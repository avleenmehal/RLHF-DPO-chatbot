"""SQLAlchemy database setup and models."""

import base64
import hashlib
import uuid
from datetime import datetime

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, Text, create_engine, event
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.types import TypeDecorator

from core.config import Config


# ── Encryption helpers ────────────────────────────────────────────────────────

def _get_fernet() -> Fernet:
    """Return a Fernet instance using DB_ENCRYPTION_KEY from config.

    Falls back to a key derived from JWT_SECRET in development.
    Prints a loud warning if the fallback is used.
    """
    key = Config.DB_ENCRYPTION_KEY
    if not key:
        import warnings
        warnings.warn(
            "DB_ENCRYPTION_KEY is not set — deriving encryption key from JWT_SECRET. "
            "This is NOT safe for production. Set DB_ENCRYPTION_KEY in your .env file.",
            stacklevel=2,
        )
        raw = hashlib.sha256(Config.JWT_SECRET.encode()).digest()
        key = base64.urlsafe_b64encode(raw).decode()
    return Fernet(key.encode() if isinstance(key, str) else key)


class EncryptedText(TypeDecorator):
    """Transparently encrypts/decrypts text columns using Fernet (AES-128-CBC + HMAC).

    Encrypted values are stored as base64 ciphertext strings.
    Existing plaintext values (pre-migration) are returned as-is and will be
    re-encrypted on the next write — no manual migration required.
    """
    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        """Encrypt before writing to DB."""
        if value is None:
            return None
        return _get_fernet().encrypt(value.encode()).decode()

    def process_result_value(self, value, dialect):
        """Decrypt after reading from DB. Falls back gracefully for plaintext rows."""
        if value is None:
            return None
        try:
            return _get_fernet().decrypt(value.encode()).decode()
        except (InvalidToken, Exception):
            # Pre-encryption plaintext row — return as-is; will encrypt on next write
            return value

_is_sqlite = Config.DATABASE_URL.startswith("sqlite")

engine = create_engine(
    Config.DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30} if _is_sqlite else {},
)

if _is_sqlite:
    @event.listens_for(engine, "connect")
    def _set_wal_mode(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        finally:
            cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")


class ChatSession(Base):
    __tablename__ = "sessions"

    session_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False, index=True)
    title = Column(String, default="New Conversation")
    model_type = Column(String, default="openai")
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="sessions")
    messages = relationship(
        "Message", back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.timestamp",
    )


class Message(Base):
    __tablename__ = "messages"

    message_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, ForeignKey("sessions.session_id"), nullable=False, index=True)
    role = Column(String, nullable=False)           # "user" | "assistant"
    content = Column(EncryptedText, nullable=False) # encrypted at rest
    timestamp = Column(DateTime, default=datetime.utcnow)
    tools_used = Column(String, nullable=True)      # JSON-encoded list e.g. '["rag","web"]'

    session = relationship("ChatSession", back_populates="messages")


class Preference(Base):
    __tablename__ = "preferences"

    preference_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    # user_id intentionally omitted — preference training data is not linked to identity
    session_id = Column(String, nullable=True, index=True)
    query = Column(EncryptedText, nullable=False)           # encrypted at rest
    chosen_response = Column(EncryptedText, nullable=False) # encrypted at rest
    rejected_response = Column(EncryptedText, nullable=False)
    chosen_variant = Column(String, nullable=True)          # "empathetic" or "clinical"
    rejected_variant = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create all tables if they don't exist, and migrate any missing columns."""
    Base.metadata.create_all(bind=engine)
    _migrate_preferences_variants()


def _migrate_preferences_variants():
    """Add chosen_variant / rejected_variant to preferences if they don't exist yet.

    SQLAlchemy create_all won't add columns to existing tables, so we handle
    this with a lightweight ALTER TABLE for both SQLite and PostgreSQL.
    """
    new_cols = {"chosen_variant": "VARCHAR", "rejected_variant": "VARCHAR"}
    with engine.connect() as conn:
        if _is_sqlite:
            existing = {
                row[1]
                for row in conn.execute(
                    __import__("sqlalchemy").text("PRAGMA table_info(preferences)")
                )
            }
            for col, col_type in new_cols.items():
                if col not in existing:
                    conn.execute(
                        __import__("sqlalchemy").text(
                            f"ALTER TABLE preferences ADD COLUMN {col} {col_type}"
                        )
                    )
            conn.commit()
        else:
            # PostgreSQL — use information_schema
            for col, col_type in new_cols.items():
                result = conn.execute(
                    __import__("sqlalchemy").text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name='preferences' AND column_name=:col"
                    ),
                    {"col": col},
                )
                if result.fetchone() is None:
                    conn.execute(
                        __import__("sqlalchemy").text(
                            f"ALTER TABLE preferences ADD COLUMN {col} VARCHAR"
                        )
                    )
            conn.commit()


def get_db():
    """Dependency-style DB session — yields and closes cleanly."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
