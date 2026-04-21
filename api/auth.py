"""Authentication — registration, login, JWT, bcrypt."""

from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from core.config import Config
from db.database import User


class AuthManager:

    # ── Password helpers ──────────────────────────────────────────────────

    @staticmethod
    def hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")

    @staticmethod
    def verify_password(plain: str, hashed: str) -> bool:
        return bcrypt.checkpw(plain.encode("utf-8")[:72], hashed.encode("utf-8"))

    # ── JWT helpers ───────────────────────────────────────────────────────

    @staticmethod
    def create_token(user_id: str) -> str:
        expire = datetime.utcnow() + timedelta(hours=Config.JWT_EXPIRE_HOURS)
        return jwt.encode(
            {"sub": user_id, "exp": expire},
            Config.JWT_SECRET,
            algorithm="HS256",
        )

    @staticmethod
    def decode_token(token: str) -> Optional[str]:
        """Returns user_id if token is valid, else None."""
        try:
            payload = jwt.decode(token, Config.JWT_SECRET, algorithms=["HS256"])
            return payload.get("sub")
        except JWTError:
            return None

    # ── User operations ───────────────────────────────────────────────────

    @staticmethod
    def register(db: Session, email: str, username: str, password: str) -> User:
        """Create a new user. Raises ValueError on duplicate email/username."""
        if db.query(User).filter(User.email == email).first():
            raise ValueError("Email already registered.")
        if db.query(User).filter(User.username == username).first():
            raise ValueError("Username already taken.")

        user = User(
            email=email,
            username=username,
            password_hash=AuthManager.hash_password(password),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def login(db: Session, email: str, password: str) -> Optional[User]:
        """Return user if credentials are valid, else None."""
        user = db.query(User).filter(User.email == email).first()
        if not user or not AuthManager.verify_password(password, user.password_hash):
            return None
        if not user.is_active:
            return None
        return user

    @staticmethod
    def get_user_by_id(db: Session, user_id: str) -> Optional[User]:
        return db.query(User).filter(User.user_id == user_id).first()
