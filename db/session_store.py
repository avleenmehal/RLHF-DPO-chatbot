"""Session and message persistence — scoped per user."""

import json
from typing import Optional

from db.database import ChatSession, Message, SessionLocal


class SessionStore:

    # ── Sessions ──────────────────────────────────────────────────────────

    def create_session(self, user_id: str, model_type: str = "openai") -> str:
        """Create a new chat session and return its session_id."""
        db = SessionLocal()
        try:
            session = ChatSession(user_id=user_id, model_type=model_type)
            db.add(session)
            db.commit()
            db.refresh(session)
            return session.session_id
        finally:
            db.close()

    def list_sessions(self, user_id: str) -> list[dict]:
        """Return all sessions for a user, newest first."""
        db = SessionLocal()
        try:
            rows = (
                db.query(ChatSession)
                .filter(ChatSession.user_id == user_id)
                .order_by(ChatSession.created_at.desc())
                .all()
            )
            return [
                {
                    "session_id": r.session_id,
                    "title": r.title,
                    "model_type": r.model_type,
                    "created_at": r.created_at.strftime("%b %d, %Y %H:%M"),
                }
                for r in rows
            ]
        finally:
            db.close()

    def set_title(self, session_id: str, first_message: str):
        """Set session title from the first user message.

        Uses only the first 5 words to avoid surfacing sensitive medical
        content directly in the session sidebar.
        """
        words = first_message.split()
        truncated = " ".join(words[:5])
        if len(words) > 5:
            truncated += "…"
        db = SessionLocal()
        try:
            session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
            if session:
                session.title = truncated[:60]
                db.commit()
        finally:
            db.close()

    # ── Messages ──────────────────────────────────────────────────────────

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tools_used: Optional[list] = None,
    ):
        """Append a single message to a session."""
        db = SessionLocal()
        try:
            msg = Message(
                session_id=session_id,
                role=role,
                content=content,
                tools_used=json.dumps(tools_used) if tools_used else None,
            )
            db.add(msg)
            db.commit()
        finally:
            db.close()

    def get_user_info(self, user_id: str) -> dict | None:
        """Return username and email for a given user_id."""
        from db.database import User
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.user_id == user_id).first()
            if user:
                return {"username": user.username, "email": user.email}
            return None
        finally:
            db.close()

    def save_preference(
        self,
        session_id: str,
        query: str,
        chosen_response: str,
        rejected_response: str,
        chosen_variant: str = None,
        rejected_variant: str = None,
    ):
        """Persist a single A/B preference pair.

        user_id is intentionally excluded — preference/training data
        should not be linked to individual user identity.
        """
        from db.database import Preference
        db = SessionLocal()
        try:
            pref = Preference(
                session_id=session_id,
                query=query,
                chosen_response=chosen_response,
                rejected_response=rejected_response,
                chosen_variant=chosen_variant,
                rejected_variant=rejected_variant,
            )
            db.add(pref)
            db.commit()
        finally:
            db.close()

    def load_messages(self, session_id: str) -> list[dict]:
        """Return all messages for a session, in chronological order."""
        db = SessionLocal()
        try:
            rows = (
                db.query(Message)
                .filter(Message.session_id == session_id)
                .order_by(Message.timestamp)
                .all()
            )
            return [
                {
                    "role": r.role,
                    "content": r.content,
                    "tools_used": json.loads(r.tools_used) if r.tools_used else [],
                }
                for r in rows
            ]
        finally:
            db.close()
