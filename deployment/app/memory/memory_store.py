"""In-memory session store implementation.

Provides fast, non-persistent storage for development and testing.
Data is lost when the application restarts.
"""

import logging
import os
from datetime import datetime, timedelta
from threading import Lock
from typing import Any

from app.memory.base import (
    BaseSessionStore,
    Message,
    Session,
    SessionMetadata,
)

# Maximum number of messages to keep per session (0 = unlimited).
# Set MAX_HISTORY_MESSAGES environment variable to override.
MAX_HISTORY_MESSAGES: int = int(os.getenv("MAX_HISTORY_MESSAGES", "0"))

logger = logging.getLogger(__name__)


class InMemorySessionStore(BaseSessionStore):
    """In-memory session storage.

    Thread-safe implementation using a dictionary.
    Suitable for development, testing, and single-instance deployments
    where persistence is not required.
    """

    def __init__(self, max_sessions: int = 10000) -> None:
        """Initialize in-memory store.

        Args:
            max_sessions: Maximum number of sessions to store.
        """
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()
        self._max_sessions = max_sessions

    def _key(self, tenant_id: str, session_id: str) -> str:
        """Build namespaced key for the internal dict.

        Args:
            tenant_id: Tenant identifier.
            session_id: Session identifier.

        Returns:
            Namespaced key string.
        """
        return f"{tenant_id}:{session_id}"

    def create_session(
        self,
        agent_type: str,
        user_id: str = "",
        metadata: dict | None = None,
        ttl_hours: int | None = None,
        tenant_id: str = "default",
    ) -> str:
        """Create a new session.

        Args:
            agent_type: Type of agent for this session.
            user_id: User identifier.
            metadata: Additional metadata.
            ttl_hours: Session TTL in hours (None for no expiry).
            tenant_id: Tenant identifier for session isolation.

        Returns:
            Session ID.
        """
        with self._lock:
            # Cleanup if at capacity
            if len(self._sessions) >= self._max_sessions:
                self._cleanup_oldest()

            # Create session
            session_metadata = SessionMetadata(
                user_id=user_id,
                agent_type=agent_type,
                custom=metadata or {},
                tenant_id=tenant_id,
            )

            expires_at = None
            if ttl_hours:
                expires_at = datetime.now() + timedelta(hours=ttl_hours)

            session = Session(
                metadata=session_metadata,
                expires_at=expires_at,
            )

            key = self._key(tenant_id, session.id)
            self._sessions[key] = session
            logger.debug(f"Created session {session.id} for agent {agent_type} (tenant={tenant_id})")

            return session.id

    def get_session(self, session_id: str, tenant_id: str = "default") -> Session | None:
        """Get a session by ID.

        Returns a deep copy to prevent callers from modifying internal state.

        Args:
            session_id: Session identifier.
            tenant_id: Tenant identifier for session isolation.

        Returns:
            Deep copy of Session or None if not found.
        """
        with self._lock:
            key = self._key(tenant_id, session_id)
            session = self._sessions.get(key)

            if session and session.is_expired:
                del self._sessions[key]
                return None

            # Return deep copy to prevent external modifications
            return session.copy() if session else None

    def update_session(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: dict | None = None,
        tenant_id: str = "default",
    ) -> bool:
        """Update session with new messages.

        Args:
            session_id: Session identifier.
            user_message: User's message.
            assistant_message: Assistant's response.
            metadata: Optional additional metadata.
            tenant_id: Tenant identifier for session isolation.

        Returns:
            True if updated successfully.
        """
        with self._lock:
            key = self._key(tenant_id, session_id)
            session = self._sessions.get(key)
            if not session:
                return False

            if session.is_expired:
                del self._sessions[key]
                return False

            session.add_exchange(
                user_message,
                assistant_message,
                assistant_metadata=metadata,
            )

            # Trim history if a limit is configured.
            if MAX_HISTORY_MESSAGES > 0:
                session.messages = session.messages[-MAX_HISTORY_MESSAGES:]

            return True

    def delete_session(self, session_id: str, tenant_id: str = "default") -> bool:
        """Delete a session.

        Args:
            session_id: Session identifier.
            tenant_id: Tenant identifier for session isolation.

        Returns:
            True if deleted successfully.
        """
        with self._lock:
            key = self._key(tenant_id, session_id)
            if key in self._sessions:
                del self._sessions[key]
                logger.debug(f"Deleted session {session_id} (tenant={tenant_id})")
                return True
            return False

    def list_sessions(
        self,
        user_id: str | None = None,
        agent_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
        tenant_id: str | None = None,
    ) -> list[Session]:
        """List sessions with optional filters.

        Returns deep copies to prevent external modifications.

        Args:
            user_id: Filter by user.
            agent_type: Filter by agent type.
            limit: Maximum number of sessions.
            offset: Offset for pagination.
            tenant_id: Filter by tenant. When provided, only sessions for that
                tenant are returned.

        Returns:
            List of session deep copies.
        """
        with self._lock:
            results = []

            for key, session in self._sessions.items():
                if session.is_expired:
                    continue
                if tenant_id and session.metadata.tenant_id != tenant_id:
                    continue
                if user_id and session.metadata.user_id != user_id:
                    continue
                if agent_type and session.metadata.agent_type != agent_type:
                    continue
                results.append(session)

            # Sort by updated_at descending
            results.sort(key=lambda s: s.updated_at, reverse=True)

            # Return deep copies
            return [s.copy() for s in results[offset : offset + limit]]

    def get_history(
        self,
        session_id: str,
        limit: int | None = None,
        tenant_id: str = "default",
    ) -> list[Message]:
        """Get conversation history for a session.

        Args:
            session_id: Session identifier.
            limit: Maximum number of messages.
            tenant_id: Tenant identifier for session isolation.

        Returns:
            List of messages.
        """
        session = self.get_session(session_id, tenant_id=tenant_id)
        if not session:
            return []

        return session.get_history(limit)

    def clear_session(self, session_id: str, tenant_id: str = "default") -> bool:
        """Clear messages from a session.

        Args:
            session_id: Session identifier.
            tenant_id: Tenant identifier for session isolation.

        Returns:
            True if cleared successfully.
        """
        with self._lock:
            key = self._key(tenant_id, session_id)
            session = self._sessions.get(key)
            if not session:
                return False

            session.clear_messages()
            return True

    def set_context(
        self,
        session_id: str,
        context: dict[str, Any],
        tenant_id: str = "default",
    ) -> bool:
        """Set session context.

        Args:
            session_id: Session identifier.
            context: Context data to set.
            tenant_id: Tenant identifier for session isolation.

        Returns:
            True if set successfully.
        """
        with self._lock:
            key = self._key(tenant_id, session_id)
            session = self._sessions.get(key)
            if not session:
                return False

            session.context.update(context)
            session.updated_at = datetime.now()
            return True

    def get_context(self, session_id: str, tenant_id: str = "default") -> dict[str, Any]:
        """Get session context.

        Args:
            session_id: Session identifier.
            tenant_id: Tenant identifier for session isolation.

        Returns:
            Context data.
        """
        session = self.get_session(session_id, tenant_id=tenant_id)
        if not session:
            return {}

        return session.context.copy()

    def cleanup_expired(self) -> int:
        """Clean up expired sessions.

        Returns:
            Number of sessions removed.
        """
        with self._lock:
            expired = [
                key for key, session in self._sessions.items()
                if session.is_expired
            ]

            for key in expired:
                del self._sessions[key]

            if expired:
                logger.info(f"Cleaned up {len(expired)} expired sessions")

            return len(expired)

    def _cleanup_oldest(self) -> None:
        """Remove oldest sessions when at capacity."""
        # Sort by updated_at and remove oldest 10%
        sessions_list = sorted(
            self._sessions.items(),
            key=lambda x: x[1].updated_at,
        )

        to_remove = max(1, len(sessions_list) // 10)
        for sid, _ in sessions_list[:to_remove]:
            del self._sessions[sid]

        logger.info(f"Removed {to_remove} oldest sessions due to capacity")

    @property
    def session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)

    def close(self) -> None:
        """Close the store (no-op for in-memory)."""
        pass
