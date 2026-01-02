"""Redis session store implementation.

Provides distributed, persistent storage using Redis.
Suitable for production deployments with multiple instances.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from app.memory.base import (
    BaseSessionStore,
    Message,
    Session,
    SessionMetadata,
)

logger = logging.getLogger(__name__)


class RedisSessionStore(BaseSessionStore):
    """Redis-based session storage.

    Provides persistent, distributed storage for sessions.
    Supports TTL-based expiration and is suitable for production
    deployments with multiple application instances.

    Requires:
        pip install redis
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "session:",
        default_ttl_hours: int = 24,
        db: int = 0,
    ) -> None:
        """Initialize Redis store.

        Args:
            url: Redis connection URL.
            prefix: Key prefix for sessions.
            default_ttl_hours: Default session TTL in hours.
            db: Redis database number.
        """
        self._url = url
        self._prefix = prefix
        self._default_ttl_hours = default_ttl_hours
        self._db = db
        self._client = None
        self._connect()

    def _connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis

            self._client = redis.from_url(
                self._url,
                db=self._db,
                decode_responses=True,
            )
            # Test connection
            self._client.ping()
            logger.info(f"Connected to Redis at {self._url}")
        except ImportError:
            logger.error("Redis package not installed. Run: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _key(self, session_id: str) -> str:
        """Get Redis key for session."""
        return f"{self._prefix}{session_id}"

    def _user_index_key(self, user_id: str) -> str:
        """Get Redis key for user session index."""
        return f"{self._prefix}user:{user_id}"

    def _agent_index_key(self, agent_type: str) -> str:
        """Get Redis key for agent session index."""
        return f"{self._prefix}agent:{agent_type}"

    def create_session(
        self,
        agent_type: str,
        user_id: str = "",
        metadata: dict | None = None,
        ttl_hours: int | None = None,
    ) -> str:
        """Create a new session.

        Args:
            agent_type: Type of agent for this session.
            user_id: User identifier.
            metadata: Additional metadata.
            ttl_hours: Session TTL in hours (None for default).

        Returns:
            Session ID.
        """
        # Create session
        session_metadata = SessionMetadata(
            user_id=user_id,
            agent_type=agent_type,
            custom=metadata or {},
        )

        ttl = ttl_hours or self._default_ttl_hours
        expires_at = datetime.now() + timedelta(hours=ttl)

        session = Session(
            metadata=session_metadata,
            expires_at=expires_at,
        )

        # Store in Redis
        key = self._key(session.id)
        ttl_seconds = ttl * 3600

        self._client.setex(
            key,
            ttl_seconds,
            json.dumps(session.to_dict()),
        )

        # Add to indices
        if user_id:
            self._client.sadd(self._user_index_key(user_id), session.id)
            self._client.expire(self._user_index_key(user_id), ttl_seconds)

        self._client.sadd(self._agent_index_key(agent_type), session.id)
        self._client.expire(self._agent_index_key(agent_type), ttl_seconds)

        logger.debug(f"Created Redis session {session.id} for agent {agent_type}")
        return session.id

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            Session or None if not found.
        """
        key = self._key(session_id)
        data = self._client.get(key)

        if not data:
            return None

        try:
            session = Session.from_dict(json.loads(data))
            return session
        except Exception as e:
            logger.error(f"Failed to parse session {session_id}: {e}")
            return None

    def update_session(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: dict | None = None,
    ) -> bool:
        """Update session with new messages.

        Args:
            session_id: Session identifier.
            user_message: User's message.
            assistant_message: Assistant's response.
            metadata: Optional additional metadata.

        Returns:
            True if updated successfully.
        """
        session = self.get_session(session_id)
        if not session:
            return False

        session.add_exchange(
            user_message,
            assistant_message,
            assistant_metadata=metadata,
        )

        # Get remaining TTL
        key = self._key(session_id)
        ttl = self._client.ttl(key)
        if ttl < 0:
            ttl = self._default_ttl_hours * 3600

        # Update in Redis
        self._client.setex(
            key,
            ttl,
            json.dumps(session.to_dict()),
        )

        return True

    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session identifier.

        Returns:
            True if deleted successfully.
        """
        session = self.get_session(session_id)
        if not session:
            return False

        key = self._key(session_id)
        self._client.delete(key)

        # Remove from indices
        if session.metadata.user_id:
            self._client.srem(
                self._user_index_key(session.metadata.user_id),
                session_id,
            )

        self._client.srem(
            self._agent_index_key(session.metadata.agent_type),
            session_id,
        )

        logger.debug(f"Deleted Redis session {session_id}")
        return True

    def list_sessions(
        self,
        user_id: str | None = None,
        agent_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Session]:
        """List sessions with optional filters.

        Args:
            user_id: Filter by user.
            agent_type: Filter by agent type.
            limit: Maximum number of sessions.
            offset: Offset for pagination.

        Returns:
            List of sessions.
        """
        session_ids = set()

        # Get from indices
        if user_id:
            user_sessions = self._client.smembers(self._user_index_key(user_id))
            session_ids.update(user_sessions)
        elif agent_type:
            agent_sessions = self._client.smembers(self._agent_index_key(agent_type))
            session_ids.update(agent_sessions)
        else:
            # Scan all sessions (expensive, use with caution)
            cursor = 0
            pattern = f"{self._prefix}[0-9a-f]*"
            while True:
                cursor, keys = self._client.scan(cursor, match=pattern, count=1000)
                for key in keys:
                    sid = key.replace(self._prefix, "")
                    if not sid.startswith("user:") and not sid.startswith("agent:"):
                        session_ids.add(sid)
                if cursor == 0:
                    break

        # Fetch sessions
        sessions = []
        for sid in session_ids:
            session = self.get_session(sid)
            if session:
                # Apply filters
                if user_id and session.metadata.user_id != user_id:
                    continue
                if agent_type and session.metadata.agent_type != agent_type:
                    continue
                sessions.append(session)

        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.updated_at, reverse=True)

        return sessions[offset : offset + limit]

    def get_history(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[Message]:
        """Get conversation history for a session.

        Args:
            session_id: Session identifier.
            limit: Maximum number of messages.

        Returns:
            List of messages.
        """
        session = self.get_session(session_id)
        if not session:
            return []

        return session.get_history(limit)

    def clear_session(self, session_id: str) -> bool:
        """Clear messages from a session.

        Args:
            session_id: Session identifier.

        Returns:
            True if cleared successfully.
        """
        session = self.get_session(session_id)
        if not session:
            return False

        session.clear_messages()

        # Get remaining TTL
        key = self._key(session_id)
        ttl = self._client.ttl(key)
        if ttl < 0:
            ttl = self._default_ttl_hours * 3600

        # Update in Redis
        self._client.setex(
            key,
            ttl,
            json.dumps(session.to_dict()),
        )

        return True

    def set_context(
        self,
        session_id: str,
        context: dict[str, Any],
    ) -> bool:
        """Set session context.

        Args:
            session_id: Session identifier.
            context: Context data to set.

        Returns:
            True if set successfully.
        """
        session = self.get_session(session_id)
        if not session:
            return False

        session.context.update(context)
        session.updated_at = datetime.now()

        # Get remaining TTL
        key = self._key(session_id)
        ttl = self._client.ttl(key)
        if ttl < 0:
            ttl = self._default_ttl_hours * 3600

        # Update in Redis
        self._client.setex(
            key,
            ttl,
            json.dumps(session.to_dict()),
        )

        return True

    def get_context(self, session_id: str) -> dict[str, Any]:
        """Get session context.

        Args:
            session_id: Session identifier.

        Returns:
            Context data.
        """
        session = self.get_session(session_id)
        if not session:
            return {}

        return session.context.copy()

    def extend_ttl(self, session_id: str, hours: int) -> bool:
        """Extend session TTL.

        Args:
            session_id: Session identifier.
            hours: Hours to extend by.

        Returns:
            True if extended successfully.
        """
        key = self._key(session_id)
        current_ttl = self._client.ttl(key)

        if current_ttl < 0:
            return False

        new_ttl = current_ttl + (hours * 3600)
        return self._client.expire(key, new_ttl)

    def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            self._client.close()
            logger.info("Closed Redis connection")
