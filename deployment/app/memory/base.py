"""Base classes and types for memory storage.

Defines the abstract interface for session stores and common data types.
"""

import copy
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Message:
    """A message in a conversation.

    Attributes:
        role: Message role (user, assistant, system).
        content: Message content.
        timestamp: When the message was created.
        metadata: Additional message metadata.
    """

    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
        )


@dataclass
class SessionMetadata:
    """Metadata for a session.

    Attributes:
        user_id: User identifier.
        agent_type: Type of agent.
        tags: Optional tags for categorization.
        custom: Custom metadata fields.
        tenant_id: Tenant identifier for multi-tenancy isolation.
    """

    user_id: str = ""
    agent_type: str = ""
    tags: list[str] = field(default_factory=list)
    custom: dict[str, Any] = field(default_factory=dict)
    tenant_id: str = "default"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "agent_type": self.agent_type,
            "tags": self.tags,
            "custom": self.custom,
            "tenant_id": self.tenant_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMetadata":
        """Create from dictionary."""
        return cls(
            user_id=data.get("user_id", ""),
            agent_type=data.get("agent_type", ""),
            tags=data.get("tags", []),
            custom=data.get("custom", {}),
            tenant_id=data.get("tenant_id", "default"),
        )


@dataclass
class Session:
    """A conversation session.

    Attributes:
        id: Unique session identifier.
        metadata: Session metadata.
        messages: List of messages in the session.
        context: Session context for agent state.
        created_at: When the session was created.
        updated_at: When the session was last updated.
        expires_at: Optional expiration time.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: SessionMetadata = field(default_factory=SessionMetadata)
    messages: list[Message] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None

    @property
    def message_count(self) -> int:
        """Get number of messages."""
        return len(self.messages)

    @property
    def is_expired(self) -> bool:
        """Check if session is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def add_message(self, role: str, content: str, metadata: dict | None = None) -> Message:
        """Add a message to the session.

        Args:
            role: Message role.
            content: Message content.
            metadata: Optional metadata.

        Returns:
            The created message.
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message

    def add_exchange(
        self,
        user_message: str,
        assistant_message: str,
        user_metadata: dict | None = None,
        assistant_metadata: dict | None = None,
    ) -> tuple[Message, Message]:
        """Add a user/assistant message exchange.

        Args:
            user_message: User's message.
            assistant_message: Assistant's response.
            user_metadata: Optional user message metadata.
            assistant_metadata: Optional assistant message metadata.

        Returns:
            Tuple of (user_message, assistant_message).
        """
        user_msg = self.add_message("user", user_message, user_metadata)
        assistant_msg = self.add_message("assistant", assistant_message, assistant_metadata)
        return user_msg, assistant_msg

    def get_history(self, limit: int | None = None) -> list[Message]:
        """Get message history.

        Args:
            limit: Maximum number of messages (None for all).

        Returns:
            List of messages.
        """
        if limit is None:
            return self.messages.copy()
        return self.messages[-limit:]

    def clear_messages(self) -> None:
        """Clear all messages."""
        self.messages = []
        self.updated_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "metadata": self.metadata.to_dict(),
            "messages": [m.to_dict() for m in self.messages],
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.now()

        expires_at = data.get("expires_at")
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)

        return cls(
            id=data["id"],
            metadata=SessionMetadata.from_dict(data.get("metadata", {})),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            context=data.get("context", {}),
            created_at=created_at,
            updated_at=updated_at,
            expires_at=expires_at,
        )

    def copy(self) -> "Session":
        """Create a deep copy of this session.

        Returns:
            A new Session instance with copied data.
        """
        return Session(
            id=self.id,
            metadata=SessionMetadata(
                user_id=self.metadata.user_id,
                agent_type=self.metadata.agent_type,
                tags=copy.deepcopy(self.metadata.tags),
                custom=copy.deepcopy(self.metadata.custom),
                tenant_id=self.metadata.tenant_id,
            ),
            messages=[
                Message(
                    role=m.role,
                    content=m.content,
                    timestamp=m.timestamp,
                    metadata=copy.deepcopy(m.metadata),
                )
                for m in self.messages
            ],
            context=copy.deepcopy(self.context),
            created_at=self.created_at,
            updated_at=self.updated_at,
            expires_at=self.expires_at,
        )


class BaseSessionStore(ABC):
    """Abstract base class for session storage.

    All session store implementations must inherit from this class
    and implement the required methods.
    """

    @abstractmethod
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
        pass

    @abstractmethod
    def get_session(self, session_id: str, tenant_id: str = "default") -> Session | None:
        """Get a session by ID.

        Args:
            session_id: Session identifier.
            tenant_id: Tenant identifier for session isolation.

        Returns:
            Session or None if not found.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def delete_session(self, session_id: str, tenant_id: str = "default") -> bool:
        """Delete a session.

        Args:
            session_id: Session identifier.
            tenant_id: Tenant identifier for session isolation.

        Returns:
            True if deleted successfully.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def clear_session(self, session_id: str, tenant_id: str = "default") -> bool:
        """Clear messages from a session.

        Args:
            session_id: Session identifier.
            tenant_id: Tenant identifier for session isolation.

        Returns:
            True if cleared successfully.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_context(self, session_id: str, tenant_id: str = "default") -> dict[str, Any]:
        """Get session context.

        Args:
            session_id: Session identifier.
            tenant_id: Tenant identifier for session isolation.

        Returns:
            Context data.
        """
        pass

    def session_exists(self, session_id: str, tenant_id: str = "default") -> bool:
        """Check if session exists.

        Args:
            session_id: Session identifier.
            tenant_id: Tenant identifier for session isolation.

        Returns:
            True if session exists.
        """
        return self.get_session(session_id, tenant_id=tenant_id) is not None

    def cleanup_expired(self) -> int:
        """Clean up expired sessions.

        Returns:
            Number of sessions removed.
        """
        # Default implementation does nothing
        return 0

    def close(self) -> None:
        """Close the store and release resources."""
        pass
