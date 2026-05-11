"""Conversation memory for LangChain/LangGraph integration.

Provides conversation memory that integrates with session stores
and supports LangChain message formats.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.memory.base import Message, Session

logger = logging.getLogger(__name__)


@dataclass
class ConversationSummary:
    """Summary of a conversation.

    Attributes:
        session_id: Session identifier.
        agent_type: Type of agent.
        user_id: User identifier.
        message_count: Number of messages.
        started_at: When the conversation started.
        last_message_at: When the last message was sent.
        topics: Extracted topics (if available).
        summary: Text summary (if generated).
    """

    session_id: str
    agent_type: str
    user_id: str
    message_count: int
    started_at: datetime
    last_message_at: datetime
    topics: list[str] = field(default_factory=list)
    summary: str = ""


class ConversationMemory:
    """Conversation memory with LangChain integration.

    Provides memory management for LangChain chains and LangGraph agents
    with support for multiple storage backends.
    """

    def __init__(
        self,
        session_store: Any = None,
        max_messages: int = 100,
        summarize_after: int = 50,
    ) -> None:
        """Initialize conversation memory.

        Args:
            session_store: Session store instance.
            max_messages: Maximum messages to keep in memory.
            summarize_after: Number of messages after which to summarize.
        """
        from app.memory.config import get_session_store

        self._store = session_store or get_session_store()
        self._max_messages = max_messages
        self._summarize_after = summarize_after

    def create_session(
        self,
        agent_type: str,
        user_id: str = "",
        metadata: dict | None = None,
    ) -> str:
        """Create a new conversation session.

        Args:
            agent_type: Type of agent.
            user_id: User identifier.
            metadata: Additional metadata.

        Returns:
            Session ID.
        """
        return self._store.create_session(
            agent_type=agent_type,
            user_id=user_id,
            metadata=metadata,
        )

    def add_user_message(
        self,
        session_id: str,
        content: str,
        metadata: dict | None = None,
    ) -> bool:
        """Add a user message to the conversation.

        Args:
            session_id: Session identifier.
            content: Message content.
            metadata: Optional metadata.

        Returns:
            True if added successfully.
        """
        session = self._store.get_session(session_id)
        if not session:
            return False

        session.add_message("user", content, metadata)
        self._save_session(session)
        return True

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        metadata: dict | None = None,
    ) -> bool:
        """Add an assistant message to the conversation.

        Args:
            session_id: Session identifier.
            content: Message content.
            metadata: Optional metadata.

        Returns:
            True if added successfully.
        """
        session = self._store.get_session(session_id)
        if not session:
            return False

        session.add_message("assistant", content, metadata)
        self._save_session(session)
        return True

    def add_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        user_metadata: dict | None = None,
        assistant_metadata: dict | None = None,
    ) -> bool:
        """Add a user/assistant exchange.

        Args:
            session_id: Session identifier.
            user_message: User's message.
            assistant_message: Assistant's response.
            user_metadata: Optional user message metadata.
            assistant_metadata: Optional assistant message metadata.

        Returns:
            True if added successfully.
        """
        return self._store.update_session(
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
            metadata=assistant_metadata,
        )

    def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[Message]:
        """Get messages from a session.

        Args:
            session_id: Session identifier.
            limit: Maximum messages to return.

        Returns:
            List of messages.
        """
        return self._store.get_history(session_id, limit)

    def get_langchain_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[Any]:
        """Get messages in LangChain format.

        Args:
            session_id: Session identifier.
            limit: Maximum messages to return.

        Returns:
            List of LangChain message objects.
        """
        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        except ImportError:
            logger.warning("langchain_core not installed, returning raw messages")
            return self.get_messages(session_id, limit)

        messages = self.get_messages(session_id, limit)
        lc_messages = []

        for msg in messages:
            if msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))

        return lc_messages

    def get_chat_history_string(
        self,
        session_id: str,
        limit: int | None = None,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
    ) -> str:
        """Get chat history as a formatted string.

        Args:
            session_id: Session identifier.
            limit: Maximum messages to include.
            human_prefix: Prefix for human messages.
            ai_prefix: Prefix for AI messages.

        Returns:
            Formatted chat history string.
        """
        messages = self.get_messages(session_id, limit)

        lines = []
        for msg in messages:
            if msg.role == "user":
                lines.append(f"{human_prefix}: {msg.content}")
            elif msg.role == "assistant":
                lines.append(f"{ai_prefix}: {msg.content}")

        return "\n".join(lines)

    def clear_session(self, session_id: str) -> bool:
        """Clear all messages from a session.

        Args:
            session_id: Session identifier.

        Returns:
            True if cleared successfully.
        """
        return self._store.clear_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session entirely.

        Args:
            session_id: Session identifier.

        Returns:
            True if deleted successfully.
        """
        return self._store.delete_session(session_id)

    def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID.

        Args:
            session_id: Session identifier.

        Returns:
            Session or None if not found.
        """
        return self._store.get_session(session_id)

    def set_context(
        self,
        session_id: str,
        context: dict[str, Any],
    ) -> bool:
        """Set session context.

        Args:
            session_id: Session identifier.
            context: Context data.

        Returns:
            True if set successfully.
        """
        return self._store.set_context(session_id, context)

    def get_context(self, session_id: str) -> dict[str, Any]:
        """Get session context.

        Args:
            session_id: Session identifier.

        Returns:
            Context data.
        """
        return self._store.get_context(session_id)

    def get_summary(self, session_id: str) -> ConversationSummary | None:
        """Get conversation summary.

        Args:
            session_id: Session identifier.

        Returns:
            Conversation summary or None.
        """
        session = self._store.get_session(session_id)
        if not session:
            return None

        return ConversationSummary(
            session_id=session.id,
            agent_type=session.metadata.agent_type,
            user_id=session.metadata.user_id,
            message_count=session.message_count,
            started_at=session.created_at,
            last_message_at=session.updated_at,
            topics=session.metadata.tags,
        )

    def list_sessions(
        self,
        user_id: str | None = None,
        agent_type: str | None = None,
        limit: int = 100,
    ) -> list[ConversationSummary]:
        """List conversation sessions.

        Args:
            user_id: Filter by user.
            agent_type: Filter by agent type.
            limit: Maximum sessions to return.

        Returns:
            List of conversation summaries.
        """
        sessions = self._store.list_sessions(
            user_id=user_id,
            agent_type=agent_type,
            limit=limit,
        )

        summaries = []
        for session in sessions:
            summaries.append(
                ConversationSummary(
                    session_id=session.id,
                    agent_type=session.metadata.agent_type,
                    user_id=session.metadata.user_id,
                    message_count=session.message_count,
                    started_at=session.created_at,
                    last_message_at=session.updated_at,
                    topics=session.metadata.tags,
                )
            )

        return summaries

    def _save_session(self, session: Session) -> None:
        """Save session back to store.

        For in-memory store this is a no-op since we have
        a reference to the same object. For other stores,
        this would persist the changes.
        """
        # Check if we need to trim messages
        if len(session.messages) > self._max_messages:
            session.messages = session.messages[-self._max_messages :]


# Singleton pattern
_conversation_memory: ConversationMemory | None = None


def get_conversation_memory(
    session_store: Any = None,
) -> ConversationMemory:
    """Get or create global conversation memory instance.

    Args:
        session_store: Optional session store.

    Returns:
        Conversation memory instance.
    """
    global _conversation_memory
    if _conversation_memory is None:
        _conversation_memory = ConversationMemory(session_store=session_store)
    return _conversation_memory


def reset_conversation_memory() -> None:
    """Reset global conversation memory instance."""
    global _conversation_memory
    _conversation_memory = None
