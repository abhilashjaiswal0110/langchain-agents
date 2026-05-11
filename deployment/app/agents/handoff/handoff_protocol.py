"""Data models for the agent-to-agent handoff protocol."""

from pydantic import BaseModel, Field


class HandoffRequest(BaseModel):
    """Request to transfer a conversation from one agent to another.

    Attributes:
        from_agent: The agent initiating the handoff.
        to_agent: The target agent that should take over.
        reason: Human-readable explanation for the transfer.
        session_id: The conversation session to transfer.
        conversation_summary: Optional digest of prior context.
        key_entities: Optional structured context entities to carry over.
        user_id: Optional user identifier.
    """

    from_agent: str
    to_agent: str
    reason: str
    session_id: str
    conversation_summary: str = ""
    key_entities: dict = Field(default_factory=dict)
    user_id: str | None = None


class HandoffResult(BaseModel):
    """Result of an agent handoff attempt.

    Attributes:
        success: Whether the handoff succeeded.
        new_agent: The agent now handling the session (empty on failure).
        session_id: The session that was transferred.
        error: Error message on failure.
        context_preserved: Whether prior conversation context was retained.
    """

    success: bool
    new_agent: str
    session_id: str
    error: str | None = None
    context_preserved: bool = True
