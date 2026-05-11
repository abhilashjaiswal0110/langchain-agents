"""Agent-to-agent handoff manager with context preservation.

Usage:
    from app.agents.handoff import HandoffManager, HandoffRequest

    manager = HandoffManager()
    result = await manager.execute_handoff(request, conversation_manager)
"""

import logging
from typing import Any

from app.agents.handoff.handoff_protocol import HandoffRequest, HandoffResult

logger = logging.getLogger(__name__)

# Agents that can be targeted in a handoff.
# Mirrors ConversationManager.AVAILABLE_AGENTS but kept local
# so the handoff package has no circular dependency.
_VALID_HANDOFF_TARGETS = {
    "it_helpdesk",
    "servicenow",
    "document_intelligence",
    "employee_experience",
}


class HandoffManager:
    """Orchestrates agent-to-agent session transfers.

    Snapshots the current session context, delegates to
    ``ConversationManager.switch_agent()``, and returns a structured result.
    """

    def list_valid_targets(self) -> list[str]:
        """Return agent types that can receive a handoff.

        Returns:
            Sorted list of valid target agent names.
        """
        return sorted(_VALID_HANDOFF_TARGETS)

    async def execute_handoff(
        self,
        request: HandoffRequest,
        conversation_manager: Any,
    ) -> HandoffResult:
        """Transfer a session from one agent to another.

        Snapshots the current session, calls ``switch_agent`` on the
        conversation manager, and returns the outcome.

        Args:
            request: Handoff parameters (source, target, reason, context).
            conversation_manager: ``ConversationManager`` instance that owns
                the session store.

        Returns:
            HandoffResult with success status and error detail on failure.
        """
        logger.info(
            "Handoff requested: %s → %s for session %s (reason: %s)",
            request.from_agent,
            request.to_agent,
            request.session_id,
            request.reason,
        )

        try:
            # Snapshot current session so context can be preserved
            session = conversation_manager.session_store.get_session(request.session_id)

            handoff_context = {
                "from_agent": request.from_agent,
                "reason": request.reason,
                "conversation_summary": request.conversation_summary,
                "key_entities": request.key_entities,
            }

            await conversation_manager.switch_agent(
                session_id=request.session_id,
                new_agent_type=request.to_agent,
                preserve_context=True,
                handoff_context=handoff_context,
            )

            logger.info(
                "Handoff complete: session %s now handled by %s",
                request.session_id,
                request.to_agent,
            )
            return HandoffResult(
                success=True,
                new_agent=request.to_agent,
                session_id=request.session_id,
                context_preserved=True,
            )

        except Exception as exc:
            logger.error(
                "Handoff failed for session %s: %s",
                request.session_id,
                exc,
            )
            return HandoffResult(
                success=False,
                new_agent="",
                session_id=request.session_id,
                error=str(exc),
                context_preserved=False,
            )
