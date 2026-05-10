"""Agent-to-agent handoff framework."""
from app.agents.handoff.handoff_protocol import HandoffRequest, HandoffResult
from app.agents.handoff.handoff_manager import HandoffManager

__all__ = ["HandoffRequest", "HandoffResult", "HandoffManager"]
