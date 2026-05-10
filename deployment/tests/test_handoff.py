"""Tests for the agent-to-agent handoff framework."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.handoff.handoff_protocol import HandoffRequest, HandoffResult
from app.agents.handoff.handoff_manager import HandoffManager


class TestHandoffRequest:
    def test_valid_handoff_request(self) -> None:
        req = HandoffRequest(
            from_agent="it_helpdesk",
            to_agent="servicenow",
            reason="User needs ticket created",
            session_id="sess-123",
        )
        assert req.from_agent == "it_helpdesk"
        assert req.to_agent == "servicenow"
        assert req.session_id == "sess-123"

    def test_handoff_request_optional_fields(self) -> None:
        req = HandoffRequest(
            from_agent="research",
            to_agent="content",
            reason="Generate article from research",
            session_id="sess-456",
        )
        assert req.conversation_summary == ""
        assert req.key_entities == {}

    def test_handoff_request_with_context(self) -> None:
        req = HandoffRequest(
            from_agent="it_helpdesk",
            to_agent="servicenow",
            reason="Escalate ticket",
            session_id="sess-789",
            conversation_summary="User reported login failure",
            key_entities={"incident_type": "login", "priority": "high"},
        )
        assert req.conversation_summary == "User reported login failure"
        assert req.key_entities["priority"] == "high"


class TestHandoffResult:
    def test_successful_result(self) -> None:
        result = HandoffResult(
            success=True,
            new_agent="servicenow",
            session_id="sess-123",
        )
        assert result.success is True
        assert result.new_agent == "servicenow"
        assert result.error is None

    def test_failed_result(self) -> None:
        result = HandoffResult(
            success=False,
            new_agent="",
            session_id="sess-123",
            error="Agent not found",
        )
        assert result.success is False
        assert result.error == "Agent not found"


class TestHandoffManager:
    @pytest.fixture()
    def manager(self) -> HandoffManager:
        return HandoffManager()

    @pytest.mark.asyncio
    async def test_execute_handoff_success(self, manager: HandoffManager) -> None:
        mock_cm = MagicMock()
        mock_cm.switch_agent = AsyncMock(return_value=None)
        mock_cm.session_store = MagicMock()
        mock_cm.session_store.get_session = MagicMock(return_value=MagicMock())

        req = HandoffRequest(
            from_agent="it_helpdesk",
            to_agent="servicenow",
            reason="Escalate",
            session_id="sess-abc",
        )
        result = await manager.execute_handoff(req, mock_cm)

        assert result.success is True
        assert result.new_agent == "servicenow"
        mock_cm.switch_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_handoff_unknown_agent(self, manager: HandoffManager) -> None:
        mock_cm = MagicMock()
        mock_cm.switch_agent = AsyncMock(side_effect=ValueError("Unknown agent: ghost"))
        mock_cm.session_store = MagicMock()
        mock_cm.session_store.get_session = MagicMock(return_value=MagicMock())

        req = HandoffRequest(
            from_agent="it_helpdesk",
            to_agent="ghost",
            reason="test",
            session_id="sess-xyz",
        )
        result = await manager.execute_handoff(req, mock_cm)

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_handoff_preserves_session_id(self, manager: HandoffManager) -> None:
        mock_cm = MagicMock()
        mock_cm.switch_agent = AsyncMock()
        mock_cm.session_store = MagicMock()
        mock_cm.session_store.get_session = MagicMock(return_value=MagicMock())

        req = HandoffRequest(
            from_agent="a",
            to_agent="b",
            reason="test",
            session_id="my-session-id",
        )
        result = await manager.execute_handoff(req, mock_cm)
        assert result.session_id == "my-session-id"

    def test_list_valid_targets(self, manager: HandoffManager) -> None:
        targets = manager.list_valid_targets()
        assert isinstance(targets, list)
        assert len(targets) > 0

    @pytest.mark.asyncio
    async def test_switch_agent_called_with_to_agent(self, manager: HandoffManager) -> None:
        mock_cm = MagicMock()
        mock_cm.switch_agent = AsyncMock()
        mock_cm.session_store = MagicMock()
        mock_cm.session_store.get_session = MagicMock(return_value=MagicMock())

        req = HandoffRequest(
            from_agent="it_helpdesk",
            to_agent="servicenow",
            reason="escalate",
            session_id="s1",
        )
        await manager.execute_handoff(req, mock_cm)
        call_args = mock_cm.switch_agent.call_args
        assert "servicenow" in str(call_args)
