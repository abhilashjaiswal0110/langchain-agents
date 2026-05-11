"""Tests for the Master Orchestrator."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.supervisors.master_orchestrator import (
    MasterOrchestrator,
    _CLUSTER_DOMAIN,
    _CLUSTER_IT_SUPPORT,
    _CLUSTER_RESEARCH,
    _CLUSTER_DEEP,
)


def _make_routing_result(intent: str, confidence: float = 0.9):
    from app.agents.supervisors.domain_router import RoutingResult

    return RoutingResult(
        intent=intent,
        confidence=confidence,
        keywords_matched=[intent],
        requires_supervisor=False,
        reasoning="test",
    )


class TestClusterClassification:
    @pytest.fixture()
    def orchestrator(self) -> MasterOrchestrator:
        orch = MasterOrchestrator()
        mock_router = MagicMock()
        orch._domain_router = mock_router
        return orch

    def test_general_it_maps_to_it_support(self, orchestrator: MasterOrchestrator) -> None:
        orchestrator._domain_router.classify.return_value = _make_routing_result("general")
        cluster = orchestrator._classify_cluster("reset my password")
        assert cluster == _CLUSTER_IT_SUPPORT

    def test_cloud_keyword_maps_to_domain(self, orchestrator: MasterOrchestrator) -> None:
        orchestrator._domain_router.classify.return_value = _make_routing_result("cloud")
        cluster = orchestrator._classify_cluster("list Azure VMs in my subscription")
        assert cluster == _CLUSTER_DOMAIN

    def test_hr_keyword_maps_to_domain(self, orchestrator: MasterOrchestrator) -> None:
        orchestrator._domain_router.classify.return_value = _make_routing_result("hr")
        cluster = orchestrator._classify_cluster("check my PTO balance")
        assert cluster == _CLUSTER_DOMAIN

    def test_deep_agent_keyword_triggers_deep(self, orchestrator: MasterOrchestrator) -> None:
        # deep-agent check happens BEFORE routing — mock doesn't need to be called
        cluster = orchestrator._classify_cluster("I need a root cause analysis for the outage")
        assert cluster == _CLUSTER_DEEP

    def test_rca_keyword(self, orchestrator: MasterOrchestrator) -> None:
        cluster = orchestrator._classify_cluster("run an RCA for the incident")
        assert cluster == _CLUSTER_DEEP

    def test_finance_keyword_maps_to_domain(self, orchestrator: MasterOrchestrator) -> None:
        orchestrator._domain_router.classify.return_value = _make_routing_result("finance")
        cluster = orchestrator._classify_cluster("analyze the Q1 budget for Engineering")
        assert cluster == _CLUSTER_DOMAIN

    def test_unknown_maps_to_research(self, orchestrator: MasterOrchestrator) -> None:
        orchestrator._domain_router.classify.return_value = _make_routing_result("unknown", 0.1)
        cluster = orchestrator._classify_cluster("what is the capital of Australia?")
        assert cluster in {_CLUSTER_RESEARCH, _CLUSTER_IT_SUPPORT}


class TestRouteMethod:
    @pytest.mark.asyncio
    async def test_route_returns_cluster_key(self) -> None:
        orchestrator = MasterOrchestrator()
        orchestrator._classify_cluster = MagicMock(return_value=_CLUSTER_RESEARCH)
        orchestrator._route_research = AsyncMock(return_value={
            "cluster": _CLUSTER_RESEARCH,
            "agent_type": "research",
            "response": "ok",
        })
        result = await orchestrator.route("anything")
        assert result["cluster"] == _CLUSTER_RESEARCH

    @pytest.mark.asyncio
    async def test_route_returns_error_on_exception(self) -> None:
        orchestrator = MasterOrchestrator()
        orchestrator._classify_cluster = MagicMock(return_value=_CLUSTER_RESEARCH)
        orchestrator._route_research = AsyncMock(side_effect=RuntimeError("boom"))
        result = await orchestrator.route("anything")
        assert result["agent_type"] == "error"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_route_it_support_calls_cm(self) -> None:
        mock_cm = MagicMock()
        # start_conversation is a synchronous method — use MagicMock, not AsyncMock
        mock_cm.start_conversation = MagicMock(return_value={"session_id": "s1"})
        mock_cm.achat = AsyncMock(return_value={"response": "done"})

        orchestrator = MasterOrchestrator(conversation_manager=mock_cm)
        orchestrator._classify_cluster = MagicMock(return_value=_CLUSTER_IT_SUPPORT)
        result = await orchestrator.route("reset my password")
        assert result["agent_type"] == "it_helpdesk"
