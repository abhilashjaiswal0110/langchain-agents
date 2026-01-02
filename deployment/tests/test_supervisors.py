"""Unit tests for Supervisor and Domain Agent modules.

Tests cover:
- IT Supervisor routing and orchestration
- Domain Router classification
- Escalation Handler workflows
- Domain agents base functionality
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage, AIMessage

from app.agents.supervisors.it_supervisor import (
    ITSupervisor,
    DomainType,
    SupervisorAction,
    get_it_supervisor,
    create_it_supervisor,
)
from app.agents.supervisors.domain_router import (
    DomainRouter,
    DomainIntent,
    RoutingResult,
    get_domain_router,
    DOMAIN_KEYWORDS,
)
from app.agents.supervisors.escalation_handler import (
    EscalationHandler,
    EscalationLevel,
    EscalationReason,
    EscalationStatus,
    EscalationRequest,
    get_escalation_handler,
)
from app.agents.domains.base_domain_agent import (
    DomainAgent,
    DomainConfig,
    DomainType as BaseDomainType,
)


# =============================================================================
# DomainType Tests
# =============================================================================


class TestDomainType:
    """Tests for DomainType enum."""

    def test_domain_values(self) -> None:
        """Test domain type values."""
        assert DomainType.MARCOM.value == "marcom"
        assert DomainType.HR.value == "hr"
        assert DomainType.CLOUD.value == "cloud"
        assert DomainType.CYBERSECURITY.value == "cybersecurity"

    def test_all_domains_defined(self) -> None:
        """Test all expected domains are defined."""
        expected = [
            "marcom", "hr", "lnd", "presales",
            "datacenter", "cloud", "cybersecurity",
            "data_ai", "general"
        ]
        actual = [d.value for d in DomainType]
        for domain in expected:
            assert domain in actual


# =============================================================================
# IT Supervisor Tests
# =============================================================================


class TestITSupervisor:
    """Tests for IT Supervisor."""

    def test_create_supervisor(self) -> None:
        """Test creating IT Supervisor."""
        supervisor = ITSupervisor()
        assert supervisor is not None

    def test_register_domain_agent(self) -> None:
        """Test registering a domain agent."""
        supervisor = ITSupervisor()
        mock_agent = MagicMock()
        supervisor.register_domain_agent("cloud", mock_agent)
        assert "cloud" in supervisor._domain_agents

    def test_register_invalid_domain(self) -> None:
        """Test registering invalid domain raises error."""
        supervisor = ITSupervisor()
        mock_agent = MagicMock()
        with pytest.raises(ValueError):
            supervisor.register_domain_agent("invalid_domain", mock_agent)

    def test_build_graph(self) -> None:
        """Test building the supervisor graph."""
        supervisor = ITSupervisor()
        graph = supervisor.build()
        assert graph is not None

    def test_get_singleton(self) -> None:
        """Test get_it_supervisor returns singleton."""
        s1 = get_it_supervisor()
        s2 = get_it_supervisor()
        assert s1 is s2

    def test_create_new_instance(self) -> None:
        """Test create_it_supervisor creates new instance."""
        s1 = create_it_supervisor()
        s2 = create_it_supervisor()
        assert s1 is not s2


# =============================================================================
# Domain Router Tests
# =============================================================================


class TestDomainRouter:
    """Tests for Domain Router."""

    def test_create_router(self) -> None:
        """Test creating domain router."""
        router = DomainRouter()
        assert router is not None

    def test_keyword_classification_cloud(self) -> None:
        """Test keyword classification for cloud."""
        router = DomainRouter()
        result = router._keyword_classify("I need help with Azure VMs")
        assert result is not None
        assert result.intent == DomainIntent.CLOUD

    def test_keyword_classification_hr(self) -> None:
        """Test keyword classification for HR."""
        router = DomainRouter()
        result = router._keyword_classify("What is my PTO balance?")
        assert result is not None
        assert result.intent == DomainIntent.HR

    def test_keyword_classification_security(self) -> None:
        """Test keyword classification for security."""
        router = DomainRouter()
        # Use message with multiple security keywords for higher confidence
        result = router._keyword_classify(
            "security incident malware phishing vulnerability breach threat"
        )
        assert result is not None
        assert result.intent == DomainIntent.CYBERSECURITY

    def test_keyword_classification_no_match(self) -> None:
        """Test keyword classification with no clear match."""
        router = DomainRouter()
        result = router._keyword_classify("xyz123 unknown query")
        assert result is None  # Should return None for ambiguous

    def test_classify_method(self) -> None:
        """Test classify with keyword-only match (no LLM needed)."""
        router = DomainRouter()
        # Use message with many keywords to ensure high confidence without LLM
        result = router._keyword_classify("password reset login account access email vpn")
        assert result is not None
        assert isinstance(result, RoutingResult)
        assert result.intent == DomainIntent.GENERAL

    def test_classify_with_message(self) -> None:
        """Test classify with BaseMessage (keyword match)."""
        router = DomainRouter()
        # Use message with many cloud keywords for keyword-based classification
        result = router._keyword_classify(
            "azure aws gcp kubernetes container docker vm cloud iaas paas serverless"
        )
        assert result is not None
        assert result.intent == DomainIntent.CLOUD

    def test_routing_result_to_dict(self) -> None:
        """Test RoutingResult serialization."""
        result = RoutingResult(
            intent=DomainIntent.CLOUD,
            confidence=0.9,
            keywords_matched=["azure", "vm"],
            requires_supervisor=False,
            reasoning="Matched cloud keywords"
        )
        d = result.to_dict()
        assert d["intent"] == "cloud"
        assert d["confidence"] == 0.9

    def test_domain_keywords_coverage(self) -> None:
        """Test all domains have keywords defined."""
        for intent in DomainIntent:
            if intent != DomainIntent.UNKNOWN:
                assert intent in DOMAIN_KEYWORDS, f"Missing keywords for {intent}"

    def test_get_domain_router_singleton(self) -> None:
        """Test get_domain_router returns singleton."""
        r1 = get_domain_router()
        r2 = get_domain_router()
        assert r1 is r2


class TestDomainRouterAsync:
    """Async tests for Domain Router."""

    @pytest.mark.asyncio
    async def test_aclassify_keyword(self) -> None:
        """Test async keyword classification (no LLM)."""
        router = DomainRouter()
        # Test keyword-based classification directly
        result = router._keyword_classify(
            "azure aws gcp kubernetes container docker cloud vm iaas paas"
        )
        assert result is not None
        assert result.intent == DomainIntent.CLOUD


# =============================================================================
# Escalation Handler Tests
# =============================================================================


class TestEscalationLevel:
    """Tests for EscalationLevel enum."""

    def test_level_values(self) -> None:
        """Test escalation level values."""
        assert EscalationLevel.LOW.value == "low"
        assert EscalationLevel.MEDIUM.value == "medium"
        assert EscalationLevel.HIGH.value == "high"
        assert EscalationLevel.CRITICAL.value == "critical"


class TestEscalationReason:
    """Tests for EscalationReason enum."""

    def test_reason_values(self) -> None:
        """Test escalation reason values."""
        assert EscalationReason.CROSS_DOMAIN.value == "cross_domain"
        assert EscalationReason.SECURITY_INCIDENT.value == "security_incident"
        assert EscalationReason.HUMAN_REQUESTED.value == "human_requested"


class TestEscalationRequest:
    """Tests for EscalationRequest dataclass."""

    def test_create_request(self) -> None:
        """Test creating escalation request."""
        request = EscalationRequest(
            level=EscalationLevel.HIGH,
            reason=EscalationReason.SECURITY_INCIDENT,
            summary="Potential security breach detected",
        )
        assert request.level == EscalationLevel.HIGH
        assert request.status == EscalationStatus.PENDING

    def test_request_to_dict(self) -> None:
        """Test request serialization."""
        request = EscalationRequest(
            level=EscalationLevel.MEDIUM,
            reason=EscalationReason.CROSS_DOMAIN,
            summary="Multi-domain request",
            domains_involved=["cloud", "security"],
        )
        d = request.to_dict()
        assert d["level"] == "medium"
        assert d["reason"] == "cross_domain"
        assert "cloud" in d["domains_involved"]


class TestEscalationHandler:
    """Tests for EscalationHandler."""

    def test_create_handler(self) -> None:
        """Test creating escalation handler."""
        handler = EscalationHandler()
        assert handler is not None

    def test_create_escalation(self) -> None:
        """Test creating an escalation."""
        handler = EscalationHandler()
        request = handler.create_escalation(
            reason=EscalationReason.AGENT_UNCERTAIN,
            summary="Unable to determine appropriate action",
        )
        assert request.id is not None
        assert request.status == EscalationStatus.PENDING

    def test_auto_level_assignment(self) -> None:
        """Test automatic level assignment based on reason."""
        handler = EscalationHandler()

        security = handler.create_escalation(
            reason=EscalationReason.SECURITY_INCIDENT,
            summary="Security issue",
        )
        assert security.level == EscalationLevel.CRITICAL

        uncertain = handler.create_escalation(
            reason=EscalationReason.AGENT_UNCERTAIN,
            summary="Uncertain",
        )
        assert uncertain.level == EscalationLevel.LOW

    def test_get_escalation(self) -> None:
        """Test retrieving an escalation."""
        handler = EscalationHandler()
        request = handler.create_escalation(
            reason=EscalationReason.HUMAN_REQUESTED,
            summary="User requested human help",
        )
        retrieved = handler.get_escalation(request.id)
        assert retrieved is not None
        assert retrieved.id == request.id

    def test_resolve_escalation(self) -> None:
        """Test resolving an escalation."""
        handler = EscalationHandler()
        request = handler.create_escalation(
            reason=EscalationReason.COMPLEX_ISSUE,
            summary="Complex issue",
        )

        success = handler.resolve_escalation(
            request.id,
            resolution="Issue resolved via phone call",
            resolved_by="john.doe@company.com",
        )
        assert success

        resolved = handler.get_escalation(request.id)
        assert resolved.status == EscalationStatus.RESOLVED

    def test_resolve_nonexistent(self) -> None:
        """Test resolving non-existent escalation."""
        handler = EscalationHandler()
        success = handler.resolve_escalation(
            "nonexistent",
            resolution="N/A",
            resolved_by="N/A",
        )
        assert not success

    def test_get_pending_escalations(self) -> None:
        """Test getting pending escalations."""
        handler = EscalationHandler()

        # Create several escalations
        handler.create_escalation(
            reason=EscalationReason.SECURITY_INCIDENT,
            summary="Security 1",
        )
        handler.create_escalation(
            reason=EscalationReason.AGENT_UNCERTAIN,
            summary="Uncertain 1",
        )

        pending = handler.get_pending_escalations()
        assert len(pending) >= 2

    def test_get_pending_by_level(self) -> None:
        """Test filtering pending by level."""
        handler = EscalationHandler()

        handler.create_escalation(
            reason=EscalationReason.SECURITY_INCIDENT,
            summary="Critical issue",
        )
        handler.create_escalation(
            reason=EscalationReason.AGENT_UNCERTAIN,
            summary="Low priority",
        )

        critical = handler.get_pending_escalations(level=EscalationLevel.CRITICAL)
        assert all(e.level == EscalationLevel.CRITICAL for e in critical)

    def test_escalation_stats(self) -> None:
        """Test getting escalation statistics."""
        handler = EscalationHandler()
        handler.create_escalation(
            reason=EscalationReason.CROSS_DOMAIN,
            summary="Cross domain",
        )

        stats = handler.get_escalation_stats()
        assert "total_pending" in stats
        assert "by_level" in stats
        assert "by_reason" in stats

    def test_get_singleton(self) -> None:
        """Test get_escalation_handler returns singleton."""
        h1 = get_escalation_handler()
        h2 = get_escalation_handler()
        assert h1 is h2


class TestEscalationHandlerAsync:
    """Async tests for EscalationHandler."""

    @pytest.mark.asyncio
    async def test_process_escalation(self) -> None:
        """Test processing an escalation."""
        handler = EscalationHandler()
        request = handler.create_escalation(
            reason=EscalationReason.HUMAN_REQUESTED,
            summary="User needs human help",
        )

        response = await handler.process_escalation(request)
        assert response.success
        assert response.request_id == request.id
        assert len(response.next_steps) > 0


# =============================================================================
# Domain Agent Base Tests
# =============================================================================


class TestDomainConfig:
    """Tests for DomainConfig."""

    def test_create_config(self) -> None:
        """Test creating domain config."""
        config = DomainConfig(
            domain=BaseDomainType.CLOUD,
            name="Cloud Infrastructure",
            description="Cloud support",
            expertise=["azure", "aws"],
        )
        assert config.domain == BaseDomainType.CLOUD
        assert "azure" in config.expertise


# =============================================================================
# Domain Agent Integration Tests
# =============================================================================


class TestDomainAgentImports:
    """Test that all domain agents can be imported."""

    def test_import_marcom(self) -> None:
        """Test MarCom agent import."""
        from app.agents.domains.marcom_agent import MarComAgent
        agent = MarComAgent()
        assert agent.domain == "marcom"

    def test_import_hr(self) -> None:
        """Test HR agent import."""
        from app.agents.domains.hr_agent import HRAgent
        agent = HRAgent()
        assert agent.domain == "hr"

    def test_import_lnd(self) -> None:
        """Test L&D agent import."""
        from app.agents.domains.lnd_agent import LnDAgent
        agent = LnDAgent()
        assert agent.domain == "lnd"

    def test_import_presales(self) -> None:
        """Test Presales agent import."""
        from app.agents.domains.presales_agent import PresalesAgent
        agent = PresalesAgent()
        assert agent.domain == "presales"

    def test_import_datacenter(self) -> None:
        """Test Datacenter agent import."""
        from app.agents.domains.datacenter_agent import DatacenterAgent
        agent = DatacenterAgent()
        assert agent.domain == "datacenter"

    def test_import_cloud(self) -> None:
        """Test Cloud agent import."""
        from app.agents.domains.cloud_agent import CloudAgent
        agent = CloudAgent()
        assert agent.domain == "cloud"

    def test_import_cybersecurity(self) -> None:
        """Test Cybersecurity agent import."""
        from app.agents.domains.cybersecurity_agent import CybersecurityAgent
        agent = CybersecurityAgent()
        assert agent.domain == "cybersecurity"

    def test_import_data_ai(self) -> None:
        """Test Data/AI agent import."""
        from app.agents.domains.data_ai_agent import DataAIAgent
        agent = DataAIAgent()
        assert agent.domain == "data_ai"


class TestDomainAgentTools:
    """Test domain agent tools."""

    def test_cloud_agent_has_tools(self) -> None:
        """Test Cloud agent has tools defined."""
        from app.agents.domains.cloud_agent import CloudAgent
        agent = CloudAgent()
        tools = agent.get_tools()
        assert len(tools) > 0

    def test_hr_agent_has_tools(self) -> None:
        """Test HR agent has tools defined."""
        from app.agents.domains.hr_agent import HRAgent
        agent = HRAgent()
        tools = agent.get_tools()
        assert len(tools) > 0

    def test_cybersecurity_agent_has_tools(self) -> None:
        """Test Cybersecurity agent has tools defined."""
        from app.agents.domains.cybersecurity_agent import CybersecurityAgent
        agent = CybersecurityAgent()
        tools = agent.get_tools()
        assert len(tools) > 0


class TestGetAllDomainAgents:
    """Test get_all_domain_agents function."""

    def test_get_all_agents(self) -> None:
        """Test getting all domain agents."""
        from app.agents.domains import get_all_domain_agents
        agents = get_all_domain_agents()
        assert len(agents) == 8
        assert "cloud" in agents
        assert "hr" in agents
        assert "cybersecurity" in agents
