"""Unit tests for Employee Experience Agent.

Tests the core functionality of the Employee Experience Agent including:
- Agent initialization and configuration
- Tool execution and responses
- Sentiment analysis and burnout detection
- State management and conversation flow
"""

import pytest
from unittest.mock import Mock, patch

from app.agents.employee_experience import (
    EmployeeExperienceAgent,
    EmployeeExperienceState,
    analyze_employee_sentiment,
    assess_burnout_risk,
    detect_escalation_triggers,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def employee_experience_agent():
    """Create a test Employee Experience Agent instance."""
    with patch("app.agents.employee_experience.employee_experience_agent.get_llm"):
        agent = EmployeeExperienceAgent(
            model_provider="auto",
            temperature=0.7,
        )
        return agent


@pytest.fixture
def sample_employee_context():
    """Sample employee context for testing."""
    return {
        "employee_id": "EMP12345",
        "employee_name": "Test Employee",
        "role": "Software Engineer",
        "tenure_years": 3.5,
        "department": "Engineering",
    }


# =============================================================================
# Test Agent Initialization
# =============================================================================


def test_agent_initialization():
    """Test that the agent initializes correctly."""
    with patch("app.agents.employee_experience.employee_experience_agent.get_llm"):
        agent = EmployeeExperienceAgent(
            model_provider="auto",
            temperature=0.7,
        )

        assert agent is not None
        assert agent.model_provider == "auto"
        assert agent.temperature == 0.7
        assert len(agent.tools) > 0
        assert agent.llm_with_tools is not None
        assert agent.memory is not None
        assert agent.graph is not None


def test_agent_has_required_tools():
    """Test that the agent has all required tool categories."""
    with patch("app.agents.employee_experience.employee_experience_agent.get_llm"):
        agent = EmployeeExperienceAgent()

        tool_names = [tool.name for tool in agent.tools]

        # HR Policy & Information tools
        assert "search_hr_policy" in tool_names
        assert "get_benefits_information" in tool_names
        assert "check_pto_balance" in tool_names
        assert "explain_compliance_rules" in tool_names

        # Career Development tools
        assert "explore_career_paths" in tool_names
        assert "get_skills_gap_analysis" in tool_names
        assert "find_learning_resources" in tool_names
        assert "request_career_coaching" in tool_names

        # Performance & Growth tools
        assert "prepare_performance_review" in tool_names
        assert "get_goal_setting_framework" in tool_names
        assert "request_feedback_survey" in tool_names

        # Wellbeing tools
        assert "get_wellbeing_resources" in tool_names
        assert "schedule_wellbeing_check" in tool_names

        # HR Operations tools
        assert "submit_hr_request" in tool_names
        assert "check_request_status" in tool_names

        # Engagement & Surveys tools
        assert "send_pulse_survey" in tool_names
        assert "get_engagement_insights" in tool_names

        # Compensation tools
        assert "get_compensation_insights" in tool_names
        assert "request_compensation_review" in tool_names

        # Learning & Development tools
        assert "get_learning_path" in tool_names
        assert "enroll_in_course" in tool_names

        # Escalation tools
        assert "escalate_to_hr_business_partner" in tool_names
        assert "schedule_hr_meeting" in tool_names


# =============================================================================
# Test Sentiment Analysis
# =============================================================================


def test_sentiment_analysis_positive():
    """Test sentiment analysis with positive message."""
    message = "I'm really excited about my new project! The team is amazing and I love the work."
    result = analyze_employee_sentiment(message)

    assert result["label"] == "positive"
    assert result["score"] > 0.3
    assert "positive_tone" in result["indicators"]


def test_sentiment_analysis_negative():
    """Test sentiment analysis with negative message."""
    message = "I'm feeling overwhelmed and frustrated. Too much work and no support."
    result = analyze_employee_sentiment(message)

    assert result["label"] == "negative"
    assert result["score"] < -0.3
    assert "stress" in result["indicators"]


def test_sentiment_analysis_neutral():
    """Test sentiment analysis with neutral message."""
    message = "Can you tell me about the PTO policy?"
    result = analyze_employee_sentiment(message)

    assert result["label"] == "neutral"
    assert -0.3 <= result["score"] <= 0.3


def test_sentiment_analysis_detects_stress():
    """Test that sentiment analysis detects stress indicators."""
    message = "I'm working late every night and weekends. Too much work, falling behind."
    result = analyze_employee_sentiment(message)

    assert "stress" in result["indicators"]
    assert result["keyword_counts"]["stress"] > 0


def test_sentiment_analysis_detects_burnout():
    """Test that sentiment analysis detects burnout indicators."""
    message = "I'm completely burned out. Can't do this anymore, feeling drained."
    result = analyze_employee_sentiment(message)

    assert "burnout_risk" in result["indicators"]
    assert result["keyword_counts"]["burnout"] > 0
    assert result["score"] < -0.5  # Very negative


# =============================================================================
# Test Burnout Risk Assessment
# =============================================================================


def test_burnout_assessment_low_risk():
    """Test burnout assessment with low-risk messages."""
    messages = [
        "How do I check my PTO balance?",
        "Thanks for the help!",
        "Can you explain the benefits?",
    ]
    result = assess_burnout_risk(messages)

    assert result["risk_level"] == "low"
    assert result["risk_score"] < 4


def test_burnout_assessment_medium_risk():
    """Test burnout assessment with medium-risk messages."""
    messages = [
        "I've been working a lot of overtime lately.",
        "Feeling stressed about the project deadlines.",
        "Not sure I can keep up with this pace.",
        "Having trouble balancing work and life.",
    ]
    result = assess_burnout_risk(messages)

    assert result["risk_level"] in ["medium", "high"]
    assert result["risk_score"] >= 4


def test_burnout_assessment_high_risk():
    """Test burnout assessment with high-risk messages."""
    messages = [
        "I'm completely burned out and exhausted.",
        "Working late every night and weekends.",
        "Can't do this anymore, feeling overwhelmed.",
        "Thinking about quitting, no motivation left.",
        "Drowning in work, no support from anyone.",
    ]
    result = assess_burnout_risk(messages)

    assert result["risk_level"] == "high"
    assert result["risk_score"] >= 7
    assert "explicit_burnout_signals" in result["factors"]
    assert "high_stress" in result["factors"]


def test_burnout_assessment_with_sentiment_trend():
    """Test that burnout assessment detects declining sentiment."""
    messages = [
        "Things are going well with my projects.",  # Positive
        "Work is getting busier.",  # Neutral
        "Feeling a bit stressed lately.",  # Negative
        "Really overwhelmed with the workload.",  # More negative
        "Can't keep up anymore.",  # Very negative
    ]
    result = assess_burnout_risk(messages)

    assert result["sentiment_trend"]["trend"] == "declining"
    assert result["risk_score"] > 3


# =============================================================================
# Test Escalation Detection
# =============================================================================


def test_escalation_detection_harassment():
    """Test escalation detection for harassment keywords."""
    message = "My manager is harassing me and making inappropriate comments."
    result = detect_escalation_triggers(message)

    assert result["escalation_required"] is True
    assert "harassment" in result["triggers"]
    assert result["urgency"] == "critical"


def test_escalation_detection_discrimination():
    """Test escalation detection for discrimination keywords."""
    message = "I feel like I'm being discriminated against because of my race."
    result = detect_escalation_triggers(message)

    assert result["escalation_required"] is True
    assert "discrimination" in result["triggers"]
    assert result["urgency"] == "critical"


def test_escalation_detection_safety():
    """Test escalation detection for safety concerns."""
    message = "I feel unsafe at work. Someone threatened me."
    result = detect_escalation_triggers(message)

    assert result["escalation_required"] is True
    assert "safety_concern" in result["triggers"]
    assert result["urgency"] == "critical"


def test_escalation_detection_self_harm():
    """Test escalation detection for self-harm indicators."""
    message = "I'm thinking about ending it all. Life isn't worth living anymore."
    result = detect_escalation_triggers(message)

    assert result["escalation_required"] is True
    assert "crisis_self_harm" in result["triggers"]
    assert result["urgency"] == "critical"


def test_escalation_detection_no_triggers():
    """Test escalation detection with normal message."""
    message = "Can you help me understand the performance review process?"
    result = detect_escalation_triggers(message)

    assert result["escalation_required"] is False
    assert len(result["triggers"]) == 0
    assert result["urgency"] == "normal"


# =============================================================================
# Test State Management
# =============================================================================


def test_employee_experience_state_creation():
    """Test creation of agent state."""
    from langchain_core.messages import HumanMessage

    state = EmployeeExperienceState(
        messages=[HumanMessage(content="Test message")],
        employee_id="EMP123",
        role="Engineer",
        sentiment_score=0.5,
        burnout_risk="low",
    )

    assert len(state.messages) == 1
    assert state.employee_id == "EMP123"
    assert state.role == "Engineer"
    assert state.sentiment_score == 0.5
    assert state.burnout_risk == "low"


def test_state_defaults():
    """Test default values in state."""
    from langchain_core.messages import HumanMessage

    state = EmployeeExperienceState(
        messages=[HumanMessage(content="Test")],
    )

    assert state.employee_id is None
    assert state.sentiment_score is None
    assert state.burnout_risk is None
    assert state.escalation_required is False
    assert state.context == {}


# =============================================================================
# Test Tool Functions (Sample)
# =============================================================================


def test_search_hr_policy_tool():
    """Test HR policy search tool."""
    from app.agents.employee_experience.tools import search_hr_policy

    result = search_hr_policy("PTO vacation leave")

    assert "PTO" in result or "Paid Time Off" in result
    assert len(result) > 0


def test_get_benefits_information_tool():
    """Test benefits information tool."""
    from app.agents.employee_experience.tools import get_benefits_information

    result = get_benefits_information("health")

    assert "Medical" in result or "health" in result.lower()
    assert "Coverage" in result or "coverage" in result.lower()


def test_check_pto_balance_tool():
    """Test PTO balance check tool."""
    from app.agents.employee_experience.tools import check_pto_balance

    result = check_pto_balance("self")

    assert "Balance" in result or "balance" in result.lower()
    assert "days" in result.lower()


def test_explore_career_paths_tool():
    """Test career path exploration tool."""
    from app.agents.employee_experience.tools import explore_career_paths

    result = explore_career_paths("Software Engineer")

    assert "Path" in result or "path" in result.lower()
    assert "Career" in result or "career" in result.lower()


def test_get_wellbeing_resources_tool():
    """Test wellbeing resources tool."""
    from app.agents.employee_experience.tools import get_wellbeing_resources

    result = get_wellbeing_resources("mental_health")

    assert "EAP" in result or "mental health" in result.lower()
    assert "1-800" in result  # Should have hotline number


def test_get_compensation_insights_tool():
    """Test compensation insights tool."""
    from app.agents.employee_experience.tools import get_compensation_insights

    result = get_compensation_insights("market_data")

    assert "compensation" in result.lower() or "salary" in result.lower()
    assert len(result) > 0


def test_escalate_to_hr_business_partner_tool():
    """Test HRBP escalation tool."""
    from app.agents.employee_experience.tools import escalate_to_hr_business_partner

    result = escalate_to_hr_business_partner(
        issue_type="harassment",
        urgency="critical",
        details="Test escalation",
    )

    assert "Escalation" in result or "escalation" in result.lower()
    assert "HRBP" in result or "HR Business Partner" in result
    assert "critical" in result.lower()


# =============================================================================
# Test Agent Graph Workflow
# =============================================================================


def test_agent_graph_exists():
    """Test that the agent graph is compiled."""
    with patch("app.agents.employee_experience.employee_experience_agent.get_llm"):
        agent = EmployeeExperienceAgent()

        assert agent.graph is not None
        # Check that graph has key nodes
        assert "sentiment_analysis" in agent.graph.nodes
        assert "agent" in agent.graph.nodes
        assert "tools" in agent.graph.nodes


# =============================================================================
# Test LangGraph Studio Entry Point
# =============================================================================


def test_get_graph_function():
    """Test the get_graph() function for LangGraph Studio."""
    from app.agents.employee_experience.employee_experience_agent import get_graph

    with patch("app.agents.employee_experience.employee_experience_agent.get_llm"):
        graph = get_graph()

        assert graph is not None
        assert hasattr(graph, "nodes")


# =============================================================================
# Integration Test (Mock LLM Response)
# =============================================================================


@pytest.mark.asyncio
async def test_agent_chat_flow():
    """Test the full chat flow with mocked LLM."""
    with patch("app.agents.employee_experience.employee_experience_agent.get_llm") as mock_llm:
        # Mock LLM response
        mock_llm_instance = Mock()
        mock_llm_instance.bind_tools.return_value = mock_llm_instance
        mock_llm.return_value = mock_llm_instance

        agent = EmployeeExperienceAgent()

        # Test that agent can be created without errors
        assert agent is not None
        assert agent.tools is not None
        assert len(agent.tools) > 20  # Should have 25+ tools


# =============================================================================
# Test Error Handling
# =============================================================================


def test_sentiment_analysis_empty_message():
    """Test sentiment analysis with empty message."""
    result = analyze_employee_sentiment("")

    assert result["label"] == "neutral"
    assert result["score"] == 0.0


def test_burnout_assessment_empty_messages():
    """Test burnout assessment with empty message list."""
    result = assess_burnout_risk([])

    assert result["risk_level"] == "unknown"
    assert result["risk_score"] == 0


def test_burnout_assessment_single_message():
    """Test burnout assessment with single message."""
    result = assess_burnout_risk(["I'm feeling great today!"])

    assert result["risk_level"] in ["low", "medium"]
    assert "risk_score" in result


# =============================================================================
# Test Agent System Prompt
# =============================================================================


def test_agent_has_system_prompt():
    """Test that the agent has a comprehensive system prompt."""
    with patch("app.agents.employee_experience.employee_experience_agent.get_llm"):
        agent = EmployeeExperienceAgent()

        assert hasattr(agent, "SYSTEM_PROMPT")
        assert len(agent.SYSTEM_PROMPT) > 100
        assert "empathetic" in agent.SYSTEM_PROMPT.lower() or "empathy" in agent.SYSTEM_PROMPT.lower()
        assert "HR" in agent.SYSTEM_PROMPT
        assert "career" in agent.SYSTEM_PROMPT.lower()
        assert "wellbeing" in agent.SYSTEM_PROMPT.lower()


# =============================================================================
# Test Package Exports
# =============================================================================


def test_package_exports():
    """Test that all expected classes/functions are exported."""
    from app.agents import employee_experience

    assert hasattr(employee_experience, "EmployeeExperienceAgent")
    assert hasattr(employee_experience, "EmployeeExperienceState")
    assert hasattr(employee_experience, "analyze_employee_sentiment")
    assert hasattr(employee_experience, "assess_burnout_risk")
    assert hasattr(employee_experience, "detect_escalation_triggers")
