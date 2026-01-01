"""Tests for the evaluation framework."""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.evals.evaluators import (
    BaseEvaluator,
    EvaluationResult,
    FactualAccuracyEvaluator,
    ResponseQualityEvaluator,
    TaskCompletionEvaluator,
    create_evaluation_summary,
    evaluate_agent_response,
)
from app.agents.evals.datasets import (
    EvalDataset,
    TestCase,
    get_dataset,
    get_test_cases_by_tag,
    get_test_cases_by_difficulty,
)


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_create_result(self):
        """Test creating an evaluation result."""
        result = EvaluationResult(
            score=0.85,
            passed=True,
            feedback="Good response",
        )
        assert result.score == 0.85
        assert result.passed is True
        assert result.feedback == "Good response"
        assert result.details is None

    def test_create_result_with_details(self):
        """Test creating result with details."""
        result = EvaluationResult(
            score=0.5,
            passed=False,
            feedback="Needs improvement",
            details={"issues": ["too short"]},
        )
        assert result.details == {"issues": ["too short"]}


class TestResponseQualityEvaluator:
    """Tests for ResponseQualityEvaluator."""

    def test_good_response(self):
        """Test evaluation of a good response."""
        evaluator = ResponseQualityEvaluator(min_length=10)
        result = evaluator.evaluate(
            input_text="What is LangGraph?",
            output_text="LangGraph is a framework for building stateful agents. "
            "It provides tools for creating complex workflows with LLMs.",
        )
        assert result.score >= 0.7
        assert result.passed is True

    def test_empty_response(self):
        """Test evaluation of empty response."""
        evaluator = ResponseQualityEvaluator()
        result = evaluator.evaluate(
            input_text="What is LangGraph?",
            output_text="",
        )
        assert result.score == 0.0
        assert result.passed is False
        assert "empty" in result.feedback.lower()

    def test_too_short_response(self):
        """Test evaluation of too short response."""
        evaluator = ResponseQualityEvaluator(min_length=100)
        result = evaluator.evaluate(
            input_text="What is LangGraph?",
            output_text="LangGraph is a framework.",
        )
        assert result.score < 1.0
        assert "short" in result.feedback.lower()

    def test_missing_required_elements(self):
        """Test evaluation when required elements are missing."""
        evaluator = ResponseQualityEvaluator(
            min_length=10,
            required_elements=["LangGraph", "agents", "workflow"],
        )
        result = evaluator.evaluate(
            input_text="What is LangGraph?",
            output_text="This is a framework for building applications with LLMs.",
        )
        assert result.score < 1.0
        assert len(result.details["missing_elements"]) > 0

    def test_all_required_elements_present(self):
        """Test evaluation when all required elements present."""
        evaluator = ResponseQualityEvaluator(
            min_length=10,
            required_elements=["framework", "LLM"],
        )
        result = evaluator.evaluate(
            input_text="What is this?",
            output_text="This is a framework for building applications with LLMs.",
        )
        assert result.details["missing_elements"] == []


class TestTaskCompletionEvaluator:
    """Tests for TaskCompletionEvaluator."""

    def test_successful_task(self):
        """Test evaluation of successful task."""
        evaluator = TaskCompletionEvaluator()
        result = evaluator.evaluate(
            input_text="Generate a report",
            output_text="Here is your report. The task has been completed successfully.",
        )
        assert result.passed is True
        assert result.score >= 0.8

    def test_failed_task(self):
        """Test evaluation of failed task."""
        evaluator = TaskCompletionEvaluator()
        result = evaluator.evaluate(
            input_text="Generate a report",
            output_text="Sorry, I was unable to generate the report. An error occurred.",
        )
        assert result.passed is False
        assert result.score < 0.5

    def test_unclear_task_status(self):
        """Test evaluation when task status is unclear."""
        evaluator = TaskCompletionEvaluator()
        result = evaluator.evaluate(
            input_text="Generate a report",
            output_text="Processing your request now.",
        )
        assert result.score == 0.5
        assert result.passed is False
        assert "unclear" in result.feedback.lower()

    def test_custom_indicators(self):
        """Test custom success/failure indicators."""
        evaluator = TaskCompletionEvaluator(
            success_indicators=["complete", "finished"],
            failure_indicators=["broken", "crashed"],
        )
        result = evaluator.evaluate(
            input_text="Run the task",
            output_text="Task is now complete and finished.",
        )
        assert result.passed is True
        assert result.details["success_indicators"] == 2


class TestFactualAccuracyEvaluator:
    """Tests for FactualAccuracyEvaluator."""

    def test_no_facts_to_verify(self):
        """Test evaluation with no facts."""
        evaluator = FactualAccuracyEvaluator()
        result = evaluator.evaluate(
            input_text="Tell me about Python",
            output_text="Python is a programming language.",
        )
        assert result.score == 1.0
        assert result.passed is True

    def test_correct_facts(self):
        """Test evaluation with correct facts."""
        evaluator = FactualAccuracyEvaluator(
            facts={"Python": "programming language"}
        )
        result = evaluator.evaluate(
            input_text="Tell me about Python",
            output_text="Python is a programming language used for many purposes.",
        )
        assert result.passed is True
        assert result.details["verified"] == 1

    def test_incorrect_facts(self):
        """Test evaluation with incorrect facts."""
        evaluator = FactualAccuracyEvaluator(
            facts={"Python": "compiled language"}
        )
        result = evaluator.evaluate(
            input_text="Tell me about Python",
            output_text="Python is an interpreted programming language.",
        )
        # Pattern found but expected fact not present
        assert result.passed is False


class TestEvaluateAgentResponse:
    """Tests for evaluate_agent_response function."""

    def test_default_evaluators(self):
        """Test with default evaluators."""
        results = evaluate_agent_response(
            input_text="What is LangGraph?",
            output_text="LangGraph is a framework for building stateful agents. "
            "It has been successfully implemented by many teams.",
        )
        assert "response_quality" in results
        assert "task_completion" in results
        assert isinstance(results["response_quality"], EvaluationResult)

    def test_custom_evaluators(self):
        """Test with custom evaluators."""
        evaluators = [
            ResponseQualityEvaluator(min_length=10),
            FactualAccuracyEvaluator(facts={"test": "value"}),
        ]
        results = evaluate_agent_response(
            input_text="Test input",
            output_text="This is a test output with good value content.",
            evaluators=evaluators,
        )
        assert "response_quality" in results
        assert "factual_accuracy" in results

    def test_evaluator_error_handling(self):
        """Test that evaluator errors are handled gracefully."""

        class BrokenEvaluator(BaseEvaluator):
            name = "broken"

            def evaluate(self, input_text, output_text, expected=None):
                raise RuntimeError("Intentional error")

        results = evaluate_agent_response(
            input_text="Test",
            output_text="Output",
            evaluators=[BrokenEvaluator()],
        )
        assert "broken" in results
        assert results["broken"].score == 0.0
        assert "error" in results["broken"].feedback.lower()


class TestCreateEvaluationSummary:
    """Tests for create_evaluation_summary function."""

    def test_summary_format(self):
        """Test summary output format."""
        results = {
            "quality": EvaluationResult(0.8, True, "Good"),
            "completion": EvaluationResult(0.9, True, "Complete"),
        }
        summary = create_evaluation_summary(results)
        assert "Evaluation Summary" in summary
        assert "quality" in summary
        assert "completion" in summary
        assert "PASS" in summary
        assert "Overall Score" in summary

    def test_summary_with_failures(self):
        """Test summary with failed evaluations."""
        results = {
            "quality": EvaluationResult(0.4, False, "Poor response"),
            "completion": EvaluationResult(0.3, False, "Task failed"),
        }
        summary = create_evaluation_summary(results)
        assert "FAIL" in summary
        assert "NEEDS IMPROVEMENT" in summary

    def test_empty_results(self):
        """Test summary with empty results."""
        summary = create_evaluation_summary({})
        assert "Overall Score: 0.00" in summary


# =============================================================================
# Dataset Tests
# =============================================================================


class TestDatasets:
    """Tests for evaluation datasets."""

    def test_get_dataset_exists(self):
        """Test getting an existing dataset."""
        dataset = get_dataset("research")
        assert dataset is not None
        assert dataset.name == "research_agent_eval"
        assert len(dataset.test_cases) > 0

    def test_get_dataset_not_exists(self):
        """Test getting a non-existent dataset."""
        dataset = get_dataset("nonexistent")
        assert dataset is None

    def test_get_test_cases_by_tag(self):
        """Test filtering test cases by tag."""
        cases = get_test_cases_by_tag("security")
        assert len(cases) > 0
        for case in cases:
            assert "security" in case.tags

    def test_get_test_cases_by_difficulty(self):
        """Test filtering test cases by difficulty."""
        easy_cases = get_test_cases_by_difficulty("easy")
        hard_cases = get_test_cases_by_difficulty("hard")
        assert len(easy_cases) > 0
        assert len(hard_cases) > 0
        for case in easy_cases:
            assert case.difficulty == "easy"

    def test_test_case_creation(self):
        """Test TestCase dataclass creation."""
        case = TestCase(
            id="test-001",
            input="What is AI?",
            expected_keywords=["artificial", "intelligence"],
            tags=["basic"],
            difficulty="easy",
        )
        assert case.id == "test-001"
        assert "artificial" in case.expected_keywords
        assert case.difficulty == "easy"


# =============================================================================
# Multi-Turn Evaluator Tests
# =============================================================================


class TestMultiTurnEvaluators:
    """Tests for multi-turn conversation evaluators."""

    def test_conversation_turn_from_human_message(self):
        """Test creating ConversationTurn from HumanMessage."""
        from app.agents.evals.multi_turn_evaluator import ConversationTurn

        msg = HumanMessage(content="Hello, I need help")
        turn = ConversationTurn.from_message(msg, 1)

        assert turn.turn_number == 1
        assert turn.role == "user"
        assert turn.content == "Hello, I need help"
        assert turn.tool_calls == []

    def test_conversation_turn_from_ai_message(self):
        """Test creating ConversationTurn from AIMessage."""
        from app.agents.evals.multi_turn_evaluator import ConversationTurn

        msg = AIMessage(content="I can help you with that.")
        turn = ConversationTurn.from_message(msg, 2)

        assert turn.turn_number == 2
        assert turn.role == "assistant"
        assert turn.content == "I can help you with that."

    def test_conversation_turn_with_tool_calls(self):
        """Test creating ConversationTurn with tool calls."""
        from app.agents.evals.multi_turn_evaluator import ConversationTurn

        msg = AIMessage(
            content="Let me search for that.",
            tool_calls=[{"name": "search", "args": {"query": "test"}, "id": "call_123"}],
        )
        turn = ConversationTurn.from_message(msg, 1)

        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0]["name"] == "search"


class TestIntentCompletionEvaluator:
    """Tests for IntentCompletionEvaluator."""

    def test_all_intents_completed(self):
        """Test when all intents are completed."""
        from app.agents.evals.multi_turn_evaluator import IntentCompletionEvaluator

        evaluator = IntentCompletionEvaluator(
            intents=["password reset", "account help"]
        )
        result = evaluator.evaluate(
            input_text="I need help",
            output_text="I have helped you with the password reset and account help.",
        )

        assert result.score == 1.0
        assert result.passed is True
        assert result.details["completed"] == ["password reset", "account help"]

    def test_partial_intent_completion(self):
        """Test when some intents are incomplete."""
        from app.agents.evals.multi_turn_evaluator import IntentCompletionEvaluator

        evaluator = IntentCompletionEvaluator(
            intents=["password reset", "VPN setup", "email config"]
        )
        result = evaluator.evaluate(
            input_text="Help needed",
            output_text="I have completed the password reset for you.",
        )

        assert result.score < 1.0
        assert "password reset" in result.details["completed"]
        assert "VPN setup" in result.details["incomplete"]

    def test_no_intents_specified(self):
        """Test when no intents are specified."""
        from app.agents.evals.multi_turn_evaluator import IntentCompletionEvaluator

        evaluator = IntentCompletionEvaluator()
        result = evaluator.evaluate("input", "output")

        assert result.score == 1.0
        assert result.passed is True


class TestContextCoherenceEvaluator:
    """Tests for ContextCoherenceEvaluator."""

    def test_context_maintained(self):
        """Test when context is properly maintained."""
        from app.agents.evals.multi_turn_evaluator import ContextCoherenceEvaluator

        evaluator = ContextCoherenceEvaluator(
            context_requirements=["laptop", "Chrome"]
        )
        result = evaluator.evaluate(
            input_text="My laptop has Chrome issues",
            output_text="I see you're having issues with Chrome on your laptop. Let me help.",
        )

        assert result.score >= 0.7
        assert result.passed is True
        assert len(result.details["missing_context"]) == 0

    def test_missing_context(self):
        """Test when context is missing."""
        from app.agents.evals.multi_turn_evaluator import ContextCoherenceEvaluator

        evaluator = ContextCoherenceEvaluator(
            context_requirements=["laptop", "Chrome", "Windows"]
        )
        result = evaluator.evaluate(
            input_text="Help with laptop",
            output_text="I can help with your computer issues.",
        )

        assert result.score < 1.0
        assert len(result.details["missing_context"]) > 0


class TestToolSequenceEvaluator:
    """Tests for ToolSequenceEvaluator."""

    def test_correct_tool_sequence(self):
        """Test correct tool call sequence."""
        from app.agents.evals.multi_turn_evaluator import ToolSequenceEvaluator

        evaluator = ToolSequenceEvaluator(
            expected_sequence=["search", "create_ticket"]
        )
        result = evaluator.evaluate_tool_calls([
            {"name": "search", "args": {}},
            {"name": "create_ticket", "args": {}},
        ])

        assert result.score == 1.0
        assert result.passed is True
        assert result.details["matched"] == ["search", "create_ticket"]

    def test_missing_tools(self):
        """Test when expected tools are missing."""
        from app.agents.evals.multi_turn_evaluator import ToolSequenceEvaluator

        evaluator = ToolSequenceEvaluator(
            expected_sequence=["search", "create_ticket", "notify"]
        )
        result = evaluator.evaluate_tool_calls([
            {"name": "search", "args": {}},
        ])

        assert result.score < 1.0
        assert "create_ticket" not in result.details["matched"]

    def test_extra_tools_allowed(self):
        """Test with extra tools when allowed."""
        from app.agents.evals.multi_turn_evaluator import ToolSequenceEvaluator

        evaluator = ToolSequenceEvaluator(
            expected_sequence=["search"],
            allow_extra_tools=True,
        )
        result = evaluator.evaluate_tool_calls([
            {"name": "search", "args": {}},
            {"name": "extra_tool", "args": {}},
        ])

        assert result.score == 1.0
        assert result.passed is True


class TestConversationFlowEvaluator:
    """Tests for ConversationFlowEvaluator."""

    def test_good_flow(self):
        """Test good conversation flow."""
        from app.agents.evals.multi_turn_evaluator import ConversationFlowEvaluator

        evaluator = ConversationFlowEvaluator(min_response_length=10)
        result = evaluator.evaluate(
            input_text="Help me",
            output_text="I would be happy to help you with your request. "
            "Let me know what you need assistance with.",
        )

        assert result.passed is True
        assert result.score >= 0.7

    def test_too_brief_response(self):
        """Test response that is too brief."""
        from app.agents.evals.multi_turn_evaluator import ConversationFlowEvaluator

        evaluator = ConversationFlowEvaluator(min_response_length=100)
        result = evaluator.evaluate(
            input_text="Complex question",
            output_text="OK.",
        )

        assert result.score < 1.0
        assert "brief" in result.feedback.lower()

    def test_incomplete_response(self):
        """Test response that appears incomplete."""
        from app.agents.evals.multi_turn_evaluator import ConversationFlowEvaluator

        evaluator = ConversationFlowEvaluator()
        result = evaluator.evaluate(
            input_text="Question",
            output_text="I will help you with" * 10,  # No proper ending
        )

        assert result.score < 1.0


class TestMultiTurnEvaluator:
    """Tests for the comprehensive MultiTurnEvaluator."""

    def test_full_conversation_evaluation(self):
        """Test evaluating a full conversation."""
        from app.agents.evals.multi_turn_evaluator import (
            MultiTurnEvaluator,
            MultiTurnTestCase,
        )

        messages = [
            HumanMessage(content="I need help with my password"),
            AIMessage(content="I can help you with password reset. Let me guide you."),
            HumanMessage(content="Yes, please reset it"),
            AIMessage(content="Your password has been reset. Check your email for the new password."),
        ]

        test_case = MultiTurnTestCase(
            id="test-001",
            expected_intents=["password", "reset"],
            context_requirements=["password"],
        )

        evaluator = MultiTurnEvaluator()
        result = evaluator.evaluate_conversation(messages, test_case)

        assert result.overall_score > 0
        assert len(result.turn_by_turn) == 4
        assert "score" in result.intent_completion

    def test_evaluate_multi_turn_convenience_function(self):
        """Test the convenience function for multi-turn evaluation."""
        from app.agents.evals.multi_turn_evaluator import evaluate_multi_turn_conversation

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hello! How can I assist you today?"),
        ]

        result = evaluate_multi_turn_conversation(
            messages=messages,
            expected_intents=["greet"],
        )

        assert result.overall_score > 0
        assert isinstance(result.feedback, str)


# =============================================================================
# Business Metrics Tests
# =============================================================================


class TestBusinessMetrics:
    """Tests for IT support business metrics evaluators."""

    def test_ticket_metrics_creation(self):
        """Test TicketMetrics dataclass."""
        from app.agents.evals.business_metrics import TicketMetrics

        metrics = TicketMetrics(
            ticket_id="INC001",
            priority="high",
            response_time_ms=5000,
            resolution_time_ms=90000,
        )

        assert metrics.ticket_id == "INC001"
        assert metrics.priority == "high"

    def test_ticket_resolution_evaluator(self):
        """Test TicketResolutionEvaluator."""
        from app.agents.evals.business_metrics import TicketResolutionEvaluator

        evaluator = TicketResolutionEvaluator()

        result = evaluator.evaluate(
            input_text="My laptop won't start",
            output_text="I have resolved your issue. Your laptop needed a power reset.",
        )

        assert result.passed is True
        assert result.score >= 0.8

    def test_ticket_resolution_unresolved(self):
        """Test TicketResolutionEvaluator with unresolved ticket."""
        from app.agents.evals.business_metrics import TicketResolutionEvaluator

        evaluator = TicketResolutionEvaluator()

        result = evaluator.evaluate(
            input_text="Complex issue",
            output_text="Sorry, I cannot resolve this. Escalating to L2.",
        )

        assert result.passed is False
        assert result.score < 0.7


class TestEscalationEvaluator:
    """Tests for EscalationEvaluator."""

    def test_appropriate_escalation(self):
        """Test appropriate escalation is detected."""
        from app.agents.evals.business_metrics import EscalationEvaluator

        evaluator = EscalationEvaluator()

        result = evaluator.evaluate(
            input_text="There's a security breach in the system",
            output_text="This is a security incident. I'm escalating to the security team immediately.",
        )

        assert result.passed is True
        assert result.score >= 0.7

    def test_unnecessary_escalation(self):
        """Test detection of unnecessary escalation."""
        from app.agents.evals.business_metrics import EscalationEvaluator

        evaluator = EscalationEvaluator()

        result = evaluator.evaluate(
            input_text="How do I change my password?",
            output_text="I'm escalating this to a senior agent.",
        )

        # Simple password reset shouldn't need escalation
        assert result.score < 1.0


class TestResponseTimeEvaluator:
    """Tests for ResponseTimeEvaluator."""

    def test_fast_response(self):
        """Test evaluation of fast response time."""
        from app.agents.evals.business_metrics import ResponseTimeEvaluator

        evaluator = ResponseTimeEvaluator(target_first_response_ms=30000)  # 30 seconds
        result = evaluator.evaluate_timing(
            response_time_ms=10000,  # 10 seconds
            priority="medium",
        )

        assert result.passed is True
        assert result.score >= 0.9

    def test_slow_response(self):
        """Test evaluation of slow response time."""
        from app.agents.evals.business_metrics import ResponseTimeEvaluator

        evaluator = ResponseTimeEvaluator(target_first_response_ms=5000)  # 5 seconds
        result = evaluator.evaluate_timing(
            response_time_ms=20000,  # 20 seconds - 4x the target
            priority="medium",
        )

        assert result.passed is False
        assert result.score < 0.7


class TestUserSatisfactionEvaluator:
    """Tests for UserSatisfactionEvaluator."""

    def test_positive_indicators(self):
        """Test detection of positive satisfaction indicators."""
        from app.agents.evals.business_metrics import UserSatisfactionEvaluator

        evaluator = UserSatisfactionEvaluator()
        result = evaluator.evaluate(
            input_text="Can you help?",
            output_text="Absolutely! I've resolved your issue. Is there anything else I can help with?",
        )

        assert result.score >= 0.7
        assert result.passed is True

    def test_negative_indicators(self):
        """Test detection of negative satisfaction indicators."""
        from app.agents.evals.business_metrics import UserSatisfactionEvaluator

        evaluator = UserSatisfactionEvaluator()
        result = evaluator.evaluate(
            input_text="Help!",
            output_text="Sorry, I cannot help with that. I don't understand your request.",
        )

        assert result.score < 0.8


class TestSLAComplianceEvaluator:
    """Tests for SLAComplianceEvaluator."""

    def test_sla_met(self):
        """Test when SLA is met."""
        from app.agents.evals.business_metrics import (
            SLAComplianceEvaluator,
            TicketMetrics,
        )

        evaluator = SLAComplianceEvaluator()
        metrics = TicketMetrics(
            ticket_id="INC001",
            priority="medium",
            response_time_ms=60000,  # 1 minute - under 30 min target
            resolution_time_ms=1200000,  # 20 minutes - under 8 hour target
            resolved=True,
        )

        result = evaluator.evaluate_sla(metrics)

        assert result.passed is True
        assert result.score >= 0.8

    def test_sla_breached(self):
        """Test when SLA is breached."""
        from app.agents.evals.business_metrics import (
            SLAComplianceEvaluator,
            TicketMetrics,
        )

        evaluator = SLAComplianceEvaluator()
        metrics = TicketMetrics(
            ticket_id="INC001",
            priority="critical",
            response_time_ms=600000,  # 10 minutes - over 5 min critical target
            resolution_time_ms=7200000,  # 2 hours - over 1 hour critical target
            resolved=True,
        )

        result = evaluator.evaluate_sla(metrics)

        assert result.passed is False
        assert result.score < 0.7


class TestEvaluateITSupportInteraction:
    """Tests for the convenience evaluation function."""

    def test_full_evaluation(self):
        """Test full IT support interaction evaluation."""
        from app.agents.evals.business_metrics import (
            evaluate_it_support_interaction,
            TicketMetrics,
        )

        metrics = TicketMetrics(
            ticket_id="INC001",
            priority="medium",
            response_time_ms=5000,
            resolution_time_ms=60000,
            resolved=True,
        )

        results = evaluate_it_support_interaction(
            input_text="My email isn't working",
            output_text="I've resolved your email issue. Your account was locked and I've unlocked it.",
            metrics=metrics,
        )

        assert "ticket_resolution" in results
        assert "user_satisfaction_proxy" in results


# =============================================================================
# Regression Runner Tests
# =============================================================================


class TestRegressionConfig:
    """Tests for RegressionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from app.agents.evals.regression_runner import RegressionConfig

        config = RegressionConfig()
        assert config.pass_threshold == 0.7
        assert config.max_concurrent == 5
        assert config.timeout_seconds == 60
        assert config.output_format == "json"

    def test_from_env(self, monkeypatch):
        """Test loading config from environment."""
        from app.agents.evals.regression_runner import RegressionConfig

        monkeypatch.setenv("EVAL_THRESHOLD_PASS", "0.8")
        monkeypatch.setenv("EVAL_MAX_CONCURRENT", "10")
        monkeypatch.setenv("EVAL_OUTPUT_FORMAT", "markdown")

        config = RegressionConfig.from_env()

        assert config.pass_threshold == 0.8
        assert config.max_concurrent == 10
        assert config.output_format == "markdown"


class TestTestResult:
    """Tests for TestResult dataclass."""

    def test_create_result(self):
        """Test creating a test result."""
        from app.agents.evals.regression_runner import TestResult

        result = TestResult(
            test_case_id="test-001",
            passed=True,
            score=0.85,
            execution_time_ms=1500,
            input="Test input",
            output="Test output",
        )

        assert result.test_case_id == "test-001"
        assert result.passed is True
        assert result.score == 0.85

    def test_result_with_error(self):
        """Test creating a result with an error."""
        from app.agents.evals.regression_runner import TestResult

        result = TestResult(
            test_case_id="test-002",
            passed=False,
            score=0.0,
            error="Timeout error",
        )

        assert result.passed is False
        assert result.error == "Timeout error"


class TestRegressionReport:
    """Tests for RegressionReport dataclass."""

    def test_overall_passed_above_threshold(self):
        """Test overall_passed when above threshold."""
        from app.agents.evals.regression_runner import RegressionReport

        report = RegressionReport(
            pass_rate=80.0,
            config={"pass_threshold": 0.7},
        )

        assert report.overall_passed is True

    def test_overall_passed_below_threshold(self):
        """Test overall_passed when below threshold."""
        from app.agents.evals.regression_runner import RegressionReport

        report = RegressionReport(
            pass_rate=50.0,
            config={"pass_threshold": 0.7},
        )

        assert report.overall_passed is False


class TestRegressionRunner:
    """Tests for RegressionRunner."""

    @pytest.fixture
    def mock_agent_func(self):
        """Create a mock agent function."""
        async def agent(input_text: str) -> str:
            return f"Response to: {input_text}"
        return agent

    @pytest.fixture
    def simple_dataset(self):
        """Create a simple test dataset."""
        return EvalDataset(
            name="test_dataset",
            description="Test dataset",
            agent_type="test",
            test_cases=[
                TestCase(
                    id="test-001",
                    input="What is AI?",
                    expected_keywords=["artificial", "intelligence"],
                ),
                TestCase(
                    id="test-002",
                    input="Hello",
                    expected_keywords=["hello", "hi"],
                ),
            ],
        )

    @pytest.mark.asyncio
    async def test_run_test_case(self, mock_agent_func):
        """Test running a single test case."""
        from app.agents.evals.regression_runner import RegressionRunner, RegressionConfig

        runner = RegressionRunner(
            config=RegressionConfig(verbose=False)
        )
        test_case = TestCase(
            id="test-001",
            input="Test question",
        )

        result = await runner.run_test_case(test_case, mock_agent_func)

        assert result.test_case_id == "test-001"
        assert result.output == "Response to: Test question"
        assert result.execution_time_ms >= 0  # May be 0 on fast systems

    @pytest.mark.asyncio
    async def test_run_test_case_with_timeout(self):
        """Test test case with timeout."""
        from app.agents.evals.regression_runner import RegressionRunner, RegressionConfig

        async def slow_agent(input_text: str) -> str:
            await asyncio.sleep(10)
            return "Done"

        runner = RegressionRunner(
            config=RegressionConfig(timeout_seconds=1, verbose=False)
        )
        test_case = TestCase(id="test-001", input="Test")

        result = await runner.run_test_case(test_case, slow_agent)

        assert result.passed is False
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_run_dataset(self, mock_agent_func, simple_dataset):
        """Test running a full dataset."""
        from app.agents.evals.regression_runner import RegressionRunner, RegressionConfig

        runner = RegressionRunner(
            config=RegressionConfig(verbose=False)
        )

        report = await runner.run_dataset(simple_dataset, mock_agent_func)

        assert report.dataset_name == "test_dataset"
        assert report.total_tests == 2
        assert len(report.results) == 2
        assert report.duration_seconds > 0


class TestRegressionRunnerConvenienceFunctions:
    """Tests for convenience functions."""

    def test_run_regression_sync_invalid_agent(self):
        """Test sync runner with invalid agent type."""
        from app.agents.evals.regression_runner import run_regression_sync

        def dummy_agent(x):
            return x

        with pytest.raises(ValueError, match="No dataset found"):
            run_regression_sync(dummy_agent, "invalid_agent_type")

    def test_check_regression_passed(self):
        """Test check_regression_passed function."""
        from app.agents.evals.regression_runner import (
            RegressionReport,
            check_regression_passed,
        )

        passing_report = RegressionReport(
            pass_rate=85.0,
            config={"pass_threshold": 0.7},
        )
        failing_report = RegressionReport(
            pass_rate=50.0,
            config={"pass_threshold": 0.7},
        )

        assert check_regression_passed(passing_report) is True
        assert check_regression_passed(failing_report) is False


# =============================================================================
# LangSmith Evaluator Tests (Mocked)
# =============================================================================


class TestLangSmithConfig:
    """Tests for LangSmithConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from app.agents.evals.langsmith_evaluator import LangSmithConfig

        config = LangSmithConfig()

        assert config.api_key is None
        assert config.project_name == "enterprise-agents-eval"
        assert config.sampling_rate == 0.1

    def test_from_env(self, monkeypatch):
        """Test loading config from environment."""
        from app.agents.evals.langsmith_evaluator import LangSmithConfig

        monkeypatch.setenv("LANGCHAIN_API_KEY", "test-api-key")
        monkeypatch.setenv("EVAL_PROJECT_NAME", "test-project")
        monkeypatch.setenv("EVAL_ONLINE_SAMPLING_RATE", "0.25")

        config = LangSmithConfig.from_env()

        assert config.api_key == "test-api-key"
        assert config.project_name == "test-project"
        assert config.sampling_rate == 0.25


class TestEvaluationExperiment:
    """Tests for EvaluationExperiment dataclass."""

    def test_create_experiment(self):
        """Test creating an experiment."""
        from app.agents.evals.langsmith_evaluator import EvaluationExperiment

        experiment = EvaluationExperiment(
            name="test-experiment",
            dataset_name="test-dataset",
        )

        assert experiment.name == "test-experiment"
        assert experiment.dataset_name == "test-dataset"
        assert len(experiment.id) > 0
        assert experiment.results == []
        assert experiment.metrics == {}


class TestLangSmithEvaluator:
    """Tests for LangSmithEvaluator (with mocked client)."""

    def test_register_evaluator(self):
        """Test registering custom evaluator."""
        from app.agents.evals.langsmith_evaluator import LangSmithEvaluator, LangSmithConfig

        evaluator = LangSmithEvaluator(
            config=LangSmithConfig(api_key="test-key")
        )
        custom_evaluator = ResponseQualityEvaluator()

        evaluator.register_evaluator(custom_evaluator)

        assert custom_evaluator in evaluator._evaluators

    def test_should_evaluate_online_sampling(self):
        """Test sampling rate for online evaluation."""
        from app.agents.evals.langsmith_evaluator import LangSmithEvaluator, LangSmithConfig

        # With 100% sampling rate, should always evaluate
        evaluator = LangSmithEvaluator(
            config=LangSmithConfig(api_key="test", sampling_rate=1.0)
        )
        assert evaluator.should_evaluate_online() is True

        # With 0% sampling rate, should never evaluate
        evaluator_no_sample = LangSmithEvaluator(
            config=LangSmithConfig(api_key="test", sampling_rate=0.0)
        )
        assert evaluator_no_sample.should_evaluate_online() is False

    def test_calculate_metrics(self):
        """Test metrics calculation."""
        from app.agents.evals.langsmith_evaluator import LangSmithEvaluator, LangSmithConfig

        evaluator = LangSmithEvaluator(config=LangSmithConfig(api_key="test"))

        results = [
            {"evaluations": {"quality": {"score": 0.8}, "completion": {"score": 0.9}}},
            {"evaluations": {"quality": {"score": 0.6}, "completion": {"score": 0.7}}},
        ]

        metrics = evaluator._calculate_metrics(results)

        assert "quality_avg" in metrics
        assert "completion_avg" in metrics
        assert metrics["quality_avg"] == 0.7  # (0.8 + 0.6) / 2
        assert metrics["completion_avg"] == 0.8  # (0.9 + 0.7) / 2


class TestLangSmithSingleton:
    """Tests for LangSmith singleton pattern."""

    def test_get_langsmith_evaluator_singleton(self):
        """Test singleton pattern."""
        from app.agents.evals.langsmith_evaluator import (
            get_langsmith_evaluator,
            reset_langsmith_evaluator,
        )

        reset_langsmith_evaluator()

        eval1 = get_langsmith_evaluator()
        eval2 = get_langsmith_evaluator()

        assert eval1 is eval2

        reset_langsmith_evaluator()

    def test_reset_langsmith_evaluator(self):
        """Test resetting singleton."""
        from app.agents.evals.langsmith_evaluator import (
            get_langsmith_evaluator,
            reset_langsmith_evaluator,
        )

        reset_langsmith_evaluator()
        eval1 = get_langsmith_evaluator()

        reset_langsmith_evaluator()
        eval2 = get_langsmith_evaluator()

        assert eval1 is not eval2

        reset_langsmith_evaluator()
