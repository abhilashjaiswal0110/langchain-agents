"""Tests for the evaluation framework."""

import pytest
from app.agents.evals.evaluators import (
    BaseEvaluator,
    EvaluationResult,
    FactualAccuracyEvaluator,
    ResponseQualityEvaluator,
    TaskCompletionEvaluator,
    create_evaluation_summary,
    evaluate_agent_response,
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
