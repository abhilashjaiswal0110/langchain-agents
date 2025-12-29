"""Custom evaluators for enterprise IT agents.

This module provides evaluation metrics for assessing
agent response quality and task completion.

Following Enterprise Development Standards:
- Data Architect: Structured evaluation metrics
- Software Engineer: Type-safe, testable evaluators
"""

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel


@dataclass
class EvaluationResult:
    """Result from an evaluation."""

    score: float  # 0.0 to 1.0
    passed: bool
    feedback: str
    details: dict[str, Any] | None = None


class BaseEvaluator:
    """Base class for evaluators."""

    name: str = "base"

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: str | None = None,
    ) -> EvaluationResult:
        """Evaluate an agent response.

        Args:
            input_text: User input
            output_text: Agent output
            expected: Expected output (optional)

        Returns:
            Evaluation result
        """
        raise NotImplementedError


class ResponseQualityEvaluator(BaseEvaluator):
    """Evaluates the quality of agent responses."""

    name = "response_quality"

    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 10000,
        required_elements: list[str] | None = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            min_length: Minimum response length
            max_length: Maximum response length
            required_elements: Required keywords/phrases
        """
        self.min_length = min_length
        self.max_length = max_length
        self.required_elements = required_elements or []

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: str | None = None,
    ) -> EvaluationResult:
        """Evaluate response quality."""
        issues = []
        score = 1.0

        # Length check
        if len(output_text) < self.min_length:
            issues.append(f"Response too short ({len(output_text)} < {self.min_length})")
            score -= 0.3
        elif len(output_text) > self.max_length:
            issues.append(f"Response too long ({len(output_text)} > {self.max_length})")
            score -= 0.1

        # Required elements check
        output_lower = output_text.lower()
        missing_elements = []
        for element in self.required_elements:
            if element.lower() not in output_lower:
                missing_elements.append(element)

        if missing_elements:
            issues.append(f"Missing elements: {', '.join(missing_elements)}")
            score -= 0.2 * len(missing_elements)

        # Empty response check
        if not output_text.strip():
            return EvaluationResult(
                score=0.0,
                passed=False,
                feedback="Response is empty",
            )

        # Relevance check (simple keyword overlap)
        input_words = set(input_text.lower().split())
        output_words = set(output_text.lower().split())
        overlap = len(input_words & output_words)
        if overlap < 2 and len(input_words) > 3:
            issues.append("Response may not be relevant to input")
            score -= 0.2

        score = max(0.0, min(1.0, score))

        return EvaluationResult(
            score=score,
            passed=score >= 0.7,
            feedback="; ".join(issues) if issues else "Response quality is good",
            details={
                "length": len(output_text),
                "missing_elements": missing_elements,
            },
        )


class TaskCompletionEvaluator(BaseEvaluator):
    """Evaluates whether a task was completed successfully."""

    name = "task_completion"

    def __init__(
        self,
        success_indicators: list[str] | None = None,
        failure_indicators: list[str] | None = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            success_indicators: Phrases indicating success
            failure_indicators: Phrases indicating failure
        """
        self.success_indicators = success_indicators or [
            "successfully",
            "completed",
            "done",
            "here is",
            "here are",
        ]
        self.failure_indicators = failure_indicators or [
            "error",
            "failed",
            "unable to",
            "cannot",
            "sorry",
        ]

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: str | None = None,
    ) -> EvaluationResult:
        """Evaluate task completion."""
        output_lower = output_text.lower()

        success_count = sum(
            1 for indicator in self.success_indicators
            if indicator in output_lower
        )
        failure_count = sum(
            1 for indicator in self.failure_indicators
            if indicator in output_lower
        )

        # Calculate score
        if failure_count > success_count:
            score = 0.3
            passed = False
            feedback = "Task appears to have failed"
        elif success_count > 0:
            score = 0.8 + (0.2 * min(success_count / 3, 1))
            passed = True
            feedback = "Task appears completed successfully"
        else:
            score = 0.5
            passed = False
            feedback = "Task completion status unclear"

        return EvaluationResult(
            score=score,
            passed=passed,
            feedback=feedback,
            details={
                "success_indicators": success_count,
                "failure_indicators": failure_count,
            },
        )


class FactualAccuracyEvaluator(BaseEvaluator):
    """Evaluates factual accuracy of responses."""

    name = "factual_accuracy"

    def __init__(self, facts: dict[str, str] | None = None) -> None:
        """Initialize with known facts.

        Args:
            facts: Dictionary of fact checks (pattern -> expected)
        """
        self.facts = facts or {}

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: str | None = None,
    ) -> EvaluationResult:
        """Evaluate factual accuracy."""
        if not self.facts:
            return EvaluationResult(
                score=1.0,
                passed=True,
                feedback="No facts to verify",
            )

        verified = 0
        failed = []

        for pattern, expected_fact in self.facts.items():
            if pattern.lower() in output_text.lower():
                if expected_fact.lower() in output_text.lower():
                    verified += 1
                else:
                    failed.append(f"Incorrect fact about: {pattern}")

        if not verified and not failed:
            return EvaluationResult(
                score=0.5,
                passed=True,
                feedback="No relevant facts found in response",
            )

        total = verified + len(failed)
        score = verified / total if total > 0 else 0.5

        return EvaluationResult(
            score=score,
            passed=len(failed) == 0,
            feedback="; ".join(failed) if failed else "All facts verified",
            details={"verified": verified, "failed": len(failed)},
        )


def evaluate_agent_response(
    input_text: str,
    output_text: str,
    evaluators: list[BaseEvaluator] | None = None,
    expected: str | None = None,
) -> dict[str, EvaluationResult]:
    """Run multiple evaluators on an agent response.

    Args:
        input_text: User input
        output_text: Agent output
        evaluators: List of evaluators to run
        expected: Expected output (optional)

    Returns:
        Dictionary of evaluator name to result
    """
    if evaluators is None:
        evaluators = [
            ResponseQualityEvaluator(),
            TaskCompletionEvaluator(),
        ]

    results = {}
    for evaluator in evaluators:
        try:
            result = evaluator.evaluate(input_text, output_text, expected)
            results[evaluator.name] = result
        except Exception as e:
            results[evaluator.name] = EvaluationResult(
                score=0.0,
                passed=False,
                feedback=f"Evaluation error: {e}",
            )

    return results


def create_evaluation_summary(results: dict[str, EvaluationResult]) -> str:
    """Create a human-readable evaluation summary.

    Args:
        results: Dictionary of evaluation results

    Returns:
        Formatted summary string
    """
    lines = ["Evaluation Summary", "=" * 40]

    total_score = 0
    for name, result in results.items():
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"\n{name}: [{status}] Score: {result.score:.2f}")
        lines.append(f"  {result.feedback}")
        total_score += result.score

    avg_score = total_score / len(results) if results else 0
    lines.append(f"\nOverall Score: {avg_score:.2f}")
    lines.append(f"Status: {'PASSED' if avg_score >= 0.7 else 'NEEDS IMPROVEMENT'}")

    return "\n".join(lines)
