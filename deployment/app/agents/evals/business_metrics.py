"""Business metrics evaluators for enterprise IT agents.

Provides IT support-specific evaluation metrics:
- Ticket resolution effectiveness
- Escalation appropriateness
- Response time tracking
- User satisfaction proxy
- SLA compliance
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from app.agents.evals.evaluators import BaseEvaluator, EvaluationResult


@dataclass
class TicketMetrics:
    """Metrics for a support ticket interaction.

    Attributes:
        ticket_id: Ticket identifier
        category: Issue category
        priority: Ticket priority
        response_time_ms: Time to first response
        resolution_time_ms: Time to resolution
        escalated: Whether ticket was escalated
        escalation_level: Level of escalation (L1, L2, L3)
        resolved: Whether issue was resolved
        user_satisfaction: User satisfaction score (1-5)
        agent_type: Type of agent handling ticket
    """

    ticket_id: str = ""
    category: str = "general"
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    response_time_ms: int = 0
    resolution_time_ms: int = 0
    escalated: bool = False
    escalation_level: str | None = None
    resolved: bool = False
    user_satisfaction: int | None = None
    agent_type: str = ""


class TicketResolutionEvaluator(BaseEvaluator):
    """Evaluates ticket resolution effectiveness."""

    name = "ticket_resolution"

    def __init__(
        self,
        resolution_indicators: list[str] | None = None,
        unresolved_indicators: list[str] | None = None,
        check_actionable: bool = True,
    ) -> None:
        """Initialize the evaluator.

        Args:
            resolution_indicators: Phrases indicating resolution.
            unresolved_indicators: Phrases indicating unresolved issues.
            check_actionable: Whether to check for actionable steps.
        """
        self.resolution_indicators = resolution_indicators or [
            "resolved",
            "fixed",
            "completed",
            "issue has been",
            "problem solved",
            "successfully",
            "done",
            "working now",
        ]
        self.unresolved_indicators = unresolved_indicators or [
            "unable to",
            "cannot",
            "failed",
            "escalating",
            "need more information",
            "waiting for",
            "pending",
            "investigating",
        ]
        self.check_actionable = check_actionable

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: str | None = None,
    ) -> EvaluationResult:
        """Evaluate ticket resolution."""
        output_lower = output_text.lower()

        # Count indicators
        resolution_count = sum(
            1 for ind in self.resolution_indicators
            if ind.lower() in output_lower
        )
        unresolved_count = sum(
            1 for ind in self.unresolved_indicators
            if ind.lower() in output_lower
        )

        # Check for actionable steps
        has_actionable = False
        if self.check_actionable:
            actionable_patterns = [
                r"\d+\.",  # Numbered steps
                r"step \d",
                r"first,",
                r"then,",
                r"please",
                r"you can",
                r"try",
                r"click",
                r"go to",
            ]
            has_actionable = any(
                re.search(p, output_lower) for p in actionable_patterns
            )

        # Calculate score
        if resolution_count > unresolved_count:
            base_score = 0.8
            if has_actionable:
                base_score += 0.2
        elif unresolved_count > resolution_count:
            base_score = 0.4
            # Partial credit if actionable steps provided
            if has_actionable:
                base_score += 0.2
        else:
            base_score = 0.6
            if has_actionable:
                base_score += 0.1

        score = min(1.0, base_score)

        status = "resolved" if resolution_count > unresolved_count else "pending"

        return EvaluationResult(
            score=score,
            passed=score >= 0.7,
            feedback=f"Ticket status: {status}",
            details={
                "resolution_indicators": resolution_count,
                "unresolved_indicators": unresolved_count,
                "has_actionable_steps": has_actionable,
                "status": status,
            },
        )


class EscalationEvaluator(BaseEvaluator):
    """Evaluates appropriateness of escalation decisions."""

    name = "escalation_appropriateness"

    def __init__(
        self,
        escalation_triggers: list[str] | None = None,
        l2_triggers: list[str] | None = None,
        l3_triggers: list[str] | None = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            escalation_triggers: General escalation trigger phrases.
            l2_triggers: L2 escalation trigger phrases.
            l3_triggers: L3 escalation trigger phrases.
        """
        self.escalation_triggers = escalation_triggers or [
            "escalate",
            "escalating",
            "transfer",
            "specialist",
            "senior",
            "manager",
            "higher level",
        ]
        self.l2_triggers = l2_triggers or [
            "technical specialist",
            "senior technician",
            "level 2",
            "l2 support",
            "advanced support",
        ]
        self.l3_triggers = l3_triggers or [
            "engineering",
            "development team",
            "level 3",
            "l3 support",
            "architect",
            "vendor",
        ]
        # Issues that typically warrant escalation
        self.complex_issues = [
            "security breach",
            "data loss",
            "system down",
            "critical",
            "production",
            "emergency",
            "outage",
            "corruption",
        ]

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: str | None = None,
    ) -> EvaluationResult:
        """Evaluate escalation appropriateness."""
        input_lower = input_text.lower()
        output_lower = output_text.lower()

        # Detect if issue is complex
        is_complex = any(issue in input_lower for issue in self.complex_issues)

        # Detect escalation in response
        is_escalated = any(
            trigger in output_lower for trigger in self.escalation_triggers
        )
        is_l2 = any(trigger in output_lower for trigger in self.l2_triggers)
        is_l3 = any(trigger in output_lower for trigger in self.l3_triggers)

        # Determine escalation level
        escalation_level = None
        if is_l3:
            escalation_level = "L3"
        elif is_l2:
            escalation_level = "L2"
        elif is_escalated:
            escalation_level = "L1+"

        # Evaluate appropriateness
        if is_complex and is_escalated:
            score = 1.0
            feedback = "Appropriate escalation for complex issue"
        elif is_complex and not is_escalated:
            score = 0.5
            feedback = "Complex issue may warrant escalation"
        elif not is_complex and is_escalated:
            score = 0.6
            feedback = "Escalation may be premature for routine issue"
        else:
            score = 0.9
            feedback = "Appropriate handling without escalation"

        return EvaluationResult(
            score=score,
            passed=score >= 0.6,
            feedback=feedback,
            details={
                "is_complex_issue": is_complex,
                "is_escalated": is_escalated,
                "escalation_level": escalation_level,
            },
        )


class ResponseTimeEvaluator(BaseEvaluator):
    """Evaluates response time against SLA targets."""

    name = "response_time"

    def __init__(
        self,
        target_first_response_ms: int = 30000,  # 30 seconds
        target_resolution_ms: int = 900000,  # 15 minutes
        priority_multipliers: dict[str, float] | None = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            target_first_response_ms: Target time for first response.
            target_resolution_ms: Target time for resolution.
            priority_multipliers: Multipliers for different priorities.
        """
        self.target_first_response_ms = target_first_response_ms
        self.target_resolution_ms = target_resolution_ms
        self.priority_multipliers = priority_multipliers or {
            "critical": 0.5,  # Expect 50% faster
            "high": 0.75,
            "medium": 1.0,
            "low": 1.5,  # Allow 50% more time
        }

    def evaluate_timing(
        self,
        response_time_ms: int,
        resolution_time_ms: int | None = None,
        priority: str = "medium",
    ) -> EvaluationResult:
        """Evaluate response and resolution times.

        Args:
            response_time_ms: Time to first response in milliseconds.
            resolution_time_ms: Time to resolution in milliseconds.
            priority: Ticket priority level.

        Returns:
            EvaluationResult with timing analysis.
        """
        multiplier = self.priority_multipliers.get(priority, 1.0)

        adjusted_response_target = self.target_first_response_ms * multiplier
        adjusted_resolution_target = self.target_resolution_ms * multiplier

        scores = []
        details = {}

        # Evaluate first response time
        if response_time_ms <= adjusted_response_target:
            response_score = 1.0
        else:
            # Gradual degradation
            excess_ratio = response_time_ms / adjusted_response_target
            response_score = max(0.0, 1.0 - (excess_ratio - 1) * 0.5)

        scores.append(response_score)
        details["response_time_ms"] = response_time_ms
        details["response_target_ms"] = adjusted_response_target
        details["response_score"] = response_score

        # Evaluate resolution time if provided
        if resolution_time_ms is not None:
            if resolution_time_ms <= adjusted_resolution_target:
                resolution_score = 1.0
            else:
                excess_ratio = resolution_time_ms / adjusted_resolution_target
                resolution_score = max(0.0, 1.0 - (excess_ratio - 1) * 0.3)

            scores.append(resolution_score)
            details["resolution_time_ms"] = resolution_time_ms
            details["resolution_target_ms"] = adjusted_resolution_target
            details["resolution_score"] = resolution_score

        overall_score = sum(scores) / len(scores)

        feedback_parts = []
        if response_score < 0.7:
            feedback_parts.append("First response time exceeded target")
        if "resolution_score" in details and details["resolution_score"] < 0.7:
            feedback_parts.append("Resolution time exceeded target")

        return EvaluationResult(
            score=overall_score,
            passed=overall_score >= 0.7,
            feedback="; ".join(feedback_parts) if feedback_parts else "Response times within SLA",
            details=details,
        )

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: str | None = None,
    ) -> EvaluationResult:
        """Evaluate (use evaluate_timing for actual timing evaluation)."""
        return EvaluationResult(
            score=1.0,
            passed=True,
            feedback="Use evaluate_timing method for response time evaluation",
        )


class UserSatisfactionEvaluator(BaseEvaluator):
    """Evaluates user satisfaction proxy based on conversation patterns."""

    name = "user_satisfaction_proxy"

    def __init__(self) -> None:
        """Initialize the evaluator."""
        self.positive_signals = [
            "thank you",
            "thanks",
            "great",
            "perfect",
            "awesome",
            "excellent",
            "helpful",
            "appreciate",
            "worked",
            "solved",
        ]
        self.negative_signals = [
            "frustrated",
            "unhappy",
            "disappointed",
            "still not working",
            "doesn't work",
            "useless",
            "waste of time",
            "terrible",
            "awful",
            "annoying",
        ]
        self.confusion_signals = [
            "don't understand",
            "confused",
            "what do you mean",
            "unclear",
            "not sure",
            "huh",
            "???",
        ]

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: str | None = None,
    ) -> EvaluationResult:
        """Evaluate user satisfaction proxy."""
        # Combine all text (user messages typically indicate satisfaction)
        combined_lower = (input_text + " " + output_text).lower()

        positive_count = sum(
            1 for signal in self.positive_signals
            if signal in combined_lower
        )
        negative_count = sum(
            1 for signal in self.negative_signals
            if signal in combined_lower
        )
        confusion_count = sum(
            1 for signal in self.confusion_signals
            if signal in combined_lower
        )

        # Calculate satisfaction score
        if negative_count > 0:
            base_score = 0.3
        elif confusion_count > 1:
            base_score = 0.5
        elif positive_count > 0:
            base_score = 0.8 + min(positive_count * 0.05, 0.2)
        else:
            base_score = 0.6  # Neutral

        # Adjust for confusion
        if confusion_count > 0:
            base_score -= confusion_count * 0.1

        score = max(0.0, min(1.0, base_score))

        # Map to satisfaction rating (1-5)
        satisfaction_rating = round(score * 4 + 1)

        return EvaluationResult(
            score=score,
            passed=score >= 0.6,
            feedback=f"Estimated satisfaction: {satisfaction_rating}/5",
            details={
                "positive_signals": positive_count,
                "negative_signals": negative_count,
                "confusion_signals": confusion_count,
                "estimated_rating": satisfaction_rating,
            },
        )


class SLAComplianceEvaluator(BaseEvaluator):
    """Evaluates SLA compliance for IT support interactions."""

    name = "sla_compliance"

    def __init__(
        self,
        sla_config: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            sla_config: SLA configuration by priority level.
        """
        self.sla_config = sla_config or {
            "critical": {
                "first_response_minutes": 5,
                "resolution_minutes": 60,
                "update_frequency_minutes": 15,
            },
            "high": {
                "first_response_minutes": 15,
                "resolution_minutes": 240,
                "update_frequency_minutes": 30,
            },
            "medium": {
                "first_response_minutes": 30,
                "resolution_minutes": 480,
                "update_frequency_minutes": 60,
            },
            "low": {
                "first_response_minutes": 60,
                "resolution_minutes": 1440,
                "update_frequency_minutes": 120,
            },
        }

    def evaluate_sla(
        self,
        metrics: TicketMetrics,
    ) -> EvaluationResult:
        """Evaluate SLA compliance based on ticket metrics.

        Args:
            metrics: Ticket metrics to evaluate.

        Returns:
            EvaluationResult with SLA compliance analysis.
        """
        sla = self.sla_config.get(metrics.priority, self.sla_config["medium"])

        violations = []
        score = 1.0

        # Check first response time
        target_response_ms = sla["first_response_minutes"] * 60 * 1000
        if metrics.response_time_ms > target_response_ms:
            violations.append("First response SLA breached")
            excess = metrics.response_time_ms / target_response_ms
            score -= min(0.3 * (excess - 1), 0.3)

        # Check resolution time
        if metrics.resolved:
            target_resolution_ms = sla["resolution_minutes"] * 60 * 1000
            if metrics.resolution_time_ms > target_resolution_ms:
                violations.append("Resolution SLA breached")
                excess = metrics.resolution_time_ms / target_resolution_ms
                score -= min(0.4 * (excess - 1), 0.4)
        elif metrics.resolution_time_ms > 0:
            # Ticket still open - check if close to breach
            target_resolution_ms = sla["resolution_minutes"] * 60 * 1000
            if metrics.resolution_time_ms > target_resolution_ms * 0.8:
                violations.append("Resolution SLA at risk")
                score -= 0.2

        score = max(0.0, score)

        return EvaluationResult(
            score=score,
            passed=len(violations) == 0,
            feedback="; ".join(violations) if violations else "SLA compliant",
            details={
                "priority": metrics.priority,
                "sla_targets": sla,
                "violations": violations,
            },
        )

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: str | None = None,
    ) -> EvaluationResult:
        """Evaluate (use evaluate_sla for SLA compliance)."""
        return EvaluationResult(
            score=1.0,
            passed=True,
            feedback="Use evaluate_sla method with TicketMetrics",
        )


def evaluate_it_support_interaction(
    input_text: str,
    output_text: str,
    metrics: TicketMetrics | None = None,
) -> dict[str, EvaluationResult]:
    """Evaluate an IT support interaction with business metrics.

    Args:
        input_text: User input/query.
        output_text: Agent response.
        metrics: Optional ticket metrics for SLA evaluation.

    Returns:
        Dictionary of evaluator name to result.
    """
    evaluators = [
        TicketResolutionEvaluator(),
        EscalationEvaluator(),
        UserSatisfactionEvaluator(),
    ]

    results = {}
    for evaluator in evaluators:
        try:
            result = evaluator.evaluate(input_text, output_text)
            results[evaluator.name] = result
        except Exception as e:
            results[evaluator.name] = EvaluationResult(
                score=0.0,
                passed=False,
                feedback=f"Evaluation error: {e}",
            )

    # Add SLA evaluation if metrics provided
    if metrics:
        sla_evaluator = SLAComplianceEvaluator()
        results["sla_compliance"] = sla_evaluator.evaluate_sla(metrics)

        response_evaluator = ResponseTimeEvaluator()
        results["response_time"] = response_evaluator.evaluate_timing(
            response_time_ms=metrics.response_time_ms,
            resolution_time_ms=metrics.resolution_time_ms if metrics.resolved else None,
            priority=metrics.priority,
        )

    return results
