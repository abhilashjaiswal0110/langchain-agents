"""Multi-turn conversation evaluation.

Provides:
- Conversation trajectory analysis
- Context coherence evaluation
- Intent completion tracking
- Tool call sequence analysis
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.agents.evals.evaluators import BaseEvaluator, EvaluationResult


@dataclass
class ConversationTurn:
    """A single turn in a conversation.

    Attributes:
        turn_number: Sequential turn number (1-based)
        role: Role of the speaker (user, assistant, tool)
        content: Message content
        tool_calls: Tool calls made (if assistant)
        tool_results: Tool results (if tool message)
        metadata: Additional metadata
    """

    turn_number: int
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_message(cls, msg: BaseMessage, turn_number: int) -> "ConversationTurn":
        """Create from a LangChain message.

        Args:
            msg: LangChain message.
            turn_number: Turn number in conversation.

        Returns:
            ConversationTurn instance.
        """
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        else:
            role = "assistant"

        tool_calls = []
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls = [
                {"name": tc.get("name", ""), "args": tc.get("args", {})}
                for tc in msg.tool_calls
            ]

        return cls(
            turn_number=turn_number,
            role=role,
            content=str(msg.content) if msg.content else "",
            tool_calls=tool_calls,
            metadata=getattr(msg, "additional_kwargs", {}),
        )


@dataclass
class MultiTurnTestCase:
    """Test case for multi-turn conversation evaluation.

    Attributes:
        id: Unique test case identifier
        conversation: List of conversation turns
        expected_intents: List of intents that should be completed
        expected_tool_sequence: Expected sequence of tool calls
        context_requirements: Context that should be maintained
        success_criteria: Criteria for successful completion
        tags: Tags for categorization
        difficulty: Difficulty level
    """

    id: str
    conversation: list[ConversationTurn] = field(default_factory=list)
    expected_intents: list[str] = field(default_factory=list)
    expected_tool_sequence: list[str] = field(default_factory=list)
    context_requirements: list[str] = field(default_factory=list)
    success_criteria: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    difficulty: str = "medium"


@dataclass
class MultiTurnEvaluationResult:
    """Result from multi-turn evaluation.

    Attributes:
        overall_score: Overall evaluation score (0.0 to 1.0)
        passed: Whether evaluation passed
        intent_completion: Intent completion details
        context_coherence: Context coherence score
        tool_accuracy: Tool usage accuracy
        conversation_flow: Conversation flow quality
        turn_by_turn: Per-turn analysis
        feedback: Human-readable feedback
    """

    overall_score: float
    passed: bool
    intent_completion: dict[str, Any] = field(default_factory=dict)
    context_coherence: dict[str, Any] = field(default_factory=dict)
    tool_accuracy: dict[str, Any] = field(default_factory=dict)
    conversation_flow: dict[str, Any] = field(default_factory=dict)
    turn_by_turn: list[dict[str, Any]] = field(default_factory=list)
    feedback: str = ""


class IntentCompletionEvaluator(BaseEvaluator):
    """Evaluates whether user intents are completed across turns."""

    name = "intent_completion"

    def __init__(self, intents: list[str] | None = None) -> None:
        """Initialize with expected intents.

        Args:
            intents: List of intents to check for completion.
        """
        self.intents = intents or []

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: str | None = None,
    ) -> EvaluationResult:
        """Evaluate intent completion.

        Args:
            input_text: Combined user messages.
            output_text: Combined assistant responses.
            expected: Not used.

        Returns:
            EvaluationResult with intent completion score.
        """
        if not self.intents:
            return EvaluationResult(
                score=1.0,
                passed=True,
                feedback="No intents specified to evaluate",
            )

        output_lower = output_text.lower()
        completed = []
        incomplete = []

        for intent in self.intents:
            # Simple keyword-based intent detection
            intent_keywords = intent.lower().split()
            if all(kw in output_lower for kw in intent_keywords):
                completed.append(intent)
            else:
                incomplete.append(intent)

        score = len(completed) / len(self.intents) if self.intents else 1.0

        return EvaluationResult(
            score=score,
            passed=score >= 0.8,
            feedback=f"Completed {len(completed)}/{len(self.intents)} intents",
            details={
                "completed": completed,
                "incomplete": incomplete,
            },
        )


class ContextCoherenceEvaluator(BaseEvaluator):
    """Evaluates context coherence across conversation turns."""

    name = "context_coherence"

    def __init__(
        self,
        context_requirements: list[str] | None = None,
        check_references: bool = True,
    ) -> None:
        """Initialize the evaluator.

        Args:
            context_requirements: Context that should be maintained.
            check_references: Whether to check for back-references.
        """
        self.context_requirements = context_requirements or []
        self.check_references = check_references

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: str | None = None,
    ) -> EvaluationResult:
        """Evaluate context coherence."""
        issues = []
        score = 1.0

        # Check for context requirements
        output_lower = output_text.lower()
        missing_context = []
        for req in self.context_requirements:
            if req.lower() not in output_lower:
                missing_context.append(req)

        if missing_context:
            score -= 0.3 * min(len(missing_context) / len(self.context_requirements), 1)
            issues.append(f"Missing context: {', '.join(missing_context[:3])}")

        # Check for pronouns without clear referents (simple heuristic)
        if self.check_references:
            vague_pronouns = ["it", "this", "that", "they"]
            for pronoun in vague_pronouns:
                # Check if pronoun appears at start of sentence without context
                if f". {pronoun} " in output_lower or output_lower.startswith(f"{pronoun} "):
                    # Simple check - in real implementation, use NLP
                    pass

        # Check for contradictions (simple keyword check)
        contradiction_pairs = [
            ("yes", "no"),
            ("can", "cannot"),
            ("will", "won't"),
            ("is", "isn't"),
        ]
        for pos, neg in contradiction_pairs:
            if f" {pos} " in output_lower and f" {neg} " in output_lower:
                # Potential contradiction - needs deeper analysis
                pass

        score = max(0.0, min(1.0, score))

        return EvaluationResult(
            score=score,
            passed=score >= 0.7,
            feedback="; ".join(issues) if issues else "Context appears coherent",
            details={
                "missing_context": missing_context,
            },
        )


class ToolSequenceEvaluator(BaseEvaluator):
    """Evaluates tool call sequences in agent responses."""

    name = "tool_sequence"

    def __init__(
        self,
        expected_sequence: list[str] | None = None,
        allow_extra_tools: bool = True,
    ) -> None:
        """Initialize the evaluator.

        Args:
            expected_sequence: Expected tool call sequence.
            allow_extra_tools: Whether to allow additional tool calls.
        """
        self.expected_sequence = expected_sequence or []
        self.allow_extra_tools = allow_extra_tools

    def evaluate_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> EvaluationResult:
        """Evaluate a sequence of tool calls.

        Args:
            tool_calls: List of tool call dictionaries.

        Returns:
            EvaluationResult with tool sequence analysis.
        """
        if not self.expected_sequence:
            return EvaluationResult(
                score=1.0,
                passed=True,
                feedback="No expected tool sequence specified",
            )

        actual_sequence = [tc.get("name", "") for tc in tool_calls]

        # Check if expected tools are called in order
        expected_idx = 0
        matched = []
        extra = []

        for tool in actual_sequence:
            if expected_idx < len(self.expected_sequence):
                if tool == self.expected_sequence[expected_idx]:
                    matched.append(tool)
                    expected_idx += 1
                elif not self.allow_extra_tools:
                    extra.append(tool)
            else:
                extra.append(tool)

        # Calculate score
        if self.expected_sequence:
            sequence_score = len(matched) / len(self.expected_sequence)
        else:
            sequence_score = 1.0

        if not self.allow_extra_tools and extra:
            sequence_score *= 0.8

        issues = []
        if expected_idx < len(self.expected_sequence):
            missing = self.expected_sequence[expected_idx:]
            issues.append(f"Missing tools: {', '.join(missing)}")
        if extra and not self.allow_extra_tools:
            issues.append(f"Unexpected tools: {', '.join(extra)}")

        return EvaluationResult(
            score=sequence_score,
            passed=sequence_score >= 0.8,
            feedback="; ".join(issues) if issues else "Tool sequence is correct",
            details={
                "matched": matched,
                "extra": extra,
                "expected": self.expected_sequence,
                "actual": actual_sequence,
            },
        )

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: str | None = None,
    ) -> EvaluationResult:
        """Evaluate (not applicable for tool sequence - use evaluate_tool_calls)."""
        return EvaluationResult(
            score=1.0,
            passed=True,
            feedback="Use evaluate_tool_calls for tool sequence evaluation",
        )


class ConversationFlowEvaluator(BaseEvaluator):
    """Evaluates the natural flow of conversation."""

    name = "conversation_flow"

    def __init__(
        self,
        max_response_length: int = 2000,
        min_response_length: int = 20,
        check_greeting: bool = False,
        check_closing: bool = False,
    ) -> None:
        """Initialize the evaluator.

        Args:
            max_response_length: Maximum acceptable response length.
            min_response_length: Minimum acceptable response length.
            check_greeting: Whether to check for appropriate greeting.
            check_closing: Whether to check for appropriate closing.
        """
        self.max_response_length = max_response_length
        self.min_response_length = min_response_length
        self.check_greeting = check_greeting
        self.check_closing = check_closing

    def evaluate(
        self,
        input_text: str,
        output_text: str,
        expected: str | None = None,
    ) -> EvaluationResult:
        """Evaluate conversation flow quality."""
        issues = []
        score = 1.0

        # Length check
        if len(output_text) < self.min_response_length:
            issues.append("Response too brief")
            score -= 0.2
        elif len(output_text) > self.max_response_length:
            issues.append("Response too verbose")
            score -= 0.1

        # Check for abrupt endings
        if output_text and not output_text.rstrip().endswith((".", "!", "?", ":", "```")):
            issues.append("Response may be incomplete")
            score -= 0.1

        # Check for repetition (simple check)
        sentences = output_text.split(".")
        unique_sentences = set(s.strip().lower() for s in sentences if s.strip())
        if len(sentences) > 3 and len(unique_sentences) < len(sentences) * 0.7:
            issues.append("Response contains repetition")
            score -= 0.2

        # Check for greeting if required
        if self.check_greeting:
            greetings = ["hello", "hi", "hey", "good", "welcome"]
            output_lower = output_text.lower()
            if not any(g in output_lower[:100] for g in greetings):
                issues.append("Missing greeting")
                score -= 0.1

        score = max(0.0, min(1.0, score))

        return EvaluationResult(
            score=score,
            passed=score >= 0.7,
            feedback="; ".join(issues) if issues else "Conversation flow is natural",
            details={
                "length": len(output_text),
                "sentence_count": len(sentences),
            },
        )


class MultiTurnEvaluator:
    """Comprehensive multi-turn conversation evaluator.

    Combines multiple evaluation dimensions:
    - Intent completion
    - Context coherence
    - Tool sequence accuracy
    - Conversation flow
    """

    def __init__(
        self,
        intent_evaluator: IntentCompletionEvaluator | None = None,
        context_evaluator: ContextCoherenceEvaluator | None = None,
        tool_evaluator: ToolSequenceEvaluator | None = None,
        flow_evaluator: ConversationFlowEvaluator | None = None,
    ) -> None:
        """Initialize the multi-turn evaluator.

        Args:
            intent_evaluator: Intent completion evaluator.
            context_evaluator: Context coherence evaluator.
            tool_evaluator: Tool sequence evaluator.
            flow_evaluator: Conversation flow evaluator.
        """
        self.intent_evaluator = intent_evaluator or IntentCompletionEvaluator()
        self.context_evaluator = context_evaluator or ContextCoherenceEvaluator()
        self.tool_evaluator = tool_evaluator or ToolSequenceEvaluator()
        self.flow_evaluator = flow_evaluator or ConversationFlowEvaluator()

    def evaluate_conversation(
        self,
        messages: list[BaseMessage],
        test_case: MultiTurnTestCase | None = None,
    ) -> MultiTurnEvaluationResult:
        """Evaluate a complete conversation.

        Args:
            messages: List of conversation messages.
            test_case: Optional test case with expected values.

        Returns:
            MultiTurnEvaluationResult with detailed analysis.
        """
        # Convert messages to turns
        turns = [
            ConversationTurn.from_message(msg, i + 1)
            for i, msg in enumerate(messages)
        ]

        # Separate user and assistant content
        user_content = "\n".join(
            t.content for t in turns if t.role == "user"
        )
        assistant_content = "\n".join(
            t.content for t in turns if t.role == "assistant"
        )

        # Collect all tool calls
        all_tool_calls = []
        for turn in turns:
            all_tool_calls.extend(turn.tool_calls)

        # Update evaluators with test case expectations
        if test_case:
            self.intent_evaluator.intents = test_case.expected_intents
            self.context_evaluator.context_requirements = test_case.context_requirements
            self.tool_evaluator.expected_sequence = test_case.expected_tool_sequence

        # Run evaluations
        intent_result = self.intent_evaluator.evaluate(
            user_content, assistant_content
        )
        context_result = self.context_evaluator.evaluate(
            user_content, assistant_content
        )
        tool_result = self.tool_evaluator.evaluate_tool_calls(all_tool_calls)
        flow_result = self.flow_evaluator.evaluate(
            user_content, assistant_content
        )

        # Per-turn analysis
        turn_analysis = []
        for turn in turns:
            turn_analysis.append({
                "turn": turn.turn_number,
                "role": turn.role,
                "content_length": len(turn.content),
                "tool_calls": len(turn.tool_calls),
            })

        # Calculate overall score (weighted average)
        weights = {
            "intent": 0.3,
            "context": 0.25,
            "tool": 0.25,
            "flow": 0.2,
        }
        overall_score = (
            intent_result.score * weights["intent"]
            + context_result.score * weights["context"]
            + tool_result.score * weights["tool"]
            + flow_result.score * weights["flow"]
        )

        # Generate feedback
        feedback_parts = []
        if not intent_result.passed:
            feedback_parts.append(f"Intent: {intent_result.feedback}")
        if not context_result.passed:
            feedback_parts.append(f"Context: {context_result.feedback}")
        if not tool_result.passed:
            feedback_parts.append(f"Tools: {tool_result.feedback}")
        if not flow_result.passed:
            feedback_parts.append(f"Flow: {flow_result.feedback}")

        return MultiTurnEvaluationResult(
            overall_score=overall_score,
            passed=overall_score >= 0.7,
            intent_completion={
                "score": intent_result.score,
                "passed": intent_result.passed,
                "details": intent_result.details,
            },
            context_coherence={
                "score": context_result.score,
                "passed": context_result.passed,
                "details": context_result.details,
            },
            tool_accuracy={
                "score": tool_result.score,
                "passed": tool_result.passed,
                "details": tool_result.details,
            },
            conversation_flow={
                "score": flow_result.score,
                "passed": flow_result.passed,
                "details": flow_result.details,
            },
            turn_by_turn=turn_analysis,
            feedback="; ".join(feedback_parts) if feedback_parts else "Conversation evaluation passed",
        )


def evaluate_multi_turn_conversation(
    messages: list[BaseMessage],
    expected_intents: list[str] | None = None,
    expected_tool_sequence: list[str] | None = None,
    context_requirements: list[str] | None = None,
) -> MultiTurnEvaluationResult:
    """Convenience function to evaluate a multi-turn conversation.

    Args:
        messages: List of conversation messages.
        expected_intents: Intents that should be completed.
        expected_tool_sequence: Expected tool call sequence.
        context_requirements: Context to maintain.

    Returns:
        MultiTurnEvaluationResult with analysis.
    """
    test_case = MultiTurnTestCase(
        id="inline_test",
        expected_intents=expected_intents or [],
        expected_tool_sequence=expected_tool_sequence or [],
        context_requirements=context_requirements or [],
    )

    evaluator = MultiTurnEvaluator()
    return evaluator.evaluate_conversation(messages, test_case)
