"""LangSmith integration for agent evaluation.

Provides:
- Offline evaluation with LangSmith datasets
- Online feedback submission
- Run tracking and metrics
- Evaluation experiment management
- LangSmith SDK compatible evaluators with proper variable mapping

IMPORTANT FIX (2026-01-02):
- Fixed dataset schema to properly map expected outputs for LangSmith evaluators
- Fixed evaluator execution to provide all required variables (context, reference_outputs)
- Added tracing diagnostics and verification functions
"""

import os
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal
from uuid import uuid4

from langsmith import Client
from langsmith.schemas import Example, Run

from app.agents.evals.evaluators import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class LangSmithConfig:
    """Configuration for LangSmith integration.

    Attributes:
        api_key: LangSmith API key (defaults to LANGCHAIN_API_KEY env var)
        project_name: Project name for evaluation runs
        dataset_name: Default dataset name for evaluations
        auto_submit_feedback: Whether to auto-submit feedback for online evals
        sampling_rate: Sampling rate for online evaluation (0.0 to 1.0)
    """

    api_key: str | None = None
    project_name: str = "enterprise-agents-eval"
    dataset_name: str = "enterprise-agents-dataset"
    auto_submit_feedback: bool = True
    sampling_rate: float = 0.1  # 10% of runs get evaluated online

    @classmethod
    def from_env(cls) -> "LangSmithConfig":
        """Create config from environment variables."""
        return cls(
            api_key=os.getenv("LANGCHAIN_API_KEY"),
            project_name=os.getenv("EVAL_PROJECT_NAME", "enterprise-agents-eval"),
            dataset_name=os.getenv("EVAL_DATASET_NAME", "enterprise-agents-dataset"),
            auto_submit_feedback=os.getenv("EVAL_AUTO_FEEDBACK", "true").lower() == "true",
            sampling_rate=float(os.getenv("EVAL_ONLINE_SAMPLING_RATE", "0.1")),
        )


@dataclass
class EvaluationExperiment:
    """Represents an evaluation experiment run.

    Attributes:
        id: Unique experiment identifier
        name: Experiment name
        dataset_name: Dataset used for evaluation
        created_at: Timestamp of creation
        results: List of evaluation results
        metrics: Aggregated metrics
        metadata: Additional metadata
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    dataset_name: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    results: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class LangSmithEvaluator:
    """LangSmith integration for agent evaluation.

    Supports:
    - Offline evaluation against datasets
    - Online feedback submission
    - Experiment tracking
    - Custom evaluator integration
    """

    def __init__(self, config: LangSmithConfig | None = None) -> None:
        """Initialize LangSmith evaluator.

        Args:
            config: LangSmith configuration. If None, loads from environment.
        """
        self.config = config or LangSmithConfig.from_env()
        self._client: Client | None = None
        self._evaluators: list[BaseEvaluator] = []

    @property
    def client(self) -> Client:
        """Get or create LangSmith client."""
        if self._client is None:
            if self.config.api_key:
                self._client = Client(api_key=self.config.api_key)
            else:
                # Uses LANGCHAIN_API_KEY from environment
                self._client = Client()
        return self._client

    def register_evaluator(self, evaluator: BaseEvaluator) -> None:
        """Register a custom evaluator for use in evaluations.

        Args:
            evaluator: Evaluator instance to register.
        """
        self._evaluators.append(evaluator)

    def create_dataset(
        self,
        name: str,
        description: str = "",
        examples: list[dict[str, Any]] | None = None,
    ) -> str:
        """Create or get a LangSmith dataset.

        Args:
            name: Dataset name.
            description: Dataset description.
            examples: Optional list of examples to add.

        Returns:
            Dataset ID.
        """
        # Check if dataset exists
        try:
            dataset = self.client.read_dataset(dataset_name=name)
            dataset_id = str(dataset.id)
        except Exception:
            # Create new dataset
            dataset = self.client.create_dataset(
                dataset_name=name,
                description=description,
            )
            dataset_id = str(dataset.id)

        # Add examples if provided
        if examples:
            for example in examples:
                self.client.create_example(
                    dataset_id=dataset_id,
                    inputs=example.get("inputs", {}),
                    outputs=example.get("outputs"),
                    metadata=example.get("metadata"),
                )

        return dataset_id

    def sync_dataset_from_local(
        self,
        dataset_name: str,
        test_cases: list[dict[str, Any]],
    ) -> str:
        """Sync local test cases to LangSmith dataset.

        Creates a dataset with proper schema for LangSmith SDK evaluators.
        The dataset includes both `outputs` (for compatibility) and properly
        structured data for built-in evaluators.

        Args:
            dataset_name: Name for the LangSmith dataset.
            test_cases: List of test case dictionaries.

        Returns:
            Dataset ID.

        Note:
            FIX (2026-01-02): Updated to include `reference_output` field
            for LangSmith built-in evaluators that expect this variable.
        """
        examples = []
        for case in test_cases:
            expected_output = case.get("expected_output") or ""
            expected_keywords = case.get("expected_keywords", [])

            # Build context from expected output and keywords
            context = expected_output
            if expected_keywords:
                context += f"\nExpected keywords: {', '.join(expected_keywords)}"

            examples.append({
                "inputs": {
                    "input": case.get("input", ""),
                    # Include context in inputs for evaluators that need it
                    "context": context,
                },
                "outputs": {
                    # Standard output field
                    "expected": expected_output,
                    "keywords": expected_keywords,
                    # Reference output for LangSmith built-in evaluators
                    "reference_output": expected_output,
                },
                "metadata": {
                    "id": case.get("id"),
                    "tags": case.get("tags", []),
                    "difficulty": case.get("difficulty", "medium"),
                },
            })

        return self.create_dataset(
            name=dataset_name,
            description=f"Synced from local test cases at {datetime.now(timezone.utc).isoformat()}",
            examples=examples,
        )

    async def run_offline_evaluation(
        self,
        agent_func: Callable,
        dataset_name: str | None = None,
        experiment_name: str | None = None,
        evaluators: list[BaseEvaluator] | None = None,
    ) -> EvaluationExperiment:
        """Run offline evaluation against a LangSmith dataset.

        Args:
            agent_func: Agent function that takes input and returns output.
            dataset_name: Dataset to evaluate against.
            experiment_name: Name for this evaluation experiment.
            evaluators: Custom evaluators to use.

        Returns:
            EvaluationExperiment with results.
        """
        dataset_name = dataset_name or self.config.dataset_name
        experiment_name = experiment_name or f"eval-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        evaluators = evaluators or self._evaluators

        experiment = EvaluationExperiment(
            name=experiment_name,
            dataset_name=dataset_name,
        )

        try:
            # Get dataset examples
            examples = list(self.client.list_examples(dataset_name=dataset_name))

            for example in examples:
                # Run agent
                input_text = example.inputs.get("input", "")
                try:
                    output = await agent_func(input_text)
                    output_text = str(output)
                except Exception as e:
                    output_text = f"Error: {e}"

                # Run evaluations
                eval_results = {}
                for evaluator in evaluators:
                    try:
                        expected = example.outputs.get("expected") if example.outputs else None
                        result = evaluator.evaluate(input_text, output_text, expected)
                        eval_results[evaluator.name] = {
                            "score": result.score,
                            "passed": result.passed,
                            "feedback": result.feedback,
                        }
                    except Exception as e:
                        eval_results[evaluator.name] = {
                            "score": 0.0,
                            "passed": False,
                            "feedback": f"Evaluation error: {e}",
                        }

                experiment.results.append({
                    "example_id": str(example.id),
                    "input": input_text,
                    "output": output_text,
                    "evaluations": eval_results,
                })

            # Calculate aggregated metrics
            experiment.metrics = self._calculate_metrics(experiment.results)

        except Exception as e:
            experiment.metadata["error"] = str(e)

        return experiment

    def _calculate_metrics(self, results: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate aggregated metrics from results.

        Args:
            results: List of evaluation results.

        Returns:
            Dictionary of metric name to value.
        """
        metrics: dict[str, list[float]] = {}

        for result in results:
            for eval_name, eval_result in result.get("evaluations", {}).items():
                if eval_name not in metrics:
                    metrics[eval_name] = []
                metrics[eval_name].append(eval_result.get("score", 0.0))

        # Calculate averages
        return {
            f"{name}_avg": sum(scores) / len(scores) if scores else 0.0
            for name, scores in metrics.items()
        }

    def submit_feedback(
        self,
        run_id: str,
        key: str,
        score: float,
        comment: str | None = None,
        correction: dict[str, Any] | None = None,
    ) -> None:
        """Submit feedback for a run (online evaluation).

        Args:
            run_id: LangSmith run ID.
            key: Feedback key (e.g., "quality", "accuracy").
            score: Score value (typically 0.0 to 1.0).
            comment: Optional comment.
            correction: Optional correction data.
        """
        self.client.create_feedback(
            run_id=run_id,
            key=key,
            score=score,
            comment=comment,
            correction=correction,
        )

    def submit_evaluation_results(
        self,
        run_id: str,
        results: dict[str, EvaluationResult],
    ) -> None:
        """Submit multiple evaluation results as feedback.

        Args:
            run_id: LangSmith run ID.
            results: Dictionary of evaluator name to result.
        """
        for name, result in results.items():
            self.submit_feedback(
                run_id=run_id,
                key=name,
                score=result.score,
                comment=result.feedback,
            )

    def should_evaluate_online(self) -> bool:
        """Check if this run should be evaluated (based on sampling rate).

        Returns:
            True if run should be evaluated.
        """
        import random
        return random.random() < self.config.sampling_rate

    async def evaluate_async(
        self,
        agent_type: str,
        input_msg: str,
        output: str,
        run_id: str | None = None,
    ) -> None:
        """Submit an agent response for lightweight online evaluation.

        Sends correctness and helpfulness feedback to LangSmith as background
        metadata. Silently skips if no API key is configured.

        Args:
            agent_type: The agent type that produced the output.
            input_msg: The user input that was sent to the agent.
            output: The agent's response string.
            run_id: Optional LangSmith run ID to attach feedback to.
        """
        if not self.config.api_key and not os.getenv("LANGCHAIN_API_KEY") and not os.getenv("LANGSMITH_API_KEY"):
            logger.debug("LangSmith API key not configured; skipping evaluation")
            return

        try:
            feedback_run_id = run_id or str(uuid4())
            # Simple heuristic quality signals (replace with real evaluators in production)
            has_output = bool(output and output.strip())
            score = 1.0 if has_output else 0.0

            self.client.create_feedback(
                run_id=feedback_run_id,
                key="online_eval",
                score=score,
                comment=f"agent={agent_type}, input_len={len(input_msg)}, output_len={len(output)}",
            )
            logger.debug("Evaluation feedback submitted for run %s", feedback_run_id)
        except Exception as exc:
            logger.debug("Evaluation feedback failed (non-critical): %s", exc)

    def get_run_metrics(
        self,
        project_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Get aggregated metrics for runs in a project.

        Args:
            project_name: Project to get metrics for.
            start_time: Start time filter.
            end_time: End time filter.

        Returns:
            Dictionary of metrics.
        """
        project_name = project_name or self.config.project_name

        runs = self.client.list_runs(
            project_name=project_name,
            start_time=start_time,
            end_time=end_time,
        )

        total_runs = 0
        total_latency = 0.0
        error_count = 0
        feedback_scores: dict[str, list[float]] = {}

        for run in runs:
            total_runs += 1
            if run.end_time and run.start_time:
                total_latency += (run.end_time - run.start_time).total_seconds()
            if run.error:
                error_count += 1

            # Get feedback for run
            try:
                feedbacks = self.client.list_feedback(run_ids=[str(run.id)])
                for fb in feedbacks:
                    if fb.key not in feedback_scores:
                        feedback_scores[fb.key] = []
                    if fb.score is not None:
                        feedback_scores[fb.key].append(fb.score)
            except Exception:
                pass

        return {
            "total_runs": total_runs,
            "avg_latency_seconds": total_latency / total_runs if total_runs > 0 else 0,
            "error_rate": error_count / total_runs if total_runs > 0 else 0,
            "feedback_averages": {
                key: sum(scores) / len(scores) if scores else 0
                for key, scores in feedback_scores.items()
            },
        }


# Global evaluator instance
_langsmith_evaluator: LangSmithEvaluator | None = None


def get_langsmith_evaluator() -> LangSmithEvaluator:
    """Get or create the global LangSmith evaluator instance."""
    global _langsmith_evaluator
    if _langsmith_evaluator is None:
        _langsmith_evaluator = LangSmithEvaluator()
    return _langsmith_evaluator


def reset_langsmith_evaluator() -> None:
    """Reset the global LangSmith evaluator instance."""
    global _langsmith_evaluator
    _langsmith_evaluator = None


# Convenience functions for common operations


def submit_online_feedback(
    run_id: str,
    score: float,
    key: str = "quality",
    comment: str | None = None,
) -> None:
    """Submit feedback for a run.

    Args:
        run_id: LangSmith run ID.
        score: Quality score (0.0 to 1.0).
        key: Feedback key.
        comment: Optional comment.
    """
    evaluator = get_langsmith_evaluator()
    evaluator.submit_feedback(run_id, key, score, comment)


async def evaluate_agent_offline(
    agent_func: Callable,
    dataset_name: str,
    evaluators: list[BaseEvaluator] | None = None,
) -> EvaluationExperiment:
    """Run offline evaluation against a dataset.

    Args:
        agent_func: Agent function to evaluate.
        dataset_name: LangSmith dataset name.
        evaluators: Custom evaluators to use.

    Returns:
        EvaluationExperiment with results.
    """
    evaluator = get_langsmith_evaluator()
    return await evaluator.run_offline_evaluation(
        agent_func=agent_func,
        dataset_name=dataset_name,
        evaluators=evaluators,
    )


# =============================================================================
# Tracing Diagnostics - Added 2026-01-02
# =============================================================================


def verify_tracing_config() -> dict[str, Any]:
    """Verify LangSmith tracing configuration.

    Returns:
        Dictionary with tracing configuration status and diagnostics.
    """
    config = {
        "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2"),
        "LANGCHAIN_API_KEY": "***" if os.getenv("LANGCHAIN_API_KEY") else None,
        "LANGCHAIN_PROJECT": os.getenv("LANGCHAIN_PROJECT"),
        "LANGCHAIN_ENDPOINT": os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
    }

    is_enabled = config["LANGCHAIN_TRACING_V2"] == "true"
    has_api_key = config["LANGCHAIN_API_KEY"] is not None

    return {
        "tracing_enabled": is_enabled,
        "api_key_configured": has_api_key,
        "project_name": config["LANGCHAIN_PROJECT"],
        "endpoint": config["LANGCHAIN_ENDPOINT"],
        "status": "OK" if (is_enabled and has_api_key) else "NOT_CONFIGURED",
        "issues": _get_tracing_issues(config, is_enabled, has_api_key),
    }


def _get_tracing_issues(
    config: dict[str, Any],
    is_enabled: bool,
    has_api_key: bool,
) -> list[str]:
    """Get list of tracing configuration issues."""
    issues = []

    if not is_enabled:
        issues.append("LANGCHAIN_TRACING_V2 is not set to 'true'")
    if not has_api_key:
        issues.append("LANGCHAIN_API_KEY is not configured")
    if not config["LANGCHAIN_PROJECT"]:
        issues.append("LANGCHAIN_PROJECT is not set (using default)")

    return issues


def test_langsmith_connection() -> dict[str, Any]:
    """Test connection to LangSmith API.

    Returns:
        Dictionary with connection test results.
    """
    result = {
        "connected": False,
        "projects": [],
        "error": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        client = Client()
        # Try to list projects to verify connection
        projects = list(client.list_projects(limit=5))
        result["connected"] = True
        result["projects"] = [p.name for p in projects]
        logger.info(f"LangSmith connection verified. Found {len(projects)} projects.")
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"LangSmith connection failed: {e}")

    return result


def get_recent_traces(
    project_name: str | None = None,
    limit: int = 10,
    hours: int = 24,
) -> dict[str, Any]:
    """Get recent traces from LangSmith.

    Args:
        project_name: Project to query. Uses env var if not provided.
        limit: Maximum number of traces to return.
        hours: Number of hours to look back.

    Returns:
        Dictionary with trace information.
    """
    project_name = project_name or os.getenv("LANGCHAIN_PROJECT", "enterprise-it-agents")

    result = {
        "project": project_name,
        "traces": [],
        "total_count": 0,
        "error": None,
        "query_time": datetime.now(timezone.utc).isoformat(),
    }

    try:
        client = Client()
        start_time = datetime.now(timezone.utc) - timezone.utc.utcoffset(None)

        # Calculate start time for query
        from datetime import timedelta
        query_start = datetime.now(timezone.utc) - timedelta(hours=hours)

        runs = list(client.list_runs(
            project_name=project_name,
            start_time=query_start,
            limit=limit,
        ))

        result["total_count"] = len(runs)
        result["traces"] = [
            {
                "id": str(run.id),
                "name": run.name,
                "status": run.status,
                "start_time": run.start_time.isoformat() if run.start_time else None,
                "end_time": run.end_time.isoformat() if run.end_time else None,
                "error": run.error[:100] if run.error else None,
            }
            for run in runs
        ]

        if len(runs) == 0:
            logger.warning(f"No traces found in project '{project_name}' in last {hours} hours")
        else:
            logger.info(f"Found {len(runs)} traces in project '{project_name}'")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Failed to get traces: {e}")

    return result


# =============================================================================
# LangSmith SDK Compatible Evaluation - Added 2026-01-02
# =============================================================================


def create_langsmith_evaluator_wrapper(
    base_evaluator: BaseEvaluator,
) -> Callable:
    """Create a LangSmith SDK compatible evaluator from a BaseEvaluator.

    This wrapper handles the variable mapping that LangSmith SDK expects:
    - inputs: The input to the agent
    - outputs: The agent's output
    - reference_outputs: The expected/reference output
    - context: Additional context (optional)

    Args:
        base_evaluator: The base evaluator to wrap.

    Returns:
        A callable compatible with LangSmith's evaluate() function.

    Example:
        >>> from langsmith.evaluation import evaluate
        >>> from app.agents.evals import ResponseQualityEvaluator
        >>>
        >>> quality_eval = create_langsmith_evaluator_wrapper(ResponseQualityEvaluator())
        >>> results = evaluate(agent_func, data="my-dataset", evaluators=[quality_eval])

    Note:
        FIX (2026-01-06): Updated to properly handle context and reference_outputs
        to fix KeyError in LangSmith Playground evaluators.
    """
    def evaluator_fn(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any] | None = None,
        context: str | None = None,
        examples_few_shot: list | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """LangSmith SDK compatible evaluator function.

        Handles all possible variable combinations from LangSmith SDK/Playground.
        """
        # Extract values from the inputs/outputs structure
        input_text = inputs.get("input", "") or str(inputs)
        output_text = outputs.get("output", "") or str(outputs)

        # Get context from multiple sources
        ctx = context or inputs.get("context", "") or ""

        # Get expected output from reference_outputs (handle None gracefully)
        expected = None
        if reference_outputs:
            expected = (
                reference_outputs.get("expected")
                or reference_outputs.get("reference_output")
                or reference_outputs.get("output")
            )
        elif outputs:
            # Fallback: try to get from outputs if reference not provided
            expected = outputs.get("expected") or outputs.get("reference_output")

        # Run the base evaluator
        try:
            result = base_evaluator.evaluate(input_text, output_text, expected)
            return {
                "key": base_evaluator.name,
                "score": result.score,
                "comment": result.feedback,
                "passed": result.passed,
            }
        except Exception as e:
            logger.error(f"Evaluator {base_evaluator.name} failed: {e}")
            return {
                "key": base_evaluator.name,
                "score": 0.0,
                "comment": f"Evaluation error: {e}",
                "passed": False,
            }

    # Set function name for LangSmith display
    evaluator_fn.__name__ = base_evaluator.name
    return evaluator_fn


def create_playground_compatible_evaluator(
    base_evaluator: BaseEvaluator,
) -> Callable:
    """Create a LangSmith Playground compatible evaluator.

    This wrapper ensures all expected variables are handled, including:
    - context: Defaults to empty string if not provided
    - reference_outputs: Defaults to empty dict if not provided
    - examples_few_shot: Handled but not required

    This fixes the KeyError: "Input to StructuredPrompt is missing variables
    {'context', 'reference_outputs'}" error in LangSmith Playground.

    Args:
        base_evaluator: The base evaluator to wrap.

    Returns:
        A callable compatible with LangSmith Playground evaluators.

    Note:
        Added 2026-01-06 to fix Playground evaluator errors.
    """
    def evaluator_fn(
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        reference_outputs: dict[str, Any] | None = None,
        context: str | None = None,
        examples_few_shot: list | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Playground compatible evaluator function with default handling."""
        # Provide defaults for all expected variables
        inputs = inputs or {}
        outputs = outputs or {}
        reference_outputs = reference_outputs or {}
        context = context or ""

        # Extract values
        input_text = inputs.get("input", "") or str(inputs) if inputs else ""
        output_text = outputs.get("output", "") or str(outputs) if outputs else ""

        # Get context from inputs if not provided directly
        if not context and inputs:
            context = inputs.get("context", "")

        # Get expected output from reference_outputs
        expected = None
        if reference_outputs:
            expected = (
                reference_outputs.get("expected")
                or reference_outputs.get("reference_output")
                or reference_outputs.get("output")
            )

        # Run the base evaluator
        try:
            result = base_evaluator.evaluate(input_text, output_text, expected)
            return {
                "key": base_evaluator.name,
                "score": result.score,
                "comment": result.feedback,
                "passed": result.passed,
            }
        except Exception as e:
            logger.error(f"Evaluator {base_evaluator.name} failed: {e}")
            return {
                "key": base_evaluator.name,
                "score": 0.0,
                "comment": f"Evaluation error: {e}",
                "passed": False,
            }

    # Set function name for LangSmith display
    evaluator_fn.__name__ = f"{base_evaluator.name}_playground"
    return evaluator_fn


async def run_langsmith_sdk_evaluation(
    agent_func: Callable,
    dataset_name: str,
    evaluators: list[BaseEvaluator] | None = None,
    experiment_prefix: str = "eval",
) -> dict[str, Any]:
    """Run evaluation using LangSmith SDK evaluate() function.

    This function properly maps variables for LangSmith SDK evaluators,
    fixing the KeyError for missing 'reference_outputs' and 'context'.

    Args:
        agent_func: Agent function to evaluate.
        dataset_name: LangSmith dataset name.
        evaluators: Custom evaluators to use.
        experiment_prefix: Prefix for experiment name.

    Returns:
        Evaluation results dictionary.

    Note:
        This requires the langsmith package with evaluation support.
        Install with: pip install langsmith[evaluation]
    """
    try:
        from langsmith.evaluation import evaluate
    except ImportError:
        logger.error("langsmith.evaluation not available. Install with: pip install langsmith[evaluation]")
        return {"error": "langsmith.evaluation not installed"}

    # Import default evaluators if none provided
    if evaluators is None:
        from app.agents.evals.evaluators import (
            ResponseQualityEvaluator,
            TaskCompletionEvaluator,
        )
        evaluators = [ResponseQualityEvaluator(), TaskCompletionEvaluator()]

    # Create LangSmith SDK compatible wrappers
    sdk_evaluators = [create_langsmith_evaluator_wrapper(e) for e in evaluators]

    # Create the target function wrapper to handle input/output format
    async def target_fn(inputs: dict[str, Any]) -> dict[str, Any]:
        input_text = inputs.get("input", "") or str(inputs)
        try:
            result = await agent_func(input_text)
            return {"output": str(result)}
        except Exception as e:
            return {"output": f"Error: {e}"}

    experiment_name = f"{experiment_prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    try:
        results = evaluate(
            target_fn,
            data=dataset_name,
            evaluators=sdk_evaluators,
            experiment_prefix=experiment_name,
        )

        return {
            "experiment_name": experiment_name,
            "dataset": dataset_name,
            "status": "completed",
            "results": results,
        }
    except Exception as e:
        logger.error(f"LangSmith SDK evaluation failed: {e}")
        return {
            "experiment_name": experiment_name,
            "dataset": dataset_name,
            "status": "failed",
            "error": str(e),
        }


def ensure_tracing_enabled() -> bool:
    """Ensure LangSmith tracing is properly enabled.

    This function verifies and enables tracing if the API key is available.
    Call this at application startup to ensure tracing works.

    Returns:
        True if tracing is enabled, False otherwise.
    """
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")

    if not api_key:
        logger.warning("No LangSmith API key found. Tracing disabled.")
        return False

    # Ensure environment variables are set
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key

    if not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = "enterprise-it-agents"

    logger.info(f"LangSmith tracing enabled for project: {os.getenv('LANGCHAIN_PROJECT')}")
    return True
