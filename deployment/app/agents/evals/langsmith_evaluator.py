"""LangSmith integration for agent evaluation.

Provides:
- Offline evaluation with LangSmith datasets
- Online feedback submission
- Run tracking and metrics
- Evaluation experiment management
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal
from uuid import uuid4

from langsmith import Client
from langsmith.schemas import Example, Run

from app.agents.evals.evaluators import BaseEvaluator, EvaluationResult


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

        Args:
            dataset_name: Name for the LangSmith dataset.
            test_cases: List of test case dictionaries.

        Returns:
            Dataset ID.
        """
        examples = []
        for case in test_cases:
            examples.append({
                "inputs": {"input": case.get("input", "")},
                "outputs": {
                    "expected": case.get("expected_output"),
                    "keywords": case.get("expected_keywords", []),
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
