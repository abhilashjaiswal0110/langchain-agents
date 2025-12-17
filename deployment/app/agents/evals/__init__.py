"""Evaluation framework for enterprise IT agents."""

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
    ALL_DATASETS,
    EvalDataset,
    TestCase,
    get_dataset,
    get_test_cases_by_difficulty,
    get_test_cases_by_tag,
)

__all__ = [
    # Evaluators
    "BaseEvaluator",
    "EvaluationResult",
    "ResponseQualityEvaluator",
    "TaskCompletionEvaluator",
    "FactualAccuracyEvaluator",
    "evaluate_agent_response",
    "create_evaluation_summary",
    # Datasets
    "EvalDataset",
    "TestCase",
    "ALL_DATASETS",
    "get_dataset",
    "get_test_cases_by_tag",
    "get_test_cases_by_difficulty",
]
