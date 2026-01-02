"""Evaluation framework for enterprise IT agents.

This module provides:
- Base evaluators for response quality and task completion
- LangSmith integration for offline/online evaluation
- Multi-turn conversation evaluation
- Business metrics for IT support
- Regression test runner for CI/CD
"""

# Base evaluators
from app.agents.evals.evaluators import (
    BaseEvaluator,
    EvaluationResult,
    FactualAccuracyEvaluator,
    ResponseQualityEvaluator,
    TaskCompletionEvaluator,
    create_evaluation_summary,
    evaluate_agent_response,
)

# Datasets
from app.agents.evals.datasets import (
    ALL_DATASETS,
    EvalDataset,
    TestCase,
    get_dataset,
    get_test_cases_by_difficulty,
    get_test_cases_by_tag,
)

# LangSmith integration
from app.agents.evals.langsmith_evaluator import (
    EvaluationExperiment,
    LangSmithConfig,
    LangSmithEvaluator,
    evaluate_agent_offline,
    get_langsmith_evaluator,
    reset_langsmith_evaluator,
    submit_online_feedback,
    # Tracing diagnostics (added 2026-01-02)
    verify_tracing_config,
    test_langsmith_connection,
    get_recent_traces,
    ensure_tracing_enabled,
    # LangSmith SDK compatible evaluation (added 2026-01-02)
    create_langsmith_evaluator_wrapper,
    run_langsmith_sdk_evaluation,
)

# Multi-turn evaluation
from app.agents.evals.multi_turn_evaluator import (
    ContextCoherenceEvaluator,
    ConversationFlowEvaluator,
    ConversationTurn,
    IntentCompletionEvaluator,
    MultiTurnEvaluationResult,
    MultiTurnEvaluator,
    MultiTurnTestCase,
    ToolSequenceEvaluator,
    evaluate_multi_turn_conversation,
)

# Business metrics
from app.agents.evals.business_metrics import (
    EscalationEvaluator,
    ResponseTimeEvaluator,
    SLAComplianceEvaluator,
    TicketMetrics,
    TicketResolutionEvaluator,
    UserSatisfactionEvaluator,
    evaluate_it_support_interaction,
)

# Regression runner
from app.agents.evals.regression_runner import (
    RegressionConfig,
    RegressionReport,
    RegressionRunner,
    TestResult,
    check_regression_passed,
    run_regression_async,
    run_regression_sync,
)

__all__ = [
    # Base Evaluators
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
    # LangSmith
    "LangSmithConfig",
    "LangSmithEvaluator",
    "EvaluationExperiment",
    "get_langsmith_evaluator",
    "reset_langsmith_evaluator",
    "submit_online_feedback",
    "evaluate_agent_offline",
    # Tracing diagnostics
    "verify_tracing_config",
    "test_langsmith_connection",
    "get_recent_traces",
    "ensure_tracing_enabled",
    # LangSmith SDK compatible evaluation
    "create_langsmith_evaluator_wrapper",
    "run_langsmith_sdk_evaluation",
    # Multi-Turn
    "ConversationTurn",
    "MultiTurnTestCase",
    "MultiTurnEvaluationResult",
    "IntentCompletionEvaluator",
    "ContextCoherenceEvaluator",
    "ToolSequenceEvaluator",
    "ConversationFlowEvaluator",
    "MultiTurnEvaluator",
    "evaluate_multi_turn_conversation",
    # Business Metrics
    "TicketMetrics",
    "TicketResolutionEvaluator",
    "EscalationEvaluator",
    "ResponseTimeEvaluator",
    "UserSatisfactionEvaluator",
    "SLAComplianceEvaluator",
    "evaluate_it_support_interaction",
    # Regression Runner
    "RegressionConfig",
    "RegressionReport",
    "RegressionRunner",
    "TestResult",
    "run_regression_sync",
    "run_regression_async",
    "check_regression_passed",
]
