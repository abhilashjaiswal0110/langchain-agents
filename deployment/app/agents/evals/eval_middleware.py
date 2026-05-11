"""Background evaluation sampler for enterprise agent invocations.

Usage:
    from app.agents.evals.eval_middleware import submit_for_evaluation
    from fastapi import BackgroundTasks

    # Inside an endpoint handler:
    submit_for_evaluation(background_tasks, agent_type, input_msg, output, run_id=run_id)
"""

import logging
import os
import random

from fastapi import BackgroundTasks

from app.agents.evals.langsmith_evaluator import LangSmithConfig, LangSmithEvaluator

logger = logging.getLogger(__name__)


class EvalMiddleware:
    """Probabilistic sampler that submits agent responses to LangSmith.

    Args:
        sampling_rate: Fraction of requests to evaluate (0.0–1.0).
                       Defaults to ``EVAL_SAMPLING_RATE`` env var or ``0.1``.
    """

    def __init__(self, sampling_rate: float | None = None) -> None:
        if sampling_rate is None:
            sampling_rate = float(os.getenv("EVAL_SAMPLING_RATE", "0.1"))
        self.sampling_rate = sampling_rate
        self._evaluator: LangSmithEvaluator | None = None

    def _get_evaluator(self) -> LangSmithEvaluator:
        if self._evaluator is None:
            self._evaluator = LangSmithEvaluator(config=LangSmithConfig.from_env())
        return self._evaluator

    def maybe_evaluate(
        self,
        background_tasks: BackgroundTasks,
        agent_type: str,
        input_msg: str,
        output: str,
        run_id: str | None = None,
    ) -> None:
        """Probabilistically enqueue an evaluation as a background task.

        Args:
            background_tasks: FastAPI BackgroundTasks scheduler.
            agent_type: The agent that produced the response.
            input_msg: User input.
            output: Agent response.
            run_id: Optional LangSmith run ID.
        """
        if random.random() >= self.sampling_rate:
            return

        evaluator = self._get_evaluator()
        background_tasks.add_task(
            evaluator.evaluate_async,
            agent_type,
            input_msg,
            output,
            run_id,
        )
        logger.debug("Enqueued evaluation for agent=%s run_id=%s", agent_type, run_id)


_middleware = EvalMiddleware()


def submit_for_evaluation(
    background_tasks: BackgroundTasks,
    agent_type: str,
    input_msg: str,
    output: str,
    run_id: str | None = None,
) -> None:
    """Submit an agent response for sampled LangSmith evaluation.

    Convenience wrapper around the module-level EvalMiddleware singleton.

    Args:
        background_tasks: FastAPI BackgroundTasks scheduler from the endpoint.
        agent_type: The agent type string (e.g. ``"research"``).
        input_msg: The user input passed to the agent.
        output: The agent's response string.
        run_id: Optional LangSmith run ID for feedback attachment.
    """
    _middleware.maybe_evaluate(background_tasks, agent_type, input_msg, output, run_id=run_id)
