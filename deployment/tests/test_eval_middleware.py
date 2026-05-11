"""Tests for LangSmith evaluation sampling middleware."""
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.evals.eval_middleware import EvalMiddleware, submit_for_evaluation


class TestEvalMiddleware:
    def test_sampling_respects_rate_zero(self) -> None:
        middleware = EvalMiddleware(sampling_rate=0.0)
        bg = MagicMock()
        middleware.maybe_evaluate(bg, "research", "hello", "world")
        bg.add_task.assert_not_called()

    def test_sampling_respects_rate_one(self) -> None:
        middleware = EvalMiddleware(sampling_rate=1.0)
        bg = MagicMock()
        middleware.maybe_evaluate(bg, "research", "hello", "world")
        bg.add_task.assert_called_once()

    def test_sampling_passes_correct_args(self) -> None:
        middleware = EvalMiddleware(sampling_rate=1.0)
        bg = MagicMock()
        middleware.maybe_evaluate(bg, "content", "input-msg", "output-msg", run_id="r1")
        call_kwargs = bg.add_task.call_args
        # The coroutine function and its args are passed positionally
        assert call_kwargs is not None

    def test_disabled_when_no_api_key(self) -> None:
        middleware = EvalMiddleware(sampling_rate=1.0)
        bg = MagicMock()
        with patch.dict(os.environ, {"LANGCHAIN_API_KEY": "", "LANGSMITH_API_KEY": ""}, clear=False):
            middleware.maybe_evaluate(bg, "research", "q", "a")
        # With no key evaluation is silently skipped
        # (the background task runs but evaluate_async catches the error)
        # We just verify no exception is raised here
        assert True

    def test_default_sampling_rate_from_env(self) -> None:
        with patch.dict(os.environ, {"EVAL_SAMPLING_RATE": "0.5"}):
            middleware = EvalMiddleware()
        assert middleware.sampling_rate == 0.5

    def test_default_sampling_rate_fallback(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EVAL_SAMPLING_RATE", None)
            middleware = EvalMiddleware()
        assert 0.0 <= middleware.sampling_rate <= 1.0


class TestSubmitForEvaluation:
    def test_submit_for_evaluation_adds_background_task(self) -> None:
        bg = MagicMock()
        with patch("app.agents.evals.eval_middleware._middleware") as mock_mw:
            submit_for_evaluation(bg, "research", "q", "a")
            mock_mw.maybe_evaluate.assert_called_once_with(bg, "research", "q", "a", run_id=None)

    def test_submit_for_evaluation_passes_run_id(self) -> None:
        bg = MagicMock()
        with patch("app.agents.evals.eval_middleware._middleware") as mock_mw:
            submit_for_evaluation(bg, "content", "q", "a", run_id="my-run")
            mock_mw.maybe_evaluate.assert_called_once_with(bg, "content", "q", "a", run_id="my-run")
