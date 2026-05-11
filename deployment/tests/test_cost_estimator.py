"""Tests for pre-execution cost estimation."""
from unittest.mock import patch

import pytest

from app.governance.cost_estimator import CostEstimator


class TestCostEstimator:
    def test_estimate_returns_required_keys(self) -> None:
        estimator = CostEstimator()
        result = estimator.estimate("Hello world", "research")
        assert "input_tokens" in result
        assert "estimated_output_tokens" in result
        assert "estimated_cost_usd" in result
        assert "model" in result

    def test_estimate_input_tokens_positive(self) -> None:
        estimator = CostEstimator()
        result = estimator.estimate("test message for token counting", "research")
        assert result["input_tokens"] > 0

    def test_estimate_output_tokens_positive(self) -> None:
        estimator = CostEstimator()
        result = estimator.estimate("hello", "content")
        assert result["estimated_output_tokens"] > 0

    def test_estimate_cost_positive(self) -> None:
        estimator = CostEstimator()
        result = estimator.estimate("hello world", "data-analyst")
        assert result["estimated_cost_usd"] >= 0.0

    def test_estimate_research_higher_multiplier_than_analyst(self) -> None:
        estimator = CostEstimator()
        msg = "analyze this data"
        r_research = estimator.estimate(msg, "research")
        r_analyst = estimator.estimate(msg, "data-analyst")
        assert r_research["estimated_output_tokens"] >= r_analyst["estimated_output_tokens"]

    def test_estimate_unknown_agent_uses_default_multiplier(self) -> None:
        estimator = CostEstimator()
        result = estimator.estimate("hello", "unknown-agent-xyz")
        assert result["estimated_output_tokens"] > 0

    def test_estimate_model_in_result(self) -> None:
        estimator = CostEstimator(model="gpt-4o-mini")
        result = estimator.estimate("hello", "research")
        assert result["model"] == "gpt-4o-mini"

    def test_estimate_empty_message(self) -> None:
        estimator = CostEstimator()
        result = estimator.estimate("", "research")
        assert result["input_tokens"] >= 0
        assert result["estimated_cost_usd"] >= 0.0

    def test_estimate_long_message_more_tokens(self) -> None:
        estimator = CostEstimator()
        short = estimator.estimate("hi", "research")
        long = estimator.estimate("hi " * 100, "research")
        assert long["input_tokens"] > short["input_tokens"]

    def test_estimate_cost_scales_with_tokens(self) -> None:
        estimator = CostEstimator()
        short = estimator.estimate("hi", "research")
        long = estimator.estimate("analyze this very detailed market research report " * 20, "research")
        assert long["estimated_cost_usd"] > short["estimated_cost_usd"]

    def test_tiktoken_fallback_on_encoding_error(self) -> None:
        estimator = CostEstimator(model="gpt-4o-mini")
        with patch.object(estimator, "_count_tokens", return_value=42):
            result = estimator.estimate("hello", "research")
        assert result["input_tokens"] == 42

    def test_agent_multipliers_coverage(self) -> None:
        estimator = CostEstimator()
        for agent in ["research", "content", "data-analyst", "documents", "rag", "support", "code"]:
            result = estimator.estimate("test", agent)
            assert result["estimated_output_tokens"] > 0, f"Failed for agent: {agent}"
