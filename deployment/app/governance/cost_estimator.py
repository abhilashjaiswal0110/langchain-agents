"""Pre-execution cost estimation for enterprise agent invocations.

Counts input tokens with tiktoken and applies per-agent output multipliers
to produce a cost estimate before a request is sent to the LLM.

Usage:
    from app.governance.cost_estimator import CostEstimator

    estimator = CostEstimator()
    result = estimator.estimate("Summarize the Q3 report", "research")
    print(f"Estimated cost: ${result['estimated_cost_usd']:.6f}")
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Conservative output-to-input token multiplier per agent type.
# Research and content agents produce much longer outputs than support agents.
_OUTPUT_MULTIPLIERS: dict[str, int] = {
    "research": 5,
    "content": 4,
    "data-analyst": 3,
    "documents": 3,
    "rag": 3,
    "support": 2,
    "code": 3,
    "document-intelligence": 3,
}
_DEFAULT_MULTIPLIER = 3

# Pricing per 1 000 tokens (USD) for common models
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "o4-mini": (0.00015, 0.0006),
}
_DEFAULT_PRICING = (0.00015, 0.0006)  # gpt-4o-mini rates as safe default


class CostEstimator:
    """Estimates token counts and USD cost before executing an agent request.

    Args:
        model: OpenAI-compatible model name used for tiktoken encoding and pricing.
    """

    def __init__(
        self,
        model: str = os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
    ) -> None:
        self._model = model
        self._encoder: Any = None

    def _get_encoder(self) -> Any:
        """Return (and cache) the tiktoken encoder for the configured model.

        Returns:
            Tiktoken encoding object, or None if tiktoken is unavailable.
        """
        if self._encoder is not None:
            return self._encoder
        try:
            import tiktoken  # type: ignore[import]

            try:
                self._encoder = tiktoken.encoding_for_model(self._model)
            except KeyError:
                self._encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning("tiktoken not installed; using character-based token estimate")
        return self._encoder

    def _count_tokens(self, text: str) -> int:
        """Count tokens in *text* using tiktoken, falling back to char/4.

        Args:
            text: Text to count tokens for.

        Returns:
            Estimated token count.
        """
        encoder = self._get_encoder()
        if encoder is not None:
            return len(encoder.encode(text))
        return max(1, len(text) // 4)

    def estimate(self, message: str, agent_type: str) -> dict[str, Any]:
        """Estimate the cost of invoking an agent with a given message.

        Args:
            message: User input that will be sent to the agent.
            agent_type: Agent identifier used to select the output multiplier.

        Returns:
            Dict with keys:
                - ``input_tokens``: Counted input tokens.
                - ``estimated_output_tokens``: Projected output tokens.
                - ``estimated_cost_usd``: Estimated USD cost.
                - ``model``: Model name used for estimation.
        """
        input_tokens = self._count_tokens(message)
        multiplier = _OUTPUT_MULTIPLIERS.get(agent_type, _DEFAULT_MULTIPLIER)
        estimated_output_tokens = input_tokens * multiplier

        input_price, output_price = _PRICING.get(self._model, _DEFAULT_PRICING)
        estimated_cost = (
            (input_tokens / 1000) * input_price
            + (estimated_output_tokens / 1000) * output_price
        )

        return {
            "input_tokens": input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_cost_usd": round(estimated_cost, 8),
            "model": self._model,
        }


_estimator_instance: CostEstimator | None = None


def get_cost_estimator() -> CostEstimator:
    """Return the module-level CostEstimator singleton.

    Returns:
        Shared CostEstimator instance.
    """
    global _estimator_instance
    if _estimator_instance is None:
        _estimator_instance = CostEstimator()
    return _estimator_instance
