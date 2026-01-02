"""Cost tracking for LLM token usage.

Provides token usage tracking and cost estimation for enterprise
governance and budgeting. Tracks:
- Input/output tokens per request
- Costs by model, user, agent, and session
- Daily/monthly budgets and alerts
- Usage analytics and reporting

Usage:
    from app.governance.cost_tracker import (
        CostTracker, TokenUsage, CostConfig,
        get_cost_tracker, track_usage, get_usage_summary,
    )

    # Track usage
    usage = track_usage(
        model="gpt-4o-mini",
        input_tokens=100,
        output_tokens=50,
        user_id="user123",
        agent_type="research",
    )

    # Get summary
    summary = get_usage_summary(user_id="user123")
    print(f"Total cost: ${summary.total_cost:.4f}")

    # Check budget
    tracker = get_cost_tracker()
    if tracker.check_budget("user123"):
        # Within budget
        pass
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """LLM model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    CUSTOM = "custom"


@dataclass
class ModelPricing:
    """Pricing information for a model.

    Attributes:
        model_name: Model identifier.
        provider: Model provider.
        input_price_per_1k: Price per 1000 input tokens (USD).
        output_price_per_1k: Price per 1000 output tokens (USD).
        cached_input_price_per_1k: Price for cached input tokens.
    """

    model_name: str
    provider: ModelProvider
    input_price_per_1k: float
    output_price_per_1k: float
    cached_input_price_per_1k: float = 0.0

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """Calculate cost for token usage.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            cached_tokens: Number of cached input tokens.

        Returns:
            Total cost in USD.
        """
        input_cost = (input_tokens / 1000) * self.input_price_per_1k
        output_cost = (output_tokens / 1000) * self.output_price_per_1k
        cached_cost = (cached_tokens / 1000) * self.cached_input_price_per_1k

        return input_cost + output_cost + cached_cost


# Default pricing (as of late 2024/early 2025)
DEFAULT_PRICING: dict[str, ModelPricing] = {
    # OpenAI models
    "gpt-4o": ModelPricing(
        model_name="gpt-4o",
        provider=ModelProvider.OPENAI,
        input_price_per_1k=0.005,
        output_price_per_1k=0.015,
        cached_input_price_per_1k=0.0025,
    ),
    "gpt-4o-mini": ModelPricing(
        model_name="gpt-4o-mini",
        provider=ModelProvider.OPENAI,
        input_price_per_1k=0.00015,
        output_price_per_1k=0.0006,
        cached_input_price_per_1k=0.000075,
    ),
    "gpt-4-turbo": ModelPricing(
        model_name="gpt-4-turbo",
        provider=ModelProvider.OPENAI,
        input_price_per_1k=0.01,
        output_price_per_1k=0.03,
    ),
    "gpt-3.5-turbo": ModelPricing(
        model_name="gpt-3.5-turbo",
        provider=ModelProvider.OPENAI,
        input_price_per_1k=0.0005,
        output_price_per_1k=0.0015,
    ),
    # Anthropic models
    "claude-3-5-sonnet-20241022": ModelPricing(
        model_name="claude-3-5-sonnet-20241022",
        provider=ModelProvider.ANTHROPIC,
        input_price_per_1k=0.003,
        output_price_per_1k=0.015,
        cached_input_price_per_1k=0.0003,
    ),
    "claude-3-opus-20240229": ModelPricing(
        model_name="claude-3-opus-20240229",
        provider=ModelProvider.ANTHROPIC,
        input_price_per_1k=0.015,
        output_price_per_1k=0.075,
        cached_input_price_per_1k=0.0015,
    ),
    "claude-3-haiku-20240307": ModelPricing(
        model_name="claude-3-haiku-20240307",
        provider=ModelProvider.ANTHROPIC,
        input_price_per_1k=0.00025,
        output_price_per_1k=0.00125,
        cached_input_price_per_1k=0.00003,
    ),
}


@dataclass
class TokenUsage:
    """Record of token usage for a request.

    Attributes:
        request_id: Unique request identifier.
        model: Model used.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        cached_tokens: Number of cached input tokens.
        total_tokens: Total tokens used.
        cost: Estimated cost in USD.
        user_id: User who made the request.
        agent_type: Type of agent used.
        session_id: Session identifier.
        timestamp: When the usage occurred.
        metadata: Additional metadata.
    """

    request_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    user_id: str = ""
    agent_type: str = ""
    session_id: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate total tokens if not set."""
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class UsageSummary:
    """Summary of token usage.

    Attributes:
        total_requests: Number of requests.
        total_input_tokens: Total input tokens.
        total_output_tokens: Total output tokens.
        total_cached_tokens: Total cached tokens.
        total_tokens: Total tokens overall.
        total_cost: Total cost in USD.
        by_model: Usage breakdown by model.
        by_agent: Usage breakdown by agent type.
        by_user: Usage breakdown by user.
        period_start: Start of the period.
        period_end: End of the period.
    """

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    by_model: dict[str, dict] = field(default_factory=dict)
    by_agent: dict[str, dict] = field(default_factory=dict)
    by_user: dict[str, dict] = field(default_factory=dict)
    period_start: datetime | None = None
    period_end: datetime | None = None


@dataclass
class BudgetConfig:
    """Budget configuration.

    Attributes:
        daily_limit: Daily budget limit in USD.
        monthly_limit: Monthly budget limit in USD.
        per_user_daily: Per-user daily limit.
        per_user_monthly: Per-user monthly limit.
        per_agent_daily: Per-agent daily limit.
        alert_threshold: Percentage at which to alert (0.0-1.0).
        block_on_exceed: Whether to block on budget exceeded.
    """

    daily_limit: float = 100.0
    monthly_limit: float = 2000.0
    per_user_daily: float = 10.0
    per_user_monthly: float = 200.0
    per_agent_daily: float = 50.0
    alert_threshold: float = 0.8
    block_on_exceed: bool = False


@dataclass
class CostConfig:
    """Configuration for cost tracker.

    Attributes:
        enabled: Whether cost tracking is enabled.
        pricing: Custom model pricing.
        budget: Budget configuration.
        retention_days: How long to keep usage records.
        alert_callback: Callback for budget alerts.
    """

    enabled: bool = True
    pricing: dict[str, ModelPricing] = field(default_factory=lambda: DEFAULT_PRICING.copy())
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    retention_days: int = 90
    alert_callback: Callable[[str, float, float], None] | None = None


class CostTracker:
    """Tracks token usage and costs.

    Provides:
    - Token usage recording
    - Cost estimation
    - Budget monitoring
    - Usage analytics
    """

    def __init__(self, config: CostConfig | None = None) -> None:
        """Initialize cost tracker.

        Args:
            config: Tracker configuration.
        """
        self.config = config or CostConfig()
        self._usage_records: list[TokenUsage] = []
        self._daily_usage: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._monthly_usage: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def get_pricing(self, model: str) -> ModelPricing | None:
        """Get pricing for a model.

        Args:
            model: Model name.

        Returns:
            Pricing info or None if not found.
        """
        # Try exact match
        if model in self.config.pricing:
            return self.config.pricing[model]

        # Try partial match
        for name, pricing in self.config.pricing.items():
            if name in model or model in name:
                return pricing

        return None

    def add_pricing(self, pricing: ModelPricing) -> None:
        """Add or update model pricing.

        Args:
            pricing: Model pricing to add.
        """
        self.config.pricing[pricing.model_name] = pricing

    def track(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        user_id: str = "",
        agent_type: str = "",
        session_id: str = "",
        request_id: str = "",
        metadata: dict | None = None,
    ) -> TokenUsage:
        """Track token usage for a request.

        Args:
            model: Model used.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            cached_tokens: Number of cached tokens.
            user_id: User identifier.
            agent_type: Agent type.
            session_id: Session identifier.
            request_id: Request identifier.
            metadata: Additional metadata.

        Returns:
            Token usage record.
        """
        if not self.config.enabled:
            return TokenUsage(
                request_id=request_id or "",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        # Generate request ID if not provided
        if not request_id:
            import uuid

            request_id = str(uuid.uuid4())

        # Calculate cost
        pricing = self.get_pricing(model)
        cost = 0.0
        if pricing:
            cost = pricing.calculate_cost(input_tokens, output_tokens, cached_tokens)
        else:
            # Default fallback pricing
            cost = ((input_tokens + output_tokens) / 1000) * 0.002
            logger.warning(f"No pricing found for model {model}, using fallback")

        # Create usage record
        usage = TokenUsage(
            request_id=request_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost=cost,
            user_id=user_id,
            agent_type=agent_type,
            session_id=session_id,
            metadata=metadata or {},
        )

        # Store record
        self._usage_records.append(usage)

        # Update aggregates
        today = datetime.now().strftime("%Y-%m-%d")
        month = datetime.now().strftime("%Y-%m")

        self._daily_usage[today][user_id] += cost
        self._daily_usage[today][f"agent:{agent_type}"] += cost
        self._daily_usage[today]["total"] += cost

        self._monthly_usage[month][user_id] += cost
        self._monthly_usage[month][f"agent:{agent_type}"] += cost
        self._monthly_usage[month]["total"] += cost

        # Check budget alerts
        self._check_budget_alerts(user_id, agent_type)

        logger.debug(
            f"Tracked usage: model={model}, tokens={usage.total_tokens}, cost=${cost:.6f}"
        )

        return usage

    def _check_budget_alerts(self, user_id: str, agent_type: str) -> None:
        """Check and trigger budget alerts.

        Args:
            user_id: User to check.
            agent_type: Agent type to check.
        """
        today = datetime.now().strftime("%Y-%m-%d")
        month = datetime.now().strftime("%Y-%m")
        budget = self.config.budget
        threshold = budget.alert_threshold

        # Check daily user budget
        user_daily = self._daily_usage[today].get(user_id, 0)
        if user_daily >= budget.per_user_daily * threshold:
            self._trigger_alert(
                f"User {user_id} daily budget",
                user_daily,
                budget.per_user_daily,
            )

        # Check monthly user budget
        user_monthly = self._monthly_usage[month].get(user_id, 0)
        if user_monthly >= budget.per_user_monthly * threshold:
            self._trigger_alert(
                f"User {user_id} monthly budget",
                user_monthly,
                budget.per_user_monthly,
            )

        # Check daily agent budget
        agent_key = f"agent:{agent_type}"
        agent_daily = self._daily_usage[today].get(agent_key, 0)
        if agent_daily >= budget.per_agent_daily * threshold:
            self._trigger_alert(
                f"Agent {agent_type} daily budget",
                agent_daily,
                budget.per_agent_daily,
            )

        # Check total daily budget
        total_daily = self._daily_usage[today].get("total", 0)
        if total_daily >= budget.daily_limit * threshold:
            self._trigger_alert("Daily total budget", total_daily, budget.daily_limit)

        # Check total monthly budget
        total_monthly = self._monthly_usage[month].get("total", 0)
        if total_monthly >= budget.monthly_limit * threshold:
            self._trigger_alert("Monthly total budget", total_monthly, budget.monthly_limit)

    def _trigger_alert(self, alert_type: str, current: float, limit: float) -> None:
        """Trigger a budget alert.

        Args:
            alert_type: Type of alert.
            current: Current usage.
            limit: Budget limit.
        """
        percentage = (current / limit) * 100
        logger.warning(
            f"Budget alert: {alert_type} at {percentage:.1f}% "
            f"(${current:.4f} / ${limit:.2f})"
        )

        if self.config.alert_callback:
            try:
                self.config.alert_callback(alert_type, current, limit)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def check_budget(
        self,
        user_id: str = "",
        agent_type: str = "",
        period: str = "daily",
    ) -> bool:
        """Check if within budget limits.

        Args:
            user_id: User to check (empty for total).
            agent_type: Agent type to check.
            period: "daily" or "monthly".

        Returns:
            True if within budget.
        """
        budget = self.config.budget
        today = datetime.now().strftime("%Y-%m-%d")
        month = datetime.now().strftime("%Y-%m")

        if period == "daily":
            usage_data = self._daily_usage[today]
            if user_id:
                return usage_data.get(user_id, 0) < budget.per_user_daily
            if agent_type:
                return usage_data.get(f"agent:{agent_type}", 0) < budget.per_agent_daily
            return usage_data.get("total", 0) < budget.daily_limit
        else:
            usage_data = self._monthly_usage[month]
            if user_id:
                return usage_data.get(user_id, 0) < budget.per_user_monthly
            return usage_data.get("total", 0) < budget.monthly_limit

    def get_usage(
        self,
        user_id: str = "",
        agent_type: str = "",
        session_id: str = "",
        model: str = "",
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 100,
    ) -> list[TokenUsage]:
        """Get usage records with filtering.

        Args:
            user_id: Filter by user.
            agent_type: Filter by agent type.
            session_id: Filter by session.
            model: Filter by model.
            start_time: Filter by start timestamp.
            end_time: Filter by end timestamp.
            limit: Maximum records to return.

        Returns:
            Filtered usage records.
        """
        records = self._usage_records

        if user_id:
            records = [r for r in records if r.user_id == user_id]
        if agent_type:
            records = [r for r in records if r.agent_type == agent_type]
        if session_id:
            records = [r for r in records if r.session_id == session_id]
        if model:
            records = [r for r in records if r.model == model]
        if start_time:
            records = [r for r in records if r.timestamp >= start_time]
        if end_time:
            records = [r for r in records if r.timestamp <= end_time]

        # Sort by timestamp descending and limit
        records = sorted(records, key=lambda r: r.timestamp, reverse=True)
        return records[:limit]

    def get_summary(
        self,
        user_id: str = "",
        agent_type: str = "",
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> UsageSummary:
        """Get usage summary.

        Args:
            user_id: Filter by user.
            agent_type: Filter by agent type.
            start_time: Period start timestamp.
            end_time: Period end timestamp.

        Returns:
            Usage summary.
        """
        # Get filtered records
        records = self.get_usage(
            user_id=user_id,
            agent_type=agent_type,
            start_time=start_time,
            end_time=end_time,
            limit=10000,
        )

        # Calculate summary
        summary = UsageSummary(
            total_requests=len(records),
            period_start=datetime.fromtimestamp(start_time) if start_time else None,
            period_end=datetime.fromtimestamp(end_time) if end_time else None,
        )

        by_model: dict[str, dict] = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0})
        by_agent: dict[str, dict] = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0})
        by_user: dict[str, dict] = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0})

        for record in records:
            summary.total_input_tokens += record.input_tokens
            summary.total_output_tokens += record.output_tokens
            summary.total_cached_tokens += record.cached_tokens
            summary.total_tokens += record.total_tokens
            summary.total_cost += record.cost

            # By model
            by_model[record.model]["requests"] += 1
            by_model[record.model]["tokens"] += record.total_tokens
            by_model[record.model]["cost"] += record.cost

            # By agent
            if record.agent_type:
                by_agent[record.agent_type]["requests"] += 1
                by_agent[record.agent_type]["tokens"] += record.total_tokens
                by_agent[record.agent_type]["cost"] += record.cost

            # By user
            if record.user_id:
                by_user[record.user_id]["requests"] += 1
                by_user[record.user_id]["tokens"] += record.total_tokens
                by_user[record.user_id]["cost"] += record.cost

        summary.by_model = dict(by_model)
        summary.by_agent = dict(by_agent)
        summary.by_user = dict(by_user)

        return summary

    def get_daily_cost(self, date: str | None = None) -> float:
        """Get total cost for a day.

        Args:
            date: Date string (YYYY-MM-DD), default today.

        Returns:
            Total cost for the day.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        return self._daily_usage.get(date, {}).get("total", 0.0)

    def get_monthly_cost(self, month: str | None = None) -> float:
        """Get total cost for a month.

        Args:
            month: Month string (YYYY-MM), default current month.

        Returns:
            Total cost for the month.
        """
        if month is None:
            month = datetime.now().strftime("%Y-%m")
        return self._monthly_usage.get(month, {}).get("total", 0.0)

    def cleanup_old_records(self) -> int:
        """Remove records older than retention period.

        Returns:
            Number of records removed.
        """
        cutoff = time.time() - (self.config.retention_days * 24 * 60 * 60)
        original_count = len(self._usage_records)

        self._usage_records = [r for r in self._usage_records if r.timestamp >= cutoff]

        removed = original_count - len(self._usage_records)
        if removed > 0:
            logger.info(f"Cleaned up {removed} old usage records")

        return removed

    def export_records(
        self,
        filepath: str,
        format: str = "jsonl",
    ) -> int:
        """Export usage records to file.

        Args:
            filepath: Output file path.
            format: Output format (jsonl, csv).

        Returns:
            Number of records exported.
        """
        import json

        if format == "jsonl":
            with open(filepath, "w") as f:
                for record in self._usage_records:
                    f.write(
                        json.dumps(
                            {
                                "request_id": record.request_id,
                                "model": record.model,
                                "input_tokens": record.input_tokens,
                                "output_tokens": record.output_tokens,
                                "cached_tokens": record.cached_tokens,
                                "total_tokens": record.total_tokens,
                                "cost": record.cost,
                                "user_id": record.user_id,
                                "agent_type": record.agent_type,
                                "session_id": record.session_id,
                                "timestamp": record.timestamp,
                                "metadata": record.metadata,
                            }
                        )
                        + "\n"
                    )
        elif format == "csv":
            import csv

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "request_id",
                        "model",
                        "input_tokens",
                        "output_tokens",
                        "cached_tokens",
                        "total_tokens",
                        "cost",
                        "user_id",
                        "agent_type",
                        "session_id",
                        "timestamp",
                    ]
                )
                for record in self._usage_records:
                    writer.writerow(
                        [
                            record.request_id,
                            record.model,
                            record.input_tokens,
                            record.output_tokens,
                            record.cached_tokens,
                            record.total_tokens,
                            record.cost,
                            record.user_id,
                            record.agent_type,
                            record.session_id,
                            record.timestamp,
                        ]
                    )

        return len(self._usage_records)


# Singleton pattern for global tracker
_cost_tracker: CostTracker | None = None


def get_cost_tracker(config: CostConfig | None = None) -> CostTracker:
    """Get or create global cost tracker instance.

    Args:
        config: Optional configuration (used only on first call).

    Returns:
        Global cost tracker instance.
    """
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker(config)
    return _cost_tracker


def reset_cost_tracker() -> None:
    """Reset global cost tracker instance."""
    global _cost_tracker
    _cost_tracker = None


def track_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    user_id: str = "",
    agent_type: str = "",
    session_id: str = "",
    request_id: str = "",
    metadata: dict | None = None,
) -> TokenUsage:
    """Convenience function to track token usage.

    Args:
        model: Model used.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        cached_tokens: Number of cached tokens.
        user_id: User identifier.
        agent_type: Agent type.
        session_id: Session identifier.
        request_id: Request identifier.
        metadata: Additional metadata.

    Returns:
        Token usage record.
    """
    tracker = get_cost_tracker()
    return tracker.track(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        user_id=user_id,
        agent_type=agent_type,
        session_id=session_id,
        request_id=request_id,
        metadata=metadata,
    )


def get_usage_summary(
    user_id: str = "",
    agent_type: str = "",
    start_time: float | None = None,
    end_time: float | None = None,
) -> UsageSummary:
    """Convenience function to get usage summary.

    Args:
        user_id: Filter by user.
        agent_type: Filter by agent type.
        start_time: Period start timestamp.
        end_time: Period end timestamp.

    Returns:
        Usage summary.
    """
    tracker = get_cost_tracker()
    return tracker.get_summary(
        user_id=user_id,
        agent_type=agent_type,
        start_time=start_time,
        end_time=end_time,
    )


class BudgetExceededError(Exception):
    """Raised when budget limit is exceeded."""

    def __init__(
        self,
        message: str,
        budget_type: str,
        current: float,
        limit: float,
    ) -> None:
        """Initialize error.

        Args:
            message: Error message.
            budget_type: Type of budget exceeded.
            current: Current usage.
            limit: Budget limit.
        """
        super().__init__(message)
        self.budget_type = budget_type
        self.current = current
        self.limit = limit
