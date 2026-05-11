"""In-process metrics collector for the analytics dashboard.

Thread-safe counters and histograms covering:
- Request counts and error rates per agent
- Latency histograms (p50/p95/p99)
- Token usage totals
- Active session counts

Usage:
    from app.analytics.metrics_collector import get_metrics_collector

    m = get_metrics_collector()
    with m.track_request("research"):
        # ... invoke agent ...
        pass
    m.record_tokens("research", input_tokens=120, output_tokens=350)
"""

import contextlib
import logging
import time
from collections import defaultdict, deque
from collections.abc import Generator
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

_LATENCY_WINDOW = 1000  # keep last N samples per agent for percentile calculation


@dataclass
class AgentStats:
    """Per-agent counters and histograms.

    Attributes:
        requests_total: Total invocations.
        errors_total: Total failed invocations.
        input_tokens_total: Cumulative input token count.
        output_tokens_total: Cumulative output token count.
        latencies_ms: Rolling window of recent latency samples (ms).
    """

    requests_total: int = 0
    errors_total: int = 0
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=_LATENCY_WINDOW))


class MetricsCollector:
    """Thread-safe in-process metrics store for agent observability."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._agents: dict[str, AgentStats] = defaultdict(AgentStats)
        self._active_sessions: int = 0
        self._started_at: float = time.time()

    def _stats(self, agent_type: str) -> AgentStats:
        return self._agents[agent_type]

    @contextlib.contextmanager
    def track_request(self, agent_type: str) -> Generator[None, None, None]:
        """Context manager that records latency and increments counters.

        Args:
            agent_type: Agent identifier string.

        Yields:
            None — wraps the agent invocation.
        """
        start = time.perf_counter()
        with self._lock:
            self._stats(agent_type).requests_total += 1
        try:
            yield
        except Exception:
            with self._lock:
                self._stats(agent_type).errors_total += 1
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            with self._lock:
                self._stats(agent_type).latencies_ms.append(elapsed_ms)

    def record_tokens(
        self,
        agent_type: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record token usage for an agent invocation.

        Args:
            agent_type: Agent identifier string.
            input_tokens: Number of input tokens used.
            output_tokens: Number of output tokens produced.
        """
        with self._lock:
            s = self._stats(agent_type)
            s.input_tokens_total += input_tokens
            s.output_tokens_total += output_tokens

    def set_active_sessions(self, count: int) -> None:
        """Update the global active session count.

        Args:
            count: Current number of active sessions.
        """
        with self._lock:
            self._active_sessions = count

    def increment_active_sessions(self, delta: int = 1) -> None:
        """Increment or decrement the active session counter.

        Args:
            delta: Amount to add (negative to decrement).
        """
        with self._lock:
            self._active_sessions = max(0, self._active_sessions + delta)

    def _percentile(self, samples: deque, pct: float) -> float:
        if not samples:
            return 0.0
        sorted_s = sorted(samples)
        idx = int(len(sorted_s) * pct / 100)
        return sorted_s[min(idx, len(sorted_s) - 1)]

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time snapshot of all metrics.

        Returns:
            Dict suitable for JSON serialisation.
        """
        with self._lock:
            agents_out = {}
            for agent_type, s in self._agents.items():
                error_rate = s.errors_total / s.requests_total if s.requests_total else 0.0
                agents_out[agent_type] = {
                    "requests_total": s.requests_total,
                    "errors_total": s.errors_total,
                    "error_rate": round(error_rate, 4),
                    "input_tokens_total": s.input_tokens_total,
                    "output_tokens_total": s.output_tokens_total,
                    "latency_p50_ms": round(self._percentile(s.latencies_ms, 50), 1),
                    "latency_p95_ms": round(self._percentile(s.latencies_ms, 95), 1),
                    "latency_p99_ms": round(self._percentile(s.latencies_ms, 99), 1),
                }

            total_requests = sum(s.requests_total for s in self._agents.values())
            total_errors = sum(s.errors_total for s in self._agents.values())

            return {
                "uptime_seconds": round(time.time() - self._started_at, 1),
                "active_sessions": self._active_sessions,
                "total_requests": total_requests,
                "total_errors": total_errors,
                "agents": agents_out,
            }


_collector_instance: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Return the module-level MetricsCollector singleton.

    Returns:
        Shared MetricsCollector instance.
    """
    global _collector_instance
    if _collector_instance is None:
        _collector_instance = MetricsCollector()
    return _collector_instance
