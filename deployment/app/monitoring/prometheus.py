"""Prometheus metrics definitions for the enterprise agents platform.

Exposes key observability signals at /metrics in OpenMetrics format,
compatible with Prometheus scraping and Grafana dashboards.
"""
import logging

from fastapi import FastAPI

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram
    from prometheus_fastapi_instrumentator import Instrumentator

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus-client or prometheus-fastapi-instrumentator not installed; "
                   "metrics endpoint will be unavailable")


def _noop_counter(*args, **kwargs):  # type: ignore[no-untyped-def]
    class _Noop:
        def labels(self, **kw):  # type: ignore[no-untyped-def]
            return self

        def inc(self, amount: float = 1) -> None:
            pass

        def observe(self, amount: float) -> None:
            pass

        def set(self, value: float) -> None:
            pass

    return _Noop()


if _PROMETHEUS_AVAILABLE:
    agent_requests_total = Counter(
        "agent_requests_total",
        "Total agent invocations",
        ["agent_type", "status"],
    )
    agent_latency_seconds = Histogram(
        "agent_latency_seconds",
        "Agent response latency in seconds",
        ["agent_type"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
    )
    llm_tokens_total = Counter(
        "llm_tokens_total",
        "LLM tokens consumed",
        ["model", "token_type"],
    )
    active_sessions = Gauge(
        "active_sessions",
        "Number of active conversation sessions",
    )
else:
    agent_requests_total = _noop_counter()  # type: ignore[assignment]
    agent_latency_seconds = _noop_counter()  # type: ignore[assignment]
    llm_tokens_total = _noop_counter()  # type: ignore[assignment]
    active_sessions = _noop_counter()  # type: ignore[assignment]


def setup_metrics(app: FastAPI) -> None:
    """Instrument a FastAPI app with Prometheus metrics at /metrics.

    Args:
        app: The FastAPI application instance to instrument.
    """
    if not _PROMETHEUS_AVAILABLE:
        logger.warning("Skipping Prometheus setup — library not available")
        return

    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=False,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics", "/health"],
        inprogress_name="http_requests_inprogress",
        inprogress_labels=True,
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    logger.info("Prometheus metrics endpoint mounted at /metrics")
