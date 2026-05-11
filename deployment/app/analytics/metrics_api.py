"""FastAPI router for the analytics metrics API."""

from fastapi import APIRouter

from app.analytics.metrics_collector import get_metrics_collector

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


@router.get("/metrics")
async def get_metrics() -> dict:
    """Return a snapshot of all platform metrics.

    Returns:
        Metrics snapshot including per-agent stats, token usage,
        latency percentiles, and active session count.
    """
    return get_metrics_collector().snapshot()
