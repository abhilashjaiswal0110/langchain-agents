"""Tests for Prometheus metrics module."""
from unittest.mock import MagicMock, patch

import pytest


class TestPrometheusMetrics:
    def test_module_imports(self) -> None:
        from app.monitoring.prometheus import (
            active_sessions,
            agent_latency_seconds,
            agent_requests_total,
            llm_tokens_total,
            setup_metrics,
        )
        assert agent_requests_total is not None
        assert agent_latency_seconds is not None
        assert llm_tokens_total is not None
        assert active_sessions is not None
        assert callable(setup_metrics)

    def test_noop_metrics_dont_raise(self) -> None:
        with patch.dict("sys.modules", {"prometheus_client": None, "prometheus_fastapi_instrumentator": None}):
            import importlib
            import app.monitoring.prometheus as prom_mod
            importlib.reload(prom_mod)
            # Noop objects should not raise on use
            prom_mod.agent_requests_total.labels(agent_type="research", status="ok").inc()
            prom_mod.agent_latency_seconds.labels(agent_type="research").observe(1.5)
            prom_mod.llm_tokens_total.labels(model="gpt-4o-mini", token_type="input").inc(100)
            prom_mod.active_sessions.set(3)
            # Reload back to normal state
            importlib.reload(prom_mod)

    def test_setup_metrics_called_with_app(self) -> None:
        from app.monitoring.prometheus import setup_metrics

        mock_app = MagicMock()
        # Should not raise — either instruments or logs a warning
        try:
            setup_metrics(mock_app)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"setup_metrics raised unexpectedly: {exc}")
