"""Tests for the analytics metrics collector and API."""
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.analytics.metrics_collector import AgentStats, MetricsCollector, get_metrics_collector


# ---------------------------------------------------------------------------
# MetricsCollector unit tests
# ---------------------------------------------------------------------------

class TestAgentStats:
    def test_defaults(self) -> None:
        s = AgentStats()
        assert s.requests_total == 0
        assert s.errors_total == 0
        assert s.input_tokens_total == 0
        assert s.output_tokens_total == 0
        assert len(s.latencies_ms) == 0

    def test_latency_deque_max_len(self) -> None:
        s = AgentStats()
        for i in range(1200):
            s.latencies_ms.append(float(i))
        assert len(s.latencies_ms) == 1000


class TestMetricsCollector:
    def setup_method(self) -> None:
        self.mc = MetricsCollector()

    def test_track_request_increments_counter(self) -> None:
        with self.mc.track_request("research"):
            pass
        snap = self.mc.snapshot()
        assert snap["agents"]["research"]["requests_total"] == 1
        assert snap["agents"]["research"]["errors_total"] == 0

    def test_track_request_increments_error_on_exception(self) -> None:
        with pytest.raises(ValueError):
            with self.mc.track_request("research"):
                raise ValueError("boom")
        snap = self.mc.snapshot()
        assert snap["agents"]["research"]["errors_total"] == 1

    def test_track_request_records_latency(self) -> None:
        with self.mc.track_request("content"):
            time.sleep(0.01)
        snap = self.mc.snapshot()
        assert snap["agents"]["content"]["latency_p50_ms"] > 0

    def test_record_tokens(self) -> None:
        self.mc.record_tokens("research", input_tokens=100, output_tokens=200)
        snap = self.mc.snapshot()
        assert snap["agents"]["research"]["input_tokens_total"] == 100
        assert snap["agents"]["research"]["output_tokens_total"] == 200

    def test_record_tokens_accumulates(self) -> None:
        self.mc.record_tokens("research", input_tokens=50, output_tokens=100)
        self.mc.record_tokens("research", input_tokens=50, output_tokens=100)
        snap = self.mc.snapshot()
        assert snap["agents"]["research"]["input_tokens_total"] == 100
        assert snap["agents"]["research"]["output_tokens_total"] == 200

    def test_set_active_sessions(self) -> None:
        self.mc.set_active_sessions(5)
        assert self.mc.snapshot()["active_sessions"] == 5

    def test_increment_active_sessions(self) -> None:
        self.mc.increment_active_sessions(3)
        assert self.mc.snapshot()["active_sessions"] == 3
        self.mc.increment_active_sessions(-1)
        assert self.mc.snapshot()["active_sessions"] == 2

    def test_increment_active_sessions_floor_zero(self) -> None:
        self.mc.increment_active_sessions(-99)
        assert self.mc.snapshot()["active_sessions"] == 0

    def test_snapshot_totals(self) -> None:
        with self.mc.track_request("research"):
            pass
        with self.mc.track_request("content"):
            pass
        with pytest.raises(RuntimeError):
            with self.mc.track_request("research"):
                raise RuntimeError("fail")
        snap = self.mc.snapshot()
        assert snap["total_requests"] == 3
        assert snap["total_errors"] == 1

    def test_snapshot_uptime(self) -> None:
        snap = self.mc.snapshot()
        assert snap["uptime_seconds"] >= 0

    def test_percentile_empty(self) -> None:
        snap = self.mc.snapshot()
        # No requests yet — latency stats should be zero
        assert "agents" in snap

    def test_error_rate_calculation(self) -> None:
        with self.mc.track_request("research"):
            pass
        with pytest.raises(ValueError):
            with self.mc.track_request("research"):
                raise ValueError("err")
        snap = self.mc.snapshot()
        assert snap["agents"]["research"]["error_rate"] == pytest.approx(0.5, abs=0.001)

    def test_multiple_agents_isolated(self) -> None:
        with self.mc.track_request("research"):
            pass
        self.mc.record_tokens("research", input_tokens=10, output_tokens=20)
        with self.mc.track_request("content"):
            pass
        self.mc.record_tokens("content", input_tokens=5, output_tokens=15)
        snap = self.mc.snapshot()
        assert snap["agents"]["research"]["input_tokens_total"] == 10
        assert snap["agents"]["content"]["input_tokens_total"] == 5

    def test_percentile_values(self) -> None:
        for i in range(100):
            self.mc._stats("research").latencies_ms.append(float(i + 1))
        snap = self.mc.snapshot()
        a = snap["agents"]["research"]
        assert a["latency_p50_ms"] > 0
        assert a["latency_p95_ms"] >= a["latency_p50_ms"]
        assert a["latency_p99_ms"] >= a["latency_p95_ms"]


class TestGetMetricsCollectorSingleton:
    def test_returns_same_instance(self) -> None:
        a = get_metrics_collector()
        b = get_metrics_collector()
        assert a is b


# ---------------------------------------------------------------------------
# Analytics API tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    from app.server import app
    return TestClient(app, raise_server_exceptions=True)


class TestAnalyticsEndpoint:
    def test_metrics_returns_200(self, client: TestClient) -> None:
        r = client.get("/api/analytics/metrics")
        assert r.status_code == 200

    def test_metrics_response_shape(self, client: TestClient) -> None:
        r = client.get("/api/analytics/metrics")
        data = r.json()
        assert "uptime_seconds" in data
        assert "active_sessions" in data
        assert "total_requests" in data
        assert "total_errors" in data
        assert "agents" in data

    def test_analytics_html_route(self, client: TestClient) -> None:
        r = client.get("/analytics")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")
        assert b"Analytics" in r.content
