"""Tests for the response caching layer.

Covers:
- Cache disabled by default (CACHE_ENABLED not set → get/set are no-ops)
- Cache hit returns stored response when enabled
- Whitespace normalisation produces identical keys
- TTL expiry: stale entries are treated as misses
- Size-cap eviction: oldest entry removed when at MAX_CACHE_SIZE
- Cache stats and clear endpoints
"""

import importlib
import os
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Unit tests for AgentResponseCache
# ---------------------------------------------------------------------------


def _fresh_cache():
    """Return a brand-new AgentResponseCache instance (not the singleton)."""
    from app.cache.response_cache import AgentResponseCache
    return AgentResponseCache()


def test_cache_disabled_by_default(monkeypatch):
    """With CACHE_ENABLED unset, get and set must be no-ops."""
    monkeypatch.delenv("CACHE_ENABLED", raising=False)
    cache = _fresh_cache()
    cache.set("research", "test query", "some response")
    assert cache.get("research", "test query") is None


def test_cache_enabled_stores_and_retrieves(monkeypatch):
    """With CACHE_ENABLED=true, a stored response is returned on the next get."""
    monkeypatch.setenv("CACHE_ENABLED", "true")
    cache = _fresh_cache()
    cache.set("research", "AI trends", "Here are the AI trends…")
    result = cache.get("research", "AI trends")
    assert result == "Here are the AI trends…"


def test_cache_hit_after_enable(monkeypatch):
    """Cache returns None for an unseen key even when enabled."""
    monkeypatch.setenv("CACHE_ENABLED", "true")
    cache = _fresh_cache()
    assert cache.get("research", "never asked before") is None


def test_cache_normalizes_whitespace():
    """Keys for queries with different whitespace must be identical."""
    cache = _fresh_cache()
    key1 = cache._key("research", "AI  trends")
    key2 = cache._key("research", "AI trends")
    assert key1 == key2


def test_cache_normalizes_leading_trailing_whitespace():
    """Leading and trailing whitespace is stripped before hashing."""
    cache = _fresh_cache()
    key1 = cache._key("research", "  AI trends  ")
    key2 = cache._key("research", "AI trends")
    assert key1 == key2


def test_cache_normalizes_case():
    """Query is lower-cased before hashing."""
    cache = _fresh_cache()
    key1 = cache._key("research", "AI Trends")
    key2 = cache._key("research", "ai trends")
    assert key1 == key2


def test_cache_key_differs_by_agent_type():
    """Different agent types produce different keys for the same query."""
    cache = _fresh_cache()
    key1 = cache._key("research", "hello")
    key2 = cache._key("content", "hello")
    assert key1 != key2


def test_cache_clear(monkeypatch):
    """After clear(), size drops to zero and stored entries are gone."""
    monkeypatch.setenv("CACHE_ENABLED", "true")
    cache = _fresh_cache()
    cache.set("research", "query one", "response one")
    cache.set("research", "query two", "response two")
    assert cache.size() == 2
    cache.clear()
    assert cache.size() == 0
    assert cache.get("research", "query one") is None


def test_cache_size_disabled(monkeypatch):
    """Size reflects raw store count even when cache is disabled (clear is always active)."""
    monkeypatch.delenv("CACHE_ENABLED", raising=False)
    cache = _fresh_cache()
    # set is a no-op when disabled
    cache.set("research", "q", "r")
    assert cache.size() == 0


def test_get_cache_singleton():
    """get_cache() returns the same instance on repeated calls."""
    from app.cache.response_cache import get_cache
    c1 = get_cache()
    c2 = get_cache()
    assert c1 is c2


def test_cache_entry_expires_after_ttl(monkeypatch):
    """An entry whose TTL has elapsed is treated as a cache miss and removed."""
    monkeypatch.setenv("CACHE_ENABLED", "true")
    import app.cache.response_cache as rc

    cache = rc.AgentResponseCache()

    # Patch time.time so the entry was stored 2 seconds ago and TTL is 1 second.
    base_time = 1_000_000.0
    with patch("app.cache.response_cache.time") as mock_time:
        # At store time the clock reads base_time.
        mock_time.time.return_value = base_time
        cache.set("research", "expiry test", "some response")
        assert cache.size() == 1

        # Advance clock past TTL (TTL is read from module constant, default 3600 s,
        # but we set the expiry directly via the mocked clock).
        mock_time.time.return_value = base_time + rc.CACHE_TTL_SECONDS + 1
        result = cache.get("research", "expiry test")

    assert result is None, "Expired entry should return None"
    assert cache.size() == 0, "Expired entry should be removed from the store"


def test_cache_evicts_oldest_when_full(monkeypatch):
    """When the store is at MAX_CACHE_SIZE the oldest entry is evicted (FIFO)."""
    monkeypatch.setenv("CACHE_ENABLED", "true")
    import app.cache.response_cache as rc

    # Override MAX_CACHE_SIZE to 2 for this test.
    original_max = rc.MAX_CACHE_SIZE
    rc.MAX_CACHE_SIZE = 2
    try:
        cache = rc.AgentResponseCache()
        cache.set("research", "query one", "response one")
        cache.set("research", "query two", "response two")
        assert cache.size() == 2

        # Adding a third entry should evict "query one" (the oldest).
        cache.set("research", "query three", "response three")

        assert cache.size() == 2, "Size must not exceed MAX_CACHE_SIZE"
        assert cache.get("research", "query one") is None, "Oldest entry should be evicted"
        assert cache.get("research", "query two") == "response two"
        assert cache.get("research", "query three") == "response three"
    finally:
        rc.MAX_CACHE_SIZE = original_max


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """TestClient for the FastAPI application."""
    from app.server import app
    return TestClient(app)


def test_cache_stats_endpoint(client):
    """GET /api/cache/stats returns enabled and size fields."""
    r = client.get("/api/cache/stats")
    assert r.status_code == 200
    data = r.json()
    assert "enabled" in data
    assert "size" in data
    assert isinstance(data["enabled"], bool)
    assert isinstance(data["size"], int)


def test_cache_stats_disabled_by_default(monkeypatch, client):
    """Cache stats show enabled=false when CACHE_ENABLED is not set."""
    monkeypatch.delenv("CACHE_ENABLED", raising=False)
    r = client.get("/api/cache/stats")
    assert r.status_code == 200
    assert r.json()["enabled"] is False


def test_cache_clear_endpoint(client):
    """DELETE /api/cache/clear returns cleared=true."""
    r = client.delete("/api/cache/clear")
    assert r.status_code == 200
    assert r.json() == {"cleared": True}


def test_cache_clear_empties_store(monkeypatch):
    """After calling clear via the singleton, size is zero."""
    monkeypatch.setenv("CACHE_ENABLED", "true")
    from app.cache.response_cache import get_cache
    cache = get_cache()
    cache.set("research", "persistent query", "response")
    # confirm it was stored
    assert cache.size() >= 1
    # clear via endpoint logic (simulate what the endpoint does)
    cache.clear()
    assert cache.size() == 0


def test_research_invoke_returns_cached_field_false_when_no_api_key(client):
    """Research invoke with no API key returns 503, not a cache response."""
    r = client.post(
        "/api/enterprise/research/invoke",
        json={"query": "What is LangChain?"},
    )
    # Without API keys enterprise agents are unavailable
    assert r.status_code == 503
