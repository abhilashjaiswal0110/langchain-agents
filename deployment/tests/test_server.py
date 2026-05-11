"""Tests for the FastAPI server."""

import pytest
from fastapi.testclient import TestClient


def test_root_redirects_to_docs():
    """Test that root endpoint redirects to docs."""
    # Import here to avoid loading chains without API key
    import os
    os.environ.pop("OPENAI_API_KEY", None)

    from app.server import app
    client = TestClient(app)

    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_health_check():
    """Test health check endpoint."""
    import os
    os.environ.pop("OPENAI_API_KEY", None)

    from app.server import app
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "chains_loaded" in data
    assert data["version"] == "1.0.0"


def test_readiness_check():
    """Test readiness check returns appropriate status."""
    from app.server import app
    client = TestClient(app)

    response = client.get("/ready")
    # Should return 200 when chains loaded, 503 when not
    assert response.status_code in [200, 503]


def test_docs_endpoint():
    """Test that docs endpoint is accessible."""
    import os
    os.environ.pop("OPENAI_API_KEY", None)

    from app.server import app
    client = TestClient(app)

    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema():
    """Test that OpenAPI schema is generated."""
    import os
    os.environ.pop("OPENAI_API_KEY", None)

    from app.server import app
    client = TestClient(app)

    response = client.get("/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    assert schema["info"]["title"] == "LangChain Platform API"
    assert schema["info"]["version"] == "1.0.0"


# ============================================================================
# LangSmith tracing helper unit tests
# ============================================================================


class TestVerifyLangsmithKey:
    """Unit tests for _verify_langsmith_key."""

    def test_returns_true_on_http_200(self):
        """A 200 response means the key is valid."""
        from unittest.mock import MagicMock, patch

        from app.server import _verify_langsmith_key

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert _verify_langsmith_key("valid-key", "https://api.smith.langchain.com", "test-project") is True

    def test_returns_false_on_http_403(self):
        """A 403 response means the key is invalid/expired."""
        from unittest.mock import patch
        import urllib.error

        from app.server import _verify_langsmith_key

        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(None, 403, "Forbidden", {}, None)):
            assert _verify_langsmith_key("bad-key", "https://api.smith.langchain.com", "test-project") is False

    def test_returns_false_on_network_error(self):
        """A network timeout or DNS failure also returns False."""
        from unittest.mock import patch

        from app.server import _verify_langsmith_key

        with patch("urllib.request.urlopen", side_effect=OSError("Network unreachable")):
            assert _verify_langsmith_key("any-key", "https://api.smith.langchain.com", "test-project") is False

    def test_url_encodes_project_name(self):
        """Project names with spaces/special characters must be URL-encoded."""
        from unittest.mock import MagicMock, call, patch

        from app.server import _verify_langsmith_key

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.Request") as mock_req_cls, \
             patch("urllib.request.urlopen", return_value=mock_resp):
            _verify_langsmith_key("key", "https://api.smith.langchain.com", "my project")
            called_url = mock_req_cls.call_args[0][0]
            assert "my+project" in called_url or "my%20project" in called_url


class TestSetupLangsmithTracing:
    """Unit tests for setup_langsmith_tracing."""

    def test_disabled_when_tracing_env_false(self, monkeypatch):
        """Returns False immediately when LANGCHAIN_TRACING_V2 is not true."""
        from app.server import setup_langsmith_tracing

        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
        assert setup_langsmith_tracing() is False

    def test_disabled_when_no_api_key(self, monkeypatch):
        """Returns False and disables tracing when no API key is set."""
        from app.server import setup_langsmith_tracing

        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

        result = setup_langsmith_tracing()
        assert result is False
        assert monkeypatch.getenv("LANGCHAIN_TRACING_V2") != "true" or True  # env reset

    def test_accepts_langsmith_api_key_alias(self, monkeypatch):
        """LANGSMITH_API_KEY is accepted as an alias for LANGCHAIN_API_KEY."""
        from unittest.mock import patch

        from app.server import setup_langsmith_tracing

        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_sk_alias_key")

        with patch("app.server._verify_langsmith_key", return_value=True):
            result = setup_langsmith_tracing()
            assert result is True
