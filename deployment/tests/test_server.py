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


def test_readiness_check_without_chains():
    """Test readiness check fails without chains loaded."""
    import os
    os.environ.pop("OPENAI_API_KEY", None)

    from app.server import app
    client = TestClient(app)

    response = client.get("/ready")
    # Should return 503 when chains are not loaded
    assert response.status_code == 503


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
