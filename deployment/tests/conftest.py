"""Pytest configuration and shared fixtures."""

import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def clear_env_vars():
    """Clear API keys before each test to ensure clean state."""
    env_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "LANGCHAIN_API_KEY",
        "TAVILY_API_KEY",
    ]
    original_values = {}
    for var in env_vars:
        original_values[var] = os.environ.pop(var, None)

    yield

    # Restore original values
    for var, value in original_values.items():
        if value is not None:
            os.environ[var] = value


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    from app.server import app
    return TestClient(app)


@pytest.fixture
def mock_openai_key():
    """Set a mock OpenAI API key for testing."""
    os.environ["OPENAI_API_KEY"] = "sk-test-key-for-testing"
    yield
    os.environ.pop("OPENAI_API_KEY", None)
