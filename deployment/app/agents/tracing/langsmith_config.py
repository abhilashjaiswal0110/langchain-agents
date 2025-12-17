"""LangSmith tracing configuration for enterprise IT agents.

This module provides centralized LangSmith configuration
for tracing and evaluation of all agents.

Following Enterprise Development Standards:
- Security Architect: No hardcoded API keys
- Data Architect: Structured tracing metadata
- Software Engineer: Clean configuration API
"""

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class TracingConfig:
    """LangSmith tracing configuration."""

    enabled: bool = False
    api_key: str | None = None
    project: str = "enterprise-it-agents"
    endpoint: str = "https://api.smith.langchain.com"


_tracing_config: TracingConfig | None = None


def setup_tracing(
    project: str = "enterprise-it-agents",
    endpoint: str | None = None,
) -> bool:
    """Configure LangSmith tracing for all agents.

    Args:
        project: LangSmith project name for grouping traces
        endpoint: Custom LangSmith endpoint (optional)

    Returns:
        True if tracing is enabled, False otherwise
    """
    global _tracing_config

    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

    if tracing_enabled and api_key:
        # Set environment variables
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint or "https://api.smith.langchain.com"

        _tracing_config = TracingConfig(
            enabled=True,
            api_key=api_key[:10] + "...",  # Masked for security
            project=project,
            endpoint=os.environ["LANGCHAIN_ENDPOINT"],
        )

        print(f"LangSmith tracing enabled for project: {project}")
        return True

    if tracing_enabled and not api_key:
        print("Warning: LANGCHAIN_TRACING_V2=true but no API key found")
        print("Set LANGCHAIN_API_KEY or LANGSMITH_API_KEY")

    _tracing_config = TracingConfig(enabled=False)
    return False


def get_tracing_status() -> dict[str, Any]:
    """Get current tracing configuration status.

    Returns:
        Dictionary with tracing status information
    """
    if _tracing_config is None:
        setup_tracing()

    return {
        "enabled": _tracing_config.enabled if _tracing_config else False,
        "project": _tracing_config.project if _tracing_config else None,
        "endpoint": _tracing_config.endpoint if _tracing_config else None,
    }


def get_run_metadata(
    agent_name: str,
    session_id: str | None = None,
    user_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Generate metadata for LangSmith runs.

    Args:
        agent_name: Name of the agent
        session_id: Optional session identifier
        user_id: Optional user identifier
        **extra: Additional metadata

    Returns:
        Metadata dictionary for tracing
    """
    metadata = {
        "agent": agent_name,
        "framework": "langgraph",
        "version": "1.0.0",
    }

    if session_id:
        metadata["session_id"] = session_id
    if user_id:
        metadata["user_id"] = user_id

    metadata.update(extra)
    return metadata
