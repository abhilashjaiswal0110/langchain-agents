"""Specialized Subagents for IT Managed Services.

Each subagent provides focused expertise for a specific IT domain.
"""

from app.deepagents.subagents.definitions import (
    INCIDENT_AGENT,
    CHANGE_AGENT,
    PROBLEM_AGENT,
    ASSET_AGENT,
    SLA_AGENT,
    KNOWLEDGE_AGENT,
    get_all_subagents,
    get_subagent_tools,
)

__all__ = [
    "INCIDENT_AGENT",
    "CHANGE_AGENT",
    "PROBLEM_AGENT",
    "ASSET_AGENT",
    "SLA_AGENT",
    "KNOWLEDGE_AGENT",
    "get_all_subagents",
    "get_subagent_tools",
]
