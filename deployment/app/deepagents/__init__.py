"""Deep Agents Module for IT Managed Services.

This module implements a Deep Agents system inspired by LangChain's deepagents library,
providing planning capabilities, file system context management, and subagent spawning
for complex IT operations workflows.

Key Components:
- ITOperationsDeepAgent: Main coordinator for IT managed services
- SalesIntelligenceDeepAgent: Sales & Pre-Sales Intelligence agent
- RecruitmentDeepAgent: Recruitment & Talent Acquisition agent
- Subagents: Specialized agents for incident, change, problem, asset, SLA, and knowledge
- Planning: Task decomposition and progress tracking (TodoList)
- Context: File system for storing investigation notes and reports
"""

from app.deepagents.core.deep_agent import DeepAgent, create_deep_agent
from app.deepagents.core.state import DeepAgentState
from app.deepagents.core.types import (
    DeepAgentConfig,
    SubAgentDefinition,
    Todo,
    TodoStatus,
)
from app.deepagents.it_operations_agent import (
    ITOperationsDeepAgent,
    create_it_operations_agent,
)
from app.deepagents.recruitment_agent import (
    RecruitmentDeepAgent,
    create_recruitment_agent,
)

__all__ = [
    # Types
    "Todo",
    "TodoStatus",
    "DeepAgentConfig",
    "SubAgentDefinition",
    # State
    "DeepAgentState",
    # Core
    "create_deep_agent",
    "DeepAgent",
    # IT Operations
    "ITOperationsDeepAgent",
    "create_it_operations_agent",
    # Recruitment
    "RecruitmentDeepAgent",
    "create_recruitment_agent",
]

__version__ = "1.0.0"
