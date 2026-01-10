"""Deep Agents Module for IT Managed Services.

This module implements a Deep Agents system inspired by LangChain's deepagents library,
providing planning capabilities, file system context management, and subagent spawning
for complex IT operations workflows.

Key Components:
- ITOperationsDeepAgent: Main coordinator for IT managed services
- Subagents: Specialized agents for incident, change, problem, asset, SLA, and knowledge
- Planning: Task decomposition and progress tracking (TodoList)
- Context: File system for storing investigation notes and reports
"""

from app.deepagents.core.types import (
    Todo,
    TodoStatus,
    DeepAgentConfig,
    SubAgentDefinition,
)
from app.deepagents.core.state import DeepAgentState
from app.deepagents.core.deep_agent import create_deep_agent, DeepAgent
from app.deepagents.it_operations_agent import (
    ITOperationsDeepAgent,
    create_it_operations_agent,
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
]

__version__ = "1.0.0"
