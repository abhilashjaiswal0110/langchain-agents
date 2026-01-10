"""Type definitions for Deep Agents.

This module defines the core data structures used throughout the Deep Agents system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field


class TodoStatus(str, Enum):
    """Status of a todo item."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class Todo(BaseModel):
    """A single todo item for task tracking.

    Deep agents use todos to break down complex tasks into discrete steps,
    track progress, and adapt plans as new information emerges.
    """

    id: str = Field(description="Unique identifier for the todo")
    content: str = Field(description="Description of the task")
    status: TodoStatus = Field(default=TodoStatus.PENDING, description="Current status")
    priority: int = Field(default=0, description="Priority (0=normal, 1=high, 2=critical)")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = None
    parent_id: str | None = Field(default=None, description="Parent todo for subtasks")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def mark_in_progress(self) -> "Todo":
        """Mark todo as in progress."""
        self.status = TodoStatus.IN_PROGRESS
        self.updated_at = datetime.now()
        return self

    def mark_completed(self) -> "Todo":
        """Mark todo as completed."""
        self.status = TodoStatus.COMPLETED
        self.updated_at = datetime.now()
        self.completed_at = datetime.now()
        return self

    def mark_blocked(self, reason: str | None = None) -> "Todo":
        """Mark todo as blocked."""
        self.status = TodoStatus.BLOCKED
        self.updated_at = datetime.now()
        if reason:
            self.metadata["blocked_reason"] = reason
        return self


class FileEntry(BaseModel):
    """A file in the Deep Agent's virtual file system.

    Used for context management - storing investigation notes, reports,
    and intermediate results.
    """

    path: str = Field(description="File path relative to agent workspace")
    content: str = Field(description="File content")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    file_type: str = Field(default="text", description="Type: text, markdown, json, etc.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubAgentDefinition(BaseModel):
    """Definition of a subagent that can be spawned by the main agent.

    Subagents provide context isolation - their work doesn't clutter the main
    agent's context window while still allowing deep investigation of subtasks.
    """

    name: str = Field(description="Unique identifier for the subagent")
    description: str = Field(description="What this subagent does")
    system_prompt: str = Field(description="Instructions for the subagent")
    tools: list[str] = Field(default_factory=list, description="Tool names available to subagent")
    model: str | None = Field(default=None, description="Override model for this subagent")
    max_iterations: int = Field(default=10, description="Max tool-calling iterations")
    interrupt_on: dict[str, bool] = Field(
        default_factory=dict,
        description="Tools that require human approval",
    )


class DeepAgentConfig(BaseModel):
    """Configuration for a Deep Agent.

    Controls the agent's behavior, capabilities, and resource limits.
    """

    name: str = Field(default="deep_agent", description="Agent name")
    model: str = Field(default="gpt-4o-mini", description="LLM model to use")
    model_provider: Literal["openai", "anthropic", "auto"] = Field(
        default="auto",
        description="LLM provider",
    )
    temperature: float = Field(default=0, description="LLM temperature")
    system_prompt: str | None = Field(default=None, description="Custom system prompt")

    # Planning configuration
    max_todos: int = Field(default=20, description="Maximum active todos")
    auto_planning: bool = Field(default=True, description="Auto-create planning todos")

    # File system configuration
    workspace_path: str = Field(default="./workspace", description="Workspace directory")
    max_file_size: int = Field(default=100000, description="Max file size in chars")
    persistent_storage: bool = Field(default=True, description="Persist files to disk")

    # Subagent configuration
    max_subagents: int = Field(default=5, description="Max concurrent subagents")
    subagent_timeout: int = Field(default=300, description="Subagent timeout in seconds")

    # Memory configuration
    enable_long_term_memory: bool = Field(default=True, description="Enable LangGraph Store")
    memory_namespace: str = Field(default="deep_agent", description="Memory namespace")

    # Authentication
    require_auth: bool = Field(default=False, description="Require authentication")


class SubAgentResult(BaseModel):
    """Result from a subagent execution."""

    subagent_name: str
    task_description: str
    result: str
    success: bool
    execution_time: float
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class ContextFile(BaseModel):
    """A context file stored in the agent's workspace."""

    path: str
    content: str
    file_type: str = "text"
    size: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    modified_at: datetime = Field(default_factory=datetime.now)


class AgentEvent(BaseModel):
    """Event emitted by the Deep Agent during execution."""

    event_type: Literal[
        "todo_created",
        "todo_updated",
        "file_created",
        "file_updated",
        "subagent_started",
        "subagent_completed",
        "tool_called",
        "message_received",
    ]
    timestamp: datetime = Field(default_factory=datetime.now)
    data: dict[str, Any] = Field(default_factory=dict)
