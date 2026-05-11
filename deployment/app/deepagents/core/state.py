"""Deep Agent State Management.

This module defines the state structure used by Deep Agents for tracking
conversations, todos, files, and subagent executions.
"""

from datetime import datetime
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from app.deepagents.core.types import FileEntry, SubAgentResult, Todo


def merge_todos(existing: list[Todo], new: list[Todo]) -> list[Todo]:
    """Merge todo lists, updating existing todos by ID."""
    todo_map = {t.id: t for t in existing}
    for todo in new:
        todo_map[todo.id] = todo
    return list(todo_map.values())


def merge_files(existing: dict[str, FileEntry], new: dict[str, FileEntry]) -> dict[str, FileEntry]:
    """Merge file dictionaries."""
    result = existing.copy()
    result.update(new)
    return result


class DeepAgentState(BaseModel):
    """State for Deep Agent execution.

    This state tracks:
    - Conversation messages
    - Planning todos for task decomposition
    - Virtual file system for context management
    - Subagent execution results
    - Session metadata
    """

    # Conversation
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # Planning (TodoList middleware)
    todos: list[Todo] = Field(default_factory=list)
    current_todo_id: str | None = None

    # File System (Filesystem middleware)
    files: dict[str, FileEntry] = Field(default_factory=dict)
    working_directory: str = "/"

    # Subagent Results (SubAgent middleware)
    subagent_results: list[SubAgentResult] = Field(default_factory=list)
    active_subagents: list[str] = Field(default_factory=list)

    # Session metadata
    session_id: str | None = None
    user_id: str | None = None
    started_at: datetime = Field(default_factory=datetime.now)
    last_activity: datetime = Field(default_factory=datetime.now)

    # Context for IT Operations
    current_incident: str | None = None
    current_change: str | None = None
    current_problem: str | None = None
    affected_cis: list[str] = Field(default_factory=list)

    # Execution metadata
    iteration_count: int = 0
    total_tool_calls: int = 0

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def get_pending_todos(self) -> list[Todo]:
        """Get all pending todos."""
        from app.deepagents.core.types import TodoStatus

        return [t for t in self.todos if t.status == TodoStatus.PENDING]

    def get_in_progress_todos(self) -> list[Todo]:
        """Get all in-progress todos."""
        from app.deepagents.core.types import TodoStatus

        return [t for t in self.todos if t.status == TodoStatus.IN_PROGRESS]

    def get_completed_todos(self) -> list[Todo]:
        """Get all completed todos."""
        from app.deepagents.core.types import TodoStatus

        return [t for t in self.todos if t.status == TodoStatus.COMPLETED]

    def get_todo_summary(self) -> str:
        """Get a summary of todo status."""
        pending = len(self.get_pending_todos())
        in_progress = len(self.get_in_progress_todos())
        completed = len(self.get_completed_todos())
        total = len(self.todos)

        if total == 0:
            return "No tasks planned yet."

        return f"Tasks: {completed}/{total} completed, {in_progress} in progress, {pending} pending"

    def get_file_list(self) -> list[str]:
        """Get list of all file paths."""
        return list(self.files.keys())

    def read_file(self, path: str) -> str | None:
        """Read a file from the virtual file system."""
        entry = self.files.get(path)
        return entry.content if entry else None

    def get_context_summary(self) -> str:
        """Get a summary of current context."""
        parts = []

        if self.current_incident:
            parts.append(f"Incident: {self.current_incident}")
        if self.current_change:
            parts.append(f"Change: {self.current_change}")
        if self.current_problem:
            parts.append(f"Problem: {self.current_problem}")
        if self.affected_cis:
            parts.append(f"Affected CIs: {', '.join(self.affected_cis)}")

        return " | ".join(parts) if parts else "No active IT context"
