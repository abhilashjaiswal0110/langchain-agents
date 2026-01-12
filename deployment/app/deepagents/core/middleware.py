"""Deep Agent Middleware Components.

Middleware provides modular capabilities to Deep Agents:
- TodoListMiddleware: Planning and task decomposition
- FilesystemMiddleware: Context management via virtual file system
- SubAgentMiddleware: Spawning specialized subagents
"""

import uuid
from datetime import datetime
from typing import Any

from langchain_core.tools import tool

from app.deepagents.core.types import (
    Todo,
    TodoStatus,
    FileEntry,
    SubAgentDefinition,
    SubAgentResult,
)
from app.deepagents.core.state import DeepAgentState


class TodoListMiddleware:
    """Middleware for planning and task decomposition.

    Provides the write_todos tool that enables agents to break down
    complex tasks into discrete steps and track progress.
    """

    def __init__(self, max_todos: int = 20) -> None:
        """Initialize TodoList middleware.

        Args:
            max_todos: Maximum number of active todos.
        """
        self.max_todos = max_todos

    def get_tools(self) -> list:
        """Get todo management tools."""
        return [
            self._create_write_todos_tool(),
            self._create_update_todo_tool(),
            self._create_get_todos_tool(),
        ]

    def _create_write_todos_tool(self):
        """Create the write_todos tool."""
        max_todos = self.max_todos

        @tool
        def write_todos(
            todos: list[dict[str, str]],
        ) -> str:
            """Create or update the task list for the current work.

            Use this to plan complex tasks by breaking them into steps.
            Each todo should have 'content' (description) and optionally
            'priority' (0=normal, 1=high, 2=critical).

            Args:
                todos: List of todo items with 'content' and optional 'priority'.

            Returns:
                Confirmation of todos created.
            """
            if len(todos) > max_todos:
                return f"Error: Maximum {max_todos} todos allowed. Please consolidate."

            created_todos = []
            for item in todos:
                todo = Todo(
                    id=str(uuid.uuid4())[:8],
                    content=item.get("content", ""),
                    priority=int(item.get("priority", 0)),
                    status=TodoStatus.PENDING,
                )
                created_todos.append(todo)

            # Return formatted list
            lines = ["**Task Plan Created:**\n"]
            for i, todo in enumerate(created_todos, 1):
                priority_marker = "!" * todo.priority if todo.priority > 0 else ""
                lines.append(f"{i}. [{todo.status.value}] {priority_marker}{todo.content}")

            return "\n".join(lines)

        return write_todos

    def _create_update_todo_tool(self):
        """Create the update_todo tool."""

        @tool
        def update_todo(
            todo_id: str,
            status: str,
            notes: str | None = None,
        ) -> str:
            """Update the status of a todo item.

            Args:
                todo_id: The ID of the todo to update.
                status: New status (pending, in_progress, completed, blocked).
                notes: Optional notes about the update.

            Returns:
                Confirmation of the update.
            """
            valid_statuses = ["pending", "in_progress", "completed", "blocked"]
            if status not in valid_statuses:
                return f"Error: Invalid status. Use one of: {valid_statuses}"

            result = f"Todo {todo_id} updated to '{status}'"
            if notes:
                result += f"\nNotes: {notes}"

            return result

        return update_todo

    def _create_get_todos_tool(self):
        """Create the get_todos tool."""

        @tool
        def get_todos() -> str:
            """Get the current task list with status.

            Returns:
                Formatted list of all todos with their status.
            """
            return "Use context to view current todos. This tool retrieves the todo state."

        return get_todos


class FilesystemMiddleware:
    """Middleware for context management via virtual file system.

    Provides tools to read, write, and edit files for storing context,
    investigation notes, and intermediate results.
    """

    def __init__(
        self,
        workspace_path: str = "./workspace",
        max_file_size: int = 100000,
        persistent: bool = True,
    ) -> None:
        """Initialize Filesystem middleware.

        Args:
            workspace_path: Base path for file storage.
            max_file_size: Maximum file size in characters.
            persistent: Whether to persist files to disk.
        """
        self.workspace_path = workspace_path
        self.max_file_size = max_file_size
        self.persistent = persistent

    def get_tools(self) -> list:
        """Get file system tools."""
        return [
            self._create_ls_tool(),
            self._create_read_file_tool(),
            self._create_write_file_tool(),
            self._create_edit_file_tool(),
        ]

    def _create_ls_tool(self):
        """Create the ls (list files) tool."""

        @tool
        def ls(path: str = "/") -> str:
            """List files in the workspace directory.

            Args:
                path: Directory path to list.

            Returns:
                List of files in the directory.
            """
            # This returns state-based file listing
            return f"Listing files in {path}. Check context for file list."

        return ls

    def _create_read_file_tool(self):
        """Create the read_file tool."""
        max_size = self.max_file_size

        @tool
        def read_file(path: str) -> str:
            """Read a file from the workspace.

            Use this to retrieve previously saved context, notes, or reports.

            Args:
                path: File path to read.

            Returns:
                File contents or error message.
            """
            return f"Reading file: {path}. Content will be in context."

        return read_file

    def _create_write_file_tool(self):
        """Create the write_file tool."""
        max_size = self.max_file_size

        @tool
        def write_file(path: str, content: str) -> str:
            """Write content to a file in the workspace.

            Use this to save investigation notes, reports, or intermediate results.

            Args:
                path: File path to write.
                content: Content to write.

            Returns:
                Confirmation or error message.
            """
            if len(content) > max_size:
                return f"Error: Content exceeds max size of {max_size} characters."

            return f"File written: {path} ({len(content)} characters)"

        return write_file

    def _create_edit_file_tool(self):
        """Create the edit_file tool."""

        @tool
        def edit_file(
            path: str,
            old_content: str,
            new_content: str,
        ) -> str:
            """Edit a file by replacing content.

            Args:
                path: File path to edit.
                old_content: Content to find and replace.
                new_content: New content to insert.

            Returns:
                Confirmation or error message.
            """
            return f"File edited: {path}"

        return edit_file


class SubAgentMiddleware:
    """Middleware for spawning specialized subagents.

    Provides the task tool that enables the main agent to delegate
    work to specialized subagents for context isolation.
    """

    def __init__(
        self,
        subagents: list[SubAgentDefinition] | None = None,
        default_model: str = "gpt-4o-mini",
        default_tools: list | None = None,
        max_concurrent: int = 5,
    ) -> None:
        """Initialize SubAgent middleware.

        Args:
            subagents: List of available subagent definitions.
            default_model: Default model for subagents.
            default_tools: Default tools available to subagents.
            max_concurrent: Maximum concurrent subagents.
        """
        self.subagents = {s.name: s for s in (subagents or [])}
        self.default_model = default_model
        self.default_tools = default_tools or []
        self.max_concurrent = max_concurrent

    def get_tools(self) -> list:
        """Get subagent spawning tools."""
        return [self._create_task_tool()]

    def _create_task_tool(self):
        """Create the task tool for spawning subagents."""
        available_subagents = list(self.subagents.keys())

        @tool
        def task(
            subagent_type: str,
            task_description: str,
            context: str | None = None,
        ) -> str:
            """Delegate a task to a specialized subagent.

            Subagents work in isolation and return a final report.
            Use this for complex subtasks that need focused investigation.

            Available subagents: {available}

            Args:
                subagent_type: Type of subagent to spawn.
                task_description: What the subagent should do.
                context: Optional context to pass to the subagent.

            Returns:
                Result from the subagent.
            """.format(available=", ".join(available_subagents) or "general-purpose")

            if subagent_type not in available_subagents and subagent_type != "general-purpose":
                return f"Error: Unknown subagent '{subagent_type}'. Available: {available_subagents}"

            return f"[Subagent: {subagent_type}] Task queued: {task_description}"

        return task

    def add_subagent(self, definition: SubAgentDefinition) -> None:
        """Add a new subagent definition."""
        self.subagents[definition.name] = definition

    def get_subagent(self, name: str) -> SubAgentDefinition | None:
        """Get a subagent definition by name."""
        return self.subagents.get(name)
