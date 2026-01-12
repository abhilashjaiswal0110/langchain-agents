"""Base storage interface for Deep Agent context management."""

from abc import ABC, abstractmethod
from typing import Any

from app.deepagents.core.types import FileEntry, Todo


class BaseStorage(ABC):
    """Abstract base class for Deep Agent storage backends.

    Storage backends handle persistence of:
    - Context files (investigation notes, reports, etc.)
    - Todo lists for task tracking
    - Session metadata
    """

    @abstractmethod
    def save_file(self, session_id: str, path: str, content: str, metadata: dict | None = None) -> FileEntry:
        """Save a file to storage.

        Args:
            session_id: Session identifier.
            path: File path.
            content: File content.
            metadata: Optional metadata.

        Returns:
            Created FileEntry.
        """
        pass

    @abstractmethod
    def read_file(self, session_id: str, path: str) -> FileEntry | None:
        """Read a file from storage.

        Args:
            session_id: Session identifier.
            path: File path.

        Returns:
            FileEntry or None if not found.
        """
        pass

    @abstractmethod
    def delete_file(self, session_id: str, path: str) -> bool:
        """Delete a file from storage.

        Args:
            session_id: Session identifier.
            path: File path.

        Returns:
            True if deleted, False if not found.
        """
        pass

    @abstractmethod
    def list_files(self, session_id: str, directory: str = "/") -> list[str]:
        """List files in a directory.

        Args:
            session_id: Session identifier.
            directory: Directory path.

        Returns:
            List of file paths.
        """
        pass

    @abstractmethod
    def save_todos(self, session_id: str, todos: list[Todo]) -> None:
        """Save todos to storage.

        Args:
            session_id: Session identifier.
            todos: List of todos.
        """
        pass

    @abstractmethod
    def get_todos(self, session_id: str) -> list[Todo]:
        """Get todos from storage.

        Args:
            session_id: Session identifier.

        Returns:
            List of todos.
        """
        pass

    @abstractmethod
    def save_session_metadata(self, session_id: str, metadata: dict[str, Any]) -> None:
        """Save session metadata.

        Args:
            session_id: Session identifier.
            metadata: Metadata dictionary.
        """
        pass

    @abstractmethod
    def get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
        """Get session metadata.

        Args:
            session_id: Session identifier.

        Returns:
            Metadata dictionary or None.
        """
        pass

    @abstractmethod
    def clear_session(self, session_id: str) -> None:
        """Clear all data for a session.

        Args:
            session_id: Session identifier.
        """
        pass
