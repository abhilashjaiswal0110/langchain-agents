"""In-memory storage backend for Deep Agent."""

from datetime import datetime
from typing import Any

from app.deepagents.core.types import FileEntry, Todo
from app.deepagents.storage.base import BaseStorage


class MemoryStorage(BaseStorage):
    """In-memory storage backend.

    Fast, non-persistent storage suitable for development and testing.
    All data is lost when the process terminates.
    """

    def __init__(self) -> None:
        """Initialize memory storage."""
        self._files: dict[str, dict[str, FileEntry]] = {}  # session_id -> {path: FileEntry}
        self._todos: dict[str, list[Todo]] = {}  # session_id -> [Todo]
        self._metadata: dict[str, dict[str, Any]] = {}  # session_id -> metadata

    def save_file(
        self,
        session_id: str,
        path: str,
        content: str,
        metadata: dict | None = None,
    ) -> FileEntry:
        """Save a file to memory."""
        if session_id not in self._files:
            self._files[session_id] = {}

        now = datetime.now()
        existing = self._files[session_id].get(path)

        entry = FileEntry(
            path=path,
            content=content,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            file_type=self._detect_file_type(path),
            metadata=metadata or {},
        )

        self._files[session_id][path] = entry
        return entry

    def read_file(self, session_id: str, path: str) -> FileEntry | None:
        """Read a file from memory."""
        return self._files.get(session_id, {}).get(path)

    def delete_file(self, session_id: str, path: str) -> bool:
        """Delete a file from memory."""
        if session_id in self._files and path in self._files[session_id]:
            del self._files[session_id][path]
            return True
        return False

    def list_files(self, session_id: str, directory: str = "/") -> list[str]:
        """List files in memory."""
        if session_id not in self._files:
            return []

        files = []
        for path in self._files[session_id].keys():
            if directory == "/" or path.startswith(directory):
                files.append(path)

        return sorted(files)

    def save_todos(self, session_id: str, todos: list[Todo]) -> None:
        """Save todos to memory."""
        self._todos[session_id] = todos

    def get_todos(self, session_id: str) -> list[Todo]:
        """Get todos from memory."""
        return self._todos.get(session_id, [])

    def save_session_metadata(self, session_id: str, metadata: dict[str, Any]) -> None:
        """Save session metadata to memory."""
        if session_id not in self._metadata:
            self._metadata[session_id] = {}
        self._metadata[session_id].update(metadata)

    def get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
        """Get session metadata from memory."""
        return self._metadata.get(session_id)

    def clear_session(self, session_id: str) -> None:
        """Clear all data for a session."""
        self._files.pop(session_id, None)
        self._todos.pop(session_id, None)
        self._metadata.pop(session_id, None)

    def _detect_file_type(self, path: str) -> str:
        """Detect file type from path."""
        if path.endswith(".md"):
            return "markdown"
        elif path.endswith(".json"):
            return "json"
        elif path.endswith(".yaml") or path.endswith(".yml"):
            return "yaml"
        elif path.endswith(".py"):
            return "python"
        else:
            return "text"
