"""Persistent file-based storage backend for Deep Agent."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from app.deepagents.core.types import FileEntry, Todo
from app.deepagents.storage.base import BaseStorage


class PersistentStorage(BaseStorage):
    """Persistent file-based storage backend.

    Stores Deep Agent context to disk for durability across restarts.
    Suitable for production single-instance deployments.
    """

    def __init__(self, base_path: str = "./data/deepagent_context") -> None:
        """Initialize persistent storage.

        Args:
            base_path: Base directory for file storage.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        """Get the path for a session's data."""
        # Sanitize session_id for filesystem
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        path = self.base_path / safe_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_files_dir(self, session_id: str) -> Path:
        """Get the files directory for a session."""
        path = self._get_session_path(session_id) / "files"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_file(
        self,
        session_id: str,
        path: str,
        content: str,
        metadata: dict | None = None,
    ) -> FileEntry:
        """Save a file to disk."""
        files_dir = self._get_files_dir(session_id)

        # Build and validate path to prevent directory traversal
        base_dir = files_dir.resolve()
        requested_path = (files_dir / path.lstrip("/")).resolve()
        if base_dir != requested_path and base_dir not in requested_path.parents:
            raise ValueError(f"Invalid file path outside of session directory: {path!r}")
        file_path = requested_path

        # Create parent directories
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists for created_at
        now = datetime.now()
        created_at = now
        if file_path.exists():
            # Read existing metadata
            meta_path = file_path.with_suffix(file_path.suffix + ".meta")
            if meta_path.exists():
                try:
                    with open(meta_path, "r") as f:
                        existing_meta = json.load(f)
                        created_at = datetime.fromisoformat(existing_meta.get("created_at", now.isoformat()))
                except (json.JSONDecodeError, ValueError):
                    pass

        # Write content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Write metadata
        entry = FileEntry(
            path=path,
            content=content,
            created_at=created_at,
            updated_at=now,
            file_type=self._detect_file_type(path),
            metadata=metadata or {},
        )

        meta_path = file_path.with_suffix(file_path.suffix + ".meta")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "created_at": entry.created_at.isoformat(),
                    "updated_at": entry.updated_at.isoformat(),
                    "file_type": entry.file_type,
                    "metadata": entry.metadata,
                },
                f,
            )

        return entry

    def read_file(self, session_id: str, path: str) -> FileEntry | None:
        """Read a file from disk."""
        files_dir = self._get_files_dir(session_id)

        # Build and validate path to prevent directory traversal
        base_dir = files_dir.resolve()
        requested_path = (files_dir / path.lstrip("/")).resolve()
        if base_dir != requested_path and base_dir not in requested_path.parents:
            raise ValueError(f"Invalid file path outside of session directory: {path!r}")
        file_path = requested_path

        if not file_path.exists():
            return None

        # Read content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Read metadata
        meta_path = file_path.with_suffix(file_path.suffix + ".meta")
        metadata = {}
        created_at = datetime.now()
        updated_at = datetime.now()
        file_type = self._detect_file_type(path)

        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    created_at = datetime.fromisoformat(meta.get("created_at", created_at.isoformat()))
                    updated_at = datetime.fromisoformat(meta.get("updated_at", updated_at.isoformat()))
                    file_type = meta.get("file_type", file_type)
                    metadata = meta.get("metadata", {})
            except (json.JSONDecodeError, ValueError):
                pass

        return FileEntry(
            path=path,
            content=content,
            created_at=created_at,
            updated_at=updated_at,
            file_type=file_type,
            metadata=metadata,
        )

    def delete_file(self, session_id: str, path: str) -> bool:
        """Delete a file from disk."""
        files_dir = self._get_files_dir(session_id)

        # Build and validate path to prevent directory traversal
        base_dir = files_dir.resolve()
        requested_path = (files_dir / path.lstrip("/")).resolve()
        if base_dir != requested_path and base_dir not in requested_path.parents:
            raise ValueError(f"Invalid file path outside of session directory: {path!r}")
        file_path = requested_path

        if not file_path.exists():
            return False

        file_path.unlink()

        # Also delete metadata
        meta_path = file_path.with_suffix(file_path.suffix + ".meta")
        if meta_path.exists():
            meta_path.unlink()

        return True

    def list_files(self, session_id: str, directory: str = "/") -> list[str]:
        """List files on disk."""
        files_dir = self._get_files_dir(session_id)

        if not files_dir.exists():
            return []

        files = []
        for file_path in files_dir.rglob("*"):
            if file_path.is_file() and not file_path.suffix == ".meta":
                # Convert to relative path
                rel_path = "/" + str(file_path.relative_to(files_dir)).replace("\\", "/")
                if directory == "/" or rel_path.startswith(directory):
                    files.append(rel_path)

        return sorted(files)

    def save_todos(self, session_id: str, todos: list[Todo]) -> None:
        """Save todos to disk."""
        session_path = self._get_session_path(session_id)
        todos_path = session_path / "todos.json"

        with open(todos_path, "w", encoding="utf-8") as f:
            json.dump([t.model_dump(mode="json") for t in todos], f, indent=2, default=str)

    def get_todos(self, session_id: str) -> list[Todo]:
        """Get todos from disk."""
        session_path = self._get_session_path(session_id)
        todos_path = session_path / "todos.json"

        if not todos_path.exists():
            return []

        try:
            with open(todos_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [Todo(**t) for t in data]
        except (json.JSONDecodeError, ValueError):
            return []

    def save_session_metadata(self, session_id: str, metadata: dict[str, Any]) -> None:
        """Save session metadata to disk."""
        session_path = self._get_session_path(session_id)
        meta_path = session_path / "session.json"

        existing = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                pass

        existing.update(metadata)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, default=str)

    def get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
        """Get session metadata from disk."""
        session_path = self._get_session_path(session_id)
        meta_path = session_path / "session.json"

        if not meta_path.exists():
            return None

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None

    def clear_session(self, session_id: str) -> None:
        """Clear all data for a session."""
        import shutil

        session_path = self._get_session_path(session_id)
        if session_path.exists():
            shutil.rmtree(session_path)

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
