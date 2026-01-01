"""Checkpointer factory for persistent agent state management.

Supports multiple backends:
- PostgreSQL (recommended for production)
- SQLite (good for development/single instance)
- Memory (in-memory, non-persistent - for testing)
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver


class CheckpointerBackend(str, Enum):
    """Supported checkpointer backends."""

    POSTGRES = "postgres"
    SQLITE = "sqlite"
    MEMORY = "memory"


@dataclass
class CheckpointerConfig:
    """Configuration for checkpointer initialization.

    Args:
        backend: The storage backend to use.
        connection_string: Database connection string (for postgres/sqlite).
        pool_size: Connection pool size for PostgreSQL.
        max_overflow: Max overflow connections for PostgreSQL.
    """

    backend: CheckpointerBackend = CheckpointerBackend.MEMORY
    connection_string: str | None = None
    pool_size: int = 5
    max_overflow: int = 10

    @classmethod
    def from_env(cls) -> "CheckpointerConfig":
        """Create config from environment variables.

        Environment variables:
            MEMORY_BACKEND: postgres, sqlite, or memory (default: memory)
            DATABASE_URL: Connection string for postgres/sqlite
            DB_POOL_SIZE: Connection pool size (default: 5)
            DB_MAX_OVERFLOW: Max overflow connections (default: 10)

        Returns:
            CheckpointerConfig instance
        """
        backend_str = os.getenv("MEMORY_BACKEND", "memory").lower()
        try:
            backend = CheckpointerBackend(backend_str)
        except ValueError:
            print(f"Warning: Unknown backend '{backend_str}', using memory")
            backend = CheckpointerBackend.MEMORY

        return cls(
            backend=backend,
            connection_string=os.getenv("DATABASE_URL"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
        )


# Global checkpointer instance (singleton pattern)
_checkpointer: BaseCheckpointSaver | None = None


def create_checkpointer(
    config: CheckpointerConfig | None = None,
) -> BaseCheckpointSaver:
    """Create a checkpointer based on configuration.

    Args:
        config: Checkpointer configuration. If None, loads from environment.

    Returns:
        A checkpointer instance (PostgresSaver, SqliteSaver, or MemorySaver).

    Raises:
        ValueError: If required configuration is missing.
        ImportError: If required backend package is not installed.
    """
    if config is None:
        config = CheckpointerConfig.from_env()

    if config.backend == CheckpointerBackend.POSTGRES:
        return _create_postgres_checkpointer(config)
    elif config.backend == CheckpointerBackend.SQLITE:
        return _create_sqlite_checkpointer(config)
    else:
        return MemorySaver()


def _create_postgres_checkpointer(config: CheckpointerConfig) -> BaseCheckpointSaver:
    """Create PostgreSQL checkpointer.

    Args:
        config: Checkpointer configuration with connection string.

    Returns:
        PostgresSaver instance.

    Raises:
        ValueError: If DATABASE_URL is not set.
        ImportError: If langgraph-checkpoint-postgres is not installed.
    """
    if not config.connection_string:
        raise ValueError(
            "DATABASE_URL environment variable is required for PostgreSQL backend. "
            "Set MEMORY_BACKEND=memory to use in-memory storage."
        )

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as e:
        raise ImportError(
            "PostgreSQL checkpointer requires 'langgraph-checkpoint-postgres' package. "
            "Install with: pip install langgraph-checkpoint-postgres"
        ) from e

    # Create connection pool with psycopg
    try:
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(
            conninfo=config.connection_string,
            min_size=1,
            max_size=config.pool_size,
            kwargs={"autocommit": True},
        )

        checkpointer = PostgresSaver(pool)
        # Setup tables if they don't exist
        checkpointer.setup()
        print(f"PostgreSQL checkpointer initialized (pool_size={config.pool_size})")
        return checkpointer

    except ImportError:
        # Fallback to sync connection if psycopg_pool not available
        print("Warning: psycopg_pool not available, using sync connection")
        return PostgresSaver.from_conn_string(config.connection_string)


def _create_sqlite_checkpointer(config: CheckpointerConfig) -> BaseCheckpointSaver:
    """Create SQLite checkpointer.

    Args:
        config: Checkpointer configuration with connection string.

    Returns:
        SqliteSaver instance.

    Raises:
        ImportError: If langgraph-checkpoint-sqlite is not installed.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as e:
        raise ImportError(
            "SQLite checkpointer requires 'langgraph-checkpoint-sqlite' package. "
            "Install with: pip install langgraph-checkpoint-sqlite"
        ) from e

    # Use connection string or default path
    db_path = config.connection_string or "sqlite:///./data/checkpoints.db"

    # Ensure data directory exists
    if db_path.startswith("sqlite:///"):
        import pathlib

        file_path = db_path.replace("sqlite:///", "")
        pathlib.Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    checkpointer = SqliteSaver.from_conn_string(db_path)
    print(f"SQLite checkpointer initialized: {db_path}")
    return checkpointer


def get_checkpointer() -> BaseCheckpointSaver:
    """Get or create the global checkpointer instance.

    Uses singleton pattern to ensure single checkpointer instance
    across the application.

    Returns:
        The global checkpointer instance.
    """
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = create_checkpointer()
    return _checkpointer


def reset_checkpointer() -> None:
    """Reset the global checkpointer instance.

    Useful for testing or reconfiguration.
    """
    global _checkpointer
    _checkpointer = None


async def get_async_checkpointer() -> BaseCheckpointSaver:
    """Get async-compatible checkpointer.

    For PostgreSQL, returns AsyncPostgresSaver if available.
    For SQLite, returns AsyncSqliteSaver if available.

    Returns:
        Async-compatible checkpointer instance.
    """
    config = CheckpointerConfig.from_env()

    if config.backend == CheckpointerBackend.POSTGRES:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            if not config.connection_string:
                raise ValueError("DATABASE_URL required for async PostgreSQL")

            checkpointer = AsyncPostgresSaver.from_conn_string(config.connection_string)
            await checkpointer.setup()
            return checkpointer
        except ImportError:
            print("Warning: AsyncPostgresSaver not available, using sync")
            return get_checkpointer()

    elif config.backend == CheckpointerBackend.SQLITE:
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            db_path = config.connection_string or "sqlite:///./data/checkpoints.db"
            return AsyncSqliteSaver.from_conn_string(db_path)
        except ImportError:
            print("Warning: AsyncSqliteSaver not available, using sync")
            return get_checkpointer()

    return MemorySaver()
