"""Memory configuration and factory functions.

Provides configuration management and factory functions for
creating session stores and LangGraph checkpointers.

Thread-safe singleton pattern is used for global instances.
"""

import logging
import os
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.memory.base import BaseSessionStore

logger = logging.getLogger(__name__)

# Lock for thread-safe singleton initialization
_config_lock = threading.Lock()
_store_lock = threading.Lock()


class MemoryBackend(str, Enum):
    """Available memory storage backends."""

    MEMORY = "memory"
    REDIS = "redis"
    SQLITE = "sqlite"


class CheckpointerType(str, Enum):
    """Available LangGraph checkpointer types."""

    MEMORY = "memory"
    REDIS = "redis"
    SQLITE = "sqlite"
    POSTGRES = "postgres"


@dataclass
class MemoryConfig:
    """Configuration for memory storage.

    Attributes:
        backend: Storage backend to use.
        redis_url: Redis connection URL.
        sqlite_path: SQLite database path.
        session_ttl_hours: Default session TTL.
        max_sessions: Maximum sessions (memory backend).
        key_prefix: Key prefix for Redis.
    """

    backend: MemoryBackend = MemoryBackend.MEMORY
    redis_url: str = "redis://localhost:6379"
    sqlite_path: str = "data/sessions.db"
    session_ttl_hours: int = 24
    max_sessions: int = 10000
    key_prefix: str = "session:"

    @classmethod
    def from_env(cls) -> "MemoryConfig":
        """Create config from environment variables.

        Environment variables:
            MEMORY_BACKEND: memory, redis, or sqlite
            REDIS_URL: Redis connection URL
            SQLITE_PATH: SQLite database path
            SESSION_TTL_HOURS: Session TTL in hours
            MAX_SESSIONS: Maximum sessions (memory backend)
            SESSION_KEY_PREFIX: Key prefix for Redis

        Returns:
            Memory configuration.
        """
        backend_str = os.getenv("MEMORY_BACKEND", "memory").lower()
        try:
            backend = MemoryBackend(backend_str)
        except ValueError:
            logger.warning(f"Unknown memory backend '{backend_str}', using 'memory'")
            backend = MemoryBackend.MEMORY

        return cls(
            backend=backend,
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            sqlite_path=os.getenv("SQLITE_PATH", "data/sessions.db"),
            session_ttl_hours=int(os.getenv("SESSION_TTL_HOURS", "24")),
            max_sessions=int(os.getenv("MAX_SESSIONS", "10000")),
            key_prefix=os.getenv("SESSION_KEY_PREFIX", "session:"),
        )


# Global instances (protected by locks for thread safety)
_memory_config: MemoryConfig | None = None
_session_store: BaseSessionStore | None = None


def get_memory_config() -> MemoryConfig:
    """Get or create global memory configuration.

    Thread-safe singleton pattern using double-checked locking.

    Returns:
        Memory configuration.
    """
    global _memory_config
    if _memory_config is None:
        with _config_lock:
            # Double-check after acquiring lock
            if _memory_config is None:
                _memory_config = MemoryConfig.from_env()
    return _memory_config


def set_memory_config(config: MemoryConfig) -> None:
    """Set global memory configuration.

    Thread-safe update that also resets the session store.

    Args:
        config: Memory configuration.
    """
    global _memory_config, _session_store
    with _config_lock:
        with _store_lock:
            _memory_config = config
            # Close existing store before resetting
            if _session_store:
                _session_store.close()
            _session_store = None


def reset_memory_config() -> None:
    """Reset global memory configuration.

    Thread-safe reset. The next call to get_memory_config will re-read
    from environment.
    """
    global _memory_config
    with _config_lock:
        _memory_config = None


def get_session_store(config: MemoryConfig | None = None) -> BaseSessionStore:
    """Get or create global session store.

    Thread-safe singleton pattern using double-checked locking.

    Args:
        config: Optional configuration override.

    Returns:
        Session store instance.
    """
    global _session_store

    # Fast path: return existing store if available and no override
    if _session_store is not None and config is None:
        return _session_store

    with _store_lock:
        # Double-check after acquiring lock
        if _session_store is not None and config is None:
            return _session_store

        cfg = config or get_memory_config()

        if cfg.backend == MemoryBackend.MEMORY:
            from app.memory.memory_store import InMemorySessionStore

            _session_store = InMemorySessionStore(max_sessions=cfg.max_sessions)
            logger.info("Using in-memory session store")

        elif cfg.backend == MemoryBackend.REDIS:
            from app.memory.redis_store import RedisSessionStore

            _session_store = RedisSessionStore(
                url=cfg.redis_url,
                prefix=cfg.key_prefix,
                default_ttl_hours=cfg.session_ttl_hours,
            )
            logger.info(f"Using Redis session store at {cfg.redis_url}")

        elif cfg.backend == MemoryBackend.SQLITE:
            from app.memory.sqlite_store import SQLiteSessionStore

            _session_store = SQLiteSessionStore(
                db_path=cfg.sqlite_path,
                default_ttl_hours=cfg.session_ttl_hours,
            )
            logger.info(f"Using SQLite session store at {cfg.sqlite_path}")

        else:
            # Fallback to memory
            from app.memory.memory_store import InMemorySessionStore

            _session_store = InMemorySessionStore()
            logger.warning(f"Unknown backend {cfg.backend}, using in-memory store")

        return _session_store


def reset_session_store() -> None:
    """Reset global session store instance.

    Thread-safe reset that closes the existing store.
    """
    global _session_store
    with _store_lock:
        if _session_store:
            _session_store.close()
        _session_store = None


def get_checkpointer(
    checkpointer_type: CheckpointerType | None = None,
    config: MemoryConfig | None = None,
) -> Any:
    """Get a LangGraph checkpointer.

    Args:
        checkpointer_type: Type of checkpointer.
        config: Optional memory configuration.

    Returns:
        LangGraph checkpointer instance.
    """
    cfg = config or get_memory_config()

    # Default to matching session store backend
    if checkpointer_type is None:
        checkpointer_type = CheckpointerType(cfg.backend.value)

    if checkpointer_type == CheckpointerType.MEMORY:
        try:
            from langgraph.checkpoint.memory import MemorySaver

            logger.debug("Using MemorySaver checkpointer")
            return MemorySaver()
        except ImportError:
            logger.error("langgraph not installed")
            raise

    elif checkpointer_type == CheckpointerType.SQLITE:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            logger.debug(f"Using SqliteSaver checkpointer at {cfg.sqlite_path}")
            return SqliteSaver.from_conn_string(cfg.sqlite_path)
        except ImportError:
            logger.warning("SqliteSaver not available, falling back to MemorySaver")
            from langgraph.checkpoint.memory import MemorySaver

            return MemorySaver()

    elif checkpointer_type == CheckpointerType.POSTGRES:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            postgres_url = os.getenv("POSTGRES_URL", "")
            if not postgres_url:
                raise ValueError("POSTGRES_URL environment variable not set")

            logger.debug("Using PostgresSaver checkpointer")
            return PostgresSaver.from_conn_string(postgres_url)
        except ImportError:
            logger.warning("PostgresSaver not available, falling back to MemorySaver")
            from langgraph.checkpoint.memory import MemorySaver

            return MemorySaver()

    elif checkpointer_type == CheckpointerType.REDIS:
        # Redis checkpointer requires custom implementation or falls back
        logger.warning("Redis checkpointer not natively supported, using MemorySaver")
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    else:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()


def create_session_store(backend: MemoryBackend, **kwargs: Any) -> BaseSessionStore:
    """Create a session store with specific backend.

    Args:
        backend: Storage backend to use.
        **kwargs: Backend-specific arguments.

    Returns:
        Session store instance.
    """
    if backend == MemoryBackend.MEMORY:
        from app.memory.memory_store import InMemorySessionStore

        return InMemorySessionStore(
            max_sessions=kwargs.get("max_sessions", 10000),
        )

    elif backend == MemoryBackend.REDIS:
        from app.memory.redis_store import RedisSessionStore

        return RedisSessionStore(
            url=kwargs.get("url", "redis://localhost:6379"),
            prefix=kwargs.get("prefix", "session:"),
            default_ttl_hours=kwargs.get("ttl_hours", 24),
        )

    elif backend == MemoryBackend.SQLITE:
        from app.memory.sqlite_store import SQLiteSessionStore

        return SQLiteSessionStore(
            db_path=kwargs.get("db_path", "sessions.db"),
            default_ttl_hours=kwargs.get("ttl_hours", 168),
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")
