"""In-memory semantic response cache for enterprise agents.

Caching is opt-in via the ``CACHE_ENABLED`` environment variable (default
``false``) so that all existing behaviour is preserved when the flag is not
set.  Only identical or near-identical queries (after whitespace normalisation)
are served from cache; everything else passes through to the LLM unchanged.

Entries expire after ``CACHE_TTL_SECONDS`` seconds (default 3600).  When the
store reaches ``MAX_CACHE_SIZE`` entries the oldest entry is evicted (FIFO)
before the new one is added.
"""

import hashlib
import os
import time

# ---------------------------------------------------------------------------
# Module-level configuration (read once at import time so that test monkeypatching
# that sets/unsets env vars before importing will see the correct value; however,
# tests that need dynamic toggling should use the CACHE_ENABLED check inside
# get/set rather than the module constant.  The property is re-evaluated on
# every call for maximum flexibility.
# ---------------------------------------------------------------------------

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "1000"))


def is_cache_enabled() -> bool:
    """Return True if response caching is enabled via environment variable.

    Re-evaluated on every call so that environment changes (e.g. in tests)
    are picked up without requiring a module reload.

    Returns:
        ``True`` when ``CACHE_ENABLED=true`` (case-insensitive), ``False``
        otherwise.
    """
    return os.getenv("CACHE_ENABLED", "false").lower() == "true"


class AgentResponseCache:
    """In-memory cache for agent responses keyed by agent type and query.

    Cache lookups and writes are no-ops when ``CACHE_ENABLED`` is not ``true``,
    ensuring zero impact on existing behaviour in the default configuration.

    Entries carry an expiry timestamp derived from ``CACHE_TTL_SECONDS``.  A
    stale entry is treated as a miss and removed on the next ``get`` call.
    When the store reaches ``MAX_CACHE_SIZE`` the oldest entry (by insertion
    order) is evicted before the new one is added.

    Example:
        >>> cache = AgentResponseCache()
        >>> cache.set("research", "AI trends", "Here are the trends…")
        >>> cache.get("research", "AI  trends")  # normalised → same key
        'Here are the trends…'
    """

    def __init__(self) -> None:
        """Initialise an empty in-memory store."""
        # Stored as key -> (response, expiry_time).  Insertion order is
        # preserved by dict (Python 3.7+) which enables cheap FIFO eviction.
        self._store: dict[str, tuple[str, float]] = {}

    def _key(self, agent_type: str, message: str) -> str:
        """Derive a deterministic cache key from agent type and query text.

        Whitespace in ``message`` is normalised (collapsed to single spaces,
        stripped, lowercased) so that minor formatting differences do not
        produce separate cache entries.

        Args:
            agent_type: Identifier for the agent (e.g. ``"research"``).
            message: The user query or message text.

        Returns:
            A hex-encoded SHA-256 digest of the normalised cache key string.
        """
        normalized = " ".join(message.lower().split())
        raw = f"{agent_type}:{normalized}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, agent_type: str, message: str) -> str | None:
        """Return a cached response, or ``None`` when not found or cache is off.

        Expired entries are removed on access so that the store does not
        accumulate stale data indefinitely.

        Args:
            agent_type: Identifier for the agent.
            message: The user query used as part of the cache key.

        Returns:
            The previously stored response string, or ``None`` if the cache is
            disabled, the entry does not exist, or the entry has expired.
        """
        if not is_cache_enabled():
            return None
        key = self._key(agent_type, message)
        entry = self._store.get(key)
        if entry is None:
            return None
        response, expiry = entry
        if time.time() > expiry:
            del self._store[key]
            return None
        return response

    def set(self, agent_type: str, message: str, response: str) -> None:
        """Store a response in the cache.

        This is a no-op when ``CACHE_ENABLED`` is not ``true``.  When the
        store is at capacity (``MAX_CACHE_SIZE``), the oldest entry is evicted
        before the new one is inserted.

        Args:
            agent_type: Identifier for the agent.
            message: The user query used as part of the cache key.
            response: The response text to store.
        """
        if not is_cache_enabled():
            return
        key = self._key(agent_type, message)
        expiry = time.time() + CACHE_TTL_SECONDS
        # Enforce size limit: evict oldest entry (FIFO) when at capacity.
        if len(self._store) >= MAX_CACHE_SIZE:
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
        self._store[key] = (response, expiry)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._store.clear()

    def size(self) -> int:
        """Return the number of entries currently held in the cache.

        Returns:
            Count of cached entries.
        """
        return len(self._store)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_cache: AgentResponseCache | None = None


def get_cache() -> AgentResponseCache:
    """Return the process-wide singleton ``AgentResponseCache`` instance.

    The instance is created lazily on first call and reused for the lifetime
    of the process.

    Returns:
        The shared ``AgentResponseCache`` singleton.
    """
    global _cache
    if _cache is None:
        _cache = AgentResponseCache()
    return _cache
