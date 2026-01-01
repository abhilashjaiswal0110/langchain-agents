"""Rate limiting for enterprise IT agents.

Provides:
- Token bucket rate limiting
- Per-user and per-agent limits
- Redis-backed for distributed limiting
- In-memory fallback for development
- Configurable burst allowance
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.

    Attributes:
        enabled: Whether rate limiting is enabled
        backend: Storage backend ("redis" or "memory")
        redis_url: Redis connection URL
        default_limit: Default requests per window (e.g., "100/minute")
        burst_multiplier: Multiplier for burst allowance
        window_seconds: Default window size in seconds
    """

    enabled: bool = True
    backend: Literal["redis", "memory"] = "memory"
    redis_url: str = "redis://localhost:6379"
    default_limit: int = 100
    burst_multiplier: float = 1.5
    window_seconds: int = 60

    @classmethod
    def from_env(cls) -> "RateLimitConfig":
        """Create config from environment variables."""
        return cls(
            enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
            backend=os.getenv("RATE_LIMIT_BACKEND", "memory"),  # type: ignore[arg-type]
            redis_url=os.getenv("RATE_LIMIT_REDIS_URL", "redis://localhost:6379"),
            default_limit=int(os.getenv("RATE_LIMIT_DEFAULT", "100")),
            burst_multiplier=float(os.getenv("RATE_LIMIT_BURST_MULTIPLIER", "1.5")),
            window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
        )


@dataclass
class RateLimitResult:
    """Result of a rate limit check.

    Attributes:
        allowed: Whether the request is allowed
        remaining: Remaining requests in current window
        limit: Total limit for the window
        reset_at: Unix timestamp when the window resets
        retry_after: Seconds to wait before retrying (if not allowed)
    """

    allowed: bool
    remaining: int
    limit: int
    reset_at: float
    retry_after: float | None = None


@dataclass
class RateLimitRule:
    """A rate limit rule for a specific scope.

    Attributes:
        scope: The scope this rule applies to (e.g., "user", "agent", "global")
        limit: Maximum requests per window
        window_seconds: Window size in seconds
        burst_limit: Maximum burst requests (if higher than limit)
    """

    scope: str
    limit: int
    window_seconds: int = 60
    burst_limit: int | None = None

    @property
    def effective_burst(self) -> int:
        """Get effective burst limit."""
        return self.burst_limit or self.limit


class TokenBucket:
    """In-memory token bucket for rate limiting.

    Attributes:
        capacity: Maximum tokens in bucket
        refill_rate: Tokens added per second
        tokens: Current token count
        last_update: Last update timestamp
    """

    def __init__(self, capacity: int, refill_rate: float) -> None:
        """Initialize token bucket.

        Args:
            capacity: Maximum tokens.
            refill_rate: Tokens per second to refill.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_update = time.time()

    def consume(self, tokens: int = 1) -> tuple[bool, float]:
        """Attempt to consume tokens.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            Tuple of (success, remaining_tokens).
        """
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now

        # Refill tokens
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True, self.tokens
        return False, self.tokens

    def time_until_available(self, tokens: int = 1) -> float:
        """Calculate time until tokens are available.

        Args:
            tokens: Number of tokens needed.

        Returns:
            Seconds until tokens are available.
        """
        if self.tokens >= tokens:
            return 0.0
        needed = tokens - self.tokens
        return needed / self.refill_rate


class InMemoryRateLimiter:
    """In-memory rate limiter using token buckets."""

    def __init__(self, config: RateLimitConfig) -> None:
        """Initialize in-memory rate limiter.

        Args:
            config: Rate limit configuration.
        """
        self.config = config
        self._buckets: dict[str, TokenBucket] = {}
        self._rules: dict[str, RateLimitRule] = {}
        self._lock = asyncio.Lock()

    def add_rule(self, rule: RateLimitRule) -> None:
        """Add a rate limit rule.

        Args:
            rule: Rule to add.
        """
        self._rules[rule.scope] = rule

    def _get_bucket(self, key: str, rule: RateLimitRule) -> TokenBucket:
        """Get or create a token bucket for a key.

        Args:
            key: The bucket key.
            rule: Rule to use for bucket creation.

        Returns:
            Token bucket for the key.
        """
        if key not in self._buckets:
            refill_rate = rule.limit / rule.window_seconds
            self._buckets[key] = TokenBucket(
                capacity=rule.effective_burst,
                refill_rate=refill_rate,
            )
        return self._buckets[key]

    async def check(
        self,
        key: str,
        scope: str = "default",
        tokens: int = 1,
    ) -> RateLimitResult:
        """Check rate limit for a key.

        Args:
            key: Identifier for rate limiting (e.g., user_id, api_key).
            scope: Rule scope to apply.
            tokens: Number of tokens to consume.

        Returns:
            Rate limit result.
        """
        if not self.config.enabled:
            return RateLimitResult(
                allowed=True,
                remaining=self.config.default_limit,
                limit=self.config.default_limit,
                reset_at=time.time() + self.config.window_seconds,
            )

        rule = self._rules.get(scope)
        if not rule:
            # Create default rule
            rule = RateLimitRule(
                scope=scope,
                limit=self.config.default_limit,
                window_seconds=self.config.window_seconds,
                burst_limit=int(self.config.default_limit * self.config.burst_multiplier),
            )

        bucket_key = f"{scope}:{key}"

        async with self._lock:
            bucket = self._get_bucket(bucket_key, rule)
            allowed, remaining = bucket.consume(tokens)

            reset_at = time.time() + rule.window_seconds
            retry_after = None if allowed else bucket.time_until_available(tokens)

            return RateLimitResult(
                allowed=allowed,
                remaining=int(remaining),
                limit=rule.limit,
                reset_at=reset_at,
                retry_after=retry_after,
            )

    async def reset(self, key: str, scope: str = "default") -> None:
        """Reset rate limit for a key.

        Args:
            key: Identifier to reset.
            scope: Rule scope.
        """
        bucket_key = f"{scope}:{key}"
        async with self._lock:
            if bucket_key in self._buckets:
                del self._buckets[bucket_key]

    async def cleanup_expired(self, max_age_seconds: int = 3600) -> int:
        """Clean up expired buckets.

        Args:
            max_age_seconds: Maximum age for inactive buckets.

        Returns:
            Number of buckets cleaned up.
        """
        now = time.time()
        expired = []

        async with self._lock:
            for key, bucket in self._buckets.items():
                if now - bucket.last_update > max_age_seconds:
                    expired.append(key)

            for key in expired:
                del self._buckets[key]

        return len(expired)


class RedisRateLimiter:
    """Redis-backed rate limiter for distributed systems."""

    def __init__(self, config: RateLimitConfig) -> None:
        """Initialize Redis rate limiter.

        Args:
            config: Rate limit configuration.
        """
        self.config = config
        self._redis: Any | None = None
        self._rules: dict[str, RateLimitRule] = {}

    async def _get_redis(self) -> Any:
        """Get or create Redis connection."""
        if self._redis is None:
            if not REDIS_AVAILABLE:
                msg = "redis package not installed. Install with: pip install redis"
                raise RuntimeError(msg)
            self._redis = await aioredis.from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis

    def add_rule(self, rule: RateLimitRule) -> None:
        """Add a rate limit rule.

        Args:
            rule: Rule to add.
        """
        self._rules[rule.scope] = rule

    async def check(
        self,
        key: str,
        scope: str = "default",
        tokens: int = 1,
    ) -> RateLimitResult:
        """Check rate limit for a key using sliding window.

        Args:
            key: Identifier for rate limiting.
            scope: Rule scope to apply.
            tokens: Number of tokens to consume.

        Returns:
            Rate limit result.
        """
        if not self.config.enabled:
            return RateLimitResult(
                allowed=True,
                remaining=self.config.default_limit,
                limit=self.config.default_limit,
                reset_at=time.time() + self.config.window_seconds,
            )

        rule = self._rules.get(scope)
        if not rule:
            rule = RateLimitRule(
                scope=scope,
                limit=self.config.default_limit,
                window_seconds=self.config.window_seconds,
                burst_limit=int(self.config.default_limit * self.config.burst_multiplier),
            )

        redis_client = await self._get_redis()
        redis_key = f"ratelimit:{scope}:{key}"
        now = time.time()
        window_start = now - rule.window_seconds

        # Sliding window log implementation
        pipe = redis_client.pipeline()

        # Remove expired entries
        pipe.zremrangebyscore(redis_key, "-inf", window_start)

        # Count current window
        pipe.zcard(redis_key)

        # Execute pipeline
        results = await pipe.execute()
        current_count = results[1]

        # Check if allowed
        if current_count + tokens <= rule.effective_burst:
            # Add new entries
            pipe2 = redis_client.pipeline()
            for _ in range(tokens):
                pipe2.zadd(redis_key, {f"{now}:{id(pipe2)}": now})
            pipe2.expire(redis_key, rule.window_seconds + 1)
            await pipe2.execute()

            return RateLimitResult(
                allowed=True,
                remaining=rule.limit - current_count - tokens,
                limit=rule.limit,
                reset_at=now + rule.window_seconds,
            )

        # Rate limited
        # Get oldest entry to calculate retry time
        oldest = await redis_client.zrange(redis_key, 0, 0, withscores=True)
        retry_after = None
        if oldest:
            oldest_time = oldest[0][1]
            retry_after = oldest_time + rule.window_seconds - now

        return RateLimitResult(
            allowed=False,
            remaining=0,
            limit=rule.limit,
            reset_at=now + rule.window_seconds,
            retry_after=retry_after if retry_after and retry_after > 0 else 1.0,
        )

    async def reset(self, key: str, scope: str = "default") -> None:
        """Reset rate limit for a key.

        Args:
            key: Identifier to reset.
            scope: Rule scope.
        """
        redis_client = await self._get_redis()
        redis_key = f"ratelimit:{scope}:{key}"
        await redis_client.delete(redis_key)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


class RateLimiter:
    """Unified rate limiter with backend abstraction.

    Supports both Redis (distributed) and in-memory (single instance) backends.
    """

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration. If None, loads from environment.
        """
        self.config = config or RateLimitConfig.from_env()
        self._backend: InMemoryRateLimiter | RedisRateLimiter

        if self.config.backend == "redis" and REDIS_AVAILABLE:
            self._backend = RedisRateLimiter(self.config)
        else:
            self._backend = InMemoryRateLimiter(self.config)

    def add_rule(self, rule: RateLimitRule) -> None:
        """Add a rate limit rule.

        Args:
            rule: Rule to add.
        """
        self._backend.add_rule(rule)

    async def check(
        self,
        key: str,
        scope: str = "default",
        tokens: int = 1,
    ) -> RateLimitResult:
        """Check rate limit for a key.

        Args:
            key: Identifier for rate limiting (e.g., user_id, api_key).
            scope: Rule scope to apply.
            tokens: Number of tokens to consume.

        Returns:
            Rate limit result.
        """
        return await self._backend.check(key, scope, tokens)

    async def check_user(self, user_id: str, tokens: int = 1) -> RateLimitResult:
        """Check rate limit for a user.

        Args:
            user_id: User identifier.
            tokens: Number of tokens to consume.

        Returns:
            Rate limit result.
        """
        return await self.check(user_id, scope="user", tokens=tokens)

    async def check_agent(
        self,
        user_id: str,
        agent_type: str,
        tokens: int = 1,
    ) -> RateLimitResult:
        """Check rate limit for a specific agent.

        Args:
            user_id: User identifier.
            agent_type: Agent type being invoked.
            tokens: Number of tokens to consume.

        Returns:
            Rate limit result.
        """
        key = f"{user_id}:{agent_type}"
        return await self.check(key, scope="agent", tokens=tokens)

    async def check_global(self, tokens: int = 1) -> RateLimitResult:
        """Check global rate limit.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            Rate limit result.
        """
        return await self.check("global", scope="global", tokens=tokens)

    async def reset(self, key: str, scope: str = "default") -> None:
        """Reset rate limit for a key.

        Args:
            key: Identifier to reset.
            scope: Rule scope.
        """
        await self._backend.reset(key, scope)

    async def close(self) -> None:
        """Close any open connections."""
        if isinstance(self._backend, RedisRateLimiter):
            await self._backend.close()


# Default rate limit rules
DEFAULT_RULES: list[RateLimitRule] = [
    RateLimitRule(scope="user", limit=100, window_seconds=60, burst_limit=150),
    RateLimitRule(scope="agent", limit=30, window_seconds=60, burst_limit=45),
    RateLimitRule(scope="global", limit=1000, window_seconds=60, burst_limit=1500),
]


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter.

    Returns:
        Global rate limiter instance.
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
        # Add default rules
        for rule in DEFAULT_RULES:
            _rate_limiter.add_rule(rule)
    return _rate_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter."""
    global _rate_limiter
    _rate_limiter = None


# Convenience functions


async def check_rate_limit(
    user_id: str,
    agent_type: str | None = None,
) -> RateLimitResult:
    """Check rate limit for a user/agent combination.

    Args:
        user_id: User identifier.
        agent_type: Optional agent type.

    Returns:
        Rate limit result.
    """
    limiter = get_rate_limiter()

    # Check user limit first
    user_result = await limiter.check_user(user_id)
    if not user_result.allowed:
        return user_result

    # Check agent-specific limit if provided
    if agent_type:
        agent_result = await limiter.check_agent(user_id, agent_type)
        if not agent_result.allowed:
            return agent_result

    # Check global limit
    global_result = await limiter.check_global()
    return global_result


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        result: RateLimitResult,
    ) -> None:
        """Initialize error.

        Args:
            message: Error message.
            result: Rate limit result with details.
        """
        super().__init__(message)
        self.result = result
        self.retry_after = result.retry_after


async def require_rate_limit(
    user_id: str,
    agent_type: str | None = None,
) -> RateLimitResult:
    """Require rate limit check to pass.

    Args:
        user_id: User identifier.
        agent_type: Optional agent type.

    Returns:
        Rate limit result if allowed.

    Raises:
        RateLimitExceededError: If rate limit is exceeded.
    """
    result = await check_rate_limit(user_id, agent_type)
    if not result.allowed:
        raise RateLimitExceededError(
            f"Rate limit exceeded. Retry after {result.retry_after:.1f} seconds.",
            result,
        )
    return result
