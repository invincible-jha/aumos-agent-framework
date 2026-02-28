"""Sandbox guard adapter — rate limiting and resource enforcement for the agent playground.

Provides per-tenant rate limiting for playground/sandbox workflow executions (Gap #130).
Backed by Redis with a sliding window counter.
"""

import time
import uuid
from dataclasses import dataclass

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_SANDBOX_RATE_LIMIT_PER_MINUTE = 10
_SANDBOX_MAX_EXECUTION_SECONDS = 60
_SANDBOX_KEY_PREFIX = "agf:sandbox:rate:"


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining: int
    retry_after_seconds: int


class SandboxGuard:
    """Enforce per-tenant rate limits for sandbox/playground workflow executions.

    Uses Redis sliding window counters with 60-second TTL.
    Each tenant is limited to _SANDBOX_RATE_LIMIT_PER_MINUTE executions per minute.
    """

    def __init__(self, redis_client: "Any") -> None:  # type: ignore[name-defined]  # noqa: F821
        """Initialize with a Redis client.

        Args:
            redis_client: Async Redis client (redis.asyncio or similar).
        """
        self._redis = redis_client

    async def check_rate_limit(self, tenant_id: str | uuid.UUID) -> RateLimitResult:
        """Check whether the tenant is within the sandbox rate limit.

        Uses an atomic INCR + EXPIRE sliding window. If the counter exceeds
        the per-minute limit, returns allowed=False with retry_after_seconds.

        Args:
            tenant_id: Tenant UUID or string identifier.

        Returns:
            RateLimitResult with allowed flag, remaining quota, and retry delay.
        """
        key = f"{_SANDBOX_KEY_PREFIX}{tenant_id}"
        now = int(time.time())
        window_start = now - 60

        pipeline = self._redis.pipeline()
        pipeline.zadd(key, {str(now): now})
        pipeline.zremrangebyscore(key, 0, window_start)
        pipeline.zcard(key)
        pipeline.expire(key, 60)
        results = await pipeline.execute()

        current_count: int = results[2]

        if current_count > _SANDBOX_RATE_LIMIT_PER_MINUTE:
            logger.warning(
                "Sandbox rate limit exceeded",
                tenant_id=str(tenant_id),
                count=current_count,
                limit=_SANDBOX_RATE_LIMIT_PER_MINUTE,
            )
            return RateLimitResult(
                allowed=False,
                remaining=0,
                retry_after_seconds=60,
            )

        remaining = max(0, _SANDBOX_RATE_LIMIT_PER_MINUTE - current_count)
        logger.info(
            "Sandbox rate limit check passed",
            tenant_id=str(tenant_id),
            count=current_count,
            remaining=remaining,
        )
        return RateLimitResult(allowed=True, remaining=remaining, retry_after_seconds=0)

    async def enforce(self, tenant_id: str | uuid.UUID) -> None:
        """Enforce rate limit; raise ValueError if exceeded.

        Args:
            tenant_id: Tenant UUID or string identifier.

        Raises:
            ValueError: If the tenant has exceeded the sandbox rate limit.
        """
        result = await self.check_rate_limit(tenant_id)
        if not result.allowed:
            raise ValueError(
                f"Sandbox rate limit exceeded. Retry after {result.retry_after_seconds} seconds."
            )


# Type alias for annotation — actual import happens lazily to avoid circular deps
from typing import Any  # noqa: E402
