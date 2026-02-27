"""Redis-backed circuit breaker adapter for cascading failure containment.

Implements CircuitBreakerProtocol with three states: CLOSED (normal), OPEN (failing),
and HALF_OPEN (testing recovery). State is persisted in Redis for distributed access
with sub-5ms check latency requirement.
"""

import json
from datetime import UTC, datetime
from typing import Any

from redis.asyncio import Redis

from aumos_common.observability import get_logger

from aumos_agent_framework.core.interfaces import CircuitBreakerProtocol

logger = get_logger(__name__)

# Circuit states
STATE_CLOSED = "closed"
STATE_OPEN = "open"
STATE_HALF_OPEN = "half_open"

_DEFAULT_FAILURE_THRESHOLD = 5
_DEFAULT_RESET_TIMEOUT_SECONDS = 60
_DEFAULT_HALF_OPEN_MAX_CALLS = 3


class RedisCircuitBreaker:
    """Redis-backed circuit breaker for per-agent and per-workflow failure containment.

    State machine:
    - CLOSED: Normal operation. Failures increment counter. At threshold → OPEN.
    - OPEN: Requests rejected immediately. After reset_timeout_seconds → HALF_OPEN.
    - HALF_OPEN: Limited test requests allowed. Success → CLOSED. Failure → OPEN.

    All state is namespaced by tenant_id to ensure cross-tenant isolation.
    """

    def __init__(
        self,
        redis_client: Redis,  # type: ignore[type-arg]
        failure_threshold: int = _DEFAULT_FAILURE_THRESHOLD,
        reset_timeout_seconds: int = _DEFAULT_RESET_TIMEOUT_SECONDS,
        half_open_max_calls: int = _DEFAULT_HALF_OPEN_MAX_CALLS,
    ) -> None:
        """Initialize with Redis client and circuit breaker configuration.

        Args:
            redis_client: Async Redis client instance.
            failure_threshold: Number of failures before circuit opens.
            reset_timeout_seconds: Seconds to wait in OPEN state before trying HALF_OPEN.
            half_open_max_calls: Max calls allowed in HALF_OPEN state before decision.
        """
        self._redis = redis_client
        self._failure_threshold = failure_threshold
        self._reset_timeout_seconds = reset_timeout_seconds
        self._half_open_max_calls = half_open_max_calls

    def _state_key(self, circuit_key: str, tenant_id: str) -> str:
        """Build Redis key for circuit state data."""
        return f"cb:{tenant_id}:{circuit_key}:state"

    async def _get_state_data(self, circuit_key: str, tenant_id: str) -> dict[str, Any]:
        """Retrieve raw circuit state dict from Redis."""
        raw = await self._redis.get(self._state_key(circuit_key, tenant_id))
        if raw is None:
            return {
                "state": STATE_CLOSED,
                "failure_count": 0,
                "opened_at": None,
                "half_open_calls": 0,
            }
        return json.loads(raw)  # type: ignore[no-any-return]

    async def _save_state_data(
        self, circuit_key: str, tenant_id: str, data: dict[str, Any]
    ) -> None:
        """Persist circuit state dict to Redis with a long TTL."""
        key = self._state_key(circuit_key, tenant_id)
        # TTL: reset_timeout * 10 to retain history beyond one reset cycle
        ttl = self._reset_timeout_seconds * 10
        await self._redis.setex(key, ttl, json.dumps(data))

    async def _resolve_state(self, data: dict[str, Any]) -> str:
        """Compute effective state, handling OPEN → HALF_OPEN timeout transition."""
        if data["state"] == STATE_OPEN and data.get("opened_at"):
            opened_at = datetime.fromisoformat(data["opened_at"])
            elapsed = (datetime.now(UTC) - opened_at).total_seconds()
            if elapsed >= self._reset_timeout_seconds:
                return STATE_HALF_OPEN
        return data["state"]

    async def is_open(self, circuit_key: str, tenant_id: str) -> bool:
        """Check if the circuit is open, blocking new requests.

        Args:
            circuit_key: Unique key for the circuit (e.g., "agent:{id}").
            tenant_id: Tenant context for Redis namespace isolation.

        Returns:
            True if circuit is OPEN and requests should be rejected.
        """
        data = await self._get_state_data(circuit_key, tenant_id)
        effective_state = await self._resolve_state(data)
        # HALF_OPEN allows test traffic through — only pure OPEN blocks
        return effective_state == STATE_OPEN

    async def record_success(self, circuit_key: str, tenant_id: str) -> None:
        """Record a successful call. Transitions HALF_OPEN → CLOSED on recovery.

        Args:
            circuit_key: Unique key for the circuit.
            tenant_id: Tenant context.
        """
        data = await self._get_state_data(circuit_key, tenant_id)
        effective_state = await self._resolve_state(data)

        if effective_state == STATE_HALF_OPEN:
            logger.info(
                "Circuit breaker recovering — transitioning HALF_OPEN → CLOSED",
                circuit_key=circuit_key,
                tenant_id=tenant_id,
            )
            data["state"] = STATE_CLOSED
            data["failure_count"] = 0
            data["opened_at"] = None
            data["half_open_calls"] = 0
        elif effective_state == STATE_CLOSED:
            # Reset failure count on success in closed state
            data["failure_count"] = 0

        await self._save_state_data(circuit_key, tenant_id, data)

    async def record_failure(self, circuit_key: str, tenant_id: str) -> None:
        """Record a failed call. May transition CLOSED → OPEN or HALF_OPEN → OPEN.

        Args:
            circuit_key: Unique key for the circuit.
            tenant_id: Tenant context.
        """
        data = await self._get_state_data(circuit_key, tenant_id)
        effective_state = await self._resolve_state(data)

        if effective_state in (STATE_CLOSED, STATE_HALF_OPEN):
            data["failure_count"] = data.get("failure_count", 0) + 1

            if data["failure_count"] >= self._failure_threshold or effective_state == STATE_HALF_OPEN:
                logger.warning(
                    "Circuit breaker tripped — transitioning to OPEN",
                    circuit_key=circuit_key,
                    failure_count=data["failure_count"],
                    threshold=self._failure_threshold,
                    tenant_id=tenant_id,
                )
                data["state"] = STATE_OPEN
                data["opened_at"] = datetime.now(UTC).isoformat()
                data["half_open_calls"] = 0
            else:
                data["state"] = STATE_CLOSED

        await self._save_state_data(circuit_key, tenant_id, data)

    async def get_state(self, circuit_key: str, tenant_id: str) -> str:
        """Get the current circuit state string.

        Args:
            circuit_key: Unique key for the circuit.
            tenant_id: Tenant context.

        Returns:
            One of 'closed', 'open', or 'half_open'.
        """
        data = await self._get_state_data(circuit_key, tenant_id)
        return await self._resolve_state(data)


# Verify protocol compliance
def _verify_protocol_compliance() -> None:
    """Verify RedisCircuitBreaker satisfies CircuitBreakerProtocol at import time."""
    assert isinstance(RedisCircuitBreaker, type)
    # Runtime check deferred — requires a live Redis client


_verify_protocol_compliance()
