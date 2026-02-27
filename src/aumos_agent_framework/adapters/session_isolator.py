"""Redis-backed session isolation adapter for per-tenant agent memory.

Implements SessionIsolatorProtocol with strict namespace isolation between tenants.
Private sessions are scoped to (tenant, agent). Shared sessions are scoped to
(tenant, execution). TTL is configurable via settings.
"""

import json
from typing import Any
from uuid import UUID

from redis.asyncio import Redis

from aumos_common.observability import get_logger

from aumos_agent_framework.core.interfaces import SessionIsolatorProtocol

logger = get_logger(__name__)

_DEFAULT_SESSION_TTL_SECONDS = 86400  # 24 hours


class RedisSessionIsolator:
    """Redis-backed session isolator enforcing per-tenant namespace isolation.

    Key schema:
    - Private session:  session:{tenant_id}:agent:{agent_id}
    - Scoped private:   session:{tenant_id}:agent:{agent_id}:exec:{execution_id}
    - Shared session:   session:{tenant_id}:shared:exec:{execution_id}

    The tenant_id prefix in every key guarantees cross-tenant data isolation at the
    Redis layer even if application-level RLS is bypassed.
    """

    def __init__(
        self,
        redis_client: Redis,  # type: ignore[type-arg]
        session_ttl_seconds: int = _DEFAULT_SESSION_TTL_SECONDS,
    ) -> None:
        """Initialize with Redis client and TTL configuration.

        Args:
            redis_client: Async Redis client instance.
            session_ttl_seconds: Time-to-live for session keys in seconds.
        """
        self._redis = redis_client
        self._ttl = session_ttl_seconds

    def _private_key(self, agent_id: UUID, tenant_id: str, execution_id: UUID | None) -> str:
        """Build Redis key for private agent session."""
        if execution_id is not None:
            return f"session:{tenant_id}:agent:{agent_id}:exec:{execution_id}"
        return f"session:{tenant_id}:agent:{agent_id}"

    def _shared_key(self, execution_id: UUID, tenant_id: str) -> str:
        """Build Redis key for shared execution session."""
        return f"session:{tenant_id}:shared:exec:{execution_id}"

    async def _redis_get(self, key: str) -> dict[str, Any]:
        """Retrieve and deserialize a JSON session from Redis."""
        raw = await self._redis.get(key)
        if raw is None:
            return {}
        return json.loads(raw)  # type: ignore[no-any-return]

    async def _redis_merge(self, key: str, updates: dict[str, Any]) -> None:
        """Merge updates into existing session data and persist with TTL refresh."""
        existing = await self._redis_get(key)
        existing.update(updates)
        await self._redis.setex(key, self._ttl, json.dumps(existing))

    async def get_session(
        self,
        agent_id: UUID,
        tenant_id: str,
        execution_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Retrieve agent private session data.

        Args:
            agent_id: Agent whose session to retrieve.
            tenant_id: Tenant context — enforces namespace isolation.
            execution_id: Optional workflow execution for scoped sessions.

        Returns:
            Session data dict, or empty dict if no session exists.
        """
        key = self._private_key(agent_id, tenant_id, execution_id)
        data = await self._redis_get(key)

        logger.debug(
            "Agent session retrieved",
            agent_id=str(agent_id),
            execution_id=str(execution_id) if execution_id else None,
            tenant_id=tenant_id,
            key_count=len(data),
        )
        return data

    async def update_session(
        self,
        agent_id: UUID,
        tenant_id: str,
        updates: dict[str, Any],
        execution_id: UUID | None = None,
    ) -> None:
        """Update agent private session with partial updates.

        Args:
            agent_id: Agent whose session to update.
            tenant_id: Tenant context.
            updates: Key-value pairs to merge into existing session.
            execution_id: Optional execution scope.
        """
        key = self._private_key(agent_id, tenant_id, execution_id)
        await self._redis_merge(key, updates)

        logger.debug(
            "Agent session updated",
            agent_id=str(agent_id),
            execution_id=str(execution_id) if execution_id else None,
            tenant_id=tenant_id,
            updated_keys=list(updates.keys()),
        )

    async def clear_session(
        self,
        agent_id: UUID,
        tenant_id: str,
        execution_id: UUID | None = None,
    ) -> None:
        """Clear agent private session data.

        Args:
            agent_id: Agent whose session to clear.
            tenant_id: Tenant context.
            execution_id: Optional execution scope.
        """
        key = self._private_key(agent_id, tenant_id, execution_id)
        await self._redis.delete(key)

        logger.info(
            "Agent session cleared",
            agent_id=str(agent_id),
            execution_id=str(execution_id) if execution_id else None,
            tenant_id=tenant_id,
        )

    async def get_shared_session(
        self,
        execution_id: UUID,
        tenant_id: str,
    ) -> dict[str, Any]:
        """Retrieve shared session data for all agents in a workflow execution.

        Args:
            execution_id: Workflow execution whose shared session to retrieve.
            tenant_id: Tenant context — enforces namespace isolation.

        Returns:
            Shared session data dict, or empty dict if none exists.
        """
        key = self._shared_key(execution_id, tenant_id)
        data = await self._redis_get(key)

        logger.debug(
            "Shared session retrieved",
            execution_id=str(execution_id),
            tenant_id=tenant_id,
            key_count=len(data),
        )
        return data

    async def update_shared_session(
        self,
        execution_id: UUID,
        tenant_id: str,
        updates: dict[str, Any],
    ) -> None:
        """Update shared session data for all agents in a workflow execution.

        Args:
            execution_id: Workflow execution whose shared session to update.
            tenant_id: Tenant context.
            updates: Key-value pairs to merge into existing shared session.
        """
        key = self._shared_key(execution_id, tenant_id)
        await self._redis_merge(key, updates)

        logger.debug(
            "Shared session updated",
            execution_id=str(execution_id),
            tenant_id=tenant_id,
            updated_keys=list(updates.keys()),
        )
