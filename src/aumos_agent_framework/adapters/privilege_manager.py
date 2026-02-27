"""Agent privilege level management adapter.

Enforces the 5-level privilege system for agent actions. All agent invocations
must pass through privilege checks before execution. Privilege level 3+
automatically triggers HITL gate evaluation per the CLAUDE.md specification.
"""

from uuid import UUID

from aumos_common.errors import PermissionDeniedError
from aumos_common.observability import get_logger
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_agent_framework.core.models import AgentDefinition

logger = get_logger(__name__)

# Privilege level constants — mirrors services.py for consistency
PRIVILEGE_READ_ONLY = 1
PRIVILEGE_EXECUTE_SAFE = 2
PRIVILEGE_EXECUTE_RISKY = 3
PRIVILEGE_ADMIN = 4
PRIVILEGE_SUPER_ADMIN = 5

PRIVILEGE_NAMES: dict[int, str] = {
    PRIVILEGE_READ_ONLY: "READ_ONLY",
    PRIVILEGE_EXECUTE_SAFE: "EXECUTE_SAFE",
    PRIVILEGE_EXECUTE_RISKY: "EXECUTE_RISKY",
    PRIVILEGE_ADMIN: "ADMIN",
    PRIVILEGE_SUPER_ADMIN: "SUPER_ADMIN",
}

# Privilege levels that require HITL gate evaluation
HITL_REQUIRED_PRIVILEGE_THRESHOLD = PRIVILEGE_EXECUTE_RISKY


class PrivilegeManager:
    """Enforces agent privilege level policies for action gating and HITL triggering.

    The privilege system enforces least-privilege access. Agents are assigned a
    maximum privilege level at registration. Each action declares a required
    privilege level. Calls with insufficient privilege are rejected immediately.

    HITL gate policy:
    - privilege_level >= 3 (EXECUTE_RISKY): always requires HITL evaluation
    - privilege_level >= 4 (ADMIN): requires explicit HITL approval
    - privilege_level == 5 (SUPER_ADMIN): platform-level ops, requires approval chain
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with database session for agent lookups.

        Args:
            session: Async SQLAlchemy session with tenant RLS context set.
        """
        self._session = session

    async def get_effective_privilege(
        self,
        agent_id: UUID,
        tenant_id: str,
        requested_privilege: int,
    ) -> int:
        """Compute the effective privilege level for an agent action.

        The effective level is min(agent.privilege_level, requested_privilege).
        This prevents privilege escalation beyond what the agent is authorized for.

        Args:
            agent_id: Agent requesting the action.
            tenant_id: Tenant context for agent lookup.
            requested_privilege: Privilege level the action requires.

        Returns:
            Effective privilege level (capped at agent's registered level).

        Raises:
            PermissionDeniedError: If agent's privilege_level < requested_privilege.
        """
        from sqlalchemy import select

        stmt = select(AgentDefinition).where(
            AgentDefinition.id == agent_id,
            AgentDefinition.tenant_id == UUID(tenant_id),
            AgentDefinition.status == "active",
        )
        result = await self._session.execute(stmt)
        agent = result.scalar_one_or_none()

        if agent is None:
            raise PermissionDeniedError(
                f"Agent {agent_id} is not active or does not exist in tenant {tenant_id}"
            )

        if agent.privilege_level < requested_privilege:
            agent_level_name = PRIVILEGE_NAMES.get(agent.privilege_level, str(agent.privilege_level))
            required_level_name = PRIVILEGE_NAMES.get(requested_privilege, str(requested_privilege))
            logger.warning(
                "Privilege check failed — agent level insufficient",
                agent_id=str(agent_id),
                agent_privilege=agent.privilege_level,
                agent_privilege_name=agent_level_name,
                required_privilege=requested_privilege,
                required_privilege_name=required_level_name,
                tenant_id=tenant_id,
            )
            raise PermissionDeniedError(
                f"Agent {agent_id} has privilege level {agent_level_name} "
                f"but action requires {required_level_name}"
            )

        effective = min(agent.privilege_level, requested_privilege)
        logger.debug(
            "Privilege check passed",
            agent_id=str(agent_id),
            agent_privilege=agent.privilege_level,
            requested_privilege=requested_privilege,
            effective_privilege=effective,
            tenant_id=tenant_id,
        )
        return effective

    def requires_hitl(self, privilege_level: int) -> bool:
        """Check whether a given privilege level requires HITL gate evaluation.

        Args:
            privilege_level: The effective privilege level for the action.

        Returns:
            True if HITL gate must be evaluated before proceeding.
        """
        return privilege_level >= HITL_REQUIRED_PRIVILEGE_THRESHOLD

    async def assert_min_privilege(
        self,
        agent_id: UUID,
        tenant_id: str,
        min_required: int,
    ) -> None:
        """Assert that an agent has at minimum the required privilege level.

        This is a convenience method for gates that need a simple yes/no check
        without computing the effective level.

        Args:
            agent_id: Agent to check.
            tenant_id: Tenant context.
            min_required: Minimum privilege level required.

        Raises:
            PermissionDeniedError: If agent privilege is insufficient.
        """
        await self.get_effective_privilege(agent_id, tenant_id, min_required)

    def get_privilege_name(self, privilege_level: int) -> str:
        """Get human-readable name for a privilege level.

        Args:
            privilege_level: Numeric privilege level (1-5).

        Returns:
            Human-readable privilege level name.
        """
        return PRIVILEGE_NAMES.get(privilege_level, f"UNKNOWN({privilege_level})")
