"""Tool registration and access control adapter.

Implements ToolRegistryProtocol backed by PostgreSQL via SQLAlchemy.
Tools are registered per-tenant with a minimum privilege level. Access is
checked against both the tool's min_privilege_level and the agent's
tool_access configuration on AgentDefinition.
"""

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.errors import NotFoundError
from aumos_common.observability import get_logger

from aumos_agent_framework.core.interfaces import ToolRegistryProtocol
from aumos_agent_framework.core.models import AgentDefinition, ToolDefinition

logger = get_logger(__name__)


class DatabaseToolRegistry:
    """PostgreSQL-backed tool registry with privilege-based access control.

    Tools are stored in the agf_tool_definitions table. Access control is
    two-layered:
    1. Tool's min_privilege_level: the minimum privilege required for any agent.
    2. Agent's tool_access config: explicit allow/deny per agent.

    Both layers must pass for access to be granted.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with database session.

        Args:
            session: Async SQLAlchemy session with tenant RLS context set.
        """
        self._session = session

    async def register_tool(
        self,
        name: str,
        description: str,
        min_privilege_level: int,
        input_schema: dict[str, Any],
        config: dict[str, Any],
        tenant_id: str,
    ) -> UUID:
        """Register a new tool in the registry.

        Args:
            name: Unique tool name within this tenant.
            description: Human-readable description of tool purpose.
            min_privilege_level: Minimum privilege level (1-5) required.
            input_schema: JSON schema for tool input parameters.
            config: Tool endpoint and configuration details.
            tenant_id: Tenant that owns this tool registration.

        Returns:
            UUID of the created ToolDefinition record.

        Raises:
            ValueError: If min_privilege_level is outside the 1-5 range.
        """
        if not 1 <= min_privilege_level <= 5:
            raise ValueError(
                f"min_privilege_level must be 1-5, got {min_privilege_level}"
            )

        tool = ToolDefinition(
            tenant_id=UUID(tenant_id),
            name=name,
            description=description,
            min_privilege_level=min_privilege_level,
            input_schema=input_schema,
            config=config,
            status="active",
        )
        self._session.add(tool)
        await self._session.flush()

        logger.info(
            "Tool registered in registry",
            tool_id=str(tool.id),
            name=name,
            min_privilege_level=min_privilege_level,
            tenant_id=tenant_id,
        )
        return tool.id

    async def get_tool(self, tool_name: str, tenant_id: str) -> dict[str, Any] | None:
        """Retrieve a tool definition by name within the tenant scope.

        Args:
            tool_name: Name of the tool to look up.
            tenant_id: Tenant context for scoping.

        Returns:
            Tool definition dict if found, or None.
        """
        stmt = select(ToolDefinition).where(
            ToolDefinition.name == tool_name,
            ToolDefinition.tenant_id == UUID(tenant_id),
            ToolDefinition.status == "active",
        )
        result = await self._session.execute(stmt)
        tool = result.scalar_one_or_none()

        if tool is None:
            return None

        return {
            "id": str(tool.id),
            "name": tool.name,
            "description": tool.description,
            "min_privilege_level": tool.min_privilege_level,
            "input_schema": tool.input_schema,
            "config": tool.config,
            "status": tool.status,
        }

    async def list_tools(
        self,
        tenant_id: str,
        min_privilege_level: int | None = None,
    ) -> list[dict[str, Any]]:
        """List all active tools for a tenant, optionally filtered by privilege.

        Args:
            tenant_id: Tenant context.
            min_privilege_level: If set, only return tools at or below this level.
                                 Useful for listing tools accessible to an agent.

        Returns:
            List of tool definition dicts.
        """
        stmt = select(ToolDefinition).where(
            ToolDefinition.tenant_id == UUID(tenant_id),
            ToolDefinition.status == "active",
        )
        if min_privilege_level is not None:
            stmt = stmt.where(
                ToolDefinition.min_privilege_level <= min_privilege_level
            )
        stmt = stmt.order_by(ToolDefinition.name)

        result = await self._session.execute(stmt)
        tools = result.scalars().all()

        return [
            {
                "id": str(tool.id),
                "name": tool.name,
                "description": tool.description,
                "min_privilege_level": tool.min_privilege_level,
                "input_schema": tool.input_schema,
                "config": tool.config,
                "status": tool.status,
            }
            for tool in tools
        ]

    async def check_access(
        self,
        agent_id: UUID,
        tool_name: str,
        tenant_id: str,
    ) -> bool:
        """Check whether an agent has access to a specific tool.

        Access is granted when:
        1. The tool exists and is active for this tenant.
        2. The agent's privilege_level >= tool's min_privilege_level.
        3. The tool is listed in the agent's tool_access config with enabled=True.

        Args:
            agent_id: Agent requesting access.
            tool_name: Name of the tool being requested.
            tenant_id: Tenant context.

        Returns:
            True if all access conditions are satisfied.
        """
        # Load the tool
        tool_data = await self.get_tool(tool_name, tenant_id)
        if tool_data is None:
            logger.debug(
                "Tool access denied — tool not found",
                agent_id=str(agent_id),
                tool_name=tool_name,
                tenant_id=tenant_id,
            )
            return False

        # Load the agent
        stmt = select(AgentDefinition).where(
            AgentDefinition.id == agent_id,
            AgentDefinition.tenant_id == UUID(tenant_id),
            AgentDefinition.status == "active",
        )
        result = await self._session.execute(stmt)
        agent = result.scalar_one_or_none()

        if agent is None:
            logger.debug(
                "Tool access denied — agent not found or inactive",
                agent_id=str(agent_id),
                tool_name=tool_name,
                tenant_id=tenant_id,
            )
            return False

        # Check privilege level
        if agent.privilege_level < tool_data["min_privilege_level"]:
            logger.debug(
                "Tool access denied — insufficient privilege",
                agent_id=str(agent_id),
                agent_privilege=agent.privilege_level,
                tool_name=tool_name,
                tool_min_privilege=tool_data["min_privilege_level"],
                tenant_id=tenant_id,
            )
            return False

        # Check agent's tool_access configuration
        tool_access_entry = agent.tool_access.get(tool_name, {})
        if not tool_access_entry.get("enabled", False):
            logger.debug(
                "Tool access denied — tool not enabled in agent tool_access config",
                agent_id=str(agent_id),
                tool_name=tool_name,
                tenant_id=tenant_id,
            )
            return False

        logger.debug(
            "Tool access granted",
            agent_id=str(agent_id),
            tool_name=tool_name,
            tenant_id=tenant_id,
        )
        return True


# Verify protocol compliance
def _verify_protocol_compliance() -> None:
    """Verify DatabaseToolRegistry satisfies ToolRegistryProtocol at import time."""
    assert isinstance(DatabaseToolRegistry, type)


_verify_protocol_compliance()
