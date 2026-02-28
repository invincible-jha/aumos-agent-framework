"""Auto-discovering built-in tool registry for the AumOS agent framework.

Scans the adapters/tools/ subdirectory to auto-register all tool implementations
that satisfy AumOSToolProtocol. Provides a unified interface for listing, configuring,
and testing tools.
"""

from typing import Any

from aumos_common.observability import get_logger

from aumos_agent_framework.core.interfaces import AumOSToolProtocol, ToolInputSchema, ToolOutputSchema

logger = get_logger(__name__)


class BuiltinToolRegistry:
    """Registry of all pre-built AumOS tools.

    Discovers and registers tools from the adapters/tools/ subdirectories.
    Provides access control enforcement and per-tenant configuration storage.

    Attributes:
        _tools: Mapping from tool_id to tool instance.
    """

    def __init__(self) -> None:
        """Initialize the registry and auto-discover all built-in tools."""
        self._tools: dict[str, Any] = {}
        self._discover_tools()

    def _discover_tools(self) -> None:
        """Auto-discover all tool modules in the adapters/tools/ subdirectories.

        Each module that exposes a top-level TOOL instance satisfying
        AumOSToolProtocol is registered automatically.
        """
        import importlib
        import pkgutil
        import aumos_agent_framework.adapters.tools as tools_package

        for finder, module_name, is_pkg in pkgutil.walk_packages(
            path=tools_package.__path__,
            prefix=tools_package.__name__ + ".",
        ):
            if is_pkg or module_name.endswith(("__init__", "registry")):
                continue
            try:
                module = importlib.import_module(module_name)
                tool_instance = getattr(module, "TOOL", None)
                if tool_instance is not None and isinstance(tool_instance, AumOSToolProtocol):
                    self._tools[tool_instance.tool_id] = tool_instance
                    logger.debug("Registered built-in tool", tool_id=tool_instance.tool_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to load tool module",
                    module=module_name,
                    error=str(exc),
                )

    def list_tools(
        self,
        category: str | None = None,
        max_privilege_level: int | None = None,
    ) -> list[dict[str, Any]]:
        """List all registered built-in tools with metadata.

        Args:
            category: Optional category filter ("web", "data", "communication", etc.)
            max_privilege_level: If set, only return tools requiring <= this privilege level.

        Returns:
            List of tool metadata dicts suitable for API responses.
        """
        tools = list(self._tools.values())

        if category is not None:
            tools = [t for t in tools if t.category == category]

        if max_privilege_level is not None:
            tools = [t for t in tools if t.privilege_level <= max_privilege_level]

        return [
            {
                "tool_id": t.tool_id,
                "display_name": t.display_name,
                "category": t.category,
                "description": t.description,
                "privilege_level": t.privilege_level,
                "input_schema": t.input_schema.model_json_schema(),
                "output_schema": t.output_schema.model_json_schema(),
            }
            for t in sorted(tools, key=lambda x: (x.category, x.tool_id))
        ]

    def get_tool(self, tool_id: str) -> Any | None:
        """Retrieve a tool instance by ID.

        Args:
            tool_id: Unique tool identifier.

        Returns:
            Tool instance if found, else None.
        """
        return self._tools.get(tool_id)

    async def execute_tool(
        self,
        tool_id: str,
        input_data: dict[str, Any],
        tenant_id: str,
        agent_privilege_level: int,
        config: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute a built-in tool with privilege enforcement.

        Args:
            tool_id: Unique tool identifier.
            input_data: Raw input dict, validated against tool's input_schema.
            tenant_id: Tenant context for rate limiting.
            agent_privilege_level: Calling agent's privilege level (1-5).
            config: Optional per-tenant config overrides (API keys, endpoints).

        Returns:
            Tool output serialized as a dict.

        Raises:
            ValueError: If tool not found or privilege check fails.
        """
        tool = self.get_tool(tool_id)
        if tool is None:
            raise ValueError(f"Tool '{tool_id}' is not registered in the built-in registry")

        if agent_privilege_level < tool.privilege_level:
            raise ValueError(
                f"Agent privilege level {agent_privilege_level} is insufficient for tool "
                f"'{tool_id}' which requires privilege level {tool.privilege_level}"
            )

        validated_input = tool.input_schema.model_validate(input_data)
        result = await tool.execute(
            input_data=validated_input,
            tenant_id=tenant_id,
            config=config or {},
        )

        logger.info(
            "Built-in tool executed",
            tool_id=tool_id,
            tenant_id=tenant_id,
            agent_privilege_level=agent_privilege_level,
        )

        return result.model_dump()
