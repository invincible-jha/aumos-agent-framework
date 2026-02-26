"""LangGraph graph-based workflow execution adapter.

Implements WorkflowEngineProtocol using LangGraph StateGraph for graph-based
multi-agent workflow orchestration with typed state and conditional routing.
"""

from typing import Any, Callable
from uuid import UUID

from aumos_common.observability import get_logger

from aumos_agent_framework.core.interfaces import WorkflowEngineProtocol

logger = get_logger(__name__)


class LangGraphEngine:
    """LangGraph-based workflow engine.

    Executes workflows defined as LangGraph StateGraph definitions.
    Each workflow definition includes nodes (callables), edges (transitions),
    and optionally conditional edges (routing functions).

    Graph definition format:
    {
        "state_schema": {"field_name": "type_hint_string"},
        "nodes": {
            "node_name": "callable_ref"  // resolved via node_registry
        },
        "edges": [
            {"from": "START", "to": "first_node"},
            {"from": "node_a", "to": "node_b"}
        ],
        "conditional_edges": [
            {
                "from": "router_node",
                "condition": "callable_ref",
                "mapping": {"route_a": "node_a", "route_b": "node_b"}
            }
        ],
        "entry_point": "first_node"
    }
    """

    def __init__(self, node_registry: dict[str, Callable[..., Any]] | None = None) -> None:
        """Initialize with optional node callable registry.

        Args:
            node_registry: Maps callable reference strings to Python callables.
                           Callables should be async functions with signature:
                           async def node_fn(state: dict) -> dict
        """
        self._node_registry: dict[str, Callable[..., Any]] = node_registry or {}

    def register_node(self, name: str, callable_fn: Callable[..., Any]) -> None:
        """Register a node callable in the registry.

        Args:
            name: The callable reference string used in workflow definitions.
            callable_fn: Async callable that takes state dict and returns state dict.
        """
        self._node_registry[name] = callable_fn
        logger.debug("Node registered in LangGraph engine", node_name=name)

    async def execute_workflow(
        self,
        workflow_definition: dict[str, Any],
        input_data: dict[str, Any],
        execution_id: str,
        tenant_id: str,
    ) -> dict[str, Any]:
        """Execute a workflow graph and return the final state.

        Args:
            workflow_definition: Serialized LangGraph graph definition.
            input_data: Initial state for the graph.
            execution_id: Unique execution ID for tracking/checkpointing.
            tenant_id: Tenant context for isolation.

        Returns:
            Final state dict after graph execution completes.

        Raises:
            ValueError: If workflow definition is malformed.
            RuntimeError: If graph execution fails.
        """
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError("langgraph package is required for LangGraphEngine") from exc

        nodes = workflow_definition.get("nodes", {})
        edges = workflow_definition.get("edges", [])
        conditional_edges = workflow_definition.get("conditional_edges", [])
        entry_point = workflow_definition.get("entry_point")

        if not nodes:
            raise ValueError("Workflow definition must have at least one node")
        if not entry_point:
            raise ValueError("Workflow definition must specify an entry_point")

        logger.info(
            "Building LangGraph workflow",
            execution_id=execution_id,
            node_count=len(nodes),
            tenant_id=tenant_id,
        )

        # Build StateGraph — use dict as state type for flexibility
        graph: StateGraph = StateGraph(dict)

        # Add nodes
        for node_name, callable_ref in nodes.items():
            callable_fn = self._resolve_callable(callable_ref)
            graph.add_node(node_name, callable_fn)

        # Add edges
        for edge in edges:
            from_node = edge.get("from", "")
            to_node = edge.get("to", "")
            if from_node == "START":
                graph.add_edge(START, to_node)
            elif to_node == "END":
                graph.add_edge(from_node, END)
            else:
                graph.add_edge(from_node, to_node)

        # Add conditional edges
        for cond_edge in conditional_edges:
            from_node = cond_edge["from"]
            condition_ref = cond_edge["condition"]
            mapping = cond_edge.get("mapping", {})
            condition_fn = self._resolve_callable(condition_ref)
            graph.add_conditional_edges(from_node, condition_fn, mapping)

        # Set entry point
        graph.set_entry_point(entry_point)

        # Compile and execute
        compiled = graph.compile()

        logger.info(
            "Executing LangGraph workflow",
            execution_id=execution_id,
            entry_point=entry_point,
            tenant_id=tenant_id,
        )

        # Inject execution context into initial state
        initial_state = {
            **input_data,
            "__execution_id__": execution_id,
            "__tenant_id__": tenant_id,
        }

        final_state = await compiled.ainvoke(initial_state)

        logger.info(
            "LangGraph workflow completed",
            execution_id=execution_id,
            tenant_id=tenant_id,
        )
        return dict(final_state)

    async def get_graph_nodes(self, workflow_definition: dict[str, Any]) -> list[str]:
        """Return all node names defined in the workflow graph.

        Args:
            workflow_definition: Serialized LangGraph graph definition.

        Returns:
            List of node name strings.
        """
        return list(workflow_definition.get("nodes", {}).keys())

    def _resolve_callable(self, callable_ref: str | Callable[..., Any]) -> Callable[..., Any]:
        """Resolve a callable reference to an actual Python callable.

        Args:
            callable_ref: Either a string key in the node registry or a direct callable.

        Returns:
            Callable function.

        Raises:
            ValueError: If string reference is not found in registry.
        """
        if callable(callable_ref):
            return callable_ref  # type: ignore[return-value]

        if isinstance(callable_ref, str):
            if callable_ref not in self._node_registry:
                raise ValueError(
                    f"Node callable '{callable_ref}' not found in registry. "
                    f"Available: {list(self._node_registry.keys())}"
                )
            return self._node_registry[callable_ref]

        raise ValueError(f"Invalid callable reference: {callable_ref!r}")


# Verify protocol compliance
def _verify_protocol_compliance() -> None:
    """Verify LangGraphEngine satisfies WorkflowEngineProtocol at import time."""
    assert isinstance(LangGraphEngine(), WorkflowEngineProtocol)


_verify_protocol_compliance()
