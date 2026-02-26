"""Protocol interfaces for aumos-agent-framework adapters.

All adapters must implement these protocols to ensure the core services
remain framework-independent and testable.
"""

from typing import Any, Protocol, runtime_checkable
from uuid import UUID


@runtime_checkable
class WorkflowEngineProtocol(Protocol):
    """Executes graph-based workflows using LangGraph StateGraph."""

    async def execute_workflow(
        self,
        workflow_definition: dict[str, Any],
        input_data: dict[str, Any],
        execution_id: str,
        tenant_id: str,
    ) -> dict[str, Any]:
        """Execute a workflow graph and return the output state.

        Args:
            workflow_definition: Serialized LangGraph graph definition.
            input_data: Initial state to inject into the graph.
            execution_id: Unique execution ID for tracking.
            tenant_id: Tenant context for isolation.

        Returns:
            Final state dict from the completed workflow graph.
        """
        ...

    async def get_graph_nodes(self, workflow_definition: dict[str, Any]) -> list[str]:
        """Return all node names in the workflow graph.

        Args:
            workflow_definition: Serialized LangGraph graph definition.

        Returns:
            List of node name strings.
        """
        ...


@runtime_checkable
class AgentExecutorProtocol(Protocol):
    """Executes individual agent actions within a workflow node."""

    async def invoke_agent(
        self,
        agent_id: UUID,
        input_data: dict[str, Any],
        session_id: str,
        tenant_id: str,
        privilege_level: int,
    ) -> dict[str, Any]:
        """Invoke an agent and return its output.

        Args:
            agent_id: ID of the registered agent to invoke.
            input_data: Input data for the agent.
            session_id: Session ID for memory context.
            tenant_id: Tenant context for isolation.
            privilege_level: Effective privilege level for this invocation.

        Returns:
            Agent output as a dict.
        """
        ...


@runtime_checkable
class HITLGateProtocol(Protocol):
    """Manages human-in-the-loop approval gate flow."""

    async def trigger_gate(
        self,
        execution_id: UUID,
        gate_name: str,
        agent_id: UUID,
        action_description: str,
        context_data: dict[str, Any],
        tenant_id: str,
    ) -> UUID:
        """Create a pending HITL approval and pause workflow execution.

        Args:
            execution_id: ID of the workflow execution being paused.
            gate_name: Name of the HITL gate as defined in workflow definition.
            agent_id: Agent that triggered the gate.
            action_description: Human-readable description of what needs approval.
            context_data: Additional context for the approver.
            tenant_id: Tenant context.

        Returns:
            UUID of the created HITLApproval record.
        """
        ...

    async def approve(
        self,
        approval_id: UUID,
        decided_by: UUID,
        notes: str | None,
        tenant_id: str,
    ) -> bool:
        """Approve a pending HITL gate and resume workflow.

        Returns:
            True if workflow was successfully resumed.
        """
        ...

    async def reject(
        self,
        approval_id: UUID,
        decided_by: UUID,
        notes: str | None,
        tenant_id: str,
    ) -> bool:
        """Reject a pending HITL gate and fail the workflow.

        Returns:
            True if workflow was successfully cancelled.
        """
        ...


@runtime_checkable
class CircuitBreakerProtocol(Protocol):
    """Per-agent and per-workflow circuit breaker for cascading failure containment."""

    async def is_open(self, circuit_key: str, tenant_id: str) -> bool:
        """Check if the circuit is open (blocking requests).

        Args:
            circuit_key: Unique key identifying the circuit (e.g., "agent:{id}" or "workflow:{id}")
            tenant_id: Tenant context for isolation.

        Returns:
            True if circuit is OPEN and requests should be rejected.
        """
        ...

    async def record_success(self, circuit_key: str, tenant_id: str) -> None:
        """Record a successful call — may transition HALF_OPEN to CLOSED."""
        ...

    async def record_failure(self, circuit_key: str, tenant_id: str) -> None:
        """Record a failed call — may transition CLOSED to OPEN."""
        ...

    async def get_state(self, circuit_key: str, tenant_id: str) -> str:
        """Get current circuit state: 'closed', 'open', or 'half_open'."""
        ...


@runtime_checkable
class SessionIsolatorProtocol(Protocol):
    """Per-tenant agent session isolation with Redis-backed state."""

    async def get_session(
        self,
        agent_id: UUID,
        tenant_id: str,
        execution_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Retrieve agent session data.

        Args:
            agent_id: Agent whose session to retrieve.
            tenant_id: Tenant context (enforces namespace isolation).
            execution_id: Optional workflow execution for scoped sessions.

        Returns:
            Session data dict, empty dict if no session exists.
        """
        ...

    async def update_session(
        self,
        agent_id: UUID,
        tenant_id: str,
        updates: dict[str, Any],
        execution_id: UUID | None = None,
    ) -> None:
        """Update agent session data with partial updates."""
        ...

    async def clear_session(
        self,
        agent_id: UUID,
        tenant_id: str,
        execution_id: UUID | None = None,
    ) -> None:
        """Clear agent session data."""
        ...

    async def get_shared_session(
        self,
        execution_id: UUID,
        tenant_id: str,
    ) -> dict[str, Any]:
        """Retrieve shared session data for all agents in a workflow execution."""
        ...

    async def update_shared_session(
        self,
        execution_id: UUID,
        tenant_id: str,
        updates: dict[str, Any],
    ) -> None:
        """Update shared session data accessible to all agents in an execution."""
        ...


@runtime_checkable
class ToolRegistryProtocol(Protocol):
    """Registry for tools available to agents."""

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

        Returns:
            UUID of the created ToolDefinition.
        """
        ...

    async def get_tool(self, tool_name: str, tenant_id: str) -> dict[str, Any] | None:
        """Retrieve a tool definition by name.

        Returns:
            Tool definition dict, or None if not found.
        """
        ...

    async def list_tools(
        self,
        tenant_id: str,
        min_privilege_level: int | None = None,
    ) -> list[dict[str, Any]]:
        """List all available tools, optionally filtered by privilege level."""
        ...

    async def check_access(
        self,
        agent_id: UUID,
        tool_name: str,
        tenant_id: str,
    ) -> bool:
        """Check whether an agent has access to a specific tool.

        Returns:
            True if access is allowed.
        """
        ...


@runtime_checkable
class DurableExecutorProtocol(Protocol):
    """Durable workflow execution via Temporal for failure-resilient processing."""

    async def start_workflow(
        self,
        workflow_id: str,
        workflow_definition: dict[str, Any],
        input_data: dict[str, Any],
        tenant_id: str,
    ) -> str:
        """Start a durable workflow execution via Temporal.

        Returns:
            Temporal workflow run ID.
        """
        ...

    async def cancel_workflow(self, workflow_id: str, tenant_id: str) -> bool:
        """Cancel a running durable workflow.

        Returns:
            True if successfully cancelled.
        """
        ...

    async def get_workflow_status(self, workflow_id: str, tenant_id: str) -> dict[str, Any]:
        """Get current status of a durable workflow execution."""
        ...
