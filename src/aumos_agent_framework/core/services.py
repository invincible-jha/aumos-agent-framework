"""Business logic services for aumos-agent-framework.

All services operate on tenant-scoped data and enforce privilege, circuit breaker,
and HITL gate policies. No framework dependencies — adapters are injected via interfaces.
"""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.errors import NotFoundError, PermissionDeniedError
from aumos_common.observability import get_logger

from aumos_agent_framework.core.interfaces import (
    CircuitBreakerProtocol,
    DurableExecutorProtocol,
    HITLGateProtocol,
    SessionIsolatorProtocol,
    ToolRegistryProtocol,
    WorkflowEngineProtocol,
)
from aumos_agent_framework.core.models import (
    AgentDefinition,
    AgentSession,
    HITLApproval,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowExecution,
)

logger = get_logger(__name__)

# Privilege level constants
PRIVILEGE_READ_ONLY = 1
PRIVILEGE_EXECUTE_SAFE = 2
PRIVILEGE_EXECUTE_RISKY = 3
PRIVILEGE_ADMIN = 4
PRIVILEGE_SUPER_ADMIN = 5

HITL_REQUIRED_PRIVILEGE_THRESHOLD = PRIVILEGE_EXECUTE_RISKY


class AgentRegistryService:
    """Manages agent registration, capabilities, privilege levels, and tool access.

    Agents are the fundamental execution units. Each agent has a privilege level
    that controls what actions it can perform and whether HITL gates are triggered.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with database session.

        Args:
            session: Async SQLAlchemy session with tenant RLS context set.
        """
        self._session = session

    async def register_agent(
        self,
        tenant_id: str,
        name: str,
        description: str | None,
        capabilities: dict[str, Any],
        privilege_level: int,
        tool_access: dict[str, Any],
        resource_limits: dict[str, Any],
    ) -> AgentDefinition:
        """Register a new agent definition.

        Args:
            tenant_id: Tenant that owns this agent.
            name: Human-readable agent name.
            description: Optional description of agent purpose.
            capabilities: Dict of capability tags the agent can perform.
            privilege_level: 1-5 privilege level (see constants above).
            tool_access: Dict mapping tool names to access config.
            resource_limits: Dict of resource consumption limits.

        Returns:
            Created AgentDefinition instance.

        Raises:
            ValueError: If privilege_level is out of range 1-5.
        """
        if not 1 <= privilege_level <= 5:
            raise ValueError(f"privilege_level must be 1-5, got {privilege_level}")

        agent = AgentDefinition(
            tenant_id=uuid.UUID(tenant_id),
            name=name,
            description=description,
            capabilities=capabilities,
            privilege_level=privilege_level,
            tool_access=tool_access,
            resource_limits=resource_limits,
            status="active",
        )
        self._session.add(agent)
        await self._session.flush()

        logger.info(
            "Agent registered",
            agent_id=str(agent.id),
            name=name,
            privilege_level=privilege_level,
            tenant_id=tenant_id,
        )
        return agent

    async def get_agent(self, agent_id: uuid.UUID, tenant_id: str) -> AgentDefinition:
        """Retrieve an agent by ID within the tenant scope.

        Raises:
            NotFoundError: If agent does not exist for this tenant.
        """
        from sqlalchemy import select

        stmt = select(AgentDefinition).where(
            AgentDefinition.id == agent_id,
            AgentDefinition.tenant_id == uuid.UUID(tenant_id),
            AgentDefinition.status != "retired",
        )
        result = await self._session.execute(stmt)
        agent = result.scalar_one_or_none()

        if agent is None:
            raise NotFoundError(f"Agent {agent_id} not found")

        return agent

    async def list_agents(
        self,
        tenant_id: str,
        status: str | None = None,
        capability: str | None = None,
    ) -> list[AgentDefinition]:
        """List all agents for a tenant with optional filters.

        Args:
            tenant_id: Tenant scope.
            status: Optional status filter (active/suspended/retired).
            capability: Optional capability filter.

        Returns:
            List of AgentDefinition instances.
        """
        from sqlalchemy import select

        stmt = select(AgentDefinition).where(
            AgentDefinition.tenant_id == uuid.UUID(tenant_id)
        )
        if status:
            stmt = stmt.where(AgentDefinition.status == status)
        if capability:
            stmt = stmt.where(AgentDefinition.capabilities[capability].isnot(None))

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def update_tool_access(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        tool_access: dict[str, Any],
    ) -> AgentDefinition:
        """Update the tool access configuration for an agent.

        Args:
            agent_id: Agent to update.
            tenant_id: Tenant scope.
            tool_access: New tool access configuration.

        Returns:
            Updated AgentDefinition instance.
        """
        agent = await self.get_agent(agent_id, tenant_id)
        agent.tool_access = tool_access
        await self._session.flush()

        logger.info(
            "Agent tool access updated",
            agent_id=str(agent_id),
            tools=list(tool_access.keys()),
            tenant_id=tenant_id,
        )
        return agent

    async def suspend_agent(self, agent_id: uuid.UUID, tenant_id: str) -> AgentDefinition:
        """Suspend an agent — prevents new workflow assignments."""
        agent = await self.get_agent(agent_id, tenant_id)
        agent.status = "suspended"
        await self._session.flush()
        logger.info("Agent suspended", agent_id=str(agent_id), tenant_id=tenant_id)
        return agent


class WorkflowService:
    """Creates and manages workflow definitions using LangGraph graph DSL."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with database session."""
        self._session = session

    async def create_workflow(
        self,
        tenant_id: str,
        name: str,
        graph_definition: dict[str, Any],
        agents: dict[str, Any],
        hitl_gates: dict[str, Any],
        circuit_breaker_config: dict[str, Any],
    ) -> WorkflowDefinition:
        """Create a new workflow definition.

        Args:
            tenant_id: Tenant that owns this workflow.
            name: Human-readable workflow name.
            graph_definition: LangGraph StateGraph definition (nodes, edges, state schema).
            agents: Maps node names to agent IDs: {"node_name": "agent-uuid"}.
            hitl_gates: HITL gate configs keyed by gate name.
            circuit_breaker_config: Circuit breaker settings for this workflow.

        Returns:
            Created WorkflowDefinition instance.
        """
        workflow = WorkflowDefinition(
            tenant_id=uuid.UUID(tenant_id),
            name=name,
            graph_definition=graph_definition,
            agents=agents,
            hitl_gates=hitl_gates,
            circuit_breaker_config=circuit_breaker_config,
        )
        self._session.add(workflow)
        await self._session.flush()

        logger.info(
            "Workflow definition created",
            workflow_id=str(workflow.id),
            name=name,
            node_count=len(graph_definition.get("nodes", {})),
            tenant_id=tenant_id,
        )
        return workflow

    async def get_workflow(self, workflow_id: uuid.UUID, tenant_id: str) -> WorkflowDefinition:
        """Retrieve a workflow definition.

        Raises:
            NotFoundError: If workflow does not exist for this tenant.
        """
        from sqlalchemy import select

        stmt = select(WorkflowDefinition).where(
            WorkflowDefinition.id == workflow_id,
            WorkflowDefinition.tenant_id == uuid.UUID(tenant_id),
        )
        result = await self._session.execute(stmt)
        workflow = result.scalar_one_or_none()

        if workflow is None:
            raise NotFoundError(f"Workflow {workflow_id} not found")

        return workflow

    async def list_workflows(self, tenant_id: str) -> list[WorkflowDefinition]:
        """List all workflow definitions for a tenant."""
        from sqlalchemy import select

        stmt = select(WorkflowDefinition).where(
            WorkflowDefinition.tenant_id == uuid.UUID(tenant_id)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())


class ExecutionService:
    """Orchestrates workflow execution — node transitions, state management, lifecycle."""

    def __init__(
        self,
        session: AsyncSession,
        workflow_engine: WorkflowEngineProtocol,
        durable_executor: DurableExecutorProtocol,
        circuit_breaker: CircuitBreakerProtocol,
        hitl_gate: HITLGateProtocol,
        session_isolator: SessionIsolatorProtocol,
    ) -> None:
        """Initialize with all required adapters.

        Args:
            session: Async SQLAlchemy session.
            workflow_engine: LangGraph workflow engine adapter.
            durable_executor: Temporal durable execution adapter.
            circuit_breaker: Circuit breaker adapter.
            hitl_gate: HITL gate adapter.
            session_isolator: Session isolation adapter.
        """
        self._session = session
        self._workflow_engine = workflow_engine
        self._durable_executor = durable_executor
        self._circuit_breaker = circuit_breaker
        self._hitl_gate = hitl_gate
        self._session_isolator = session_isolator

    async def start_execution(
        self,
        workflow_id: uuid.UUID,
        tenant_id: str,
        input_data: dict[str, Any],
    ) -> WorkflowExecution:
        """Start a new workflow execution.

        Checks workflow circuit breaker before starting. Delegates to Temporal
        for durable execution management. Creates execution record.

        Args:
            workflow_id: ID of the workflow definition to execute.
            tenant_id: Tenant context.
            input_data: Initial input state for the workflow.

        Returns:
            Created WorkflowExecution record in 'running' state.

        Raises:
            NotFoundError: If workflow definition does not exist.
            PermissionDeniedError: If workflow circuit breaker is OPEN.
        """
        from aumos_agent_framework.core.services import WorkflowService

        workflow_service = WorkflowService(self._session)
        workflow = await workflow_service.get_workflow(workflow_id, tenant_id)

        # Check circuit breaker before starting
        circuit_key = f"workflow:{workflow_id}"
        if await self._circuit_breaker.is_open(circuit_key, tenant_id):
            circuit_state = await self._circuit_breaker.get_state(circuit_key, tenant_id)
            logger.warning(
                "Workflow circuit breaker is open — execution rejected",
                workflow_id=str(workflow_id),
                circuit_state=circuit_state,
                tenant_id=tenant_id,
            )
            raise PermissionDeniedError(
                f"Workflow {workflow_id} circuit breaker is {circuit_state} — execution blocked"
            )

        # Create execution record
        execution = WorkflowExecution(
            tenant_id=uuid.UUID(tenant_id),
            workflow_id=workflow_id,
            status="running",
            input_data=input_data,
            execution_history=[
                {
                    "event": "execution_started",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "input_data": input_data,
                }
            ],
        )
        self._session.add(execution)
        await self._session.flush()

        # Start durable execution via Temporal
        try:
            temporal_run_id = await self._durable_executor.start_workflow(
                workflow_id=str(execution.id),
                workflow_definition=workflow.graph_definition,
                input_data=input_data,
                tenant_id=tenant_id,
            )
            execution.temporal_workflow_id = temporal_run_id
            await self._session.flush()
        except Exception as exc:
            await self._circuit_breaker.record_failure(circuit_key, tenant_id)
            execution.status = "failed"
            execution.error_details = {"error": str(exc), "stage": "temporal_start"}
            execution.completed_at = datetime.now(UTC)
            await self._session.flush()
            logger.error(
                "Failed to start Temporal workflow",
                execution_id=str(execution.id),
                error=str(exc),
                tenant_id=tenant_id,
            )
            raise

        logger.info(
            "Workflow execution started",
            execution_id=str(execution.id),
            workflow_id=str(workflow_id),
            temporal_run_id=temporal_run_id,
            tenant_id=tenant_id,
        )
        return execution

    async def cancel_execution(self, execution_id: uuid.UUID, tenant_id: str) -> WorkflowExecution:
        """Cancel a running or paused workflow execution.

        Raises:
            NotFoundError: If execution does not exist.
            ValueError: If execution is already in a terminal state.
        """
        execution = await self.get_execution(execution_id, tenant_id)

        terminal_statuses = {"completed", "failed", "cancelled"}
        if execution.status in terminal_statuses:
            raise ValueError(
                f"Cannot cancel execution in status '{execution.status}' — already terminal"
            )

        # Cancel via Temporal if we have a workflow ID
        if execution.temporal_workflow_id:
            await self._durable_executor.cancel_workflow(
                str(execution.id), tenant_id
            )

        execution.status = "cancelled"
        execution.completed_at = datetime.now(UTC)
        execution.execution_history = [
            *execution.execution_history,
            {
                "event": "execution_cancelled",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        ]
        await self._session.flush()

        logger.info(
            "Workflow execution cancelled",
            execution_id=str(execution_id),
            tenant_id=tenant_id,
        )
        return execution

    async def get_execution(self, execution_id: uuid.UUID, tenant_id: str) -> WorkflowExecution:
        """Retrieve a workflow execution by ID.

        Raises:
            NotFoundError: If execution does not exist for this tenant.
        """
        from sqlalchemy import select

        stmt = select(WorkflowExecution).where(
            WorkflowExecution.id == execution_id,
            WorkflowExecution.tenant_id == uuid.UUID(tenant_id),
        )
        result = await self._session.execute(stmt)
        execution = result.scalar_one_or_none()

        if execution is None:
            raise NotFoundError(f"Execution {execution_id} not found")

        return execution

    async def record_node_transition(
        self,
        execution_id: uuid.UUID,
        tenant_id: str,
        node_name: str,
        node_output: dict[str, Any],
    ) -> None:
        """Record a node transition in execution history.

        Called by the workflow engine adapter as nodes complete execution.
        """
        execution = await self.get_execution(execution_id, tenant_id)
        execution.current_node = node_name
        execution.execution_history = [
            *execution.execution_history,
            {
                "event": "node_completed",
                "node": node_name,
                "timestamp": datetime.now(UTC).isoformat(),
                "output": node_output,
            },
        ]
        await self._session.flush()

    async def complete_execution(
        self,
        execution_id: uuid.UUID,
        tenant_id: str,
        output_data: dict[str, Any],
    ) -> WorkflowExecution:
        """Mark an execution as completed with final output.

        Also records circuit breaker success for the workflow.
        """
        execution = await self.get_execution(execution_id, tenant_id)
        execution.status = "completed"
        execution.output_data = output_data
        execution.completed_at = datetime.now(UTC)
        execution.execution_history = [
            *execution.execution_history,
            {
                "event": "execution_completed",
                "timestamp": datetime.now(UTC).isoformat(),
            },
        ]
        await self._session.flush()

        # Record circuit breaker success
        circuit_key = f"workflow:{execution.workflow_id}"
        await self._circuit_breaker.record_success(circuit_key, tenant_id)

        logger.info(
            "Workflow execution completed",
            execution_id=str(execution_id),
            tenant_id=tenant_id,
        )
        return execution

    async def fail_execution(
        self,
        execution_id: uuid.UUID,
        tenant_id: str,
        error_details: dict[str, Any],
    ) -> WorkflowExecution:
        """Mark an execution as failed with error details.

        Also records circuit breaker failure for the workflow.
        """
        execution = await self.get_execution(execution_id, tenant_id)
        execution.status = "failed"
        execution.error_details = error_details
        execution.completed_at = datetime.now(UTC)
        execution.execution_history = [
            *execution.execution_history,
            {
                "event": "execution_failed",
                "timestamp": datetime.now(UTC).isoformat(),
                "error": error_details,
            },
        ]
        await self._session.flush()

        # Record circuit breaker failure
        circuit_key = f"workflow:{execution.workflow_id}"
        await self._circuit_breaker.record_failure(circuit_key, tenant_id)

        logger.error(
            "Workflow execution failed",
            execution_id=str(execution_id),
            error=error_details,
            tenant_id=tenant_id,
        )
        return execution


class HITLService:
    """Manages human-in-the-loop approval gates, pending approvals, and decisions."""

    def __init__(
        self,
        session: AsyncSession,
        hitl_gate: HITLGateProtocol,
    ) -> None:
        """Initialize with database session and HITL gate adapter."""
        self._session = session
        self._hitl_gate = hitl_gate

    async def get_pending_approvals(
        self,
        tenant_id: str,
        execution_id: uuid.UUID | None = None,
    ) -> list[HITLApproval]:
        """List all pending HITL approvals for a tenant.

        Args:
            tenant_id: Tenant scope.
            execution_id: Optional filter by workflow execution.

        Returns:
            List of pending HITLApproval instances.
        """
        from sqlalchemy import select

        stmt = select(HITLApproval).where(
            HITLApproval.tenant_id == uuid.UUID(tenant_id),
            HITLApproval.status == "pending",
        )
        if execution_id is not None:
            stmt = stmt.where(HITLApproval.execution_id == execution_id)

        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_approval(self, approval_id: uuid.UUID, tenant_id: str) -> HITLApproval:
        """Retrieve a specific HITL approval by ID.

        Raises:
            NotFoundError: If approval does not exist for this tenant.
        """
        from sqlalchemy import select

        stmt = select(HITLApproval).where(
            HITLApproval.id == approval_id,
            HITLApproval.tenant_id == uuid.UUID(tenant_id),
        )
        result = await self._session.execute(stmt)
        approval = result.scalar_one_or_none()

        if approval is None:
            raise NotFoundError(f"HITL approval {approval_id} not found")

        return approval

    async def approve(
        self,
        approval_id: uuid.UUID,
        decided_by: uuid.UUID,
        notes: str | None,
        tenant_id: str,
    ) -> HITLApproval:
        """Approve a pending HITL gate and resume the workflow.

        Args:
            approval_id: HITL approval to approve.
            decided_by: UUID of the user approving.
            notes: Optional approval notes.
            tenant_id: Tenant scope.

        Returns:
            Updated HITLApproval with approved status.

        Raises:
            NotFoundError: If approval not found.
            ValueError: If approval is not in pending state.
        """
        approval = await self.get_approval(approval_id, tenant_id)

        if approval.status != "pending":
            raise ValueError(
                f"Cannot approve HITL gate in status '{approval.status}' — not pending"
            )

        approval.status = "approved"
        approval.decided_by = decided_by
        approval.decided_at = datetime.now(UTC)
        approval.decision_notes = notes
        await self._session.flush()

        # Resume workflow via HITL gate adapter
        await self._hitl_gate.approve(
            approval_id=approval_id,
            decided_by=decided_by,
            notes=notes,
            tenant_id=tenant_id,
        )

        logger.info(
            "HITL approval granted",
            approval_id=str(approval_id),
            execution_id=str(approval.execution_id),
            gate_name=approval.gate_name,
            decided_by=str(decided_by),
            tenant_id=tenant_id,
        )
        return approval

    async def reject(
        self,
        approval_id: uuid.UUID,
        decided_by: uuid.UUID,
        notes: str | None,
        tenant_id: str,
    ) -> HITLApproval:
        """Reject a pending HITL gate and fail the workflow.

        Args:
            approval_id: HITL approval to reject.
            decided_by: UUID of the user rejecting.
            notes: Optional rejection reason.
            tenant_id: Tenant scope.

        Returns:
            Updated HITLApproval with rejected status.
        """
        approval = await self.get_approval(approval_id, tenant_id)

        if approval.status != "pending":
            raise ValueError(
                f"Cannot reject HITL gate in status '{approval.status}' — not pending"
            )

        approval.status = "rejected"
        approval.decided_by = decided_by
        approval.decided_at = datetime.now(UTC)
        approval.decision_notes = notes
        await self._session.flush()

        # Fail workflow via HITL gate adapter
        await self._hitl_gate.reject(
            approval_id=approval_id,
            decided_by=decided_by,
            notes=notes,
            tenant_id=tenant_id,
        )

        logger.info(
            "HITL approval rejected",
            approval_id=str(approval_id),
            execution_id=str(approval.execution_id),
            gate_name=approval.gate_name,
            decided_by=str(decided_by),
            notes=notes,
            tenant_id=tenant_id,
        )
        return approval


class CircuitBreakerService:
    """Manages per-agent and per-workflow circuit breakers with cascading failure containment.

    Circuit breakers prevent cascading failures by short-circuiting calls to failing
    agents or workflows. State is persisted in Redis for fast, distributed access.
    """

    def __init__(self, circuit_breaker: CircuitBreakerProtocol) -> None:
        """Initialize with circuit breaker adapter."""
        self._circuit_breaker = circuit_breaker

    def _agent_key(self, agent_id: uuid.UUID) -> str:
        """Generate circuit breaker key for an agent."""
        return f"agent:{agent_id}"

    def _workflow_key(self, workflow_id: uuid.UUID) -> str:
        """Generate circuit breaker key for a workflow."""
        return f"workflow:{workflow_id}"

    async def check_agent(self, agent_id: uuid.UUID, tenant_id: str) -> str:
        """Check agent circuit breaker state.

        Args:
            agent_id: Agent to check.
            tenant_id: Tenant context.

        Returns:
            Circuit state: 'closed', 'open', or 'half_open'.
        """
        return await self._circuit_breaker.get_state(self._agent_key(agent_id), tenant_id)

    async def is_agent_available(self, agent_id: uuid.UUID, tenant_id: str) -> bool:
        """Check if an agent circuit breaker allows requests.

        Returns:
            True if circuit is CLOSED or HALF_OPEN (allowing test requests).
        """
        return not await self._circuit_breaker.is_open(self._agent_key(agent_id), tenant_id)

    async def record_agent_success(self, agent_id: uuid.UUID, tenant_id: str) -> None:
        """Record a successful agent invocation — may close circuit."""
        await self._circuit_breaker.record_success(self._agent_key(agent_id), tenant_id)

    async def record_agent_failure(self, agent_id: uuid.UUID, tenant_id: str) -> None:
        """Record a failed agent invocation — may open circuit."""
        await self._circuit_breaker.record_failure(self._agent_key(agent_id), tenant_id)
        logger.warning(
            "Agent failure recorded in circuit breaker",
            agent_id=str(agent_id),
            tenant_id=tenant_id,
        )

    async def get_all_circuit_states(
        self,
        agent_ids: list[uuid.UUID],
        tenant_id: str,
    ) -> dict[str, str]:
        """Retrieve circuit breaker states for multiple agents.

        Useful for dashboard/monitoring views.

        Returns:
            Dict mapping agent_id string to circuit state.
        """
        states: dict[str, str] = {}
        for agent_id in agent_ids:
            states[str(agent_id)] = await self._circuit_breaker.get_state(
                self._agent_key(agent_id), tenant_id
            )
        return states


class SessionService:
    """Per-tenant agent session isolation with memory scoping (private/shared).

    Private sessions are scoped to (tenant, agent) and persist across executions.
    Shared sessions are scoped to (tenant, execution) and are accessible to all
    agents participating in that workflow execution.
    """

    def __init__(
        self,
        session: AsyncSession,
        session_isolator: SessionIsolatorProtocol,
    ) -> None:
        """Initialize with database session and session isolator adapter."""
        self._session = session
        self._session_isolator = session_isolator

    async def get_agent_memory(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        execution_id: uuid.UUID | None = None,
    ) -> dict[str, Any]:
        """Get agent's private memory state.

        Args:
            agent_id: Agent whose memory to retrieve.
            tenant_id: Tenant context for isolation.
            execution_id: Optional execution scope for retrieval context.

        Returns:
            Dict of agent memory key-value pairs.
        """
        return await self._session_isolator.get_session(agent_id, tenant_id, execution_id)

    async def update_agent_memory(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        updates: dict[str, Any],
        execution_id: uuid.UUID | None = None,
    ) -> None:
        """Update agent's private memory with partial updates."""
        await self._session_isolator.update_session(agent_id, tenant_id, updates, execution_id)

    async def get_shared_memory(
        self,
        execution_id: uuid.UUID,
        tenant_id: str,
    ) -> dict[str, Any]:
        """Get shared memory accessible to all agents in a workflow execution."""
        return await self._session_isolator.get_shared_session(execution_id, tenant_id)

    async def update_shared_memory(
        self,
        execution_id: uuid.UUID,
        tenant_id: str,
        updates: dict[str, Any],
    ) -> None:
        """Update shared memory for all agents in a workflow execution."""
        await self._session_isolator.update_shared_session(execution_id, tenant_id, updates)

    async def create_session_record(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        memory_scope: str,
        execution_id: uuid.UUID | None = None,
    ) -> AgentSession:
        """Create a session record in the database.

        Args:
            agent_id: Agent for this session.
            tenant_id: Tenant context.
            memory_scope: 'private' or 'shared'.
            execution_id: Optional workflow execution association.

        Returns:
            Created AgentSession instance.
        """
        if memory_scope not in {"private", "shared"}:
            raise ValueError(f"memory_scope must be 'private' or 'shared', got '{memory_scope}'")

        agent_session = AgentSession(
            tenant_id=uuid.UUID(tenant_id),
            agent_id=agent_id,
            execution_id=execution_id,
            session_data={},
            memory_scope=memory_scope,
        )
        self._session.add(agent_session)
        await self._session.flush()
        return agent_session


class ToolRegistryService:
    """Registers and manages tools available to agents.

    Tools are registered by the system and assigned to agents via the
    tool_access field on AgentDefinition. Access is checked against
    both the tool's min_privilege_level and the agent's tool_access config.
    """

    def __init__(
        self,
        session: AsyncSession,
        tool_registry: ToolRegistryProtocol,
    ) -> None:
        """Initialize with database session and tool registry adapter."""
        self._session = session
        self._tool_registry = tool_registry

    async def register_tool(
        self,
        tenant_id: str,
        name: str,
        description: str,
        min_privilege_level: int,
        input_schema: dict[str, Any],
        config: dict[str, Any],
    ) -> ToolDefinition:
        """Register a new tool in the registry.

        Args:
            tenant_id: Tenant registering the tool.
            name: Unique tool name.
            description: Human-readable tool description.
            min_privilege_level: Minimum privilege level required to use this tool.
            input_schema: JSON schema for tool input parameters.
            config: Tool endpoint/configuration details.

        Returns:
            Created ToolDefinition instance.
        """
        if not 1 <= min_privilege_level <= 5:
            raise ValueError(f"min_privilege_level must be 1-5, got {min_privilege_level}")

        tool = ToolDefinition(
            tenant_id=uuid.UUID(tenant_id),
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
            "Tool registered",
            tool_id=str(tool.id),
            name=name,
            min_privilege_level=min_privilege_level,
            tenant_id=tenant_id,
        )
        return tool

    async def check_agent_tool_access(
        self,
        agent: AgentDefinition,
        tool_name: str,
        tenant_id: str,
    ) -> bool:
        """Check if an agent has access to a specific tool.

        Validates both tool existence and agent's configured tool_access.

        Args:
            agent: Agent definition to check.
            tool_name: Name of tool being requested.
            tenant_id: Tenant context.

        Returns:
            True if access is permitted.
        """
        return await self._tool_registry.check_access(agent.id, tool_name, tenant_id)

    async def list_tools(
        self,
        tenant_id: str,
        min_privilege_level: int | None = None,
    ) -> list[dict[str, Any]]:
        """List all available tools with optional privilege filter."""
        return await self._tool_registry.list_tools(tenant_id, min_privilege_level)
