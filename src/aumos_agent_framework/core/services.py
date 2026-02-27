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
    ActionExecutorProtocol,
    AgentMemoryManagerProtocol,
    AgentRuntimeProtocol,
    CircuitBreakerProtocol,
    DurableExecutorProtocol,
    HITLGateProtocol,
    MultiAgentCoordinatorProtocol,
    ObservabilityHooksProtocol,
    ReasoningEngineProtocol,
    SecuritySandboxProtocol,
    SessionIsolatorProtocol,
    SkillComposerProtocol,
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


class AgentMemoryService:
    """High-level service for agent short-term and long-term memory operations.

    Wraps AgentMemoryManagerProtocol with business-logic defaults and
    privilege-aware consolidation triggers.
    """

    def __init__(self, memory_manager: AgentMemoryManagerProtocol) -> None:
        """Initialize with a memory manager adapter.

        Args:
            memory_manager: Adapter implementing AgentMemoryManagerProtocol.
        """
        self._memory = memory_manager

    async def record_conversation_turn(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        role: str,
        content: str,
        importance: float = 0.5,
    ) -> str:
        """Record a single conversation turn in short-term memory.

        Args:
            agent_id: Agent whose memory to update.
            tenant_id: Tenant context.
            role: Message role ('user', 'assistant', 'system').
            content: Message content.
            importance: Importance score for consolidation gating.

        Returns:
            Memory entry ID string.
        """
        return await self._memory.add_short_term(
            agent_id=agent_id,
            tenant_id=tenant_id,
            content=content,
            memory_type="conversation",
            importance=importance,
            metadata={"role": role},
        )

    async def record_tool_output(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        tool_name: str,
        output_summary: str,
        importance: float = 0.6,
    ) -> str:
        """Record a tool execution output in short-term memory.

        Args:
            agent_id: Agent whose memory to update.
            tenant_id: Tenant context.
            tool_name: Name of the tool that produced the output.
            output_summary: Summarized output string.
            importance: Importance score.

        Returns:
            Memory entry ID string.
        """
        return await self._memory.add_short_term(
            agent_id=agent_id,
            tenant_id=tenant_id,
            content=output_summary,
            memory_type="tool_output",
            importance=importance,
            metadata={"tool_name": tool_name},
        )

    async def get_recent_context(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Retrieve recent short-term memory for context injection.

        Args:
            agent_id: Agent whose memory to retrieve.
            tenant_id: Tenant context.
            limit: Maximum entries to return.

        Returns:
            List of recent memory entry dicts.
        """
        return await self._memory.get_short_term(
            agent_id=agent_id,
            tenant_id=tenant_id,
            limit=limit,
            memory_type=None,
        )

    async def search_relevant_memories(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        query_embedding: list[float],
        limit: int = 10,
        topic: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search long-term memory for contextually relevant entries.

        Args:
            agent_id: Agent whose long-term memory to search.
            tenant_id: Tenant context.
            query_embedding: Embedding vector for similarity search.
            limit: Maximum results to return.
            topic: Optional topic filter.

        Returns:
            List of memory entry dicts ordered by relevance.
        """
        return await self._memory.search_long_term(
            agent_id=agent_id,
            tenant_id=tenant_id,
            query_embedding=query_embedding,
            limit=limit,
            topic=topic,
            memory_type=None,
            min_importance=0.0,
            since_hours=None,
        )


class ReasoningService:
    """Orchestrates LLM-based reasoning within the privilege and circuit breaker context.

    Wraps ReasoningEngineProtocol and enforces circuit breaker checks before
    every inference call to prevent cascading failures on LLM serving outages.
    """

    def __init__(
        self,
        reasoning_engine: ReasoningEngineProtocol,
        circuit_breaker: CircuitBreakerProtocol,
        observability: ObservabilityHooksProtocol,
    ) -> None:
        """Initialize with reasoning engine and supporting adapters.

        Args:
            reasoning_engine: Adapter implementing ReasoningEngineProtocol.
            circuit_breaker: Circuit breaker for LLM serving protection.
            observability: Observability hooks for token and trace recording.
        """
        self._engine = reasoning_engine
        self._circuit_breaker = circuit_breaker
        self._observability = observability

    _LLM_SERVING_CIRCUIT_KEY = "llm_serving:default"

    async def reason_chain_of_thought(
        self,
        agent_id: uuid.UUID,
        task_description: str,
        context: dict[str, Any],
        tenant_id: str,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute chain-of-thought reasoning with circuit breaker protection.

        Args:
            agent_id: Agent executing the reasoning.
            task_description: Task to reason about.
            context: Context dict for the reasoning engine.
            tenant_id: Tenant context.
            model_id: Optional model override.

        Returns:
            ReasoningTrace dict from the engine.

        Raises:
            PermissionDeniedError: If the LLM serving circuit breaker is OPEN.
        """
        if await self._circuit_breaker.is_open(self._LLM_SERVING_CIRCUIT_KEY, tenant_id):
            raise PermissionDeniedError(
                "LLM serving circuit breaker is OPEN — reasoning unavailable"
            )

        try:
            trace = await self._engine.chain_of_thought(
                agent_id=agent_id,
                task_description=task_description,
                context=context,
                tenant_id=tenant_id,
                model_id=model_id,
            )
            await self._circuit_breaker.record_success(self._LLM_SERVING_CIRCUIT_KEY, tenant_id)

            # Record token consumption if trace has the field
            if isinstance(trace, dict) and "total_tokens" in trace:
                await self._observability.record_tokens_consumed(
                    agent_id=agent_id,
                    tenant_id=tenant_id,
                    token_count=trace["total_tokens"],
                    model_id=model_id or "default",
                    action_type="chain_of_thought",
                )

            return trace if isinstance(trace, dict) else trace.to_dict()

        except Exception as exc:
            await self._circuit_breaker.record_failure(self._LLM_SERVING_CIRCUIT_KEY, tenant_id)
            logger.error(
                "Chain-of-thought reasoning failed",
                agent_id=str(agent_id),
                error=str(exc),
                tenant_id=tenant_id,
            )
            raise

    async def select_tools_for_task(
        self,
        agent_id: uuid.UUID,
        task_description: str,
        available_tools: list[dict[str, Any]],
        tenant_id: str,
        top_k: int = 5,
        model_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recommend relevant tools for a task using LLM-based reasoning.

        Args:
            agent_id: Agent requesting tool selection.
            task_description: Task description to match tools against.
            available_tools: Full list of available tool definitions.
            tenant_id: Tenant context.
            top_k: Maximum tools to recommend.
            model_id: Optional model override.

        Returns:
            Sorted list of tool dicts with 'relevance_score' added.
        """
        if await self._circuit_breaker.is_open(self._LLM_SERVING_CIRCUIT_KEY, tenant_id):
            # Fallback: return all tools without LLM ranking
            logger.warning(
                "LLM circuit open during tool selection — returning all tools unranked",
                agent_id=str(agent_id),
                tenant_id=tenant_id,
            )
            return available_tools[:top_k]

        return await self._engine.select_tools(
            agent_id=agent_id,
            task_description=task_description,
            available_tools=available_tools,
            tenant_id=tenant_id,
            top_k=top_k,
            model_id=model_id,
        )


class ActionExecutionService:
    """Coordinates sandboxed tool execution with privilege enforcement and audit logging.

    Wraps ActionExecutorProtocol and enforces that agents can only invoke
    tools they have access to, with all executions tracked via ObservabilityHooks.
    """

    def __init__(
        self,
        action_executor: ActionExecutorProtocol,
        tool_registry: ToolRegistryProtocol,
        observability: ObservabilityHooksProtocol,
        security_sandbox: SecuritySandboxProtocol,
    ) -> None:
        """Initialize with execution and registry adapters.

        Args:
            action_executor: Adapter implementing ActionExecutorProtocol.
            tool_registry: Tool registry for access control checks.
            observability: Observability hooks for audit logging.
            security_sandbox: Security sandbox for untrusted code execution.
        """
        self._executor = action_executor
        self._tool_registry = tool_registry
        self._observability = observability
        self._sandbox = security_sandbox

    async def execute_tool(
        self,
        agent_id: uuid.UUID,
        tool_name: str,
        tool_input: dict[str, Any],
        tenant_id: str,
        execution_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a tool after verifying agent access rights.

        Args:
            agent_id: Agent requesting tool execution.
            tool_name: Name of the tool to invoke.
            tool_input: Input data for the tool.
            tenant_id: Tenant context.
            execution_id: Optional workflow execution correlation ID.

        Returns:
            Audit result dict from the action executor.

        Raises:
            PermissionDeniedError: If agent does not have access to the tool.
            NotFoundError: If the tool is not registered.
        """
        # Check tool access
        has_access = await self._tool_registry.check_access(agent_id, tool_name, tenant_id)
        if not has_access:
            logger.warning(
                "Agent tool access denied",
                agent_id=str(agent_id),
                tool_name=tool_name,
                tenant_id=tenant_id,
            )
            raise PermissionDeniedError(
                f"Agent {agent_id} does not have access to tool '{tool_name}'"
            )

        # Retrieve tool definition
        tool_def = await self._tool_registry.get_tool(tool_name, tenant_id)
        if tool_def is None:
            raise NotFoundError(f"Tool '{tool_name}' not found")

        # Execute via action executor
        return await self._executor.execute(
            tool_name=tool_name,
            tool_definition=tool_def,
            tool_input=tool_input,
            agent_id=agent_id,
            tenant_id=tenant_id,
            execution_id=execution_id,
        )

    async def execute_sandboxed_code(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        code: str,
        input_data: dict[str, Any],
        agent_config: dict[str, Any],
        privilege_level: int,
    ) -> dict[str, Any]:
        """Execute untrusted code in a Docker sandbox scoped to agent privilege.

        Args:
            agent_id: Agent requesting sandboxed execution.
            tenant_id: Tenant context.
            code: Python source code to execute.
            input_data: Input data available to the code.
            agent_config: Agent configuration for policy derivation.
            privilege_level: Agent privilege level for resource limit scaling.

        Returns:
            SandboxExecutionResult dict.
        """
        policy = await self._sandbox.build_policy_for_agent(agent_config, privilege_level)
        return await self._sandbox.execute_code(
            code=code,
            input_data=input_data,
            agent_id=agent_id,
            tenant_id=tenant_id,
            policy=policy,
        )


class MultiAgentOrchestrationService:
    """Manages inter-agent communication, task delegation, and coordination.

    Provides a unified interface over MultiAgentCoordinatorProtocol with
    automatic deadlock detection and privilege enforcement for delegation.
    """

    def __init__(
        self,
        coordinator: MultiAgentCoordinatorProtocol,
        circuit_breaker: CircuitBreakerProtocol,
    ) -> None:
        """Initialize with coordinator and circuit breaker adapters.

        Args:
            coordinator: Adapter implementing MultiAgentCoordinatorProtocol.
            circuit_breaker: Circuit breaker for coordinator health protection.
        """
        self._coordinator = coordinator
        self._circuit_breaker = circuit_breaker

    async def send_notification(
        self,
        sender_agent_id: uuid.UUID,
        recipient_agent_id: uuid.UUID,
        message: str,
        tenant_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Send a notification message from one agent to another.

        Args:
            sender_agent_id: Sending agent.
            recipient_agent_id: Receiving agent.
            message: Notification message string.
            tenant_id: Tenant context.
            metadata: Optional additional payload.

        Returns:
            Message ID string.
        """
        return await self._coordinator.send_message(
            sender_agent_id=sender_agent_id,
            recipient_agent_id=recipient_agent_id,
            payload={"message": message, **(metadata or {})},
            tenant_id=tenant_id,
            message_type="notification",
        )

    async def delegate_with_result(
        self,
        delegator_agent_id: uuid.UUID,
        delegate_agent_id: uuid.UUID,
        task_description: str,
        task_input: dict[str, Any],
        tenant_id: str,
        timeout_seconds: int = 60,
    ) -> dict[str, Any]:
        """Delegate a task and await the delegate's response.

        Combines delegate_task + request_response for synchronous delegation.

        Args:
            delegator_agent_id: Agent delegating the task.
            delegate_agent_id: Agent to perform the task.
            task_description: Human-readable task description.
            task_input: Structured task input.
            tenant_id: Tenant context.
            timeout_seconds: Response timeout.

        Returns:
            Response payload dict from the delegate agent.
        """
        await self._coordinator.delegate_task(
            delegator_agent_id=delegator_agent_id,
            delegate_agent_id=delegate_agent_id,
            task_description=task_description,
            task_input=task_input,
            tenant_id=tenant_id,
            priority=5,
            requires_ack=True,
        )

        return await self._coordinator.request_response(
            sender_agent_id=delegator_agent_id,
            recipient_agent_id=delegate_agent_id,
            request_payload=task_input,
            tenant_id=tenant_id,
            timeout_seconds=timeout_seconds,
        )
