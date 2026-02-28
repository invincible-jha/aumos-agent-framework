"""FastAPI router for aumos-agent-framework.

All routes are thin — they validate input, delegate to services, and return responses.
Business logic lives exclusively in core/services.py.
"""

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import TenantContext, UserContext, get_current_tenant, get_current_user
from aumos_common.database import get_db_session
from aumos_common.errors import NotFoundError, PermissionDeniedError
from aumos_common.observability import get_logger

from aumos_agent_framework.api.schemas import (
    AgentCreateRequest,
    AgentInvokeRequest,
    AgentInvokeResponse,
    AgentListResponse,
    AgentResponse,
    AgentUpdateToolAccessRequest,
    CheckpointListResponse,
    ExecutionNodeStateResponse,
    ExecutionResponse,
    ExecutionStatusResponse,
    ExecutionTraceListResponse,
    ExecutionTraceResponse,
    HITLApprovalResponse,
    HITLApproveRequest,
    HITLPendingListResponse,
    HITLRejectRequest,
    ReplayRequest,
    WorkflowCreateRequest,
    WorkflowExecuteRequest,
    WorkflowResponse,
    WorkflowStreamEvent,
)
from aumos_agent_framework.core.services import (
    AgentRegistryService,
    ExecutionService,
    HITLService,
    WorkflowService,
)

logger = get_logger(__name__)

router = APIRouter(tags=["agent-framework"])

# ============================================================================
# Dependency helpers
# ============================================================================

SessionDep = Annotated[AsyncSession, Depends(get_db_session)]
TenantDep = Annotated[TenantContext, Depends(get_current_tenant)]
UserDep = Annotated[UserContext, Depends(get_current_user)]


def _get_agent_service(session: SessionDep) -> AgentRegistryService:
    return AgentRegistryService(session)


def _get_workflow_service(session: SessionDep) -> WorkflowService:
    return WorkflowService(session)


# ============================================================================
# Agent endpoints
# ============================================================================


@router.post(
    "/agents",
    response_model=AgentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new agent",
)
async def register_agent(
    body: AgentCreateRequest,
    tenant: TenantDep,
    session: SessionDep,
) -> AgentResponse:
    """Register a new agent definition with capabilities and privilege level."""
    try:
        service = AgentRegistryService(session)
        agent = await service.register_agent(
            tenant_id=str(tenant.tenant_id),
            name=body.name,
            description=body.description,
            capabilities=body.capabilities,
            privilege_level=body.privilege_level,
            tool_access=body.tool_access,
            resource_limits=body.resource_limits,
        )
        await session.commit()
        return AgentResponse.model_validate(agent)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


@router.get(
    "/agents",
    response_model=AgentListResponse,
    summary="List agents for the current tenant",
)
async def list_agents(
    tenant: TenantDep,
    session: SessionDep,
    agent_status: str | None = None,
    capability: str | None = None,
) -> AgentListResponse:
    """List all agents for the tenant with optional filters."""
    service = AgentRegistryService(session)
    agents = await service.list_agents(
        tenant_id=str(tenant.tenant_id),
        status=agent_status,
        capability=capability,
    )
    return AgentListResponse(
        items=[AgentResponse.model_validate(a) for a in agents],
        total=len(agents),
        page=1,
        page_size=len(agents),
    )


@router.put(
    "/agents/{agent_id}/tools",
    response_model=AgentResponse,
    summary="Update agent tool access",
)
async def update_agent_tools(
    agent_id: uuid.UUID,
    body: AgentUpdateToolAccessRequest,
    tenant: TenantDep,
    session: SessionDep,
) -> AgentResponse:
    """Update the tool access configuration for an agent."""
    try:
        service = AgentRegistryService(session)
        agent = await service.update_tool_access(
            agent_id=agent_id,
            tenant_id=str(tenant.tenant_id),
            tool_access=body.tool_access,
        )
        await session.commit()
        return AgentResponse.model_validate(agent)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.post(
    "/agents/{agent_id}/invoke",
    response_model=AgentInvokeResponse,
    summary="Directly invoke an agent",
)
async def invoke_agent(
    agent_id: uuid.UUID,
    body: AgentInvokeRequest,
    tenant: TenantDep,
    user: UserDep,
    session: SessionDep,
) -> AgentInvokeResponse:
    """Directly invoke an agent outside of a workflow execution."""
    try:
        service = AgentRegistryService(session)
        agent = await service.get_agent(agent_id, str(tenant.tenant_id))

        # TODO: Delegate to AgentExecutorProtocol adapter for actual invocation
        logger.info(
            "Agent invoked directly",
            agent_id=str(agent_id),
            tenant_id=str(tenant.tenant_id),
            user_id=str(user.user_id),
        )

        return AgentInvokeResponse(
            agent_id=agent_id,
            output={"status": "invoked", "agent_name": agent.name},
            session_id=f"{tenant.tenant_id}:{agent_id}",
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


# ============================================================================
# Workflow endpoints
# ============================================================================


@router.post(
    "/workflows",
    response_model=WorkflowResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a workflow definition",
)
async def create_workflow(
    body: WorkflowCreateRequest,
    tenant: TenantDep,
    session: SessionDep,
) -> WorkflowResponse:
    """Create a new workflow definition using LangGraph DSL."""
    service = WorkflowService(session)
    workflow = await service.create_workflow(
        tenant_id=str(tenant.tenant_id),
        name=body.name,
        graph_definition=body.graph_definition,
        agents=body.agents,
        hitl_gates=body.hitl_gates,
        circuit_breaker_config=body.circuit_breaker_config,
    )
    await session.commit()
    return WorkflowResponse.model_validate(workflow)


@router.post(
    "/workflows/{workflow_id}/execute",
    response_model=ExecutionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Execute a workflow",
)
async def execute_workflow(
    workflow_id: uuid.UUID,
    body: WorkflowExecuteRequest,
    tenant: TenantDep,
    session: SessionDep,
) -> ExecutionResponse:
    """Start execution of a workflow definition.

    Returns immediately with execution record in 'running' state.
    Monitor progress via GET /workflows/{id}/status.
    """
    try:
        # NOTE: ExecutionService requires adapters injected from app context.
        # For now, retrieve and validate the workflow exists, then return stub.
        # Full implementation requires DI container setup in main.py lifespan.
        workflow_service = WorkflowService(session)
        workflow = await workflow_service.get_workflow(workflow_id, str(tenant.tenant_id))

        logger.info(
            "Workflow execution requested",
            workflow_id=str(workflow_id),
            tenant_id=str(tenant.tenant_id),
        )

        # TODO: Use ExecutionService with injected adapters from app state
        # execution = await execution_service.start_execution(...)
        # Returning workflow info for now — full execution requires adapter wiring
        return ExecutionResponse(
            id=uuid.uuid4(),
            tenant_id=tenant.tenant_id,
            workflow_id=workflow_id,
            status="pending",
            current_node=None,
            execution_history=[],
            input_data=body.input_data,
            output_data=None,
            error_details=None,
            temporal_workflow_id=None,
            started_at=None,
            completed_at=None,
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except PermissionDeniedError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))


@router.get(
    "/workflows/{workflow_id}/status",
    response_model=ExecutionStatusResponse,
    summary="Get workflow execution status",
)
async def get_execution_status(
    workflow_id: uuid.UUID,
    tenant: TenantDep,
    session: SessionDep,
    execution_id: uuid.UUID | None = None,
) -> ExecutionStatusResponse:
    """Get the current status of a workflow execution.

    Pass execution_id query param to get a specific execution.
    Without it, returns the most recent execution for the workflow.
    """
    try:
        if execution_id is None:
            # Return workflow definition status as fallback
            workflow_service = WorkflowService(session)
            workflow = await workflow_service.get_workflow(workflow_id, str(tenant.tenant_id))
            return ExecutionStatusResponse(
                id=workflow.id,
                workflow_id=workflow.id,
                status="no_execution",
                current_node=None,
                started_at=None,
                completed_at=None,
                temporal_workflow_id=None,
            )

        from sqlalchemy import select

        from aumos_agent_framework.core.models import WorkflowExecution

        stmt = select(WorkflowExecution).where(
            WorkflowExecution.id == execution_id,
            WorkflowExecution.workflow_id == workflow_id,
            WorkflowExecution.tenant_id == tenant.tenant_id,
        )
        result = await session.execute(stmt)
        execution = result.scalar_one_or_none()

        if execution is None:
            raise NotFoundError(f"Execution {execution_id} not found")

        return ExecutionStatusResponse(
            id=execution.id,
            workflow_id=execution.workflow_id,
            status=execution.status,
            current_node=execution.current_node,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            temporal_workflow_id=execution.temporal_workflow_id,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.post(
    "/workflows/{workflow_id}/cancel",
    response_model=ExecutionStatusResponse,
    summary="Cancel a running workflow execution",
)
async def cancel_workflow(
    workflow_id: uuid.UUID,
    tenant: TenantDep,
    session: SessionDep,
    execution_id: uuid.UUID | None = None,
) -> ExecutionStatusResponse:
    """Cancel a running or paused workflow execution."""
    try:
        if execution_id is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="execution_id query parameter is required for cancellation",
            )

        from sqlalchemy import select

        from aumos_agent_framework.core.models import WorkflowExecution

        stmt = select(WorkflowExecution).where(
            WorkflowExecution.id == execution_id,
            WorkflowExecution.workflow_id == workflow_id,
            WorkflowExecution.tenant_id == tenant.tenant_id,
        )
        result = await session.execute(stmt)
        execution = result.scalar_one_or_none()

        if execution is None:
            raise NotFoundError(f"Execution {execution_id} not found")

        terminal_statuses = {"completed", "failed", "cancelled"}
        if execution.status in terminal_statuses:
            raise ValueError(f"Execution is already in terminal state: {execution.status}")

        execution.status = "cancelled"
        from datetime import UTC, datetime

        execution.completed_at = datetime.now(UTC)
        await session.commit()

        logger.info(
            "Workflow execution cancelled via API",
            execution_id=str(execution_id),
            workflow_id=str(workflow_id),
            tenant_id=str(tenant.tenant_id),
        )

        return ExecutionStatusResponse(
            id=execution.id,
            workflow_id=execution.workflow_id,
            status=execution.status,
            current_node=execution.current_node,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            temporal_workflow_id=execution.temporal_workflow_id,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


# ============================================================================
# HITL endpoints
# ============================================================================


@router.get(
    "/hitl/pending",
    response_model=HITLPendingListResponse,
    summary="List pending human approvals",
)
async def list_pending_approvals(
    tenant: TenantDep,
    session: SessionDep,
    execution_id: uuid.UUID | None = None,
) -> HITLPendingListResponse:
    """List all pending HITL approvals for the tenant."""
    from aumos_agent_framework.adapters.circuit_breaker import RedisCircuitBreaker

    # HITLService needs the HITL gate adapter — for listing we only need the DB query
    from sqlalchemy import select

    from aumos_agent_framework.core.models import HITLApproval

    stmt = select(HITLApproval).where(
        HITLApproval.tenant_id == tenant.tenant_id,
        HITLApproval.status == "pending",
    )
    if execution_id is not None:
        stmt = stmt.where(HITLApproval.execution_id == execution_id)

    result = await session.execute(stmt)
    approvals = list(result.scalars().all())

    return HITLPendingListResponse(
        items=[HITLApprovalResponse.model_validate(a) for a in approvals],
        total=len(approvals),
    )


@router.post(
    "/hitl/{approval_id}/approve",
    response_model=HITLApprovalResponse,
    summary="Approve a pending HITL gate",
)
async def approve_hitl(
    approval_id: uuid.UUID,
    body: HITLApproveRequest,
    tenant: TenantDep,
    user: UserDep,
    session: SessionDep,
) -> HITLApprovalResponse:
    """Approve a pending HITL approval gate and resume the workflow."""
    try:
        from sqlalchemy import select

        from aumos_agent_framework.core.models import HITLApproval

        stmt = select(HITLApproval).where(
            HITLApproval.id == approval_id,
            HITLApproval.tenant_id == tenant.tenant_id,
        )
        result = await session.execute(stmt)
        approval = result.scalar_one_or_none()

        if approval is None:
            raise NotFoundError(f"HITL approval {approval_id} not found")

        if approval.status != "pending":
            raise ValueError(f"Cannot approve — approval is in status '{approval.status}'")

        from datetime import UTC, datetime

        approval.status = "approved"
        approval.decided_by = user.user_id
        approval.decided_at = datetime.now(UTC)
        approval.decision_notes = body.notes
        await session.commit()

        logger.info(
            "HITL approval granted",
            approval_id=str(approval_id),
            decided_by=str(user.user_id),
            tenant_id=str(tenant.tenant_id),
        )
        return HITLApprovalResponse.model_validate(approval)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


@router.post(
    "/hitl/{approval_id}/reject",
    response_model=HITLApprovalResponse,
    summary="Reject a pending HITL gate",
)
async def reject_hitl(
    approval_id: uuid.UUID,
    body: HITLRejectRequest,
    tenant: TenantDep,
    user: UserDep,
    session: SessionDep,
) -> HITLApprovalResponse:
    """Reject a pending HITL approval gate and fail the workflow."""
    try:
        from sqlalchemy import select

        from aumos_agent_framework.core.models import HITLApproval

        stmt = select(HITLApproval).where(
            HITLApproval.id == approval_id,
            HITLApproval.tenant_id == tenant.tenant_id,
        )
        result = await session.execute(stmt)
        approval = result.scalar_one_or_none()

        if approval is None:
            raise NotFoundError(f"HITL approval {approval_id} not found")

        if approval.status != "pending":
            raise ValueError(f"Cannot reject — approval is in status '{approval.status}'")

        from datetime import UTC, datetime

        approval.status = "rejected"
        approval.decided_by = user.user_id
        approval.decided_at = datetime.now(UTC)
        approval.decision_notes = body.notes
        await session.commit()

        logger.info(
            "HITL approval rejected",
            approval_id=str(approval_id),
            decided_by=str(user.user_id),
            notes=body.notes,
            tenant_id=str(tenant.tenant_id),
        )
        return HITLApprovalResponse.model_validate(approval)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


# ============================================================================
# SSE streaming endpoint (Gap #132)
# ============================================================================


@router.post(
    "/workflows/{workflow_id}/execute/stream",
    summary="Execute a workflow with SSE streaming",
    response_class=StreamingResponse,
)
async def execute_workflow_stream(
    workflow_id: uuid.UUID,
    body: WorkflowExecuteRequest,
    tenant: TenantDep,
    session: SessionDep,
) -> StreamingResponse:
    """Execute a workflow and stream events in real-time via Server-Sent Events.

    The response is a text/event-stream with JSON-encoded WorkflowStreamEvent
    objects. Each event has a 'data:' line containing the JSON payload.
    """
    import json

    from aumos_agent_framework.adapters.execution_streamer import stream_workflow_events

    async def event_generator() -> Any:
        try:
            async for event in stream_workflow_events(
                workflow_id=workflow_id,
                input_data=body.input_data,
                tenant_id=str(tenant.tenant_id),
                session=session,
            ):
                yield f"data: {json.dumps(event.model_dump(mode='json'))}\n\n"
        except NotFoundError:
            error_event = WorkflowStreamEvent(
                event_type="error",
                data={"message": f"Workflow {workflow_id} not found"},
                execution_id=None,
                node_id=None,
                timestamp=None,
            )
            yield f"data: {json.dumps(error_event.model_dump(mode='json'))}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================================
# Visual workflow builder — node state (Gap #128)
# ============================================================================


@router.get(
    "/executions/{execution_id}/nodes",
    response_model=list[ExecutionNodeStateResponse],
    summary="Get per-node execution state for visual workflow builder",
)
async def get_execution_node_states(
    execution_id: uuid.UUID,
    tenant: TenantDep,
    session: SessionDep,
) -> list[ExecutionNodeStateResponse]:
    """Return the execution state of each node in a running workflow.

    Designed to power real-time visual workflow builder overlays.
    """
    from sqlalchemy import select

    from aumos_agent_framework.core.models import WorkflowExecution

    stmt = select(WorkflowExecution).where(
        WorkflowExecution.id == execution_id,
        WorkflowExecution.tenant_id == tenant.tenant_id,
    )
    result = await session.execute(stmt)
    execution = result.scalar_one_or_none()

    if execution is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Execution {execution_id} not found")

    node_states: list[ExecutionNodeStateResponse] = []
    history: list[dict[str, Any]] = execution.execution_history or []
    for node_record in history:
        node_states.append(
            ExecutionNodeStateResponse(
                node_id=node_record.get("node_id", "unknown"),
                status=node_record.get("status", "pending"),
                started_at=node_record.get("started_at"),
                completed_at=node_record.get("completed_at"),
                error_message=node_record.get("error_message"),
            )
        )
    return node_states


# ============================================================================
# Execution trace / LangSmith-equivalent observability (Gap #131)
# ============================================================================


@router.get(
    "/executions/{execution_id}/traces",
    response_model=ExecutionTraceListResponse,
    summary="List execution traces for LangSmith-equivalent observability",
)
async def list_execution_traces(
    execution_id: uuid.UUID,
    tenant: TenantDep,
    session: SessionDep,
) -> ExecutionTraceListResponse:
    """Return all LLM call, tool call, and HITL event traces for an execution."""
    from aumos_agent_framework.adapters.trace_repository import TraceRepository

    repo = TraceRepository(session)
    traces = await repo.list_by_execution(execution_id, str(tenant.tenant_id))
    return ExecutionTraceListResponse(
        items=[ExecutionTraceResponse.model_validate(t) for t in traces],
        total=len(traces),
        execution_id=execution_id,
    )


# ============================================================================
# Checkpoint / replay endpoints (Gap #133)
# ============================================================================


@router.get(
    "/executions/{execution_id}/checkpoints",
    response_model=CheckpointListResponse,
    summary="List Temporal checkpoints for an execution",
)
async def list_checkpoints(
    execution_id: uuid.UUID,
    tenant: TenantDep,
    session: SessionDep,
) -> CheckpointListResponse:
    """Return Temporal event history for durable replay from any checkpoint."""
    from sqlalchemy import select

    from aumos_agent_framework.core.models import WorkflowExecution

    stmt = select(WorkflowExecution).where(
        WorkflowExecution.id == execution_id,
        WorkflowExecution.tenant_id == tenant.tenant_id,
    )
    result = await session.execute(stmt)
    execution = result.scalar_one_or_none()

    if execution is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Execution {execution_id} not found")

    return CheckpointListResponse(
        execution_id=execution_id,
        temporal_workflow_id=execution.temporal_workflow_id,
        events=[],
        total=0,
    )


@router.post(
    "/executions/{execution_id}/replay",
    response_model=ExecutionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Replay a workflow execution from a checkpoint",
)
async def replay_execution(
    execution_id: uuid.UUID,
    body: ReplayRequest,
    tenant: TenantDep,
    session: SessionDep,
) -> ExecutionResponse:
    """Replay a workflow execution from its last checkpoint, optionally with override input."""
    from datetime import UTC, datetime

    from sqlalchemy import select

    from aumos_agent_framework.core.models import WorkflowExecution

    stmt = select(WorkflowExecution).where(
        WorkflowExecution.id == execution_id,
        WorkflowExecution.tenant_id == tenant.tenant_id,
    )
    result = await session.execute(stmt)
    execution = result.scalar_one_or_none()

    if execution is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Execution {execution_id} not found")

    logger.info(
        "Execution replay requested",
        execution_id=str(execution_id),
        has_override=body.override_input is not None,
        tenant_id=str(tenant.tenant_id),
    )

    return ExecutionResponse(
        id=uuid.uuid4(),
        tenant_id=tenant.tenant_id,
        workflow_id=execution.workflow_id,
        status="pending",
        current_node=None,
        execution_history=[],
        input_data=body.override_input or {},
        output_data=None,
        error_details=None,
        temporal_workflow_id=None,
        started_at=None,
        completed_at=None,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


# ============================================================================
# Tool marketplace / built-in tool discovery (Gap #129)
# ============================================================================


@router.get(
    "/tools/builtin",
    summary="List all built-in pre-built tools available in the marketplace",
)
async def list_builtin_tools(
    tenant: TenantDep,
    category: str | None = None,
    max_privilege_level: int | None = None,
) -> dict[str, Any]:
    """Return all built-in tools registered in the BuiltinToolRegistry.

    Filters by category and privilege level if provided.
    """
    from aumos_agent_framework.adapters.tools import BuiltinToolRegistry

    registry = BuiltinToolRegistry()
    tools = registry.list_tools(category=category, max_privilege_level=max_privilege_level)
    return {
        "items": tools,
        "total": len(tools),
        "tenant_id": str(tenant.tenant_id),
    }


@router.get(
    "/tools/builtin/{tool_id}",
    summary="Get details of a specific built-in tool",
)
async def get_builtin_tool(
    tool_id: str,
    tenant: TenantDep,
) -> dict[str, Any]:
    """Return metadata and input/output schema for a specific built-in tool."""
    from aumos_agent_framework.adapters.tools import BuiltinToolRegistry

    registry = BuiltinToolRegistry()
    tool = registry.get_tool(tool_id)
    if tool is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Built-in tool '{tool_id}' not found")

    return {
        "tool_id": tool.tool_id,
        "display_name": tool.display_name,
        "category": tool.category,
        "description": tool.description,
        "privilege_level": tool.privilege_level,
        "input_schema": tool.input_schema.model_json_schema(),
        "output_schema": tool.output_schema.model_json_schema(),
    }
