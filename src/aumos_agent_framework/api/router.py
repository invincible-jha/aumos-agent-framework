"""FastAPI router for aumos-agent-framework.

All routes are thin — they validate input, delegate to services, and return responses.
Business logic lives exclusively in core/services.py.
"""

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
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
    ExecutionResponse,
    ExecutionStatusResponse,
    HITLApprovalResponse,
    HITLApproveRequest,
    HITLPendingListResponse,
    HITLRejectRequest,
    WorkflowCreateRequest,
    WorkflowExecuteRequest,
    WorkflowResponse,
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
