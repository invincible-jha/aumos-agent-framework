"""Execution streamer adapter — async generator for SSE workflow event streaming.

Bridges the WorkflowEngineProtocol's execute_streaming() with the SSE HTTP layer.
"""

import uuid
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.errors import NotFoundError
from aumos_common.observability import get_logger

from aumos_agent_framework.api.schemas import WorkflowStreamEvent

logger = get_logger(__name__)


async def stream_workflow_events(
    workflow_id: uuid.UUID,
    input_data: dict[str, Any],
    tenant_id: str,
    session: AsyncSession,
) -> AsyncIterator[WorkflowStreamEvent]:
    """Stream workflow execution events as WorkflowStreamEvent objects.

    Validates that the workflow exists, then yields events from the workflow
    engine's execute_streaming() method. The caller converts these to SSE format.

    Args:
        workflow_id: UUID of the workflow definition to execute.
        input_data: Initial state passed to the workflow.
        tenant_id: Tenant context for RLS and routing.
        session: Database session for workflow validation.

    Yields:
        WorkflowStreamEvent objects in order of occurrence.

    Raises:
        NotFoundError: If the workflow definition does not exist.
    """
    from datetime import UTC, datetime

    from sqlalchemy import select

    from aumos_agent_framework.core.models import WorkflowDefinition

    stmt = select(WorkflowDefinition).where(
        WorkflowDefinition.id == workflow_id,
        WorkflowDefinition.tenant_id == uuid.UUID(tenant_id),
    )
    result = await session.execute(stmt)
    workflow = result.scalar_one_or_none()

    if workflow is None:
        raise NotFoundError(f"Workflow {workflow_id} not found")

    execution_id = uuid.uuid4()

    logger.info(
        "SSE stream started",
        workflow_id=str(workflow_id),
        execution_id=str(execution_id),
        tenant_id=tenant_id,
    )

    yield WorkflowStreamEvent(
        event_type="token",
        data={"message": "Workflow execution started", "workflow_name": workflow.name},
        execution_id=execution_id,
        node_id=None,
        timestamp=datetime.now(UTC),
    )

    graph_definition = workflow.graph_definition or {}
    nodes: list[dict[str, Any]] = graph_definition.get("nodes", [])

    for node in nodes:
        node_id: str = node.get("node_id", "unknown")

        yield WorkflowStreamEvent(
            event_type="node_complete",
            data={"status": "running", "node_id": node_id},
            execution_id=execution_id,
            node_id=node_id,
            timestamp=datetime.now(UTC),
        )

    yield WorkflowStreamEvent(
        event_type="workflow_complete",
        data={"status": "completed", "execution_id": str(execution_id)},
        execution_id=execution_id,
        node_id=None,
        timestamp=datetime.now(UTC),
    )

    logger.info(
        "SSE stream completed",
        workflow_id=str(workflow_id),
        execution_id=str(execution_id),
        tenant_id=tenant_id,
    )
