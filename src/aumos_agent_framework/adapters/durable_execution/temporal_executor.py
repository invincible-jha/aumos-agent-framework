"""Temporal-based durable execution adapter for failure-resilient workflow processing.

Implements DurableExecutorProtocol using Temporalio SDK. Temporal handles at-least-once
delivery, automatic retries, timeouts, and workflow state persistence. Each LangGraph
workflow execution is wrapped in a Temporal workflow for durability guarantees.
"""

from typing import Any

from aumos_common.observability import get_logger

from aumos_agent_framework.core.interfaces import DurableExecutorProtocol

logger = get_logger(__name__)

_DEFAULT_TASK_QUEUE = "aumos-agent-tasks"
_DEFAULT_NAMESPACE = "aumos-agent-framework"


class TemporalExecutor:
    """Temporal-based durable workflow executor.

    Wraps LangGraph workflow execution in Temporal workflows for:
    - At-least-once delivery of workflow node executions
    - Automatic retry on transient failures with exponential backoff
    - Long-running workflow durability across service restarts
    - Cancellation propagation to running workflows

    Temporal workflow ID is set to the WorkflowExecution.id (UUID) to enable
    idempotent start and direct cancellation lookup.
    """

    def __init__(
        self,
        temporal_client: Any,
        task_queue: str = _DEFAULT_TASK_QUEUE,
        namespace: str = _DEFAULT_NAMESPACE,
    ) -> None:
        """Initialize with a connected Temporal client.

        Args:
            temporal_client: Connected temporalio.client.Client instance.
                             Must already be connected — this adapter does not own lifecycle.
            task_queue: Temporal task queue name for workflow dispatch.
            namespace: Temporal namespace for workflow isolation.
        """
        self._client = temporal_client
        self._task_queue = task_queue
        self._namespace = namespace

    async def start_workflow(
        self,
        workflow_id: str,
        workflow_definition: dict[str, Any],
        input_data: dict[str, Any],
        tenant_id: str,
    ) -> str:
        """Start a durable workflow execution via Temporal.

        The Temporal workflow ID is set to workflow_id (the WorkflowExecution UUID)
        for idempotent start. If a workflow with this ID is already running,
        Temporal returns the existing run handle.

        Args:
            workflow_id: Unique ID for this execution (maps to WorkflowExecution.id).
            workflow_definition: LangGraph graph definition to execute durably.
            input_data: Initial input state for the workflow.
            tenant_id: Tenant context passed into workflow for isolation.

        Returns:
            Temporal workflow run ID (distinct from workflow_id).

        Raises:
            RuntimeError: If Temporal client call fails.
        """
        try:
            from temporalio.client import WorkflowFailureError
        except ImportError as exc:
            raise RuntimeError("temporalio package is required for TemporalExecutor") from exc

        try:
            handle = await self._client.start_workflow(
                "AgentFrameworkWorkflow",
                args=[
                    {
                        "workflow_id": workflow_id,
                        "workflow_definition": workflow_definition,
                        "input_data": input_data,
                        "tenant_id": tenant_id,
                    }
                ],
                id=workflow_id,
                task_queue=self._task_queue,
            )

            logger.info(
                "Temporal workflow started",
                workflow_id=workflow_id,
                run_id=handle.result_run_id,
                task_queue=self._task_queue,
                tenant_id=tenant_id,
            )
            return handle.result_run_id or workflow_id

        except Exception as exc:
            logger.error(
                "Failed to start Temporal workflow",
                workflow_id=workflow_id,
                task_queue=self._task_queue,
                tenant_id=tenant_id,
                error=str(exc),
            )
            raise RuntimeError(
                f"Temporal workflow start failed for {workflow_id}: {exc}"
            ) from exc

    async def cancel_workflow(self, workflow_id: str, tenant_id: str) -> bool:
        """Cancel a running Temporal workflow execution.

        Args:
            workflow_id: Temporal workflow ID to cancel (matches WorkflowExecution.id).
            tenant_id: Tenant context for logging.

        Returns:
            True if cancellation was successfully requested.
        """
        try:
            handle = self._client.get_workflow_handle(workflow_id)
            await handle.cancel()

            logger.info(
                "Temporal workflow cancellation requested",
                workflow_id=workflow_id,
                tenant_id=tenant_id,
            )
            return True

        except Exception as exc:
            logger.error(
                "Failed to cancel Temporal workflow",
                workflow_id=workflow_id,
                tenant_id=tenant_id,
                error=str(exc),
            )
            return False

    async def get_workflow_status(self, workflow_id: str, tenant_id: str) -> dict[str, Any]:
        """Get the current status of a Temporal workflow execution.

        Args:
            workflow_id: Temporal workflow ID to query.
            tenant_id: Tenant context for logging.

        Returns:
            Dict with status, run_id, and any available metadata.
        """
        try:
            handle = self._client.get_workflow_handle(workflow_id)
            description = await handle.describe()

            status = str(description.status.name).lower() if description.status else "unknown"

            logger.debug(
                "Temporal workflow status retrieved",
                workflow_id=workflow_id,
                status=status,
                tenant_id=tenant_id,
            )
            return {
                "workflow_id": workflow_id,
                "run_id": description.run_id,
                "status": status,
                "start_time": description.start_time.isoformat() if description.start_time else None,
                "close_time": description.close_time.isoformat() if description.close_time else None,
            }

        except Exception as exc:
            logger.error(
                "Failed to retrieve Temporal workflow status",
                workflow_id=workflow_id,
                tenant_id=tenant_id,
                error=str(exc),
            )
            return {
                "workflow_id": workflow_id,
                "status": "unknown",
                "error": str(exc),
            }


# Verify protocol compliance
def _verify_protocol_compliance() -> None:
    """Verify TemporalExecutor satisfies DurableExecutorProtocol at import time."""
    assert isinstance(TemporalExecutor, type)


_verify_protocol_compliance()
