"""AumOS workflow trigger tool — start another workflow via the agent-framework API.

Privilege level 2 — triggers an internal AumOS workflow execution.
"""

from typing import Any

import httpx
from pydantic import Field

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class WorkflowTriggerInput(ToolInputSchema):
    """Input schema for the workflow trigger tool."""

    workflow_id: str = Field(..., description="ID of the workflow definition to trigger")
    initial_state: dict[str, Any] = Field(
        default_factory=dict,
        description="Initial state dict to pass to the workflow",
    )
    wait_for_completion: bool = Field(
        default=False,
        description="If True, poll until the workflow finishes (up to 5 minutes)",
    )


class WorkflowTriggerOutput(ToolOutputSchema):
    """Output schema for the workflow trigger tool."""

    result: dict[str, Any] = Field(description="Workflow execution details including execution_id and status")
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowTriggerTool:
    """Trigger another AumOS workflow execution from within an agent.

    Calls the agent-framework /workflows/{id}/execute endpoint.
    Requires 'AGENT_FRAMEWORK_URL' in config; tenant JWT is passed via X-Tenant-ID header.
    """

    tool_id: str = "workflow_trigger"
    display_name: str = "AumOS Workflow Trigger"
    category: str = "aumos"
    description: str = "Trigger another AumOS workflow by ID, optionally passing initial state and waiting for completion."
    privilege_level: int = 2
    input_schema: type[WorkflowTriggerInput] = WorkflowTriggerInput
    output_schema: type[WorkflowTriggerOutput] = WorkflowTriggerOutput

    async def execute(
        self,
        input_data: WorkflowTriggerInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> WorkflowTriggerOutput:
        """Trigger a workflow execution via the agent-framework REST API.

        Args:
            input_data: Workflow ID, initial state, and wait flag.
            tenant_id: Tenant context passed to the downstream API.
            config: Must contain 'AGENT_FRAMEWORK_URL'; optionally 'SERVICE_TOKEN'.

        Returns:
            WorkflowTriggerOutput with execution ID and status.
        """
        import asyncio

        base_url = config.get("AGENT_FRAMEWORK_URL", "").rstrip("/")
        if not base_url:
            return WorkflowTriggerOutput(
                result={},
                metadata={"error": "AGENT_FRAMEWORK_URL not configured"},
            )

        service_token = config.get("SERVICE_TOKEN", "")
        headers: dict[str, str] = {
            "X-Tenant-ID": tenant_id,
            "Content-Type": "application/json",
        }
        if service_token:
            headers["Authorization"] = f"Bearer {service_token}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{base_url}/workflows/{input_data.workflow_id}/execute",
                headers=headers,
                json={"initial_state": input_data.initial_state},
            )
            response.raise_for_status()
            execution_data: dict[str, Any] = response.json()

        execution_id: str = execution_data.get("execution_id", "")

        if input_data.wait_for_completion and execution_id:
            async with httpx.AsyncClient(timeout=10.0) as client:
                for _ in range(60):
                    await asyncio.sleep(5)
                    status_response = await client.get(
                        f"{base_url}/workflows/{input_data.workflow_id}/status",
                        headers=headers,
                        params={"execution_id": execution_id},
                    )
                    if status_response.status_code == 200:
                        status_data: dict[str, Any] = status_response.json()
                        if status_data.get("status") in ("completed", "failed", "cancelled"):
                            execution_data.update(status_data)
                            break

        logger.info(
            "Workflow triggered",
            workflow_id=input_data.workflow_id,
            execution_id=execution_id,
            tenant_id=tenant_id,
        )

        return WorkflowTriggerOutput(
            result=execution_data,
            metadata={"workflow_id": input_data.workflow_id, "execution_id": execution_id},
        )


TOOL = WorkflowTriggerTool()
