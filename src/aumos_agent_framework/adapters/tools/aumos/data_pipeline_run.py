"""AumOS data pipeline trigger tool — start a data pipeline job.

Privilege level 2 — triggers an internal aumos-data-pipeline job.
"""

from typing import Any

import httpx
from pydantic import Field

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class DataPipelineRunInput(ToolInputSchema):
    """Input schema for the data pipeline run tool."""

    pipeline_id: str = Field(..., description="ID of the data pipeline to run")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Runtime parameters to pass to the pipeline job",
    )
    wait_for_completion: bool = Field(
        default=False,
        description="If True, poll until the pipeline job finishes (up to 10 minutes)",
    )


class DataPipelineRunOutput(ToolOutputSchema):
    """Output schema for the data pipeline run tool."""

    result: dict[str, Any] = Field(description="Pipeline job details including job_id and status")
    metadata: dict[str, Any] = Field(default_factory=dict)


class DataPipelineRunTool:
    """Trigger an aumos-data-pipeline job from within an agent workflow.

    Calls the data-pipeline service REST API to start a pipeline run.
    Requires 'DATA_PIPELINE_URL' in config.
    """

    tool_id: str = "data_pipeline_run"
    display_name: str = "AumOS Data Pipeline Run"
    category: str = "aumos"
    description: str = "Trigger an aumos-data-pipeline job by pipeline ID, optionally waiting for completion."
    privilege_level: int = 2
    input_schema: type[DataPipelineRunInput] = DataPipelineRunInput
    output_schema: type[DataPipelineRunOutput] = DataPipelineRunOutput

    async def execute(
        self,
        input_data: DataPipelineRunInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> DataPipelineRunOutput:
        """Trigger a data pipeline job.

        Args:
            input_data: Pipeline ID, runtime parameters, and wait flag.
            tenant_id: Tenant context passed to the downstream API.
            config: Must contain 'DATA_PIPELINE_URL'; optionally 'SERVICE_TOKEN'.

        Returns:
            DataPipelineRunOutput with job ID and status.
        """
        import asyncio

        base_url = config.get("DATA_PIPELINE_URL", "").rstrip("/")
        if not base_url:
            return DataPipelineRunOutput(
                result={},
                metadata={"error": "DATA_PIPELINE_URL not configured"},
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
                f"{base_url}/pipelines/{input_data.pipeline_id}/runs",
                headers=headers,
                json={"parameters": input_data.parameters},
            )
            response.raise_for_status()
            job_data: dict[str, Any] = response.json()

        job_id: str = job_data.get("job_id", "")

        if input_data.wait_for_completion and job_id:
            async with httpx.AsyncClient(timeout=10.0) as client:
                for _ in range(120):
                    await asyncio.sleep(5)
                    status_response = await client.get(
                        f"{base_url}/runs/{job_id}",
                        headers=headers,
                    )
                    if status_response.status_code == 200:
                        status_data: dict[str, Any] = status_response.json()
                        if status_data.get("status") in ("completed", "failed", "cancelled"):
                            job_data.update(status_data)
                            break

        logger.info(
            "Data pipeline triggered",
            pipeline_id=input_data.pipeline_id,
            job_id=job_id,
            tenant_id=tenant_id,
        )

        return DataPipelineRunOutput(
            result=job_data,
            metadata={"pipeline_id": input_data.pipeline_id, "job_id": job_id},
        )


TOOL = DataPipelineRunTool()
