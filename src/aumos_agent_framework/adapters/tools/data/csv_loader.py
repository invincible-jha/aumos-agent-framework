"""CSV loader tool — fetches a CSV from a URL or file path and returns rows.

Privilege level 1 — safe read-only data loading.
"""

from typing import Any

import httpx
from pydantic import Field

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class CSVLoaderInput(ToolInputSchema):
    """Input schema for the CSV loader tool."""

    source_url: str = Field(..., description="URL to fetch the CSV from")
    max_rows: int = Field(default=100, ge=1, le=10000, description="Maximum rows to return")
    delimiter: str = Field(default=",", max_length=1, description="CSV field delimiter character")


class CSVLoaderOutput(ToolOutputSchema):
    """Output schema for the CSV loader tool."""

    result: list[dict[str, Any]] = Field(description="List of row dicts with column headers as keys")
    metadata: dict[str, Any] = Field(default_factory=dict)


class CSVLoaderTool:
    """Load a CSV file from a URL and return its rows as a list of dicts.

    Each row is represented as a dict with column names as keys.
    """

    tool_id: str = "csv_loader"
    display_name: str = "CSV Loader"
    category: str = "data"
    description: str = "Fetch a CSV file from a URL and return its rows as structured data. Useful for data analysis workflows."
    privilege_level: int = 1
    input_schema: type[CSVLoaderInput] = CSVLoaderInput
    output_schema: type[CSVLoaderOutput] = CSVLoaderOutput

    async def execute(
        self,
        input_data: CSVLoaderInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> CSVLoaderOutput:
        """Fetch and parse a CSV file.

        Args:
            input_data: Source URL, row limit, and delimiter.
            tenant_id: Tenant context.
            config: Optional bearer token via 'AUTH_TOKEN'.

        Returns:
            CSVLoaderOutput with parsed rows.
        """
        import csv
        import io

        headers: dict[str, str] = {}
        auth_token = config.get("AUTH_TOKEN")
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(input_data.source_url, headers=headers)
            response.raise_for_status()

        reader = csv.DictReader(
            io.StringIO(response.text),
            delimiter=input_data.delimiter,
        )
        rows: list[dict[str, Any]] = []
        for i, row in enumerate(reader):
            if i >= input_data.max_rows:
                break
            rows.append(dict(row))

        logger.info(
            "CSV loaded",
            source_url=input_data.source_url,
            rows_loaded=len(rows),
            tenant_id=tenant_id,
        )

        return CSVLoaderOutput(
            result=rows,
            metadata={"source_url": input_data.source_url, "rows_loaded": len(rows)},
        )


TOOL = CSVLoaderTool()
