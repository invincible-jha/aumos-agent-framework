"""Generic authenticated JSON API caller tool.

Privilege level 2 — read or write JSON endpoints with optional auth.
"""

from typing import Any

import httpx
from pydantic import Field

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class JSONAPIInput(ToolInputSchema):
    """Input schema for the JSON API caller tool."""

    url: str = Field(..., description="Full URL of the JSON API endpoint")
    method: str = Field(
        default="GET",
        description="HTTP method: GET, POST, PUT, PATCH, DELETE",
    )
    body: dict[str, Any] | None = Field(default=None, description="JSON request body (POST/PUT/PATCH)")
    query_params: dict[str, str] | None = Field(default=None, description="URL query parameters")
    extra_headers: dict[str, str] | None = Field(default=None, description="Additional HTTP headers")


class JSONAPIOutput(ToolOutputSchema):
    """Output schema for the JSON API caller tool."""

    result: dict[str, Any] | list[Any] = Field(description="Parsed JSON response body")
    metadata: dict[str, Any] = Field(default_factory=dict)


class JSONAPITool:
    """Call an external JSON API endpoint with optional authentication.

    Supports Bearer token auth via config key 'AUTH_TOKEN', or API key via 'API_KEY'
    header configured as 'API_KEY_HEADER_NAME' + 'API_KEY'.
    """

    tool_id: str = "json_api"
    display_name: str = "JSON API Caller"
    category: str = "data"
    description: str = "Make authenticated HTTP requests to JSON API endpoints and return the parsed response."
    privilege_level: int = 2
    input_schema: type[JSONAPIInput] = JSONAPIInput
    output_schema: type[JSONAPIOutput] = JSONAPIOutput

    async def execute(
        self,
        input_data: JSONAPIInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> JSONAPIOutput:
        """Execute a JSON API HTTP request.

        Args:
            input_data: URL, method, body, and optional headers.
            tenant_id: Tenant context for logging.
            config: Optional 'AUTH_TOKEN', 'API_KEY', 'API_KEY_HEADER_NAME'.

        Returns:
            JSONAPIOutput with parsed response body.
        """
        headers: dict[str, str] = {"Accept": "application/json", "Content-Type": "application/json"}

        if input_data.extra_headers:
            headers.update(input_data.extra_headers)

        auth_token = config.get("AUTH_TOKEN")
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        api_key = config.get("API_KEY")
        api_key_header = config.get("API_KEY_HEADER_NAME", "X-API-Key")
        if api_key:
            headers[api_key_header] = api_key

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.request(
                method=input_data.method.upper(),
                url=input_data.url,
                headers=headers,
                params=input_data.query_params,
                json=input_data.body,
            )
            response.raise_for_status()
            data: dict[str, Any] | list[Any] = response.json()

        logger.info(
            "JSON API call executed",
            url=input_data.url,
            method=input_data.method,
            status_code=response.status_code,
            tenant_id=tenant_id,
        )

        return JSONAPIOutput(
            result=data,
            metadata={
                "url": input_data.url,
                "method": input_data.method,
                "status_code": response.status_code,
            },
        )


TOOL = JSONAPITool()
