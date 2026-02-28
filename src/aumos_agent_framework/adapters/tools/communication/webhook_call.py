"""Generic webhook HTTP POST tool.

Privilege level 2 — outbound HTTP POST to a configured webhook URL.
"""

from typing import Any

import httpx
from pydantic import Field

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class WebhookCallInput(ToolInputSchema):
    """Input schema for the webhook call tool."""

    url: str = Field(..., description="Webhook endpoint URL to POST to")
    payload: dict[str, Any] = Field(..., description="JSON payload to send in the POST body")
    extra_headers: dict[str, str] | None = Field(default=None, description="Additional HTTP headers")
    timeout_seconds: float = Field(default=15.0, ge=1.0, le=60.0, description="Request timeout in seconds")


class WebhookCallOutput(ToolOutputSchema):
    """Output schema for the webhook call tool."""

    result: dict[str, Any] = Field(description="Webhook response details")
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebhookCallTool:
    """Send an HTTP POST request to a webhook URL with a JSON payload.

    Supports HMAC signature via config key 'WEBHOOK_SECRET' (signed as
    X-Aumos-Signature-256 using SHA-256 HMAC of the request body).
    """

    tool_id: str = "webhook_call"
    display_name: str = "Webhook Call"
    category: str = "communication"
    description: str = "POST a JSON payload to a webhook URL. Supports HMAC signing via WEBHOOK_SECRET config key."
    privilege_level: int = 2
    input_schema: type[WebhookCallInput] = WebhookCallInput
    output_schema: type[WebhookCallOutput] = WebhookCallOutput

    async def execute(
        self,
        input_data: WebhookCallInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> WebhookCallOutput:
        """POST a JSON payload to a webhook endpoint.

        Args:
            input_data: Target URL, payload, and optional headers.
            tenant_id: Tenant context for audit logging.
            config: Optional 'WEBHOOK_SECRET' for HMAC signing.

        Returns:
            WebhookCallOutput with HTTP status and response body.
        """
        import hashlib
        import hmac
        import json

        body_bytes = json.dumps(input_data.payload, separators=(",", ":")).encode()

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if input_data.extra_headers:
            headers.update(input_data.extra_headers)

        webhook_secret = config.get("WEBHOOK_SECRET")
        if webhook_secret:
            signature = hmac.new(webhook_secret.encode(), body_bytes, hashlib.sha256).hexdigest()
            headers["X-Aumos-Signature-256"] = f"sha256={signature}"

        async with httpx.AsyncClient(timeout=input_data.timeout_seconds) as client:
            response = await client.post(input_data.url, headers=headers, content=body_bytes)

        try:
            response_body: Any = response.json()
        except Exception:
            response_body = {"raw": response.text}

        logger.info(
            "Webhook called",
            url=input_data.url,
            status_code=response.status_code,
            tenant_id=tenant_id,
        )

        return WebhookCallOutput(
            result={"status_code": response.status_code, "body": response_body},
            metadata={"url": input_data.url, "status_code": response.status_code},
        )


TOOL = WebhookCallTool()
