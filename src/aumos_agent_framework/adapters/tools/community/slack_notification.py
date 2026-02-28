"""Community integration: Slack notification via incoming webhook.

Privilege level 2 — posts a notification to a preconfigured Slack webhook URL.
No bot token required — simpler than the full slack_message tool.
"""

from typing import Any

import httpx
from pydantic import Field

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class SlackNotificationInput(ToolInputSchema):
    """Input schema for the Slack notification tool."""

    message: str = Field(..., max_length=4000, description="Notification message text")
    title: str | None = Field(default=None, max_length=200, description="Optional bold title above the message")
    color: str = Field(
        default="#36a64f",
        description="Attachment color hex code: green=#36a64f, red=#ff0000, blue=#0055ff",
    )


class SlackNotificationOutput(ToolOutputSchema):
    """Output schema for the Slack notification tool."""

    result: dict[str, Any] = Field(description="HTTP response status from Slack webhook")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SlackNotificationTool:
    """Send a formatted notification to Slack via an incoming webhook URL.

    Requires 'SLACK_WEBHOOK_URL' in config — no token management needed.
    Ideal for simple alerting and notification use cases.
    """

    tool_id: str = "slack_notification"
    display_name: str = "Slack Notification (Webhook)"
    category: str = "community"
    description: str = "Post a formatted notification to Slack using an incoming webhook URL. No bot token required."
    privilege_level: int = 2
    input_schema: type[SlackNotificationInput] = SlackNotificationInput
    output_schema: type[SlackNotificationOutput] = SlackNotificationOutput

    async def execute(
        self,
        input_data: SlackNotificationInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> SlackNotificationOutput:
        """Post a notification to Slack via incoming webhook.

        Args:
            input_data: Message text, optional title, and color.
            tenant_id: Tenant context for audit logging.
            config: Must contain 'SLACK_WEBHOOK_URL'.

        Returns:
            SlackNotificationOutput with HTTP response status.
        """
        webhook_url = config.get("SLACK_WEBHOOK_URL", "")
        if not webhook_url:
            return SlackNotificationOutput(result={}, metadata={"error": "SLACK_WEBHOOK_URL not configured"})

        attachment: dict[str, Any] = {"color": input_data.color, "text": input_data.message}
        if input_data.title:
            attachment["title"] = input_data.title

        payload: dict[str, Any] = {"attachments": [attachment]}

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(webhook_url, json=payload)

        logger.info(
            "Slack notification sent",
            status_code=response.status_code,
            tenant_id=tenant_id,
        )

        return SlackNotificationOutput(
            result={"status_code": response.status_code, "ok": response.status_code == 200},
            metadata={"webhook_url_configured": True},
        )


TOOL = SlackNotificationTool()
