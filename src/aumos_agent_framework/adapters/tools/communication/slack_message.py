"""Slack Web API message sender tool.

Privilege level 3 — sends an outbound message to a Slack channel or user.
"""

from typing import Any

import httpx
from pydantic import Field

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class SlackMessageInput(ToolInputSchema):
    """Input schema for the Slack message tool."""

    channel: str = Field(..., description="Slack channel ID or user ID to post to (e.g. C01234 or U01234)")
    text: str = Field(..., max_length=4000, description="Message text (supports Slack mrkdwn formatting)")
    thread_ts: str | None = Field(default=None, description="Thread timestamp to reply within a thread")
    username: str | None = Field(default=None, description="Override bot display name")
    icon_emoji: str | None = Field(default=None, description="Override bot icon emoji (e.g. :robot_face:)")


class SlackMessageOutput(ToolOutputSchema):
    """Output schema for the Slack message tool."""

    result: dict[str, Any] = Field(description="Slack API response with ts and channel")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SlackMessageTool:
    """Send a message to a Slack channel via the Slack Web API.

    Requires config key 'SLACK_BOT_TOKEN' (xoxb-...).
    """

    tool_id: str = "slack_message"
    display_name: str = "Slack Message"
    category: str = "communication"
    description: str = "Post a message to a Slack channel or DM a user using the Slack Web API."
    privilege_level: int = 3
    input_schema: type[SlackMessageInput] = SlackMessageInput
    output_schema: type[SlackMessageOutput] = SlackMessageOutput

    async def execute(
        self,
        input_data: SlackMessageInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> SlackMessageOutput:
        """Post a message to Slack.

        Args:
            input_data: Channel, text, and optional thread/username overrides.
            tenant_id: Tenant context for audit logging.
            config: Must contain 'SLACK_BOT_TOKEN'.

        Returns:
            SlackMessageOutput with Slack message timestamp.
        """
        bot_token = config.get("SLACK_BOT_TOKEN", "")
        if not bot_token:
            return SlackMessageOutput(result={}, metadata={"error": "SLACK_BOT_TOKEN not configured"})

        payload: dict[str, Any] = {
            "channel": input_data.channel,
            "text": input_data.text,
        }
        if input_data.thread_ts:
            payload["thread_ts"] = input_data.thread_ts
        if input_data.username:
            payload["username"] = input_data.username
        if input_data.icon_emoji:
            payload["icon_emoji"] = input_data.icon_emoji

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {bot_token}", "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()

        if not data.get("ok"):
            error = data.get("error", "unknown_error")
            logger.warning("Slack API error", error=error, tenant_id=tenant_id)
            return SlackMessageOutput(result=data, metadata={"error": error})

        logger.info(
            "Slack message sent",
            channel=input_data.channel,
            ts=data.get("ts"),
            tenant_id=tenant_id,
        )

        return SlackMessageOutput(
            result=data,
            metadata={"channel": input_data.channel, "ts": data.get("ts")},
        )


TOOL = SlackMessageTool()
