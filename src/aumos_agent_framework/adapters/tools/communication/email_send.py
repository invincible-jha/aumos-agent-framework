"""Email sending tool via Resend API.

Privilege level 3 — triggers an outbound communication action.
"""

from typing import Any

import httpx
from pydantic import Field

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class EmailSendInput(ToolInputSchema):
    """Input schema for the email send tool."""

    to: list[str] = Field(..., description="List of recipient email addresses")
    subject: str = Field(..., max_length=998, description="Email subject line")
    html_body: str = Field(..., description="HTML body of the email")
    from_address: str = Field(
        default="noreply@aumos.ai",
        description="Sender address (must be verified in Resend)",
    )
    cc: list[str] = Field(default_factory=list, description="CC recipients")
    reply_to: str | None = Field(default=None, description="Reply-To address")


class EmailSendOutput(ToolOutputSchema):
    """Output schema for the email send tool."""

    result: dict[str, Any] = Field(description="Resend API response with message ID")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmailSendTool:
    """Send a transactional email via the Resend API.

    Requires config key 'RESEND_API_KEY'. Privilege level 3 means the
    workflow must pass a HITL gate or explicitly approve the action.
    """

    tool_id: str = "email_send"
    display_name: str = "Email Send (Resend)"
    category: str = "communication"
    description: str = "Send a transactional HTML email via the Resend API to one or more recipients."
    privilege_level: int = 3
    input_schema: type[EmailSendInput] = EmailSendInput
    output_schema: type[EmailSendOutput] = EmailSendOutput

    async def execute(
        self,
        input_data: EmailSendInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> EmailSendOutput:
        """Send an email via Resend.

        Args:
            input_data: Recipients, subject, and HTML body.
            tenant_id: Tenant context for audit logging.
            config: Must contain 'RESEND_API_KEY'.

        Returns:
            EmailSendOutput with Resend message ID.
        """
        api_key = config.get("RESEND_API_KEY", "")
        if not api_key:
            return EmailSendOutput(result={}, metadata={"error": "RESEND_API_KEY not configured"})

        payload: dict[str, Any] = {
            "from": input_data.from_address,
            "to": input_data.to,
            "subject": input_data.subject,
            "html": input_data.html_body,
        }
        if input_data.cc:
            payload["cc"] = input_data.cc
        if input_data.reply_to:
            payload["reply_to"] = input_data.reply_to

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()

        logger.info(
            "Email sent",
            to=input_data.to,
            subject=input_data.subject,
            message_id=data.get("id"),
            tenant_id=tenant_id,
        )

        return EmailSendOutput(
            result=data,
            metadata={"message_id": data.get("id"), "recipient_count": len(input_data.to)},
        )


TOOL = EmailSendTool()
