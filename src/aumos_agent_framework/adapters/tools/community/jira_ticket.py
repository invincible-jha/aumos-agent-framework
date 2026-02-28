"""Community integration: Jira ticket creation tool.

Privilege level 2 — creates a Jira issue in the configured project.
"""

from typing import Any

import httpx
from pydantic import Field

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class JiraTicketInput(ToolInputSchema):
    """Input schema for the Jira ticket creation tool."""

    project_key: str = Field(..., description="Jira project key (e.g. 'ENG', 'OPS')")
    summary: str = Field(..., max_length=255, description="Issue summary / title")
    description: str = Field(default="", max_length=32767, description="Issue description (plain text or ADF JSON)")
    issue_type: str = Field(default="Task", description="Issue type: Task, Bug, Story, Epic")
    priority: str = Field(default="Medium", description="Priority: Highest, High, Medium, Low, Lowest")
    assignee_account_id: str | None = Field(default=None, description="Jira account ID to assign the issue to")
    labels: list[str] = Field(default_factory=list, description="Labels to attach to the issue")


class JiraTicketOutput(ToolOutputSchema):
    """Output schema for the Jira ticket creation tool."""

    result: dict[str, Any] = Field(description="Created Jira issue with ID, key, and URL")
    metadata: dict[str, Any] = Field(default_factory=dict)


class JiraTicketTool:
    """Create a Jira issue using the Jira REST API v3.

    Requires 'JIRA_BASE_URL', 'JIRA_USER_EMAIL', and 'JIRA_API_TOKEN' in config.
    Authentication uses HTTP Basic Auth (email:api_token).
    """

    tool_id: str = "jira_ticket"
    display_name: str = "Jira Ticket Creator"
    category: str = "community"
    description: str = "Create a Jira issue in a specified project using the Jira REST API v3."
    privilege_level: int = 2
    input_schema: type[JiraTicketInput] = JiraTicketInput
    output_schema: type[JiraTicketOutput] = JiraTicketOutput

    async def execute(
        self,
        input_data: JiraTicketInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> JiraTicketOutput:
        """Create a Jira issue.

        Args:
            input_data: Project key, summary, description, and metadata.
            tenant_id: Tenant context for audit logging.
            config: Must contain 'JIRA_BASE_URL', 'JIRA_USER_EMAIL', 'JIRA_API_TOKEN'.

        Returns:
            JiraTicketOutput with created issue ID, key, and self URL.
        """
        import base64

        base_url = config.get("JIRA_BASE_URL", "").rstrip("/")
        user_email = config.get("JIRA_USER_EMAIL", "")
        api_token = config.get("JIRA_API_TOKEN", "")

        if not all([base_url, user_email, api_token]):
            return JiraTicketOutput(
                result={},
                metadata={"error": "JIRA_BASE_URL, JIRA_USER_EMAIL, and JIRA_API_TOKEN are required"},
            )

        credentials = base64.b64encode(f"{user_email}:{api_token}".encode()).decode()
        headers = {
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        fields: dict[str, Any] = {
            "project": {"key": input_data.project_key},
            "summary": input_data.summary,
            "issuetype": {"name": input_data.issue_type},
            "priority": {"name": input_data.priority},
        }

        if input_data.description:
            fields["description"] = {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": input_data.description}],
                    }
                ],
            }

        if input_data.assignee_account_id:
            fields["assignee"] = {"accountId": input_data.assignee_account_id}

        if input_data.labels:
            fields["labels"] = input_data.labels

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{base_url}/rest/api/3/issue",
                headers=headers,
                json={"fields": fields},
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()

        logger.info(
            "Jira ticket created",
            project_key=input_data.project_key,
            issue_key=data.get("key"),
            issue_id=data.get("id"),
            tenant_id=tenant_id,
        )

        return JiraTicketOutput(
            result={"id": data.get("id"), "key": data.get("key"), "self": data.get("self")},
            metadata={"project_key": input_data.project_key, "issue_type": input_data.issue_type},
        )


TOOL = JiraTicketTool()
