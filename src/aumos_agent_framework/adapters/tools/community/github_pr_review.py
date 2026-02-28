"""Community integration: GitHub PR review comment tool.

Privilege level 3 — posts review comments to a GitHub pull request.
"""

from typing import Any

import httpx
from pydantic import Field

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class GitHubPRReviewInput(ToolInputSchema):
    """Input schema for the GitHub PR review tool."""

    owner: str = Field(..., description="GitHub repository owner (org or user)")
    repo: str = Field(..., description="GitHub repository name")
    pull_number: int = Field(..., ge=1, description="Pull request number")
    body: str = Field(..., max_length=65536, description="Review comment body (Markdown supported)")
    event: str = Field(
        default="COMMENT",
        description="Review event: COMMENT (no approval), APPROVE, or REQUEST_CHANGES",
    )


class GitHubPRReviewOutput(ToolOutputSchema):
    """Output schema for the GitHub PR review tool."""

    result: dict[str, Any] = Field(description="GitHub API response with review ID and URL")
    metadata: dict[str, Any] = Field(default_factory=dict)


class GitHubPRReviewTool:
    """Post a review comment to a GitHub pull request.

    Requires 'GITHUB_TOKEN' in config (classic PAT with repo scope or
    fine-grained token with pull_requests:write permission).
    """

    tool_id: str = "github_pr_review"
    display_name: str = "GitHub PR Review"
    category: str = "community"
    description: str = "Post a review comment (approve, request changes, or comment) on a GitHub pull request."
    privilege_level: int = 3
    input_schema: type[GitHubPRReviewInput] = GitHubPRReviewInput
    output_schema: type[GitHubPRReviewOutput] = GitHubPRReviewOutput

    async def execute(
        self,
        input_data: GitHubPRReviewInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> GitHubPRReviewOutput:
        """Post a review to a GitHub pull request.

        Args:
            input_data: Repo owner, name, PR number, body, and event type.
            tenant_id: Tenant context for audit logging.
            config: Must contain 'GITHUB_TOKEN'.

        Returns:
            GitHubPRReviewOutput with review ID and HTML URL.
        """
        github_token = config.get("GITHUB_TOKEN", "")
        if not github_token:
            return GitHubPRReviewOutput(result={}, metadata={"error": "GITHUB_TOKEN not configured"})

        url = f"https://api.github.com/repos/{input_data.owner}/{input_data.repo}/pulls/{input_data.pull_number}/reviews"
        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                url,
                headers=headers,
                json={"body": input_data.body, "event": input_data.event},
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()

        logger.info(
            "GitHub PR review posted",
            owner=input_data.owner,
            repo=input_data.repo,
            pull_number=input_data.pull_number,
            event=input_data.event,
            review_id=data.get("id"),
            tenant_id=tenant_id,
        )

        return GitHubPRReviewOutput(
            result={"id": data.get("id"), "html_url": data.get("html_url"), "state": data.get("state")},
            metadata={"pull_number": input_data.pull_number, "event": input_data.event},
        )


TOOL = GitHubPRReviewTool()
