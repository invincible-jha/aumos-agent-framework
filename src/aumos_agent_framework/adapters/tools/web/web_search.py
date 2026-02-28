"""Web search tool using the Serper API (Google Search).

Privilege level 1 — safe read-only web query.
"""

from typing import Any

import httpx
from pydantic import Field

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class WebSearchInput(ToolInputSchema):
    """Input schema for the web search tool."""

    query: str = Field(..., min_length=1, max_length=1000, description="Search query string")
    num_results: int = Field(default=5, ge=1, le=20, description="Number of results to return")


class WebSearchOutput(ToolOutputSchema):
    """Output schema for the web search tool."""

    result: list[dict[str, Any]] = Field(description="List of search result dicts with title, url, snippet")
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebSearchTool:
    """Web search via the Serper API (Google Search results).

    Requires config key 'SERPER_API_KEY' to be set in per-tenant tool config.
    """

    tool_id: str = "web_search_serper"
    display_name: str = "Web Search (Serper)"
    category: str = "web"
    description: str = "Search the web using Google via the Serper API and return top results with titles, URLs, and snippets."
    privilege_level: int = 1
    input_schema: type[WebSearchInput] = WebSearchInput
    output_schema: type[WebSearchOutput] = WebSearchOutput

    async def execute(
        self,
        input_data: WebSearchInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> WebSearchOutput:
        """Execute a Google search via the Serper API.

        Args:
            input_data: Validated search query and result count.
            tenant_id: Tenant context for rate limiting.
            config: Must contain 'SERPER_API_KEY'.

        Returns:
            WebSearchOutput with search results.
        """
        api_key = config.get("SERPER_API_KEY", "")
        if not api_key:
            logger.warning("Serper API key not configured", tenant_id=tenant_id)
            return WebSearchOutput(result=[], metadata={"error": "SERPER_API_KEY not configured"})

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={"q": input_data.query, "num": input_data.num_results},
            )
            response.raise_for_status()
            data = response.json()

        results = [
            {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
            for r in data.get("organic", [])[: input_data.num_results]
        ]

        logger.info(
            "Web search executed",
            query=input_data.query,
            results_count=len(results),
            tenant_id=tenant_id,
        )

        return WebSearchOutput(
            result=results,
            metadata={"query": input_data.query, "total_results": len(results)},
        )


TOOL = WebSearchTool()
