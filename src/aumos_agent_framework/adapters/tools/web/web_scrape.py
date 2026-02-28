"""Web scraping tool using httpx + BeautifulSoup.

Privilege level 1 — safe read-only HTTP fetch.
"""

from typing import Any

import httpx
from pydantic import Field, HttpUrl

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class WebScrapeInput(ToolInputSchema):
    """Input schema for the web scraping tool."""

    url: str = Field(..., description="URL to scrape")
    css_selector: str | None = Field(
        None,
        description="Optional CSS selector to extract specific content. If omitted, returns all text.",
    )
    max_length: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum characters to return from scraped content",
    )


class WebScrapeOutput(ToolOutputSchema):
    """Output schema for the web scraping tool."""

    result: str = Field(description="Scraped text content")
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebScrapeTool:
    """Scrape a web page and extract its text content.

    Uses httpx for HTTP fetching and beautifulsoup4 for HTML parsing.
    Returns cleaned text content, optionally filtered by CSS selector.
    """

    tool_id: str = "web_scrape"
    display_name: str = "Web Scraper"
    category: str = "web"
    description: str = "Fetch a URL and extract text content using HTML parsing. Optionally filter by CSS selector."
    privilege_level: int = 1
    input_schema: type[WebScrapeInput] = WebScrapeInput
    output_schema: type[WebScrapeOutput] = WebScrapeOutput

    async def execute(
        self,
        input_data: WebScrapeInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> WebScrapeOutput:
        """Fetch and parse a web page.

        Args:
            input_data: URL and optional CSS selector.
            tenant_id: Tenant context for rate limiting.
            config: Optional proxy settings via 'HTTP_PROXY'.

        Returns:
            WebScrapeOutput with extracted text.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return WebScrapeOutput(
                result="",
                metadata={"error": "beautifulsoup4 not installed"},
            )

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(input_data.url)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        if input_data.css_selector:
            elements = soup.select(input_data.css_selector)
            content = " ".join(el.get_text(separator=" ", strip=True) for el in elements)
        else:
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            content = soup.get_text(separator=" ", strip=True)

        content = content[: input_data.max_length]

        logger.info(
            "Web page scraped",
            url=input_data.url,
            content_length=len(content),
            tenant_id=tenant_id,
        )

        return WebScrapeOutput(
            result=content,
            metadata={
                "url": input_data.url,
                "status_code": response.status_code,
                "content_length": len(content),
            },
        )


TOOL = WebScrapeTool()
