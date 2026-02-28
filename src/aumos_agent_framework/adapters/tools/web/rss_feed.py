"""RSS/Atom feed reader tool.

Privilege level 1 — safe read-only feed fetch.
"""

from typing import Any

import httpx
from pydantic import Field

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class RSSFeedInput(ToolInputSchema):
    """Input schema for the RSS feed reader."""

    feed_url: str = Field(..., description="URL of the RSS or Atom feed")
    max_items: int = Field(default=10, ge=1, le=100, description="Maximum feed items to return")


class RSSFeedOutput(ToolOutputSchema):
    """Output schema for the RSS feed reader."""

    result: list[dict[str, Any]] = Field(description="List of feed items with title, link, summary, published")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RSSFeedTool:
    """Read and parse an RSS or Atom feed.

    Returns a list of feed items with title, URL, summary, and publication date.
    """

    tool_id: str = "rss_feed"
    display_name: str = "RSS/Atom Feed Reader"
    category: str = "web"
    description: str = "Fetch and parse an RSS or Atom feed URL, returning the latest items with titles and summaries."
    privilege_level: int = 1
    input_schema: type[RSSFeedInput] = RSSFeedInput
    output_schema: type[RSSFeedOutput] = RSSFeedOutput

    async def execute(
        self,
        input_data: RSSFeedInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> RSSFeedOutput:
        """Fetch and parse an RSS/Atom feed.

        Args:
            input_data: Feed URL and item count limit.
            tenant_id: Tenant context.
            config: Unused for this tool.

        Returns:
            RSSFeedOutput with parsed feed items.
        """
        try:
            import feedparser
        except ImportError:
            return RSSFeedOutput(result=[], metadata={"error": "feedparser not installed"})

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(input_data.feed_url)
            response.raise_for_status()
            raw_content = response.text

        feed = feedparser.parse(raw_content)
        items = []
        for entry in feed.entries[: input_data.max_items]:
            items.append(
                {
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "summary": entry.get("summary", ""),
                    "published": entry.get("published", ""),
                }
            )

        logger.info(
            "RSS feed fetched",
            feed_url=input_data.feed_url,
            items_count=len(items),
            tenant_id=tenant_id,
        )

        return RSSFeedOutput(
            result=items,
            metadata={"feed_title": feed.feed.get("title", ""), "item_count": len(items)},
        )


TOOL = RSSFeedTool()
