"""HTML to Markdown conversion tool.

Privilege level 1 — safe read-only content transformation.
"""

from typing import Any

import httpx
from pydantic import Field

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class HTMLToMarkdownInput(ToolInputSchema):
    """Input schema for the HTML to Markdown tool."""

    source: str = Field(
        ...,
        description="URL to fetch HTML from, or raw HTML string (detected by '<' prefix)",
    )
    max_chars: int = Field(
        default=20000,
        ge=500,
        le=200000,
        description="Maximum characters of Markdown to return",
    )
    strip_images: bool = Field(default=False, description="Whether to strip image tags from output")


class HTMLToMarkdownOutput(ToolOutputSchema):
    """Output schema for the HTML to Markdown tool."""

    result: str = Field(description="Converted Markdown text")
    metadata: dict[str, Any] = Field(default_factory=dict)


class HTMLToMarkdownTool:
    """Convert HTML (from a URL or raw string) to clean Markdown.

    Uses markdownify for HTML-to-Markdown conversion. Strips scripts,
    styles, navs, and footers before conversion.
    """

    tool_id: str = "html_to_markdown"
    display_name: str = "HTML to Markdown"
    category: str = "document"
    description: str = "Fetch a URL or convert raw HTML into clean Markdown text for LLM consumption."
    privilege_level: int = 1
    input_schema: type[HTMLToMarkdownInput] = HTMLToMarkdownInput
    output_schema: type[HTMLToMarkdownOutput] = HTMLToMarkdownOutput

    async def execute(
        self,
        input_data: HTMLToMarkdownInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> HTMLToMarkdownOutput:
        """Fetch HTML from a URL or convert raw HTML to Markdown.

        Args:
            input_data: URL or raw HTML, character limit, image stripping.
            tenant_id: Tenant context for logging.
            config: Unused for this tool.

        Returns:
            HTMLToMarkdownOutput with Markdown text.
        """
        try:
            from bs4 import BeautifulSoup
            from markdownify import markdownify
        except ImportError:
            return HTMLToMarkdownOutput(
                result="",
                metadata={"error": "markdownify or beautifulsoup4 not installed"},
            )

        raw_html: str
        source_url: str | None = None

        if input_data.source.lstrip().startswith("<"):
            raw_html = input_data.source
        else:
            source_url = input_data.source
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.get(source_url)
                response.raise_for_status()
                raw_html = response.text

        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        strip_tags = ["img"] if input_data.strip_images else []
        markdown_text: str = markdownify(str(soup), strip=strip_tags, heading_style="ATX")
        markdown_text = markdown_text[: input_data.max_chars]

        logger.info(
            "HTML converted to Markdown",
            source_url=source_url,
            chars_out=len(markdown_text),
            tenant_id=tenant_id,
        )

        return HTMLToMarkdownOutput(
            result=markdown_text,
            metadata={
                "source_url": source_url,
                "chars_out": len(markdown_text),
                "images_stripped": input_data.strip_images,
            },
        )


TOOL = HTMLToMarkdownTool()
