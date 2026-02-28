"""PDF text extraction tool via pypdf.

Privilege level 1 — safe read-only document parsing.
"""

from typing import Any

import httpx
from pydantic import Field

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class PDFExtractInput(ToolInputSchema):
    """Input schema for the PDF extraction tool."""

    source_url: str = Field(..., description="URL to fetch the PDF from")
    max_pages: int = Field(default=50, ge=1, le=500, description="Maximum pages to extract text from")
    max_chars: int = Field(
        default=50000,
        ge=1000,
        le=500000,
        description="Maximum characters to return across all pages",
    )


class PDFExtractOutput(ToolOutputSchema):
    """Output schema for the PDF extraction tool."""

    result: str = Field(description="Extracted text content from the PDF")
    metadata: dict[str, Any] = Field(default_factory=dict)


class PDFExtractTool:
    """Extract text from a PDF file fetched from a URL.

    Uses pypdf for pure-Python PDF parsing. Images and scanned PDFs
    are not supported — text layer only.
    """

    tool_id: str = "pdf_extract"
    display_name: str = "PDF Text Extractor"
    category: str = "document"
    description: str = "Fetch a PDF from a URL and extract its text content page by page."
    privilege_level: int = 1
    input_schema: type[PDFExtractInput] = PDFExtractInput
    output_schema: type[PDFExtractOutput] = PDFExtractOutput

    async def execute(
        self,
        input_data: PDFExtractInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> PDFExtractOutput:
        """Fetch and extract text from a PDF.

        Args:
            input_data: PDF URL, page limit, and character limit.
            tenant_id: Tenant context for logging.
            config: Optional 'AUTH_TOKEN' for authenticated PDF URLs.

        Returns:
            PDFExtractOutput with extracted text.
        """
        try:
            import io

            from pypdf import PdfReader
        except ImportError:
            return PDFExtractOutput(result="", metadata={"error": "pypdf not installed"})

        headers: dict[str, str] = {}
        auth_token = config.get("AUTH_TOKEN")
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(input_data.source_url, headers=headers)
            response.raise_for_status()

        reader = PdfReader(io.BytesIO(response.content))
        total_pages = len(reader.pages)
        pages_to_read = min(total_pages, input_data.max_pages)

        text_parts: list[str] = []
        for i in range(pages_to_read):
            page_text = reader.pages[i].extract_text() or ""
            text_parts.append(page_text)

        full_text = "\n\n".join(text_parts)[: input_data.max_chars]

        logger.info(
            "PDF extracted",
            source_url=input_data.source_url,
            pages_read=pages_to_read,
            chars_extracted=len(full_text),
            tenant_id=tenant_id,
        )

        return PDFExtractOutput(
            result=full_text,
            metadata={
                "source_url": input_data.source_url,
                "total_pages": total_pages,
                "pages_read": pages_to_read,
                "chars_extracted": len(full_text),
            },
        )


TOOL = PDFExtractTool()
