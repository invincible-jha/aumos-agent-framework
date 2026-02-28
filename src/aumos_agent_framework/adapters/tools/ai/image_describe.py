"""Image description tool via vision LLM.

Privilege level 1 — safe read-only vision inference call.
"""

from typing import Any

import httpx
from pydantic import Field

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class ImageDescribeInput(ToolInputSchema):
    """Input schema for the image description tool."""

    image_url: str = Field(..., description="Public URL of the image to describe")
    prompt: str = Field(
        default="Describe this image in detail.",
        max_length=1000,
        description="Prompt to guide the image description",
    )
    max_tokens: int = Field(default=500, ge=50, le=4096, description="Maximum tokens in the response")


class ImageDescribeOutput(ToolOutputSchema):
    """Output schema for the image description tool."""

    result: str = Field(description="LLM-generated description of the image")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImageDescribeTool:
    """Describe an image using a vision-capable LLM via aumos-llm-serving.

    Passes the image URL directly to the model's vision capability.
    Requires 'LLM_SERVING_URL' and a vision-capable model in config.
    """

    tool_id: str = "image_describe"
    display_name: str = "Image Describer (Vision LLM)"
    category: str = "ai"
    description: str = "Describe the content of an image URL using a vision-capable LLM model."
    privilege_level: int = 1
    input_schema: type[ImageDescribeInput] = ImageDescribeInput
    output_schema: type[ImageDescribeOutput] = ImageDescribeOutput

    async def execute(
        self,
        input_data: ImageDescribeInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> ImageDescribeOutput:
        """Describe an image using a vision LLM.

        Args:
            input_data: Image URL and description prompt.
            tenant_id: Tenant context for routing.
            config: Must contain 'LLM_SERVING_URL'; optionally 'LLM_VISION_MODEL'.

        Returns:
            ImageDescribeOutput with the generated description.
        """
        serving_url = config.get("LLM_SERVING_URL", "").rstrip("/")
        if not serving_url:
            return ImageDescribeOutput(result="", metadata={"error": "LLM_SERVING_URL not configured"})

        model = config.get("LLM_VISION_MODEL", "claude-sonnet-4-6")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": input_data.image_url}},
                    {"type": "text", "text": input_data.prompt},
                ],
            }
        ]

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{serving_url}/v1/chat/completions",
                headers={"X-Tenant-ID": tenant_id, "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "max_tokens": input_data.max_tokens},
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()

        description: str = data["choices"][0]["message"]["content"]

        logger.info(
            "Image described",
            image_url=input_data.image_url,
            description_length=len(description),
            model=model,
            tenant_id=tenant_id,
        )

        return ImageDescribeOutput(
            result=description,
            metadata={"model": model, "image_url": input_data.image_url, "description_length": len(description)},
        )


TOOL = ImageDescribeTool()
