"""LLM text summarization tool via aumos-llm-serving.

Privilege level 1 — safe read-only LLM inference call.
"""

from typing import Any

import httpx
from pydantic import Field

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class LLMSummarizeInput(ToolInputSchema):
    """Input schema for the LLM summarization tool."""

    text: str = Field(..., max_length=100000, description="Text to summarize")
    max_summary_words: int = Field(
        default=150,
        ge=20,
        le=2000,
        description="Approximate maximum word count for the summary",
    )
    style: str = Field(
        default="concise",
        description="Summary style: 'concise', 'bullet_points', or 'executive'",
    )


class LLMSummarizeOutput(ToolOutputSchema):
    """Output schema for the LLM summarization tool."""

    result: str = Field(description="Generated summary text")
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMSummarizeTool:
    """Summarize text using the aumos-llm-serving inference service.

    Calls the /v1/chat/completions endpoint on the configured LLM serving URL.
    Requires 'LLM_SERVING_URL' and optionally 'LLM_MODEL' in config.
    """

    tool_id: str = "llm_summarize"
    display_name: str = "LLM Text Summarizer"
    category: str = "ai"
    description: str = "Summarize a long text using an LLM via aumos-llm-serving. Supports concise, bullet-point, and executive styles."
    privilege_level: int = 1
    input_schema: type[LLMSummarizeInput] = LLMSummarizeInput
    output_schema: type[LLMSummarizeOutput] = LLMSummarizeOutput

    _STYLE_PROMPTS: dict[str, str] = {
        "concise": "Provide a concise summary in {max_words} words or fewer.",
        "bullet_points": "Summarize as {max_words}-word bullet points, one key point per line.",
        "executive": "Write an executive summary in {max_words} words highlighting business impact.",
    }

    async def execute(
        self,
        input_data: LLMSummarizeInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> LLMSummarizeOutput:
        """Summarize text via aumos-llm-serving.

        Args:
            input_data: Text, word count target, and summary style.
            tenant_id: Tenant context for routing.
            config: Must contain 'LLM_SERVING_URL'; optionally 'LLM_MODEL'.

        Returns:
            LLMSummarizeOutput with the generated summary.
        """
        serving_url = config.get("LLM_SERVING_URL", "").rstrip("/")
        if not serving_url:
            return LLMSummarizeOutput(result="", metadata={"error": "LLM_SERVING_URL not configured"})

        model = config.get("LLM_MODEL", "claude-sonnet-4-6")
        style_prompt = self._STYLE_PROMPTS.get(
            input_data.style,
            self._STYLE_PROMPTS["concise"],
        ).format(max_words=input_data.max_summary_words)

        messages = [
            {
                "role": "system",
                "content": f"You are a summarization assistant. {style_prompt}",
            },
            {
                "role": "user",
                "content": f"Summarize the following text:\n\n{input_data.text}",
            },
        ]

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{serving_url}/v1/chat/completions",
                headers={"X-Tenant-ID": tenant_id, "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "max_tokens": input_data.max_summary_words * 2},
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()

        summary: str = data["choices"][0]["message"]["content"]

        logger.info(
            "LLM summarization complete",
            input_length=len(input_data.text),
            summary_length=len(summary),
            model=model,
            tenant_id=tenant_id,
        )

        return LLMSummarizeOutput(
            result=summary,
            metadata={"model": model, "style": input_data.style, "summary_length": len(summary)},
        )


TOOL = LLMSummarizeTool()
