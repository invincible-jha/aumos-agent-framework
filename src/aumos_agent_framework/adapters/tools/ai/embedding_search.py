"""Semantic embedding search tool via aumos-llm-serving.

Privilege level 1 — safe read-only embedding + similarity search.
"""

from typing import Any

import httpx
from pydantic import Field

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class EmbeddingSearchInput(ToolInputSchema):
    """Input schema for the embedding search tool."""

    query: str = Field(..., max_length=8000, description="Query text to embed and search")
    documents: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of document strings to search over",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top results to return")


class EmbeddingSearchOutput(ToolOutputSchema):
    """Output schema for the embedding search tool."""

    result: list[dict[str, Any]] = Field(
        description="Ranked list of matching documents with index, text, and similarity score"
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingSearchTool:
    """Perform semantic similarity search using embeddings from aumos-llm-serving.

    Embeds the query and all documents, then ranks documents by cosine
    similarity. Requires 'LLM_SERVING_URL' in config.
    """

    tool_id: str = "embedding_search"
    display_name: str = "Embedding Similarity Search"
    category: str = "ai"
    description: str = (
        "Embed a query and a list of documents, then return the top-k most semantically similar documents."
    )
    privilege_level: int = 1
    input_schema: type[EmbeddingSearchInput] = EmbeddingSearchInput
    output_schema: type[EmbeddingSearchOutput] = EmbeddingSearchOutput

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def execute(
        self,
        input_data: EmbeddingSearchInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> EmbeddingSearchOutput:
        """Embed query and documents then return ranked results.

        Args:
            input_data: Query, document list, and top-k count.
            tenant_id: Tenant context for routing.
            config: Must contain 'LLM_SERVING_URL'; optionally 'EMBEDDING_MODEL'.

        Returns:
            EmbeddingSearchOutput with ranked document matches.
        """
        serving_url = config.get("LLM_SERVING_URL", "").rstrip("/")
        if not serving_url:
            return EmbeddingSearchOutput(result=[], metadata={"error": "LLM_SERVING_URL not configured"})

        model = config.get("EMBEDDING_MODEL", "text-embedding-3-small")
        all_texts = [input_data.query] + input_data.documents

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{serving_url}/v1/embeddings",
                headers={"X-Tenant-ID": tenant_id, "Content-Type": "application/json"},
                json={"model": model, "input": all_texts},
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()

        embeddings: list[list[float]] = [item["embedding"] for item in data["data"]]
        query_embedding = embeddings[0]
        doc_embeddings = embeddings[1:]

        scored: list[dict[str, Any]] = []
        for i, (doc, emb) in enumerate(zip(input_data.documents, doc_embeddings)):
            score = self._cosine_similarity(query_embedding, emb)
            scored.append({"index": i, "text": doc, "similarity": round(score, 6)})

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = scored[: input_data.top_k]

        logger.info(
            "Embedding search complete",
            query_length=len(input_data.query),
            doc_count=len(input_data.documents),
            top_k=input_data.top_k,
            tenant_id=tenant_id,
        )

        return EmbeddingSearchOutput(
            result=top_results,
            metadata={"model": model, "doc_count": len(input_data.documents), "top_k": input_data.top_k},
        )


TOOL = EmbeddingSearchTool()
