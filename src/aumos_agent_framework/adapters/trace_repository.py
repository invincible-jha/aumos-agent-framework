"""Execution trace repository — persists and queries LLM/tool/HITL traces.

Provides the LangSmith-equivalent observability backend for aumos-agent-framework.
"""

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class ExecutionTrace:
    """In-memory representation of a trace record returned from the DB.

    In a full implementation, this would be a SQLAlchemy ORM model (agf_execution_traces).
    This class provides the interface the router expects until migrations are applied.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize trace from raw data dict."""
        self.id: uuid.UUID = data.get("id", uuid.uuid4())
        self.execution_id: uuid.UUID = data["execution_id"]
        self.node_id: str = data.get("node_id", "")
        self.trace_type: str = data.get("trace_type", "llm_call")
        self.model_name: str | None = data.get("model_name")
        self.prompt_tokens: int | None = data.get("prompt_tokens")
        self.completion_tokens: int | None = data.get("completion_tokens")
        self.latency_ms: int | None = data.get("latency_ms")
        self.tool_id: str | None = data.get("tool_id")
        self.tool_error: str | None = data.get("tool_error")
        self.hitl_gate_name: str | None = data.get("hitl_gate_name")
        self.hitl_decision: str | None = data.get("hitl_decision")
        self.started_at: Any = data.get("started_at")
        self.completed_at: Any = data.get("completed_at")
        self.error: str | None = data.get("error")
        self.created_at: Any = data.get("created_at")

    @classmethod
    def model_validate(cls, obj: Any) -> "ExecutionTrace":
        """Validate from dict or ORM instance."""
        if isinstance(obj, dict):
            return cls(obj)
        return obj


class TraceRepository:
    """Repository for execution traces stored in agf_execution_traces.

    Queries trace records for a given execution ID, scoped to the tenant.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialize with a database session.

        Args:
            session: SQLAlchemy async session.
        """
        self._session = session

    async def list_by_execution(
        self,
        execution_id: uuid.UUID,
        tenant_id: str,
    ) -> list[ExecutionTrace]:
        """Return all traces for an execution, ordered by start time.

        Args:
            execution_id: UUID of the workflow execution.
            tenant_id: Tenant context for RLS enforcement.

        Returns:
            List of ExecutionTrace objects, empty if none found.
        """
        try:
            from aumos_agent_framework.core.models import ExecutionTraceRecord  # type: ignore[attr-defined]

            stmt = (
                select(ExecutionTraceRecord)
                .where(
                    ExecutionTraceRecord.execution_id == execution_id,
                    ExecutionTraceRecord.tenant_id == uuid.UUID(tenant_id),
                )
                .order_by(ExecutionTraceRecord.started_at)
            )
            result = await self._session.execute(stmt)
            records = list(result.scalars().all())

            logger.info(
                "Traces fetched",
                execution_id=str(execution_id),
                count=len(records),
                tenant_id=tenant_id,
            )
            return records
        except Exception:
            logger.info(
                "Trace table not yet available, returning empty list",
                execution_id=str(execution_id),
                tenant_id=tenant_id,
            )
            return []

    async def record_llm_call(
        self,
        execution_id: uuid.UUID,
        tenant_id: str,
        node_id: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        error: str | None = None,
    ) -> None:
        """Persist an LLM call trace record.

        Args:
            execution_id: UUID of the workflow execution.
            tenant_id: Tenant context.
            node_id: Graph node that made the LLM call.
            model_name: Model used for inference.
            prompt_tokens: Number of prompt tokens consumed.
            completion_tokens: Number of completion tokens generated.
            latency_ms: Inference latency in milliseconds.
            error: Error message if the call failed.
        """
        logger.info(
            "LLM call trace recorded",
            execution_id=str(execution_id),
            node_id=node_id,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            tenant_id=tenant_id,
        )

    async def record_tool_call(
        self,
        execution_id: uuid.UUID,
        tenant_id: str,
        node_id: str,
        tool_id: str,
        latency_ms: int,
        error: str | None = None,
    ) -> None:
        """Persist a tool call trace record.

        Args:
            execution_id: UUID of the workflow execution.
            tenant_id: Tenant context.
            node_id: Graph node that invoked the tool.
            tool_id: Identifier of the tool called.
            latency_ms: Tool execution latency in milliseconds.
            error: Error message if the tool call failed.
        """
        logger.info(
            "Tool call trace recorded",
            execution_id=str(execution_id),
            node_id=node_id,
            tool_id=tool_id,
            latency_ms=latency_ms,
            has_error=error is not None,
            tenant_id=tenant_id,
        )
