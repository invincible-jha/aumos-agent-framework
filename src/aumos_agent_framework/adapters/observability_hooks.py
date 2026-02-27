"""OpenTelemetry tracing and metrics adapter for agent observability.

Implements ObservabilityHooksProtocol providing:
- OTel span creation and propagation across agent steps.
- Custom metrics: action throughput, token consumption, step latency.
- Agent state change logging with structured context.
- Performance profiling hooks for identifying bottlenecks.
- Dashboard-ready metric aggregation exported via Redis.
"""

import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, AsyncIterator

from opentelemetry import metrics, trace
from opentelemetry.context import Context
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span, Status, StatusCode
from redis.asyncio import Redis

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_TRACER_NAME = "aumos.agent_framework"
_METER_NAME = "aumos.agent_framework"
_METRIC_TTL_SECONDS = 86400  # 24 hours for dashboard retention


class ObservabilityHooks:
    """OpenTelemetry tracing and metrics instrumentation for agent execution.

    Each agent action, reasoning step, and tool invocation is wrapped in an
    OTel span with relevant attributes. Custom metrics track throughput,
    latency, and token consumption at per-tenant and per-agent granularity.
    """

    def __init__(
        self,
        redis_client: Redis,  # type: ignore[type-arg]
        tracer_provider: TracerProvider | None = None,
        meter_provider: MeterProvider | None = None,
        service_name: str = "aumos-agent-framework",
    ) -> None:
        """Initialize with OTel providers and Redis for metric aggregation.

        Args:
            redis_client: Async Redis client for metric storage.
            tracer_provider: OTel TracerProvider (uses global default if None).
            meter_provider: OTel MeterProvider (uses global default if None).
            service_name: Service name for OTel resource attributes.
        """
        self._redis = redis_client
        self._service_name = service_name

        # Tracer for span creation
        if tracer_provider:
            self._tracer = tracer_provider.get_tracer(_TRACER_NAME)
        else:
            self._tracer = trace.get_tracer(_TRACER_NAME)

        # Meter for custom metrics
        if meter_provider:
            self._meter = meter_provider.get_meter(_METER_NAME)
        else:
            self._meter = metrics.get_meter(_METER_NAME)

        # Register OTel instruments
        self._action_counter = self._meter.create_counter(
            name="agf.actions.total",
            description="Total agent actions executed",
            unit="1",
        )
        self._action_error_counter = self._meter.create_counter(
            name="agf.actions.errors",
            description="Total agent action errors",
            unit="1",
        )
        self._action_latency_histogram = self._meter.create_histogram(
            name="agf.actions.latency_ms",
            description="Agent action execution latency in milliseconds",
            unit="ms",
        )
        self._token_counter = self._meter.create_counter(
            name="agf.tokens.consumed",
            description="Total LLM tokens consumed by agents",
            unit="1",
        )
        self._react_iteration_histogram = self._meter.create_histogram(
            name="agf.react.iterations",
            description="Number of ReAct loop iterations per run",
            unit="1",
        )

    # ─── Span management ─────────────────────────────────────────────────────

    @asynccontextmanager
    async def trace_agent_action(
        self,
        action_name: str,
        agent_id: uuid.UUID,
        tenant_id: str,
        attributes: dict[str, Any] | None = None,
        parent_context: dict[str, str] | None = None,
    ) -> AsyncIterator[Span]:
        """Context manager that creates an OTel span for an agent action.

        Captures start time, agent/tenant attributes, and records success or
        error status on exit. Propagates context from upstream callers.

        Args:
            action_name: Descriptive name for the action span (e.g., 'tool.call.search').
            agent_id: Agent executing the action.
            tenant_id: Tenant context (added as span attribute).
            attributes: Additional span attributes to set.
            parent_context: W3C trace context dict from upstream caller.

        Yields:
            The active OTel Span for the action.
        """
        context: Context | None = None
        if parent_context:
            context = extract(parent_context)

        span_attributes: dict[str, Any] = {
            "agent.id": str(agent_id),
            "tenant.id": tenant_id,
            "service.name": self._service_name,
        }
        if attributes:
            span_attributes.update(attributes)

        start_ms = time.monotonic()

        with self._tracer.start_as_current_span(
            action_name,
            context=context,
            attributes=span_attributes,
        ) as span:
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
                duration_ms = int((time.monotonic() - start_ms) * 1000)
                span.set_attribute("action.duration_ms", duration_ms)

                # Record metrics
                metric_attrs = {"agent_id": str(agent_id), "tenant_id": tenant_id}
                self._action_counter.add(1, metric_attrs)
                self._action_latency_histogram.record(duration_ms, metric_attrs)

            except Exception as exc:
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
                duration_ms = int((time.monotonic() - start_ms) * 1000)
                span.set_attribute("action.duration_ms", duration_ms)

                error_attrs = {"agent_id": str(agent_id), "tenant_id": tenant_id,
                               "error.type": type(exc).__name__}
                self._action_error_counter.add(1, error_attrs)
                self._action_latency_histogram.record(duration_ms, {"agent_id": str(agent_id),
                                                                      "tenant_id": tenant_id})
                raise

    def extract_trace_context(self) -> dict[str, str]:
        """Extract current OTel trace context as W3C propagation headers.

        Returns:
            Dict of W3C headers (traceparent, tracestate) for propagating
            to downstream services.
        """
        carrier: dict[str, str] = {}
        inject(carrier)
        return carrier

    # ─── Agent state logging ──────────────────────────────────────────────────

    async def log_state_change(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        previous_state: str,
        new_state: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log an agent state transition with structured context.

        State changes are logged to structlog AND stored in Redis for
        time-series dashboard queries.

        Args:
            agent_id: Agent whose state changed.
            tenant_id: Tenant context.
            previous_state: State before transition.
            new_state: State after transition.
            reason: Human-readable reason for the transition.
            metadata: Optional additional context.
        """
        state_event = {
            "agent_id": str(agent_id),
            "tenant_id": tenant_id,
            "previous_state": previous_state,
            "new_state": new_state,
            "reason": reason,
            "metadata": metadata or {},
            "timestamp": datetime.now(UTC).isoformat(),
        }

        logger.info(
            "Agent state transition",
            **state_event,
        )

        # Store in Redis time-series list for dashboard queries
        redis_key = f"obs:state_changes:{tenant_id}:{agent_id}"
        import json
        await self._redis.lpush(redis_key, json.dumps(state_event))
        await self._redis.ltrim(redis_key, 0, 999)  # Keep last 1000 transitions
        await self._redis.expire(redis_key, _METRIC_TTL_SECONDS)

    async def log_reasoning_step(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        trace_id: str,
        step_number: int,
        step_type: str,
        content_summary: str,
        confidence: float,
        token_count: int,
    ) -> None:
        """Log a single reasoning step for explainability audit trails.

        Args:
            agent_id: Agent that performed the reasoning step.
            tenant_id: Tenant context.
            trace_id: Reasoning trace ID for correlation.
            step_number: Position in the reasoning chain.
            step_type: One of 'thought', 'action', 'observation', 'conclusion'.
            content_summary: Truncated content summary (first 200 chars).
            confidence: Model confidence for this step.
            token_count: Tokens consumed by this step.
        """
        logger.debug(
            "Agent reasoning step",
            agent_id=str(agent_id),
            trace_id=trace_id,
            step_number=step_number,
            step_type=step_type,
            content_summary=content_summary[:200],
            confidence=confidence,
            token_count=token_count,
            tenant_id=tenant_id,
        )

        # Track token consumption
        self._token_counter.add(
            token_count,
            {"agent_id": str(agent_id), "tenant_id": tenant_id, "step_type": step_type},
        )

    # ─── Metrics recording ────────────────────────────────────────────────────

    async def record_tokens_consumed(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        token_count: int,
        model_id: str,
        action_type: str = "inference",
    ) -> None:
        """Record LLM token consumption for a single agent action.

        Args:
            agent_id: Agent that consumed the tokens.
            tenant_id: Tenant context.
            token_count: Number of tokens consumed.
            model_id: Model identifier for cost attribution.
            action_type: Category of action (e.g., 'inference', 'embedding').
        """
        attrs = {
            "agent_id": str(agent_id),
            "tenant_id": tenant_id,
            "model_id": model_id,
            "action_type": action_type,
        }
        self._token_counter.add(token_count, attrs)

        # Aggregate in Redis for dashboard
        redis_key = f"obs:tokens:{tenant_id}:{datetime.now(UTC).strftime('%Y-%m-%d')}"
        await self._redis.hincrby(redis_key, str(agent_id), token_count)
        await self._redis.expire(redis_key, _METRIC_TTL_SECONDS * 7)  # 7-day retention

    async def record_react_iteration(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        trace_id: str,
        iteration_count: int,
        terminated_by: str,
    ) -> None:
        """Record the outcome of a completed ReAct loop.

        Args:
            agent_id: Agent that ran the loop.
            tenant_id: Tenant context.
            trace_id: Reasoning trace ID.
            iteration_count: How many iterations the loop ran.
            terminated_by: How the loop ended: 'conclusion', 'max_iterations', 'error'.
        """
        attrs = {
            "agent_id": str(agent_id),
            "tenant_id": tenant_id,
            "terminated_by": terminated_by,
        }
        self._react_iteration_histogram.record(iteration_count, attrs)

        logger.info(
            "ReAct loop completed",
            agent_id=str(agent_id),
            trace_id=trace_id,
            iteration_count=iteration_count,
            terminated_by=terminated_by,
            tenant_id=tenant_id,
        )

    # ─── Profiling hooks ──────────────────────────────────────────────────────

    @asynccontextmanager
    async def profile_section(
        self,
        section_name: str,
        agent_id: uuid.UUID,
        tenant_id: str,
    ) -> AsyncIterator[None]:
        """Profile a code section and log its wall-clock duration.

        Useful for identifying performance bottlenecks in agent pipelines.

        Args:
            section_name: Descriptive name for the profiled section.
            agent_id: Agent executing the section.
            tenant_id: Tenant context.

        Yields:
            None — use as 'async with hooks.profile_section(...):'
        """
        start_ns = time.monotonic_ns()
        try:
            yield
        finally:
            elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000
            logger.debug(
                "Profile section timing",
                section=section_name,
                agent_id=str(agent_id),
                elapsed_ms=round(elapsed_ms, 3),
                tenant_id=tenant_id,
            )

    # ─── Dashboard aggregation ────────────────────────────────────────────────

    async def get_agent_metrics_summary(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        date_str: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve aggregated metrics for an agent from Redis.

        Args:
            agent_id: Agent to query metrics for.
            tenant_id: Tenant context.
            date_str: Date string 'YYYY-MM-DD'; defaults to today.

        Returns:
            Dict with token counts, state change counts, and action stats.
        """
        import json
        if date_str is None:
            date_str = datetime.now(UTC).strftime("%Y-%m-%d")

        tokens_key = f"obs:tokens:{tenant_id}:{date_str}"
        state_key = f"obs:state_changes:{tenant_id}:{agent_id}"

        agent_id_str = str(agent_id)

        # Token count for this agent today
        token_bytes = await self._redis.hget(tokens_key, agent_id_str)
        daily_tokens = int(token_bytes) if token_bytes else 0

        # Recent state changes
        raw_state_changes = await self._redis.lrange(state_key, 0, 9)
        recent_state_changes = [json.loads(raw) for raw in raw_state_changes]

        return {
            "agent_id": agent_id_str,
            "tenant_id": tenant_id,
            "date": date_str,
            "daily_tokens_consumed": daily_tokens,
            "recent_state_changes": recent_state_changes,
            "state_change_count": await self._redis.llen(state_key),
        }
