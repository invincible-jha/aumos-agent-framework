"""Kafka event publisher adapter for agent lifecycle events.

Publishes domain events to the agf.* Kafka topics after state changes in
agent workflows, HITL gates, and circuit breakers. Uses confluent-kafka
with structured JSON payloads and tenant context on every message.
"""

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Kafka topic constants — must match CLAUDE.md specification
TOPIC_WORKFLOW_STARTED = "agf.workflow.started"
TOPIC_WORKFLOW_COMPLETED = "agf.workflow.completed"
TOPIC_WORKFLOW_FAILED = "agf.workflow.failed"
TOPIC_WORKFLOW_PAUSED_HITL = "agf.workflow.paused_hitl"
TOPIC_AGENT_INVOKED = "agf.agent.invoked"
TOPIC_HITL_APPROVAL_REQUESTED = "agf.hitl.approval_requested"
TOPIC_HITL_APPROVED = "agf.hitl.approved"
TOPIC_HITL_REJECTED = "agf.hitl.rejected"
TOPIC_CIRCUIT_BREAKER_OPENED = "agf.circuit_breaker.opened"
TOPIC_CIRCUIT_BREAKER_CLOSED = "agf.circuit_breaker.closed"


class AgentFrameworkKafkaPublisher:
    """Publishes agent framework lifecycle events to Kafka.

    All events include:
    - tenant_id: for downstream tenant-scoped consumers
    - event_type: for consumer routing
    - occurred_at: ISO8601 UTC timestamp
    - payload: event-specific data

    Message key is always the tenant_id to ensure tenant-level ordering
    on partitioned topics.
    """

    def __init__(self, producer: Any, bootstrap_servers: str | None = None) -> None:
        """Initialize with a confluent-kafka Producer instance.

        Args:
            producer: Configured confluent_kafka.Producer instance.
                      Must already be initialized — this adapter does not own lifecycle.
            bootstrap_servers: Optional for logging context only.
        """
        self._producer = producer
        self._bootstrap_servers = bootstrap_servers

    def _build_event(
        self,
        event_type: str,
        tenant_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a structured event envelope.

        Args:
            event_type: The event type string (e.g., TOPIC_WORKFLOW_STARTED).
            tenant_id: Tenant context for downstream consumers.
            payload: Event-specific data dict.

        Returns:
            Structured event dict ready for JSON serialization.
        """
        return {
            "event_type": event_type,
            "tenant_id": tenant_id,
            "occurred_at": datetime.now(UTC).isoformat(),
            "payload": payload,
        }

    def _publish(self, topic: str, tenant_id: str, event: dict[str, Any]) -> None:
        """Serialize and publish an event to Kafka.

        Uses tenant_id as message key for partition affinity.
        Calls poll(0) to serve delivery callbacks without blocking.

        Args:
            topic: Kafka topic to publish to.
            tenant_id: Tenant context — used as message key.
            event: Structured event dict to serialize.
        """
        try:
            self._producer.produce(
                topic=topic,
                key=tenant_id.encode("utf-8"),
                value=json.dumps(event).encode("utf-8"),
            )
            self._producer.poll(0)
        except Exception as exc:
            # Log but do not raise — Kafka publish failure must not block agent execution
            logger.error(
                "Failed to publish Kafka event",
                topic=topic,
                event_type=event.get("event_type"),
                tenant_id=tenant_id,
                error=str(exc),
            )

    async def publish_workflow_started(
        self,
        execution_id: UUID,
        workflow_id: UUID,
        tenant_id: str,
        input_data: dict[str, Any],
    ) -> None:
        """Publish agf.workflow.started event.

        Args:
            execution_id: The new workflow execution ID.
            workflow_id: The workflow definition ID being executed.
            tenant_id: Tenant context.
            input_data: Initial input state for the workflow.
        """
        event = self._build_event(
            TOPIC_WORKFLOW_STARTED,
            tenant_id,
            {
                "execution_id": str(execution_id),
                "workflow_id": str(workflow_id),
                "input_data": input_data,
            },
        )
        self._publish(TOPIC_WORKFLOW_STARTED, tenant_id, event)
        logger.info(
            "Published workflow.started event",
            execution_id=str(execution_id),
            workflow_id=str(workflow_id),
            tenant_id=tenant_id,
        )

    async def publish_workflow_completed(
        self,
        execution_id: UUID,
        workflow_id: UUID,
        tenant_id: str,
        output_data: dict[str, Any],
    ) -> None:
        """Publish agf.workflow.completed event."""
        event = self._build_event(
            TOPIC_WORKFLOW_COMPLETED,
            tenant_id,
            {
                "execution_id": str(execution_id),
                "workflow_id": str(workflow_id),
                "output_data": output_data,
            },
        )
        self._publish(TOPIC_WORKFLOW_COMPLETED, tenant_id, event)
        logger.info(
            "Published workflow.completed event",
            execution_id=str(execution_id),
            tenant_id=tenant_id,
        )

    async def publish_workflow_failed(
        self,
        execution_id: UUID,
        workflow_id: UUID,
        tenant_id: str,
        error_details: dict[str, Any],
    ) -> None:
        """Publish agf.workflow.failed event."""
        event = self._build_event(
            TOPIC_WORKFLOW_FAILED,
            tenant_id,
            {
                "execution_id": str(execution_id),
                "workflow_id": str(workflow_id),
                "error_details": error_details,
            },
        )
        self._publish(TOPIC_WORKFLOW_FAILED, tenant_id, event)
        logger.warning(
            "Published workflow.failed event",
            execution_id=str(execution_id),
            tenant_id=tenant_id,
        )

    async def publish_workflow_paused_hitl(
        self,
        execution_id: UUID,
        workflow_id: UUID,
        approval_id: UUID,
        gate_name: str,
        tenant_id: str,
    ) -> None:
        """Publish agf.workflow.paused_hitl event."""
        event = self._build_event(
            TOPIC_WORKFLOW_PAUSED_HITL,
            tenant_id,
            {
                "execution_id": str(execution_id),
                "workflow_id": str(workflow_id),
                "approval_id": str(approval_id),
                "gate_name": gate_name,
            },
        )
        self._publish(TOPIC_WORKFLOW_PAUSED_HITL, tenant_id, event)
        logger.info(
            "Published workflow.paused_hitl event",
            execution_id=str(execution_id),
            gate_name=gate_name,
            tenant_id=tenant_id,
        )

    async def publish_agent_invoked(
        self,
        agent_id: UUID,
        execution_id: UUID | None,
        tenant_id: str,
        privilege_level: int,
    ) -> None:
        """Publish agf.agent.invoked event."""
        event = self._build_event(
            TOPIC_AGENT_INVOKED,
            tenant_id,
            {
                "agent_id": str(agent_id),
                "execution_id": str(execution_id) if execution_id else None,
                "privilege_level": privilege_level,
            },
        )
        self._publish(TOPIC_AGENT_INVOKED, tenant_id, event)

    async def publish_hitl_approval_requested(
        self,
        approval_id: UUID,
        execution_id: UUID,
        gate_name: str,
        agent_id: UUID,
        action_description: str,
        tenant_id: str,
    ) -> None:
        """Publish agf.hitl.approval_requested event."""
        event = self._build_event(
            TOPIC_HITL_APPROVAL_REQUESTED,
            tenant_id,
            {
                "approval_id": str(approval_id),
                "execution_id": str(execution_id),
                "gate_name": gate_name,
                "agent_id": str(agent_id),
                "action_description": action_description,
            },
        )
        self._publish(TOPIC_HITL_APPROVAL_REQUESTED, tenant_id, event)
        logger.info(
            "Published hitl.approval_requested event",
            approval_id=str(approval_id),
            gate_name=gate_name,
            tenant_id=tenant_id,
        )

    async def publish_hitl_approved(
        self,
        approval_id: UUID,
        decided_by: UUID,
        tenant_id: str,
    ) -> None:
        """Publish agf.hitl.approved event."""
        event = self._build_event(
            TOPIC_HITL_APPROVED,
            tenant_id,
            {
                "approval_id": str(approval_id),
                "decided_by": str(decided_by),
            },
        )
        self._publish(TOPIC_HITL_APPROVED, tenant_id, event)

    async def publish_hitl_rejected(
        self,
        approval_id: UUID,
        decided_by: UUID,
        notes: str | None,
        tenant_id: str,
    ) -> None:
        """Publish agf.hitl.rejected event."""
        event = self._build_event(
            TOPIC_HITL_REJECTED,
            tenant_id,
            {
                "approval_id": str(approval_id),
                "decided_by": str(decided_by),
                "notes": notes,
            },
        )
        self._publish(TOPIC_HITL_REJECTED, tenant_id, event)

    async def publish_circuit_breaker_opened(
        self,
        circuit_key: str,
        tenant_id: str,
        failure_count: int,
    ) -> None:
        """Publish agf.circuit_breaker.opened event."""
        event = self._build_event(
            TOPIC_CIRCUIT_BREAKER_OPENED,
            tenant_id,
            {
                "circuit_key": circuit_key,
                "failure_count": failure_count,
            },
        )
        self._publish(TOPIC_CIRCUIT_BREAKER_OPENED, tenant_id, event)
        logger.warning(
            "Published circuit_breaker.opened event",
            circuit_key=circuit_key,
            failure_count=failure_count,
            tenant_id=tenant_id,
        )

    async def publish_circuit_breaker_closed(
        self,
        circuit_key: str,
        tenant_id: str,
    ) -> None:
        """Publish agf.circuit_breaker.closed event."""
        event = self._build_event(
            TOPIC_CIRCUIT_BREAKER_CLOSED,
            tenant_id,
            {"circuit_key": circuit_key},
        )
        self._publish(TOPIC_CIRCUIT_BREAKER_CLOSED, tenant_id, event)
        logger.info(
            "Published circuit_breaker.closed event",
            circuit_key=circuit_key,
            tenant_id=tenant_id,
        )
