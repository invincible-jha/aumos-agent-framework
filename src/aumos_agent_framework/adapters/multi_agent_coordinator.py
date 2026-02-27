"""Inter-agent coordination adapter via Kafka message dispatch.

Implements MultiAgentCoordinatorProtocol providing:
- Point-to-point agent messaging with correlation IDs.
- Broadcast messaging to all agents in a named group.
- Request-response pattern with timeout and response aggregation.
- Task delegation protocol with acknowledgement tracking.
- Deadlock detection via dependency graph cycle analysis using NetworkX.
- Coordinator health monitoring via Redis heartbeat.
"""

import asyncio
import json
import uuid
from datetime import UTC, datetime
from typing import Any

import networkx as nx
from confluent_kafka import Producer, Consumer, KafkaException
from redis.asyncio import Redis

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_DEFAULT_RESPONSE_TIMEOUT_SECONDS = 30
_DEFAULT_HEARTBEAT_TTL_SECONDS = 60
_DEFAULT_MAX_CORRELATION_CACHE = 1000

# Kafka topic conventions
_TOPIC_AGENT_MESSAGES = "agf.agent.messages"
_TOPIC_AGENT_BROADCAST = "agf.agent.broadcast"
_TOPIC_TASK_DELEGATION = "agf.task.delegation"
_TOPIC_TASK_ACKNOWLEDGEMENT = "agf.task.ack"


class DeadlockDetectedError(Exception):
    """Raised when a dependency cycle is detected in agent task graph.

    Attributes:
        cycle: List of agent IDs forming the dependency cycle.
    """

    def __init__(self, cycle: list[str]) -> None:
        super().__init__(f"Deadlock detected in agent dependency graph: {' -> '.join(cycle)}")
        self.cycle = cycle


class MultiAgentCoordinator:
    """Kafka-backed coordinator for inter-agent messaging and task delegation.

    Manages point-to-point and broadcast messaging between agents. Tracks
    pending tasks in Redis for timeout enforcement and response aggregation.
    Uses NetworkX to detect dependency cycles that would cause deadlocks.
    """

    def __init__(
        self,
        kafka_producer: Producer,
        kafka_consumer: Consumer,
        redis_client: Redis,  # type: ignore[type-arg]
        response_timeout_seconds: int = _DEFAULT_RESPONSE_TIMEOUT_SECONDS,
        heartbeat_ttl_seconds: int = _DEFAULT_HEARTBEAT_TTL_SECONDS,
    ) -> None:
        """Initialize with Kafka producer/consumer and Redis.

        Args:
            kafka_producer: Confluent Kafka producer for publishing messages.
            kafka_consumer: Confluent Kafka consumer for receiving responses.
            redis_client: Async Redis client for correlation tracking.
            response_timeout_seconds: Timeout for request-response operations.
            heartbeat_ttl_seconds: TTL for coordinator heartbeat in Redis.
        """
        self._producer = kafka_producer
        self._consumer = kafka_consumer
        self._redis = redis_client
        self._response_timeout = response_timeout_seconds
        self._heartbeat_ttl = heartbeat_ttl_seconds
        # In-process response futures keyed by correlation_id
        self._pending_responses: dict[str, asyncio.Future[dict[str, Any]]] = {}
        # Dependency graph for deadlock detection
        self._dependency_graph: nx.DiGraph = nx.DiGraph()

    # ─── Messaging ────────────────────────────────────────────────────────────

    def _build_message(
        self,
        sender_agent_id: uuid.UUID,
        recipient_agent_id: uuid.UUID,
        message_type: str,
        payload: dict[str, Any],
        tenant_id: str,
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        """Build a standard inter-agent message envelope.

        Args:
            sender_agent_id: Agent sending the message.
            recipient_agent_id: Intended recipient agent.
            message_type: Category tag (e.g., 'request', 'response', 'notification').
            payload: Arbitrary message payload dict.
            tenant_id: Tenant context for routing and isolation.
            correlation_id: Optional ID linking request to response.

        Returns:
            Complete message envelope dict.
        """
        return {
            "message_id": str(uuid.uuid4()),
            "correlation_id": correlation_id or str(uuid.uuid4()),
            "sender_agent_id": str(sender_agent_id),
            "recipient_agent_id": str(recipient_agent_id),
            "message_type": message_type,
            "payload": payload,
            "tenant_id": tenant_id,
            "sent_at": datetime.now(UTC).isoformat(),
        }

    async def send_message(
        self,
        sender_agent_id: uuid.UUID,
        recipient_agent_id: uuid.UUID,
        payload: dict[str, Any],
        tenant_id: str,
        message_type: str = "notification",
    ) -> str:
        """Send a fire-and-forget message from one agent to another.

        Publishes to the Kafka agent messages topic partitioned by recipient.

        Args:
            sender_agent_id: Agent sending the message.
            recipient_agent_id: Intended recipient agent ID.
            payload: Message payload dict.
            tenant_id: Tenant context.
            message_type: Message category for recipient routing.

        Returns:
            Message ID string for tracking.
        """
        envelope = self._build_message(
            sender_agent_id=sender_agent_id,
            recipient_agent_id=recipient_agent_id,
            message_type=message_type,
            payload=payload,
            tenant_id=tenant_id,
        )

        self._producer.produce(
            topic=_TOPIC_AGENT_MESSAGES,
            key=str(recipient_agent_id).encode(),
            value=json.dumps(envelope).encode(),
            headers={"tenant_id": tenant_id.encode()},
        )
        self._producer.poll(0)

        logger.info(
            "Agent message dispatched",
            message_id=envelope["message_id"],
            sender=str(sender_agent_id),
            recipient=str(recipient_agent_id),
            message_type=message_type,
            tenant_id=tenant_id,
        )
        return envelope["message_id"]

    async def request_response(
        self,
        sender_agent_id: uuid.UUID,
        recipient_agent_id: uuid.UUID,
        request_payload: dict[str, Any],
        tenant_id: str,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Send a request to an agent and await its response.

        Creates a correlation ID, registers a future, publishes the request,
        and waits for the recipient to publish a response on the same channel.

        Args:
            sender_agent_id: Agent sending the request.
            recipient_agent_id: Agent expected to respond.
            request_payload: Request data dict.
            tenant_id: Tenant context.
            timeout_seconds: Override default response timeout.

        Returns:
            Response payload dict from the recipient agent.

        Raises:
            asyncio.TimeoutError: If the recipient does not respond in time.
        """
        correlation_id = str(uuid.uuid4())
        timeout = timeout_seconds or self._response_timeout

        # Register pending response future
        loop = asyncio.get_event_loop()
        response_future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_responses[correlation_id] = response_future

        # Store correlation metadata in Redis for distributed tracking
        await self._redis.setex(
            f"coord:pending:{tenant_id}:{correlation_id}",
            timeout,
            json.dumps({
                "sender": str(sender_agent_id),
                "recipient": str(recipient_agent_id),
                "requested_at": datetime.now(UTC).isoformat(),
            }),
        )

        envelope = self._build_message(
            sender_agent_id=sender_agent_id,
            recipient_agent_id=recipient_agent_id,
            message_type="request",
            payload=request_payload,
            tenant_id=tenant_id,
            correlation_id=correlation_id,
        )

        self._producer.produce(
            topic=_TOPIC_AGENT_MESSAGES,
            key=str(recipient_agent_id).encode(),
            value=json.dumps(envelope).encode(),
            headers={"tenant_id": tenant_id.encode(), "correlation_id": correlation_id.encode()},
        )
        self._producer.poll(0)

        logger.info(
            "Agent request dispatched",
            correlation_id=correlation_id,
            sender=str(sender_agent_id),
            recipient=str(recipient_agent_id),
            timeout_seconds=timeout,
            tenant_id=tenant_id,
        )

        try:
            response = await asyncio.wait_for(response_future, timeout=timeout)
            await self._redis.delete(f"coord:pending:{tenant_id}:{correlation_id}")
            return response
        except asyncio.TimeoutError:
            self._pending_responses.pop(correlation_id, None)
            await self._redis.delete(f"coord:pending:{tenant_id}:{correlation_id}")
            logger.error(
                "Agent request timed out",
                correlation_id=correlation_id,
                recipient=str(recipient_agent_id),
                timeout_seconds=timeout,
                tenant_id=tenant_id,
            )
            raise

    def resolve_response(self, correlation_id: str, response_payload: dict[str, Any]) -> None:
        """Resolve a pending request-response future with the received response.

        Called by the Kafka consumer callback when a response message arrives.

        Args:
            correlation_id: The correlation ID from the response envelope.
            response_payload: The response payload to deliver.
        """
        future = self._pending_responses.pop(correlation_id, None)
        if future is not None and not future.done():
            future.set_result(response_payload)
            logger.debug(
                "Agent response resolved",
                correlation_id=correlation_id,
            )

    async def broadcast(
        self,
        sender_agent_id: uuid.UUID,
        group_name: str,
        payload: dict[str, Any],
        tenant_id: str,
        message_type: str = "broadcast",
    ) -> str:
        """Broadcast a message to all agents in a named group.

        Publishes to the broadcast topic using the group name as the routing key.

        Args:
            sender_agent_id: Agent broadcasting the message.
            group_name: Agent group identifier (all subscribers receive the message).
            payload: Broadcast payload dict.
            tenant_id: Tenant context.
            message_type: Message category for subscriber handling.

        Returns:
            Message ID string.
        """
        message_id = str(uuid.uuid4())
        envelope = {
            "message_id": message_id,
            "sender_agent_id": str(sender_agent_id),
            "group_name": group_name,
            "message_type": message_type,
            "payload": payload,
            "tenant_id": tenant_id,
            "sent_at": datetime.now(UTC).isoformat(),
        }

        self._producer.produce(
            topic=_TOPIC_AGENT_BROADCAST,
            key=f"{tenant_id}:{group_name}".encode(),
            value=json.dumps(envelope).encode(),
            headers={"tenant_id": tenant_id.encode()},
        )
        self._producer.poll(0)

        logger.info(
            "Agent broadcast dispatched",
            message_id=message_id,
            sender=str(sender_agent_id),
            group_name=group_name,
            tenant_id=tenant_id,
        )
        return message_id

    # ─── Task delegation ──────────────────────────────────────────────────────

    async def delegate_task(
        self,
        delegator_agent_id: uuid.UUID,
        delegate_agent_id: uuid.UUID,
        task_description: str,
        task_input: dict[str, Any],
        tenant_id: str,
        priority: int = 5,
        requires_ack: bool = True,
    ) -> str:
        """Delegate a task from one agent to another via Kafka.

        Checks for dependency cycles before dispatching to prevent deadlocks.

        Args:
            delegator_agent_id: Agent delegating the task.
            delegate_agent_id: Agent receiving the task.
            task_description: Human-readable task description.
            task_input: Structured task input data.
            tenant_id: Tenant context.
            priority: Task priority 1-10 (10=highest).
            requires_ack: If True, wait for acknowledgement message.

        Returns:
            Task delegation ID for tracking.

        Raises:
            DeadlockDetectedError: If this delegation would create a dependency cycle.
        """
        # Add edge and check for cycles before sending
        delegator_str = str(delegator_agent_id)
        delegate_str = str(delegate_agent_id)

        self._dependency_graph.add_edge(delegator_str, delegate_str)
        if not nx.is_directed_acyclic_graph(self._dependency_graph):
            # Remove the just-added edge and raise
            self._dependency_graph.remove_edge(delegator_str, delegate_str)
            try:
                cycle = nx.find_cycle(self._dependency_graph, source=delegator_str)
                cycle_nodes = [u for u, _ in cycle] + [cycle[-1][1]]
            except nx.NetworkXNoCycle:
                cycle_nodes = [delegator_str, delegate_str]
            raise DeadlockDetectedError(cycle=cycle_nodes)

        task_id = str(uuid.uuid4())
        delegation_envelope = {
            "task_id": task_id,
            "delegator_agent_id": delegator_str,
            "delegate_agent_id": delegate_str,
            "task_description": task_description,
            "task_input": task_input,
            "priority": priority,
            "requires_ack": requires_ack,
            "tenant_id": tenant_id,
            "delegated_at": datetime.now(UTC).isoformat(),
        }

        self._producer.produce(
            topic=_TOPIC_TASK_DELEGATION,
            key=delegate_str.encode(),
            value=json.dumps(delegation_envelope).encode(),
            headers={"tenant_id": tenant_id.encode(), "priority": str(priority).encode()},
        )
        self._producer.poll(0)

        # Track pending delegation in Redis
        await self._redis.setex(
            f"coord:task:{tenant_id}:{task_id}",
            _DEFAULT_RESPONSE_TIMEOUT_SECONDS * 10,
            json.dumps({"status": "delegated", "delegate": delegate_str}),
        )

        logger.info(
            "Task delegated to agent",
            task_id=task_id,
            delegator=delegator_str,
            delegate=delegate_str,
            priority=priority,
            tenant_id=tenant_id,
        )
        return task_id

    async def acknowledge_task(
        self,
        task_id: str,
        acknowledging_agent_id: uuid.UUID,
        tenant_id: str,
        accepted: bool,
        reason: str | None = None,
    ) -> None:
        """Publish a task acknowledgement back to the delegator.

        Args:
            task_id: Task ID being acknowledged.
            acknowledging_agent_id: Agent sending the acknowledgement.
            tenant_id: Tenant context.
            accepted: True if agent accepted the task, False if rejected.
            reason: Optional reason for rejection.
        """
        ack_envelope = {
            "task_id": task_id,
            "agent_id": str(acknowledging_agent_id),
            "accepted": accepted,
            "reason": reason,
            "tenant_id": tenant_id,
            "acknowledged_at": datetime.now(UTC).isoformat(),
        }

        self._producer.produce(
            topic=_TOPIC_TASK_ACKNOWLEDGEMENT,
            key=task_id.encode(),
            value=json.dumps(ack_envelope).encode(),
        )
        self._producer.poll(0)

        # Update task status in Redis
        await self._redis.hset(
            f"coord:task:{tenant_id}:{task_id}",
            mapping={"status": "accepted" if accepted else "rejected"},
        )

        logger.info(
            "Task acknowledged",
            task_id=task_id,
            agent_id=str(acknowledging_agent_id),
            accepted=accepted,
            tenant_id=tenant_id,
        )

    # ─── Response aggregation ─────────────────────────────────────────────────

    async def gather_responses(
        self,
        sender_agent_id: uuid.UUID,
        recipient_agent_ids: list[uuid.UUID],
        request_payload: dict[str, Any],
        tenant_id: str,
        timeout_seconds: int | None = None,
        require_all: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Send requests to multiple agents and aggregate their responses.

        Dispatches requests concurrently and collects responses up to the
        timeout. Partial results are returned if require_all is False.

        Args:
            sender_agent_id: Agent orchestrating the gather.
            recipient_agent_ids: List of agents to query.
            request_payload: Payload sent to all recipients.
            tenant_id: Tenant context.
            timeout_seconds: Timeout for response collection.
            require_all: If True, raises TimeoutError if any agent fails to respond.

        Returns:
            Dict mapping agent_id string to response payload dict.
            Missing responses are omitted if require_all is False.
        """
        tasks = {
            str(recipient_id): asyncio.create_task(
                self.request_response(
                    sender_agent_id=sender_agent_id,
                    recipient_agent_id=recipient_id,
                    request_payload=request_payload,
                    tenant_id=tenant_id,
                    timeout_seconds=timeout_seconds,
                )
            )
            for recipient_id in recipient_agent_ids
        }

        responses: dict[str, dict[str, Any]] = {}
        for agent_id_str, task in tasks.items():
            try:
                responses[agent_id_str] = await task
            except asyncio.TimeoutError:
                if require_all:
                    raise
                logger.warning(
                    "Agent did not respond in gather window",
                    agent_id=agent_id_str,
                    tenant_id=tenant_id,
                )

        logger.info(
            "Response gather complete",
            sender=str(sender_agent_id),
            requested=len(recipient_agent_ids),
            received=len(responses),
            tenant_id=tenant_id,
        )
        return responses

    # ─── Health monitoring ────────────────────────────────────────────────────

    async def emit_heartbeat(self, coordinator_id: str, tenant_id: str) -> None:
        """Emit a coordinator heartbeat to Redis for liveness tracking.

        Args:
            coordinator_id: Unique identifier for this coordinator instance.
            tenant_id: Tenant context.
        """
        await self._redis.setex(
            f"coord:heartbeat:{tenant_id}:{coordinator_id}",
            self._heartbeat_ttl,
            datetime.now(UTC).isoformat(),
        )

    async def is_coordinator_alive(self, coordinator_id: str, tenant_id: str) -> bool:
        """Check if a coordinator instance is alive via Redis heartbeat.

        Args:
            coordinator_id: Coordinator instance to check.
            tenant_id: Tenant context.

        Returns:
            True if heartbeat key exists in Redis (not expired).
        """
        key = f"coord:heartbeat:{tenant_id}:{coordinator_id}"
        exists = await self._redis.exists(key)
        return bool(exists)

    def release_dependency(
        self,
        delegator_agent_id: uuid.UUID,
        delegate_agent_id: uuid.UUID,
    ) -> None:
        """Remove a task dependency edge from the deadlock detection graph.

        Called when a delegated task completes and the dependency is resolved.

        Args:
            delegator_agent_id: Agent that delegated the task.
            delegate_agent_id: Agent that completed the task.
        """
        delegator_str = str(delegator_agent_id)
        delegate_str = str(delegate_agent_id)

        if self._dependency_graph.has_edge(delegator_str, delegate_str):
            self._dependency_graph.remove_edge(delegator_str, delegate_str)
            logger.debug(
                "Dependency edge released",
                delegator=delegator_str,
                delegate=delegate_str,
            )
