"""Agent lifecycle management runtime — initialization, run loop, and graceful shutdown.

Implements AgentRuntimeProtocol providing:
- Agent initialization from config, memory, and tool registry.
- Observe → Think → Act → Observe run loop with configurable step limits.
- Pause/resume with full state serialization to Redis.
- Graceful shutdown with resource cleanup.
- Health check and heartbeat emission.
- State persistence for crash recovery.
- Concurrent agent instance management with capacity enforcement.
"""

import asyncio
import json
import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from redis.asyncio import Redis

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_DEFAULT_MAX_RUN_STEPS = 50
_DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 10
_DEFAULT_STATE_TTL_SECONDS = 86400  # 24 hours
_DEFAULT_MAX_CONCURRENT_INSTANCES = 100

# Redis key patterns
_RUNTIME_STATE_KEY = "runtime:state:{tenant_id}:{agent_id}:{instance_id}"
_RUNTIME_HEARTBEAT_KEY = "runtime:heartbeat:{tenant_id}:{agent_id}:{instance_id}"
_RUNTIME_CAPACITY_KEY = "runtime:instances:{tenant_id}"


class AgentRuntimeState(str, Enum):
    """Valid states for an agent runtime instance.

    Attributes:
        INITIALIZING: Agent is loading config, memory, and tools.
        IDLE: Agent is initialized but not running.
        RUNNING: Agent is actively executing the run loop.
        PAUSED: Agent run loop is suspended, state persisted.
        STOPPING: Agent received shutdown signal, cleaning up.
        STOPPED: Agent has completed cleanup and exited.
        ERROR: Agent encountered an unrecoverable error.
    """

    INITIALIZING = "initializing"
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class AgentRuntimeError(Exception):
    """Raised when the agent runtime encounters an unrecoverable error.

    Attributes:
        agent_id: ID of the affected agent.
        instance_id: Runtime instance ID.
        phase: Run loop phase where the error occurred.
    """

    def __init__(
        self,
        message: str,
        agent_id: uuid.UUID,
        instance_id: str,
        phase: str,
    ) -> None:
        super().__init__(message)
        self.agent_id = agent_id
        self.instance_id = instance_id
        self.phase = phase


class AgentRuntime:
    """Manages the full lifecycle of a single agent execution instance.

    One AgentRuntime instance corresponds to one run of an agent against a
    specific task. Multiple runtime instances can coexist for the same agent
    (different tasks). Capacity is enforced via Redis counters per tenant.
    """

    def __init__(
        self,
        redis_client: Redis,  # type: ignore[type-arg]
        max_run_steps: int = _DEFAULT_MAX_RUN_STEPS,
        heartbeat_interval_seconds: int = _DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        state_ttl_seconds: int = _DEFAULT_STATE_TTL_SECONDS,
        max_concurrent_instances: int = _DEFAULT_MAX_CONCURRENT_INSTANCES,
    ) -> None:
        """Initialize the agent runtime manager.

        Args:
            redis_client: Async Redis client for state persistence and heartbeat.
            max_run_steps: Maximum observe-think-act cycles before forced stop.
            heartbeat_interval_seconds: Interval between heartbeat emissions.
            state_ttl_seconds: TTL for serialized state in Redis.
            max_concurrent_instances: Maximum concurrent agent instances per tenant.
        """
        self._redis = redis_client
        self._max_run_steps = max_run_steps
        self._heartbeat_interval = heartbeat_interval_seconds
        self._state_ttl = state_ttl_seconds
        self._max_concurrent = max_concurrent_instances

        # In-process instance registry
        self._instances: dict[str, dict[str, Any]] = {}
        # Per-instance cancellation events
        self._stop_events: dict[str, asyncio.Event] = {}
        self._pause_events: dict[str, asyncio.Event] = {}

    # ─── Instance management ──────────────────────────────────────────────────

    async def _check_capacity(self, tenant_id: str) -> bool:
        """Check if capacity allows a new agent instance for this tenant.

        Args:
            tenant_id: Tenant to check capacity for.

        Returns:
            True if a new instance can be created.
        """
        current_count = await self._redis.get(_RUNTIME_CAPACITY_KEY.format(tenant_id=tenant_id))
        count = int(current_count) if current_count else 0
        return count < self._max_concurrent

    async def _increment_instance_count(self, tenant_id: str) -> None:
        """Increment the running instance count for a tenant in Redis."""
        await self._redis.incr(_RUNTIME_CAPACITY_KEY.format(tenant_id=tenant_id))

    async def _decrement_instance_count(self, tenant_id: str) -> None:
        """Decrement the running instance count for a tenant in Redis."""
        key = _RUNTIME_CAPACITY_KEY.format(tenant_id=tenant_id)
        current = await self._redis.get(key)
        if current and int(current) > 0:
            await self._redis.decr(key)

    # ─── State persistence ────────────────────────────────────────────────────

    async def _persist_state(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        instance_id: str,
        runtime_state: AgentRuntimeState,
        agent_state: dict[str, Any],
        step_count: int,
    ) -> None:
        """Serialize and persist agent runtime state to Redis.

        Args:
            agent_id: Agent whose state to persist.
            tenant_id: Tenant context.
            instance_id: Runtime instance ID.
            runtime_state: Current AgentRuntimeState enum value.
            agent_state: Agent's internal state dict (memory, goals, context).
            step_count: Number of run loop steps completed.
        """
        state_doc = {
            "agent_id": str(agent_id),
            "tenant_id": tenant_id,
            "instance_id": instance_id,
            "runtime_state": runtime_state.value,
            "agent_state": agent_state,
            "step_count": step_count,
            "persisted_at": datetime.now(UTC).isoformat(),
        }

        key = _RUNTIME_STATE_KEY.format(
            tenant_id=tenant_id,
            agent_id=str(agent_id),
            instance_id=instance_id,
        )
        await self._redis.setex(key, self._state_ttl, json.dumps(state_doc))

    async def _load_persisted_state(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        instance_id: str,
    ) -> dict[str, Any] | None:
        """Load previously persisted agent state from Redis.

        Args:
            agent_id: Agent whose state to load.
            tenant_id: Tenant context.
            instance_id: Runtime instance ID.

        Returns:
            State dict if found, None if no persisted state exists.
        """
        key = _RUNTIME_STATE_KEY.format(
            tenant_id=tenant_id,
            agent_id=str(agent_id),
            instance_id=instance_id,
        )
        raw = await self._redis.get(key)
        if raw is None:
            return None
        return json.loads(raw)  # type: ignore[no-any-return]

    # ─── Run loop ─────────────────────────────────────────────────────────────

    async def initialize(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        agent_config: dict[str, Any],
        instance_id: str | None = None,
        resume_from_crash: bool = False,
    ) -> str:
        """Initialize a new agent runtime instance.

        Loads configuration, checks capacity, and transitions to IDLE state.
        If resume_from_crash is True, attempts to restore persisted state.

        Args:
            agent_id: Agent to initialize.
            tenant_id: Tenant context.
            agent_config: Agent configuration dict with tools, memory, and settings.
            instance_id: Optional instance ID (generated if not provided).
            resume_from_crash: If True, attempt to restore persisted state.

        Returns:
            Instance ID string for this runtime instance.

        Raises:
            RuntimeError: If tenant instance capacity is exceeded.
        """
        if not await self._check_capacity(tenant_id):
            raise RuntimeError(
                f"Agent instance capacity exceeded for tenant {tenant_id} "
                f"(max: {self._max_concurrent})"
            )

        instance_id = instance_id or str(uuid.uuid4())
        initial_state: dict[str, Any] = {"goals": [], "memory": {}, "context": {}}

        if resume_from_crash:
            persisted = await self._load_persisted_state(agent_id, tenant_id, instance_id)
            if persisted:
                initial_state = persisted.get("agent_state", initial_state)
                logger.info(
                    "Agent runtime resuming from persisted state",
                    agent_id=str(agent_id),
                    instance_id=instance_id,
                    persisted_at=persisted.get("persisted_at"),
                    tenant_id=tenant_id,
                )

        self._instances[instance_id] = {
            "agent_id": str(agent_id),
            "tenant_id": tenant_id,
            "config": agent_config,
            "state": AgentRuntimeState.IDLE,
            "agent_state": initial_state,
            "step_count": 0,
            "initialized_at": datetime.now(UTC).isoformat(),
        }
        self._stop_events[instance_id] = asyncio.Event()
        self._pause_events[instance_id] = asyncio.Event()

        await self._increment_instance_count(tenant_id)
        await self._persist_state(
            agent_id=agent_id,
            tenant_id=tenant_id,
            instance_id=instance_id,
            runtime_state=AgentRuntimeState.IDLE,
            agent_state=initial_state,
            step_count=0,
        )

        logger.info(
            "Agent runtime initialized",
            agent_id=str(agent_id),
            instance_id=instance_id,
            resumed=resume_from_crash,
            tenant_id=tenant_id,
        )
        return instance_id

    async def run(
        self,
        instance_id: str,
        observe_fn: Any,  # Callable[[dict], Awaitable[dict]]
        think_fn: Any,   # Callable[[dict, dict], Awaitable[dict]]
        act_fn: Any,     # Callable[[dict, dict], Awaitable[dict]]
    ) -> dict[str, Any]:
        """Execute the observe-think-act run loop for an agent instance.

        Runs until: max_run_steps reached, stop event set, or think_fn returns
        {'done': True} to signal task completion.

        Args:
            instance_id: Runtime instance to run.
            observe_fn: Async callable that observes environment and returns observation dict.
            think_fn: Async callable(observation, agent_state) → action_plan dict.
            act_fn: Async callable(action_plan, agent_state) → execution_result dict.

        Returns:
            Final agent state dict after the run loop completes.

        Raises:
            AgentRuntimeError: If an unrecoverable error occurs in any phase.
        """
        instance = self._instances.get(instance_id)
        if instance is None:
            raise AgentRuntimeError(
                f"Instance {instance_id} not found",
                agent_id=uuid.UUID(instance.get("agent_id", str(uuid.uuid4()))) if instance else uuid.uuid4(),
                instance_id=instance_id,
                phase="run_start",
            )

        agent_id = uuid.UUID(instance["agent_id"])
        tenant_id = instance["tenant_id"]

        instance["state"] = AgentRuntimeState.RUNNING
        await self._persist_state(
            agent_id=agent_id,
            tenant_id=tenant_id,
            instance_id=instance_id,
            runtime_state=AgentRuntimeState.RUNNING,
            agent_state=instance["agent_state"],
            step_count=instance["step_count"],
        )

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(
            self._emit_heartbeat_loop(agent_id, tenant_id, instance_id)
        )

        logger.info(
            "Agent run loop started",
            agent_id=str(agent_id),
            instance_id=instance_id,
            max_steps=self._max_run_steps,
            tenant_id=tenant_id,
        )

        stop_event = self._stop_events[instance_id]
        pause_event = self._pause_events[instance_id]

        try:
            for step in range(self._max_run_steps):
                # Check for stop signal
                if stop_event.is_set():
                    logger.info(
                        "Agent run loop stopped by signal",
                        agent_id=str(agent_id),
                        instance_id=instance_id,
                        step=step,
                        tenant_id=tenant_id,
                    )
                    break

                # Handle pause — wait until resumed or stopped
                if pause_event.is_set():
                    logger.info(
                        "Agent run loop paused",
                        agent_id=str(agent_id),
                        instance_id=instance_id,
                        step=step,
                        tenant_id=tenant_id,
                    )
                    instance["state"] = AgentRuntimeState.PAUSED
                    await self._persist_state(
                        agent_id=agent_id,
                        tenant_id=tenant_id,
                        instance_id=instance_id,
                        runtime_state=AgentRuntimeState.PAUSED,
                        agent_state=instance["agent_state"],
                        step_count=instance["step_count"],
                    )
                    # Wait until unpaused or stopped
                    while pause_event.is_set() and not stop_event.is_set():
                        await asyncio.sleep(0.5)
                    if stop_event.is_set():
                        break
                    instance["state"] = AgentRuntimeState.RUNNING
                    logger.info(
                        "Agent run loop resumed",
                        agent_id=str(agent_id),
                        instance_id=instance_id,
                        tenant_id=tenant_id,
                    )

                agent_state = instance["agent_state"]
                instance["step_count"] = step + 1

                # OBSERVE phase
                try:
                    observation = await observe_fn(agent_state)
                except Exception as exc:
                    raise AgentRuntimeError(
                        str(exc), agent_id, instance_id, "observe"
                    ) from exc

                # THINK phase
                try:
                    action_plan = await think_fn(observation, agent_state)
                except Exception as exc:
                    raise AgentRuntimeError(
                        str(exc), agent_id, instance_id, "think"
                    ) from exc

                # Check if agent signals task completion
                if action_plan.get("done", False):
                    logger.info(
                        "Agent signaled task completion",
                        agent_id=str(agent_id),
                        instance_id=instance_id,
                        step=step + 1,
                        tenant_id=tenant_id,
                    )
                    break

                # ACT phase
                try:
                    execution_result = await act_fn(action_plan, agent_state)
                except Exception as exc:
                    raise AgentRuntimeError(
                        str(exc), agent_id, instance_id, "act"
                    ) from exc

                # Update agent state with execution result
                instance["agent_state"]["last_observation"] = observation
                instance["agent_state"]["last_action"] = action_plan
                instance["agent_state"]["last_result"] = execution_result

                logger.debug(
                    "Agent run loop step complete",
                    agent_id=str(agent_id),
                    instance_id=instance_id,
                    step=step + 1,
                    tenant_id=tenant_id,
                )

            instance["state"] = AgentRuntimeState.IDLE
            return instance["agent_state"]

        except AgentRuntimeError:
            instance["state"] = AgentRuntimeState.ERROR
            raise

        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            await self._persist_state(
                agent_id=agent_id,
                tenant_id=tenant_id,
                instance_id=instance_id,
                runtime_state=instance["state"],
                agent_state=instance["agent_state"],
                step_count=instance["step_count"],
            )
            logger.info(
                "Agent run loop finished",
                agent_id=str(agent_id),
                instance_id=instance_id,
                steps_completed=instance["step_count"],
                final_state=instance["state"].value,
                tenant_id=tenant_id,
            )

    async def _emit_heartbeat_loop(
        self,
        agent_id: uuid.UUID,
        tenant_id: str,
        instance_id: str,
    ) -> None:
        """Continuously emit heartbeat signals while the run loop is active.

        Args:
            agent_id: Agent to emit heartbeats for.
            tenant_id: Tenant context.
            instance_id: Runtime instance ID.
        """
        key = _RUNTIME_HEARTBEAT_KEY.format(
            tenant_id=tenant_id,
            agent_id=str(agent_id),
            instance_id=instance_id,
        )
        while True:
            try:
                await self._redis.setex(
                    key,
                    self._heartbeat_interval * 3,
                    datetime.now(UTC).isoformat(),
                )
                await asyncio.sleep(self._heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(
                    "Heartbeat emission failed",
                    agent_id=str(agent_id),
                    instance_id=instance_id,
                    error=str(exc),
                )
                await asyncio.sleep(self._heartbeat_interval)

    # ─── Pause / Resume / Stop ────────────────────────────────────────────────

    def pause(self, instance_id: str) -> None:
        """Signal an agent run loop to pause after the current step.

        Args:
            instance_id: Runtime instance to pause.
        """
        event = self._pause_events.get(instance_id)
        if event and not event.is_set():
            event.set()
            logger.info("Agent pause signal sent", instance_id=instance_id)

    def resume(self, instance_id: str) -> None:
        """Resume a paused agent run loop.

        Args:
            instance_id: Runtime instance to resume.
        """
        event = self._pause_events.get(instance_id)
        if event and event.is_set():
            event.clear()
            logger.info("Agent resume signal sent", instance_id=instance_id)

    def request_stop(self, instance_id: str) -> None:
        """Signal an agent run loop to stop after the current step.

        Args:
            instance_id: Runtime instance to stop.
        """
        event = self._stop_events.get(instance_id)
        if event and not event.is_set():
            event.set()
            logger.info("Agent stop signal sent", instance_id=instance_id)

    async def shutdown(self, instance_id: str) -> None:
        """Gracefully shut down an agent instance with full cleanup.

        Signals stop, waits for run loop to complete, persists final state,
        and releases capacity slot.

        Args:
            instance_id: Runtime instance to shut down.
        """
        instance = self._instances.get(instance_id)
        if not instance:
            return

        agent_id = uuid.UUID(instance["agent_id"])
        tenant_id = instance["tenant_id"]

        instance["state"] = AgentRuntimeState.STOPPING
        self.request_stop(instance_id)

        logger.info(
            "Agent runtime shutting down",
            agent_id=str(agent_id),
            instance_id=instance_id,
            tenant_id=tenant_id,
        )

        # Allow brief period for run loop to acknowledge stop
        await asyncio.sleep(0.1)

        instance["state"] = AgentRuntimeState.STOPPED
        await self._persist_state(
            agent_id=agent_id,
            tenant_id=tenant_id,
            instance_id=instance_id,
            runtime_state=AgentRuntimeState.STOPPED,
            agent_state=instance["agent_state"],
            step_count=instance["step_count"],
        )

        # Clean up in-process tracking
        self._instances.pop(instance_id, None)
        self._stop_events.pop(instance_id, None)
        self._pause_events.pop(instance_id, None)

        await self._decrement_instance_count(tenant_id)

        logger.info(
            "Agent runtime stopped",
            agent_id=str(agent_id),
            instance_id=instance_id,
            tenant_id=tenant_id,
        )

    # ─── Health and monitoring ────────────────────────────────────────────────

    async def health_check(self, instance_id: str, tenant_id: str) -> dict[str, Any]:
        """Return health status for a runtime instance.

        Args:
            instance_id: Runtime instance to check.
            tenant_id: Tenant context.

        Returns:
            Dict with 'healthy', 'state', 'step_count', and 'heartbeat_age_seconds'.
        """
        instance = self._instances.get(instance_id)
        if not instance:
            return {"healthy": False, "reason": "Instance not found"}

        agent_id_str = instance["agent_id"]
        heartbeat_key = _RUNTIME_HEARTBEAT_KEY.format(
            tenant_id=tenant_id,
            agent_id=agent_id_str,
            instance_id=instance_id,
        )
        heartbeat_raw = await self._redis.get(heartbeat_key)

        heartbeat_age: float | None = None
        if heartbeat_raw:
            last_beat = datetime.fromisoformat(heartbeat_raw.decode())
            heartbeat_age = (datetime.now(UTC) - last_beat).total_seconds()

        current_state = instance["state"]
        healthy = current_state in (
            AgentRuntimeState.RUNNING,
            AgentRuntimeState.IDLE,
            AgentRuntimeState.PAUSED,
        )

        return {
            "healthy": healthy,
            "instance_id": instance_id,
            "agent_id": agent_id_str,
            "state": current_state.value,
            "step_count": instance["step_count"],
            "heartbeat_age_seconds": heartbeat_age,
            "initialized_at": instance.get("initialized_at"),
        }

    def get_active_instance_ids(self) -> list[str]:
        """Return all currently active instance IDs in this runtime.

        Returns:
            List of instance ID strings.
        """
        return list(self._instances.keys())
