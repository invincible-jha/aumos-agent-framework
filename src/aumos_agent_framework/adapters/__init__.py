"""Adapters for external integrations — LangGraph, Temporal, Redis, Kafka, Docker."""

from aumos_agent_framework.adapters.action_executor import ActionExecutor, ActionExecutionError
from aumos_agent_framework.adapters.agent_runtime import AgentRuntime, AgentRuntimeState, AgentRuntimeError
from aumos_agent_framework.adapters.circuit_breaker import RedisCircuitBreaker
from aumos_agent_framework.adapters.durable_execution.temporal_executor import TemporalExecutor
from aumos_agent_framework.adapters.kafka import AgentFrameworkKafkaPublisher
from aumos_agent_framework.adapters.memory_manager import AgentMemoryManager
from aumos_agent_framework.adapters.multi_agent_coordinator import (
    MultiAgentCoordinator,
    DeadlockDetectedError,
)
from aumos_agent_framework.adapters.observability_hooks import ObservabilityHooks
from aumos_agent_framework.adapters.privilege_manager import PrivilegeManager
from aumos_agent_framework.adapters.reasoning_engine import (
    ReasoningEngine,
    ReasoningTrace,
    ReasoningStep,
)
from aumos_agent_framework.adapters.security_sandbox import (
    SecuritySandbox,
    SandboxPolicy,
    SandboxExecutionResult,
)
from aumos_agent_framework.adapters.session_isolator import RedisSessionIsolator
from aumos_agent_framework.adapters.skill_composer import (
    SkillComposer,
    SkillDefinition,
    SkillExecutionResult,
    SkillNotFoundError,
    SkillDependencyCycleError,
)
from aumos_agent_framework.adapters.tool_registry import DatabaseToolRegistry
from aumos_agent_framework.adapters.workflow_engine.langgraph_engine import LangGraphEngine

__all__ = [
    # Original adapters
    "AgentFrameworkKafkaPublisher",
    "DatabaseToolRegistry",
    "LangGraphEngine",
    "PrivilegeManager",
    "RedisCircuitBreaker",
    "RedisSessionIsolator",
    "TemporalExecutor",
    # New Phase 1B adapters
    "ActionExecutor",
    "ActionExecutionError",
    "AgentMemoryManager",
    "AgentRuntime",
    "AgentRuntimeError",
    "AgentRuntimeState",
    "DeadlockDetectedError",
    "MultiAgentCoordinator",
    "ObservabilityHooks",
    "ReasoningEngine",
    "ReasoningStep",
    "ReasoningTrace",
    "SandboxExecutionResult",
    "SandboxPolicy",
    "SecuritySandbox",
    "SkillComposer",
    "SkillDefinition",
    "SkillDependencyCycleError",
    "SkillExecutionResult",
    "SkillNotFoundError",
]
