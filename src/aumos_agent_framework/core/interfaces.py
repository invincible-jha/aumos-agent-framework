"""Protocol interfaces for aumos-agent-framework adapters.

All adapters must implement these protocols to ensure the core services
remain framework-independent and testable.
"""

from typing import Any, AsyncIterator, Callable, Awaitable, Protocol, runtime_checkable
from uuid import UUID

import pydantic


# ============================================================================
# AumOS-native workflow graph types (Gap #134 — LangGraph abstraction layer)
# These types appear ONLY here and in langgraph_engine.py.
# No other file should import from langgraph directly.
# ============================================================================

NodeFunction = Callable[[dict[str, Any]], dict[str, Any]]


class WorkflowNodeDefinition(pydantic.BaseModel):
    """AumOS-native definition of a workflow graph node.

    This is the abstraction layer between the AumOS API and the underlying
    graph execution engine (currently LangGraph). Swapping the engine requires
    changes only in adapters/workflow_engine/langgraph_engine.py.
    """

    node_id: str = pydantic.Field(..., description="Unique node identifier within the workflow")
    metadata: dict[str, Any] = pydantic.Field(
        default_factory=dict,
        description="Engine-specific node metadata (agent_id, system_prompt, etc.)",
    )


class WorkflowEdgeDefinition(pydantic.BaseModel):
    """AumOS-native definition of an edge between workflow nodes."""

    source_node_id: str = pydantic.Field(..., description="Source node identifier")
    target_node_id: str = pydantic.Field(..., description="Target node identifier")
    condition: str | None = pydantic.Field(
        None,
        description="Optional Python expression evaluated at runtime to decide traversal",
    )


class WorkflowGraphDefinition(pydantic.BaseModel):
    """Complete AumOS workflow graph structure.

    Consumed by WorkflowEngineProtocol.compile() to produce an executable graph.
    No LangGraph types appear in this schema.
    """

    entry_point: str = pydantic.Field(..., description="Node ID where execution begins")
    nodes: list[WorkflowNodeDefinition] = pydantic.Field(
        ..., min_length=1, description="All nodes in the workflow graph"
    )
    edges: list[WorkflowEdgeDefinition] = pydantic.Field(
        default_factory=list, description="Directed edges connecting nodes"
    )
    state_schema: dict[str, Any] = pydantic.Field(
        default_factory=dict, description="JSON Schema for the graph state object"
    )


# ============================================================================
# Pre-built tool protocol (Gap #129 — Tool marketplace)
# ============================================================================


class ToolInputSchema(pydantic.BaseModel):
    """Base class for all tool input schemas. Override in each pre-built tool."""

    pass


class ToolOutputSchema(pydantic.BaseModel):
    """Base class for all tool output schemas. Override in each pre-built tool."""

    result: Any
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)


@runtime_checkable
class AumOSToolProtocol(Protocol):
    """Protocol that all AumOS pre-built tools must implement.

    Tools are the atomic capabilities that agents can invoke.
    Every tool is stateless — all context comes via input_data.

    Attributes:
        tool_id: Unique identifier, e.g. "web_search_serper".
        display_name: Human-readable name for UI display.
        category: Category bucket: "web", "data", "communication", "code", "document", "ai", "aumos".
        description: What this tool does (shown in tool selector UI).
        privilege_level: Minimum agent privilege to invoke (1-5).
        input_schema: Pydantic model class for input validation.
        output_schema: Pydantic model class for output validation.
    """

    tool_id: str
    display_name: str
    category: str
    description: str
    privilege_level: int
    input_schema: type[ToolInputSchema]
    output_schema: type[ToolOutputSchema]

    async def execute(
        self,
        input_data: ToolInputSchema,
        tenant_id: str,
        config: dict[str, str],
    ) -> ToolOutputSchema:
        """Execute the tool with the provided input.

        Args:
            input_data: Validated input matching input_schema.
            tenant_id: Tenant context for per-tenant rate limiting.
            config: Tool-specific configuration (API keys, endpoints).

        Returns:
            Validated output matching output_schema.
        """
        ...


@runtime_checkable
class WorkflowEngineProtocol(Protocol):
    """Executes graph-based workflows using the AumOS workflow engine abstraction.

    This protocol intentionally hides all LangGraph types. Callers interact only
    with AumOS-native types (WorkflowGraphDefinition, WorkflowStreamEvent).
    The LangGraph adapter in adapters/workflow_engine/langgraph_engine.py is the
    only module that imports from langgraph.
    """

    async def compile(self, definition: WorkflowGraphDefinition) -> Any:
        """Compile a workflow graph definition into an executable graph object.

        Args:
            definition: AumOS-native workflow graph definition.

        Returns:
            Engine-specific compiled graph object (opaque to callers).
        """
        ...

    async def execute(
        self,
        compiled_graph: Any,
        initial_state: dict[str, Any],
        execution_id: str,
        tenant_id: str,
    ) -> dict[str, Any]:
        """Execute a compiled workflow graph and return the final state.

        Args:
            compiled_graph: Result of compile(), engine-specific type.
            initial_state: Initial state dict for the workflow.
            execution_id: Unique execution ID for tracing.
            tenant_id: Tenant context for RLS isolation.

        Returns:
            Final state dict from the completed workflow graph.
        """
        ...

    async def execute_streaming(
        self,
        compiled_graph: Any,
        initial_state: dict[str, Any],
        execution_id: str,
        tenant_id: str,
    ) -> AsyncIterator["WorkflowStreamEvent"]:  # type: ignore[type-arg]
        """Execute a compiled workflow graph and yield streaming events.

        Yields one event per LLM token, tool invocation, and node completion.

        Args:
            compiled_graph: Result of compile(), engine-specific type.
            initial_state: Initial state dict for the workflow.
            execution_id: Unique execution ID for tracing.
            tenant_id: Tenant context for RLS isolation.

        Yields:
            WorkflowStreamEvent for each significant workflow milestone.
        """
        ...

    async def get_graph_nodes(self, definition: WorkflowGraphDefinition) -> list[str]:
        """Return all node IDs in the workflow graph.

        Args:
            definition: AumOS-native workflow graph definition.

        Returns:
            List of node ID strings.
        """
        ...


@runtime_checkable
class AgentExecutorProtocol(Protocol):
    """Executes individual agent actions within a workflow node."""

    async def invoke_agent(
        self,
        agent_id: UUID,
        input_data: dict[str, Any],
        session_id: str,
        tenant_id: str,
        privilege_level: int,
    ) -> dict[str, Any]:
        """Invoke an agent and return its output.

        Args:
            agent_id: ID of the registered agent to invoke.
            input_data: Input data for the agent.
            session_id: Session ID for memory context.
            tenant_id: Tenant context for isolation.
            privilege_level: Effective privilege level for this invocation.

        Returns:
            Agent output as a dict.
        """
        ...


@runtime_checkable
class HITLGateProtocol(Protocol):
    """Manages human-in-the-loop approval gate flow."""

    async def trigger_gate(
        self,
        execution_id: UUID,
        gate_name: str,
        agent_id: UUID,
        action_description: str,
        context_data: dict[str, Any],
        tenant_id: str,
    ) -> UUID:
        """Create a pending HITL approval and pause workflow execution.

        Args:
            execution_id: ID of the workflow execution being paused.
            gate_name: Name of the HITL gate as defined in workflow definition.
            agent_id: Agent that triggered the gate.
            action_description: Human-readable description of what needs approval.
            context_data: Additional context for the approver.
            tenant_id: Tenant context.

        Returns:
            UUID of the created HITLApproval record.
        """
        ...

    async def approve(
        self,
        approval_id: UUID,
        decided_by: UUID,
        notes: str | None,
        tenant_id: str,
    ) -> bool:
        """Approve a pending HITL gate and resume workflow.

        Returns:
            True if workflow was successfully resumed.
        """
        ...

    async def reject(
        self,
        approval_id: UUID,
        decided_by: UUID,
        notes: str | None,
        tenant_id: str,
    ) -> bool:
        """Reject a pending HITL gate and fail the workflow.

        Returns:
            True if workflow was successfully cancelled.
        """
        ...


@runtime_checkable
class CircuitBreakerProtocol(Protocol):
    """Per-agent and per-workflow circuit breaker for cascading failure containment."""

    async def is_open(self, circuit_key: str, tenant_id: str) -> bool:
        """Check if the circuit is open (blocking requests).

        Args:
            circuit_key: Unique key identifying the circuit (e.g., "agent:{id}" or "workflow:{id}")
            tenant_id: Tenant context for isolation.

        Returns:
            True if circuit is OPEN and requests should be rejected.
        """
        ...

    async def record_success(self, circuit_key: str, tenant_id: str) -> None:
        """Record a successful call — may transition HALF_OPEN to CLOSED."""
        ...

    async def record_failure(self, circuit_key: str, tenant_id: str) -> None:
        """Record a failed call — may transition CLOSED to OPEN."""
        ...

    async def get_state(self, circuit_key: str, tenant_id: str) -> str:
        """Get current circuit state: 'closed', 'open', or 'half_open'."""
        ...


@runtime_checkable
class SessionIsolatorProtocol(Protocol):
    """Per-tenant agent session isolation with Redis-backed state."""

    async def get_session(
        self,
        agent_id: UUID,
        tenant_id: str,
        execution_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Retrieve agent session data.

        Args:
            agent_id: Agent whose session to retrieve.
            tenant_id: Tenant context (enforces namespace isolation).
            execution_id: Optional workflow execution for scoped sessions.

        Returns:
            Session data dict, empty dict if no session exists.
        """
        ...

    async def update_session(
        self,
        agent_id: UUID,
        tenant_id: str,
        updates: dict[str, Any],
        execution_id: UUID | None = None,
    ) -> None:
        """Update agent session data with partial updates."""
        ...

    async def clear_session(
        self,
        agent_id: UUID,
        tenant_id: str,
        execution_id: UUID | None = None,
    ) -> None:
        """Clear agent session data."""
        ...

    async def get_shared_session(
        self,
        execution_id: UUID,
        tenant_id: str,
    ) -> dict[str, Any]:
        """Retrieve shared session data for all agents in a workflow execution."""
        ...

    async def update_shared_session(
        self,
        execution_id: UUID,
        tenant_id: str,
        updates: dict[str, Any],
    ) -> None:
        """Update shared session data accessible to all agents in an execution."""
        ...


@runtime_checkable
class ToolRegistryProtocol(Protocol):
    """Registry for tools available to agents."""

    async def register_tool(
        self,
        name: str,
        description: str,
        min_privilege_level: int,
        input_schema: dict[str, Any],
        config: dict[str, Any],
        tenant_id: str,
    ) -> UUID:
        """Register a new tool in the registry.

        Returns:
            UUID of the created ToolDefinition.
        """
        ...

    async def get_tool(self, tool_name: str, tenant_id: str) -> dict[str, Any] | None:
        """Retrieve a tool definition by name.

        Returns:
            Tool definition dict, or None if not found.
        """
        ...

    async def list_tools(
        self,
        tenant_id: str,
        min_privilege_level: int | None = None,
    ) -> list[dict[str, Any]]:
        """List all available tools, optionally filtered by privilege level."""
        ...

    async def check_access(
        self,
        agent_id: UUID,
        tool_name: str,
        tenant_id: str,
    ) -> bool:
        """Check whether an agent has access to a specific tool.

        Returns:
            True if access is allowed.
        """
        ...


@runtime_checkable
class DurableExecutorProtocol(Protocol):
    """Durable workflow execution via Temporal for failure-resilient processing."""

    async def start_workflow(
        self,
        workflow_id: str,
        workflow_definition: dict[str, Any],
        input_data: dict[str, Any],
        tenant_id: str,
    ) -> str:
        """Start a durable workflow execution via Temporal.

        Returns:
            Temporal workflow run ID.
        """
        ...

    async def cancel_workflow(self, workflow_id: str, tenant_id: str) -> bool:
        """Cancel a running durable workflow.

        Returns:
            True if successfully cancelled.
        """
        ...

    async def get_workflow_status(self, workflow_id: str, tenant_id: str) -> dict[str, Any]:
        """Get current status of a durable workflow execution."""
        ...


# ─── New adapter protocols added in Phase 1B expansion ───────────────────────


@runtime_checkable
class AgentMemoryManagerProtocol(Protocol):
    """Dual-layer agent memory — Redis short-term + PostgreSQL long-term."""

    async def add_short_term(
        self,
        agent_id: UUID,
        tenant_id: str,
        content: str,
        memory_type: str,
        importance: float,
        metadata: dict[str, Any] | None,
    ) -> str:
        """Add an entry to the agent's short-term Redis memory.

        Args:
            agent_id: Owning agent.
            tenant_id: Tenant namespace.
            content: Memory content string.
            memory_type: Category label (e.g., 'conversation', 'reasoning').
            importance: Importance score 0.0-1.0.
            metadata: Optional extra metadata.

        Returns:
            Generated entry ID string.
        """
        ...

    async def get_short_term(
        self,
        agent_id: UUID,
        tenant_id: str,
        limit: int,
        memory_type: str | None,
    ) -> list[dict[str, Any]]:
        """Retrieve short-term memory entries, newest first.

        Args:
            agent_id: Owning agent.
            tenant_id: Tenant namespace.
            limit: Maximum entries to return.
            memory_type: Optional category filter.

        Returns:
            List of memory entry dicts.
        """
        ...

    async def store_long_term(
        self,
        agent_id: UUID,
        tenant_id: str,
        content: str,
        memory_type: str,
        topic: str,
        importance: float,
        embedding: list[float] | None,
        metadata: dict[str, Any] | None,
    ) -> str:
        """Persist a memory entry in PostgreSQL long-term storage.

        Returns:
            UUID string of the created memory record.
        """
        ...

    async def search_long_term(
        self,
        agent_id: UUID,
        tenant_id: str,
        query_embedding: list[float],
        limit: int,
        topic: str | None,
        memory_type: str | None,
        min_importance: float,
        since_hours: int | None,
    ) -> list[dict[str, Any]]:
        """Search long-term memory by embedding similarity.

        Returns:
            List of memory dicts ordered by cosine similarity.
        """
        ...

    async def consolidate_memory(
        self,
        agent_id: UUID,
        tenant_id: str,
        get_embedding_fn: Any,
    ) -> int:
        """Promote high-importance short-term entries to long-term storage.

        Returns:
            Number of entries promoted.
        """
        ...

    async def clear_short_term(self, agent_id: UUID, tenant_id: str) -> None:
        """Clear all short-term memory and working context for an agent."""
        ...


@runtime_checkable
class ReasoningEngineProtocol(Protocol):
    """LLM-backed reasoning engine — chain-of-thought and ReAct loops."""

    async def chain_of_thought(
        self,
        agent_id: UUID,
        task_description: str,
        context: dict[str, Any],
        tenant_id: str,
        model_id: str | None,
    ) -> dict[str, Any]:
        """Execute chain-of-thought multi-step reasoning.

        Args:
            agent_id: Agent executing the reasoning.
            task_description: Task or question to reason about.
            context: Additional context dict (memory, tools, facts).
            tenant_id: Tenant context.
            model_id: Optional model override.

        Returns:
            ReasoningTrace serialized as dict with 'steps' and 'final_conclusion'.
        """
        ...

    async def react_loop(
        self,
        agent_id: UUID,
        task_description: str,
        available_tools: list[dict[str, Any]],
        context: dict[str, Any],
        tenant_id: str,
        execute_tool_fn: Any,
        model_id: str | None,
    ) -> dict[str, Any]:
        """Execute a ReAct (Reason + Act) loop with tool invocations.

        Returns:
            ReasoningTrace dict with interleaved thought/action/observation steps.
        """
        ...

    async def select_tools(
        self,
        agent_id: UUID,
        task_description: str,
        available_tools: list[dict[str, Any]],
        tenant_id: str,
        top_k: int,
        model_id: str | None,
    ) -> list[dict[str, Any]]:
        """Recommend relevant tools for a task using LLM reasoning.

        Returns:
            List of tool dicts enriched with 'relevance_score'.
        """
        ...


@runtime_checkable
class ActionExecutorProtocol(Protocol):
    """Sandboxed tool invocation with timeout, retry, and audit logging."""

    async def execute(
        self,
        tool_name: str,
        tool_definition: dict[str, Any],
        tool_input: dict[str, Any],
        agent_id: UUID,
        tenant_id: str,
        execution_id: str | None,
    ) -> dict[str, Any]:
        """Execute a tool invocation with full lifecycle management.

        Args:
            tool_name: Name of the tool to invoke.
            tool_definition: Full tool definition including config and schema.
            tool_input: Validated input data.
            agent_id: Agent performing the action.
            tenant_id: Tenant context.
            execution_id: Optional workflow execution correlation ID.

        Returns:
            Audit result dict with 'output', 'success', 'duration_ms', 'attempt_count'.
        """
        ...

    async def execute_with_fallback(
        self,
        tool_name: str,
        tool_definition: dict[str, Any],
        tool_input: dict[str, Any],
        agent_id: UUID,
        tenant_id: str,
        fallback_output: dict[str, Any],
        execution_id: str | None,
    ) -> dict[str, Any]:
        """Execute a tool with a safe fallback on failure.

        Returns:
            Result dict with 'used_fallback' bool added.
        """
        ...


@runtime_checkable
class ObservabilityHooksProtocol(Protocol):
    """OpenTelemetry tracing and custom metrics for agent observability."""

    async def log_state_change(
        self,
        agent_id: UUID,
        tenant_id: str,
        previous_state: str,
        new_state: str,
        reason: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Log an agent state transition with structured context.

        Args:
            agent_id: Agent whose state changed.
            tenant_id: Tenant context.
            previous_state: State before transition.
            new_state: State after transition.
            reason: Human-readable reason for the change.
            metadata: Optional extra context.
        """
        ...

    async def record_tokens_consumed(
        self,
        agent_id: UUID,
        tenant_id: str,
        token_count: int,
        model_id: str,
        action_type: str,
    ) -> None:
        """Record LLM token consumption for cost attribution.

        Args:
            agent_id: Consuming agent.
            tenant_id: Tenant context.
            token_count: Tokens consumed.
            model_id: Model identifier.
            action_type: Category (e.g., 'inference', 'embedding').
        """
        ...

    async def get_agent_metrics_summary(
        self,
        agent_id: UUID,
        tenant_id: str,
        date_str: str | None,
    ) -> dict[str, Any]:
        """Return aggregated daily metrics for an agent.

        Returns:
            Dict with token counts, state changes, and action stats.
        """
        ...


@runtime_checkable
class MultiAgentCoordinatorProtocol(Protocol):
    """Inter-agent messaging and task delegation via Kafka."""

    async def send_message(
        self,
        sender_agent_id: UUID,
        recipient_agent_id: UUID,
        payload: dict[str, Any],
        tenant_id: str,
        message_type: str,
    ) -> str:
        """Send a fire-and-forget message to another agent.

        Returns:
            Message ID string.
        """
        ...

    async def request_response(
        self,
        sender_agent_id: UUID,
        recipient_agent_id: UUID,
        request_payload: dict[str, Any],
        tenant_id: str,
        timeout_seconds: int | None,
    ) -> dict[str, Any]:
        """Send a request and await the recipient agent's response.

        Returns:
            Response payload dict from the recipient.
        """
        ...

    async def broadcast(
        self,
        sender_agent_id: UUID,
        group_name: str,
        payload: dict[str, Any],
        tenant_id: str,
        message_type: str,
    ) -> str:
        """Broadcast a message to all agents in a named group.

        Returns:
            Message ID string.
        """
        ...

    async def delegate_task(
        self,
        delegator_agent_id: UUID,
        delegate_agent_id: UUID,
        task_description: str,
        task_input: dict[str, Any],
        tenant_id: str,
        priority: int,
        requires_ack: bool,
    ) -> str:
        """Delegate a task to another agent with deadlock detection.

        Returns:
            Task delegation ID string.
        """
        ...

    async def gather_responses(
        self,
        sender_agent_id: UUID,
        recipient_agent_ids: list[UUID],
        request_payload: dict[str, Any],
        tenant_id: str,
        timeout_seconds: int | None,
        require_all: bool,
    ) -> dict[str, dict[str, Any]]:
        """Query multiple agents and aggregate their responses.

        Returns:
            Dict mapping agent_id string to response payload.
        """
        ...


@runtime_checkable
class SkillComposerProtocol(Protocol):
    """Hierarchical skill registry and composite skill execution."""

    async def execute_skill(
        self,
        skill_name: str,
        skill_input: dict[str, Any],
        agent_id: UUID,
        tenant_id: str,
        context: dict[str, Any] | None,
        version: str | None,
    ) -> dict[str, Any]:
        """Execute a registered skill by name, resolving dependencies first.

        Returns:
            SkillExecutionResult serialized as dict.
        """
        ...

    async def execute_composite(
        self,
        mode: str,
        skill_names: list[str],
        skill_inputs: list[dict[str, Any]],
        agent_id: UUID,
        tenant_id: str,
        context: dict[str, Any] | None,
        condition_fn: Callable[[dict[str, Any]], int] | None,
    ) -> dict[str, Any]:
        """Execute a composite skill in sequence, parallel, or conditional mode.

        Returns:
            SkillExecutionResult dict with child_results populated.
        """
        ...

    def recommend_skills(
        self,
        task_description: str,
        agent_privilege_level: int,
        tags: list[str] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Recommend relevant skills based on task description matching.

        Returns:
            Sorted list of skill recommendation dicts with relevance scores.
        """
        ...

    def list_skills(
        self,
        include_deprecated: bool,
        tag: str | None,
        min_privilege_level: int | None,
    ) -> list[dict[str, Any]]:
        """List registered skills with optional filtering.

        Returns:
            List of skill summary dicts.
        """
        ...


@runtime_checkable
class SecuritySandboxProtocol(Protocol):
    """Isolated code execution in Docker containers with resource limits."""

    async def execute_code(
        self,
        code: str,
        input_data: dict[str, Any],
        agent_id: UUID,
        tenant_id: str,
        policy: Any,
    ) -> dict[str, Any]:
        """Execute Python code in an isolated Docker sandbox.

        Args:
            code: Python source code to execute.
            input_data: Input data dict available as INPUT variable in code.
            agent_id: Agent invoking the sandbox.
            tenant_id: Tenant context.
            policy: SandboxPolicy or None to use the default policy.

        Returns:
            SandboxExecutionResult serialized as dict.
        """
        ...

    async def build_policy_for_agent(
        self,
        agent_config: dict[str, Any],
        privilege_level: int,
    ) -> Any:
        """Build a SandboxPolicy appropriate for an agent's privilege level.

        Returns:
            SandboxPolicy instance.
        """
        ...


@runtime_checkable
class AgentRuntimeProtocol(Protocol):
    """Agent lifecycle management — initialization, run loop, and shutdown."""

    async def initialize(
        self,
        agent_id: UUID,
        tenant_id: str,
        agent_config: dict[str, Any],
        instance_id: str | None,
        resume_from_crash: bool,
    ) -> str:
        """Initialize a new agent runtime instance.

        Returns:
            Instance ID string.
        """
        ...

    async def run(
        self,
        instance_id: str,
        observe_fn: Any,
        think_fn: Any,
        act_fn: Any,
    ) -> dict[str, Any]:
        """Execute the observe-think-act run loop.

        Returns:
            Final agent state dict after completion.
        """
        ...

    def pause(self, instance_id: str) -> None:
        """Signal an agent instance to pause after the current step."""
        ...

    def resume(self, instance_id: str) -> None:
        """Resume a paused agent instance."""
        ...

    def request_stop(self, instance_id: str) -> None:
        """Signal an agent instance to stop after the current step."""
        ...

    async def shutdown(self, instance_id: str) -> None:
        """Gracefully shut down an agent instance with full cleanup."""
        ...

    async def health_check(self, instance_id: str, tenant_id: str) -> dict[str, Any]:
        """Return health status for a runtime instance.

        Returns:
            Dict with 'healthy', 'state', 'step_count', and 'heartbeat_age_seconds'.
        """
        ...
