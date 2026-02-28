"""Pydantic request/response schemas for aumos-agent-framework API."""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ============================================================================
# Agent Schemas
# ============================================================================


class AgentCreateRequest(BaseModel):
    """Request body for registering a new agent."""

    name: str = Field(..., min_length=1, max_length=255, description="Human-readable agent name")
    description: str | None = Field(None, description="Optional agent description")
    capabilities: dict[str, Any] = Field(
        default_factory=dict,
        description="Capability tags: {'data_analysis': true, 'code_generation': true}",
    )
    privilege_level: int = Field(
        ...,
        ge=1,
        le=5,
        description="1=READ_ONLY, 2=EXECUTE_SAFE, 3=EXECUTE_RISKY, 4=ADMIN, 5=SUPER_ADMIN",
    )
    tool_access: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool access config: {'tool_name': {'enabled': true, 'rate_limit': 100}}",
    )
    resource_limits: dict[str, Any] = Field(
        default_factory=dict,
        description="Resource limits: {'max_tokens': 4096, 'max_calls_per_minute': 60}",
    )


class AgentUpdateToolAccessRequest(BaseModel):
    """Request body for updating agent tool access."""

    tool_access: dict[str, Any] = Field(
        ...,
        description="New tool access configuration replacing existing config",
    )


class AgentInvokeRequest(BaseModel):
    """Request body for directly invoking an agent."""

    input_data: dict[str, Any] = Field(..., description="Input data for agent invocation")
    execution_id: uuid.UUID | None = Field(
        None, description="Optional workflow execution context for session scoping"
    )


class AgentResponse(BaseModel):
    """Response schema for an agent definition."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    description: str | None
    capabilities: dict[str, Any]
    privilege_level: int
    tool_access: dict[str, Any]
    resource_limits: dict[str, Any]
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class AgentInvokeResponse(BaseModel):
    """Response schema for agent invocation."""

    agent_id: uuid.UUID
    output: dict[str, Any]
    session_id: str
    tokens_used: int | None = None
    execution_time_ms: int | None = None


class AgentListResponse(BaseModel):
    """Paginated list of agents."""

    items: list[AgentResponse]
    total: int
    page: int
    page_size: int


# ============================================================================
# Workflow Schemas
# ============================================================================


class WorkflowCreateRequest(BaseModel):
    """Request body for creating a workflow definition."""

    name: str = Field(..., min_length=1, max_length=255, description="Human-readable workflow name")
    graph_definition: dict[str, Any] = Field(
        ...,
        description="LangGraph StateGraph definition: {nodes: {...}, edges: [...], state_schema: {...}}",
    )
    agents: dict[str, Any] = Field(
        default_factory=dict,
        description="Maps node names to agent IDs: {'analyze_node': 'agent-uuid'}",
    )
    hitl_gates: dict[str, Any] = Field(
        default_factory=dict,
        description="HITL gate configs: {'gate_name': {'trigger_condition': '...', 'approval_timeout_hours': 24}}",
    )
    circuit_breaker_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Circuit breaker settings: {'failure_threshold': 5, 'reset_timeout_seconds': 60}",
    )


class WorkflowExecuteRequest(BaseModel):
    """Request body for executing a workflow."""

    input_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Initial input state for the workflow execution",
    )


class WorkflowCancelRequest(BaseModel):
    """Request body for cancelling a workflow execution."""

    reason: str | None = Field(None, description="Optional cancellation reason")


class WorkflowResponse(BaseModel):
    """Response schema for a workflow definition."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    graph_definition: dict[str, Any]
    agents: dict[str, Any]
    hitl_gates: dict[str, Any]
    circuit_breaker_config: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ExecutionResponse(BaseModel):
    """Response schema for a workflow execution."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    workflow_id: uuid.UUID
    status: str
    current_node: str | None
    execution_history: list[dict[str, Any]]
    input_data: dict[str, Any]
    output_data: dict[str, Any] | None
    error_details: dict[str, Any] | None
    temporal_workflow_id: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ExecutionStatusResponse(BaseModel):
    """Lightweight execution status response."""

    id: uuid.UUID
    workflow_id: uuid.UUID
    status: str
    current_node: str | None
    started_at: datetime | None
    completed_at: datetime | None
    temporal_workflow_id: str | None


# ============================================================================
# HITL Schemas
# ============================================================================


class HITLApproveRequest(BaseModel):
    """Request body for approving a HITL gate."""

    notes: str | None = Field(None, description="Optional approval notes")


class HITLRejectRequest(BaseModel):
    """Request body for rejecting a HITL gate."""

    notes: str | None = Field(None, description="Optional rejection reason")


class HITLApprovalResponse(BaseModel):
    """Response schema for a HITL approval record."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    execution_id: uuid.UUID
    gate_name: str
    agent_id: uuid.UUID
    action_description: str
    context_data: dict[str, Any]
    status: str
    decided_by: uuid.UUID | None
    decided_at: datetime | None
    decision_notes: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class HITLPendingListResponse(BaseModel):
    """List of pending HITL approvals."""

    items: list[HITLApprovalResponse]
    total: int


# ============================================================================
# Circuit Breaker Schemas
# ============================================================================


class CircuitBreakerStateResponse(BaseModel):
    """Circuit breaker state for an agent or workflow."""

    circuit_key: str
    state: str  # closed | open | half_open
    tenant_id: str


# ============================================================================
# Tool Schemas
# ============================================================================


class ToolRegisterRequest(BaseModel):
    """Request body for registering a tool."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str = Field(..., min_length=1)
    min_privilege_level: int = Field(..., ge=1, le=5)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class ToolResponse(BaseModel):
    """Response schema for a tool definition."""

    id: uuid.UUID
    name: str
    description: str
    min_privilege_level: int
    input_schema: dict[str, Any]
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ============================================================================
# Streaming Schemas (Gap #132)
# ============================================================================


class WorkflowStreamEvent(BaseModel):
    """Single event in a workflow execution SSE stream.

    Each event has a type field indicating the event category:
    - token: An LLM reasoning token (delta)
    - tool_start: A tool invocation has begun
    - tool_end: A tool invocation has completed
    - node_complete: A workflow node has finished
    - hitl_pause: Workflow paused for human approval
    - workflow_complete: The workflow has finished
    - error: An error occurred
    """

    event_type: Literal[
        "token",
        "tool_start",
        "tool_end",
        "node_complete",
        "hitl_pause",
        "workflow_complete",
        "error",
    ] = Field(..., description="Type of SSE event")
    data: dict[str, Any] = Field(default_factory=dict, description="Event payload")
    execution_id: uuid.UUID | None = Field(None, description="Workflow execution UUID")
    node_id: str | None = Field(None, description="LangGraph node that emitted this event")
    timestamp: datetime | None = Field(None, description="Event timestamp (UTC)")


# ============================================================================
# Execution Trace Schemas (Gap #131)
# ============================================================================


class ExecutionTraceResponse(BaseModel):
    """Response schema for a single execution trace entry.

    Traces capture the complete LLM call, tool call, or HITL event data
    for each node in a workflow execution.
    """

    id: uuid.UUID = Field(..., description="Trace entry UUID")
    execution_id: uuid.UUID = Field(..., description="Parent workflow execution UUID")
    node_id: str = Field(..., description="LangGraph node that produced this trace")
    trace_type: Literal["llm_call", "tool_call", "hitl_event"] = Field(
        ..., description="Category of trace entry"
    )
    model_name: str | None = Field(None, description="LLM model used (llm_call only)")
    prompt_tokens: int | None = Field(None, description="Input token count")
    completion_tokens: int | None = Field(None, description="Output token count")
    latency_ms: int | None = Field(None, description="Wall-clock latency in milliseconds")
    tool_id: str | None = Field(None, description="Tool identifier (tool_call only)")
    tool_error: str | None = Field(None, description="Tool error message if failed")
    hitl_gate_name: str | None = Field(None, description="HITL gate name (hitl_event only)")
    hitl_decision: str | None = Field(None, description="approved | rejected | timeout")
    started_at: datetime = Field(..., description="Trace start timestamp (UTC)")
    completed_at: datetime | None = Field(None, description="Trace completion timestamp (UTC)")
    error: str | None = Field(None, description="Error message if trace failed")
    created_at: datetime = Field(..., description="Record creation timestamp")

    model_config = {"from_attributes": True}


class ExecutionTraceListResponse(BaseModel):
    """Paginated list of execution traces."""

    items: list[ExecutionTraceResponse]
    total: int
    execution_id: uuid.UUID


# ============================================================================
# Checkpoint / Replay Schemas (Gap #133)
# ============================================================================


class CheckpointEvent(BaseModel):
    """Single event from the Temporal workflow event history.

    Temporal maintains a complete event log for every workflow execution.
    This schema provides a human-readable representation of each event.
    """

    event_id: int = Field(..., description="Sequential event ID in the Temporal history")
    event_type: str = Field(..., description="Temporal event type (e.g., WorkflowExecutionStarted)")
    timestamp: datetime = Field(..., description="Event timestamp (UTC)")
    data: dict[str, Any] = Field(default_factory=dict, description="Event-specific data")


class CheckpointListResponse(BaseModel):
    """List of checkpoint events for a workflow execution."""

    execution_id: uuid.UUID
    temporal_workflow_id: str | None
    events: list[CheckpointEvent]
    total: int


class ReplayRequest(BaseModel):
    """Request body for replaying a workflow execution with modified input."""

    override_input: dict[str, Any] | None = Field(
        None,
        description="Optional input override. If omitted, uses original execution input.",
    )


# ============================================================================
# Execution Node State Schema (Gap #128)
# ============================================================================


class ExecutionNodeStateResponse(BaseModel):
    """Real-time state of a single node within a workflow execution."""

    node_id: str = Field(..., description="LangGraph node identifier")
    status: Literal["pending", "running", "completed", "failed", "hitl_paused"] = Field(
        ..., description="Current node status"
    )
    started_at: datetime | None = Field(None, description="Node start timestamp")
    completed_at: datetime | None = Field(None, description="Node completion timestamp")
    error_message: str | None = Field(None, description="Error message if status is failed")


# ============================================================================
# Sandbox Guard Schemas (Gap #130)
# ============================================================================


class SandboxRateLimitResponse(BaseModel):
    """Response when a sandbox rate limit is exceeded."""

    detail: str = Field(..., description="Rate limit error message")
    retry_after_seconds: int = Field(..., description="Seconds until the rate limit resets")
