"""SQLAlchemy ORM models for aumos-agent-framework.

All models use the agf_ table prefix and extend AumOSModel for tenant isolation.
"""

import uuid
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from aumos_common.database import AumOSModel


class AgentDefinition(AumOSModel):
    """Defines an agent with its capabilities, privilege level, and tool access.

    Agents are registered by tenants and can be assigned to workflow nodes.
    Privilege levels control what actions the agent can perform.
    """

    __tablename__ = "agf_agent_definitions"

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Capabilities this agent can perform (e.g., ["data_analysis", "code_generation"])
    capabilities: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # 1=READ_ONLY, 2=EXECUTE_SAFE, 3=EXECUTE_RISKY, 4=ADMIN, 5=SUPER_ADMIN
    privilege_level: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # Tools this agent can access: {"tool_name": {"enabled": true, "rate_limit": 100}}
    tool_access: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # Resource limits: {"max_tokens": 4096, "max_calls_per_minute": 60}
    resource_limits: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # active | suspended | retired
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="active", index=True)

    # Relationships
    sessions: Mapped[list["AgentSession"]] = relationship("AgentSession", back_populates="agent")
    hitl_approvals: Mapped[list["HITLApproval"]] = relationship(
        "HITLApproval", back_populates="agent"
    )


class WorkflowDefinition(AumOSModel):
    """Defines a workflow as a LangGraph-compatible graph with agents and gates.

    The graph_definition stores the LangGraph StateGraph definition serialized as JSON.
    Agents maps node names to agent IDs. HITL gates define approval requirements.
    """

    __tablename__ = "agf_workflow_definitions"

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)

    # LangGraph StateGraph definition: nodes, edges, conditional edges
    graph_definition: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    # Maps node names to agent definition IDs: {"analyze_node": "agent-uuid", ...}
    agents: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # HITL gate configs keyed by gate name
    hitl_gates: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # Circuit breaker config for this workflow
    # {"failure_threshold": 5, "reset_timeout_seconds": 60}
    circuit_breaker_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict
    )

    # Relationships
    executions: Mapped[list["WorkflowExecution"]] = relationship(
        "WorkflowExecution", back_populates="workflow"
    )


class WorkflowExecution(AumOSModel):
    """Tracks the runtime state of a workflow execution.

    Status transitions:
    pending → running → (paused_hitl → running)* → completed | failed | cancelled
    """

    __tablename__ = "agf_workflow_executions"

    workflow_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agf_workflow_definitions.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    # pending | running | paused_hitl | completed | failed | cancelled
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending", index=True)

    # Current node being executed in the graph
    current_node: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Full execution history: list of node transition events with timestamps and outputs
    execution_history: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )

    # Input data for the workflow
    input_data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # Output data from completed workflow
    output_data: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Error details if failed
    error_details: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Temporal workflow ID for durable execution tracking
    temporal_workflow_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    started_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), nullable=True, server_default=func.now()
    )
    completed_at: Mapped[Any | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    workflow: Mapped["WorkflowDefinition"] = relationship(
        "WorkflowDefinition", back_populates="executions"
    )
    hitl_approvals: Mapped[list["HITLApproval"]] = relationship(
        "HITLApproval", back_populates="execution"
    )


class HITLApproval(AumOSModel):
    """Records a human-in-the-loop approval gate instance.

    Created when an agent with privilege_level >= 3 attempts an action,
    or when a workflow node is configured with an explicit HITL gate.
    Workflow is paused until approved or rejected.
    """

    __tablename__ = "agf_hitl_approvals"

    execution_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agf_workflow_executions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    gate_name: Mapped[str] = mapped_column(String(255), nullable=False)

    # The agent that triggered this HITL gate
    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agf_agent_definitions.id", ondelete="RESTRICT"),
        nullable=False,
    )

    # Human-readable description of the action awaiting approval
    action_description: Mapped[str] = mapped_column(Text, nullable=False)

    # Context data for the approver to make informed decision
    context_data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # pending | approved | rejected
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pending", index=True
    )

    # UUID of the user who decided (from JWT subject claim)
    decided_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)

    decided_at: Mapped[Any | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Optional rejection reason
    decision_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    execution: Mapped["WorkflowExecution"] = relationship(
        "WorkflowExecution", back_populates="hitl_approvals"
    )
    agent: Mapped["AgentDefinition"] = relationship(
        "AgentDefinition", back_populates="hitl_approvals"
    )


class AgentSession(AumOSModel):
    """Per-tenant, per-agent session with isolated memory state.

    Sessions track agent memory across a workflow execution.
    Memory scope controls whether memory is private to the agent
    or shared with other agents in the same workflow execution.
    """

    __tablename__ = "agf_agent_sessions"

    agent_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agf_agent_definitions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Optional: link to a specific workflow execution
    execution_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True
    )

    # Session state stored in DB (Redis is primary, DB is checkpoint)
    session_data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # private = only this agent can read/write
    # shared = all agents in same workflow execution can read/write
    memory_scope: Mapped[str] = mapped_column(String(50), nullable=False, default="private")

    started_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), nullable=True, server_default=func.now()
    )
    last_activity_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), nullable=True, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    agent: Mapped["AgentDefinition"] = relationship("AgentDefinition", back_populates="sessions")


class ToolDefinition(AumOSModel):
    """Registry of tools available for agent use.

    Tools are registered by the system and assigned to agents via tool_access on AgentDefinition.
    """

    __tablename__ = "agf_tool_definitions"

    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)

    # Minimum privilege level required to use this tool
    min_privilege_level: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # JSON schema for tool input parameters
    input_schema: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # Tool configuration and endpoint details
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    # active | deprecated | disabled
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="active")
