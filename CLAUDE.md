# CLAUDE.md — AumOS Agent Framework

## Project Overview

AumOS Enterprise is a composable enterprise AI platform with 9 products + 2 services
across 62 repositories. This repo (`aumos-agent-framework`) is part of **Tier B: Open Core**:
Multi-agent orchestration infrastructure used by ALL AumOS products.

**Release Tier:** B: Open Core
**Product Mapping:** Cross-cutting — provides to ALL products
**Phase:** 1B (Months 4-8)

## Repo Purpose

Provides graph-based multi-agent workflow orchestration with durable execution via Temporal,
cascading failure containment via circuit breakers, human-in-the-loop (HITL) approval gates,
5-level agent privilege enforcement, and per-tenant session isolation. This is the agent
execution engine that powers every AI workflow in the AumOS platform.

## Architecture Position

```
aumos-platform-core
aumos-auth-gateway ──→ aumos-agent-framework ──→ ALL AumOS Products
aumos-event-bus    ──↗        ↓                    (consume workflows)
aumos-data-layer   ──↗   aumos-llm-serving
                         (invokes models)
```

**Upstream dependencies (this repo IMPORTS from):**
- `aumos-common` — auth, database, events, errors, config, health, pagination
- `aumos-proto` — Protobuf message definitions for Kafka events
- `aumos-event-bus` — Kafka transport for agent communication messages
- `aumos-auth-gateway` — JWT validation, tenant context, privilege checks
- `aumos-llm-serving` — LLM model invocation for agent reasoning

**Downstream dependents (other repos IMPORT from this):**
- ALL AumOS product repos — consume workflow execution APIs
- `aumos-data-pipeline` — data processing workflows
- `aumos-federated-learning` — FL orchestration workflows

## Tech Stack (DO NOT DEVIATE)

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.11+ | Runtime |
| FastAPI | 0.110+ | REST API framework |
| SQLAlchemy | 2.0+ (async) | Database ORM |
| asyncpg | 0.29+ | PostgreSQL async driver |
| Pydantic | 2.6+ | Data validation, settings, API schemas |
| LangGraph | 0.0.40+ | Graph-based workflow execution |
| Temporalio | 1.4.0+ | Durable workflow execution |
| NetworkX | 3.2.0+ | Graph topology analysis |
| Redis | 5.0+ | Session state, circuit breaker state |
| confluent-kafka | 2.3+ | Kafka producer/consumer |
| structlog | 24.1+ | Structured JSON logging |
| OpenTelemetry | 1.23+ | Distributed tracing |
| pytest | 8.0+ | Testing framework |
| ruff | 0.3+ | Linting and formatting |
| mypy | 1.8+ | Type checking |

## Coding Standards

### ABSOLUTE RULES (violations will break integration with other repos)

1. **Import aumos-common, never reimplement.**
   ```python
   from aumos_common.auth import get_current_tenant, get_current_user
   from aumos_common.database import get_db_session, Base, AumOSModel, BaseRepository
   from aumos_common.events import EventPublisher, Topics
   from aumos_common.errors import NotFoundError, ErrorCode
   from aumos_common.config import AumOSSettings
   from aumos_common.health import create_health_router
   from aumos_common.pagination import PageRequest, PageResponse, paginate
   from aumos_common.app import create_app
   ```

2. **Type hints on EVERY function.** No exceptions.

3. **Pydantic models for ALL API inputs/outputs.** Never return raw dicts.

4. **RLS tenant isolation via aumos-common.** Never bypass RLS.

5. **Structured logging via structlog.**
   ```python
   from aumos_common.observability import get_logger
   logger = get_logger(__name__)
   ```

6. **Publish domain events to Kafka after state changes.**

7. **Async by default.** All I/O operations must be async.

8. **Google-style docstrings** on all public classes and functions.

### Style Rules

- Max line length: **120 characters**
- Import order: stdlib → third-party → aumos-common → local
- Linter: `ruff` (select E, W, F, I, N, UP, ANN, B, A, COM, C4, PT, RUF)
- Type checker: `mypy` strict mode
- Formatter: `ruff format`

### File Structure

```
src/aumos_agent_framework/
├── __init__.py
├── main.py
├── settings.py
├── api/
│   ├── __init__.py
│   ├── router.py
│   └── schemas.py
├── core/
│   ├── __init__.py
│   ├── models.py          # SQLAlchemy ORM — agf_ table prefix
│   ├── services.py        # Business logic (400-600 lines)
│   └── interfaces.py      # Protocol classes
└── adapters/
    ├── __init__.py
    ├── repositories.py
    ├── kafka.py
    ├── workflow_engine/
    │   └── langgraph_engine.py
    ├── durable_execution/
    │   └── temporal_executor.py
    ├── privilege_manager.py
    ├── circuit_breaker.py
    ├── session_isolator.py
    └── tool_registry.py
```

## Database Conventions

- Table prefix: `agf_` (e.g., `agf_agent_definitions`, `agf_workflow_executions`)
- ALL tenant-scoped tables: extend `AumOSModel` (gets id, tenant_id, created_at, updated_at)
- RLS policy on every tenant table (created in migration)

## Agent Privilege Levels

| Level | Name | Can Do |
|-------|------|--------|
| 1 | READ_ONLY | Read data, no mutations |
| 2 | EXECUTE_SAFE | Safe tool calls, reversible actions |
| 3 | EXECUTE_RISKY | Potentially irreversible actions, needs HITL gate |
| 4 | ADMIN | Administrative actions, requires approval |
| 5 | SUPER_ADMIN | All actions, platform-level ops |

- Privilege level 3+ automatically triggers HITL gate evaluation
- HITL gates can be configured per workflow node
- Circuit breakers are per-agent AND per-workflow

## Circuit Breaker States

- **CLOSED** — normal operation, requests flow through
- **OPEN** — failure threshold exceeded, requests rejected immediately
- **HALF_OPEN** — test state after reset timeout, allows limited requests

Cascading failure containment: if a sub-workflow fails repeatedly, parent workflow
is also protected via circuit breaker chain.

## HITL Gate Configuration

```json
{
  "gate_name": "high_risk_action",
  "trigger_condition": "privilege_level >= 3",
  "approval_timeout_hours": 24,
  "escalation_policy": "notify_admin",
  "auto_reject_on_timeout": false
}
```

## Kafka Events Published

- `agf.workflow.started` — workflow execution began
- `agf.workflow.completed` — workflow execution finished
- `agf.workflow.failed` — workflow execution failed
- `agf.workflow.paused_hitl` — workflow paused awaiting human approval
- `agf.agent.invoked` — agent directly invoked
- `agf.hitl.approval_requested` — HITL approval gate triggered
- `agf.hitl.approved` — HITL gate approved
- `agf.hitl.rejected` — HITL gate rejected
- `agf.circuit_breaker.opened` — circuit breaker tripped
- `agf.circuit_breaker.closed` — circuit breaker recovered

## Repo-Specific Context

### LangGraph Integration
- Workflows are defined as LangGraph `StateGraph` with typed state
- Nodes are Python callables (async functions or agent executors)
- Edges can be conditional (using `add_conditional_edges`)
- Checkpointing is handled by Temporal for durability

### Temporal Integration
- Temporal handles durable execution, retries, timeouts
- Workflow activities map to LangGraph node executions
- Temporal ensures at-least-once delivery of node execution
- Use `temporalio.workflow.defn` and `temporalio.activity.defn` decorators

### Session Isolation
- Each (tenant_id, agent_id) pair gets an isolated Redis namespace
- Memory scope: `private` (agent-only) or `shared` (all agents in workflow)
- Session TTL configurable via `AUMOS_AGF_SESSION_TTL_SECONDS`

### Performance Requirements
- Workflow execution start latency: < 200ms (excluding agent inference)
- HITL gate evaluation: < 50ms
- Circuit breaker check: < 5ms (Redis-backed state)
- Max concurrent workflows per tenant: configurable, default 100

## What Claude Code Should NOT Do

1. **Do NOT reimplement anything in aumos-common.**
2. **Do NOT use print().** Use `get_logger(__name__)`.
3. **Do NOT return raw dicts from API endpoints.**
4. **Do NOT write raw SQL.** Use SQLAlchemy ORM.
5. **Do NOT hardcode configuration.** Use Pydantic Settings.
6. **Do NOT skip type hints.**
7. **Do NOT put business logic in API routes.**
8. **Do NOT bypass the privilege system.** All agent actions must check privilege level.
9. **Do NOT skip circuit breaker checks.** Every agent/workflow call must go through the circuit breaker.
10. **Do NOT allow cross-tenant data access.** Session isolation is critical.
