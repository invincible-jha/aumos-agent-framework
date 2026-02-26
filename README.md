# aumos-agent-framework

Multi-agent orchestration framework for AumOS Enterprise. Provides graph-based workflow execution with LangGraph, durable execution via Temporal, 5-level agent privilege enforcement, human-in-the-loop approval gates, per-agent circuit breakers, and tenant-aware session isolation.

## Features

- **Graph-based workflows** — LangGraph `StateGraph` with typed state and conditional routing
- **Durable execution** — Temporal integration for failure-resilient, retryable workflows
- **5-level privilege system** — READ_ONLY through SUPER_ADMIN with tool access control
- **HITL gates** — Human-in-the-loop approval gates at any workflow node
- **Circuit breakers** — Per-agent and per-workflow cascading failure containment
- **Session isolation** — Redis-backed per-tenant agent memory with private/shared scoping
- **Kafka communication** — Protobuf envelope agent-to-agent messaging
- **Tenant-aware** — Full RLS isolation, all operations scoped to tenant

## Quick Start

```bash
cp .env.example .env
make docker-up
make migrate
```

Service starts at `http://localhost:8007`.

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/workflows` | Create workflow definition |
| `POST` | `/api/v1/workflows/{id}/execute` | Execute workflow |
| `GET` | `/api/v1/workflows/{id}/status` | Get execution status |
| `POST` | `/api/v1/workflows/{id}/cancel` | Cancel workflow |
| `POST` | `/api/v1/agents` | Register agent |
| `GET` | `/api/v1/agents` | List agents (tenant-scoped) |
| `PUT` | `/api/v1/agents/{id}/tools` | Update agent tool access |
| `POST` | `/api/v1/agents/{id}/invoke` | Directly invoke agent |
| `GET` | `/api/v1/hitl/pending` | List pending approvals |
| `POST` | `/api/v1/hitl/{id}/approve` | Approve HITL gate |
| `POST` | `/api/v1/hitl/{id}/reject` | Reject HITL gate |

## Architecture

```
api/          FastAPI routes (thin layer)
core/         Business logic (framework-independent)
  models.py   SQLAlchemy ORM (agf_ prefix)
  services.py Agent registry, workflow, HITL, circuit breaker, session services
  interfaces.py Protocol definitions
adapters/     External integrations
  workflow_engine/langgraph_engine.py
  durable_execution/temporal_executor.py
  circuit_breaker.py
  session_isolator.py
  privilege_manager.py
  tool_registry.py
  kafka_transport.py
```

## Agent Privilege Levels

| Level | Name | Description |
|-------|------|-------------|
| 1 | READ_ONLY | Read data only |
| 2 | EXECUTE_SAFE | Safe, reversible tool calls |
| 3 | EXECUTE_RISKY | Potentially irreversible — triggers HITL |
| 4 | ADMIN | Administrative actions — requires approval |
| 5 | SUPER_ADMIN | Platform-level operations |

## Development

```bash
make dev-install  # Install dependencies
make lint         # Run ruff linter
make typecheck    # Run mypy
make test         # Run tests (80% coverage required)
make check-all    # Lint + typecheck + test
```

## Dependencies

- `aumos-common` — auth, database, events, errors, config
- `aumos-proto` — Protobuf definitions
- `aumos-event-bus` — Kafka transport
- `aumos-auth-gateway` — JWT validation
- `aumos-llm-serving` — LLM model invocation

## License

Apache 2.0 — see [LICENSE](LICENSE)
