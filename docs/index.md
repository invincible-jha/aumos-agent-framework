# AumOS Agent Framework

The **AumOS Agent Framework** is the multi-agent orchestration engine that powers every AI workflow
in the AumOS Enterprise platform. It provides graph-based workflow execution, durable processing,
human-in-the-loop gates, and a built-in tool marketplace.

## Key Features

- **Graph-based workflows** — define agent workflows as directed graphs via LangGraph, with
  AumOS-native type abstractions that hide library internals from your application code
- **Durable execution** — Temporal ensures at-least-once delivery and automatic retries for
  every workflow node, with full checkpoint-and-replay support
- **HITL gates** — configurable human-in-the-loop approval gates trigger automatically for
  high-privilege actions (level 3+) and pause workflows pending human review
- **Circuit breakers** — cascading failure containment via Redis-backed circuit breakers at
  both agent and workflow levels
- **SSE streaming** — real-time execution event streaming via Server-Sent Events for live
  workflow monitoring and visual builder overlays
- **Tool marketplace** — 20+ pre-built tools (web search, data APIs, email, Slack, PDF, code
  execution) plus a community integration layer
- **LangSmith-equivalent observability** — execution traces capture every LLM call, tool call,
  and HITL event with token counts and latency

## Quick Start

```python
import httpx

# Create a workflow
workflow = httpx.post("/workflows", json={
    "name": "Research Assistant",
    "graph_definition": {
        "entry_point": "search",
        "nodes": [
            {"node_id": "search", "metadata": {}},
            {"node_id": "summarize", "metadata": {}}
        ],
        "edges": [
            {"source_node_id": "search", "target_node_id": "summarize"}
        ],
        "state_schema": {}
    },
    "agents": [],
    "hitl_gates": [],
    "circuit_breaker_config": {}
})

# Execute with SSE streaming
with httpx.stream("POST", f"/workflows/{workflow.json()['id']}/execute/stream",
                  json={"input_data": {"query": "AumOS enterprise AI"}}) as stream:
    for line in stream.iter_lines():
        if line.startswith("data: "):
            print(line[6:])
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI REST API                       │
│  /agents  /workflows  /hitl  /tools  /executions        │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                 Core Services Layer                       │
│  AgentRegistryService  WorkflowService  HITLService      │
└───┬───────────────┬──────────────┬───────────────────────┘
    │               │              │
┌───▼───┐     ┌─────▼────┐  ┌─────▼─────────────┐
│LangGraph│   │ Temporal │  │   Tool Registry   │
│Engine  │   │ Executor │  │ (20+ built-in)    │
└───────┘     └──────────┘  └───────────────────┘
```

## Installation

```bash
pip install aumos-agent-framework
```

See [Installation](getting-started/installation.md) for full setup instructions including
database migrations and service dependencies.
