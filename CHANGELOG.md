# Changelog

All notable changes to aumos-agent-framework are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-26

### Added
- Initial scaffolding for aumos-agent-framework
- LangGraph graph-based workflow engine adapter
- Temporal durable execution integration
- 5-level agent privilege system (READ_ONLY through SUPER_ADMIN)
- Human-in-the-loop (HITL) approval gates
- Per-agent and per-workflow circuit breakers with cascading failure containment
- Tenant-aware agent session isolation with Redis-backed state
- Kafka Protobuf envelope agent communication
- Agent registry with capability and tool access management
- Workflow definition, execution, and cancellation API
- Direct agent invocation endpoint
- Full hexagonal architecture (api/ + core/ + adapters/)
- SQLAlchemy ORM models with tenant isolation (agf_ table prefix)
- Protocol interfaces for all major components
- CI/CD pipeline with GitHub Actions
- Docker multi-stage build
- Temporal service in docker-compose.dev.yml
