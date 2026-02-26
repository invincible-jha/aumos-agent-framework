# Contributing to aumos-agent-framework

Thank you for your interest in contributing to the AumOS Agent Framework.

## Development Setup

```bash
git clone <repo-url>
cd aumos-agent-framework
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
make dev-install
cp .env.example .env
```

## Code Standards

- Python 3.11+, type hints on every function
- `ruff` for linting and formatting (`make lint`, `make format`)
- `mypy` strict mode (`make typecheck`)
- 80% test coverage minimum (`make test-cov`)

## Branch Naming

- `feature/short-description`
- `fix/short-description`
- `docs/short-description`

## Commit Messages

Follow conventional commits:
- `feat:` — new feature
- `fix:` — bug fix
- `refactor:` — code change without feature/fix
- `test:` — adding tests
- `docs:` — documentation only
- `chore:` — maintenance

## Pull Requests

1. Branch from `main`
2. Run `make check-all` before pushing
3. Write tests for new functionality
4. Update CHANGELOG.md under `[Unreleased]`

## Architecture

This repo follows hexagonal architecture:
- `api/` — FastAPI routes (thin, delegates to services)
- `core/` — Business logic (no framework dependencies)
- `adapters/` — External integrations (LangGraph, Temporal, Redis, Kafka)

Never put business logic in routes. Never import framework dependencies into `core/`.
