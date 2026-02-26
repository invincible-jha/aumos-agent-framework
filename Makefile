# ============================================================================
# Makefile — aumos-agent-framework
# ============================================================================

.PHONY: help install dev-install lint format typecheck test test-cov \
        docker-build docker-up docker-down migrate clean

PYTHON := python
PIP := pip
SERVICE_NAME := aumos-agent-framework
IMAGE_NAME := aumos/agent-framework
IMAGE_TAG := latest

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	$(PIP) install -e .

dev-install: ## Install development dependencies
	$(PIP) install -e ".[dev]"

lint: ## Run ruff linter
	ruff check src/ tests/

format: ## Run ruff formatter
	ruff format src/ tests/

format-check: ## Check formatting without modifying files
	ruff format --check src/ tests/

typecheck: ## Run mypy type checker
	mypy src/

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=aumos_agent_framework --cov-report=html --cov-report=term-missing

test-unit: ## Run unit tests only
	pytest tests/ -v -m "not integration"

test-integration: ## Run integration tests only
	pytest tests/ -v -m integration

docker-build: ## Build Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-up: ## Start all services (dev)
	docker-compose -f docker-compose.dev.yml up -d

docker-down: ## Stop all services
	docker-compose -f docker-compose.dev.yml down

docker-logs: ## Follow service logs
	docker-compose -f docker-compose.dev.yml logs -f $(SERVICE_NAME)

migrate: ## Run database migrations
	alembic upgrade head

migrate-create: ## Create new migration (use: make migrate-create MSG="description")
	alembic revision --autogenerate -m "$(MSG)"

migrate-rollback: ## Rollback last migration
	alembic downgrade -1

clean: ## Remove build artifacts
	rm -rf dist/ build/ .eggs/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/

check-all: lint format-check typecheck test ## Run all quality checks
