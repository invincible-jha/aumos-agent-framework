"""AumOS Agent Framework service entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from aumos_common.app import create_app
from aumos_common.database import init_database
from aumos_common.health import HealthCheck
from aumos_common.observability import get_logger

from aumos_agent_framework.settings import Settings

logger = get_logger(__name__)
settings = Settings()


@asynccontextmanager
async def lifespan(app: object) -> AsyncGenerator[None, None]:
    """Application lifespan — startup and shutdown."""
    logger.info("Starting aumos-agent-framework", version="0.1.0")

    # Initialize database
    init_database(settings.database)
    logger.info("Database initialized")

    # TODO: Initialize Kafka publisher
    # TODO: Initialize Redis client for session isolation and circuit breaker
    # TODO: Initialize Temporal client
    logger.info("aumos-agent-framework started successfully")

    yield

    # Shutdown
    logger.info("Shutting down aumos-agent-framework")
    # TODO: Close Kafka producer
    # TODO: Close Redis connection
    # TODO: Close Temporal client


app = create_app(
    service_name="aumos-agent-framework",
    version="0.1.0",
    settings=settings,
    lifespan=lifespan,
    health_checks=[
        # HealthCheck(name="postgres", check_fn=check_db),
        # HealthCheck(name="redis", check_fn=check_redis),
        # HealthCheck(name="temporal", check_fn=check_temporal),
    ],
)

from aumos_agent_framework.api.router import router  # noqa: E402

app.include_router(router, prefix="/api/v1")
