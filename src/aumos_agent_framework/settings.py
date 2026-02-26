"""Service-specific settings extending AumOS base config."""

from pydantic_settings import SettingsConfigDict

from aumos_common.config import AumOSSettings


class Settings(AumOSSettings):
    """Agent framework service settings."""

    service_name: str = "aumos-agent-framework"

    # Temporal durable execution
    temporal_server_url: str = "localhost:7233"
    temporal_namespace: str = "aumos-agent-framework"
    temporal_task_queue: str = "aumos-agent-tasks"

    # LLM Serving
    llm_serving_url: str = "http://localhost:8001"

    # Workflow limits
    max_concurrent_workflows: int = 100
    workflow_timeout_seconds: int = 3600

    # Circuit breaker configuration
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_reset_timeout_seconds: int = 60
    circuit_breaker_half_open_max_calls: int = 3

    # HITL gate configuration
    hitl_approval_timeout_hours: int = 24
    hitl_auto_reject_on_timeout: bool = False

    # Session configuration
    session_ttl_seconds: int = 86400
    agent_max_privilege_level: int = 5

    model_config = SettingsConfigDict(env_prefix="AUMOS_AGF_")
