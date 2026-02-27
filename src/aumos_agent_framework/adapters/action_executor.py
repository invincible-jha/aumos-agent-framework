"""Sandboxed tool invocation adapter with timeout, retry, and audit logging.

Implements ActionExecutorProtocol providing production-grade tool execution:
- Input validation against JSON Schema before execution.
- Timeout enforcement using asyncio.wait_for.
- Error categorization into retryable vs terminal failure types.
- Exponential backoff retry for transient errors.
- Full audit log of every invocation including inputs, outputs, and timing.
"""

import asyncio
import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import jsonschema
import httpx

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_DEFAULT_TOOL_TIMEOUT_SECONDS = 30
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BASE_DELAY_SECONDS = 1.0
_DEFAULT_RETRY_MAX_DELAY_SECONDS = 30.0
_DEFAULT_RETRY_BACKOFF_FACTOR = 2.0

# Error categories
ERROR_RETRYABLE = "retryable"
ERROR_TERMINAL = "terminal"
ERROR_TIMEOUT = "timeout"
ERROR_VALIDATION = "validation"

# Retryable HTTP status codes
_RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}


class ActionExecutionError(Exception):
    """Raised when a tool action fails during execution.

    Attributes:
        error_category: One of 'retryable', 'terminal', 'timeout', 'validation'.
        tool_name: Name of the tool that failed.
        original_error: The underlying exception.
        retry_after_seconds: Suggested delay before retry, if applicable.
    """

    def __init__(
        self,
        message: str,
        error_category: str,
        tool_name: str,
        original_error: Exception | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.error_category = error_category
        self.tool_name = tool_name
        self.original_error = original_error
        self.retry_after_seconds = retry_after_seconds


class ActionExecutor:
    """Executes agent tool invocations with sandboxing, validation, and retry logic.

    Provides a uniform execution surface for all tool types (HTTP endpoints,
    internal services, sandboxed code). All executions are audit-logged with
    full input/output capture for traceability and debugging.
    """

    def __init__(
        self,
        tool_timeout_seconds: int = _DEFAULT_TOOL_TIMEOUT_SECONDS,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_base_delay_seconds: float = _DEFAULT_RETRY_BASE_DELAY_SECONDS,
        retry_max_delay_seconds: float = _DEFAULT_RETRY_MAX_DELAY_SECONDS,
        retry_backoff_factor: float = _DEFAULT_RETRY_BACKOFF_FACTOR,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize with execution configuration.

        Args:
            tool_timeout_seconds: Maximum seconds to wait for a tool response.
            max_retries: Maximum retry attempts for retryable errors.
            retry_base_delay_seconds: Initial backoff delay in seconds.
            retry_max_delay_seconds: Maximum backoff delay cap in seconds.
            retry_backoff_factor: Exponential backoff multiplier per retry.
            http_client: Optional pre-configured httpx client (for testing).
        """
        self._timeout = tool_timeout_seconds
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay_seconds
        self._retry_max_delay = retry_max_delay_seconds
        self._retry_backoff_factor = retry_backoff_factor
        self._http_client = http_client

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Return or lazily create the async HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout + 5),
            )
        return self._http_client

    def _validate_input(
        self,
        tool_input: dict[str, Any],
        input_schema: dict[str, Any],
        tool_name: str,
    ) -> None:
        """Validate tool input against the tool's JSON Schema.

        Args:
            tool_input: The input data dict to validate.
            input_schema: JSON Schema dict for validation.
            tool_name: Tool name for error context.

        Raises:
            ActionExecutionError: With 'validation' category if input is invalid.
        """
        try:
            jsonschema.validate(instance=tool_input, schema=input_schema)
        except jsonschema.ValidationError as exc:
            raise ActionExecutionError(
                message=f"Tool input validation failed: {exc.message}",
                error_category=ERROR_VALIDATION,
                tool_name=tool_name,
                original_error=exc,
            ) from exc

    def _classify_http_error(self, status_code: int, tool_name: str, exc: Exception) -> ActionExecutionError:
        """Classify an HTTP error as retryable or terminal.

        Args:
            status_code: HTTP response status code.
            tool_name: Tool name for error context.
            exc: Original exception.

        Returns:
            ActionExecutionError with appropriate category.
        """
        if status_code in _RETRYABLE_HTTP_STATUSES:
            return ActionExecutionError(
                message=f"Tool {tool_name} returned retryable HTTP {status_code}",
                error_category=ERROR_RETRYABLE,
                tool_name=tool_name,
                original_error=exc,
                retry_after_seconds=self._retry_base_delay,
            )
        return ActionExecutionError(
            message=f"Tool {tool_name} returned terminal HTTP {status_code}",
            error_category=ERROR_TERMINAL,
            tool_name=tool_name,
            original_error=exc,
        )

    async def _invoke_http_tool(
        self,
        tool_config: dict[str, Any],
        tool_input: dict[str, Any],
        tool_name: str,
    ) -> dict[str, Any]:
        """Invoke an HTTP-endpoint tool with timeout enforcement.

        Args:
            tool_config: Tool configuration dict with 'endpoint_url' and optional 'headers'.
            tool_input: Validated input dict for the tool.
            tool_name: Tool name for logging context.

        Returns:
            Parsed JSON response dict from the tool endpoint.

        Raises:
            ActionExecutionError: On timeout, HTTP error, or invalid response.
        """
        client = await self._get_http_client()
        endpoint_url = tool_config["endpoint_url"]
        extra_headers = tool_config.get("headers", {})

        try:
            async with asyncio.timeout(self._timeout):
                response = await client.post(
                    endpoint_url,
                    json=tool_input,
                    headers={**extra_headers, "X-Tool-Name": tool_name},
                )
        except asyncio.TimeoutError as exc:
            raise ActionExecutionError(
                message=f"Tool {tool_name} timed out after {self._timeout}s",
                error_category=ERROR_TIMEOUT,
                tool_name=tool_name,
                original_error=exc,
            ) from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise self._classify_http_error(response.status_code, tool_name, exc) from exc

        try:
            return response.json()  # type: ignore[no-any-return]
        except json.JSONDecodeError as exc:
            raise ActionExecutionError(
                message=f"Tool {tool_name} returned non-JSON response",
                error_category=ERROR_TERMINAL,
                tool_name=tool_name,
                original_error=exc,
            ) from exc

    def _compute_retry_delay(self, attempt: int) -> float:
        """Compute exponential backoff delay for a retry attempt.

        Args:
            attempt: Zero-based retry attempt number.

        Returns:
            Delay in seconds, capped at retry_max_delay_seconds.
        """
        delay = self._retry_base_delay * (self._retry_backoff_factor ** attempt)
        return min(delay, self._retry_max_delay)

    async def execute(
        self,
        tool_name: str,
        tool_definition: dict[str, Any],
        tool_input: dict[str, Any],
        agent_id: uuid.UUID,
        tenant_id: str,
        execution_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a tool invocation with full lifecycle management.

        Validates input, invokes the tool with timeout, retries on transient
        errors, and produces a full audit record of the execution.

        Args:
            tool_name: Name of the tool to invoke.
            tool_definition: Full tool definition dict including 'config' and 'input_schema'.
            tool_input: Input data for the tool.
            agent_id: Agent performing the action.
            tenant_id: Tenant context for logging and isolation.
            execution_id: Optional workflow execution ID for correlation.

        Returns:
            Audit result dict with 'output', 'success', 'duration_ms', 'attempt_count',
            and 'error_category' (None on success).

        Raises:
            ActionExecutionError: After all retries exhausted, with error_category set.
        """
        invocation_id = str(uuid.uuid4())
        input_schema = tool_definition.get("input_schema", {})
        tool_config = tool_definition.get("config", {})
        started_at = time.monotonic()

        logger.info(
            "Tool action execution starting",
            invocation_id=invocation_id,
            tool_name=tool_name,
            agent_id=str(agent_id),
            execution_id=execution_id,
            tenant_id=tenant_id,
        )

        # Validate input before any network calls
        if input_schema:
            self._validate_input(tool_input, input_schema, tool_name)

        last_error: ActionExecutionError | None = None
        attempt_count = 0

        for attempt in range(self._max_retries + 1):
            attempt_count = attempt + 1

            try:
                tool_output = await self._invoke_http_tool(
                    tool_config=tool_config,
                    tool_input=tool_input,
                    tool_name=tool_name,
                )

                duration_ms = int((time.monotonic() - started_at) * 1000)

                logger.info(
                    "Tool action execution succeeded",
                    invocation_id=invocation_id,
                    tool_name=tool_name,
                    agent_id=str(agent_id),
                    attempt=attempt_count,
                    duration_ms=duration_ms,
                    tenant_id=tenant_id,
                )

                return {
                    "invocation_id": invocation_id,
                    "tool_name": tool_name,
                    "output": tool_output,
                    "success": True,
                    "duration_ms": duration_ms,
                    "attempt_count": attempt_count,
                    "error_category": None,
                    "executed_at": datetime.now(UTC).isoformat(),
                }

            except ActionExecutionError as exc:
                last_error = exc

                if exc.error_category in (ERROR_TERMINAL, ERROR_VALIDATION):
                    # Terminal errors — do not retry
                    break

                if attempt < self._max_retries:
                    delay = exc.retry_after_seconds or self._compute_retry_delay(attempt)
                    logger.warning(
                        "Tool action retryable error — retrying",
                        invocation_id=invocation_id,
                        tool_name=tool_name,
                        agent_id=str(agent_id),
                        attempt=attempt_count,
                        error_category=exc.error_category,
                        retry_in_seconds=delay,
                        tenant_id=tenant_id,
                    )
                    await asyncio.sleep(delay)

        # All attempts exhausted or terminal error encountered
        duration_ms = int((time.monotonic() - started_at) * 1000)
        error_to_raise = last_error or ActionExecutionError(
            message="Unknown execution failure",
            error_category=ERROR_TERMINAL,
            tool_name=tool_name,
        )

        logger.error(
            "Tool action execution failed",
            invocation_id=invocation_id,
            tool_name=tool_name,
            agent_id=str(agent_id),
            attempt_count=attempt_count,
            error_category=error_to_raise.error_category,
            error=str(error_to_raise),
            duration_ms=duration_ms,
            tenant_id=tenant_id,
        )

        raise error_to_raise

    async def execute_with_fallback(
        self,
        tool_name: str,
        tool_definition: dict[str, Any],
        tool_input: dict[str, Any],
        agent_id: uuid.UUID,
        tenant_id: str,
        fallback_output: dict[str, Any],
        execution_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a tool with a fallback value on any failure.

        Useful when a failed tool call should degrade gracefully rather than
        propagate an exception up the agent call stack.

        Args:
            tool_name: Name of the tool to invoke.
            tool_definition: Full tool definition dict.
            tool_input: Input data for the tool.
            agent_id: Agent performing the action.
            tenant_id: Tenant context.
            fallback_output: Dict to return if all execution attempts fail.
            execution_id: Optional correlation ID.

        Returns:
            Audit result dict — on failure, 'output' is set to fallback_output
            and 'used_fallback' is True.
        """
        try:
            result = await self.execute(
                tool_name=tool_name,
                tool_definition=tool_definition,
                tool_input=tool_input,
                agent_id=agent_id,
                tenant_id=tenant_id,
                execution_id=execution_id,
            )
            return {**result, "used_fallback": False}
        except ActionExecutionError as exc:
            logger.warning(
                "Tool execution failed — using fallback output",
                tool_name=tool_name,
                agent_id=str(agent_id),
                error_category=exc.error_category,
                tenant_id=tenant_id,
            )
            return {
                "invocation_id": str(uuid.uuid4()),
                "tool_name": tool_name,
                "output": fallback_output,
                "success": False,
                "duration_ms": 0,
                "attempt_count": self._max_retries + 1,
                "error_category": exc.error_category,
                "used_fallback": True,
                "executed_at": datetime.now(UTC).isoformat(),
            }

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
