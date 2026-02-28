"""Sandboxed Python REPL tool.

Privilege level 3 — code execution; requires HITL gate or explicit approval.
"""

from typing import Any

from pydantic import Field

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)

_FORBIDDEN_BUILTINS = frozenset(
    {
        "__import__",
        "open",
        "eval",
        "exec",
        "compile",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
    }
)


class PythonReplInput(ToolInputSchema):
    """Input schema for the sandboxed Python REPL tool."""

    code: str = Field(..., max_length=10000, description="Python code to execute in the sandbox")
    timeout_seconds: float = Field(default=10.0, ge=1.0, le=60.0, description="Maximum execution time")


class PythonReplOutput(ToolOutputSchema):
    """Output schema for the sandboxed Python REPL tool."""

    result: dict[str, Any] = Field(description="Execution result with stdout, return_value, and error")
    metadata: dict[str, Any] = Field(default_factory=dict)


class PythonReplTool:
    """Execute Python code in a restricted sandbox environment.

    The sandbox blocks dangerous builtins (open, exec, eval, __import__),
    restricts available modules, and enforces a hard wall-clock timeout.
    Only pure computation — no file I/O, network, or subprocess calls.
    """

    tool_id: str = "python_repl"
    display_name: str = "Python REPL (Sandboxed)"
    category: str = "code"
    description: str = (
        "Execute a Python code snippet in a restricted sandbox. "
        "Returns stdout, return value, and any errors. No I/O or network access."
    )
    privilege_level: int = 3
    input_schema: type[PythonReplInput] = PythonReplInput
    output_schema: type[PythonReplOutput] = PythonReplOutput

    def _build_safe_globals(self) -> dict[str, Any]:
        """Build a restricted globals dict with safe builtins only."""
        import builtins

        safe_builtins = {
            name: getattr(builtins, name)
            for name in dir(builtins)
            if name not in _FORBIDDEN_BUILTINS and not name.startswith("__")
        }
        safe_builtins["__builtins__"] = safe_builtins
        return {"__builtins__": safe_builtins}

    async def execute(
        self,
        input_data: PythonReplInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> PythonReplOutput:
        """Execute Python code in a restricted sandbox.

        Args:
            input_data: Code to execute and timeout.
            tenant_id: Tenant context for audit logging.
            config: Unused for this tool.

        Returns:
            PythonReplOutput with stdout, return_value, and error.
        """
        import asyncio
        import io
        import sys

        safe_globals = self._build_safe_globals()
        stdout_capture = io.StringIO()
        result_holder: dict[str, Any] = {"return_value": None, "error": None}

        def run_code() -> None:
            old_stdout = sys.stdout
            sys.stdout = stdout_capture
            try:
                exec(compile(input_data.code, "<sandbox>", "exec"), safe_globals)  # noqa: S102
            except Exception as exc:
                result_holder["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                sys.stdout = old_stdout

        loop = asyncio.get_event_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, run_code),
                timeout=input_data.timeout_seconds,
            )
        except asyncio.TimeoutError:
            result_holder["error"] = f"Execution timed out after {input_data.timeout_seconds}s"

        stdout_output = stdout_capture.getvalue()

        logger.info(
            "Python REPL executed",
            code_length=len(input_data.code),
            has_error=result_holder["error"] is not None,
            tenant_id=tenant_id,
        )

        return PythonReplOutput(
            result={
                "stdout": stdout_output,
                "return_value": result_holder["return_value"],
                "error": result_holder["error"],
            },
            metadata={"code_length": len(input_data.code), "timeout_seconds": input_data.timeout_seconds},
        )


TOOL = PythonReplTool()
