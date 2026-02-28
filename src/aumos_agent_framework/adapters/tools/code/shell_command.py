"""Shell command execution tool.

Privilege level 4 — ADMIN required; arbitrary shell execution.
"""

from typing import Any

from pydantic import Field

from aumos_agent_framework.core.interfaces import ToolInputSchema, ToolOutputSchema
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class ShellCommandInput(ToolInputSchema):
    """Input schema for the shell command tool."""

    command: str = Field(..., max_length=2000, description="Shell command to execute")
    working_directory: str | None = Field(default=None, description="Working directory for the command")
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0, description="Maximum execution time")
    environment: dict[str, str] | None = Field(
        default=None,
        description="Additional environment variables (merged with a minimal safe env)",
    )


class ShellCommandOutput(ToolOutputSchema):
    """Output schema for the shell command tool."""

    result: dict[str, Any] = Field(description="Command result with stdout, stderr, and return_code")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ShellCommandTool:
    """Execute a shell command in a subprocess.

    Requires privilege level 4 (ADMIN). All executions are logged with full
    command text and tenant context for audit purposes. Commands are run with
    a minimal environment to reduce injection surface.
    """

    tool_id: str = "shell_command"
    display_name: str = "Shell Command"
    category: str = "code"
    description: str = (
        "Execute a shell command in a subprocess. Requires ADMIN privilege (level 4). "
        "All commands are audit-logged."
    )
    privilege_level: int = 4
    input_schema: type[ShellCommandInput] = ShellCommandInput
    output_schema: type[ShellCommandOutput] = ShellCommandOutput

    async def execute(
        self,
        input_data: ShellCommandInput,
        tenant_id: str,
        config: dict[str, str],
    ) -> ShellCommandOutput:
        """Execute a shell command.

        Args:
            input_data: Command string, working directory, timeout, and env vars.
            tenant_id: Tenant context for audit logging.
            config: Unused for this tool.

        Returns:
            ShellCommandOutput with stdout, stderr, and return code.
        """
        import asyncio
        import os

        base_env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": os.environ.get("HOME", "/tmp"),
        }
        if input_data.environment:
            base_env.update(input_data.environment)

        logger.warning(
            "Shell command executing",
            command=input_data.command,
            tenant_id=tenant_id,
        )

        try:
            proc = await asyncio.create_subprocess_shell(
                input_data.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=input_data.working_directory,
                env=base_env,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=input_data.timeout_seconds,
            )
            return_code = proc.returncode or 0
            stdout = stdout_bytes.decode(errors="replace")
            stderr = stderr_bytes.decode(errors="replace")
        except asyncio.TimeoutError:
            return ShellCommandOutput(
                result={"stdout": "", "stderr": f"Timed out after {input_data.timeout_seconds}s", "return_code": -1},
                metadata={"command": input_data.command, "timed_out": True},
            )

        logger.info(
            "Shell command completed",
            return_code=return_code,
            stdout_length=len(stdout),
            tenant_id=tenant_id,
        )

        return ShellCommandOutput(
            result={"stdout": stdout, "stderr": stderr, "return_code": return_code},
            metadata={"command": input_data.command, "return_code": return_code},
        )


TOOL = ShellCommandTool()
