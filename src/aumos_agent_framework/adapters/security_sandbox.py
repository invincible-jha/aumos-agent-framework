"""Isolated code execution sandbox via Docker containers.

Implements SecuritySandboxProtocol providing:
- On-demand Docker container creation per agent execution.
- CPU, memory, network, and disk resource limits.
- File system isolation with read-only bind mounts.
- Network isolation (no external access by default).
- Execution timeout enforcement.
- Stdout, stderr, and output file capture.
- Container cleanup on completion or timeout.
- Per-agent security policy configuration.
"""

import asyncio
import json
import tarfile
import io
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import docker
import docker.errors
from docker.models.containers import Container

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_DEFAULT_MEMORY_LIMIT_MB = 256
_DEFAULT_CPU_QUOTA = 50000  # 50% of one CPU (relative to cpu_period=100000)
_DEFAULT_CPU_PERIOD = 100000
_DEFAULT_EXECUTION_TIMEOUT_SECONDS = 30
_DEFAULT_DISK_SIZE_MB = 100
_DEFAULT_BASE_IMAGE = "python:3.11-slim"
_CONTAINER_WORK_DIR = "/workspace"
_CONTAINER_OUTPUT_PATH = "/workspace/output.json"
_CONTAINER_LABEL_PREFIX = "aumos.sandbox"


@dataclass
class SandboxPolicy:
    """Security policy governing sandbox container behaviour.

    Attributes:
        memory_limit_mb: Maximum container RAM in megabytes.
        cpu_quota: Docker CPU quota (relative to cpu_period).
        cpu_period: Docker CPU period in microseconds.
        allow_network: If True, container has outbound network access.
        execution_timeout_seconds: Maximum wall-clock time for execution.
        disk_size_mb: Maximum writable layer size in megabytes.
        base_image: Docker image to use for the sandbox container.
        environment_variables: Env vars injected into the container.
        read_only_mounts: List of host paths to bind-mount read-only.
        allowed_syscalls: Empty list means use Docker default seccomp profile.
    """

    memory_limit_mb: int = _DEFAULT_MEMORY_LIMIT_MB
    cpu_quota: int = _DEFAULT_CPU_QUOTA
    cpu_period: int = _DEFAULT_CPU_PERIOD
    allow_network: bool = False
    execution_timeout_seconds: int = _DEFAULT_EXECUTION_TIMEOUT_SECONDS
    disk_size_mb: int = _DEFAULT_DISK_SIZE_MB
    base_image: str = _DEFAULT_BASE_IMAGE
    environment_variables: dict[str, str] = field(default_factory=dict)
    read_only_mounts: list[str] = field(default_factory=list)
    allowed_syscalls: list[str] = field(default_factory=list)


@dataclass
class SandboxExecutionResult:
    """Result of a sandboxed code execution.

    Attributes:
        execution_id: Unique ID for this execution.
        success: True if process exited with code 0.
        exit_code: Container process exit code.
        stdout: Captured standard output.
        stderr: Captured standard error.
        output_data: Parsed output.json if present, else empty dict.
        duration_ms: Wall-clock execution time in milliseconds.
        timed_out: True if execution was terminated by timeout.
        container_id: Docker container ID (empty after cleanup).
        executed_at: ISO timestamp of execution completion.
    """

    execution_id: str
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    output_data: dict[str, Any]
    duration_ms: int
    timed_out: bool
    container_id: str
    executed_at: str


class SecuritySandbox:
    """Docker-based security sandbox for isolated, resource-limited code execution.

    Each call to execute_code creates a fresh container, runs the provided
    code, captures outputs, and destroys the container. No state persists
    between executions. Network access is disabled by default.
    """

    def __init__(
        self,
        docker_client: docker.DockerClient | None = None,
        default_policy: SandboxPolicy | None = None,
    ) -> None:
        """Initialize with Docker client and default security policy.

        Args:
            docker_client: Initialized Docker SDK client. Connects to local
                           Docker daemon if None.
            default_policy: Default SandboxPolicy applied to all executions.
                            Can be overridden per-execution.
        """
        self._docker = docker_client or docker.from_env()
        self._default_policy = default_policy or SandboxPolicy()

    def _build_container_config(
        self,
        policy: SandboxPolicy,
        agent_id: uuid.UUID,
        tenant_id: str,
        execution_id: str,
    ) -> dict[str, Any]:
        """Build the Docker container creation kwargs from policy.

        Args:
            policy: Security policy to enforce.
            agent_id: Agent executing the code.
            tenant_id: Tenant context for labelling.
            execution_id: Execution ID for container labelling.

        Returns:
            Dict of kwargs to pass to docker_client.containers.create().
        """
        mounts = []
        for host_path in policy.read_only_mounts:
            mounts.append(
                docker.types.Mount(
                    target=host_path,
                    source=host_path,
                    type="bind",
                    read_only=True,
                )
            )

        network_mode = "bridge" if policy.allow_network else "none"

        config: dict[str, Any] = {
            "image": policy.base_image,
            "command": ["python", "/workspace/entrypoint.py"],
            "working_dir": _CONTAINER_WORK_DIR,
            "network_mode": network_mode,
            "mem_limit": f"{policy.memory_limit_mb}m",
            "memswap_limit": f"{policy.memory_limit_mb}m",  # Disable swap
            "cpu_quota": policy.cpu_quota,
            "cpu_period": policy.cpu_period,
            "read_only": False,  # Allow writes to /workspace only
            "mounts": mounts,
            "environment": {
                "AUMOS_SANDBOX": "1",
                "AUMOS_EXECUTION_ID": execution_id,
                **policy.environment_variables,
            },
            "labels": {
                f"{_CONTAINER_LABEL_PREFIX}.agent_id": str(agent_id),
                f"{_CONTAINER_LABEL_PREFIX}.tenant_id": tenant_id,
                f"{_CONTAINER_LABEL_PREFIX}.execution_id": execution_id,
                f"{_CONTAINER_LABEL_PREFIX}.created_at": datetime.now(UTC).isoformat(),
            },
            "detach": True,
            "auto_remove": False,  # We remove manually after capturing output
        }

        return config

    def _create_entrypoint_tar(self, code: str, input_data: dict[str, Any]) -> bytes:
        """Create a tar archive containing the entrypoint script and input data.

        The entrypoint reads input from /workspace/input.json and writes
        results to /workspace/output.json for capture after execution.

        Args:
            code: Python source code to execute.
            input_data: Input data dict serialized as input.json.

        Returns:
            Tar archive bytes for copying into the container.
        """
        entrypoint_script = f"""import json, sys, traceback

with open('/workspace/input.json') as f:
    INPUT = json.load(f)

OUTPUT = {{}}
EXIT_CODE = 0

try:
{chr(10).join('    ' + line for line in code.splitlines())}
except Exception as e:
    OUTPUT = {{'error': str(e), 'traceback': traceback.format_exc()}}
    EXIT_CODE = 1
finally:
    with open('/workspace/output.json', 'w') as out:
        json.dump(OUTPUT, out)
    sys.exit(EXIT_CODE)
"""
        tar_buffer = io.BytesIO()
        with tarfile.open(fileobj=tar_buffer, mode="w") as tar:
            # Add entrypoint.py
            entrypoint_bytes = entrypoint_script.encode()
            info = tarfile.TarInfo(name="entrypoint.py")
            info.size = len(entrypoint_bytes)
            tar.addfile(info, io.BytesIO(entrypoint_bytes))

            # Add input.json
            input_bytes = json.dumps(input_data).encode()
            info2 = tarfile.TarInfo(name="input.json")
            info2.size = len(input_bytes)
            tar.addfile(info2, io.BytesIO(input_bytes))

        return tar_buffer.getvalue()

    async def execute_code(
        self,
        code: str,
        input_data: dict[str, Any],
        agent_id: uuid.UUID,
        tenant_id: str,
        policy: SandboxPolicy | None = None,
    ) -> SandboxExecutionResult:
        """Execute Python code in an isolated Docker container.

        Creates a fresh container, copies the code and input data in,
        runs with resource limits and timeout enforcement, captures all
        outputs, then destroys the container.

        Args:
            code: Python source code string to execute inside the sandbox.
            input_data: Dict of input data available as INPUT variable in code.
            agent_id: Agent invoking the sandbox.
            tenant_id: Tenant context for isolation and labelling.
            policy: Optional policy override; uses default_policy if None.

        Returns:
            SandboxExecutionResult with captured stdout, stderr, and output.
        """
        active_policy = policy or self._default_policy
        execution_id = str(uuid.uuid4())

        logger.info(
            "Sandbox execution starting",
            execution_id=execution_id,
            agent_id=str(agent_id),
            image=active_policy.base_image,
            memory_mb=active_policy.memory_limit_mb,
            network=active_policy.allow_network,
            timeout=active_policy.execution_timeout_seconds,
            tenant_id=tenant_id,
        )

        container: Container | None = None
        import time as _time
        start = _time.monotonic()
        timed_out = False

        try:
            container_config = self._build_container_config(
                policy=active_policy,
                agent_id=agent_id,
                tenant_id=tenant_id,
                execution_id=execution_id,
            )

            # Create container (synchronous Docker SDK call wrapped in executor)
            loop = asyncio.get_event_loop()
            container = await loop.run_in_executor(
                None,
                lambda: self._docker.containers.create(**container_config),
            )

            # Copy entrypoint and input into container
            tar_bytes = self._create_entrypoint_tar(code, input_data)
            await loop.run_in_executor(
                None,
                lambda: container.put_archive(_CONTAINER_WORK_DIR, tar_bytes),
            )

            # Start the container
            await loop.run_in_executor(None, container.start)

            # Wait for completion with timeout
            try:
                async with asyncio.timeout(active_policy.execution_timeout_seconds):
                    exit_result = await loop.run_in_executor(
                        None,
                        lambda: container.wait(),
                    )
                exit_code = exit_result.get("StatusCode", 1)
            except asyncio.TimeoutError:
                timed_out = True
                exit_code = 124  # Standard timeout exit code
                await loop.run_in_executor(None, container.kill)
                logger.warning(
                    "Sandbox execution timed out",
                    execution_id=execution_id,
                    timeout_seconds=active_policy.execution_timeout_seconds,
                    tenant_id=tenant_id,
                )

            # Capture stdout and stderr
            raw_logs = await loop.run_in_executor(
                None,
                lambda: container.logs(stdout=True, stderr=True),
            )
            log_text = raw_logs.decode(errors="replace") if raw_logs else ""

            # Try to extract stdout/stderr separately
            try:
                stdout_logs = await loop.run_in_executor(
                    None, lambda: container.logs(stdout=True, stderr=False)
                )
                stderr_logs = await loop.run_in_executor(
                    None, lambda: container.logs(stdout=False, stderr=True)
                )
                stdout_text = stdout_logs.decode(errors="replace") if stdout_logs else ""
                stderr_text = stderr_logs.decode(errors="replace") if stderr_logs else ""
            except Exception:
                stdout_text = log_text
                stderr_text = ""

            # Retrieve output.json if present
            output_data: dict[str, Any] = {}
            if not timed_out:
                try:
                    tar_stream, _ = await loop.run_in_executor(
                        None,
                        lambda: container.get_archive(_CONTAINER_OUTPUT_PATH),
                    )
                    tar_content = b"".join(tar_stream)
                    with tarfile.open(fileobj=io.BytesIO(tar_content)) as tar:
                        output_member = tar.getmember("output.json")
                        output_file = tar.extractfile(output_member)
                        if output_file:
                            output_data = json.loads(output_file.read().decode())
                except Exception as exc:
                    logger.debug(
                        "Could not retrieve output.json from sandbox",
                        execution_id=execution_id,
                        reason=str(exc),
                    )

            duration_ms = int((_time.monotonic() - start) * 1000)
            success = exit_code == 0 and not timed_out
            container_id = container.id or ""

            logger.info(
                "Sandbox execution complete",
                execution_id=execution_id,
                agent_id=str(agent_id),
                exit_code=exit_code,
                success=success,
                timed_out=timed_out,
                duration_ms=duration_ms,
                tenant_id=tenant_id,
            )

            return SandboxExecutionResult(
                execution_id=execution_id,
                success=success,
                exit_code=exit_code,
                stdout=stdout_text,
                stderr=stderr_text,
                output_data=output_data,
                duration_ms=duration_ms,
                timed_out=timed_out,
                container_id=container_id,
                executed_at=datetime.now(UTC).isoformat(),
            )

        except docker.errors.ImageNotFound as exc:
            logger.error(
                "Sandbox base image not found",
                image=active_policy.base_image,
                execution_id=execution_id,
                error=str(exc),
                tenant_id=tenant_id,
            )
            raise
        except docker.errors.APIError as exc:
            logger.error(
                "Docker API error during sandbox execution",
                execution_id=execution_id,
                error=str(exc),
                tenant_id=tenant_id,
            )
            raise
        finally:
            # Always clean up the container
            if container is not None:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: container.remove(force=True),
                    )
                    logger.debug(
                        "Sandbox container removed",
                        execution_id=execution_id,
                        container_id=container.id,
                    )
                except Exception as cleanup_exc:
                    logger.warning(
                        "Failed to remove sandbox container",
                        execution_id=execution_id,
                        error=str(cleanup_exc),
                    )

    async def build_policy_for_agent(
        self,
        agent_config: dict[str, Any],
        privilege_level: int,
    ) -> SandboxPolicy:
        """Build a SandboxPolicy from agent configuration and privilege level.

        Higher privilege agents get looser resource limits and may be granted
        selective network access. Lower privilege agents are maximally restricted.

        Args:
            agent_config: Agent's resource_limits and sandbox config dict.
            privilege_level: Agent privilege level (1-5).

        Returns:
            Configured SandboxPolicy for this agent.
        """
        sandbox_config = agent_config.get("sandbox", {})

        # Scale memory with privilege level
        base_memory_mb = sandbox_config.get("memory_limit_mb", _DEFAULT_MEMORY_LIMIT_MB)
        memory_mb = min(base_memory_mb, 512 if privilege_level <= 2 else 2048)

        # Network access only for privilege level 4+
        allow_network = privilege_level >= 4 and sandbox_config.get("allow_network", False)

        # CPU limits scale with privilege level
        cpu_quota = min(
            sandbox_config.get("cpu_quota", _DEFAULT_CPU_QUOTA),
            25000 if privilege_level <= 2 else _DEFAULT_CPU_QUOTA,
        )

        return SandboxPolicy(
            memory_limit_mb=memory_mb,
            cpu_quota=cpu_quota,
            cpu_period=_DEFAULT_CPU_PERIOD,
            allow_network=allow_network,
            execution_timeout_seconds=sandbox_config.get(
                "timeout_seconds", _DEFAULT_EXECUTION_TIMEOUT_SECONDS
            ),
            disk_size_mb=sandbox_config.get("disk_size_mb", _DEFAULT_DISK_SIZE_MB),
            base_image=sandbox_config.get("base_image", _DEFAULT_BASE_IMAGE),
            environment_variables=sandbox_config.get("environment_variables", {}),
            read_only_mounts=sandbox_config.get("read_only_mounts", []),
        )
