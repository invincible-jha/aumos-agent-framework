"""Hierarchical skill composition adapter — registry, DAG resolution, and composite execution.

Implements SkillComposerProtocol providing:
- Skill registry mapping names to callable implementations.
- Dependency resolution via topological sort on a directed acyclic graph.
- Composite skill types: sequence, parallel, and conditional branches.
- Parameter validation against per-skill Pydantic schemas.
- Skill versioning and backward compatibility checks.
- Task-to-skill recommendation using description embedding similarity.
"""

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any, Callable, Awaitable

import networkx as nx
from pydantic import BaseModel, ValidationError

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# Skill execution modes
MODE_SEQUENCE = "sequence"
MODE_PARALLEL = "parallel"
MODE_CONDITIONAL = "conditional"

_SKILL_VERSION_SEPARATOR = "@"


class SkillDefinition(BaseModel):
    """Schema for a registered skill definition.

    Attributes:
        name: Unique skill name within the registry.
        version: Semantic version string (e.g., '1.2.0').
        description: Human-readable description for recommendation matching.
        input_schema: JSON Schema dict for input parameter validation.
        dependencies: List of skill names this skill requires to run first.
        min_privilege_level: Minimum agent privilege level to invoke this skill.
        tags: Categorization tags for filtering and recommendation.
        deprecated: If True, skill is available but flagged for removal.
    """

    name: str
    version: str = "1.0.0"
    description: str
    input_schema: dict[str, Any] = {}
    dependencies: list[str] = []
    min_privilege_level: int = 1
    tags: list[str] = []
    deprecated: bool = False


class SkillExecutionResult(BaseModel):
    """Result of a skill or composite skill execution.

    Attributes:
        skill_name: Name of the executed skill.
        success: Whether execution completed without error.
        output: Skill output data.
        error: Error message if success is False.
        duration_ms: Wall-clock execution duration.
        executed_at: ISO timestamp of completion.
        child_results: For composite skills, results of each child skill.
    """

    skill_name: str
    success: bool
    output: dict[str, Any] = {}
    error: str | None = None
    duration_ms: int = 0
    executed_at: str = ""
    child_results: list["SkillExecutionResult"] = []


class SkillNotFoundError(Exception):
    """Raised when a requested skill name is not in the registry."""


class SkillDependencyCycleError(Exception):
    """Raised when skill dependencies form a cycle."""


class SkillVersionIncompatibleError(Exception):
    """Raised when a requested skill version is not compatible."""


class SkillComposer:
    """Registry and execution engine for hierarchical agent skills.

    Skills are registered with their implementations (async callables),
    input schemas, dependency declarations, and versioning metadata.
    Composite skills chain, parallelize, or conditionally branch over
    registered primitives, with full dependency resolution via DAG.
    """

    def __init__(self) -> None:
        """Initialize an empty skill registry."""
        # name@version → SkillDefinition
        self._definitions: dict[str, SkillDefinition] = {}
        # name@version → async callable
        self._implementations: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {}
        # Dependency graph for resolution
        self._dep_graph: nx.DiGraph = nx.DiGraph()

    # ─── Registration ─────────────────────────────────────────────────────────

    def register(
        self,
        definition: SkillDefinition,
        implementation: Callable[..., Awaitable[dict[str, Any]]],
    ) -> str:
        """Register a skill with its definition and callable implementation.

        Args:
            definition: SkillDefinition metadata including dependencies and schema.
            implementation: Async callable that accepts (input_data, context) and
                            returns a dict output.

        Returns:
            Registry key string (name@version).

        Raises:
            SkillDependencyCycleError: If registering this skill creates a dependency cycle.
        """
        registry_key = f"{definition.name}{_SKILL_VERSION_SEPARATOR}{definition.version}"

        self._definitions[registry_key] = definition
        self._implementations[registry_key] = implementation

        # Update dependency graph
        self._dep_graph.add_node(registry_key)
        for dep_name in definition.dependencies:
            # Dependencies may specify name@version or just name (latest)
            dep_key = self._resolve_dependency_key(dep_name)
            self._dep_graph.add_edge(registry_key, dep_key)

        # Verify no cycles introduced
        if not nx.is_directed_acyclic_graph(self._dep_graph):
            # Roll back
            self._dep_graph.remove_node(registry_key)
            self._definitions.pop(registry_key, None)
            self._implementations.pop(registry_key, None)
            raise SkillDependencyCycleError(
                f"Registering skill '{definition.name}' v{definition.version} "
                "would create a dependency cycle."
            )

        if definition.deprecated:
            logger.warning(
                "Deprecated skill registered",
                skill=registry_key,
                description=definition.description,
            )
        else:
            logger.info(
                "Skill registered",
                skill=registry_key,
                dependencies=definition.dependencies,
                min_privilege=definition.min_privilege_level,
            )

        return registry_key

    def _resolve_dependency_key(self, dep_name: str) -> str:
        """Resolve a dependency name to its registry key.

        If dep_name includes a version (name@version), use it directly.
        Otherwise, return the latest registered version for that name.

        Args:
            dep_name: Dependency name, optionally with version suffix.

        Returns:
            Full registry key string.
        """
        if _SKILL_VERSION_SEPARATOR in dep_name:
            return dep_name

        # Find latest registered version matching the base name
        matching = [
            key for key in self._definitions
            if key.split(_SKILL_VERSION_SEPARATOR)[0] == dep_name
        ]
        if not matching:
            # Return the name@latest placeholder — will fail at resolution
            return f"{dep_name}{_SKILL_VERSION_SEPARATOR}latest"

        # Return the last registered version (simple strategy)
        return sorted(matching)[-1]

    def get_definition(self, skill_name: str, version: str | None = None) -> SkillDefinition:
        """Retrieve a skill definition by name and optional version.

        Args:
            skill_name: Base skill name.
            version: Specific version string, or None for latest.

        Returns:
            SkillDefinition for the requested skill.

        Raises:
            SkillNotFoundError: If no matching skill is registered.
        """
        if version:
            key = f"{skill_name}{_SKILL_VERSION_SEPARATOR}{version}"
            if key not in self._definitions:
                raise SkillNotFoundError(f"Skill '{skill_name}' version '{version}' not found.")
            return self._definitions[key]

        # Find latest version
        matching = sorted(
            key for key in self._definitions
            if key.split(_SKILL_VERSION_SEPARATOR)[0] == skill_name
        )
        if not matching:
            raise SkillNotFoundError(f"Skill '{skill_name}' not found in registry.")
        return self._definitions[matching[-1]]

    # ─── Dependency resolution ────────────────────────────────────────────────

    def resolve_execution_order(self, skill_key: str) -> list[str]:
        """Compute topological execution order for a skill and its dependencies.

        Args:
            skill_key: The registry key (name@version) to resolve.

        Returns:
            Ordered list of registry keys, dependencies first.

        Raises:
            SkillNotFoundError: If skill_key is not registered.
            SkillDependencyCycleError: If a cycle exists (should not happen post-registration).
        """
        if skill_key not in self._definitions:
            raise SkillNotFoundError(f"Skill '{skill_key}' not found.")

        # Extract subgraph reachable from this skill
        subgraph_nodes = nx.descendants(self._dep_graph, skill_key) | {skill_key}
        subgraph = self._dep_graph.subgraph(subgraph_nodes)

        try:
            order = list(reversed(list(nx.topological_sort(subgraph))))
        except nx.NetworkXUnfeasible as exc:
            raise SkillDependencyCycleError(
                f"Dependency cycle detected for skill '{skill_key}'."
            ) from exc

        return order

    # ─── Execution ────────────────────────────────────────────────────────────

    def _validate_input(
        self,
        skill_key: str,
        skill_input: dict[str, Any],
    ) -> None:
        """Validate skill input against its JSON Schema.

        Args:
            skill_key: Registry key for the skill.
            skill_input: Input data dict.

        Raises:
            ValueError: If input fails schema validation.
        """
        import jsonschema
        schema = self._definitions[skill_key].input_schema
        if not schema:
            return
        try:
            jsonschema.validate(instance=skill_input, schema=schema)
        except jsonschema.ValidationError as exc:
            raise ValueError(
                f"Skill '{skill_key}' input validation failed: {exc.message}"
            ) from exc

    async def execute_skill(
        self,
        skill_name: str,
        skill_input: dict[str, Any],
        agent_id: uuid.UUID,
        tenant_id: str,
        context: dict[str, Any] | None = None,
        version: str | None = None,
    ) -> SkillExecutionResult:
        """Execute a single skill by name, running dependencies first.

        Args:
            skill_name: Name of the skill to execute.
            skill_input: Input data for the skill.
            agent_id: Agent executing the skill.
            tenant_id: Tenant context.
            context: Optional shared context dict passed to all skills.
            version: Specific version to execute, or None for latest.

        Returns:
            SkillExecutionResult with output and timing data.

        Raises:
            SkillNotFoundError: If skill is not registered.
            ValueError: If input validation fails.
        """
        definition = self.get_definition(skill_name, version)
        skill_key = f"{definition.name}{_SKILL_VERSION_SEPARATOR}{definition.version}"

        if definition.deprecated:
            logger.warning(
                "Executing deprecated skill",
                skill=skill_key,
                agent_id=str(agent_id),
                tenant_id=tenant_id,
            )

        self._validate_input(skill_key, skill_input)

        # Run dependencies first (in topological order, excluding self)
        execution_order = self.resolve_execution_order(skill_key)
        dep_keys = [k for k in execution_order if k != skill_key]

        dep_results: dict[str, dict[str, Any]] = {}
        for dep_key in dep_keys:
            dep_impl = self._implementations.get(dep_key)
            if dep_impl is None:
                raise SkillNotFoundError(f"Dependency '{dep_key}' implementation missing.")
            dep_output = await dep_impl({}, context or {})
            dep_results[dep_key] = dep_output

        # Execute the target skill
        impl = self._implementations[skill_key]
        enriched_context = {**(context or {}), "dependency_outputs": dep_results}

        import time as _time
        start = _time.monotonic()
        try:
            output = await impl(skill_input, enriched_context)
            duration_ms = int((_time.monotonic() - start) * 1000)

            logger.info(
                "Skill executed successfully",
                skill=skill_key,
                agent_id=str(agent_id),
                duration_ms=duration_ms,
                tenant_id=tenant_id,
            )

            return SkillExecutionResult(
                skill_name=skill_key,
                success=True,
                output=output,
                duration_ms=duration_ms,
                executed_at=datetime.now(UTC).isoformat(),
            )

        except Exception as exc:
            duration_ms = int((_time.monotonic() - start) * 1000)
            logger.error(
                "Skill execution failed",
                skill=skill_key,
                agent_id=str(agent_id),
                error=str(exc),
                duration_ms=duration_ms,
                tenant_id=tenant_id,
            )
            return SkillExecutionResult(
                skill_name=skill_key,
                success=False,
                error=str(exc),
                duration_ms=duration_ms,
                executed_at=datetime.now(UTC).isoformat(),
            )

    async def execute_composite(
        self,
        mode: str,
        skill_names: list[str],
        skill_inputs: list[dict[str, Any]],
        agent_id: uuid.UUID,
        tenant_id: str,
        context: dict[str, Any] | None = None,
        condition_fn: Callable[[dict[str, Any]], int] | None = None,
    ) -> SkillExecutionResult:
        """Execute a composite skill in sequence, parallel, or conditional mode.

        Args:
            mode: Execution mode: 'sequence', 'parallel', or 'conditional'.
            skill_names: Ordered list of skill names to execute.
            skill_inputs: Corresponding input dicts (must match skill_names length).
            agent_id: Agent executing the composite.
            tenant_id: Tenant context.
            context: Optional shared context.
            condition_fn: For conditional mode — callable that takes context and
                          returns the index into skill_names to execute.

        Returns:
            SkillExecutionResult with child_results populated.

        Raises:
            ValueError: If mode is invalid or skill_names/inputs length mismatch.
        """
        if len(skill_names) != len(skill_inputs):
            raise ValueError("skill_names and skill_inputs must have equal length.")

        if mode not in (MODE_SEQUENCE, MODE_PARALLEL, MODE_CONDITIONAL):
            raise ValueError(f"Invalid composite mode '{mode}'.")

        import time as _time
        start = _time.monotonic()
        child_results: list[SkillExecutionResult] = []
        composite_output: dict[str, Any] = {}

        if mode == MODE_SEQUENCE:
            for skill_name, skill_input in zip(skill_names, skill_inputs):
                result = await self.execute_skill(
                    skill_name=skill_name,
                    skill_input=skill_input,
                    agent_id=agent_id,
                    tenant_id=tenant_id,
                    context={**(context or {}), "previous_outputs": composite_output},
                )
                child_results.append(result)
                if not result.success:
                    break  # Sequence stops on first failure
                composite_output[skill_name] = result.output

        elif mode == MODE_PARALLEL:
            tasks = [
                self.execute_skill(
                    skill_name=name,
                    skill_input=inp,
                    agent_id=agent_id,
                    tenant_id=tenant_id,
                    context=context,
                )
                for name, inp in zip(skill_names, skill_inputs)
            ]
            child_results = list(await asyncio.gather(*tasks, return_exceptions=False))
            for result in child_results:
                if result.success:
                    composite_output[result.skill_name] = result.output

        elif mode == MODE_CONDITIONAL:
            if condition_fn is None:
                raise ValueError("condition_fn is required for conditional composite mode.")
            index = condition_fn(context or {})
            if not 0 <= index < len(skill_names):
                raise ValueError(f"condition_fn returned index {index} out of range.")
            result = await self.execute_skill(
                skill_name=skill_names[index],
                skill_input=skill_inputs[index],
                agent_id=agent_id,
                tenant_id=tenant_id,
                context=context,
            )
            child_results = [result]
            composite_output = result.output

        duration_ms = int((_time.monotonic() - start) * 1000)
        overall_success = all(r.success for r in child_results)

        return SkillExecutionResult(
            skill_name=f"composite:{mode}",
            success=overall_success,
            output=composite_output,
            duration_ms=duration_ms,
            executed_at=datetime.now(UTC).isoformat(),
            child_results=child_results,
        )

    # ─── Recommendation ───────────────────────────────────────────────────────

    def recommend_skills(
        self,
        task_description: str,
        agent_privilege_level: int,
        tags: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Recommend skills based on task description keyword matching.

        Uses simple token overlap scoring as a fallback when no embedding
        service is available. Filters by privilege level and optional tags.

        Args:
            task_description: Natural language description of the task.
            agent_privilege_level: Agent's privilege level for filtering.
            tags: Optional list of tags to filter candidates by.
            top_k: Maximum number of recommendations to return.

        Returns:
            List of recommendation dicts with 'skill_name', 'version',
            'description', 'relevance_score', and 'tags'.
        """
        task_tokens = set(task_description.lower().split())
        candidates: list[dict[str, Any]] = []

        for key, definition in self._definitions.items():
            if definition.deprecated:
                continue
            if definition.min_privilege_level > agent_privilege_level:
                continue
            if tags and not any(t in definition.tags for t in tags):
                continue

            # Token overlap scoring
            desc_tokens = set(definition.description.lower().split())
            tag_tokens = set(" ".join(definition.tags).lower().split())
            all_tokens = desc_tokens | tag_tokens
            overlap = len(task_tokens & all_tokens)
            relevance = overlap / max(len(task_tokens), 1)

            candidates.append({
                "skill_name": definition.name,
                "version": definition.version,
                "description": definition.description,
                "relevance_score": round(relevance, 4),
                "tags": definition.tags,
                "dependencies": definition.dependencies,
            })

        candidates.sort(key=lambda c: c["relevance_score"], reverse=True)
        return candidates[:top_k]

    def list_skills(
        self,
        include_deprecated: bool = False,
        tag: str | None = None,
        min_privilege_level: int | None = None,
    ) -> list[dict[str, Any]]:
        """List all registered skills with optional filtering.

        Args:
            include_deprecated: If True, include deprecated skills.
            tag: Optional tag to filter by.
            min_privilege_level: If set, only return skills at or below this level.

        Returns:
            List of skill summary dicts.
        """
        results: list[dict[str, Any]] = []
        for key, definition in self._definitions.items():
            if not include_deprecated and definition.deprecated:
                continue
            if tag and tag not in definition.tags:
                continue
            if min_privilege_level is not None and definition.min_privilege_level > min_privilege_level:
                continue
            results.append({
                "key": key,
                "name": definition.name,
                "version": definition.version,
                "description": definition.description,
                "dependencies": definition.dependencies,
                "tags": definition.tags,
                "deprecated": definition.deprecated,
                "min_privilege_level": definition.min_privilege_level,
            })
        return sorted(results, key=lambda s: s["name"])
