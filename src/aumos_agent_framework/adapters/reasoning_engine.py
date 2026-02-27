"""LLM-backed reasoning engine adapter — chain-of-thought and ReAct loop implementation.

Implements ReasoningEngineProtocol by delegating inference to aumos-llm-serving
via HTTP. Supports chain-of-thought step tracking, ReAct (Reason + Act) loop,
tool selection reasoning, and confidence scoring with full trace logging.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_DEFAULT_LLM_SERVING_BASE_URL = "http://aumos-llm-serving:8080"
_DEFAULT_MAX_REACT_ITERATIONS = 10
_DEFAULT_INFERENCE_TIMEOUT_SECONDS = 60
_DEFAULT_TEMPERATURE = 0.0
_CONFIDENCE_HIGH = 0.85
_CONFIDENCE_MEDIUM = 0.6


@dataclass
class ReasoningStep:
    """A single step in a chain-of-thought or ReAct reasoning trace.

    Attributes:
        step_number: Sequential position in the reasoning chain.
        step_type: One of 'thought', 'observation', 'action', 'conclusion'.
        content: Raw text content of this step.
        confidence: Model-estimated confidence for this step (0.0-1.0).
        tool_name: Tool selected for action steps, None otherwise.
        tool_input: Input dict for tool action steps, None otherwise.
        timestamp: UTC timestamp when this step was generated.
        token_count: Approximate token count consumed for this step.
    """

    step_number: int
    step_type: str
    content: str
    confidence: float
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for storage and API responses."""
        return {
            "step_number": self.step_number,
            "step_type": self.step_type,
            "content": self.content,
            "confidence": self.confidence,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
        }


@dataclass
class ReasoningTrace:
    """Complete trace of a reasoning session.

    Attributes:
        trace_id: Unique ID for this reasoning session.
        agent_id: Agent that performed the reasoning.
        task_description: Original task given to the agent.
        steps: Ordered list of reasoning steps.
        final_conclusion: Final answer or decision from the reasoning chain.
        total_tokens: Total tokens consumed across all steps.
        iteration_count: Number of ReAct iterations performed.
        terminated_by: 'conclusion', 'max_iterations', or 'error'.
        started_at: ISO timestamp of trace start.
        completed_at: ISO timestamp of trace completion.
    """

    trace_id: str
    agent_id: str
    task_description: str
    steps: list[ReasoningStep] = field(default_factory=list)
    final_conclusion: str | None = None
    total_tokens: int = 0
    iteration_count: int = 0
    terminated_by: str = "in_progress"
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for storage and API responses."""
        return {
            "trace_id": self.trace_id,
            "agent_id": self.agent_id,
            "task_description": self.task_description,
            "steps": [s.to_dict() for s in self.steps],
            "final_conclusion": self.final_conclusion,
            "total_tokens": self.total_tokens,
            "iteration_count": self.iteration_count,
            "terminated_by": self.terminated_by,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class ReasoningEngine:
    """LLM-powered reasoning engine supporting chain-of-thought and ReAct loops.

    Connects to aumos-llm-serving for model inference. Structures prompts for
    multi-step reasoning and parses structured outputs to extract thoughts,
    actions, tool selections, and final conclusions.
    """

    def __init__(
        self,
        llm_serving_base_url: str = _DEFAULT_LLM_SERVING_BASE_URL,
        max_react_iterations: int = _DEFAULT_MAX_REACT_ITERATIONS,
        inference_timeout_seconds: int = _DEFAULT_INFERENCE_TIMEOUT_SECONDS,
        temperature: float = _DEFAULT_TEMPERATURE,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize with LLM serving configuration.

        Args:
            llm_serving_base_url: Base URL of the aumos-llm-serving service.
            max_react_iterations: Maximum ReAct loop iterations before forced stop.
            inference_timeout_seconds: HTTP timeout for each inference call.
            temperature: LLM sampling temperature (0.0 for deterministic reasoning).
            http_client: Optional pre-configured httpx client (for testing).
        """
        self._llm_base_url = llm_serving_base_url.rstrip("/")
        self._max_react_iterations = max_react_iterations
        self._inference_timeout = inference_timeout_seconds
        self._temperature = temperature
        self._http_client = http_client

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Return or lazily create the async HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._inference_timeout),
                headers={"Content-Type": "application/json"},
            )
        return self._http_client

    async def _call_llm(
        self,
        messages: list[dict[str, str]],
        tenant_id: str,
        model_id: str | None = None,
        response_format: str = "text",
    ) -> dict[str, Any]:
        """Make an inference call to aumos-llm-serving.

        Args:
            messages: List of role/content message dicts.
            tenant_id: Tenant context for model routing and billing.
            model_id: Optional model override; uses serving default if None.
            response_format: 'text' or 'json_object' for structured output.

        Returns:
            Response dict with 'content', 'token_count', and 'model_id' keys.
        """
        client = await self._get_http_client()
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": self._temperature,
            "tenant_id": tenant_id,
            "response_format": response_format,
        }
        if model_id:
            payload["model_id"] = model_id

        try:
            response = await client.post(
                f"{self._llm_base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return {
                "content": data["choices"][0]["message"]["content"],
                "token_count": data.get("usage", {}).get("total_tokens", 0),
                "model_id": data.get("model", "unknown"),
            }
        except httpx.HTTPStatusError as exc:
            logger.error(
                "LLM serving HTTP error",
                status_code=exc.response.status_code,
                url=str(exc.request.url),
                tenant_id=tenant_id,
            )
            raise
        except httpx.TimeoutException:
            logger.error(
                "LLM serving inference timeout",
                timeout_seconds=self._inference_timeout,
                tenant_id=tenant_id,
            )
            raise

    async def chain_of_thought(
        self,
        agent_id: uuid.UUID,
        task_description: str,
        context: dict[str, Any],
        tenant_id: str,
        model_id: str | None = None,
    ) -> ReasoningTrace:
        """Execute chain-of-thought reasoning for a task.

        Generates a structured multi-step thought chain leading to a conclusion.
        Each step is tracked with confidence scoring.

        Args:
            agent_id: Agent executing the reasoning.
            task_description: The task or question to reason about.
            context: Additional context dict (memory, tools, facts).
            tenant_id: Tenant context.
            model_id: Optional model override.

        Returns:
            ReasoningTrace with all steps and final conclusion.
        """
        trace = ReasoningTrace(
            trace_id=str(uuid.uuid4()),
            agent_id=str(agent_id),
            task_description=task_description,
        )

        system_prompt = """You are a careful, step-by-step reasoning agent.
Think through the task methodically. Output your reasoning as a JSON object with this structure:
{
  "steps": [
    {"thought": "...", "confidence": 0.0-1.0},
    ...
  ],
  "conclusion": "Your final answer or decision",
  "overall_confidence": 0.0-1.0
}
Be precise, acknowledge uncertainty, and base conclusions on evidence."""

        context_str = json.dumps(context, indent=2) if context else "No additional context."
        user_message = f"Task: {task_description}\n\nContext:\n{context_str}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        logger.info(
            "Chain-of-thought reasoning started",
            trace_id=trace.trace_id,
            agent_id=str(agent_id),
            tenant_id=tenant_id,
        )

        try:
            llm_response = await self._call_llm(
                messages=messages,
                tenant_id=tenant_id,
                model_id=model_id,
                response_format="json_object",
            )

            parsed = json.loads(llm_response["content"])
            raw_steps = parsed.get("steps", [])

            for index, raw_step in enumerate(raw_steps):
                step = ReasoningStep(
                    step_number=index + 1,
                    step_type="thought",
                    content=raw_step.get("thought", ""),
                    confidence=float(raw_step.get("confidence", 0.5)),
                    token_count=llm_response["token_count"] // max(len(raw_steps), 1),
                )
                trace.steps.append(step)
                trace.total_tokens += step.token_count

            conclusion_step = ReasoningStep(
                step_number=len(trace.steps) + 1,
                step_type="conclusion",
                content=parsed.get("conclusion", ""),
                confidence=float(parsed.get("overall_confidence", 0.5)),
            )
            trace.steps.append(conclusion_step)
            trace.final_conclusion = parsed.get("conclusion")
            trace.terminated_by = "conclusion"

        except Exception as exc:
            trace.terminated_by = "error"
            logger.error(
                "Chain-of-thought reasoning failed",
                trace_id=trace.trace_id,
                agent_id=str(agent_id),
                error=str(exc),
                tenant_id=tenant_id,
            )
            raise

        trace.completed_at = datetime.now(UTC).isoformat()
        logger.info(
            "Chain-of-thought reasoning complete",
            trace_id=trace.trace_id,
            steps=len(trace.steps),
            total_tokens=trace.total_tokens,
            conclusion_confidence=conclusion_step.confidence,
            tenant_id=tenant_id,
        )
        return trace

    async def react_loop(
        self,
        agent_id: uuid.UUID,
        task_description: str,
        available_tools: list[dict[str, Any]],
        context: dict[str, Any],
        tenant_id: str,
        execute_tool_fn: Any,  # Callable[[str, dict], Awaitable[Any]]
        model_id: str | None = None,
    ) -> ReasoningTrace:
        """Execute a ReAct (Reason + Act) loop for tool-using agents.

        Alternates between Thought (reasoning about next action) and
        Act (tool invocation) until a final answer is reached or the
        maximum iteration limit is hit.

        Args:
            agent_id: Agent executing the loop.
            task_description: The task to complete.
            available_tools: List of tool definition dicts the agent can use.
            context: Initial context dict.
            tenant_id: Tenant context.
            execute_tool_fn: Async callable(tool_name, tool_input) → tool_output.
            model_id: Optional model override.

        Returns:
            ReasoningTrace with interleaved thought/action/observation steps.
        """
        trace = ReasoningTrace(
            trace_id=str(uuid.uuid4()),
            agent_id=str(agent_id),
            task_description=task_description,
        )

        tools_description = json.dumps(
            [{"name": t["name"], "description": t["description"],
              "input_schema": t.get("input_schema", {})}
             for t in available_tools],
            indent=2,
        )

        system_prompt = f"""You are a ReAct agent. For each step, output a JSON object:
{{
  "thought": "Your reasoning about what to do next",
  "action": "tool_name" or "final_answer",
  "action_input": {{...}} or "your final answer string",
  "confidence": 0.0-1.0
}}

Available tools:
{tools_description}

Use "final_answer" as action when you have enough information to answer the task.
Never guess — use tools to gather facts."""

        conversation: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task_description}\n\nInitial context: {json.dumps(context)}"},
        ]

        logger.info(
            "ReAct loop started",
            trace_id=trace.trace_id,
            agent_id=str(agent_id),
            tool_count=len(available_tools),
            tenant_id=tenant_id,
        )

        for iteration in range(self._max_react_iterations):
            trace.iteration_count = iteration + 1

            try:
                llm_response = await self._call_llm(
                    messages=conversation,
                    tenant_id=tenant_id,
                    model_id=model_id,
                    response_format="json_object",
                )
                parsed = json.loads(llm_response["content"])
            except Exception as exc:
                trace.terminated_by = "error"
                logger.error(
                    "ReAct LLM call failed",
                    trace_id=trace.trace_id,
                    iteration=iteration + 1,
                    error=str(exc),
                    tenant_id=tenant_id,
                )
                raise

            thought = parsed.get("thought", "")
            action = parsed.get("action", "")
            action_input = parsed.get("action_input", {})
            confidence = float(parsed.get("confidence", 0.5))

            thought_step = ReasoningStep(
                step_number=len(trace.steps) + 1,
                step_type="thought",
                content=thought,
                confidence=confidence,
                token_count=llm_response["token_count"],
            )
            trace.steps.append(thought_step)
            trace.total_tokens += llm_response["token_count"]

            # Check for final answer
            if action == "final_answer":
                conclusion = action_input if isinstance(action_input, str) else json.dumps(action_input)
                conclusion_step = ReasoningStep(
                    step_number=len(trace.steps) + 1,
                    step_type="conclusion",
                    content=conclusion,
                    confidence=confidence,
                )
                trace.steps.append(conclusion_step)
                trace.final_conclusion = conclusion
                trace.terminated_by = "conclusion"
                break

            # Record action step
            action_step = ReasoningStep(
                step_number=len(trace.steps) + 1,
                step_type="action",
                content=f"Calling tool: {action}",
                confidence=confidence,
                tool_name=action,
                tool_input=action_input if isinstance(action_input, dict) else {},
            )
            trace.steps.append(action_step)

            # Execute the tool
            try:
                tool_result = await execute_tool_fn(action, action_step.tool_input)
                observation_content = json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
            except Exception as exc:
                observation_content = f"Tool execution error: {exc}"
                logger.warning(
                    "ReAct tool execution failed",
                    trace_id=trace.trace_id,
                    tool_name=action,
                    error=str(exc),
                    tenant_id=tenant_id,
                )

            observation_step = ReasoningStep(
                step_number=len(trace.steps) + 1,
                step_type="observation",
                content=observation_content,
                confidence=1.0,  # Observations are factual
                tool_name=action,
            )
            trace.steps.append(observation_step)

            # Feed observation back into conversation context
            conversation.append({
                "role": "assistant",
                "content": llm_response["content"],
            })
            conversation.append({
                "role": "user",
                "content": f"Observation from {action}: {observation_content}\n\nContinue reasoning.",
            })

        else:
            trace.terminated_by = "max_iterations"
            logger.warning(
                "ReAct loop reached max iterations",
                trace_id=trace.trace_id,
                max_iterations=self._max_react_iterations,
                tenant_id=tenant_id,
            )

        trace.completed_at = datetime.now(UTC).isoformat()
        logger.info(
            "ReAct loop complete",
            trace_id=trace.trace_id,
            iterations=trace.iteration_count,
            steps=len(trace.steps),
            terminated_by=trace.terminated_by,
            total_tokens=trace.total_tokens,
            tenant_id=tenant_id,
        )
        return trace

    async def select_tools(
        self,
        agent_id: uuid.UUID,
        task_description: str,
        available_tools: list[dict[str, Any]],
        tenant_id: str,
        top_k: int = 5,
        model_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Reason about which tools are most relevant for a given task.

        Args:
            agent_id: Agent requesting tool selection.
            task_description: The task to match tools against.
            available_tools: Full list of available tool definitions.
            tenant_id: Tenant context.
            top_k: Maximum number of tools to recommend.
            model_id: Optional model override.

        Returns:
            Sorted list of recommended tool dicts with 'relevance_score' added.
        """
        if not available_tools:
            return []

        tools_summary = "\n".join(
            f"- {t['name']}: {t['description']}" for t in available_tools
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a tool selection reasoner. Given a task and a list of tools, "
                    "output JSON: {\"selected_tools\": [{\"name\": \"...\", \"relevance_score\": 0.0-1.0, "
                    "\"reason\": \"...\"}, ...]} "
                    "Select only tools that are genuinely needed for this task. "
                    f"Return at most {top_k} tools, sorted by relevance_score descending."
                ),
            },
            {
                "role": "user",
                "content": f"Task: {task_description}\n\nAvailable tools:\n{tools_summary}",
            },
        ]

        llm_response = await self._call_llm(
            messages=messages,
            tenant_id=tenant_id,
            model_id=model_id,
            response_format="json_object",
        )
        parsed = json.loads(llm_response["content"])
        selected = parsed.get("selected_tools", [])

        # Enrich with full tool definition
        tools_by_name = {t["name"]: t for t in available_tools}
        result: list[dict[str, Any]] = []
        for selection in selected[:top_k]:
            tool_def = tools_by_name.get(selection["name"])
            if tool_def:
                enriched = {**tool_def, "relevance_score": selection.get("relevance_score", 0.0),
                            "selection_reason": selection.get("reason", "")}
                result.append(enriched)

        logger.info(
            "Tool selection reasoning complete",
            agent_id=str(agent_id),
            available_count=len(available_tools),
            selected_count=len(result),
            tenant_id=tenant_id,
        )
        return result

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
