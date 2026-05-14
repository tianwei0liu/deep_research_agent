"""Deep Research Agent built with ``deepagents.create_deep_agent``.

Provides:
- ``build_deep_agent``      — compile a LangGraph agent with MCP search tools.
- ``run_deep_research``     — one-shot query → full report text.
- ``stream_deep_research``  — async generator yielding structured streaming events.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, AsyncGenerator, Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage

from deep_research_agent.agents.mcp_client import MCPSearchClient
from deep_research_agent.agents.patched_deepseek import PatchedChatDeepSeek
from deep_research_agent.agents.prompts import DeepAgentPrompts
from deep_research_agent.agents.tools import load_settings

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.types import Checkpointer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backward-compatible aliases for test patch targets.
# Tests patch "deep_research_agent.agents.agent._load_settings"
# so we must keep this name importable from this module.
# ---------------------------------------------------------------------------
_load_settings = load_settings


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _build_activate_persona_tool(
    persona_middleware: "PersonaMiddleware",
) -> Any:
    """Build the ``activate_persona`` tool for the Supervisor.

    This tool allows the Supervisor to dynamically activate a persona
    framework.  When called, it mutates the ``PersonaMiddleware``
    instance in-process.  The middleware's ``awrap_tool_call`` hook
    then persists the ``active_persona_id`` in the graph state via
    ``Command(update=...)``, ensuring the persona survives checkpointing
    and graph rebuilds in multi-turn conversations.

    Args:
        persona_middleware: The ``PersonaMiddleware`` instance that will
            be mutated when the tool is called.

    Returns:
        A LangChain ``@tool``-decorated function.
    """
    from langchain_core.tools import tool

    @tool
    def activate_persona(persona_id: str) -> str:
        """Activate a persona analysis framework for this research session.

        Call this tool to enable a specific cognitive framework (e.g.,
        value investing, first-principles thinking) that will influence
        how research tasks are decomposed and how the final report is
        written.

        Args:
            persona_id: The identifier of the persona to activate
                (e.g., "buffett", "feynman", "zhangxuefeng").

        Returns:
            A status message confirming activation or explaining failure.
        """
        try:
            success = persona_middleware.activate(persona_id)
        except RuntimeError as exc:
            return f"Activation failed: {exc}"

        if success:
            config = persona_middleware.current_persona
            display_name = config.display_name if config else persona_id
            return (
                f"Persona '{display_name}' ({persona_id}) activated successfully. "
                f"The research will now use this analytical framework."
            )
        return (
            f"Persona '{persona_id}' not found in the registry. "
            f"Proceeding with objective analysis."
        )

    return activate_persona


async def build_deep_agent(
    *,
    mcp_client: MCPSearchClient,
    checkpointer: Optional[Checkpointer] = None,
    persona_id: Optional[str] = None,
    enable_skills_discovery: bool = True,
    **overrides: Any,
) -> CompiledStateGraph:
    """Build and return a compiled deep research agent.

    Args:
        mcp_client: Active MCP client with tools already loaded.
            Must be used inside an ``async with MCPSearchClient()``
            context manager.
        checkpointer: Optional LangGraph checkpointer for state persistence
            and multi-turn conversation support (e.g. ``MemorySaver()``).
        persona_id: Optional persona framework to activate at build time
            (e.g. ``"buffett"``).  When set, the persona is pre-activated
            and Skills Discovery is skipped.
        enable_skills_discovery: Whether to enable the interactive Skills
            Discovery phase.  When ``True`` (default) and ``persona_id``
            is not set, the Supervisor will recommend personas to the
            user during the first turn.  Set to ``False`` to disable
            for backward compatibility or non-interactive use.
        **overrides: Forwarded to ``create_deep_agent``.
            - ``model``: Override the main orchestrator model.
            - ``worker_model``: Override the worker subagent model.

    Returns:
        A compiled LangGraph ``CompiledStateGraph``.
    """
    from deepagents import create_deep_agent

    from deep_research_agent.agents.citation.citation_middleware import (
        CitationDataMiddleware,
    )
    from deep_research_agent.agents.citation.models import WorkerOutput

    cfg = _load_settings()

    # --- Resolve orchestration limits from config ---
    supervisor_max_turns = cfg.get("supervisor_max_turns", 35)
    supervisor_max_search_calls = cfg.get("supervisor_max_search_calls", 10)
    worker_max_search_calls = cfg.get("worker_max_search_calls", 60)
    worker_max_turns = cfg.get("worker_max_turns", 20)
    citation_max_retries = cfg.get("citation_max_retries", 5)

    # --- Persona registry + middleware (supports runtime activation) ---
    from deep_research_agent.agents.persona_middleware import PersonaMiddleware
    from deep_research_agent.agents.skills_catalog import SkillsCatalog
    from deep_research_agent.personas.registry import PersonaRegistry

    registry = PersonaRegistry()

    # Pre-activate if persona_id is provided via CLI (backward compat)
    persona_config = None
    if persona_id:
        persona_config = registry.get(persona_id)
        if persona_config is None:
            logger.warning(
                "Persona '%s' not found in registry, running without persona",
                persona_id,
            )
        else:
            logger.info(
                "Persona activated (build-time): %s (%s)",
                persona_config.display_name,
                persona_id,
            )

    persona_middleware = PersonaMiddleware(
        persona=persona_config,
        registry=registry,
    )

    # --- Skills catalog for prompt injection ---
    catalog = SkillsCatalog(registry=registry)
    skills_table = catalog.format_skills_table() if enable_skills_discovery else ""

    # --- Format prompts with concrete limits + skills table ---
    supervisor_prompt = DeepAgentPrompts.format_supervisor_prompt(
        max_turns=supervisor_max_turns,
        max_search_calls=supervisor_max_search_calls,
        skills_table=skills_table,
    )
    worker_prompt = DeepAgentPrompts.format_worker_prompt(
        max_search_calls=worker_max_search_calls,
        max_turns=worker_max_turns,
    )

    # Use PatchedChatDeepSeek to fix reasoning_content passback for
    # DeepSeek V4 thinking-mode models.  Falls back to caller overrides
    # (which may be strings or pre-built instances for other providers).
    main_model = overrides.pop("model", None) or PatchedChatDeepSeek(
        model=cfg["planner_model"],
    )
    # Worker uses response_format (Pydantic structured output), which
    # triggers forced tool_choice — incompatible with DeepSeek thinking
    # mode.  Disable thinking for workers; the prompt-level CoT in the
    # Worker prompt compensates.
    worker_model = overrides.pop("worker_model", None) or PatchedChatDeepSeek(
        model=cfg["worker_model"],
        extra_body={"thinking": {"type": "disabled"}},
    )

    # --- MCP tools: dynamic discovery ---
    # Supervisor: web_search + activate_persona
    supervisor_tools = list(mcp_client.get_tools(names=["web_search"]))

    # Build and register the activate_persona tool
    activate_tool = _build_activate_persona_tool(persona_middleware)
    supervisor_tools.append(activate_tool)

    # Worker: web_search + scrape_url only (vertical search tools disabled
    # to reduce Bocha API consumption)
    worker_tools = mcp_client.get_tools(names=["web_search", "scrape_url"])

    research_subagent: dict[str, Any] = {
        "name": "research-worker",
        "description": (
            "Conducts focused web research on a specific sub-topic. "
            "Delegate narrow, well-scoped objectives to this agent."
        ),
        "system_prompt": worker_prompt,
        "tools": worker_tools,
        "model": worker_model,
        "response_format": WorkerOutput,
    }

    from langgraph.prebuilt import create_react_agent

    citation_graph = create_react_agent(
        worker_model,
        tools=[],
        prompt=DeepAgentPrompts.CITATION_SPECIALIST
    ).with_config({"recursion_limit": 15})

    citation_specialist: dict[str, Any] = {
        "name": "citation-specialist",
        "description": (
            "Adds accurate inline citations [1], [2]... to a draft research "
            "report. Delegate to this agent AFTER all research is complete "
            "and you have written a draft report. Pass ONLY the draft report "
            "as the task description — worker findings are auto-injected."
        ),
        "runnable": citation_graph,
    }

    # --- Budget tracking middleware ---
    from deep_research_agent.agents.budget_middleware import (
        BudgetTrackingMiddleware,
    )
    budget_middleware = BudgetTrackingMiddleware(max_turns=supervisor_max_turns)

    return create_deep_agent(
        model=main_model,
        tools=supervisor_tools,
        system_prompt=supervisor_prompt,
        subagents=[research_subagent, citation_specialist],
        middleware=[
            persona_middleware,
            budget_middleware,
            CitationDataMiddleware(max_retries=citation_max_retries),
        ],
        checkpointer=checkpointer,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Streaming runner (async generator)
# ---------------------------------------------------------------------------

async def stream_deep_research(
    query: str,
    *,
    thread_id: Optional[str] = None,
    checkpointer: Optional[Checkpointer] = None,
    persona_id: Optional[str] = None,
    enable_skills_discovery: bool = True,
    **overrides: Any,
) -> AsyncGenerator[dict[str, Any], None]:
    """Stream a deep research query, yielding structured events in real time.

    This is the primary entry point for building UIs or CLI tools that need
    incremental feedback during long-running research.

    Args:
        query: The research question (or follow-up message for multi-turn).
        thread_id: Conversation thread ID. When the same ``thread_id`` is
            reused across calls (with a checkpointer), the agent resumes
            the previous conversation — enabling multi-turn research.
            Defaults to a new UUID if not provided.
        checkpointer: Optional checkpointer for persistence. Required for
            multi-turn conversations across calls.
        persona_id: Optional persona framework to activate (e.g.
            ``"buffett"``).  See ``build_deep_agent`` for details.
        enable_skills_discovery: Whether to include the persona skills
            table in the Supervisor prompt.  When ``True`` (default) and
            ``persona_id`` is not set, the Supervisor will autonomously
            select and activate the best persona for the query.
        **overrides: Forwarded to ``build_deep_agent``.

    Yields:
        Structured event dicts with ``type`` and ``data`` keys:
        - ``{"type": "status", "data": "..."}`` — lifecycle status messages.
        - ``{"type": "tool_start", "data": {"name": ..., "input": ...}}``
        - ``{"type": "tool_end", "data": {"name": ..., "output": ...}}``
        - ``{"type": "token", "data": "..."}`` — streamed output tokens.
        - ``{"type": "final_report", "data": "..."}`` — the complete report.
    """
    resolved_thread_id = thread_id or str(uuid.uuid4())

    cfg = _load_settings()
    timeout = cfg.get("research_timeout_seconds", 600)
    # Allow 0 to mean "no timeout"
    effective_timeout: float | None = timeout if timeout > 0 else None

    config = {
        "configurable": {"thread_id": resolved_thread_id},
        "recursion_limit": 1000,
    }

    yield {"type": "status", "data": f"Starting research (thread={resolved_thread_id})"}

    # MCP session lifecycle: spawn search service → run agent → cleanup
    async with MCPSearchClient() as mcp_client:
        yield {
            "type": "status",
            "data": f"MCP search service connected ({len(mcp_client.get_tools())} tools)",
        }

        agent = await build_deep_agent(
            mcp_client=mcp_client,
            checkpointer=checkpointer,
            persona_id=persona_id,
            enable_skills_discovery=enable_skills_discovery,
            **overrides,
        )

        agent_input = {"messages": [("user", query)]}

        timed_out = False
        root_run_id_yielded = False
        citation_done = False
        try:
            cm = asyncio.timeout(effective_timeout)
            async with cm:
                async for event in agent.astream_events(
                    agent_input,
                    config=config,
                    version="v2",
                ):
                    kind = event.get("event", "")

                    if not root_run_id_yielded and kind == "on_chain_start":
                        run_id = event.get("run_id")
                        if run_id:
                            yield {"type": "run_id", "data": run_id}
                            root_run_id_yielded = True

                    if kind == "on_tool_start":
                        tool_name = event.get("name", "unknown")
                        tool_input = event.get("data", {}).get("input", {})
                        yield {"type": "tool_start", "data": {"name": tool_name, "input": tool_input}}

                    elif kind == "on_tool_end":
                        tool_name = event.get("name", "unknown")
                        tool_output = event.get("data", {}).get("output", "")
                        output_str = str(tool_output)
                        if len(output_str) > 500:
                            output_str = output_str[:500] + "..."
                        yield {"type": "tool_end", "data": {"name": tool_name, "output": output_str}}
                        # After citation-specialist returns, the report is
                        # finalized.  Suppress subsequent Supervisor tokens
                        # (e.g. spurious persona recommendations).
                        if tool_name == "citation-specialist":
                            citation_done = True

                    elif kind == "on_chat_model_stream":
                        if citation_done:
                            continue
                        langgraph_node = event.get("metadata", {}).get("langgraph_node")
                        # create_agent uses "model" as the main agent node name.
                        # Filter to the supervisor node only — exclude subagent
                        # (worker / citation) streaming which runs in child graphs.
                        if langgraph_node in ("model", "agent"):
                            chunk = event.get("data", {}).get("chunk")
                            if chunk and hasattr(chunk, "content") and chunk.content:
                                token = chunk.content
                                if isinstance(token, str):
                                    yield {"type": "token", "data": token}
        except TimeoutError:
            timed_out = True
            logger.warning(
                "Research timed out after %ds, forcing finalization", timeout
            )
            yield {
                "type": "status",
                "data": "Research timed out. Finalizing partial report...",
            }



        if timed_out:
            # --- Checkpoint resume: force Supervisor to produce a report ---
            try:
                partial_report = await _handle_timeout_resume(
                    agent, config, finalization_timeout=120
                )
                yield {
                    "type": "final_report",
                    "data": partial_report,
                    "metadata": {"truncated": True, "reason": "timeout"},
                }
            except Exception as exc:
                logger.error("Finalization failed: %s", exc)
                state = await agent.aget_state(config)
                fallback = _extract_raw_partial(state)
                yield {
                    "type": "final_report",
                    "data": fallback,
                    "metadata": {"truncated": True, "reason": "timeout_fallback"},
                }
        else:
            # --- Normal completion: extract report from final state ---
            try:
                state = await agent.aget_state(config)
                messages = state.values.get("messages", [])
                logger.debug(
                    "Report extraction: %d messages in state", len(messages)
                )
                if messages:
                    candidate_contents = [
                        msg.content for msg in messages[-10:]
                        if hasattr(msg, "content") and isinstance(msg.content, str)
                    ]
                    logger.debug(
                        "Report extraction: %d candidates from last 10 messages",
                        len(candidate_contents),
                    )
                    if candidate_contents:
                        best_content = max(candidate_contents, key=len)
                        logger.info(
                            "Final report extracted: %d chars", len(best_content)
                        )
                        yield {"type": "final_report", "data": best_content}
                    else:
                        logger.warning("No text candidates found in final state")
                else:
                    logger.warning("No messages in final state")
            except Exception as e:
                logger.error("Failed to extract final report from state: %s", e)

    yield {"type": "status", "data": "Research complete"}


# ---------------------------------------------------------------------------
# Convenience one-shot runner
# ---------------------------------------------------------------------------

async def run_deep_research(
    query: str,
    *,
    thread_id: Optional[str] = None,
    checkpointer: Optional[Checkpointer] = None,
    persona_id: Optional[str] = None,
    enable_skills_discovery: bool = False,
    **overrides: Any,
) -> str:
    """Run a deep research query end-to-end, return the final report text.

    This is a convenience wrapper around ``stream_deep_research`` that
    consumes all events and returns the final report.

    .. note::
        Skills Discovery is disabled by default here because this is a
        non-interactive runner that cannot handle HITL interrupts.
        Use ``stream_deep_research`` for interactive flows.

    Args:
        query: The research question.
        thread_id: Optional thread ID for multi-turn conversations.
        checkpointer: Optional checkpointer for state persistence.
        persona_id: Optional persona framework to activate.
        enable_skills_discovery: Default ``False`` (non-interactive).
        **overrides: Forwarded to ``build_deep_agent``.

    Returns:
        The final assistant message content (markdown report).
    """
    report = ""
    async for event in stream_deep_research(
        query,
        thread_id=thread_id,
        checkpointer=checkpointer,
        persona_id=persona_id,
        enable_skills_discovery=enable_skills_discovery,
        **overrides,
    ):
        if event["type"] == "final_report":
            report = event["data"]

    return report


# ---------------------------------------------------------------------------
# Timeout recovery helpers
# ---------------------------------------------------------------------------

async def _handle_timeout_resume(
    agent: CompiledStateGraph,
    config: dict[str, Any],
    finalization_timeout: int = 120,
) -> str:
    """Resume from checkpoint with a TIMEOUT message to force report generation.

    Injects a HumanMessage instructing the Supervisor to immediately
    synthesize findings and delegate to citation-specialist.  Uses the
    same ``thread_id`` so LangGraph resumes from the existing checkpoint.

    Args:
        agent: The compiled agent graph.
        config: LangGraph config dict (must contain the same thread_id).
        finalization_timeout: Seconds to allow for the finalization run.

    Returns:
        The partial report text extracted from the resumed run.

    Raises:
        ValueError: If no substantive report could be extracted.
        TimeoutError: If finalization itself times out.
    """
    timeout_msg = HumanMessage(content=(
        "\u23f0 RESEARCH TIMEOUT \u2014 The research phase has exceeded the time limit. "
        "You MUST immediately:\n"
        "1. Stop all pending research tasks\n"
        "2. Synthesize a report from ALL findings gathered so far\n"
        "3. Delegate to citation-specialist for final formatting\n"
        "4. Output the final report\n\n"
        "Do NOT start any new research. Use only the data you already have."
    ))

    async with asyncio.timeout(finalization_timeout):
        result = await agent.ainvoke(
            {"messages": [timeout_msg]},
            config,  # Same thread_id → resumes from checkpoint
        )

    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content and len(msg.content) > 200:
            return msg.content

    raise ValueError("No substantive report generated during finalization")


def _extract_raw_partial(state: Any) -> str:
    """Last-resort extraction from raw agent state.

    Walks backward through messages to find the last substantive
    content block and appends a truncation warning.

    Args:
        state: The agent state snapshot from ``aget_state``.

    Returns:
        Best-effort report text with truncation notice.
    """
    messages = state.values.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content and len(msg.content) > 100:
            return (
                msg.content
                + "\n\n---\n> \u26a0\ufe0f 本报告因研究超时被截断，内容可能不完整。"
            )
    return "研究超时且无法生成报告。请缩小查询范围或增加超时时间。"
