"""Deep Research Agent built with ``deepagents.create_deep_agent``.

Provides:
- ``build_deep_agent``      — compile a LangGraph agent with optional checkpointer.
- ``run_deep_research``     — one-shot query → full report text.
- ``stream_deep_research``  — async generator yielding structured streaming events.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, AsyncGenerator, Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage

from deep_research_agent.agents.patched_deepseek import PatchedChatDeepSeek
from deep_research_agent.agents.prompts import DeepAgentPrompts
from deep_research_agent.agents.tools import (
    load_settings,
    make_internet_search,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.types import Checkpointer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backward-compatible aliases for test patch targets.
# Tests patch "deep_research_agent.agents.agent._load_settings"
# and "...._make_internet_search", so we must keep these names importable
# from this module.
# ---------------------------------------------------------------------------
_load_settings = load_settings
_make_internet_search = make_internet_search


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def build_deep_agent(
    *,
    checkpointer: Optional[Checkpointer] = None,
    **overrides: Any,
) -> CompiledStateGraph:
    """Build and return a compiled deep research agent.

    Args:
        checkpointer: Optional LangGraph checkpointer for state persistence
            and multi-turn conversation support (e.g. ``MemorySaver()``).
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
    search_tool = _make_internet_search(cfg["tavily_api_key"])

    # --- Resolve orchestration limits from config ---
    supervisor_max_turns = cfg.get("supervisor_max_turns", 35)
    supervisor_max_search_calls = cfg.get("supervisor_max_search_calls", 10)
    worker_max_search_calls = cfg.get("worker_max_search_calls", 60)
    worker_max_turns = cfg.get("worker_max_turns", 20)
    citation_max_retries = cfg.get("citation_max_retries", 5)

    # --- Format prompts with concrete limits ---
    supervisor_prompt = DeepAgentPrompts.format_supervisor_prompt(
        max_turns=supervisor_max_turns,
        max_search_calls=supervisor_max_search_calls,
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

    research_subagent: dict[str, Any] = {
        "name": "research-worker",
        "description": (
            "Conducts focused web research on a specific sub-topic. "
            "Delegate narrow, well-scoped objectives to this agent."
        ),
        "system_prompt": worker_prompt,
        "tools": [search_tool],
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
        tools=[search_tool],
        system_prompt=supervisor_prompt,
        subagents=[research_subagent, citation_specialist],
        middleware=[
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
    agent = build_deep_agent(checkpointer=checkpointer, **overrides)

    cfg = _load_settings()
    timeout = cfg.get("research_timeout_seconds", 600)
    # Allow 0 to mean "no timeout"
    effective_timeout: float | None = timeout if timeout > 0 else None

    config = {
        "configurable": {"thread_id": resolved_thread_id},
        "recursion_limit": 1000,
    }

    yield {"type": "status", "data": f"Starting research (thread={resolved_thread_id})"}

    timed_out = False
    root_run_id_yielded = False
    try:
        cm = asyncio.timeout(effective_timeout)
        async with cm:
            async for event in agent.astream_events(
                {"messages": [("user", query)]},
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

                elif kind == "on_chat_model_stream":
                    langgraph_node = event.get("metadata", {}).get("langgraph_node")
                    if langgraph_node == "agent":
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
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content") and last_msg.content:
                    yield {"type": "final_report", "data": last_msg.content}
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
    **overrides: Any,
) -> str:
    """Run a deep research query end-to-end, return the final report text.

    This is a convenience wrapper around ``stream_deep_research`` that
    consumes all events and returns the final report.

    Args:
        query: The research question.
        thread_id: Optional thread ID for multi-turn conversations.
        checkpointer: Optional checkpointer for state persistence.
        **overrides: Forwarded to ``build_deep_agent``.

    Returns:
        The final assistant message content (markdown report).
    """
    report = ""
    async for event in stream_deep_research(
        query,
        thread_id=thread_id,
        checkpointer=checkpointer,
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
