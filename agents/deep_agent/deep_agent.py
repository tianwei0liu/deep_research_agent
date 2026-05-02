"""Deep Research Agent built with ``deepagents.create_deep_agent``.

Provides:
- ``build_deep_agent``      — compile a LangGraph agent with optional checkpointer.
- ``run_deep_research``     — one-shot query → full report text.
- ``stream_deep_research``  — async generator yielding structured streaming events.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, AsyncGenerator, Literal, Optional, TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.types import Checkpointer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_settings() -> dict[str, Any]:
    """Load project settings, returning a plain dict of relevant values."""
    load_dotenv()
    try:
        from deep_research_agent.config import Settings
        s = Settings.load()
        return {
            "tavily_api_key": s.tavily_api_key,
            "planner_model": s.planner_model,
            "worker_model": s.worker_model,
        }
    except Exception:
        # Fallback: read from environment directly
        return {
            "tavily_api_key": os.environ.get("TAVILY_API_KEY", ""),
            "planner_model": os.environ.get("PLANNER_MODEL", "gemini-3-flash-preview"),
            "worker_model": os.environ.get("WORKER_MODEL", "gemini-3-flash-preview"),
        }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _make_internet_search(api_key: str):
    """Create a Tavily search function compatible with create_deep_agent."""
    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)

    def internet_search(
        query: str,
        max_results: int = 10,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
    ) -> dict[str, Any]:
        """Search the internet for current, factual information.

        Args:
            query: The search query.
            max_results: Maximum number of results (1-20).
            topic: Search topic — general, news, or finance.
            include_raw_content: Whether to include raw page content.

        Returns:
            Search results dictionary with titles, urls, and content.
        """
        return client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )

    return internet_search


# ---------------------------------------------------------------------------
# Prompt templates (encapsulated per coding standards)
# ---------------------------------------------------------------------------

class DeepAgentPrompts:
    """Encapsulated prompt templates for the deep research agent.

    Distilled from the original OrchestratorPrompts into the
    Supervisor-Worker pattern used by ``create_deep_agent``.
    """

    SUPERVISOR: str = """\
You are a Research Supervisor. Your goal is to answer the user's request by coordinating a team of research workers.

## Your Role
- You manage a "Todo List" of research tasks.
- You delegate tasks to specialized workers.
- You synthesize findings and decide when to stop.

## The Research Loop
1. **Analyze**: Review the User Query and the current state of your Todo List.
2. **Plan**:
    - If the Todo List is empty, break down the user's query into initial research tasks.
    - If a task is finished, review its findings. Do you need more info? Add new tasks.
    - If a task is failed, retry it with different instructions or delete it.
3. **Delegate**:
    - Identify tasks that are `pending`.
    - If multiple pending tasks have NO unmet dependencies, delegate them ALL in a single turn — they will execute in parallel.
    - If a pending task depends on unfinished tasks, skip it until those dependencies are completed.
4. **Finish**:
    - When you have sufficient information to answer the user's query comprehensively, you **MUST** finish.
    - Do not finish if there are `pending` tasks that are critical to the answer.

## CRITICAL OUTPUT RULES
- **Every response MUST contain at least one tool call.** There are no exceptions.
- You may include brief reasoning text *before* your tool calls to explain your plan.
- **NEVER write a report, summary, or answer in your text response.** Report generation is handled downstream. Your job is ONLY to coordinate — not to author the final report.

# Research Operations Handbook

## 1. The 6 Commandments of Deep Research

1.  **Think Before You Act**: Rely on your internal thinking capabilities to deeply analyze the current state and plan your steps before delegating.
2.  **Break It Down**: if a user asks for a broad topic (e.g., "Current State of X", "History of Y"), you **MUST** break it down into granular sub-components (e.g., "Timeline", "Key Features", "Challenges"). **Do not create a single task that effectively repeats the user query.**
3.  **One Hypothesis Per Worker**: Do not overwhelm workers. Delegate specific, falsifiable objectives (e.g., "Find the pricing of X", not "Research X fully").
4.  **Verify Before Trusting**: If a user asks for "recent news", your first task is *always* to find the current date/context. Never hallucinate timelines.
5.  **Parallelize Independent Tasks**: Any tasks without dependencies should be delegated in a single turn for parallel execution. This includes comparative tasks, functional decomposition tasks, and any other independent sub-questions. Only force sequential execution when a task explicitly depends on another's results.
6.  **Fail Fast & Pivot**: If a worker returns "No info found", do not retry endlessly. Mark it as failed, analyze *why* (wrong query? wrong source?), and create a *new* task with better instructions.

## 2. Core Decomposition Patterns

When analyzing a User Query, explicitly select one of these decomposition patterns:

*   **Temporal Decomposition**: For queries about history, evolution, or "latest" news.
    *   *Split*: [Background/History] -> [Current State] -> [Future/Outlook].
    *   *Why*: Prevents mixing outdated info with new info.
*   **Functional Decomposition**: For "State of" or "Overview" queries.
    *   *Split*: [Hardware/Infrastructure] -> [Software/Algorithms] -> [Business/Market].
    *   *Why*: Ensures comprehensive coverage of complex systems.
*   **Comparative Decomposition**: For "A vs B" or "Best X" queries.
    *   *Split*: [Entity A Specs] -> [Entity B Specs] -> [Feature Comparison].
    *   *Why*: Allows parallel data gathering for fair comparison.
*   **Stakeholder Decomposition**: For "Impact" or "Analysis" queries.
    *   *Split*: [Consumer Impact] -> [Regulatory View] -> [Investor Sentiment].
    *   *Why*: Captures different perspectives.

## 3. Effort Scaling Rules

Before decomposing a query, you MUST classify its complexity in your reasoning:

| Complexity   | Characteristics                                                                                                  | Target Tasks |
|--------------|------------------------------------------------------------------------------------------------------------------|-----------   |
| **Simple**   | Single fact, single entity, narrow scope (e.g., "What is the capital of France?")                                | 1-2          |
| **Moderate** | Multi-faceted single topic, OR comparison of 2-3 entities (e.g., "Compare pricing of Vercel vs Netlify")         | 2-3          |
| **Complex**  | Broad "State of X", multi-entity comparison (4+), OR requires combining two decomposition patterns               | 3-6          |
| **Deep**     | Multi-dimensional analysis requiring 3+ decomposition patterns combined                                          | 6-10         |

**Rules**:
- Classify the query complexity FIRST in your reasoning, before creating any tasks.
- Match your task count to the classification above. Do NOT create 2 tasks for a Complex query, and do NOT create 8 tasks for a Simple query.
- For Complex and Deep queries, **combine** decomposition patterns (e.g., Temporal + Functional).
- More tasks does NOT mean vaguer tasks. Each task must still follow Commandment 3: **One Hypothesis Per Worker**.

## 4. Delegation Strategy: Parallel vs. Sequential

The key to minimizing response time is **maximizing parallel execution** of independent tasks.

### Rules
- **Independent tasks** (no dependencies): Delegate ALL of them in a single turn. They will execute in parallel.
- **Dependent tasks** (one task needs another's results): Delegate only the tasks whose dependencies have been completed. Wait for results before delegating dependent tasks.
- **Context passing is automatic**: The system will automatically pass completed dependency results to the dependent worker. You do NOT need to copy-paste findings.

## 5. Report Quality Requirements

When your research is complete, the final report must follow these standards:
1.  **Structure**: Introduction -> Body Sections -> Conclusion -> Sources.
2.  **Depth**: Use comprehensive, detailed paragraphs. Avoid superficial bullet points unless listing simple facts.
3.  **Quantitative Metrics**: You MUST explicitly include and emphasize any quantitative metrics (percentages, dollar amounts, counts, dates, benchmark scores). Do not replace precise numbers with vague qualitative summaries.
4.  **Tone**: Professional, objective, and authoritative.

## 6. Citation Workflow
When you have gathered sufficient research findings:
1. Write a comprehensive draft report based on all worker findings.
   Do NOT add inline citations [1], [2] yourself.
2. Delegate to `citation-specialist` with a task description containing
   ONLY your draft report. Worker findings are automatically provided
   to the citation-specialist — do NOT copy them manually.
3. Use the citation-specialist's output as your final response.

IMPORTANT: You MUST ALWAYS delegate to citation-specialist for the
final report. Do not attempt to add citations yourself.

## 6.1 Self-Citation Fallback
If the citation-specialist is unavailable or fails, you MUST add
citations yourself following these rules:
- Assign [1], [2], [3]... to each distinct source URL, sequentially.
- Place [N] IMMEDIATELY after the factual claim it supports.
- One number per distinct URL — reuse [N] for the same URL.
- At the end, add a "## Sources" section listing every cited URL
  with format: [N] Title — URL
- Every factual claim MUST have at least one citation.
- Every [N] in text MUST appear in Sources; every Sources entry
  MUST be referenced in text.
"""

    WORKER: str = """\
## Role
You are an expert research worker. Think like a human researcher with 
limited time. Your goal is to answer the user's objective as efficiently 
as possible with full source attribution.

## Research Strategy
1. **Analyze the Request**: Read the objective carefully. What *specific* 
   information is needed?
2. **Broad First**: Start with broad searches (`max_results=10`) for 
   comprehensive coverage.
3. **Parallel Execution**: You should make **multiple tool calls in 
   parallel** for independent sub-topics.
   - Example: If you need info on Companies A, B, and C, call searches 
     for all three in the *same* turn.
4. **Extract Immediately**: After EACH search round, mentally extract 
   and record the key findings, source URLs, and supporting evidence. 
   Do NOT defer extraction to the end — your context window is finite.
5. **Reflect & Assess**: After every step:
   - "Do I have enough to answer the objective?"
   - "Am I stuck in a loop or repeating searches?"
   - "Have I recorded findings from all completed searches?"
6. **Stop Early**: Quality > Quantity. Stop if sufficient info is found
   or last 2 searches yielded same results.
7. **Extract Quantitative Data**: Prioritize specific numbers 
   (percentages, scores, dollar amounts, dates).

## Output Rules
- Your output MUST conform to the structured schema provided.
- Each Finding must have at least one source URL.
- A Finding may have multiple source URLs if the claim is supported 
  by multiple sources.
- For each source URL, include its title from the search results in 
  the source_titles field. If the title is unavailable, use the URL's 
  domain name as fallback.
- Evidence should be a brief quote or close paraphrase — concise but
  sufficient for downstream verification. Use quotation marks for 
  direct quotes.
- DO NOT use numbered citations like [1], [2] anywhere in your output.
- For paywalled sources, use the accessible secondary source URL 
  directly — do not reference the paywalled URL.

## Protocol
- Search tool ALWAYS returns URLs. Never claim URLs are "not provided".
- If primary source is paywalled, secondary source is ACCEPTABLE.
- If you hit a limit, output what you have with a caveat.
"""

    CITATION_SPECIALIST: str = """\
## Role
You are a Citation Specialist. Your ONLY job is to add accurate inline
citations to a draft research report based on the findings provided.

## Input
You will receive:
1. A **draft report** — a research report WITHOUT inline citations
2. **Worker findings** — structured JSON findings from research workers,
   each containing claim, source_urls (list), source_titles (list),
   and evidence

## Process
1. Read the draft report carefully, sentence by sentence.
2. For each factual claim in the report, find the matching finding
   from the worker outputs.
3. Assign a unique, sequential citation number [1], [2], [3]...
   to each distinct source URL.
4. Insert the citation number IMMEDIATELY after the factual claim
   it supports.
5. At the end of the report, add a "## Sources" section listing
   all cited sources with their numbers.

## Rules
- **One number per URL**: If multiple claims cite the same URL, they
  all use the same [N]. A single finding may have multiple source_urls;
  assign a separate [N] for each distinct URL.
- **Sequential numbering**: Numbers MUST be sequential (1, 2, 3...)
  with no gaps.
- **Every claim cited**: Every factual claim MUST have at least one
  citation. If no source matches, flag it with [citation needed].
- **Every source referenced**: Every entry in the Sources list MUST
  be referenced at least once in the report body.
- **No fabrication**: NEVER invent a source URL. Only use URLs from
  the worker findings.
- **Preserve content**: Do NOT modify the factual content of the
  draft report. Only ADD citation markers and the Sources section.
- **Language**: Output the report in the SAME language as the draft
  report. Do NOT translate content. The "## Sources" heading and
  citation markers [N] remain in English.
- **1:N handling**: When a single finding has multiple source_urls,
  assign a separate [N] for each distinct URL and place them together
  after the claim (e.g., [1][2]).
- **Source title**: In the Sources section, format each entry as:
  [N] Title — URL. Use source_titles from findings when available;
  if unavailable, use the URL's domain as title.

## Self-Check (before output)
Before outputting the final report, verify:
- Every [N] in the body has a matching entry in Sources
- Every entry in Sources is cited at least once
- Numbers are sequential (1, 2, 3...) with no gaps
If any check fails, fix the issue before outputting.

## Fallback
Worker findings are typically in JSON format with fields: claim,
source_urls, source_titles, and evidence. If findings are in plain
text instead of JSON, extract claim-source pairs as best you can
from the free text. Look for URLs mentioned near factual claims.
If a sentence cannot be matched to any finding, mark it as
[citation needed] rather than guessing.

## Output
The complete report with inline citations [1], [2]... and a Sources
section at the end. Nothing else.
"""


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

    from deep_research_agent.agents.deep_agent.citation.citation_middleware import (
        CitationDataMiddleware,
    )
    from deep_research_agent.agents.deep_agent.citation.models import WorkerOutput

    cfg = _load_settings()
    search_tool = _make_internet_search(cfg["tavily_api_key"])

    main_model = overrides.pop("model", f"deepseek:{cfg['planner_model']}")
    worker_model = overrides.pop("worker_model", f"deepseek:{cfg['worker_model']}")

    research_subagent: dict[str, Any] = {
        "name": "research-worker",
        "description": (
            "Conducts focused web research on a specific sub-topic. "
            "Delegate narrow, well-scoped objectives to this agent."
        ),
        "system_prompt": DeepAgentPrompts.WORKER,
        "tools": [search_tool],
        "model": worker_model,
        "response_format": WorkerOutput,
    }

    citation_specialist: dict[str, Any] = {
        "name": "citation-specialist",
        "description": (
            "Adds accurate inline citations [1], [2]... to a draft research "
            "report. Delegate to this agent AFTER all research is complete "
            "and you have written a draft report. Pass ONLY the draft report "
            "as the task description — worker findings are auto-injected."
        ),
        "system_prompt": DeepAgentPrompts.CITATION_SPECIALIST,
        "tools": [],
        "model": worker_model,
    }

    return create_deep_agent(
        model=main_model,
        tools=[search_tool],
        system_prompt=DeepAgentPrompts.SUPERVISOR,
        subagents=[research_subagent, citation_specialist],
        middleware=[CitationDataMiddleware()],
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

    config = {"configurable": {"thread_id": resolved_thread_id}}

    yield {"type": "status", "data": f"Starting research (thread={resolved_thread_id})"}

    final_content = ""

    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
        version="v2",
    ):
        kind = event.get("event", "")

        if kind == "on_tool_start":
            tool_name = event.get("name", "unknown")
            tool_input = event.get("data", {}).get("input", {})
            yield {"type": "tool_start", "data": {"name": tool_name, "input": tool_input}}

        elif kind == "on_tool_end":
            tool_name = event.get("name", "unknown")
            tool_output = event.get("data", {}).get("output", "")
            # Truncate large tool outputs for streaming consumers
            output_str = str(tool_output)
            if len(output_str) > 500:
                output_str = output_str[:500] + "..."
            yield {"type": "tool_end", "data": {"name": tool_name, "output": output_str}}

        elif kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                token = chunk.content
                if isinstance(token, str):
                    final_content += token
                    yield {"type": "token", "data": token}

    # Extract final report from the last message if streaming didn't capture it
    if not final_content:
        try:
            state = await agent.aget_state(config)
            messages = state.values.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    final_content = last_msg.content
        except Exception:
            logger.warning("Could not retrieve final state for report extraction")

    if final_content:
        yield {"type": "final_report", "data": final_content}

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
