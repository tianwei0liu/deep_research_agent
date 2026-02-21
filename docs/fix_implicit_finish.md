# Fix: Supervisor Implicit Finish — Implementation Plan

## Problem Statement

In the benchmark run `run_20260220_185718`, **4 out of 8 cases** (`factual_002`, `temporal_001`,
`comparative_001`, `multi_hop_001`) returned an **empty `final_report`** despite the supervisor
having completed all research tasks successfully. The grader scored them 0.0 across correctness,
completeness, and citations.

### Root Cause

The supervisor LLM writes a final report as **plain text content** in its response instead of
calling the `finish()` tool. When `route_supervisor` in `graph.py` sees zero tool calls, it
routes to `END`, which **skips `compose_node` entirely** — the only node that populates
`state["final_report"]`. The supervisor's text (which IS the report) is silently discarded.

**Log evidence** — every failing case ends with:

```
Supervisor Response: <report text>... Tool Calls: 0
```

### Defense-in-Depth Strategy

| Layer          | Fix   | File                               | Purpose                                                             |
|----------------|-------|------------------------------------|---------------------------------------------------------------------|
| **Prevention** | Fix 2 | `agents/orchestrator/prompts.py`   | Stop the LLM from producing text-only responses in the first place  |
| **Recovery**   | Fix 1 | `agents/orchestrator/graph.py`     | Route to Composer even without `finish`, so findings are synthesized |

---

## Fix 1: Graph-Level Implicit Finish Detection

**File**: `agents/orchestrator/graph.py`

### Design Decision

When the supervisor returns text without tool calls, we have two options:

- **Option A — Set `final_report` directly from the supervisor's text.** Simpler, but the
  supervisor's inline text is unstructured — it was not generated with the Composer's formatting
  guidelines, citation consolidation, or source list requirements.

- **Option B (chosen) — Route to `compose_node`.** The Composer generates a proper report from
  the completed task findings already stored in `state["todos"]`. It uses a dedicated model
  (`settings.composer_model`), a specialized prompt (`build_composer_prompt`), and produces
  structured reports with headers, inline citations, and a source list.

Option B is superior because the task findings in `state["todos"]` are **always populated before
the supervisor's final turn** — workers complete inside `execute_tools`, which runs before the
supervisor is invoked again. The Composer does not depend on the `finish` tool having been
called; it only needs the state.

### Routing Rules (New)

Evaluated in order:

1. No messages at all → `END`
2. `finish` tool called → `compose_node`
3. Other tool calls present → `tool_execution_node`
4. No tool calls, but completed/partial/failed tasks exist → `compose_node` *(implicit finish)*
5. No tool calls, no actionable tasks → `END`

Rule 4 is the new addition. The condition checks for any task with status in
`{COMPLETED, PARTIAL, FAILED}` — meaning research work was done and there are findings worth
synthesizing. Only `PENDING` tasks (no work done yet) are excluded.

### Exact Changes

**Add `logging` import** to the top-level imports (around line 6):

```python
import logging
```

**Add logger to `__init__`** (replace lines 17–19):

```python
def __init__(self):
    self.logger = logging.getLogger(__name__)
    self.supervisor = Supervisor()
    self.composer = Composer()
```

**Replace `route_supervisor` method** (lines 21–46) with:

```python
def route_supervisor(self, state: OrchestratorState) -> Literal["tool_execution_node", "compose_node", "__end__"]:
    """Route based on the supervisor's output.

    Routing rules (evaluated in order):
        1. No messages at all           -> END
        2. 'finish' tool called         -> compose_node
        3. Other tool calls present     -> tool_execution_node
        4. No tool calls, but completed
           research tasks exist         -> compose_node  (implicit finish)
        5. No tool calls, no tasks      -> END
    """
    messages = state.get("messages", [])
    if not messages:
        return END

    last_msg = messages[-1]
    has_tool_calls = hasattr(last_msg, "tool_calls") and last_msg.tool_calls

    if has_tool_calls:
        for tool_call in last_msg.tool_calls:
            if tool_call["name"] in ("finish", "FinishInput"):
                return "compose_node"
        return "tool_execution_node"

    # No tool calls — check if research was performed (implicit finish).
    todos = state.get("todos", [])
    from deep_research_agent.agents.orchestrator.schemas import TaskStatus
    actionable_statuses = {TaskStatus.COMPLETED, TaskStatus.PARTIAL, TaskStatus.FAILED}
    has_findings = any(t.status in actionable_statuses for t in todos)

    if has_findings:
        self.logger.warning(
            "Implicit finish detected: supervisor returned text without "
            "calling finish(). Routing to compose_node. "
            "Text preview: %.100s",
            getattr(last_msg, "content", "") or "",
        )
        return "compose_node"

    return END
```

### Edge Cases

| Scenario | Todos State | Route | Correct? |
|----------|-------------|-------|----------|
| First turn, LLM returns text with no tool calls | Empty | `END` | Yes — no work was done, nothing to compose |
| Mid-research glitch, LLM returns text without tools | Has COMPLETED tasks | `compose_node` | Acceptable — Composer produces a partial report from available findings rather than losing everything |
| All tasks complete, LLM writes report as text | All COMPLETED | `compose_node` | Yes — this is the exact bug scenario being fixed |
| Supervisor returns error text (e.g. "Error: ...") | Has COMPLETED tasks | `compose_node` | Safe — Composer ignores the error text and synthesizes from task findings |

---

## Fix 2: Supervisor Prompt Hardening

**File**: `agents/orchestrator/prompts.py`

### Problem

Two weak spots in the current system prompt allow the LLM to bypass `finish`:

1. **"Finish" instruction (lines 204–206)** — merely says "call `finish(reason=...)`" without
   explaining what happens if the supervisor does not comply.

2. **"Output" section (lines 208–209)** — says *"Do not output conversational text unless it is
   to explain your reasoning before a tool call."* This is a loophole: the LLM generates an
   entire report as "reasoning" and then emits no tool call at all.

### Exact Changes

**Bump the cache buster** on line 183 from `3` to `4`:

```python
# Cache buster: 4
```

This is necessary because the system prompt is cached via `get_process_level_cache`. A content
change without a cache buster increment would cause the next run to reuse the stale prompt.

**Replace the "Research Loop" step 4 and "Output" section** (lines 193–209 of the return
string inside `build_supervisor_prompt`) with:

```
## The Research Loop
1. **Analyze**: Review the User Query and the current state of your Todo List.
2.  **Plan**:
    - If the Todo List is empty, break down the user's query into initial research tasks. Use `add_task(objective=..., description=...)`
    - If a task is finished, review its findings. Do you need more info? Add new tasks.
    - If a task is failed, retry it with different instructions or delete it using `remove_task(task_id=...)`.
3. **Delegate**:
    - Identify tasks that are `pending`.
    - If multiple pending tasks have NO unmet dependencies, delegate them ALL in a single turn — they will execute in parallel.
    - If a pending task depends on unfinished tasks, skip it until those dependencies are completed.
    - The system automatically passes completed dependency results to workers; you do not need to include dependency context manually.
    - Use `delegate_research(task_id, objective, instructions)` to assign them to a worker.
4. **Finish**:
    - When you have sufficient information to answer the user's query comprehensively, you **MUST** call `finish(reason="...")`.
    - The `finish` tool is the **ONLY** mechanism that triggers report generation. If you do not call it, **all research will be lost**.
    - Do not finish if there are `pending` tasks that are critical to the answer.

## CRITICAL OUTPUT RULES
- **Every response MUST contain at least one tool call.** There are no exceptions.
- You may include brief reasoning text *before* your tool calls to explain your plan. This is encouraged.
- **NEVER write a report, summary, or answer in your text response.** Report generation is handled by a downstream Composer after you call `finish`. Your job is ONLY to coordinate — not to author the final report.
- If all research is complete and you are ready to conclude, your response must be: reasoning text (optional) + `finish(reason="Research complete.")`. Nothing else.
```

### What Changed and Why

| Section | Before | After | Rationale |
|---------|--------|-------|-----------|
| Step 4 "Finish" | `call finish(reason="...")` | `finish` is the **ONLY** mechanism that triggers report generation. If you do not call it, **all research will be lost**. | Consequence framing. LLMs respond better to understanding *why* a constraint exists than to bare imperatives. |
| "Output" heading | `## Output` | `## CRITICAL OUTPUT RULES` | Capitalized heading + "CRITICAL" keyword signals priority to the LLM's attention mechanism. |
| Output body | "Do not output conversational text unless to explain reasoning before a tool call" | **NEVER write a report, summary, or answer in your text response.** Report generation is handled by a downstream Composer. | Closes the loophole explicitly. The old phrasing allowed "reasoning" that could be a full report. The new phrasing forbids reports/summaries/answers and explains *why* (a downstream Composer handles it). |
| New rule | *(absent)* | "If all research is complete... your response must be: reasoning text (optional) + `finish(reason=...)`. Nothing else." | Prescribes the exact shape of the final response, leaving no ambiguity. |

### Note on Unchanged Sections

The Research Handbook (`RESEARCH_HANDBOOK` class attribute, lines 11–176) is **not modified**.
The decomposition patterns, effort scaling rules, delegation strategy, and interaction examples
remain exactly as-is. Only the supervisor role instructions and output rules within
`build_supervisor_prompt()` are changed.

---

## Execution Order

1. **Fix 2 first** — because it changes the cached system prompt and requires a cache buster
   increment. Doing it first ensures the next test run picks up the new prompt from a fresh
   cache entry.
2. **Fix 1 second** — the structural recovery mechanism in the graph router.

## Files Changed

| File | Change Summary |
|------|----------------|
| `agents/orchestrator/graph.py` | Add `logging` import; add `self.logger` to `__init__`; rewrite `route_supervisor` with implicit finish detection |
| `agents/orchestrator/prompts.py` | Rewrite "Finish" step and "Output" section in supervisor prompt; bump cache buster from 3 → 4 |

## Files NOT Changed

| File | Reason |
|------|--------|
| `agents/orchestrator/supervisor.py` | The supervisor's `run()` and `execute_tools()` work correctly. The bug is in routing and prompting, not execution. |
| `agents/orchestrator/composer.py` | The Composer already handles invocation with completed tasks. It does not depend on `finish` having been called. |
| `agents/orchestrator/state.py` | No schema changes needed. `final_report: Optional[str]` and `todos` are sufficient. |
| `tools/control.py` | The `finish` tool itself is fine. The issue is the LLM not calling it, not the tool being broken. |
| `benchmarks/runner.py` | Runner is the benchmark harness and should not contain production-logic workarounds. |

## Risk Assessment

| Fix | Risk | Mitigation |
|-----|------|------------|
| Fix 1 | Premature routing to `compose_node` if the supervisor emits text mid-research before all tasks complete | The condition requires at least one COMPLETED/PARTIAL/FAILED task. Log analysis confirms the supervisor **never** emits text without tool calls mid-research — it only does so when it decides to "finish." Even if triggered prematurely, the Composer produces a partial report (better than an empty one). |
| Fix 2 | Prompt change could alter supervisor behavior for currently-passing cases | The new prompt only restricts a behavior that was already forbidden ("do not output conversational text"). Planning, delegation, and research handbook instructions are untouched. The two currently-passing cases (`factual_001`, `comparative_002`) both use the `finish` tool correctly and will not be affected. |
