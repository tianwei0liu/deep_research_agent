# Resource Limits — Parameter Changelog

Records all changes to the orchestration limit parameters in `config/settings.yaml`.

---

## 2026-05-03: Restore Limit Injection & Raise Defaults

### Background

After refactoring from the custom `OrchestratorGraph` / `Supervisor` to `deepagents.create_deep_agent`, two limit-injection mechanisms were silently lost:

1. **Supervisor** — The old `build_dynamic_limits_prompt(current_step, max_steps)` injected a per-turn `Recursion Budget` message into the LLM context. After refactoring, the Supervisor prompt contained **no resource awareness** — the LLM had no idea how many turns it had or when to converge.
2. **Worker** — The old Worker task description included a `## Limits` section with `max_tool_calls`, `max_turns`, and `max_output_tokens`. After refactoring, **none of these** were present in the Worker system prompt.

Additionally, the Supervisor gained a direct `internet_search` tool (previously only Workers had search), but no corresponding search-call budget was defined.

### Changes

#### New Parameters

| Parameter | Description | Added In |
|-----------|-------------|----------|
| `supervisor_max_turns` | Max Supervisor reasoning turns (replaces `default_recursion_limit`) | This change |
| `supervisor_max_search_calls` | Max direct `internet_search` calls by the Supervisor | This change |

#### Renamed Parameters

| Old Name | New Name | Reason |
|----------|----------|--------|
| `default_recursion_limit` | `supervisor_max_turns` | Explicit `supervisor_` prefix; dropped redundant `default_` |
| `default_max_parallel_workers` | `max_parallel_workers` | Dropped redundant `default_` prefix |
| `default_worker_max_tool_calls` | `worker_max_tool_calls` | Dropped redundant `default_` prefix |
| `default_worker_max_turns` | `worker_max_turns` | Dropped redundant `default_` prefix |
| `default_worker_max_output_tokens` | `worker_max_output_tokens` | Dropped redundant `default_` prefix |

#### Value Changes

| Parameter | Old Value | New Value | Rationale |
|-----------|----------:|----------:|-----------|
| `supervisor_max_turns` | 25 | **100** | Deep/Complex queries often require 20+ delegation cycles; 25 was too tight and caused premature `GraphRecursionError` |
| `supervisor_max_search_calls` | *(N/A)* | **100** | New parameter; Supervisor now has direct search capability for Trivial queries and fact-checks |
| `worker_max_tool_calls` | 40 | **500** | Workers handling broad sub-topics (e.g. multi-hop research) need significantly more search calls |
| `worker_max_turns` | 10 | **100** | Aligned with supervisor turns; prevents premature truncation on complex sub-tasks |
| `max_parallel_workers` | 10 | 10 | *(unchanged)* |
| `worker_max_output_tokens` | 8192 | 8192 | *(unchanged)* |

### How Limits Are Injected

Limits are injected into LLM prompts at **agent build time** via `DeepAgentPrompts.format_*` methods:

```
settings.yaml
    → Settings dataclass (config/settings.py)
    → load_settings() dict (agents/deep_agent/tools.py)
    → build_deep_agent() (agents/deep_agent/deep_agent.py)
        → DeepAgentPrompts.format_supervisor_prompt(max_turns=..., max_search_calls=...)
        → DeepAgentPrompts.format_worker_prompt(max_tool_calls=..., max_turns=...)
```

The formatted prompts are passed to `create_deep_agent(system_prompt=..., subagents=[{system_prompt: ...}])`.

### Supervisor Prompt Injection (§7)

```
## 7. Resource Limits
- You have at most **{supervisor_max_turns} turns** in total.
  Each turn = one reasoning step + tool calls.
- You may call `internet_search` yourself at most **{supervisor_max_search_calls} times**.
  Use these sparingly — only for Trivial queries or quick fact-checks.
- When you have fewer than 3 turns remaining, you MUST abandon any
  pending tasks, synthesize whatever partial findings you have
  gathered, and produce the final report immediately.
```

### Worker Prompt Injection

```
## Resource Limits
- You have at most **{worker_max_tool_calls} tool calls** in total.
- You have at most **{worker_max_turns} turns** (each turn = one
  reasoning step + optional tool calls).
- When you hit a limit, do **not** make another tool call; output
  your current findings immediately and add a one-sentence caveat
  (e.g. "Stopped at limit; N items found.").
```

### Files Modified

| File | Change |
|------|--------|
| `config/settings.yaml` | Renamed keys, updated values |
| `config/settings.py` | Renamed `Settings` fields to match YAML |
| `agents/deep_agent/prompts.py` | Added `## Resource Limits` sections with `{placeholders}`; added `format_supervisor_prompt()` and `format_worker_prompt()` class methods |
| `agents/deep_agent/tools.py` | `load_settings()` now returns limit fields |
| `agents/deep_agent/deep_agent.py` | `build_deep_agent()` reads limits from config and calls `format_*` methods |
| `tests/test_citation/test_agent_integration.py` | Updated `_FAKE_SETTINGS` fixture with new keys/values |
| `README.md` | Updated Configuration table |

---

## 2026-05-04: Budget Enforcement Remediation — Dynamic Injection + Timeout

### Background

After the 2026-05-03 refactoring restored static resource limits to prompts,
a 30+ minute runaway execution was observed. Root cause analysis ([long_running_agent_analysis.md]) identified that static prompt-level limits are insufficient — the LLM relies on **dynamic per-turn budget feedback** to decide when to stop, and this mechanism was lost during the migration.

### Changes

#### Renamed Parameters

| Old Name | New Name | Reason |
|----------|----------|--------|
| `worker_max_tool_calls` | `worker_max_search_calls` | More accurate — only `internet_search` is counted |

#### Value Changes

| Parameter | Old Value | New Value | Rationale |
|-----------|----------:|----------:|-----------|
| `supervisor_max_turns` | 100 | **35** | Anthropic data: Supervisor needs ~30 turns for Deep queries; 35 = 30 + 5 safety margin |
| `supervisor_max_search_calls` | 100 | **10** | Supervisor should delegate, not search; 10 for Trivial + fact-checks |
| `worker_max_search_calls` | 500 | **60** | Anthropic baseline: 10-15 calls/worker. 60 = generous headroom for multi-hop |
| `worker_max_turns` | 100 | **20** | 20 turns is ample for focused sub-tasks; prevents runaway workers |

#### New Parameters

| Parameter | Value | Description |
|-----------|------:|-------------|
| `citation_max_retries` | **5** | Max L1 validation retry cycles for the citation specialist |
| `research_timeout_seconds` | **600** | Wall-clock timeout (seconds); 0 = disabled |

### New Components

#### BudgetTrackingMiddleware (`agents/deep_agent/budget_middleware.py`)

Restores the dynamic budget injection that was lost during refactoring. Registered as a model middleware on the Supervisor agent.

**Behavior**: Before every Supervisor LLM call, counts `AIMessage` instances in history to determine current turn number, then appends a `## ⏱ Budget Status` block to the system message:

```
## ⏱ Budget Status (auto-injected, do NOT ignore)
- Current turn: 22 / 35
- Remaining turns: 13
- Status: ✅ NORMAL — continue research as planned.
```

When `remaining <= 3`:
```
- Status: 🚨 CRITICAL — You MUST stop ALL research immediately...
```

#### Wall-clock Timeout + Checkpoint Resume

`stream_deep_research()` wraps the agent execution in `asyncio.timeout(research_timeout_seconds)`. On timeout:

1. Injects a `TIMEOUT HumanMessage` into the agent via checkpoint resume
2. Supervisor uses existing findings to produce a partial report
3. If resume fails, falls back to raw state extraction
4. All timeout reports carry `{"truncated": True}` metadata

#### Citation Agent Lifecycle Limits

- `CitationDataMiddleware.__init__(max_retries=N)` — configurable retry budget
- Retry descriptions now include `## ⏱ Citation Budget` with `Retry attempt: X/Y`
- Last retry includes `⚠️ This is your LAST chance` urgency signal
- Citation graph constrained with `recursion_limit=15`

### How Dynamic Budget Is Injected

```
settings.yaml
    → Settings dataclass (config/settings.py)
    → load_settings() dict (agents/deep_agent/tools.py)
    → build_deep_agent() (agents/deep_agent/deep_agent.py)
        → BudgetTrackingMiddleware(max_turns=supervisor_max_turns)
            ↪ Every LLM call: append budget status to system message
        → CitationDataMiddleware(max_retries=citation_max_retries)
            ↪ Every retry: inject remaining count into correction prompt
```

### Supervisor Prompt §7 (updated)

```
## 7. Resource Limits
- You have at most **35 turns** in total.
- You may call `internet_search` yourself at most **10 times**.
- Your remaining budget is dynamically updated at the end of the
  system prompt. When you see a CRITICAL budget warning, you MUST
  stop immediately and produce the final report.
```

### Worker Prompt (updated)

```
## Resource Limits
- You have at most **60 search calls** in total.
- You have at most **20 turns**.
```

### Files Modified

| File | Change |
|------|--------|
| `config/settings.yaml` | Calibrated limits, renamed keys, new timeout + citation fields |
| `config/settings.py` | `worker_max_tool_calls` → `worker_max_search_calls`, new fields |
| `agents/deep_agent/budget_middleware.py` | **NEW** — Dynamic per-turn budget injection middleware |
| `agents/deep_agent/prompts.py` | §7 references dynamic injection; worker uses `search_calls` |
| `agents/deep_agent/deep_agent.py` | Registers both middlewares, timeout + checkpoint resume, `recursion_limit=1000` |
| `agents/deep_agent/citation/citation_middleware.py` | Configurable `max_retries`, remaining count injection, docstring fixes |
| `agents/deep_agent/tools.py` | Renamed + new config keys |
| `tests/test_budget_middleware.py` | **NEW** — 14 unit tests |
| `tests/test_citation/test_agent_integration.py` | Updated `_FAKE_SETTINGS` |
| `tests/test_citation/test_citation_middleware.py` | Updated retry test to use explicit `max_retries=1` |

