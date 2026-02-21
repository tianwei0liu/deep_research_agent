# Worker Design — Detailed Specification

This document specifies the design of a **worker** for a multi-agent research system. The worker is a single-purpose agent that executes a focused research task delegated by a lead (orchestrator) agent and returns structured findings. Use this as the implementation blueprint.

---

## 1. Role and Responsibility

### 1.1 Purpose

- **Single responsibility**: Execute exactly one delegated research subtask (e.g. “Find all AI agent companies founded in 2025”, “Summarize regulatory stance on X in the EU”).
- **No orchestration**: Does not spawn other agents, assign work, or make high-level strategy decisions.
- **Bounded scope**: Operates within explicit task boundaries (topic, sources, output format, effort limit) provided by the lead.


- **Output**: Returns a **findings payload** (structured or semi-structured) to the lead—i.e. **condensed** results (e.g. a list, summary, table), not the raw search queries and full search results. The worker acts as an intelligent filter: it uses queries and results internally, then distils and returns only the most important information. Raw queries/results stay in the worker’s internal state (§9); the lead receives only the findings envelope.

### 1.2 What the worker does NOT do

- Delegate to other workers.
- Change the overall research plan.
- Access tools or sources outside the set assigned for this task.
- Exceed the effort budget (max tool calls / max steps) without explicit extension.

---

## 2. Memory Architecture (Consolidated Store)

A **consolidated memory** (e.g. database or cache with an API) serves two distinct purposes and enables both **distributed execution** and **follow-up reuse**.

### 2.1 Plan store: distribution and parallelism

- **What is stored**: The lead’s **research plan** and task slices for workers.
- **Why**: Lead researcher and workers can run on different servers; both access memory via API. The lead writes the plan once (e.g. after planning, before spawning workers) so it survives context truncation (e.g. 200k tokens). Workers get their task either in the spawn payload or, in a later phase, by reading their slice from the store by `task_id`.
- **Build phasing**: (1) Initially the worker receives the full task in the spawn payload (user or lead). (2) Later, when the memory system exists, the lead can write the plan to the store and pass `task_id` (or the worker reads its slice by ID). (3) Finally the worker runs as a long-lived service for async multi-agent use.

### 2.2 Conversation / report store: follow-up questions

- **What is stored**: Past **user conversations**: user query + research report (and optionally citations, embeddings for retrieval).
- **Why**: On follow-up questions, retrieve relevant past (query, report) pairs (e.g. similarity search), inject into context, and avoid repeating research or extend from prior results.

### 2.3 Summary

| Store | Contents | Used by | Purpose |
|-------|----------|---------|---------|
| **Plan store** | Research plan, task slices for workers | Lead (write); workers or spawner (read) | Distribution, parallelism, context truncation safety |
| **Conversation / report store** | (user_query, research_report) and optionally embeddings | Lead or gateway (retrieve before running research) | Follow-up reuse, avoid repeating research |

---

## 3. Inputs (from Lead Agent)

The lead agent must supply the following when creating a worker. Treat these as the **required contract** for spawning a worker.

### 3.1 Task description (required)

| Field | Type | Description |
|-------|------|--------------|
| `objective` | string | One or two sentences: what the worker must find or produce. |
| `output_format` | string or schema | How to return results: e.g. “JSON list of {name, url, date}”, “Markdown table”, “Bullet summary”, or a JSON schema. |
| `sources_and_tools` | string or list | Which tools/sources to use (e.g. “web search only”, “only use the `company_db` tool”, “prefer academic sources”). |
| `task_boundaries` | string | What to avoid or when to stop (e.g. “Do not cover EU”, “Stop after 10 items”). **Advisory only**—the runner enforces only `max_tool_calls` and `max_turns`. |
| `context_snippet` | string (optional) | Short relevant context from the lead (e.g. “User asked about AI agents in 2025; you are responsible only for the company list.”). |

### 3.2 Effort and limits (required)

| Field | Type | Description |
|-------|------|--------------|
| `max_tool_calls` | integer | Maximum number of tool invocations (e.g. 10–15 for a typical subtask). Worker must stop when reached. |
| `max_turns` | integer (optional) | Maximum reasoning+action “turns” (each turn = one reasoning block + one or more tool calls). If omitted, no turn limit (only `max_tool_calls` applies). |

### 3.3 Tool set (required)

- **Lead supplies**: A list of **tool names** (e.g. `["tavily_search"]` for current scope). The lead has access to tool descriptions elsewhere for delegation.
- **Runner**: Resolves each name to the full definition (name, description, parameters / JSON Schema) and passes only those tools to the worker. The worker may only call tools in this set.

### 3.4 Example payload from lead to worker

```json
{
  "task": {
    "objective": "Find the top 5 AI agent companies founded or heavily marketed in 2025, with name and official website.",
    "output_format": "JSON array of objects: {\"name\": string, \"url\": string, \"founded_or_2025_focus\": string}",
    "sources_and_tools": "Use only tavily_search. Prefer company websites and news from 2025.",
    "task_boundaries": "US and Europe only. Stop after 5 companies or 8 search calls. Do not include pure LLM APIs (e.g. OpenAI); focus on agentic/AI agent products."
  },
  "limits": {
    "max_tool_calls": 10,
    "max_turns": 5
  },
  "tools": ["tavily_search"],
  "context_snippet": "User question: 'Who are the main AI agent companies in 2025?' Your job is only the company list."
}
```

---

## 4. Outputs (to Lead Agent)

### 4.1 Success: findings payload

The worker must return a **condensed, structured or semi-structured** result that the lead can merge with other workers’ results—not raw queries and full search results (see §1.1). Two parts:

1. **Findings** (required): The actual answer in the requested `output_format` (e.g. JSON, markdown table, bullets).
2. **Metadata** (recommended): Short summary of what was done and any caveats.
3. **Sources** (optional, for citations): List of sources the worker used (e.g. `[{ "url", "title", "snippet" }]`) so the lead or CitationAgent can attribute claims.

Suggested envelope (you can adapt to your lead’s expectations):

```json
{
  "status": "success",
  "findings": { ... },
  "metadata": {
    "tool_calls_used": 7,
    "sources_used": ["tavily_search (7)"],
    "caveats": "Stopped at 5 companies; some had no clear 'founded 2025' so used '2025 focus'."
  },
  "sources": [{ "url": "https://...", "title": "...", "snippet": "..." }]
}
```

- `findings` should conform to the task’s `output_format` (or explain in `metadata.caveats`). Empty or “no results found” is valid as `status: "success"` with empty/minimal findings.

**Summarization: same ReAct process, not a separate LLM call.** The condensed findings are produced **in the same loop with the same prompt**: when the model has enough information (or hits a limit), it outputs the final answer in one turn (no tool call). There is no separate “summarization” API call or different prompt. The system prompt and task prompt already tell the model to “output your final answer in the exact format requested” (§8.1). Using one process keeps context (the model has just seen all tool results), avoids extra latency and cost, and matches the typical worker design (e.g. Anthropic’s “returns findings”). Optionally, you could add a **separate summarization call** (e.g. a second LLM call with a dedicated “summarize this into format X” prompt) if you need a different model, stricter structure, or post-processing; the design does not require it.

**Citations: how to preserve.** Citations link claims in the findings to sources (URLs, snippets). Options: (1) **Bake into output format**—have the lead request an `output_format` that includes a source per item (e.g. `{"name", "url", "source_url"}` or “for each finding, cite the URL”). The worker then emits findings with source fields in the same ReAct turn. (2) **Return a sources list with findings**—the envelope can include an optional `sources` array (e.g. `[{ "url", "title", "snippet" }]`) that the worker used; the lead or a later CitationAgent can match claims to these. (3) **CitationAgent after synthesis** (Anthropic’s approach)—after the lead synthesizes the full report, a separate CitationAgent processes the report and the underlying documents to attach citations. For the worker itself, (1) and (2) keep citations traceable without a second LLM call; (3) is at the lead/final-report level and can be specified in the lead-agent design.

### 4.2 Failure or partial success

- **Hard failure** (e.g. all tools errored, task impossible): `status: "failure"`, `findings: null`, `metadata.reason` and optionally `metadata.last_error`.
- **Partial success** (e.g. hit `max_tool_calls` before complete): `status: "partial"`, `findings` with what was found, `metadata.caveats` explaining the gap.

The lead can then decide to spawn another worker, relax boundaries, or surface the partial result.

---

## 5. Internal Loop: ReAct / “Interleaved Thinking”

Single run-to-completion loop: reason → act → observe → reason → … until done or limit reached.

### 5.1 Turn structure (one turn = one cycle)

1. **Reason** (before every action): First turn—plan how to achieve the objective. Later turns—summarize last result(s), assess gaps, decide next action or finish.
2. **Act**: Call one or more tools (parallel if the API allows). **Runner**: execute at most `max_tool_calls - tool_calls_used` in this batch; reject excess and return a synthetic observation so the model can synthesize.
3. **Observe**: Receive tool results (and errors).
4. **Repeat** until: (a) the worker has enough and outputs a final answer, (b) `max_tool_calls` or `max_turns` is reached, or (c) hard failure.

### 5.2 Implementing “interleaved thinking”

- **Claude**: Use extended + interleaved thinking; preserve `thinking` blocks when sending tool results back.
- **Gemini / others**: Require reasoning in a structured block (e.g. `<reasoning>...</reasoning>`) before each tool call and after each result; parse and keep it in history (ReAct pattern).

### 5.3 Rules inside the loop

- **Reason before acting**: No tool call without a preceding reasoning step.
- **Enforce limits**: After each turn, if `tool_calls_used >= max_tool_calls` or `turns >= max_turns`, inject: “You have reached the limit. Output your final answer now in the requested format.” Accept only a final answer (no further tool calls).
- **Assigned tools only**: Reject or ignore tool calls not in the assigned list.
- **One objective**: Do not expand scope.

### 5.4 Stop vs continue (no tool call = final answer)

Each turn the worker should either (a) make at least one tool call (e.g. tavily_search) to gather more information, or (b) output the final answer if it has enough. **If the model outputs text with no tool call, treat that as the final answer**: parse it into the envelope and exit. Do not re-prompt. State this explicitly in the system prompt (§8.1).

---

## 6. Models (Google Gemini)

This design uses **Google Gemini**. Rationale: access where alternatives may be restricted, multimodal input, function calling, and long context.

### 6.1 Model versions

Use **current Gemini API model IDs** that support `generateContent` and **function calling**. Older IDs (e.g. `gemini-1.5-flash`) may return 404. Check [Gemini API docs](https://ai.google.dev/gemini-api/docs/models) for the latest list.

**Stable (recommended for production):**

| Model ID | Description |
|----------|-------------|
| `gemini-2.5-flash` | Balanced speed and scale; function calling, 1M context. Best for worker workers. |
| `gemini-2.5-pro` | Strong reasoning; function calling, 1M context. Best for lead/orchestrator. |
| `gemini-2.5-flash-lite` | Fast, cost-efficient; function calling. |

**Preview (newer, may change):**

| Model ID | Description |
|----------|-------------|
| `gemini-3-flash-preview` | Latest balanced model (speed + scale). |
| `gemini-3-pro-preview` | Most capable; best for complex orchestration. |

**Role-to-model mapping:**

| Role | Recommended model(s) |
|------|----------------------|
| **Lead (orchestrator)** | `gemini-2.5-pro` or `gemini-3-pro-preview` |
| **Worker** | `gemini-2.5-flash` (default) or `gemini-3-flash-preview` (Flash: cheaper/faster for parallel workers) |

### 6.2 Requirements for the model

- **Function calling**: Required for tool use (e.g. search). Gemini uses the standard function-declaration and `functionCall` response pattern.
- **Long context**: Worker conversations grow with tool results; prefer models with at least ~128K–1M token context where available.
- **Multimodal**: If research involves PDFs, screenshots, or charts, use a model that accepts image/document parts (e.g. Gemini 2.5 Pro/Flash or 3.x). Pass fetched PDFs or images as inline data or file references per the Gemini API.

### 6.3 "Interleaved thinking" on Gemini

Gemini does not expose a first-class "thinking block" like Claude. Use **ReAct-style prompt engineering** (§5.2): require the model to output reasoning in a structured block (e.g. `<reasoning>...</reasoning>`) before each tool call and after each tool result; parse and preserve these blocks in the conversation so the model reasons after every observation.

---

## 7. Tools (Specified)

**Current scope: only `tavily_search`.** Other tools (fetch_url, memory, etc.) can be added later. The lead assigns a subset per task; the runner resolves names to definitions.

### 7.1 Search tool (in use)

#### tavily_search

- **Purpose**: Web search for current, factual information. Tavily is an AI-oriented search API (good relevance and latency for agent use).
- **Backend**: [Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search) (POST `/search`). Authenticate with `Authorization: Bearer <TAVILY_API_KEY>` (e.g. from env).
- **Parameters** (JSON Schema for the model):

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | — | The search query. Prefer specific, focused queries; start broad then narrow if needed. |
| `search_depth` | string | no | `"basic"` | `basic` (balanced), `advanced` (highest relevance, 2 credits), `fast`, `ultra-fast`. Use `advanced` when high precision matters. |
| `max_results` | integer | no | 5 | Number of results (1–20). |
| `topic` | string | no | `"general"` | `general`, `news`, or `finance`. Use `news` for recent events. |
| `time_range` | string | no | — | Filter by recency: `day`, `week`, `month`, `year` (or `d`, `w`, `m`, `y`). |
| `include_answer` | boolean/string | no | false | If true/`basic`/`advanced`, response includes an LLM-generated answer (can save a round; prefer grounding in `results[]` when factuality matters). |

- **Response**: `results[]` with `title`, `url`, `content` (snippet/summary); optionally `answer` if requested. Return a concise summary to the model (e.g. top 5–10 results with title, url, content) to save context.

### 7.2 Optional tools (add later)

#### fetch_url (web fetch)

- **Purpose**: Retrieve full content of a URL (for verification or deep read). Add when needed.
- **Backend**: HTTP GET + HTML-to-text or markdown extraction. Parameters: `url` (required), `max_length` (optional). On failure, return error so the model can retry or skip.

#### retrieve_similar_research (search_memory)

- **Purpose**: Retrieve past **user** conversations (user query + research report) relevant to the current query for follow-up reuse. **Note**: The store holds past user sessions, not other workers’ findings from the current run.
- **Backend**: Query the conversation/report store (§2.2); return top-k (query, report) pairs.
- **Who calls it**: Usually the lead (or gateway) before spawning. Optionally expose to workers for tasks like “extend or update prior report X.”
- **Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | yes | Current user question (or task summary) for similarity search. |
| `top_k` | integer | no | 3 | Number of past (query, report) pairs to return. |

- **Response**: List of `{ user_query, report_summary_or_excerpt, report_id, created_at }` (or similar). Keep excerpts bounded (e.g. 500–1000 chars) so context does not explode.

- **Document / PDF, code execution, domain APIs**: Add as needed (e.g. `load_document`, sandboxed `python`, Semantic Scholar). Not required for initial scope.

### 7.3 Tool assignment and schema

- **Lead** passes tool **names** (e.g. `["tavily_search"]` for now); runner resolves to full definitions.
- System prompt: “You may only use the following tools: [list]. Do not use any other tool.”
- **Tool result handling**: Append result as observation; on error, present error to the model. Truncate long results (e.g. max 12K chars) with “[truncated]”.

---

## 8. Prompt Structure

### 8.1 System prompt (fixed for all workers of this type)

- **Role**: “You are a research worker. Execute exactly one delegated task and return findings in the requested format. You do not delegate or change the task.”
- **Stop vs continue**: “Each turn, either call a tool (e.g. search) to gather more information or, if you have enough, output your final answer in the requested format. If you output an answer with no tool call, the run stops and that output is your findings.”
- **Loop**: “Before every tool call, reason about your goal; after every tool result, reason about what you learned and what to do next.”
- **Format**: “Output your final answer in the exact format requested. If you cannot complete fully, return what you have and explain in metadata/caveats. If the task asks for citations or sources, include source URLs in the findings and/or a sources list (url, title, snippet) so claims can be attributed.”
- **Limits**: “Stop after at most [max_tool_calls] tool calls” and, if max_turns is set, “and [max_turns] turns.” When you hit a limit, output partial findings with a caveat. (If max_turns is omitted, omit “and [max_turns] turns” from this sentence.)

(For tag-based reasoning on non-Claude models: “Put reasoning inside <reasoning>...</reasoning>; tool calls and final answer outside.”)

### 8.2 Task prompt (injected per worker)

This is the **task description** from the lead, formatted so the model sees it once at the start. Include:

- **Objective**: [task.objective]
- **Output format**: [task.output_format]
- **Sources and tools**: [task.sources_and_tools]
- **Boundaries**: [task.task_boundaries]
- **Context**: [task.context_snippet] (if provided)
- **Limits**: max_tool_calls = [limits.max_tool_calls]; if max_turns is set, add max_turns = [limits.max_turns] (otherwise omit).

Example:

```
## Your task

**Objective:** Find the top 5 AI agent companies founded or heavily marketed in 2025, with name and official website.

**Output format:** JSON array of objects: {"name": string, "url": string, "founded_or_2025_focus": string}

**Sources and tools:** Use only tavily_search. Prefer company websites and news from 2025.

**Boundaries:** US and Europe only. Stop after 5 companies or 8 search calls. Do not include pure LLM APIs; focus on agentic/AI agent products.

**Context:** User question: "Who are the main AI agent companies in 2025?" Your job is only the company list.

**Limits:** Maximum 10 tool calls, 5 turns. When you hit a limit, return what you have and note it in caveats.
```

### 8.3 Per-turn context (optional)

If the lead has shared a **memory** or **plan** reference (e.g. “Research plan id: X”), you can add one line: “Refer to plan X for overall context; your scope is only the task above.” Do not dump the full plan into the worker context to avoid noise and token bloat.

---

## 9. State and Context

### 9.1 Worker state (what the implementation must track)

| State item | Type | Purpose |
|------------|------|---------|
| `task` | object | Full task + limits from lead (see §3). |
| `reasoning_steps` | list | ReAct / interleaved thought: reasoning block before and after each tool use (e.g. `<reasoning>...</reasoning>` or Claude thinking). |
| `search_queries_and_results` | list | For each tool invocation: the **query** (or tool params) and the **result**. With `reasoning_steps`, this *is* the conversation history. |
| `tool_calls_used` | integer | Increment on each tool invocation; enforce `max_tool_calls`. |
| `turns` | integer | Increment after each reason→act→observe cycle; enforce `max_turns`. |
| `status` | enum | `running` \| `success` \| `partial` \| `failure`. |
| `findings` | object/string | Filled when status is success/partial; null on failure. |
| `metadata` | object | Caveats, sources_used, last_error, etc. |

**Conversation history = reasoning + (query, result) pairs.** The model’s context is built from: system prompt + task prompt + for each turn a reasoning step, then tool call(s) (e.g. search query), then tool result(s). You can store either (a) a raw `messages` list in API shape, or (b) **reasoning_steps** and **search_queries_and_results** and reconstruct the message list when calling the model. Storing queries and results explicitly supports truncation (e.g. keep most relevant results), logging, and debugging. **Note:** These are internal to the worker. The worker returns only **condensed findings** to the lead (§4), not this raw state (per Anthropic’s design: workers “condense the most important tokens” for the lead).

**Recommended schemas for search state** (inspired by [deep-research-agent state](https://github.com/tarun7r/deep-research-agent/blob/main/src/state.py)):

- **Search query (per tool invocation)**  
  - `query` (string, required): The search query text.  
  - `purpose` (string, optional): Why this query is being made (e.g. “find company names”; helps debugging and truncation).  
  - `completed` (boolean, default true when results are in): Whether this query has been executed and results attached.

- **Search result (per result item, e.g. one Tavily hit)**  
  - `query` (string): The original query this result came from (for tracing and grouping).  
  - `title` (string): Result title.  
  - `url` (string): Result URL (for citations).  
  - `snippet` (string): Result snippet/summary.  
  - `content` (string, optional): Full scraped content if fetched (e.g. via fetch_url later).

One tool call produces one **query record** and a list of **result records**. Store them so you can truncate by dropping older (query, results) pairs or by keeping only the most relevant results; the `query` on each result links it back to the query for attribution. The same result shape (url, title, snippet) can be used to populate the optional **sources** field in the output envelope (§4.1) for citations.

**Example structure for one turn:**  
- `reasoning_steps[i]`: “I need to find companies; I’ll search for X.”  
- `search_queries_and_results[i]`:  
  - `query_record`: `{ "query": "AI agent companies 2025", "purpose": "find company names", "completed": true }`  
  - `results`: `[ { "query": "AI agent companies 2025", "title": "...", "url": "https://...", "snippet": "...", "content": null } ]`

### 9.2 Context window management

- **No cross-worker context**: This worker does not see other workers’ messages or findings.
- **Initial implementation**: Context limit can be deferred; worker context is typically manageable for a single task.
- **When integrating into the multi-agent system**: With explicit **search_queries_and_results**, keep the most relevant (query, result) entries (e.g. by similarity to objective or recency); drop or truncate the rest and add “[N older results omitted]” if needed. Avoid asking the model to summarize mid-run. Document max tokens / policy.
- **Thinking/reasoning**: Preserve each reasoning step in `reasoning_steps` (or in the reconstructed messages) when sending tool results back to the model.

---

## 10. Error Handling and Resilience

### 10.1 Tool errors

- **Transient** (rate limit, timeout): Retry once with backoff; if still failing, present error to the model and let it report partial/failure with `metadata.last_error`.
- **Permanent** (invalid params, auth failure): Do not retry; present error to the model, set status `failure` or `partial`.

### 10.2 Model output errors

- **Invalid tool call** (wrong name or schema): Return synthetic observation “Tool X failed: invalid arguments (expected …).” Let the model retry or stop.
- **No tool call** (model outputs only text): **Treat as final answer**: parse and return in envelope. If parsing fails, return status `partial` with raw output and `metadata.caveats`. Do not re-prompt (see §5.4).
- **Final answer not in requested format**: Return raw output with `metadata.caveats: "Output may not match requested format"`; optionally validate and set `status: "partial"`.

### 10.3 Timeouts and resource limits

- **Wall-clock timeout**: If the run exceeds T seconds (e.g. 120), stop after the current turn, set status `partial` or `failure`, return existing findings + metadata.
- **Token budget**: If context approaches the limit, stop and return partial + caveat; do not overflow.

---

## 11. API / Interface Contract

### 11.1 Spawn (input)

The **worker runner** (called by user during development, later by lead or orchestrator) accepts:

- `task`: object (objective, output_format, sources_and_tools, task_boundaries, context_snippet).
- `limits`: object (max_tool_calls, max_turns).
- `tools`: list of **tool names** (e.g. `["tavily_search"]` for current scope). Runner resolves to full definitions; lead has tool descriptions for delegation.
- `model` (optional): model id (default per §6).
- `thinking_config` (optional): e.g. `use_reasoning_tags: true` for Gemini.
- `run_id` (optional): client-provided id for tracing and deduplication.

### 11.2 Return (output)

- **Synchronous**: Return a single object: `{ status, findings, metadata }` when the worker finishes (success, partial, or failure).
- **Async**: If the system is async, the same object can be returned via a callback or future; the lead waits for all workers’ results before synthesizing.

### 11.3 Identity and tracing

- Use a unique **run_id** (or task_id) per run for logging and deduplication (e.g. avoid re-running the same spawn on client retry). Not for deterministic replay—runs are nondeterministic.

### 11.4 Deployment: Worker as an API service (recommended)

**Yes—refactoring the worker into an API service that the lead invokes via a tool is the right architecture for a multi-agent system.**

- **Lead invokes workers via a tool**: The lead has a tool (e.g. `run_worker` or `spawn_research_worker`) whose implementation is an HTTP (or gRPC) call to the worker API. The tool’s parameters are the spawn contract (§11.1); the tool’s return value is the envelope (§11.2). The lead does not need to know that the work runs on another service.
- **Benefits**:
  - **Decoupling**: Lead researcher and workers can run in different processes, languages, or nodes. The contract (task, limits, tools → status, findings, metadata) is the only interface.
  - **Scalability**: Worker API can be scaled independently (replicas, load balancer). Multiple workers can run in parallel; the lead just calls the tool once per subtask.
  - **Fits shared memory (§2)**: Lead researcher and worker services both access the plan store and conversation store via APIs; no shared process required.
  - **Easier evolution**: You can change worker implementation (model, tools, runtime) or move it to serverless without changing the lead, as long as the API contract is stable.

**Implementation outline**:

1. **Worker service**: Expose a single endpoint, e.g. `POST /v1/worker/run`, that accepts the spawn payload (§11.1), runs the worker loop to completion (or until timeout), and returns `{ status, findings, metadata }` (§11.2). Optionally support `run_id` for idempotency or tracing.
2. **Lead’s tool**: Define a function/tool the lead can call, e.g. `run_worker(task, limits, tools)`. The lead’s runtime (or a small adapter) translates that into a request to the worker API and passes the response back to the lead as the tool result.
3. **Parallelism**: When the lead spawns N workers, the orchestrator issues N tool calls (or one batch call); the client can send N requests to the worker API in parallel and collect results. The lead then synthesizes once all tool results are in.

You can implement the worker in-process first (same codebase as the lead) and **refactor later**: keep the same spawn/return contract, and replace the in-process runner with an HTTP client that calls the worker API. The design in this document already defines that contract, so the refactor is straightforward.

---

## 12. Summary Checklist for Implementation

- [ ] **Input validation**: Reject spawn if `task.objective`, `task.output_format`, `limits.max_tool_calls`, or `tools` is missing.
- [ ] **Prompt assembly**: System prompt + task prompt + only assigned tools (runner resolves tool names to definitions).
- [ ] **Loop**: Reason → act → observe; enforce max_tool_calls (cap batch size to remaining budget) and max_turns; on limit, inject “output final answer now” and accept only final answer. When model returns no tool call, treat as final answer and parse (§5.4, §10.2).
- [ ] **Thinking**: Gemini: ReAct-style tag-based reasoning (§5.2, §6.3); Claude: interleaved thinking.
- [ ] **Tools**: Implement `tavily_search` (current scope); add `fetch_url`, `retrieve_similar_research` later. See §7.
- [ ] **Output parsing**: Define how the final answer is identified (e.g. last assistant message when no tool call). Extract findings and normalize to `{ status, findings, metadata }`.
- [ ] **Errors**: Tool errors and timeouts → partial/failure + metadata. On unhandled exception, surface `status: "failure"` with `metadata.last_error`.
- [ ] **Observability**: Log run_id, task summary, tool_calls_used, turns, status, latency.
- [ ] **Deployment** (later): Expose worker as API; lead invokes via `run_worker`-style tool. See §11.4.

This design is sufficient to implement a single worker (worker) that can be driven by a lead agent in a multi-agent research system. The lead agent design (orchestrator, delegation, citation) can be specified in a separate document.
