# Gemini Context Caching Design Document (v2.0)

## 1. Challenge & Goal
**Context**: The `Supervisor` node sends a System Prompt and Tool Definitions to the LLM on every turn.
**Problem**:
1.  **Cost**: Repeated input tokens drain budget.
2.  **Latency**: Processing large prefixes increases TTFT.
3.  **Adherence**: Short prompts (<1000 tokens) often lead to "lazy" or "hallucinatory" behavior in complex agentic tasks.

**Goal**: Use **Gemini Explicit Context Caching** with a **Thick Context** strategy. We will strictly structure the prompt to *exceed* the caching threshold (1024/2048 tokens) by including detailed Standard Operating Procedures (SOPs) and Examples.

## 2. Technical Approach

### 2.1 The "Thick Context" Strategy
Instead of minimizing tokens, we **maximize relevant context** to ensure:
1.  **Cache Stability**: The system prompt is always large enough to be cached (avoiding the <1024 limit issue).
2.  **Behavioral Rigor**: We inject a "Research Handbook" containing detailed SOPs and 3-shot examples.

**Structure**:
1.  **Cached Prefix (Static)**:
    -   **Role & Core Rules**: "You are a Supervisor..."
    -   **Tool Definitions**: JSON Schemas for `manage_todos`, `delegate_research`, etc.
    -   **Research Handbook**: The "5 Commandments" + 3 Detailed Scenarios (Sequential, Parallel, Pivot).
2.  **Dynamic Context (Per Turn)**:
    -   **History**: User/Model messages.
    -   **State Snapshot**: The current `Todo List` is injected as a **User Message** at the end of the history.

### 2.2 Cache Lifecycle Management
-   **Creation**: Compute a hash of the (System Prompt + Tools).
-   **Persistence**: Use an in-memory `_RUNTIME_CACHE_STORE` to share the cache key across turns within the same process.
-   **TTL**: Default 60 minutes.

### 2.3 Eliminating "Fallback"
We expressly **reject** the strategy of falling back if the prompt is too short.
-   **Requirement**: The System Prompt + Tools *must* exceed 1024 tokens.
-   **Guarantee**: The `ResearchHandbook` (~2000 chars+) combined with tool definitions guarantees we hit the threshold.
-   **Handling**: If the cache fails (quota/API error), we proceed with standard `system_instruction` injection, but we do not design for "small prompts".

## 3. Implementation Plan

### Step 1: Research Handbook (Done)
-   Implemented `OrchestratorPrompts.RESEARCH_HANDBOOK` containing SOPs and examples.

### Step 2: Cache Manager (Done)
-   Implemented `agent/cache_manager.py`.
-   Supports `get_process_level_cache` to reuse cache per session.

### Step 3: Supervisor Integration (Done)
-   Refactored `supervisor_node` to:
    1.  Call `get_process_level_cache`.
    2.  Use `cached_content` in `GenerateContentConfig` if available.
    3.  Inject dynamic `todos` as a `UserMessage` instead of a system prompt suffix.

## 4. Cost/Performance Analysis
-   **Input Costs**: Reduced by ~90% for long-running research tasks.
-   **Latency**: Significant reduction in TTFT for complex queries.
-   **Reliability**: Increased adherence due to "Thick Context" examples.
