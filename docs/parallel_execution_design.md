# Parallel Execution Design Document

> [!NOTE]
> **Status**: Refactored (v2.0)
> **Author**: Distinguished Researcher / Principal Engineer
> **Date**: 2026-02-19
> **Supersedes**: v1.0 (Naive Parallel Loop)

## 1. Challenge & Goal
**Context**: The `Supervisor` node currently executes tool calls *sequentially*. This causes linear latency scaling (3 workers × 30s = 90s).
**Problem**: A naive "parallel loop" approach introduces **race conditions** on the shared `todos` state, causing data loss.
**Goal**: Implement **Parallel Tool Execution** using a **Map-Reduce** architecture. This ensures 30s total latency while maintaining strict **State Consistency** and **Observability**.

## 2. Technical Approach: Map-Reduce

Instead of "Loop & Await", we separate execution into two distinct phases: **Map** (Parallel Work) and **Reduce** (State Mutation).

### 2.1 The Map Phase (Parallel Execution)
The Supervisor identifies independent tasks (e.g., `delegate_research`) and launches them concurrently.
-   **Isolation**: Workers do *not* write to the `OrchestratorState` directly. They return a `WorkerResult` object.
-   **Structured Concurrency**: We use `asyncio.TaskGroup` (Python 3.11+) to manage the lifecycle of these tasks, ensuring no "dangling promises" or unhandled exceptions.

```python
# Conceptual "Map" Logic
async with asyncio.TaskGroup() as tg:
    for tool_call in parallel_tools:
        # Create task, but DO NOT await here
        tasks.append(tg.create_task(execute_worker(tool_call)))
```

### 2.2 The Reduce Phase (State Update)
Once `TaskGroup` exits (all tasks done), we process results **sequentially** in a single thread. this guarantees atomic updates to the `todos` list.

```python
# Conceptual "Reduce" Logic
for task in tasks:
    result = task.result()
    # Apply result to state locally - NO RACE CONDITION
    state["todos"] = update_todo_state(state["todos"], result)
```

### 2.3 Managing "Mutators" (Synchronous Tools)
Some tools (e.g., `manage_todos` for *creating* tasks) must run *before* parallel work or *strictly sequentially*.
-   **Strategy**: Split tool calls into `pre_mutators` (sync) and `parallel_workers` (async).
-   **Execution Order**: Run `pre_mutators` -> Run `parallel_workers` -> Run `post_mutators` (if any).

## 3. Implementation Details

### 3.1 Refactoring `supervisor.py`

The `tool_execution_node` will be rewritten to:
1.  **Classify** tool calls: `["manage_todos"]` vs `["delegate_research"]`.
2.  **Execute Mutators**: Run `manage_todos` calls one by one to update the plan.
3.  **Execute Parallel Block**:
    -   Instantiate `asyncio.TaskGroup`.
    -   Launch all `delegate_research` calls.
    -   **Context Limiting**: If > 5 tasks, use `asyncio.Semaphore(5)` to rate-limit.
4.  **Aggregate Results**: Collect findings.
5.  **State Update**: Map findings back to `todos` using `task_id`.

### 3.2 Observability & Tracing
Parallel execution kills linear log readability. We must use **Distributed Tracing**.
-   **Trace ID**: The Supervisor generates a request-scoped `trace_id` (or uses LangGraph's run ID).
-   **Propagation**: This ID is passed to `Worker.run_async(..., run_id=trace_id)`.
-   **Benefit**: In LangSmith, the trace will show a "fan-out" visualization, grouping all worker logs under the parent Supervisor step.

### 3.3 Context Management (Intermediate Summarization)
Returning 5 full reports (e.g., 25k tokens) will crash the Supervisor's context window.
-   **Constraint**: Max 4k input tokens per worker result.
-   **Mechanism**: The `Worker` is already designed to summarize if it hits limits. We will enforce a strict `max_output_tokens=4000` in the `Limits` object passed to workers.

## 4. Safety & Verification

### 4.1 Race Condition Safety
-   **Invariant**: Only the **Main Thread** (Supervisor Node) writes to `state["todos"]`.
-   **Invariant**: Workers are **Read-Only** regarding the global plan.

### 4.2 Verification Plan (`verify_parallel.py`)
1.  **Mock Worker**: Create a mock that sleeps 1s and returns a unique string.
2.  **Scenario**: Supervisor creates 5 tasks.
3.  **Assertion 1 (Speed)**: Total time < 1.5s (parallel) vs > 5s (sequential).
4.  **Assertion 2 (Data Integrity)**: `todos` list has **5 completed items**. (If race condition exists, likely only 1 item is completed).

## 5. Migration Guide
1.  **Update `supervisor.py`**:
    -   Import `asyncio.TaskGroup`.
    -   Implement `classify_tools` helper.
    -   Implement the Map-Combine logic.
2.  **Update `prompts.py`**:
    -   Explicitly tell Supervisor: "You can schedule multiple research tasks in parallel. I will handle the multitasking."
