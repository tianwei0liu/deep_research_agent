"""Tests for parallel and sequential task delegation in the Supervisor.

Verifies:
1. Parallel workers execute concurrently (wall-clock time proves it).
2. Sequential mutators execute in order.
3. add_task supports custom task_id and dependencies.
4. Supervisor auto-injects dependency context into workers.
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from langchain_core.messages import AIMessage, ToolMessage

from research_assistant.agents.orchestrator.supervisor import Supervisor
from research_assistant.agents.orchestrator.schemas import ResearchTask, TaskStatus
from research_assistant.tools.planning import PlanningTool
from research_assistant.config import Settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings():
    """Minimal settings loaded from defaults."""
    return Settings.load()


@pytest.fixture
def supervisor(settings):
    return Supervisor(settings=settings)


def _make_state(tool_calls, todos=None, max_parallel_workers=10):
    """Build a minimal OrchestratorState with an AIMessage containing tool_calls."""
    ai_msg = AIMessage(content="", tool_calls=tool_calls)
    return {
        "messages": [ai_msg],
        "todos": todos or [],
        "max_parallel_workers": max_parallel_workers,
    }


# ---------------------------------------------------------------------------
# TEST 1: Parallel workers execute concurrently
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_parallel_workers_execute_concurrently(supervisor):
    """Three delegate_research calls with 1s delay each should complete in ~1s, not ~3s."""

    # Setup: 3 pending tasks
    todos = [
        ResearchTask(id=f"T{i}", objective=f"Obj {i}", description="desc")
        for i in range(1, 4)
    ]

    tool_calls = [
        {"id": f"call_{i}", "name": "delegate_research",
         "args": {"task_id": f"T{i}", "objective": f"Obj {i}", "instructions": "instr"}}
        for i in range(1, 4)
    ]

    state = _make_state(tool_calls, todos=todos)

    # Mock: each delegate_research sleeps 1 second then returns findings
    async def mock_delegate(self, task_id, objective, instructions, context=None, limits=None):
        await asyncio.sleep(1.0)
        return {
            "status": "completed",
            "brief_summary": f"Summary for {task_id}",
            "full_findings": f"Findings for {task_id}",
            "metadata": {},
        }

    with patch(
        "research_assistant.tools.delegation.DelegationTool.delegate_research",
        mock_delegate,
    ):
        start = time.monotonic()
        result = await supervisor.execute_tools(state)
        elapsed = time.monotonic() - start

    # Assert: parallel → should take ~1s, NOT ~3s
    assert elapsed < 2.0, f"Expected parallel execution (~1s), but took {elapsed:.2f}s"

    # All 3 ToolMessages present
    messages = result["messages"]
    assert len(messages) == 3
    for msg in messages:
        assert isinstance(msg, ToolMessage)


# ---------------------------------------------------------------------------
# TEST 2: Sequential mutators execute in order
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sequential_mutators_execute_in_order(supervisor):
    """Three add_task calls should produce todos in correct insertion order."""

    tool_calls = [
        {"id": f"call_{i}", "name": "add_task",
         "args": {"objective": f"Task {i}", "description": f"Description {i}"}}
        for i in range(1, 4)
    ]

    state = _make_state(tool_calls)

    result = await supervisor.execute_tools(state)

    # Assert: 3 todos in correct order
    todos = result["todos"]
    assert len(todos) == 3
    assert todos[0].objective == "Task 1"
    assert todos[1].objective == "Task 2"
    assert todos[2].objective == "Task 3"

    # Assert: 3 ToolMessage responses
    messages = result["messages"]
    assert len(messages) == 3
    for msg in messages:
        assert isinstance(msg, ToolMessage)
        assert "Successfully added task" in msg.content


# ---------------------------------------------------------------------------
# TEST 3: add_task with custom task_id
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_task_with_custom_id(supervisor):
    """add_task(task_id='MY_ID') should create a task with id='MY_ID'."""

    tool_calls = [
        {"id": "call_1", "name": "add_task",
         "args": {"task_id": "MY_ID", "objective": "Custom", "description": "desc"}}
    ]

    state = _make_state(tool_calls)
    result = await supervisor.execute_tools(state)

    todos = result["todos"]
    assert len(todos) == 1
    assert todos[0].id == "MY_ID"
    assert todos[0].objective == "Custom"


# ---------------------------------------------------------------------------
# TEST 4: add_task with dependencies
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_add_task_with_dependencies(supervisor):
    """add_task with dependencies should store them on the ResearchTask."""

    tool_calls = [
        {"id": "call_1", "name": "add_task",
         "args": {"task_id": "T1", "objective": "Root", "description": "desc"}},
        {"id": "call_2", "name": "add_task",
         "args": {"task_id": "T2", "objective": "Dep", "description": "desc",
                  "dependencies": ["T1"]}},
    ]

    state = _make_state(tool_calls)
    result = await supervisor.execute_tools(state)

    todos = result["todos"]
    assert len(todos) == 2
    assert todos[0].id == "T1"
    assert todos[0].dependencies == []
    assert todos[1].id == "T2"
    assert todos[1].dependencies == ["T1"]


# ---------------------------------------------------------------------------
# TEST 5: Supervisor auto-injects context from dependencies
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_supervisor_injects_context_from_dependencies(supervisor):
    """When delegating T2 (depends on T1), T1's result should be injected as context."""

    # T1 is completed with findings; T2 depends on T1
    t1 = ResearchTask(id="T1", objective="Find date", description="", status=TaskStatus.COMPLETED, full_findings="Today is Feb 2026")
    t2 = ResearchTask(id="T2", objective="Find controversies", description="", dependencies=["T1"])

    tool_calls = [
        {"id": "call_1", "name": "delegate_research",
         "args": {"task_id": "T2", "objective": "Find controversies", "instructions": "instr"}}
    ]

    state = _make_state(tool_calls, todos=[t1, t2])

    captured_context = {}

    async def mock_delegate(self, task_id, objective, instructions, context=None, limits=None):
        captured_context["context"] = context
        return {
            "status": "completed",
            "brief_summary": f"Summary for {task_id}",
            "full_findings": f"Findings for {task_id}",
            "metadata": {},
        }

    with patch(
        "research_assistant.tools.delegation.DelegationTool.delegate_research",
        mock_delegate,
    ):
        await supervisor.execute_tools(state)

    # Assert: context was injected with T1's result
    assert captured_context["context"] is not None
    assert "Today is Feb 2026" in captured_context["context"]
    assert "Find date" in captured_context["context"]


# ---------------------------------------------------------------------------
# TEST 6: No context injected for tasks without dependencies
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_context_for_independent_tasks(supervisor):
    """Tasks without dependencies should not receive any context."""

    t1 = ResearchTask(id="T1", objective="Independent task", description="")

    tool_calls = [
        {"id": "call_1", "name": "delegate_research",
         "args": {"task_id": "T1", "objective": "Independent task", "instructions": "instr"}}
    ]

    state = _make_state(tool_calls, todos=[t1])

    captured_context = {}

    async def mock_delegate(self, task_id, objective, instructions, context=None, limits=None):
        captured_context["context"] = context
        return {
            "status": "completed",
            "brief_summary": f"Summary for {task_id}",
            "full_findings": f"Findings for {task_id}",
            "metadata": {},
        }

    with patch(
        "research_assistant.tools.delegation.DelegationTool.delegate_research",
        mock_delegate,
    ):
        await supervisor.execute_tools(state)

    # Assert: no context injected
    assert captured_context["context"] is None
