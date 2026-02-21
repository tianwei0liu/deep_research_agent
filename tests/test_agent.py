"""Unit tests for agent state models and worker initialization."""

import pytest
from pydantic import ValidationError

from deep_research_agent.agents.worker.schemas import Limits, SpawnTask, WorkerResult
from deep_research_agent.agents.orchestrator.schemas import TaskStatus
from deep_research_agent.agents.worker.worker import Worker


def test_spawn_task_validation():
    """Test SpawnTask Pydantic validation."""
    # Valid task
    task = SpawnTask(
        objective="Test objective",
        output_format="JSON",
        sources_and_tools="Test tools",
        task_boundaries="Test boundaries",
    )
    assert task.objective == "Test objective"

    # Missing required field (should fail if not optional, but in our case all are required except context_snippet etc)
    # Actually Pydantic models require fields unless they have defaults.
    # checking definition: objective: str, output_format: str...
    with pytest.raises(ValidationError):
        SpawnTask(objective="Just objective")  # Missing output_format etc.


def test_limits_validation():
    """Test Limits Pydantic validation."""
    limits = Limits(max_tool_calls=10, max_turns=5)
    assert limits.max_tool_calls == 10

    # Test default values if any (max_output_tokens is optional)
    assert limits.max_output_tokens is None


def test_worker_result_validation():
    """Test WorkerResult Pydantic validation."""
    result = WorkerResult(
        status=TaskStatus.COMPLETED,
        brief_summary="Found it",
        full_findings="Full markdown report",
        metadata={"k": "v"},
    )
    assert result.status == TaskStatus.COMPLETED
    assert result.brief_summary == "Found it"
    assert result.full_findings == "Full markdown report"
    assert result.metadata == {"k": "v"}


def test_worker_initialization():
    """Test Worker initialization with dependency injection."""
    worker = Worker()
    assert worker._client is None
    assert worker._settings is not None  # Should load default settings
