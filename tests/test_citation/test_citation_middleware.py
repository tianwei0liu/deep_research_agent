"""Tests for CitationDataMiddleware (data injection + L1 validation gate + retry).

Covers:
- Data injection: Worker findings auto-inject into citation-specialist input
- L1 validation gate: CitationAgent output must pass L1 before returning
- Self-correction retry: On L1 failure, re-invoke CitationAgent with corrections
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from deep_research_agent.agents.citation.models import (
    Finding,
    WorkerOutput,
)


def _make_worker_output_json(**overrides: Any) -> str:
    """Create a valid WorkerOutput JSON string."""
    base = WorkerOutput(
        summary="Test summary",
        findings=[
            Finding(
                claim="LangGraph supports checkpointing",
                source_urls=["https://langchain-ai.github.io/langgraph/"],
                source_titles=["LangGraph Docs"],
                evidence='"Built-in checkpointing support"',
            ),
        ],
        sources_consulted=["https://langchain-ai.github.io/langgraph/"],
        caveats="",
    )
    return base.model_dump_json()


def _make_tool_message(content: str, tool_call_id: str = "tc-1") -> Any:
    """Create a mock ToolMessage."""
    from langchain_core.messages import ToolMessage

    return ToolMessage(content=content, tool_call_id=tool_call_id)


def _make_valid_cited_report() -> str:
    """A report that passes L1 validation."""
    return (
        "# Research Report\n\n"
        "LangGraph supports checkpointing [1].\n\n"
        "## Sources\n\n"
        "[1] LangGraph Docs — https://langchain-ai.github.io/langgraph/\n"
    )


def _make_invalid_cited_report() -> str:
    """A report with L1 ERROR (dangling citation [2])."""
    return (
        "# Research Report\n\n"
        "LangGraph supports checkpointing [1] and streaming [2].\n\n"
        "## Sources\n\n"
        "[1] LangGraph Docs — https://langchain-ai.github.io/langgraph/\n"
    )


def _make_warning_only_report() -> str:
    """A report with L1 WARNING only (orphan source, no errors)."""
    return (
        "# Research Report\n\n"
        "LangGraph supports checkpointing [1].\n\n"
        "## Sources\n\n"
        "[1] LangGraph Docs — https://langchain-ai.github.io/langgraph/\n"
        "[2] Extra Source — https://example.com/extra\n"
    )


def _make_request(
    tool_name: str = "task",
    subagent_type: str = "citation-specialist",
    description: str = "# Draft Report\n\nLangGraph supports checkpointing.",
    state_messages: list | None = None,
) -> Any:
    """Create a mock ToolCallRequest."""
    tool_call = {
        "name": tool_name,
        "args": {
            "subagent_type": subagent_type,
            "description": description,
        },
        "id": "call-123",
        "type": "tool_call",
    }
    state = {"messages": state_messages or []}

    request = MagicMock()
    request.tool_call = tool_call
    request.state = state

    def override_fn(**overrides: Any) -> Any:
        new_req = MagicMock()
        new_req.tool_call = overrides.get("tool_call", tool_call)
        new_req.state = overrides.get("state", state)
        new_req.override = override_fn
        return new_req

    request.override = override_fn
    return request


def _make_command_result(report_content: str) -> Any:
    """Create a mock Command result from CitationAgent."""
    from langchain_core.messages import ToolMessage

    command = MagicMock()
    command.update = {
        "messages": [ToolMessage(content=report_content, tool_call_id="call-123")],
    }
    return command


# ---------------------------------------------------------------------------
# Data injection tests
# ---------------------------------------------------------------------------

class TestDataInjection:
    """Tests for Worker findings auto-injection."""

    def test_injects_findings_for_citation_specialist(self) -> None:
        """Middleware enriches description with Worker findings."""
        from deep_research_agent.agents.citation.citation_middleware import (
            CitationDataMiddleware,
        )

        mw = CitationDataMiddleware()
        worker_json = _make_worker_output_json()
        worker_msg = _make_tool_message(worker_json)

        request = _make_request(state_messages=[worker_msg])

        # handler returns valid cited report
        valid_report = _make_valid_cited_report()

        def handler(req: Any) -> Any:
            # Verify the description was enriched
            desc = req.tool_call["args"]["description"]
            assert "WORKER FINDINGS" in desc
            assert "findings" in desc
            return _make_command_result(valid_report)

        mw.wrap_tool_call(request, handler)

    def test_ignores_non_citation_specialist_tasks(self) -> None:
        """Middleware passes through non-citation-specialist tasks."""
        from deep_research_agent.agents.citation.citation_middleware import (
            CitationDataMiddleware,
        )

        mw = CitationDataMiddleware()
        request = _make_request(subagent_type="research-worker")

        handler_called_with = []

        def handler(req: Any) -> Any:
            handler_called_with.append(req)
            return _make_command_result("some result")

        mw.wrap_tool_call(request, handler)

        # Handler should have been called with original request
        assert len(handler_called_with) == 1
        assert handler_called_with[0].tool_call["args"]["subagent_type"] == "research-worker"

    def test_ignores_non_task_tools(self) -> None:
        """Middleware passes through non-task tools completely."""
        from deep_research_agent.agents.citation.citation_middleware import (
            CitationDataMiddleware,
        )

        mw = CitationDataMiddleware()
        request = _make_request(tool_name="search")

        handler_called = []

        def handler(req: Any) -> Any:
            handler_called.append(True)
            return _make_command_result("result")

        mw.wrap_tool_call(request, handler)
        assert len(handler_called) == 1

    def test_handles_no_worker_outputs_gracefully(self) -> None:
        """With no Worker ToolMessages, description is unchanged."""
        from deep_research_agent.agents.citation.citation_middleware import (
            CitationDataMiddleware,
        )

        mw = CitationDataMiddleware()
        request = _make_request(state_messages=[])
        valid_report = _make_valid_cited_report()

        def handler(req: Any) -> Any:
            return _make_command_result(valid_report)

        result = mw.wrap_tool_call(request, handler)
        # Should still succeed
        assert result is not None

    def test_handles_malformed_json_gracefully(self) -> None:
        """Malformed JSON in ToolMessage is skipped, not crash."""
        from deep_research_agent.agents.citation.citation_middleware import (
            CitationDataMiddleware,
        )

        mw = CitationDataMiddleware()
        malformed_msg = _make_tool_message('{"findings": invalid json}')
        request = _make_request(state_messages=[malformed_msg])
        valid_report = _make_valid_cited_report()

        def handler(req: Any) -> Any:
            return _make_command_result(valid_report)

        # Should not raise
        result = mw.wrap_tool_call(request, handler)
        assert result is not None


# ---------------------------------------------------------------------------
# L1 validation gate tests
# ---------------------------------------------------------------------------

class TestL1ValidationGate:
    """Tests for L1 validation gate within middleware."""

    def test_valid_report_passes_without_retry(self) -> None:
        """L1-valid report passes through — handler called exactly once."""
        from deep_research_agent.agents.citation.citation_middleware import (
            CitationDataMiddleware,
        )

        mw = CitationDataMiddleware()
        worker_msg = _make_tool_message(_make_worker_output_json())
        request = _make_request(state_messages=[worker_msg])
        valid_report = _make_valid_cited_report()

        call_count = 0

        def handler(req: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return _make_command_result(valid_report)

        mw.wrap_tool_call(request, handler)
        assert call_count == 1

    def test_invalid_report_triggers_retry(self) -> None:
        """L1-invalid report (ERROR level) triggers retry."""
        from deep_research_agent.agents.citation.citation_middleware import (
            CitationDataMiddleware,
        )

        mw = CitationDataMiddleware()
        worker_msg = _make_tool_message(_make_worker_output_json())
        request = _make_request(state_messages=[worker_msg])

        call_count = 0

        def handler(req: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_command_result(_make_invalid_cited_report())
            return _make_command_result(_make_valid_cited_report())

        mw.wrap_tool_call(request, handler)
        assert call_count == 2  # initial + 1 retry

    def test_warning_only_does_not_trigger_retry(self) -> None:
        """L1-WARNING only does NOT trigger retry."""
        from deep_research_agent.agents.citation.citation_middleware import (
            CitationDataMiddleware,
        )

        mw = CitationDataMiddleware()
        worker_msg = _make_tool_message(_make_worker_output_json())
        request = _make_request(state_messages=[worker_msg])

        call_count = 0

        def handler(req: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return _make_command_result(_make_warning_only_report())

        mw.wrap_tool_call(request, handler)
        assert call_count == 1  # No retry for warnings

    def test_retry_exhaustion_appends_warning(self) -> None:
        """When retry exhausted, report gets warning appended."""
        from deep_research_agent.agents.citation.citation_middleware import (
            CitationDataMiddleware,
        )

        mw = CitationDataMiddleware()
        worker_msg = _make_tool_message(_make_worker_output_json())
        request = _make_request(state_messages=[worker_msg])

        def handler(req: Any) -> Any:
            # Always returns invalid report
            return _make_command_result(_make_invalid_cited_report())

        result = mw.wrap_tool_call(request, handler)
        # Result should have warning appended
        content = result.update["messages"][0].content
        assert "Citation Notice" in content or "citation" in content.lower()

    def test_max_retries_limits_handler_calls(self) -> None:
        """max_retries=1 means at most 2 handler calls (initial + 1 retry)."""
        from deep_research_agent.agents.citation.citation_middleware import (
            CitationDataMiddleware,
        )

        mw = CitationDataMiddleware(max_retries=1)
        worker_msg = _make_tool_message(_make_worker_output_json())
        request = _make_request(state_messages=[worker_msg])

        call_count = 0

        def handler(req: Any) -> Any:
            nonlocal call_count
            call_count += 1
            return _make_command_result(_make_invalid_cited_report())

        mw.wrap_tool_call(request, handler)
        assert call_count == 2  # initial + max_retries(1)

    def test_retry_includes_correction_instructions(self) -> None:
        """Retry request description includes L1 error details."""
        from deep_research_agent.agents.citation.citation_middleware import (
            CitationDataMiddleware,
        )

        mw = CitationDataMiddleware()
        worker_msg = _make_tool_message(_make_worker_output_json())
        request = _make_request(state_messages=[worker_msg])

        retry_descriptions: list[str] = []

        def handler(req: Any) -> Any:
            desc = req.tool_call["args"]["description"]
            retry_descriptions.append(desc)
            if len(retry_descriptions) == 1:
                return _make_command_result(_make_invalid_cited_report())
            return _make_command_result(_make_valid_cited_report())

        mw.wrap_tool_call(request, handler)
        # Second call should have correction instructions
        assert len(retry_descriptions) == 2
        assert "CORRECTION" in retry_descriptions[1].upper() or \
               "L1-01" in retry_descriptions[1]


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------

class TestAsyncMiddleware:
    """Tests for the async version of the middleware."""

    @pytest.mark.asyncio
    async def test_async_injects_findings(self) -> None:
        """Async middleware also injects findings."""
        from deep_research_agent.agents.citation.citation_middleware import (
            CitationDataMiddleware,
        )

        mw = CitationDataMiddleware()
        worker_msg = _make_tool_message(_make_worker_output_json())
        request = _make_request(state_messages=[worker_msg])
        valid_report = _make_valid_cited_report()

        async def handler(req: Any) -> Any:
            desc = req.tool_call["args"]["description"]
            assert "WORKER FINDINGS" in desc
            return _make_command_result(valid_report)

        await mw.awrap_tool_call(request, handler)

    @pytest.mark.asyncio
    async def test_async_validates_and_retries(self) -> None:
        """Async middleware validates and retries on L1 errors."""
        from deep_research_agent.agents.citation.citation_middleware import (
            CitationDataMiddleware,
        )

        mw = CitationDataMiddleware()
        worker_msg = _make_tool_message(_make_worker_output_json())
        request = _make_request(state_messages=[worker_msg])

        call_count = 0

        async def handler(req: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_command_result(_make_invalid_cited_report())
            return _make_command_result(_make_valid_cited_report())

        await mw.awrap_tool_call(request, handler)
        assert call_count == 2
