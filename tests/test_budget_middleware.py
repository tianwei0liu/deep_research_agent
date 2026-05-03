"""Tests for BudgetTrackingMiddleware.

Verifies:
- AIMessage counting logic.
- Budget message formatting (NORMAL vs CRITICAL).
- System message injection via wrap_model_call / awrap_model_call.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deep_research_agent.agents.budget_middleware import (
    BudgetTrackingMiddleware,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(
    messages: list[Any],
    system_message: SystemMessage | None = None,
) -> MagicMock:
    """Build a mock ModelRequest with the given messages."""
    req = MagicMock()
    req.messages = messages
    req.system_message = system_message or SystemMessage(content="You are a supervisor.")

    def _override(**kwargs: Any) -> MagicMock:
        new_req = MagicMock()
        new_req.messages = kwargs.get("messages", req.messages)
        new_req.system_message = kwargs.get("system_message", req.system_message)
        new_req.override = _override
        return new_req

    req.override = _override
    return req


def _make_response() -> MagicMock:
    """Build a mock ModelResponse."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Unit tests: _count_ai_messages
# ---------------------------------------------------------------------------

class TestCountAIMessages:
    """Tests for the static AIMessage counting method."""

    def test_empty(self) -> None:
        assert BudgetTrackingMiddleware._count_ai_messages([]) == 0

    def test_only_human(self) -> None:
        messages = [HumanMessage(content="hi"), HumanMessage(content="there")]
        assert BudgetTrackingMiddleware._count_ai_messages(messages) == 0

    def test_mixed(self) -> None:
        messages = [
            HumanMessage(content="q"),
            AIMessage(content="a1"),
            ToolMessage(content="result", tool_call_id="t1"),
            AIMessage(content="a2"),
        ]
        assert BudgetTrackingMiddleware._count_ai_messages(messages) == 2

    def test_all_ai(self) -> None:
        messages = [AIMessage(content=f"turn {i}") for i in range(10)]
        assert BudgetTrackingMiddleware._count_ai_messages(messages) == 10


# ---------------------------------------------------------------------------
# Unit tests: _build_budget_message
# ---------------------------------------------------------------------------

class TestBuildBudgetMessage:
    """Tests for budget message formatting."""

    def setup_method(self) -> None:
        self.mw = BudgetTrackingMiddleware(max_turns=35, critical_threshold=3)

    def test_normal_status(self) -> None:
        msg = self.mw._build_budget_message(current_turn=5)
        assert "5 / 35" in msg
        assert "Remaining turns: 30" in msg
        assert "NORMAL" in msg
        assert "CRITICAL" not in msg

    def test_critical_at_threshold(self) -> None:
        """remaining = 35 - 32 = 3 → exactly at threshold → CRITICAL."""
        msg = self.mw._build_budget_message(current_turn=32)
        assert "32 / 35" in msg
        assert "Remaining turns: 3" in msg
        assert "CRITICAL" in msg

    def test_critical_below_threshold(self) -> None:
        msg = self.mw._build_budget_message(current_turn=34)
        assert "Remaining turns: 1" in msg
        assert "CRITICAL" in msg

    def test_critical_at_max(self) -> None:
        msg = self.mw._build_budget_message(current_turn=35)
        assert "Remaining turns: 0" in msg
        assert "CRITICAL" in msg

    def test_beyond_max_clamps_to_zero(self) -> None:
        """If somehow called beyond max_turns, remaining should be 0."""
        msg = self.mw._build_budget_message(current_turn=40)
        assert "Remaining turns: 0" in msg
        assert "CRITICAL" in msg

    def test_custom_threshold(self) -> None:
        mw = BudgetTrackingMiddleware(max_turns=20, critical_threshold=5)
        # Turn 15 → remaining = 5 → at threshold → CRITICAL
        msg = mw._build_budget_message(current_turn=15)
        assert "CRITICAL" in msg
        # Turn 14 → remaining = 6 → above threshold → NORMAL
        msg = mw._build_budget_message(current_turn=14)
        assert "NORMAL" in msg


# ---------------------------------------------------------------------------
# Integration tests: wrap_model_call
# ---------------------------------------------------------------------------

class TestWrapModelCall:
    """Tests for sync model call wrapping."""

    def test_injects_budget_into_system_message(self) -> None:
        mw = BudgetTrackingMiddleware(max_turns=10)
        messages = [
            HumanMessage(content="research X"),
            AIMessage(content="planning"),
            ToolMessage(content="ok", tool_call_id="t1"),
            AIMessage(content="delegating"),
        ]
        request = _make_request(messages)
        expected_response = _make_response()

        captured_requests: list[Any] = []

        def handler(req: Any) -> Any:
            captured_requests.append(req)
            return expected_response

        result = mw.wrap_model_call(request, handler)

        assert result is expected_response
        assert len(captured_requests) == 1
        # The overridden request should have budget in system message
        overridden = captured_requests[0]
        # Current turn = 2 AIMessages + 1 = 3
        sys_content = overridden.system_message.content
        # SystemMessage.content may be a list of content blocks
        sys_text = (
            sys_content
            if isinstance(sys_content, str)
            else " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in sys_content
            )
        )
        assert "3 / 10" in sys_text
        assert "Remaining turns: 7" in sys_text

    def test_first_turn_budget(self) -> None:
        """First turn has 0 AIMessages → current_turn = 1."""
        mw = BudgetTrackingMiddleware(max_turns=35)
        request = _make_request([HumanMessage(content="start")])
        captured: list[Any] = []

        def handler(req: Any) -> Any:
            captured.append(req)
            return _make_response()

        mw.wrap_model_call(request, handler)
        sys_content = captured[0].system_message.content
        sys_text = (
            sys_content
            if isinstance(sys_content, str)
            else " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in sys_content
            )
        )
        assert "1 / 35" in sys_text
        assert "Remaining turns: 34" in sys_text
        assert "NORMAL" in sys_text


# ---------------------------------------------------------------------------
# Integration tests: awrap_model_call
# ---------------------------------------------------------------------------

class TestAsyncWrapModelCall:
    """Tests for async model call wrapping."""

    def test_async_injects_budget(self) -> None:
        mw = BudgetTrackingMiddleware(max_turns=10)
        messages = [
            HumanMessage(content="q"),
            AIMessage(content="a"),
        ]
        request = _make_request(messages)
        expected = _make_response()
        captured: list[Any] = []

        async def handler(req: Any) -> Any:
            captured.append(req)
            return expected

        result = asyncio.get_event_loop().run_until_complete(
            mw.awrap_model_call(request, handler)
        )
        assert result is expected
        sys_content = captured[0].system_message.content
        sys_text = (
            sys_content
            if isinstance(sys_content, str)
            else " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in sys_content
            )
        )
        # 1 AIMessage → turn 2
        assert "2 / 10" in sys_text

    def test_async_critical_injection(self) -> None:
        mw = BudgetTrackingMiddleware(max_turns=5, critical_threshold=2)
        # 3 AIMessages → turn 4, remaining = 1 → CRITICAL
        messages = [AIMessage(content=f"t{i}") for i in range(3)]
        request = _make_request(messages)
        captured: list[Any] = []

        async def handler(req: Any) -> Any:
            captured.append(req)
            return _make_response()

        asyncio.get_event_loop().run_until_complete(
            mw.awrap_model_call(request, handler)
        )
        sys_content = captured[0].system_message.content
        sys_text = (
            sys_content
            if isinstance(sys_content, str)
            else " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in sys_content
            )
        )
        assert "CRITICAL" in sys_text
        assert "4 / 5" in sys_text
