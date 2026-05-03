"""Unit tests for ``PatchedChatDeepSeek`` reasoning_content passback fix.

Validates that ``reasoning_content`` stored in
``AIMessage.additional_kwargs`` survives the serialization round-trip
through ``_get_request_payload``.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_research_agent.agents.patched_deepseek import (
    PatchedChatDeepSeek,
)


@pytest.fixture()
def model() -> PatchedChatDeepSeek:
    """Create a ``PatchedChatDeepSeek`` instance for testing."""
    return PatchedChatDeepSeek(model="deepseek-v4-flash")


class TestReasoningContentPassback:
    """Verify that reasoning_content is preserved in API payloads."""

    def test_reasoning_content_injected_for_tool_call_turn(
        self, model: PatchedChatDeepSeek
    ) -> None:
        """The core bug scenario: assistant message with tool_calls + reasoning_content."""
        reasoning = "Let me think step by step about this query..."
        messages = [
            HumanMessage(content="What is LangGraph?"),
            AIMessage(
                content="",
                id="msg_001",
                tool_calls=[
                    {
                        "id": "call_abc",
                        "name": "internet_search",
                        "args": {"query": "LangGraph overview"},
                    }
                ],
                additional_kwargs={"reasoning_content": reasoning},
            ),
            ToolMessage(content="LangGraph is a framework...", tool_call_id="call_abc"),
        ]

        payload = model._get_request_payload(messages)

        assistant_msgs = [
            m for m in payload["messages"] if m["role"] == "assistant"
        ]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["reasoning_content"] == reasoning
        assert "tool_calls" in assistant_msgs[0]

    def test_no_reasoning_content_leaves_payload_unchanged(
        self, model: PatchedChatDeepSeek
    ) -> None:
        """When no reasoning_content exists, payload should not be modified."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!", id="msg_002"),
        ]

        payload = model._get_request_payload(messages)

        assistant_msgs = [
            m for m in payload["messages"] if m["role"] == "assistant"
        ]
        assert len(assistant_msgs) == 1
        assert "reasoning_content" not in assistant_msgs[0]

    def test_multiple_assistant_turns_each_preserves_reasoning(
        self, model: PatchedChatDeepSeek
    ) -> None:
        """Multi-turn conversation: each assistant message keeps its own reasoning."""
        reasoning_1 = "First thinking step..."
        reasoning_2 = "Second thinking step..."
        messages = [
            HumanMessage(content="Step 1"),
            AIMessage(
                content="",
                id="msg_010",
                tool_calls=[
                    {"id": "call_1", "name": "search", "args": {"q": "a"}},
                ],
                additional_kwargs={"reasoning_content": reasoning_1},
            ),
            ToolMessage(content="result 1", tool_call_id="call_1"),
            AIMessage(
                content="",
                id="msg_020",
                tool_calls=[
                    {"id": "call_2", "name": "search", "args": {"q": "b"}},
                ],
                additional_kwargs={"reasoning_content": reasoning_2},
            ),
            ToolMessage(content="result 2", tool_call_id="call_2"),
        ]

        payload = model._get_request_payload(messages)

        assistant_msgs = [
            m for m in payload["messages"] if m["role"] == "assistant"
        ]
        assert len(assistant_msgs) == 2
        assert assistant_msgs[0]["reasoning_content"] == reasoning_1
        assert assistant_msgs[1]["reasoning_content"] == reasoning_2

    def test_assistant_without_tool_calls_but_with_reasoning(
        self, model: PatchedChatDeepSeek
    ) -> None:
        """Reasoning content is preserved even when there are no tool calls."""
        reasoning = "Thinking about the answer..."
        messages = [
            HumanMessage(content="Explain RAG"),
            AIMessage(
                content="RAG stands for...",
                id="msg_003",
                additional_kwargs={"reasoning_content": reasoning},
            ),
        ]

        payload = model._get_request_payload(messages)

        assistant_msgs = [
            m for m in payload["messages"] if m["role"] == "assistant"
        ]
        assert assistant_msgs[0]["reasoning_content"] == reasoning

    def test_empty_reasoning_content_not_injected(
        self, model: PatchedChatDeepSeek
    ) -> None:
        """None-valued reasoning_content should not appear in the payload."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(
                content="Hi",
                id="msg_004",
                additional_kwargs={"reasoning_content": None},
            ),
        ]

        payload = model._get_request_payload(messages)

        assistant_msgs = [
            m for m in payload["messages"] if m["role"] == "assistant"
        ]
        # None is falsy for the `is not None` check, so should not be injected
        assert "reasoning_content" not in assistant_msgs[0]
