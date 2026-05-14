"""Tests for agent integration (build_deep_agent modifications).

Verifies that build_deep_agent() correctly registers citation
subagent, middleware, and response_format with MCP-backed tools.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research_agent.agents.mcp_client import MCPSearchClient


# Patch targets: _load_settings is module-level,
# but create_deep_agent is imported inside build_deep_agent from deepagents.
_SETTINGS_PATCH = "deep_research_agent.agents.agent._load_settings"
_CREATE_PATCH = "deepagents.create_deep_agent"

_FAKE_SETTINGS: dict[str, Any] = {
    "planner_model": "test-model",
    "worker_model": "test-model",
    # Orchestration limits (mirrors tools.load_settings output)
    "supervisor_max_turns": 35,
    "supervisor_max_search_calls": 10,
    "worker_max_search_calls": 60,
    "worker_max_turns": 20,
    # Citation limits
    "citation_max_retries": 5,
    # Runtime safety
    "research_timeout_seconds": 600,
}


def _make_mock_mcp_client() -> MagicMock:
    """Create a mock MCPSearchClient with 6 fake tools."""
    mock_tools = {
        "web_search": MagicMock(name="web_search"),
        "zhihu_search": MagicMock(name="zhihu_search"),
        "weibo_search": MagicMock(name="weibo_search"),
        "weixin_search": MagicMock(name="weixin_search"),
        "github_search": MagicMock(name="github_search"),
        "scrape_url": MagicMock(name="scrape_url"),
    }
    # Set .name attribute explicitly (MagicMock's name= is for repr)
    for tool_name, mock_tool in mock_tools.items():
        mock_tool.name = tool_name

    client = MagicMock(spec=MCPSearchClient)

    def get_tools(*, names=None):
        if names is None:
            return list(mock_tools.values())
        return [mock_tools[n] for n in names]

    client.get_tools = get_tools
    return client


class TestBuildDeepAgent:
    """Tests for citation integration in build_deep_agent()."""

    @pytest.mark.asyncio
    @patch(_SETTINGS_PATCH, return_value=_FAKE_SETTINGS)
    @patch(_CREATE_PATCH, return_value=MagicMock())
    async def test_research_worker_has_response_format(
        self,
        mock_create: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """research-worker subagent spec includes response_format=WorkerOutput."""
        from deep_research_agent.agents.citation.models import WorkerOutput
        from deep_research_agent.agents.agent import build_deep_agent

        mock_client = _make_mock_mcp_client()
        await build_deep_agent(mcp_client=mock_client)

        call_kwargs = mock_create.call_args[1]
        subagents = call_kwargs["subagents"]

        research = next(s for s in subagents if s["name"] == "research-worker")
        assert research["response_format"] is WorkerOutput

    @pytest.mark.asyncio
    @patch(_SETTINGS_PATCH, return_value=_FAKE_SETTINGS)
    @patch(_CREATE_PATCH, return_value=MagicMock())
    async def test_citation_specialist_registered(
        self,
        mock_create: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """citation-specialist subagent is registered with correct spec."""
        from deep_research_agent.agents.agent import build_deep_agent

        mock_client = _make_mock_mcp_client()
        await build_deep_agent(mcp_client=mock_client)

        call_kwargs = mock_create.call_args[1]
        subagents = call_kwargs["subagents"]

        citation = next(s for s in subagents if s["name"] == "citation-specialist")
        # citation-specialist is a CompiledSubAgent (has 'runnable', not 'tools')
        assert "runnable" in citation
        assert "response_format" not in citation

    @pytest.mark.asyncio
    @patch(_SETTINGS_PATCH, return_value=_FAKE_SETTINGS)
    @patch(_CREATE_PATCH, return_value=MagicMock())
    async def test_citation_middleware_registered(
        self,
        mock_create: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """CitationDataMiddleware is registered in create_deep_agent middleware."""
        from deep_research_agent.agents.citation.citation_middleware import (
            CitationDataMiddleware,
        )
        from deep_research_agent.agents.agent import build_deep_agent

        mock_client = _make_mock_mcp_client()
        await build_deep_agent(mcp_client=mock_client)

        call_kwargs = mock_create.call_args[1]
        middleware = call_kwargs.get("middleware", [])
        assert any(isinstance(m, CitationDataMiddleware) for m in middleware)

    @pytest.mark.asyncio
    @patch(_SETTINGS_PATCH, return_value=_FAKE_SETTINGS)
    @patch(_CREATE_PATCH, return_value=MagicMock())
    async def test_two_subagents_registered(
        self,
        mock_create: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Both research-worker and citation-specialist subagents are registered."""
        from deep_research_agent.agents.agent import build_deep_agent

        mock_client = _make_mock_mcp_client()
        await build_deep_agent(mcp_client=mock_client)

        call_kwargs = mock_create.call_args[1]
        subagents = call_kwargs["subagents"]

        names = {s["name"] for s in subagents}
        assert "research-worker" in names
        assert "citation-specialist" in names
        assert len(subagents) == 2

    @pytest.mark.asyncio
    @patch(_SETTINGS_PATCH, return_value=_FAKE_SETTINGS)
    @patch(_CREATE_PATCH, return_value=MagicMock())
    async def test_supervisor_gets_web_search_and_activate_persona(
        self,
        mock_create: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Supervisor tools should contain web_search + activate_persona."""
        from deep_research_agent.agents.agent import build_deep_agent

        mock_client = _make_mock_mcp_client()
        await build_deep_agent(mcp_client=mock_client)

        call_kwargs = mock_create.call_args[1]
        tools = call_kwargs["tools"]

        tool_names = {t.name for t in tools}
        assert "web_search" in tool_names
        assert "activate_persona" in tool_names
        assert len(tools) == 2

    @pytest.mark.asyncio
    @patch(_SETTINGS_PATCH, return_value=_FAKE_SETTINGS)
    @patch(_CREATE_PATCH, return_value=MagicMock())
    async def test_worker_gets_all_mcp_tools(
        self,
        mock_create: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Worker subagent should receive all 6 MCP tools."""
        from deep_research_agent.agents.agent import build_deep_agent

        mock_client = _make_mock_mcp_client()
        await build_deep_agent(mcp_client=mock_client)

        call_kwargs = mock_create.call_args[1]
        subagents = call_kwargs["subagents"]

        research = next(s for s in subagents if s["name"] == "research-worker")
        tool_names = {t.name for t in research["tools"]}
        assert tool_names == {
            "web_search", "zhihu_search", "weibo_search",
            "weixin_search", "github_search", "scrape_url",
        }
