"""Tests for agent integration (build_deep_agent modifications).

Verifies that build_deep_agent() correctly registers citation
subagent, middleware, and response_format.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Patch targets: _load_settings and _make_internet_search are module-level,
# but create_deep_agent is imported inside build_deep_agent from deepagents.
_SETTINGS_PATCH = "deep_research_agent.agents.agent._load_settings"
_SEARCH_PATCH = "deep_research_agent.agents.agent._make_internet_search"
_CREATE_PATCH = "deepagents.create_deep_agent"

_FAKE_SETTINGS: dict[str, Any] = {
    "tavily_api_key": "fake",
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


class TestBuildDeepAgent:
    """Tests for citation integration in build_deep_agent()."""

    @patch(_SETTINGS_PATCH, return_value=_FAKE_SETTINGS)
    @patch(_SEARCH_PATCH, return_value=MagicMock())
    @patch(_CREATE_PATCH, return_value=MagicMock())
    def test_research_worker_has_response_format(
        self,
        mock_create: MagicMock,
        mock_search: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """research-worker subagent spec includes response_format=WorkerOutput."""
        from deep_research_agent.agents.citation.models import WorkerOutput
        from deep_research_agent.agents.agent import build_deep_agent

        build_deep_agent()

        call_kwargs = mock_create.call_args[1]
        subagents = call_kwargs["subagents"]

        research = next(s for s in subagents if s["name"] == "research-worker")
        assert research["response_format"] is WorkerOutput

    @patch(_SETTINGS_PATCH, return_value=_FAKE_SETTINGS)
    @patch(_SEARCH_PATCH, return_value=MagicMock())
    @patch(_CREATE_PATCH, return_value=MagicMock())
    def test_citation_specialist_registered(
        self,
        mock_create: MagicMock,
        mock_search: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """citation-specialist subagent is registered with correct spec."""
        from deep_research_agent.agents.agent import build_deep_agent

        build_deep_agent()

        call_kwargs = mock_create.call_args[1]
        subagents = call_kwargs["subagents"]

        citation = next(s for s in subagents if s["name"] == "citation-specialist")
        # citation-specialist is a CompiledSubAgent (has 'runnable', not 'tools')
        assert "runnable" in citation
        assert "response_format" not in citation

    @patch(_SETTINGS_PATCH, return_value=_FAKE_SETTINGS)
    @patch(_SEARCH_PATCH, return_value=MagicMock())
    @patch(_CREATE_PATCH, return_value=MagicMock())
    def test_citation_middleware_registered(
        self,
        mock_create: MagicMock,
        mock_search: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """CitationDataMiddleware is registered in create_deep_agent middleware."""
        from deep_research_agent.agents.citation.citation_middleware import (
            CitationDataMiddleware,
        )
        from deep_research_agent.agents.agent import build_deep_agent

        build_deep_agent()

        call_kwargs = mock_create.call_args[1]
        middleware = call_kwargs.get("middleware", [])
        assert any(isinstance(m, CitationDataMiddleware) for m in middleware)

    @patch(_SETTINGS_PATCH, return_value=_FAKE_SETTINGS)
    @patch(_SEARCH_PATCH, return_value=MagicMock())
    @patch(_CREATE_PATCH, return_value=MagicMock())
    def test_two_subagents_registered(
        self,
        mock_create: MagicMock,
        mock_search: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Both research-worker and citation-specialist subagents are registered."""
        from deep_research_agent.agents.agent import build_deep_agent

        build_deep_agent()

        call_kwargs = mock_create.call_args[1]
        subagents = call_kwargs["subagents"]

        names = {s["name"] for s in subagents}
        assert "research-worker" in names
        assert "citation-specialist" in names
        assert len(subagents) == 2
