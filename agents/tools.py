"""Tool factories and configuration helpers for the deep research agent.

Provides:
- ``load_settings``  — load project settings from env / pydantic-settings.
- ``make_internet_search`` — create a Tavily search tool callable.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_settings() -> dict[str, Any]:
    """Load project settings, returning a plain dict of relevant values."""
    load_dotenv()
    try:
        from deep_research_agent.config import Settings
        s = Settings.load()
        return {
            "tavily_api_key": s.tavily_api_key,
            "planner_model": s.planner_model,
            "worker_model": s.worker_model,
            # Orchestration limits
            "supervisor_max_turns": s.supervisor_max_turns,
            "supervisor_max_search_calls": s.supervisor_max_search_calls,
            "worker_max_search_calls": s.worker_max_search_calls,
            "worker_max_turns": s.worker_max_turns,
            # Citation limits
            "citation_max_retries": s.citation_max_retries,
            # Runtime safety
            "research_timeout_seconds": s.research_timeout_seconds,
        }
    except Exception:
        # Fallback: read from environment directly
        return {
            "tavily_api_key": os.environ.get("TAVILY_API_KEY", ""),
            "planner_model": os.environ.get("PLANNER_MODEL", "gemini-3-flash-preview"),
            "worker_model": os.environ.get("WORKER_MODEL", "gemini-3-flash-preview"),
        }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def make_internet_search(api_key: str):
    """Create a Tavily search function compatible with create_deep_agent.

    Args:
        api_key: Tavily API key for authentication.

    Returns:
        A callable that performs internet searches via Tavily.
    """
    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)

    def internet_search(
        query: str,
        max_results: int = 10,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
    ) -> dict[str, Any]:
        """Search the internet for current, factual information.

        Args:
            query: The search query.
            max_results: Maximum number of results (1-20).
            topic: Search topic — general, news, or finance.
            include_raw_content: Whether to include raw page content.

        Returns:
            Search results dictionary with titles, urls, and content.
        """
        return client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )

    return internet_search
