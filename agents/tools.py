"""Configuration helpers for the deep research agent.

Provides:
- ``load_settings``  — load project settings from env / pydantic-settings.

Note: Tool factories (``make_internet_search``, ``make_scrape_url``) were
removed in the MCP migration. Tools are now dynamically discovered via
:class:`~deep_research_agent.agents.mcp_client.MCPSearchClient`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_settings() -> dict[str, Any]:
    """Load project settings, returning a plain dict of relevant values.

    Returns:
        Dictionary containing orchestration params and model names.
    """
    load_dotenv()

    try:
        from deep_research_agent.config import Settings
        s = Settings.load()
        return {
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
            "planner_model": os.environ.get("PLANNER_MODEL", "gemini-3-flash-preview"),
            "worker_model": os.environ.get("WORKER_MODEL", "gemini-3-flash-preview"),
        }
