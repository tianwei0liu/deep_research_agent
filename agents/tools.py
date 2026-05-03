"""Tool factories and configuration helpers for the deep research agent.

Provides:
- ``load_settings``  — load project settings from env / pydantic-settings.
- ``make_internet_search`` — create a SearXNG-backed async search callable.
- ``make_scrape_url`` — create a Playwright-backed async URL scraper callable.
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
    """Load project settings, returning a plain dict of relevant values.

    Returns:
        Dictionary containing SearchServiceConfig plus orchestration params.
    """
    load_dotenv()

    from search_service.config import SearchServiceConfig
    search_config = SearchServiceConfig()

    try:
        from deep_research_agent.config import Settings
        s = Settings.load()
        return {
            "search_config": search_config,
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
            "search_config": search_config,
            "planner_model": os.environ.get("PLANNER_MODEL", "gemini-3-flash-preview"),
            "worker_model": os.environ.get("WORKER_MODEL", "gemini-3-flash-preview"),
        }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def make_internet_search(config: Any) -> Any:
    """Create an async search function backed by SearXNG.

    Returns a synchronous factory that produces an async callable.
    The SearXNG client is lazily initialized on first use.
    The returned function's signature and output format are
    backward-compatible with the former Tavily-based implementation.

    Args:
        config: SearchServiceConfig instance for SearXNG connection params.

    Returns:
        An async callable that performs internet searches via SearXNG.
    """
    from search_service.backends.searxng_client import SearXNGClient
    from search_service.backends.base import SearchRouter
    from search_service.cache import NullCache

    client = SearXNGClient(config)
    router = SearchRouter(backends=[client], cache=NullCache())

    async def internet_search(
        query: str,
        max_results: int = 10,
        topic: Literal["general", "news", "finance"] = "general",
    ) -> dict[str, Any]:
        """Search the internet for current, factual information.

        Use this tool to find up-to-date information about any topic.
        For targeted searches, use site: prefix (e.g. "site:arxiv.org LLM agents").

        Args:
            query: The search query. Supports site: operator for targeted search.
            max_results: Maximum number of results (1-20).
            topic: Search topic hint — general, news, or finance.

        Returns:
            Search results dictionary with titles, urls, and content.
        """
        # Route news queries through Bing for better coverage
        engines = None
        if topic == "news":
            engines = ["bing"]

        response = await router.search(
            query, max_results=max_results, engines=engines,
        )

        # Tavily-compatible output format (Worker prompts depend on this)
        return {
            "query": response.query,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "content": r.content,
                }
                for r in response.results
            ],
        }

    return internet_search


def make_scrape_url(config: Any) -> Any:
    """Create an async URL scraping function backed by Playwright.

    Uses BrowserPool + PageScraper from the search_service package.
    BrowserPool is lazily started on first invocation to avoid
    spawning Chromium processes unless actually needed.

    Args:
        config: SearchServiceConfig instance for browser pool params.

    Returns:
        An async callable that scrapes a URL and returns Markdown content.
    """
    from search_service.backends.page_scraper import PageScraper
    from search_service.browser.pool import BrowserPool

    pool = BrowserPool(config)
    _started = False

    async def scrape_url(
        url: str,
        timeout_seconds: float = 15.0,
        max_content_length: int = 50000,
    ) -> dict[str, Any]:
        """Extract content from a URL as clean Markdown.

        Renders JavaScript-heavy pages using a headless browser.
        Removes navigation, ads, and sidebars for clean content extraction.

        Use this AFTER internet_search to deep-read high-value URLs
        that contain important detailed information.

        Args:
            url: Target URL to scrape.
            timeout_seconds: Page load timeout.
            max_content_length: Maximum content characters to return.

        Returns:
            Dictionary with url, title, content (Markdown), and content_length.
        """
        nonlocal _started
        if not _started:
            await pool.start()
            _started = True

        scraper = PageScraper(pool)
        response = await scraper.scrape(url, timeout_seconds, max_content_length)
        return response.model_dump()

    return scrape_url
