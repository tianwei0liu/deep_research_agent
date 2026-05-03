"""Integration tests for agents/tools.py ↔ search_service wiring.

Tests verify that:
- ``make_internet_search`` returns an async callable with Tavily-compat output
- ``make_scrape_url`` returns an async callable with ScrapeResponse output
- ``load_settings`` includes ``search_config``
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from search_service.config import SearchServiceConfig
from search_service.models import SearchResponse, SearchResultItem, SearchEngine, ScrapeResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def search_config() -> SearchServiceConfig:
    """Minimal SearchServiceConfig for testing."""
    return SearchServiceConfig(searxng_base_url="http://test:8080")


def _make_fake_search_response(query: str = "test") -> SearchResponse:
    """Create a realistic SearchResponse for mocking."""
    return SearchResponse(
        query=query,
        results=[
            SearchResultItem(
                title="Test Result 1",
                url="https://example.com/1",
                content="First result content",
                source_engine=SearchEngine.BAIDU,
            ),
            SearchResultItem(
                title="Test Result 2",
                url="https://example.com/2",
                content="Second result content",
                source_engine=SearchEngine.SOGOU,
            ),
        ],
        result_count=2,
        search_time_ms=150,
        engines_used=["baidu", "sogou"],
    )


def _make_fake_scrape_response(url: str = "https://example.com") -> ScrapeResponse:
    """Create a realistic ScrapeResponse for mocking."""
    return ScrapeResponse(
        url=url,
        title="Example Page",
        content="# Example\n\nPage content in markdown.",
        content_length=38,
    )


# ---------------------------------------------------------------------------
# make_internet_search
# ---------------------------------------------------------------------------

class TestMakeInternetSearch:
    """Tests for ``make_internet_search`` factory."""

    def test_returns_callable(self, search_config: SearchServiceConfig) -> None:
        """Factory returns a callable (async function)."""
        from deep_research_agent.agents.tools import make_internet_search

        tool = make_internet_search(search_config)
        assert callable(tool)

    def test_function_name_is_internet_search(self, search_config: SearchServiceConfig) -> None:
        """Inner function name is 'internet_search' for LangGraph ToolNode."""
        from deep_research_agent.agents.tools import make_internet_search

        tool = make_internet_search(search_config)
        assert tool.__name__ == "internet_search"

    def test_function_is_async(self, search_config: SearchServiceConfig) -> None:
        """Inner function is a coroutine function."""
        from deep_research_agent.agents.tools import make_internet_search

        tool = make_internet_search(search_config)
        assert asyncio.iscoroutinefunction(tool)

    @pytest.mark.asyncio
    async def test_returns_tavily_compatible_format(self, search_config: SearchServiceConfig) -> None:
        """Output dict has 'query' and 'results' keys with correct structure."""
        from deep_research_agent.agents.tools import make_internet_search

        fake_response = _make_fake_search_response("AI agent")

        with patch(
            "search_service.backends.base.SearchRouter.search",
            new_callable=AsyncMock,
            return_value=fake_response,
        ):
            tool = make_internet_search(search_config)
            result = await tool("AI agent", max_results=5)

        assert isinstance(result, dict)
        assert "query" in result
        assert "results" in result
        assert result["query"] == "AI agent"
        assert len(result["results"]) == 2

        # Each result has title, url, content
        for r in result["results"]:
            assert "title" in r
            assert "url" in r
            assert "content" in r

    @pytest.mark.asyncio
    async def test_news_topic_routes_to_bing(self, search_config: SearchServiceConfig) -> None:
        """topic='news' passes engines=['bing'] to the router."""
        from deep_research_agent.agents.tools import make_internet_search

        fake_response = _make_fake_search_response("breaking news")

        with patch(
            "search_service.backends.base.SearchRouter.search",
            new_callable=AsyncMock,
            return_value=fake_response,
        ) as mock_search:
            tool = make_internet_search(search_config)
            await tool("breaking news", topic="news")

        _, kwargs = mock_search.call_args
        assert kwargs.get("engines") == ["bing"]

    @pytest.mark.asyncio
    async def test_general_topic_no_engine_filter(self, search_config: SearchServiceConfig) -> None:
        """topic='general' passes engines=None (use SearXNG defaults)."""
        from deep_research_agent.agents.tools import make_internet_search

        fake_response = _make_fake_search_response("test")

        with patch(
            "search_service.backends.base.SearchRouter.search",
            new_callable=AsyncMock,
            return_value=fake_response,
        ) as mock_search:
            tool = make_internet_search(search_config)
            await tool("test", topic="general")

        _, kwargs = mock_search.call_args
        assert kwargs.get("engines") is None


# ---------------------------------------------------------------------------
# make_scrape_url
# ---------------------------------------------------------------------------

class TestMakeScrapeUrl:
    """Tests for ``make_scrape_url`` factory."""

    def test_returns_callable(self, search_config: SearchServiceConfig) -> None:
        """Factory returns a callable."""
        from deep_research_agent.agents.tools import make_scrape_url

        tool = make_scrape_url(search_config)
        assert callable(tool)

    def test_function_name_is_scrape_url(self, search_config: SearchServiceConfig) -> None:
        """Inner function name is 'scrape_url'."""
        from deep_research_agent.agents.tools import make_scrape_url

        tool = make_scrape_url(search_config)
        assert tool.__name__ == "scrape_url"

    def test_function_is_async(self, search_config: SearchServiceConfig) -> None:
        """Inner function is a coroutine function."""
        from deep_research_agent.agents.tools import make_scrape_url

        tool = make_scrape_url(search_config)
        assert asyncio.iscoroutinefunction(tool)

    @pytest.mark.asyncio
    async def test_returns_scrape_response_dict(self, search_config: SearchServiceConfig) -> None:
        """Output dict matches ScrapeResponse schema."""
        from deep_research_agent.agents.tools import make_scrape_url

        fake_response = _make_fake_scrape_response("https://example.com/article")

        with patch(
            "search_service.browser.pool.BrowserPool.start",
            new_callable=AsyncMock,
        ), patch(
            "search_service.backends.page_scraper.PageScraper.scrape",
            new_callable=AsyncMock,
            return_value=fake_response,
        ):
            tool = make_scrape_url(search_config)
            result = await tool("https://example.com/article")

        assert isinstance(result, dict)
        assert result["url"] == "https://example.com/article"
        assert result["title"] == "Example Page"
        assert "content" in result
        assert "content_length" in result

    @pytest.mark.asyncio
    async def test_lazy_browser_pool_start(self, search_config: SearchServiceConfig) -> None:
        """BrowserPool.start() is called only on first invocation."""
        from deep_research_agent.agents.tools import make_scrape_url

        fake_response = _make_fake_scrape_response()

        with patch(
            "search_service.browser.pool.BrowserPool.start",
            new_callable=AsyncMock,
        ) as mock_start, patch(
            "search_service.backends.page_scraper.PageScraper.scrape",
            new_callable=AsyncMock,
            return_value=fake_response,
        ):
            tool = make_scrape_url(search_config)

            # Not started yet
            mock_start.assert_not_called()

            # First call triggers start
            await tool("https://example.com/1")
            assert mock_start.call_count == 1

            # Second call does NOT restart
            await tool("https://example.com/2")
            assert mock_start.call_count == 1


# ---------------------------------------------------------------------------
# load_settings
# ---------------------------------------------------------------------------

class TestLoadSettings:
    """Tests for ``load_settings`` integration."""

    def test_returns_search_config(self) -> None:
        """load_settings always includes a SearchServiceConfig."""
        with patch.dict("os.environ", {}, clear=False):
            from deep_research_agent.agents.tools import load_settings

            result = load_settings()

        assert "search_config" in result
        assert isinstance(result["search_config"], SearchServiceConfig)

    def test_no_tavily_key_required(self) -> None:
        """load_settings no longer requires TAVILY_API_KEY."""
        with patch.dict("os.environ", {}, clear=False):
            from deep_research_agent.agents.tools import load_settings

            result = load_settings()

        # tavily_api_key should NOT be in the result
        assert "tavily_api_key" not in result
