"""Tests for search_service.server — SearchMCPServer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from search_service.config import SearchServiceConfig
from search_service.models import SearchResponse, SearchResultItem, SearchEngine, ScrapeResponse
from search_service.server import SearchMCPServer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> SearchServiceConfig:
    return SearchServiceConfig(
        searxng_base_url="http://localhost:8080",
    )


@pytest.fixture
def server(config: SearchServiceConfig) -> SearchMCPServer:
    return SearchMCPServer(config)


def _make_search_response(query: str = "test", n: int = 3) -> SearchResponse:
    items = [
        SearchResultItem(
            title=f"Result {i}", url=f"https://example.com/{i}",
            content=f"Content {i}", source_engine=SearchEngine.SEARXNG,
        )
        for i in range(n)
    ]
    return SearchResponse(
        query=query, results=items, result_count=n,
        search_time_ms=100, engines_used=["searxng"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSearchMCPServer:
    """SearchMCPServer construction and tool registration."""

    def test_server_construction(self, server: SearchMCPServer) -> None:
        """Server initializes with config and MCP instance."""
        assert server._config is not None
        assert server._mcp is not None

    def test_tools_registered(self, server: SearchMCPServer) -> None:
        """All 6 tools are registered on the MCP server."""
        tools = server._mcp._tool_manager.list_tools()
        tool_names = {t.name for t in tools}
        expected = {
            "web_search", "zhihu_search", "weibo_search",
            "weixin_search", "github_search", "scrape_url",
        }
        assert expected == tool_names


class TestServerLifecycle:
    """Server startup and shutdown."""

    @pytest.mark.asyncio
    async def test_startup_initializes_router(self, server: SearchMCPServer) -> None:
        """startup() initializes router and browser pool."""
        with patch.object(server, "_browser_pool", create=True) as mock_pool:
            # Mock BrowserPool.start
            mock_pool_instance = AsyncMock()
            with patch(
                "search_service.server.BrowserPool",
                return_value=mock_pool_instance,
            ):
                await server.startup()
                assert server._router is not None
                mock_pool_instance.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up(self, server: SearchMCPServer) -> None:
        """shutdown() closes browser pool."""
        mock_pool = AsyncMock()
        server._browser_pool = mock_pool

        await server.shutdown()
        mock_pool.shutdown.assert_called_once()


class TestWebSearchTool:
    """web_search tool via the MCP handler."""

    @pytest.mark.asyncio
    async def test_web_search_returns_results(self, server: SearchMCPServer) -> None:
        """web_search delegates to router and returns dict."""
        mock_router = AsyncMock()
        mock_router.search = AsyncMock(return_value=_make_search_response("AI agent", n=5))
        server._router = mock_router

        # Call the registered tool handler directly
        result = await server._tool_handlers["web_search"](
            query="AI agent", max_results=10, engines="", time_range="",
        )
        assert isinstance(result, dict)
        assert result["result_count"] == 5
        assert result["query"] == "AI agent"


class TestScrapeUrlTool:
    """scrape_url tool."""

    @pytest.mark.asyncio
    async def test_scrape_url_returns_response(self, server: SearchMCPServer) -> None:
        """scrape_url delegates to PageScraper."""
        mock_scraper = AsyncMock()
        mock_scraper.scrape = AsyncMock(return_value=ScrapeResponse(
            url="https://example.com", title="Test",
            content="# Hello", content_length=7,
        ))

        with patch.object(server, "_get_page_scraper", return_value=mock_scraper):
            result = await server._tool_handlers["scrape_url"](
                url="https://example.com", timeout_seconds=15.0,
                max_content_length=50000,
            )
        assert isinstance(result, dict)
        assert result["url"] == "https://example.com"


class TestGithubSearchTool:
    """github_search tool."""

    @pytest.mark.asyncio
    async def test_github_search_returns_response(self, server: SearchMCPServer) -> None:
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value=_make_search_response("langgraph"))

        with patch.object(server, "_get_github_client", return_value=mock_client):
            result = await server._tool_handlers["github_search"](
                query="langgraph", max_results=5, search_type="repositories",
            )
        assert result["query"] == "langgraph"


class TestPlatformFallback:
    """Platform tools fallback to site: search when scrapers not available."""

    @pytest.mark.asyncio
    async def test_zhihu_fallback_to_site_search(self, server: SearchMCPServer) -> None:
        """zhihu_search uses site:zhihu.com fallback in V1."""
        mock_router = AsyncMock()
        mock_router.search = AsyncMock(return_value=_make_search_response("AI"))
        server._router = mock_router

        result = await server._tool_handlers["zhihu_search"](
            query="AI agent", max_results=10,
        )
        assert isinstance(result, dict)
        # Verify router was called with site: prefix
        call_args = mock_router.search.call_args
        assert "site:zhihu.com" in call_args[0][0]
