"""MCP Search Service server entry point.

Exposes 6 search tools via the Model Context Protocol (FastMCP).
Manages lifecycle of SearXNG client, BrowserPool, and all backends.

V1 tools: web_search, github_search, scrape_url (fully functional).
V1 fallback: zhihu_search, weibo_search, weixin_search use site: prefix
via SearXNG; platform scrapers are implemented in Phase 2.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

from search_service.backends.base import SearchRouter
from search_service.backends.github_client import GitHubClient
from search_service.backends.page_scraper import PageScraper
from search_service.backends.searxng_client import SearXNGClient
from search_service.browser.pool import BrowserPool
from search_service.cache import NullCache
from search_service.config import SearchServiceConfig


class SearchMCPServer:
    """MCP Search Service server.

    Registers 6 search tools and manages backend lifecycle.

    Args:
        config: Search service configuration. Uses defaults if None.
    """

    def __init__(self, config: Optional[SearchServiceConfig] = None) -> None:
        self._config = config or SearchServiceConfig()
        self._mcp = FastMCP("search-service")
        self._router: Optional[SearchRouter] = None
        self._browser_pool: Optional[BrowserPool] = None
        self._logger = logging.getLogger(__name__)
        self._tool_handlers: dict = {}

        self._register_tools()

    def _register_tools(self) -> None:
        """Register all search tools on the MCP server."""
        # --- web_search ---
        @self._mcp.tool()
        async def web_search(
            query: str,
            max_results: int = 10,
            engines: str = "",
            time_range: str = "",
        ) -> dict:
            """Search the internet for current information.

            Supports site: operator for targeted search.
            Examples:
              - "AI agent framework 2026"
              - "site:arxiv.org multi-agent research"
              - "site:36kr.com OR site:huxiu.com AI产业分析"

            Args:
                query: Search query with optional site: prefix.
                max_results: Max results (1-20).
                engines: Comma-separated engine list (e.g. "bing,baidu").
                time_range: Time filter — "", "day", "week", "month", "year".

            Returns:
                Search results with titles, urls, and content snippets.
            """
            assert self._router is not None
            engine_list = (
                [e.strip() for e in engines.split(",") if e.strip()]
                or None
            )
            response = await self._router.search(
                query, max_results=max_results,
                engines=engine_list, time_range=time_range,
            )
            return response.model_dump()

        # --- zhihu_search (V1: site: fallback) ---
        @self._mcp.tool()
        async def zhihu_search(
            query: str, max_results: int = 10,
        ) -> dict:
            """Search Zhihu for in-depth Q&A and expert opinions.

            Best for: technical discussions, industry analysis, expert perspectives.

            Args:
                query: Search query in Chinese for best results.
                max_results: Max results.
            """
            assert self._router is not None
            # V1: site: prefix fallback (ZhihuScraper in Phase 2)
            response = await self._router.search(
                f"site:zhihu.com {query}", max_results=max_results,
            )
            return response.model_dump()

        # --- weibo_search (V1: site: fallback) ---
        @self._mcp.tool()
        async def weibo_search(
            query: str,
            max_results: int = 10,
            time_scope: str = "",
        ) -> dict:
            """Search Weibo for real-time trending content.

            Best for: breaking news, public opinion, trending topics.

            Args:
                query: Search query.
                max_results: Max results.
                time_scope: Time filter — "" (all), "hour", "day", "week".
            """
            assert self._router is not None
            # V1: site: prefix fallback (WeiboScraper in Phase 2)
            response = await self._router.search(
                f"site:weibo.com {query}", max_results=max_results,
            )
            return response.model_dump()

        # --- weixin_search (V1: site: fallback) ---
        @self._mcp.tool()
        async def weixin_search(
            query: str, max_results: int = 10,
        ) -> dict:
            """Search WeChat Official Account articles via Sogou.

            Best for: industry analysis, policy interpretation, long-form content.
            This is the ONLY way to search WeChat content from outside the app.

            Args:
                query: Search query in Chinese.
                max_results: Max results.
            """
            assert self._router is not None
            # V1: site: prefix fallback (WeixinScraper in Phase 2)
            response = await self._router.search(
                f"site:mp.weixin.qq.com {query}", max_results=max_results,
            )
            return response.model_dump()

        # --- github_search ---
        @self._mcp.tool()
        async def github_search(
            query: str,
            max_results: int = 10,
            search_type: str = "repositories",
        ) -> dict:
            """Search GitHub for repositories and code.

            Supports GitHub search syntax.
            Examples: "LangGraph agent language:python stars:>100"

            Args:
                query: GitHub search query.
                max_results: Max results (1-30).
                search_type: "repositories" or "code".
            """
            client = self._get_github_client()
            response = await client.search(query, max_results, search_type)
            return response.model_dump()

        # --- scrape_url ---
        @self._mcp.tool()
        async def scrape_url(
            url: str,
            timeout_seconds: float = 15.0,
            max_content_length: int = 50000,
        ) -> dict:
            """Extract content from a URL as Markdown.

            Renders JavaScript-heavy pages. Removes nav, ads, sidebars.

            Args:
                url: Target URL to scrape.
                timeout_seconds: Page load timeout.
                max_content_length: Max content characters.
            """
            scraper = self._get_page_scraper()
            response = await scraper.scrape(
                url, timeout_seconds, max_content_length,
            )
            return response.model_dump()

        # Store handlers for direct testing
        self._tool_handlers = {
            "web_search": web_search,
            "zhihu_search": zhihu_search,
            "weibo_search": weibo_search,
            "weixin_search": weixin_search,
            "github_search": github_search,
            "scrape_url": scrape_url,
        }

    # --- Lifecycle ---

    async def startup(self) -> None:
        """Initialize all backends."""
        cache = NullCache()
        searxng = SearXNGClient(self._config)
        self._router = SearchRouter(backends=[searxng], cache=cache)

        self._browser_pool = BrowserPool(self._config)
        await self._browser_pool.start()

        self._logger.info("search_service_started")

    async def shutdown(self) -> None:
        """Clean up all resources."""
        if self._browser_pool:
            await self._browser_pool.shutdown()
        self._logger.info("search_service_stopped")

    def run(self) -> None:
        """Start MCP Server in stdio mode."""
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        """Async entry point: startup → serve → shutdown."""
        await self.startup()
        try:
            await self._mcp.run_async(transport="stdio")
        finally:
            await self.shutdown()

    # --- Backend accessors ---

    def _get_github_client(self) -> GitHubClient:
        """Get a GitHubClient instance."""
        return GitHubClient(self._config)

    def _get_page_scraper(self) -> PageScraper:
        """Get a PageScraper instance."""
        assert self._browser_pool is not None, "Call startup() first"
        return PageScraper(self._browser_pool)
