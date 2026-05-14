# S05: MCP Server + SearchRouter

> **Phase**: 2 | **预估工时**: 2-3 天  
> **产出文件**: `search_service/server.py`  
> **依赖**: S01-S04 (全部)  
> **下游**: S06 (Agent Integration via langchain-mcp-adapters)

---

## 1. 目标

- 使用 FastMCP 实现 MCP Server 入口
- 暴露 6 个搜索工具 (web_search, zhihu_search, weibo_search, weixin_search, github_search, scrape_url)
- 统一生命周期管理 (BrowserPool, httpx clients)
- 工具级错误处理 + 结构化日志

---

## 2. MCP 工具定义

| 工具名 | 描述 | Phase | 后端 |
|:---|:---|:---|:---|
| `web_search` | 通用互联网搜索 | 1 | SearXNG |
| `zhihu_search` | 知乎深度问答搜索 | 2 | ZhihuScraper |
| `weibo_search` | 微博实时搜索 | 2 | WeiboScraper |
| `weixin_search` | 微信公众号文章搜索 | 2 | WeixinScraper |
| `github_search` | GitHub 仓库/代码搜索 | 2 | GitHubClient |
| `scrape_url` | URL 页面内容抓取 | 2 | PageScraper |

---

## 3. Server 实现

```python
from mcp.server.fastmcp import FastMCP

class SearchMCPServer:
    """MCP Search Service 入口。

    管理所有后端生命周期，注册 MCP 工具。
    """

    def __init__(self, config: SearchServiceConfig | None = None):
        self._config = config or SearchServiceConfig()
        self._mcp = FastMCP("search-service")
        self._router: SearchRouter | None = None
        self._browser_pool: BrowserPool | None = None
        self._logger = logging.getLogger(__name__)

        # 注册工具
        self._register_tools()

    def _register_tools(self) -> None:
        """注册所有搜索工具到 MCP。"""

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
            engine_list = [e.strip() for e in engines.split(",") if e.strip()] or None
            response = await self._router.search(
                query, max_results=max_results,
                engines=engine_list, time_range=time_range,
            )
            return response.model_dump()

        @self._mcp.tool()
        async def zhihu_search(query: str, max_results: int = 10) -> dict:
            """Search Zhihu for in-depth Q&A and expert opinions.

            Best for: technical discussions, industry analysis, expert perspectives.

            Args:
                query: Search query in Chinese for best results.
                max_results: Max results.
            """
            scraper = self._get_zhihu_scraper()
            try:
                response = await scraper.search(query, max_results)
                return response.model_dump()
            except CookieExpiredError:
                self._logger.warning("zhihu_cookie_expired, using site: fallback")
                assert self._router is not None
                fallback = await self._router.search(f"site:zhihu.com {query}", max_results=max_results)
                return fallback.model_dump()

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
            scraper = self._get_weibo_scraper()
            try:
                response = await scraper.search(query, max_results, time_scope)
                return response.model_dump()
            except CookieExpiredError:
                self._logger.warning("weibo_cookie_expired, using site: fallback")
                assert self._router is not None
                fallback = await self._router.search(f"site:weibo.com {query}", max_results=max_results)
                return fallback.model_dump()

        @self._mcp.tool()
        async def weixin_search(query: str, max_results: int = 10) -> dict:
            """Search WeChat Official Account articles via Sogou.

            Best for: industry analysis, policy interpretation, expert long-form content.
            This is the ONLY way to search WeChat content from outside the WeChat app.

            Args:
                query: Search query in Chinese.
                max_results: Max results.
            """
            scraper = self._get_weixin_scraper()
            response = await scraper.search(query, max_results)
            return response.model_dump()

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
                max_content_length: Max content chars.
            """
            scraper = self._get_page_scraper()
            response = await scraper.scrape(url, timeout_seconds, max_content_length)
            return response.model_dump()

    # --- Lifecycle ---

    async def startup(self) -> None:
        """初始化所有后端。"""
        cache = NullCache()  # V1
        searxng = SearXNGClient(self._config)
        self._router = SearchRouter(backends=[searxng], cache=cache)

        self._browser_pool = BrowserPool(self._config)
        await self._browser_pool.start()

        self._logger.info("search_service_started")

    async def shutdown(self) -> None:
        """清理所有资源。"""
        if self._browser_pool:
            await self._browser_pool.shutdown()
        self._logger.info("search_service_stopped")

    def run(self) -> None:
        """启动 MCP Server (stdio 模式)。"""
        import asyncio
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        await self.startup()
        try:
            await self._mcp.run_async(transport="stdio")
        finally:
            await self.shutdown()

    # --- Backend accessors (懒初始化) ---

    def _get_zhihu_scraper(self) -> ZhihuScraper:
        assert self._browser_pool is not None
        return ZhihuScraper(self._browser_pool, self._config)

    def _get_weibo_scraper(self) -> WeiboScraper:
        assert self._browser_pool is not None
        return WeiboScraper(self._browser_pool, self._config)

    def _get_weixin_scraper(self) -> WeixinScraper:
        assert self._browser_pool is not None
        return WeixinScraper(self._browser_pool, self._config)

    def _get_github_client(self) -> GitHubClient:
        return GitHubClient(self._config)

    def _get_page_scraper(self) -> PageScraper:
        assert self._browser_pool is not None
        return PageScraper(self._browser_pool)
```

---

## 4. 运行方式

```bash
# stdio 模式 (langchain-mcp-adapters 使用)
python -m search_service.server

# __main__.py 入口
# search_service/__main__.py
if __name__ == "__main__":
    server = SearchMCPServer()
    server.run()
```

---

## 5. 降级策略汇总

| 工具 | 故障场景 | 降级行为 |
|:---|:---|:---|
| `web_search` | SearXNG 宕机 | AllProvidersExhaustedError → Agent 报告 |
| `zhihu_search` | Cookie 过期 | → `web_search("site:zhihu.com ...")` |
| `weibo_search` | Cookie 过期 | → `web_search("site:weibo.com ...")` |
| `weixin_search` | 搜狗验证码 | RateLimitedError → Agent 等待重试 |
| `github_search` | API 限流 | RateLimitedError + retry_after |
| `scrape_url` | 页面超时 | ContentExtractionError → Agent 跳过 |

---

## 6. 验收标准

```bash
# 1. 启动 MCP Server
python -m search_service.server &

# 2. 通过 langchain-mcp-adapters 测试
python -c "
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def test():
    client = MultiServerMCPClient(servers=[
        {'command': 'python', 'args': ['-m', 'search_service.server'], 'transport': 'stdio'}
    ])
    tools = await client.get_tools()
    print(f'Registered tools: {[t.name for t in tools]}')
    # 期望: ['web_search', 'zhihu_search', 'weibo_search',
    #         'weixin_search', 'github_search', 'scrape_url']

asyncio.run(test())
"
```

| 验收项 | 目标 |
|:---|:---|
| 工具注册数 | 6 个 |
| web_search 可用 | 返回 ≥5 条结果 |
| scrape_url 可用 | 返回 Markdown 内容 |
| 生命周期 | startup/shutdown 无泄漏 |
| 降级测试 | Cookie 过期时自动 fallback |
