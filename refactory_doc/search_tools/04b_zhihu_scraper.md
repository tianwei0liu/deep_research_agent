# S04b: Zhihu Scraper (知乎搜索)

> **Phase**: 2 | **预估工时**: 1-2 天  
> **产出文件**: `search_service/backends/zhihu_scraper.py`  
> **依赖**: S01 (models, exceptions), S03 (BrowserPool)  
> **下游**: S05 (MCP Server 暴露 `zhihu_search` 工具)

---

## 1. 目标

- 通过 Playwright 爬取知乎搜索结果
- Cookie 持久化管理（登录态保持）
- Cookie 过期自动检测 + `site:zhihu.com` 降级
- Rate limiting (≤12 RPM)

---

## 2. 核心接口

```python
class ZhihuScraper:
    """知乎搜索爬虫。

    搜索入口: https://www.zhihu.com/search?type=content&q={query}
    需要登录 Cookie 才能访问完整内容。

    降级策略: Cookie 过期时回退到 SearXNG 的 site:zhihu.com 搜索。
    """

    SEARCH_URL = "https://www.zhihu.com/search"

    def __init__(self, browser_pool: BrowserPool, config: SearchServiceConfig):
        self._pool = browser_pool
        self._cookie_dir = config.cookie_storage_dir / "zhihu"
        self._rpm_limit = config.zhihu_rpm
        self._rate_limiter = AsyncRateLimiter(max_calls=self._rpm_limit, period=60)
        self._logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "zhihu"

    async def search(self, query: str, max_results: int = 10) -> SearchResponse:
        """搜索知乎。

        Args:
            query: 搜索关键词。
            max_results: 最大返回结果数。

        Returns:
            SearchResponse

        Raises:
            CookieExpiredError: Cookie 过期。
            RateLimitedError: 超过 RPM 限制。
        """
        await self._rate_limiter.acquire()

        cookies = await self._load_cookies()
        if not cookies:
            raise CookieExpiredError("zhihu")

        async with self._pool.acquire(cookies=cookies) as context:
            page = await context.new_page()
            try:
                await page.goto(
                    f"{self.SEARCH_URL}?type=content&q={query}",
                    wait_until="networkidle",
                    timeout=15000,
                )

                # 检测登录态
                if await self._is_login_required(page):
                    raise CookieExpiredError("zhihu")

                # 提取搜索结果
                results = await self._extract_results(page, max_results)

                return SearchResponse(
                    query=query,
                    results=results,
                    result_count=len(results),
                    search_time_ms=0,
                    engines_used=["zhihu"],
                )
            finally:
                await page.close()

    async def _extract_results(self, page, max_results: int) -> list[SearchResultItem]:
        """提取知乎搜索结果列表。"""
        items = []
        cards = await page.query_selector_all(".SearchResult-Card")

        for card in cards[:max_results]:
            try:
                title_el = await card.query_selector("h2")
                content_el = await card.query_selector(".RichText")
                link_el = await card.query_selector("a[data-za-detail-view-element_name='Title']")

                title = await title_el.inner_text() if title_el else ""
                content = await content_el.inner_text() if content_el else ""
                href = await link_el.get_attribute("href") if link_el else ""

                # 知乎内部链接转换
                url = href if href.startswith("http") else f"https://www.zhihu.com{href}"

                # 提取元数据 (赞同数、作者)
                metadata = await self._extract_metadata(card)

                items.append(SearchResultItem(
                    title=title.strip(),
                    url=url,
                    content=content.strip()[:500],
                    source_engine=SearchEngine.ZHIHU,
                    metadata=metadata,
                ))
            except Exception as e:
                self._logger.debug("zhihu_card_parse_error", extra={"error": str(e)})

        return items

    async def _extract_metadata(self, card) -> dict:
        """提取赞同数、作者等元数据。"""
        metadata: dict = {}
        try:
            vote_el = await card.query_selector(".VoteButton--up")
            if vote_el:
                vote_text = await vote_el.inner_text()
                metadata["upvotes"] = vote_text
            author_el = await card.query_selector(".AuthorInfo-name")
            if author_el:
                metadata["author"] = await author_el.inner_text()
        except Exception:
            pass
        return metadata

    async def _is_login_required(self, page) -> bool:
        """检测是否需要登录。"""
        login_modal = await page.query_selector(".Modal-backdrop")
        login_button = await page.query_selector("[data-za-detail-view-element_name='SignInButton']")
        return login_modal is not None or login_button is not None

    async def _load_cookies(self) -> list[dict] | None:
        """从磁盘加载 Cookie。"""
        cookie_file = self._cookie_dir / "cookies.json"
        if not cookie_file.exists():
            return None
        import json
        return json.loads(cookie_file.read_text())

    async def health_check(self) -> bool:
        cookies = await self._load_cookies()
        return cookies is not None
```

---

## 3. Cookie 管理

### Cookie 初始获取（手动）

```bash
# 1. 用真实浏览器登录知乎
# 2. 导出 Cookie 为 JSON 格式 (可用 EditThisCookie 扩展)
# 3. 保存到 data/cookies/zhihu/cookies.json
```

### Cookie 格式

```json
[
    {
        "name": "z_c0",
        "value": "...",
        "domain": ".zhihu.com",
        "path": "/",
        "httpOnly": true,
        "secure": true
    }
]
```

### Cookie 过期检测

知乎 Cookie 过期特征：
1. 页面出现登录弹窗 (`.Modal-backdrop`)
2. 搜索结果被截断/隐藏
3. HTTP 302 重定向到登录页

---

## 4. 降级策略

```python
# 在 SearchRouter 或 MCP Tool 层实现降级
async def zhihu_search_with_fallback(query, router, zhihu_scraper):
    try:
        return await zhihu_scraper.search(query)
    except CookieExpiredError:
        logger.warning("zhihu_cookie_expired, falling back to site: search")
        return await router.search(f"site:zhihu.com {query}")
```

---

## 5. 验收标准

| 测试用例 | 说明 |
|:---|:---|
| `test_search_with_valid_cookie` | 有效 Cookie → 返回搜索结果 |
| `test_cookie_expired_detection` | 过期 Cookie → CookieExpiredError |
| `test_no_cookie_file` | 无 Cookie 文件 → CookieExpiredError |
| `test_rate_limiting` | 超过 12 RPM → 等待 |
| `test_result_parsing` | 标题/内容/URL 正确提取 |
| `test_metadata_extraction` | upvotes/author 元数据提取 |
