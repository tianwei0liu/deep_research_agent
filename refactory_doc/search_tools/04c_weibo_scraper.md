# S04c: Weibo Scraper (微博搜索)

> **Phase**: 2 | **预估工时**: 1-2 天  
> **产出文件**: `search_service/backends/weibo_scraper.py`  
> **依赖**: S01 (models, exceptions), S03 (BrowserPool)  
> **下游**: S05 (MCP Server 暴露 `weibo_search` 工具)

---

## 1. 目标

- 通过 Playwright 爬取微博搜索结果
- Cookie 持久化 + 过期检测 + `site:weibo.com` 降级
- Rate limiting (≤10 RPM)
- 时效性内容优先（微博的核心价值）

---

## 2. 核心接口

```python
class WeiboScraper:
    """微博搜索爬虫。

    搜索入口: https://s.weibo.com/weibo?q={query}
    需要登录 Cookie 访问搜索功能。

    微博特点:
    - 内容时效性极强（分钟级更新）
    - 搜索结果包含转发数/评论数/点赞数
    - 反爬较强（需 stealth.js + Cookie）
    """

    SEARCH_URL = "https://s.weibo.com/weibo"

    def __init__(self, browser_pool: BrowserPool, config: SearchServiceConfig):
        self._pool = browser_pool
        self._cookie_dir = config.cookie_storage_dir / "weibo"
        self._rpm_limit = config.weibo_rpm
        self._rate_limiter = AsyncRateLimiter(max_calls=self._rpm_limit, period=60)
        self._logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "weibo"

    async def search(
        self,
        query: str,
        max_results: int = 10,
        time_scope: str = "",
    ) -> SearchResponse:
        """搜索微博。

        Args:
            query: 搜索关键词。
            max_results: 最大返回结果数。
            time_scope: 时间范围 — "" (全部), "hour" (1小时内),
                       "day" (24小时内), "week" (一周内)。

        Returns:
            SearchResponse
        """
        await self._rate_limiter.acquire()

        cookies = await self._load_cookies()
        if not cookies:
            raise CookieExpiredError("weibo")

        async with self._pool.acquire(cookies=cookies) as context:
            page = await context.new_page()
            try:
                params = f"?q={query}"
                if time_scope:
                    scope_map = {"hour": "1", "day": "2", "week": "3"}
                    params += f"&timescope=custom:{scope_map.get(time_scope, '')}"

                await page.goto(
                    f"{self.SEARCH_URL}{params}",
                    wait_until="networkidle",
                    timeout=15000,
                )

                if await self._is_login_required(page):
                    raise CookieExpiredError("weibo")

                results = await self._extract_results(page, max_results)

                return SearchResponse(
                    query=query, results=results,
                    result_count=len(results),
                    search_time_ms=0, engines_used=["weibo"],
                )
            finally:
                await page.close()

    async def _extract_results(self, page, max_results: int) -> list[SearchResultItem]:
        """提取微博搜索结果。"""
        items = []
        cards = await page.query_selector_all(".card-wrap[mid]")

        for card in cards[:max_results]:
            try:
                content_el = await card.query_selector(".txt")
                author_el = await card.query_selector(".name")
                time_el = await card.query_selector(".from a")

                content = await content_el.inner_text() if content_el else ""
                author = await author_el.inner_text() if author_el else ""
                pub_time = await time_el.inner_text() if time_el else ""

                # 构建微博详情 URL
                mid = await card.get_attribute("mid") or ""
                url = f"https://weibo.com/{mid}" if mid else ""

                metadata = {
                    "author": author.strip(),
                    "publish_time": pub_time.strip(),
                }

                # 提取互动数据
                act_el = await card.query_selector(".card-act")
                if act_el:
                    act_text = await act_el.inner_text()
                    metadata["engagement"] = act_text.strip()

                items.append(SearchResultItem(
                    title=f"@{author.strip()}: {content.strip()[:50]}...",
                    url=url,
                    content=content.strip()[:500],
                    source_engine=SearchEngine.WEIBO,
                    published_date=pub_time.strip(),
                    metadata=metadata,
                ))
            except Exception as e:
                self._logger.debug("weibo_card_parse_error", extra={"error": str(e)})

        return items

    async def _is_login_required(self, page) -> bool:
        """检测登录态。"""
        # 微博未登录时搜索页会跳转到 passport.weibo.com
        current_url = page.url
        return "passport.weibo.com" in current_url or "login" in current_url

    async def _load_cookies(self) -> list[dict] | None:
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

## 3. 微博搜索特性

| 特性 | 说明 |
|:---|:---|
| **时效性** | 搜索结果按时间排序，分钟级更新 |
| **time_scope** | 支持 1h/24h/1week 时间过滤 |
| **互动数据** | 转发/评论/点赞，体现内容影响力 |
| **反爬** | 频率过高触发验证码，需严格控制 RPM |
| **Cookie 有效期** | 通常 7-30 天 |

---

## 4. 验收标准

| 测试用例 | 说明 |
|:---|:---|
| `test_search_with_valid_cookie` | 有效 Cookie → 返回结果 |
| `test_time_scope_filter` | time_scope="day" 过滤生效 |
| `test_cookie_expired_redirect` | 重定向到 passport → CookieExpiredError |
| `test_rate_limiting` | 超过 10 RPM → 等待 |
| `test_metadata_extraction` | 作者/时间/互动数据提取 |
