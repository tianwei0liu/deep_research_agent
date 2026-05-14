# S04d: WeChat Official Account Scraper (微信公众号搜索)

> **Phase**: 2 | **预估工时**: 1-2 天  
> **产出文件**: `search_service/backends/weixin_scraper.py`  
> **依赖**: S01 (models, exceptions), S03 (BrowserPool)  
> **下游**: S05 (MCP Server 暴露 `weixin_search` 工具)

---

## 1. 目标

- 通过搜狗微信搜索入口 (`weixin.sogou.com`) 搜索公众号文章
- 不需要微信登录 Cookie（搜狗是公开入口）
- 提取文章标题、摘要、公众号名称、发布时间
- Rate limiting (≤8 RPM，搜狗反爬较严)

---

## 2. 背景

**为什么通过搜狗？**
- 微信公众号内容被封锁在微信生态内，不被百度/Bing 索引
- 搜狗是唯一拥有微信公众号搜索授权的外部搜索引擎
- 入口: `https://weixin.sogou.com/weixin?type=2&query={q}`
  - `type=1`: 搜索公众号
  - `type=2`: 搜索文章 ← 我们使用这个

---

## 3. 核心接口

```python
class WeixinScraper:
    """微信公众号文章搜索 (通过搜狗微信搜索)。

    搜索入口: https://weixin.sogou.com/weixin?type=2&query={q}
    不需要微信 Cookie，但需要应对搜狗的反爬机制。

    反爬特征:
    - 高频请求触发验证码
    - IP 频率限制（建议 ≤8 RPM）
    - 搜索结果链接为搜狗跳转链接，需二次解析获取原始 URL
    """

    SEARCH_URL = "https://weixin.sogou.com/weixin"

    def __init__(self, browser_pool: BrowserPool, config: SearchServiceConfig):
        self._pool = browser_pool
        self._rpm_limit = config.weixin_rpm
        self._rate_limiter = AsyncRateLimiter(max_calls=self._rpm_limit, period=60)
        self._logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "weixin"

    async def search(self, query: str, max_results: int = 10) -> SearchResponse:
        """搜索微信公众号文章。

        Args:
            query: 搜索关键词。
            max_results: 最大返回结果数。

        Returns:
            SearchResponse

        Raises:
            RateLimitedError: 触发搜狗验证码。
        """
        await self._rate_limiter.acquire()

        async with self._pool.acquire() as context:
            page = await context.new_page()
            try:
                await page.goto(
                    f"{self.SEARCH_URL}?type=2&query={query}",
                    wait_until="domcontentloaded",
                    timeout=15000,
                )

                # 检测验证码
                if await self._is_captcha_page(page):
                    raise RateLimitedError("weixin_sogou", retry_after_seconds=300)

                results = await self._extract_results(page, max_results)

                return SearchResponse(
                    query=query, results=results,
                    result_count=len(results),
                    search_time_ms=0, engines_used=["weixin"],
                )
            finally:
                await page.close()

    async def _extract_results(self, page, max_results: int) -> list[SearchResultItem]:
        """提取搜狗微信搜索结果。"""
        items = []
        cards = await page.query_selector_all(".news-list li")

        for card in cards[:max_results]:
            try:
                # 标题和链接
                title_el = await card.query_selector("h3 a")
                title = await title_el.inner_text() if title_el else ""
                sogou_url = await title_el.get_attribute("href") if title_el else ""

                # 搜狗链接为跳转链接，需获取实际微信文章 URL
                url = sogou_url if sogou_url.startswith("http") else f"https://weixin.sogou.com{sogou_url}"

                # 摘要
                summary_el = await card.query_selector(".txt-info")
                content = await summary_el.inner_text() if summary_el else ""

                # 公众号名称
                account_el = await card.query_selector(".s-p a")
                account_name = await account_el.inner_text() if account_el else ""

                # 发布时间
                time_el = await card.query_selector(".s-p")
                pub_time = ""
                if time_el:
                    time_text = await time_el.inner_text()
                    # 格式: "公众号名 日期"
                    parts = time_text.split()
                    if len(parts) > 1:
                        pub_time = parts[-1]

                items.append(SearchResultItem(
                    title=title.strip(),
                    url=url,
                    content=content.strip()[:500],
                    source_engine=SearchEngine.WEIXIN,
                    published_date=pub_time,
                    metadata={
                        "account_name": account_name.strip(),
                        "sogou_url": sogou_url,
                    },
                ))
            except Exception as e:
                self._logger.debug("weixin_card_parse_error", extra={"error": str(e)})

        return items

    async def _is_captcha_page(self, page) -> bool:
        """检测搜狗验证码页面。"""
        captcha = await page.query_selector("#seccodeImage")
        antispider = await page.query_selector(".antispider")
        return captcha is not None or antispider is not None

    async def health_check(self) -> bool:
        """搜狗微信搜索不需要 Cookie，只要页面可达即可。"""
        async with self._pool.acquire(timeout=5.0) as context:
            page = await context.new_page()
            try:
                resp = await page.goto(self.SEARCH_URL, timeout=8000)
                return resp is not None and resp.status == 200
            except Exception:
                return False
            finally:
                await page.close()
```

---

## 4. 搜狗微信搜索特性

| 特性 | 说明 |
|:---|:---|
| **无需 Cookie** | 搜狗微信搜索是公开入口，不需要微信登录态 |
| **跳转链接** | 搜索结果 URL 是搜狗跳转链接，需要访问后获取实际微信文章 URL |
| **反爬** | IP 频率限制，验证码触发阈值低（建议 ≤8 RPM） |
| **内容独占性** | 微信公众号长文是中文互联网最有价值的深度内容池之一 |
| **时效性** | 搜索结果按时间排序，支持按时间筛选 |

---

## 5. 与知乎/微博的对比

| 维度 | 知乎 | 微博 | 微信公众号 |
|:---|:---|:---|:---|
| **需要 Cookie** | ✅ 需要 | ✅ 需要 | ❌ 不需要 |
| **反爬难度** | ⭐⭐ 中 | ⭐⭐⭐ 高 | ⭐⭐ 中 |
| **内容价值** | 深度问答 | 实时舆情 | 深度长文 |
| **降级策略** | site:zhihu.com | site:weibo.com | ❌ 无法降级（百度不索引）|

> [!WARNING]
> 微信公众号搜索**没有 `site:` 降级方案**。因为百度/Bing 不索引微信公众号内容。搜狗微信搜索是唯一入口。触发限流后只能等待（建议 retry_after=5min）。

---

## 6. 验收标准

| 测试用例 | 说明 |
|:---|:---|
| `test_search_returns_results` | 搜索 "人工智能" → ≥1 条结果 |
| `test_result_has_account_name` | 结果 metadata 包含公众号名称 |
| `test_captcha_detection` | 验证码页面 → RateLimitedError |
| `test_rate_limiting` | 超过 8 RPM → 等待 |
| `test_health_check` | 搜狗微信搜索可达 |
