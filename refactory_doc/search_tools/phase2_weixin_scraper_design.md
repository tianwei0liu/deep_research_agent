# Phase 2: WeixinScraper 详细设计

> **模块**: `search_service/backends/weixin_scraper.py`  
> **依赖**: BrowserPool, AsyncRateLimiter, SearchServiceConfig  
> **MCP 工具**: `weixin_search`  
> **日期**: 2026-05-05

---

## 1. 架构概览

```
┌────────────────────────────────────────────────────────┐
│                SearchMCPServer                          │
│  weixin_search(query, max_results)                      │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────────────────────────┐                       │
│  │  WeixinScraper               │                       │
│  │  ├─ Sogou Search Page        │ ← weixin.sogou.com    │
│  │  ├─ Link Resolver            │ ← 跳转链接 → 原始 URL  │
│  │  ├─ AsyncRateLimiter         │                       │
│  │  └─ BrowserPool              │                       │
│  └──────────────────────────────┘                       │
│                                                         │
│  ⚠ 无 site: fallback — 百度/Bing 不索引微信公众号内容    │
└────────────────────────────────────────────────────────┘
```

### 关键约束

> [!CAUTION]
> **微信公众号搜索没有降级方案**。
> - 搜狗是唯一拥有微信公众号搜索授权的外部引擎
> - 百度/Bing/Google 均不索引微信公众号文章
> - 一旦搜狗反爬触发，唯一的恢复策略是等待 + 更换 IP
> 
> 因此 WeixinScraper 的**稳定性设计**比知乎/微博更为关键。

---

## 2. 核心策略

### 搜狗微信搜索特性

| 特性 | 说明 |
|:---|:---|
| **无需 Cookie** | 搜狗微信搜索是公开入口，不需要微信/搜狗登录态 |
| **跳转链接** | 搜索结果 URL 是搜狗临时链接，含加密参数，绑定 IP+SNUID |
| **反爬** | IP 频率限制，验证码触发阈值低（建议 ≤5 RPM） |
| **SNUID Cookie** | 搜狗会自动设置 SNUID，用于追踪访问者 |
| **链接时效性** | 搜狗跳转链接有时效限制，需要及时解析 |

### 选型: Playwright vs HTTP

| 维度 | Playwright | 纯 HTTP |
|:---|:---|:---|
| **SNUID 获取** | 自动（浏览器处理） | 需手动管理 |
| **验证码处理** | 可检测 DOM | 需解析响应 |
| **跳转链接解析** | 浏览器自动跟随 | 需手动 HEAD 请求 |
| **性能** | 低 | 高 |

**结论**: 选择 **Playwright** 方案。理由:
1. 搜狗的 SNUID 管理和链接跳转由浏览器自然处理
2. 验证码检测需要 DOM 访问
3. 请求频率极低（≤5 RPM），Playwright 性能开销可接受

---

## 3. 类设计

```python
class WeixinScraper:
    """微信公众号文章搜索 — 通过搜狗微信搜索。

    搜索入口: https://weixin.sogou.com/weixin?type=2&query={q}
    - type=1: 搜索公众号（不使用）
    - type=2: 搜索文章（我们使用这个）

    不需要微信/搜狗 Cookie。搜狗会自动设置 SNUID。
    但搜狗反爬极为严格，需要:
    - 严格限速 (≤5 RPM per IP)
    - 人类化行为模拟 (随机延迟)
    - IP 轮换（生产环境）

    Attributes:
        SEARCH_URL: 搜狗微信搜索入口。
        _CAPTCHA_SELECTORS: 验证码检测选择器。
    """

    SEARCH_URL: str = "https://weixin.sogou.com/weixin"

    # --- CSS 选择器 ---
    _ARTICLE_LIST_SELECTOR: str = "ul.news-list"
    _ARTICLE_ITEM_SELECTOR: str = "ul.news-list > li"
    _TITLE_SELECTOR: str = "h3 a"
    _SUMMARY_SELECTOR: str = ".txt-info"
    _ACCOUNT_SELECTOR: str = ".s-p a, a.account"
    _TIME_SELECTOR: str = ".s-p"
    _IMAGE_SELECTOR: str = ".img-box img"

    # --- 验证码检测 ---
    _CAPTCHA_SELECTORS: list[str] = [
        "#seccodeImage",       # 传统图片验证码
        ".antispider",         # 反爬虫页面
        "#verify-container",   # 新版验证码容器
    ]

    # --- 人类化延迟 ---
    _MIN_DELAY_SECONDS: float = 2.0
    _MAX_DELAY_SECONDS: float = 5.0

    def __init__(
        self,
        browser_pool: BrowserPool,
        config: SearchServiceConfig,
    ) -> None:
        self._pool = browser_pool
        self._rpm_limit = config.weixin_rpm
        self._rate_limiter = AsyncRateLimiter(
            max_calls=self._rpm_limit, period=60,
        )
        self._logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "weixin"

    async def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> SearchResponse: ...

    async def health_check(self) -> bool: ...
```

---

## 4. 搜索流程

```
search(query, max_results)
    │
    ├─ 1. rate_limiter.acquire()            # 等待 RPM 配额
    │
    ├─ 2. random_delay(2~5s)                # 人类化延迟
    │
    ├─ 3. browser_pool.acquire()            # 获取浏览器上下文
    │      (无需注入 Cookie — 搜狗自动设置 SNUID)
    │
    ├─ 4. page.goto(search_url)             # 导航到搜索页
    │      └─ 超时? ──► ContentExtractionError
    │
    ├─ 5. _detect_captcha(page)             # 检测验证码
    │      └─ 检测到? ──► RateLimitedError(retry_after=300)
    │
    ├─ 6. _extract_results(page)            # 提取搜索结果
    │
    ├─ 7. _resolve_links(results)           # 解析跳转链接 (可选)
    │
    └─ 8. 构建 SearchResponse 返回
```

### 4.1 搜索实现

```python
async def search(
    self,
    query: str,
    max_results: int = 10,
) -> SearchResponse:
    """搜索微信公众号文章。

    Args:
        query: 搜索关键词（中文效果最佳）。
        max_results: 最大返回结果数。

    Returns:
        SearchResponse

    Raises:
        RateLimitedError: 触发搜狗验证码。
        ContentExtractionError: 页面结构异常。
    """
    await self._rate_limiter.acquire()

    # 人类化延迟 — 降低 CAPTCHA 触发概率
    delay = random.uniform(self._MIN_DELAY_SECONDS, self._MAX_DELAY_SECONDS)
    await asyncio.sleep(delay)

    async with self._pool.acquire() as context:
        page = await context.new_page()
        try:
            search_url = f"{self.SEARCH_URL}?type=2&query={query}"
            await page.goto(
                search_url,
                wait_until="domcontentloaded",
                timeout=15000,
            )

            # 检测验证码
            if await self._detect_captcha(page):
                raise RateLimitedError(
                    "weixin_sogou",
                    retry_after_seconds=300,  # 5 分钟后重试
                )

            # 检测反爬重定向
            if "/antispider/" in page.url:
                raise RateLimitedError(
                    "weixin_sogou",
                    retry_after_seconds=300,
                )

            results = await self._extract_results(page, max_results)

            return SearchResponse(
                query=query,
                results=results,
                result_count=len(results),
                search_time_ms=0,
                engines_used=["weixin"],
            )
        finally:
            await page.close()
```

### 4.2 验证码检测

```python
async def _detect_captcha(self, page: Page) -> bool:
    """检测搜狗验证码页面。

    搜狗反爬触发后的表现:
    1. 重定向到 /antispider/ 页面
    2. 页面出现 #seccodeImage (图片验证码)
    3. 页面出现 .antispider 容器

    Returns:
        True 如果检测到验证码。
    """
    for selector in self._CAPTCHA_SELECTORS:
        element = await page.query_selector(selector)
        if element is not None:
            self._logger.warning(
                "weixin_captcha_detected",
                extra={"selector": selector, "url": page.url},
            )
            return True
    return False
```

### 4.3 搜索结果提取

```python
async def _extract_results(
    self,
    page: Page,
    max_results: int,
) -> list[SearchResultItem]:
    """提取搜狗微信搜索结果。

    搜索结果结构:
    <ul class="news-list">
      <li>
        <h3><a href="跳转链接">标题</a></h3>
        <p class="txt-info">摘要...</p>
        <div class="s-p">
          <a class="account">公众号名称</a>
          <span>2026-05-01</span>
        </div>
      </li>
    </ul>
    """
    items: list[SearchResultItem] = []
    cards = await page.query_selector_all(self._ARTICLE_ITEM_SELECTOR)

    for card in cards[:max_results]:
        try:
            item = await self._parse_article_card(card)
            if item is not None:
                items.append(item)
        except Exception as exc:
            self._logger.debug(
                "weixin_card_parse_error", extra={"error": str(exc)},
            )

    return items
```

### 4.4 文章卡片解析

```python
async def _parse_article_card(
    self, card: ElementHandle,
) -> Optional[SearchResultItem]:
    """解析单个文章卡片。

    Returns:
        SearchResultItem or None。
    """
    # 标题 + 链接
    title_el = await card.query_selector(self._TITLE_SELECTOR)
    if not title_el:
        return None

    title = await title_el.inner_text()
    sogou_url = await title_el.get_attribute("href") or ""

    # URL 规范化 — 搜狗链接可能是相对路径
    if sogou_url.startswith("/"):
        url = f"https://weixin.sogou.com{sogou_url}"
    elif sogou_url.startswith("http"):
        url = sogou_url
    else:
        url = ""

    # 摘要
    summary_el = await card.query_selector(self._SUMMARY_SELECTOR)
    content = (await summary_el.inner_text()) if summary_el else ""

    # 公众号名称
    account_el = await card.query_selector(self._ACCOUNT_SELECTOR)
    account_name = (await account_el.inner_text()) if account_el else ""

    # 发布时间
    pub_time = await self._extract_publish_time(card)

    return SearchResultItem(
        title=title.strip(),
        url=url,
        content=content.strip()[:500],
        source_engine=SearchEngine.WEIXIN,
        published_date=pub_time,
        metadata={
            "account_name": account_name.strip(),
            "sogou_url": sogou_url,  # 保留原始搜狗链接供调试
        },
    )
```

### 4.5 发布时间提取

```python
async def _extract_publish_time(self, card: ElementHandle) -> str:
    """提取文章发布时间。

    搜狗显示格式多样:
    - "1小时前"
    - "昨天"
    - "2026-05-01"
    - "公众号名称 2026-05-01" (需要分割)
    """
    time_el = await card.query_selector(self._TIME_SELECTOR)
    if not time_el:
        return ""

    time_text = await time_el.inner_text()
    # 格式: "公众号名称  2026-05-01" → 取最后部分
    parts = time_text.strip().split()
    if parts:
        return parts[-1]
    return ""
```

---

## 5. 跳转链接解析（可选增强）

搜狗返回的搜索结果链接是临时跳转链接，格式:
```
https://weixin.sogou.com/link?url=dn9a_-gY295K0Rci_xozVXfdo...&k=...&h=...
```

访问这个链接会 302 重定向到真正的微信文章 URL:
```
https://mp.weixin.qq.com/s/xxxxx
```

### 5.1 是否需要解析？

| 场景 | 是否需要解析 |
|:---|:---|
| Agent 只需要标题+摘要 | ❌ 不需要 — 搜狗摘要已足够 |
| Agent 需要 `scrape_url` 深度提取 | ✅ 需要 — 搜狗跳转链接可能失效 |
| 用户点击引用来源 | ⚠️ 最好有 — 跳转链接有时效限制 |

### 5.2 解析实现

```python
async def _resolve_sogou_link(
    self,
    sogou_url: str,
    page: Page,
) -> str:
    """解析搜狗跳转链接获取原始微信文章 URL。

    通过 Playwright 在同一上下文中访问跳转链接，
    浏览器自动跟随重定向，最终 URL 就是微信文章 URL。

    Args:
        sogou_url: 搜狗跳转链接。
        page: 当前页面（复用同一 context 的 cookie）。

    Returns:
        原始微信文章 URL，解析失败时返回搜狗链接本身。
    """
    try:
        new_page = await page.context.new_page()
        try:
            response = await new_page.goto(
                sogou_url,
                wait_until="commit",  # 只等待重定向完成
                timeout=8000,
            )
            return new_page.url  # 重定向后的 URL
        finally:
            await new_page.close()
    except Exception as exc:
        self._logger.debug(
            "sogou_link_resolve_failed",
            extra={"url": sogou_url, "error": str(exc)},
        )
        return sogou_url  # fallback: 返回搜狗链接
```

> [!NOTE]
> **建议**: Phase 2 第一版不实现链接解析，搜狗链接直接返回。
> 当 Agent 需要 `scrape_url` 深度提取微信文章时，在 Phase 3 再加。

---

## 6. MCP Server 集成

```python
# server.py 变更
@self._mcp.tool()
async def weixin_search(query: str, max_results: int = 10) -> dict:
    """Search WeChat Official Account articles via Sogou.

    Best for: industry analysis, policy interpretation, long-form content.
    This is the ONLY way to search WeChat content from outside the app.

    ⚠️ No fallback available — WeChat content is not indexed by
    any other search engine. If this tool fails, the content is
    not accessible through other means.
    """
    scraper = self._get_weixin_scraper()
    response = await scraper.search(query, max_results)
    return response.model_dump()

    # ⚠️ 注意: 微信搜索没有 site: fallback
    # 如果 scraper 抛出 RateLimitedError，直接上抛给 Agent
    # Agent 需要知道"微信搜索当前不可用"而不是拿到空结果
```

> [!IMPORTANT]
> 与知乎/微博不同，微信搜索**不能在 MCP Server 层 catch 异常并降级到 site: 搜索**。
> 应该让 `RateLimitedError` 通过 MCP 错误通道传递给 Agent，
> Agent 会在 prompt 中看到错误信息并调整策略。

---

## 7. Rate Limiting — 双重保护

### 7.1 应用层限流

```python
# WeixinScraper 内部
self._rate_limiter = AsyncRateLimiter(max_calls=5, period=60)
```

> [!WARNING]
> 搜狗的反爬触发阈值远低于知乎/微博。建议:
> - 开发/测试环境: 5 RPM
> - 生产环境 (有代理): 8 RPM
> - 无代理单 IP: 3-5 RPM

### 7.2 人类化行为模拟

```python
# 每次搜索前随机延迟 2-5 秒
delay = random.uniform(2.0, 5.0)
await asyncio.sleep(delay)
```

### 7.3 与 CAPTCHA 缓解策略的协同

参考 `captcha_mitigation_strategy.md`:
- L1 (降低 suspended_times) 对搜狗引擎同样有效
- L2 (代理轮转) 对微信搜索**最关键** — 因为无 fallback
- L1.5 (应用层节流) 在 WeixinScraper 中已内置

---

## 8. 健康检查

```python
async def health_check(self) -> bool:
    """搜狗微信搜索可达性检查。

    不需要 Cookie。检测搜索页是否可正常加载。
    """
    try:
        async with self._pool.acquire(timeout=5.0) as context:
            page = await context.new_page()
            try:
                response = await page.goto(
                    self.SEARCH_URL, timeout=8000,
                )
                if response is None:
                    return False
                # 检查是否被反爬拦截
                if await self._detect_captcha(page):
                    return False
                return response.status == 200
            finally:
                await page.close()
    except Exception:
        return False
```

---

## 9. 验收标准

| 测试用例 | 说明 | Mock 策略 |
|:---|:---|:---|
| `test_search_returns_results` | 正常搜索 → ≥1 条结果 | Mock page DOM |
| `test_result_has_account_name` | metadata 包含 account_name | Mock card elements |
| `test_captcha_detection_by_image` | #seccodeImage 存在 → RateLimitedError | Mock page selector |
| `test_captcha_detection_by_antispider` | .antispider 存在 → RateLimitedError | Mock page selector |
| `test_antispider_url_redirect` | URL 含 /antispider/ → RateLimitedError | Mock page.url |
| `test_rate_limiting` | 超过 5 RPM → acquire() 阻塞 | AsyncRateLimiter |
| `test_human_delay` | 搜索前有 2-5s 延迟 | Mock asyncio.sleep |
| `test_publish_time_extraction` | "公众号名 2026-05-01" → "2026-05-01" | 直接测试 |
| `test_health_check_reachable` | 搜索页可达 → True | Mock page response |
| `test_health_check_captcha` | 被反爬拦截 → False | Mock captcha selector |
| `test_no_fallback_on_error` | RateLimitedError 不被 catch → 上抛 | 集成测试 |

---

## 10. 行业价值分析与主流实现方式

### 10.1 功能实际价值评估

| 维度 | 评估 | 说明 |
|:---|:---|:---|
| **数据独特性** | ⭐⭐⭐⭐⭐ | 微信公众号是中文最大"围墙花园"，大量行业报告和深度分析**仅在公众号发布** |
| **Deep Research 贡献** | ⭐⭐⭐⭐⭐ | 公众号在"行业分析"、"政策解读"类查询中不可替代 |
| **ROI** | ⭐⭐ | 搜狗反爬极严、无降级方案、维护风险高 |
| **不可替代性** | ⭐⭐⭐⭐⭐ | 百度/Bing/Google 均不索引公众号文章 |

**核心价值与风险并存**: 公众号内容不可替代，但搜狗反爬和缺乏降级方案使其成为系统最脆弱环节。

### 10.2 业界主流实现方式

> [!IMPORTANT]
> **行业关键变化**: 搜狗微信搜索在 2025-2026 年可靠性下降。腾讯将搜索整合到微信内部"搜一搜"，搜狗仅提供有限外部接口。

| 方案 | 采用度 | 可行性 | 说明 |
|:---|:---|:---|:---|
| **搜狗微信搜索** | ⭐⭐⭐ 退化中 | 中 | 唯一免费外部入口，反爬严格 |
| **第三方数据 API** | ⭐⭐⭐⭐ 上升 | 高 | TikHub 等付费 API，约 ¥1-5/千次 |
| **微信开放平台 API** | ⭐⭐ | 低 | 需企业资质，仅管理自有账号 |
| **微信搜一搜** | ⭐⭐⭐⭐⭐ | 极低 | 仅 App 内可用，无外部接口 |

行业最佳实践：不依赖单一搜狗入口，组合搜狗 + 付费 API 的混合策略。生产环境趋势是使用付费数据服务外包反爬复杂性。

#### 竞品对比

| 产品 | 公众号搜索 | 方式 |
|:---|:---|:---|
| **Perplexity/ChatGPT/Gemini** | ❌ 不支持 | 不索引微信 |
| **秘塔 AI** | ✅ 支持 | 可能通过付费 API 或搜狗 |
| **我们** | ✅ 计划支持 | 搜狗 (Phase 2)，付费 API (Phase 3) |

> [!TIP]
> **战略建议**: Phase 2 用搜狗快速上线，Phase 3 优先评估第三方付费 API 作为备选/替代。

### 10.3 与行业最佳实践的差距

| 项 | 当前设计 | 行业实践 | 差距 | 建议 |
|:---|:---|:---|:---|:---|
| 数据源 | 仅搜狗 | 多源聚合 | 高 | Phase 3 评估第三方 API |
| 反爬韧性 | 限速+行为模拟 | 代理轮换+IP池 | 高 | 最需要代理池的功能 |
| 降级方案 | 无 | 付费 API 降级 | 高 | 集成付费 API 作为 fallback |

---

## 11. 待确认事项

> [!NOTE]

1. **搜狗 DOM 选择器验证**: 搜狗微信搜索页的 DOM 结构可能已更新。需要手动访问 `weixin.sogou.com/weixin?type=2&query=人工智能` 确认 `ul.news-list > li` 选择器是否仍有效。

2. **跳转链接解析**: Phase 2 是否实现 `_resolve_sogou_link()`？我的建议是 Phase 2 先返回搜狗链接（功能可用），Phase 3 再实现解析。

3. **RPM 限制值**: 当前设计为 5 RPM (config 字段 `weixin_rpm` 默认值 8)。是否需要修改 `SearchServiceConfig` 的默认值从 8 降到 5？

4. **生产环境代理**: 由于微信搜索无 fallback，生产环境**必须**配置 L2 代理轮转。这是否应该作为 Phase 2 的前置依赖？

5. **搜狗账号登录**: 搜狗微信搜索目前不需要登录，但如果搜狗升级反爬要求登录，是否需要预留 Cookie 注入接口？

6. **验证码解决方案**: 如果 CAPTCHA 频繁触发，是否需要集成第三方验证码识别服务（如 2Captcha/超级鹰）？成本约 ¥1-5/千次。
