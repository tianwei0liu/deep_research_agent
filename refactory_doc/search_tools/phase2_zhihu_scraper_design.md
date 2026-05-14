# Phase 2: ZhihuScraper 详细设计

> **模块**: `search_service/backends/zhihu_scraper.py`  
> **依赖**: BrowserPool, AsyncRateLimiter, SearchServiceConfig  
> **MCP 工具**: `zhihu_search`  
> **日期**: 2026-05-05

---

## 1. 架构概览

```
┌─────────────────────────────────────────────────┐
│              SearchMCPServer                     │
│  zhihu_search(query, max_results)                │
│       │                                          │
│       ▼                                          │
│  ┌──────────────────────────┐                    │
│  │  ZhihuScraper            │                    │
│  │  ├─ CookieManager        │                    │
│  │  ├─ AsyncRateLimiter     │                    │
│  │  └─ BrowserPool.acquire()│                    │
│  └──────────┬───────────────┘                    │
│             │ CookieExpiredError                  │
│             ▼                                    │
│  ┌──────────────────────────┐                    │
│  │  Fallback: site:zhihu.com│ ← SearchRouter     │
│  └──────────────────────────┘                    │
└─────────────────────────────────────────────────┘
```

### 核心策略选择

**Playwright 渲染 vs API 逆向**

| 维度 | Playwright 渲染 | API 逆向 (x-zse-96) |
|:---|:---|:---|
| **稳定性** | 高 — 浏览器自然处理签名 | 低 — 知乎频繁更新 WASM 混淆 |
| **维护成本** | 低 — CSS 选择器变化慢 | 高 — 每次更新需重新逆向 |
| **性能** | 中 — 需启动浏览器上下文 | 高 — 纯 HTTP 请求 |
| **资源消耗** | 高 — Chromium 进程 | 低 — 无浏览器 |

**结论**: 选择 **Playwright 渲染**方案。理由：

1. 我们已有 BrowserPool 基础设施，复用成本低
2. x-zse-96 的逆向需要持续投入维护，不适合小团队
3. 搜索频率低（≤12 RPM），Playwright 的性能开销可接受
4. stealth.js 已注入，可自然绕过指纹检测

---

## 2. 类设计

### 2.1 ZhihuScraper

```python
class ZhihuScraper:
    """知乎搜索爬虫 — Playwright 渲染 + Cookie 持久化。

    搜索入口: https://www.zhihu.com/search?type=content&q={query}
    需要登录 Cookie (z_c0) 才能访问完整搜索结果。

    降级策略: Cookie 过期或加载失败时，自动回退到 SearXNG
    的 site:zhihu.com 搜索。

    Attributes:
        SEARCH_URL: 知乎搜索页 URL 模板。
        _RESULT_CARD_SELECTOR: 搜索结果卡片 CSS 选择器。
        _LOGIN_WALL_SELECTORS: 登录弹窗检测选择器列表。
    """

    SEARCH_URL: str = "https://www.zhihu.com/search"

    # --- CSS 选择器 (可维护集中管理) ---
    _RESULT_CARD_SELECTOR: str = ".SearchResult-Card"
    _TITLE_SELECTOR: str = "h2.ContentItem-title a, h2 a[data-za-detail-view-element_name='Title']"
    _CONTENT_SELECTOR: str = ".RichText.SearchResult-Excerpt, .RichContent-inner"
    _AUTHOR_SELECTOR: str = ".AuthorInfo-name a, .UserLink-link"
    _UPVOTE_SELECTOR: str = ".VoteButton--up"
    _LOGIN_WALL_SELECTORS: list[str] = [
        ".Modal-backdrop",
        "[data-za-detail-view-element_name='SignInButton']",
        ".signFlowModal",
    ]

    def __init__(
        self,
        browser_pool: BrowserPool,
        config: SearchServiceConfig,
        cookie_manager: CookieManager,  # 注入依赖
    ) -> None: ...

    @property
    def name(self) -> str:
        return "zhihu"

    async def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> SearchResponse: ...

    async def health_check(self) -> bool: ...
```

### 2.2 CookieManager（提取为独立类）

> [!IMPORTANT]
> Cookie 管理逻辑在知乎、微博中完全相同（加载 JSON → 注入 → 过期检测）。
> 应提取为 `search_service/cookie_manager.py` 独立模块，三个 scraper 共用。

```python
class CookieManager:
    """平台 Cookie 持久化管理器。

    职责:
    - 从磁盘加载 Playwright 兼容的 Cookie JSON
    - Cookie 过期前的 TTL 预警
    - Cookie 有效性校验（可选: 对知乎检测 z_c0 的 expiry 字段）

    Args:
        platform: 平台标识 ("zhihu", "weibo")。
        storage_dir: Cookie JSON 文件存储目录。
        ttl_warning_hours: 过期前 N 小时触发警告日志。
    """

    def __init__(
        self,
        platform: str,
        storage_dir: Path,
        ttl_warning_hours: int = 24,
    ) -> None: ...

    async def load(self) -> list[dict]:
        """加载 Cookie，过期或不存在时抛出 CookieExpiredError。"""
        ...

    async def save(self, cookies: list[dict]) -> None:
        """保存 Cookie 到磁盘（用于手动刷新后的持久化）。"""
        ...

    def is_expired(self, cookies: list[dict]) -> bool:
        """检查关键 Cookie 是否已过期。

        知乎: 检查 z_c0 的 expires 字段。
        微博: 检查 SUB 的 expires 字段。
        """
        ...

    def check_ttl_warning(self, cookies: list[dict]) -> None:
        """距过期 < ttl_warning_hours 时输出 warning 日志。"""
        ...
```

---

## 3. 搜索流程

```
search(query, max_results)
    │
    ├─ 1. rate_limiter.acquire()        # 等待 RPM 配额
    │
    ├─ 2. cookie_manager.load()         # 加载 Cookie
    │      └─ CookieExpiredError? ──► fallback_to_site_search()
    │
    ├─ 3. browser_pool.acquire(cookies) # 获取浏览器上下文
    │
    ├─ 4. page.goto(search_url)         # 导航到搜索页
    │      └─ 超时? ──► ContentExtractionError
    │
    ├─ 5. _detect_login_wall(page)      # 检测登录弹窗
    │      └─ 检测到? ──► CookieExpiredError ──► fallback
    │
    ├─ 6. _wait_for_results(page)       # 等待搜索结果加载
    │      └─ 超时? ──► 返回空结果
    │
    ├─ 7. _extract_results(page)        # 提取搜索结果
    │
    └─ 8. 构建 SearchResponse 返回
```

### 3.1 登录弹窗检测

知乎的登录弹窗有多种形态，需要多选择器联合检测:

```python
async def _detect_login_wall(self, page: Page) -> bool:
    """检测知乎登录弹窗/墙。

    知乎在以下场景强制登录:
    1. Cookie 过期或无效 → 全屏 Modal
    2. 未登录用户浏览 N 个搜索结果后 → 底部遮罩
    3. URL 被重定向到 /signin

    Returns:
        True 如果检测到登录强制要求。
    """
    # 策略1: 检查 CSS 选择器
    for selector in self._LOGIN_WALL_SELECTORS:
        element = await page.query_selector(selector)
        if element is not None:
            return True

    # 策略2: 检查 URL 重定向
    if "/signin" in page.url or "passport" in page.url:
        return True

    return False
```

### 3.2 搜索结果提取

```python
async def _extract_results(
    self, page: Page, max_results: int,
) -> list[SearchResultItem]:
    """提取知乎搜索结果列表。

    容错策略:
    - 单条解析失败不影响整体，记录 debug 日志
    - 知乎搜索结果可能包含广告卡片，通过缺少 href 自动过滤
    """
    items: list[SearchResultItem] = []
    cards = await page.query_selector_all(self._RESULT_CARD_SELECTOR)

    for card in cards[:max_results]:
        try:
            item = await self._parse_card(card)
            if item is not None:  # 过滤广告卡片
                items.append(item)
        except Exception as exc:
            self._logger.debug(
                "zhihu_card_parse_error", extra={"error": str(exc)},
            )

    return items
```

### 3.3 元数据提取

```python
async def _parse_card(self, card: ElementHandle) -> Optional[SearchResultItem]:
    """解析单个搜索结果卡片。

    Returns:
        SearchResultItem or None (广告/无效卡片返回 None)。
    """
    # 标题 + 链接
    title_el = await card.query_selector(self._TITLE_SELECTOR)
    if not title_el:
        return None
    title = await title_el.inner_text()
    href = await title_el.get_attribute("href") or ""

    # 知乎内部链接转换
    if href.startswith("/"):
        url = f"https://www.zhihu.com{href}"
    elif href.startswith("http"):
        url = href
    else:
        return None  # 无效链接 → 可能是广告

    # 内容摘要
    content_el = await card.query_selector(self._CONTENT_SELECTOR)
    content = (await content_el.inner_text()) if content_el else ""

    # 元数据: 赞同数、作者
    metadata: dict[str, Any] = {}
    author_el = await card.query_selector(self._AUTHOR_SELECTOR)
    if author_el:
        metadata["author"] = await author_el.inner_text()
    upvote_el = await card.query_selector(self._UPVOTE_SELECTOR)
    if upvote_el:
        metadata["upvotes"] = await upvote_el.inner_text()

    return SearchResultItem(
        title=title.strip(),
        url=url,
        content=content.strip()[:500],
        source_engine=SearchEngine.ZHIHU,
        metadata=metadata,
    )
```

---

## 4. MCP Server 集成

当前 `server.py` 中的 `zhihu_search` 使用 `site:zhihu.com` fallback。
Phase 2 需替换为实际 scraper 调用 + fallback 链。

```python
# server.py 变更
@self._mcp.tool()
async def zhihu_search(query: str, max_results: int = 10) -> dict:
    """Search Zhihu for in-depth Q&A and expert opinions.

    Uses Playwright-based scraper when valid cookies are available.
    Falls back to site:zhihu.com search via SearXNG when cookies
    expire or scraper fails.
    """
    # 尝试 Playwright scraper
    try:
        zhihu = self._get_zhihu_scraper()
        response = await zhihu.search(query, max_results)
        return response.model_dump()
    except CookieExpiredError:
        self._logger.warning("zhihu_cookie_expired, fallback to site: search")
    except (RateLimitedError, ContentExtractionError) as exc:
        self._logger.warning("zhihu_scraper_failed: %s, fallback", exc)

    # Fallback: site: search
    assert self._router is not None
    response = await self._router.search(
        f"site:zhihu.com {query}", max_results=max_results,
    )
    return response.model_dump()
```

---

## 5. Cookie 管理流程

### 5.1 初始获取（手动）

```bash
# 方案 A: 浏览器扩展 (推荐 EditThisCookie)
# 1. Chrome 登录 zhihu.com
# 2. EditThisCookie → 导出 JSON → 保存到 data/cookies/zhihu/cookies.json

# 方案 B: Playwright 脚本 (半自动)
python -m search_service.scripts.cookie_helper --platform zhihu
# 打开有头浏览器 → 手动登录 → 脚本自动保存 Cookie
```

### 5.2 Cookie 格式

```json
[
    {
        "name": "z_c0",
        "value": "2|1:0|10:...",
        "domain": ".zhihu.com",
        "path": "/",
        "httpOnly": true,
        "secure": true,
        "expires": 1748000000
    },
    {
        "name": "d_c0",
        "value": "\"AKBx...\"",
        "domain": ".zhihu.com",
        "path": "/",
        "httpOnly": false,
        "secure": false
    }
]
```

> [!IMPORTANT]
> **关键 Cookie**: `z_c0` (登录凭证，有效期通常 30 天) 和 `d_c0` (设备 ID，用于频率追踪)。
> 缺少 `z_c0` 时搜索结果被大幅截断。

### 5.3 过期检测策略

| 检测方式 | 时机 | 动作 |
|:---|:---|:---|
| `expires` 字段检查 | `cookie_manager.load()` 时 | 过期 → CookieExpiredError |
| TTL 预警 | `cookie_manager.load()` 时 | <24h → `logger.warning` |
| 页面登录弹窗检测 | 搜索页加载后 | 弹窗 → CookieExpiredError |
| URL 重定向检测 | `page.goto()` 后 | /signin → CookieExpiredError |

---

## 6. Rate Limiting

| 配置 | 值 | 理由 |
|:---|:---|:---|
| `zhihu_rpm` | 12 | 知乎反爬触发阈值约 15-20 RPM |
| 实现 | `AsyncRateLimiter(max_calls=12, period=60)` | 复用现有 token-bucket |
| 位置 | `ZhihuScraper.search()` 入口 | 在 Cookie 加载之前限流 |

---

## 7. 错误处理

| 异常 | 触发条件 | 处理 |
|:---|:---|:---|
| `CookieExpiredError` | Cookie 不存在/过期/登录弹窗 | fallback to site: search |
| `RateLimitedError` | 触发知乎验证码 | fallback to site: search |
| `ContentExtractionError` | 页面结构异常/超时 | fallback to site: search |
| `BrowserPoolExhaustedError` | 浏览器池满 | 向上抛出，MCP 返回错误 |

---

## 8. 验收标准

| 测试用例 | 说明 | Mock 策略 |
|:---|:---|:---|
| `test_search_with_valid_cookie` | 有效 Cookie → 返回 SearchResponse | Mock page DOM |
| `test_cookie_expired_detection_by_expiry` | Cookie expires 字段过期 → CookieExpiredError | Mock cookie file |
| `test_cookie_expired_detection_by_login_wall` | 页面出现登录弹窗 → CookieExpiredError | Mock page selector |
| `test_no_cookie_file` | Cookie 文件不存在 → CookieExpiredError | 不创建文件 |
| `test_rate_limiting` | 超过 12 RPM → acquire() 阻塞 | AsyncRateLimiter |
| `test_result_parsing_complete` | 标题/URL/内容/作者/赞同数正确提取 | Mock card elements |
| `test_ad_card_filtered` | 广告卡片（无 href）被过滤 | Mock card without href |
| `test_url_normalization` | 内部链接 `/question/123` → 完整 URL | 直接测试 |
| `test_health_check_with_cookies` | Cookie 存在 → True | Mock file |
| `test_health_check_without_cookies` | Cookie 不存在 → False | 不创建文件 |

---

## 9. 行业价值分析与主流实现方式

### 9.1 功能实际价值评估

| 维度 | 评估 | 说明 |
|:---|:---|:---|
| **数据独特性** | ⭐⭐⭐⭐⭐ | 知乎是中文互联网最大的高质量问答社区，专业领域深度内容（技术、医学、金融）在其他平台难以获取 |
| **对 Deep Research 的贡献** | ⭐⭐⭐⭐ | 知乎内容在"技术对比"、"经验总结"、"行业分析"类查询中提供了搜索引擎难以索引的深度观点 |
| **ROI（投入产出比）** | ⭐⭐⭐ | 需要 Cookie 管理 + Playwright 渲染，维护成本中等，但数据价值高 |
| **竞品覆盖** | 中等 | Perplexity/ChatGPT Search 不索引知乎，但用户可通过通用搜索间接获取部分内容 |

**核心价值**: 知乎搜索是建立**中文 Deep Research 差异化竞争力**的关键功能。业界竞品（Perplexity、Gemini Deep Research）主要依赖英文互联网，对中文垂直平台的覆盖几乎为零。

### 9.2 业界主流实现方式

#### Playwright 渲染 vs API 逆向 — 行业共识

2025-2026 年行业实践已形成明确共识：

| 方案 | 行业采用度 | 适用场景 |
|:---|:---|:---|
| **Playwright/Puppeteer 渲染** | ⭐⭐⭐⭐⭐ 主流 | 中低频率、需要登录态的平台抓取 |
| **API 逆向（x-zse-96）** | ⭐⭐ 少数 | 高频、大规模数据采集（如学术研究/商业情报） |
| **付费 API 服务** | ⭐⭐⭐ 上升趋势 | 企业级生产环境，如 Apify/ScraperAPI 托管方案 |

> [!TIP]
> **行业验证**: 我们选择 Playwright 渲染方案与行业主流一致。2026 年的反爬系统已进化到分析 TLS 指纹、浏览器环境信号和行为模式的多层检测，纯 API 逆向的维护成本呈指数增长（知乎 WASM 混淆的 x-zse-96 是典型案例）。

#### 反爬对抗 — 行业标准实践

业界在对抗知乎级别反爬时的标准做法：

1. **TLS 指纹管理**: 标准 HTTP 库（如 requests）的 TLS 指纹会被立即识别。生产环境使用 `curl_cffi` 或 `tls-client` 模拟真实浏览器 TLS 握手（我们用 Playwright 天然规避了此问题）
2. **指纹一致性**: 确保 User-Agent、viewport、timezone、locale 之间的一致性。不一致的配置（如 Windows UA + Linux timezone）会被标记
3. **住宅/移动代理**: 数据中心 IP 容易被识别，行业标准是使用住宅或移动代理（我们当前单 IP 方案是风险点）
4. **行为模拟**: 实现随机化、非线性延迟和自然滚动模式

#### 竞品对比 — AI 搜索产品如何处理垂直平台

| 产品 | 知乎数据获取 | 方式 |
|:---|:---|:---|
| **Perplexity AI** | ❌ 不支持 | 仅依赖通用搜索引擎索引 |
| **ChatGPT Search** | ❌ 不支持 | 使用 Bing 索引，知乎内容覆盖有限 |
| **Gemini Deep Research** | ❌ 不支持 | 使用 Google 索引 |
| **秘塔 AI 搜索** | ✅ 部分支持 | 疑似通过 API/爬虫获取，具体方案未知 |
| **我们** | ✅ 计划支持 | Playwright 渲染 + Cookie 持久化 |

> [!IMPORTANT]
> **差异化机会**: 主流 AI 搜索产品均不直接索引知乎内容。实现知乎搜索将成为面向中文用户的**核心竞争力**。

### 9.3 设计方案评审 — 与行业最佳实践的差距

| 项 | 当前设计 | 行业最佳实践 | 差距 | 建议 |
|:---|:---|:---|:---|:---|
| 反指纹 | stealth.js 注入 | 深度指纹管理（WebGL/Canvas/Font） | 中 | Phase 2 先用 stealth.js，Phase 3 评估 anti-detect 方案 |
| IP 管理 | 单 IP | 住宅代理轮换 | 高 | 生产环境必须配置代理池 |
| 行为模拟 | 无 | 随机延迟 + 鼠标移动 + 自然滚动 | 中 | Phase 2 至少加随机延迟 |
| Cookie 存储 | 明文 JSON | 加密存储 + 集中管理（Redis） | 中 | Phase 3 迁移到加密存储 |
| 降级策略 | `site:zhihu.com` via SearXNG | 多级降级 + 付费 API 兜底 | 低 | 当前方案已合理 |

---

## 10. 待确认事项

> [!NOTE]
> 以下事项需要与你确认后再最终实施:

1. **CSS 选择器验证**: 上述选择器基于知乎历史稳定结构，但知乎可能已更新 DOM。需要在实现前用真实浏览器访问 `zhihu.com/search?type=content&q=AI` 验证选择器是否仍然有效。

2. **Cookie 获取工具**: 是否需要实现一个 `cookie_helper.py` 半自动脚本，还是手动导出就够用？

3. **Cookie Pool**: 当前设计是单 Cookie 文件。如果未来需要多账号轮换（规避单账号频率限制），需要扩展 CookieManager 支持 Cookie 池。是否现在就预留这个扩展点？

4. **x-zse-96 备选方案**: 如果 Playwright 渲染方案在某些场景下不稳定（如知乎升级反爬检测 stealth.js），是否需要预留 API 逆向的备选路径？

5. **搜索结果滚动加载**: 知乎搜索结果可能需要滚动才能加载更多。当前设计只取第一页结果，是否需要支持 `page.scroll()` + 加载更多？
