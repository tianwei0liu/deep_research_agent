# Phase 2: WeiboScraper 详细设计

> **模块**: `search_service/backends/weibo_scraper.py`  
> **依赖**: BrowserPool / httpx, AsyncRateLimiter, SearchServiceConfig  
> **MCP 工具**: `weibo_search`  
> **日期**: 2026-05-05

---

## 1. 架构概览

```
┌────────────────────────────────────────────────────────┐
│                 SearchMCPServer                         │
│  weibo_search(query, max_results, time_scope)           │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────────────────────────┐                       │
│  │  WeiboScraper                │                       │
│  │  ├─ Strategy A: Mobile API   │ ← 首选 (高效/稳定)    │
│  │  ├─ Strategy B: Playwright   │ ← 备选 (API 失败时)   │
│  │  ├─ CookieManager           │                       │
│  │  └─ AsyncRateLimiter        │                       │
│  └──────────┬───────────────────┘                       │
│             │ CookieExpiredError / API 失败              │
│             ▼                                           │
│  ┌──────────────────────────────┐                       │
│  │  Fallback: site:weibo.com    │ ← SearchRouter        │
│  └──────────────────────────────┘                       │
└────────────────────────────────────────────────────────┘
```

### 核心策略选择

**Mobile API vs Desktop Playwright**

| 维度 | Mobile API (m.weibo.cn) | Desktop Playwright (s.weibo.com) |
|:---|:---|:---|
| **数据格式** | JSON (结构化) | HTML (需 DOM 解析) |
| **反爬难度** | 中 — IP+Cookie | 高 — JS 混淆 + 验证码 |
| **性能** | 高 — 纯 HTTP | 低 — 需浏览器渲染 |
| **资源消耗** | 低 | 高 |
| **结果质量** | 等价 | 等价 |

**结论**: 采用**双策略**架构:

1. **首选**: Mobile API (`m.weibo.cn/api/container/getIndex`) — 高效稳定
2. **备选**: Playwright 渲染 (`s.weibo.com`) — Mobile API 失败时降级
3. **兜底**: `site:weibo.com` via SearXNG — 全部失败时

---

## 2. 类设计

### 2.1 WeiboScraper

```python
class WeiboScraper:
    """微博搜索爬虫 — Mobile API 优先 + Playwright 备选。

    微博特点:
    - 内容时效性极强（分钟级更新）
    - 搜索结果包含转发数/评论数/点赞数
    - 反爬较强（需 Cookie + 频率控制）

    搜索策略:
    1. 首选 Mobile API (m.weibo.cn) — 高效、结构化 JSON
    2. 备选 Playwright (s.weibo.com) — API 失败时降级
    3. 兜底 site:weibo.com via SearXNG

    Attributes:
        MOBILE_API_URL: 移动端搜索 API。
        DESKTOP_SEARCH_URL: 桌面端搜索页面。
    """

    MOBILE_API_URL: str = "https://m.weibo.cn/api/container/getIndex"
    DESKTOP_SEARCH_URL: str = "https://s.weibo.com/weibo"

    # Mobile API 容器 ID 格式
    _CONTAINER_ID_TEMPLATE: str = "100103type=1&q={query}"

    # Desktop CSS 选择器
    _CARD_SELECTOR: str = ".card-wrap[mid]"
    _CONTENT_SELECTOR: str = ".content .txt"
    _AUTHOR_SELECTOR: str = "a.name"
    _TIME_SELECTOR: str = ".from a"
    _ACTION_SELECTOR: str = ".card-act"

    def __init__(
        self,
        browser_pool: BrowserPool,
        config: SearchServiceConfig,
        cookie_manager: CookieManager,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None: ...

    @property
    def name(self) -> str:
        return "weibo"

    async def search(
        self,
        query: str,
        max_results: int = 10,
        time_scope: str = "",
    ) -> SearchResponse: ...

    async def health_check(self) -> bool: ...
```

---

## 3. Mobile API 搜索流程

### 3.1 API 请求

```python
async def _search_mobile_api(
    self,
    query: str,
    max_results: int,
    time_scope: str,
) -> SearchResponse:
    """通过 Mobile API 搜索微博。

    API: GET https://m.weibo.cn/api/container/getIndex
    Params:
        containerid: "100103type=1&q={query}"
        page_type: "searchall"
        page: 1

    需要 Cookie: SUB, SUBP (从 Cookie 文件加载)
    需要 Header: 合法的移动端 User-Agent

    Raises:
        CookieExpiredError: Cookie 无效。
        SearchProviderError: API 返回错误。
    """
    cookies = await self._cookie_manager.load()

    # 构建 Cookie header string
    cookie_header = "; ".join(f"{c['name']}={c['value']}" for c in cookies)

    container_id = self._CONTAINER_ID_TEMPLATE.format(query=query)
    params = {
        "containerid": container_id,
        "page_type": "searchall",
        "page": 1,
    }

    headers = {
        "User-Agent": self._MOBILE_USER_AGENT,
        "Cookie": cookie_header,
        "Referer": "https://m.weibo.cn/",
        "X-Requested-With": "XMLHttpRequest",
    }

    client = await self._ensure_http_client()
    response = await client.get(
        self.MOBILE_API_URL,
        params=params,
        headers=headers,
    )

    if response.status_code == 302 or response.status_code == 403:
        raise CookieExpiredError("weibo")

    data = response.json()

    # 检查 API 响应状态
    if data.get("ok") != 1:
        raise SearchProviderError(
            "weibo", f"Mobile API error: {data.get('msg', 'unknown')}",
        )

    return self._parse_mobile_response(query, data, max_results)
```

### 3.2 Mobile API 响应解析

```python
def _parse_mobile_response(
    self,
    query: str,
    data: dict,
    max_results: int,
) -> SearchResponse:
    """解析 Mobile API JSON 响应。

    JSON 结构:
    {
        "ok": 1,
        "data": {
            "cards": [
                {
                    "card_type": 9,     # 微博卡片
                    "mblog": {
                        "text": "...",          # HTML 格式内容
                        "user": {"screen_name": "..."},
                        "created_at": "...",
                        "reposts_count": 0,
                        "comments_count": 0,
                        "attitudes_count": 0,   # 点赞数
                        "id": "...",
                        "mid": "...",
                    }
                },
                {
                    "card_type": 11,    # 群组标题 (跳过)
                }
            ]
        }
    }
    """
    items: list[SearchResultItem] = []
    cards = data.get("data", {}).get("cards", [])

    for card in cards:
        if card.get("card_type") != 9:  # 只处理微博卡片
            continue

        mblog = card.get("mblog", {})
        if not mblog:
            continue

        user = mblog.get("user", {})
        screen_name = user.get("screen_name", "")
        text = self._clean_html(mblog.get("text", ""))
        mid = mblog.get("mid", mblog.get("id", ""))
        uid = user.get("id", "")

        items.append(SearchResultItem(
            title=f"@{screen_name}: {text[:50]}...",
            url=f"https://weibo.com/{uid}/{mid}" if uid and mid else "",
            content=text[:500],
            source_engine=SearchEngine.WEIBO,
            published_date=mblog.get("created_at", ""),
            metadata={
                "author": screen_name,
                "reposts": mblog.get("reposts_count", 0),
                "comments": mblog.get("comments_count", 0),
                "likes": mblog.get("attitudes_count", 0),
                "publish_time": mblog.get("created_at", ""),
            },
        ))

        if len(items) >= max_results:
            break

    return SearchResponse(
        query=query,
        results=items,
        result_count=len(items),
        search_time_ms=0,
        engines_used=["weibo"],
    )
```

### 3.3 HTML 清洗工具

```python
@staticmethod
def _clean_html(html_text: str) -> str:
    """清洗微博 HTML 格式文本。

    移除:
    - HTML 标签 (保留文本)
    - 表情图片 <img> → [emoji_name]
    - @提及链接 → @用户名
    """
    import re
    # 表情图片 → alt text
    text = re.sub(r'<img[^>]*alt="([^"]*)"[^>]*/>', r'[\1]', html_text)
    # 移除剩余 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 清理多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

---

## 4. Playwright 备选搜索

```python
async def _search_playwright(
    self,
    query: str,
    max_results: int,
    time_scope: str,
) -> SearchResponse:
    """通过 Playwright 渲染桌面版搜索。

    用于 Mobile API 失败时的降级方案。

    s.weibo.com 特征:
    - 需要桌面端 Cookie (与移动端不同)
    - 未登录时重定向到 passport.weibo.com
    - 搜索结果为 HTML 卡片，需要 DOM 解析
    """
    cookies = await self._cookie_manager.load()

    async with self._pool.acquire(cookies=cookies) as context:
        page = await context.new_page()
        try:
            # 构建搜索 URL
            params = f"?q={query}"
            if time_scope:
                scope_map = {"hour": "1", "day": "2", "week": "3"}
                scope_value = scope_map.get(time_scope)
                if scope_value:
                    params += f"&timescope=custom:{scope_value}"

            await page.goto(
                f"{self.DESKTOP_SEARCH_URL}{params}",
                wait_until="domcontentloaded",
                timeout=15000,
            )

            # 检测登录态
            if await self._is_login_redirect(page):
                raise CookieExpiredError("weibo")

            return await self._extract_desktop_results(page, query, max_results)
        finally:
            await page.close()
```

---

## 5. 搜索路由逻辑

```python
async def search(
    self,
    query: str,
    max_results: int = 10,
    time_scope: str = "",
) -> SearchResponse:
    """搜索微博 — 三层降级策略。

    Layer 1: Mobile API (高效)
    Layer 2: Playwright Desktop (API 失败时)
    Layer 3: site:weibo.com via SearXNG (全部失败时, 在 server.py 层处理)
    """
    await self._rate_limiter.acquire()

    # Layer 1: Mobile API
    try:
        return await self._search_mobile_api(query, max_results, time_scope)
    except CookieExpiredError:
        raise  # Cookie 过期直接上抛，交给 server.py fallback
    except Exception as exc:
        self._logger.warning(
            "weibo_mobile_api_failed, trying playwright",
            extra={"error": str(exc)},
        )

    # Layer 2: Playwright Desktop
    return await self._search_playwright(query, max_results, time_scope)
```

---

## 6. Cookie 管理

### 6.1 关键 Cookie

| Cookie | 域名 | 作用 | 有效期 |
|:---|:---|:---|:---|
| `SUB` | `.weibo.com` | 登录凭证 | 7-30 天 |
| `SUBP` | `.weibo.com` | 登录辅助凭证 | 与 SUB 同步 |
| `XSRF-TOKEN` | `.weibo.com` | CSRF 保护 | 会话级 |

### 6.2 Mobile vs Desktop Cookie 兼容性

> [!WARNING]
> 微博的 Mobile 端 (`m.weibo.cn`) 和 Desktop 端 (`s.weibo.com`) 使用**相同的 `SUB` Cookie**，但：
> - Mobile API 需要在 HTTP Header 中手动发送
> - Desktop Playwright 由 BrowserPool 自动注入
> 
> 因此两种策略可以共用同一个 `data/cookies/weibo/cookies.json` 文件。

### 6.3 过期检测

```python
async def _is_login_redirect(self, page: Page) -> bool:
    """检测微博登录重定向。"""
    url = page.url
    return "passport.weibo.com" in url or "login" in url.lower()
```

---

## 7. MCP Server 集成

```python
# server.py 变更
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
        time_scope: "" (all), "hour", "day", "week".
    """
    try:
        scraper = self._get_weibo_scraper()
        response = await scraper.search(query, max_results, time_scope)
        return response.model_dump()
    except CookieExpiredError:
        self._logger.warning("weibo_cookie_expired, fallback to site: search")
    except Exception as exc:
        self._logger.warning("weibo_scraper_failed: %s, fallback", exc)

    # Fallback: site: search
    assert self._router is not None
    response = await self._router.search(
        f"site:weibo.com {query}", max_results=max_results,
    )
    return response.model_dump()
```

---

## 8. Rate Limiting

| 配置 | 值 | 理由 |
|:---|:---|:---|
| `weibo_rpm` | 10 | 微博反爬触发阈值约 12-15 RPM |
| Mobile API 特殊处理 | 同一 Limiter | 两种策略共享配额 |
| 位置 | `search()` 入口 | 在策略选择之前限流 |

---

## 9. 验收标准

| 测试用例 | 说明 |
|:---|:---|
| `test_mobile_api_search` | Mobile API 返回 JSON → 正确解析为 SearchResponse |
| `test_mobile_api_cookie_expired` | API 返回 302 → CookieExpiredError |
| `test_mobile_api_error_fallback_to_playwright` | API 返回 500 → 降级到 Playwright |
| `test_playwright_search` | Playwright → 提取搜索结果 |
| `test_playwright_login_redirect` | passport.weibo.com 重定向 → CookieExpiredError |
| `test_time_scope_filter` | time_scope="day" → 传递正确参数 |
| `test_rate_limiting` | 超过 10 RPM → 阻塞等待 |
| `test_html_cleaning` | HTML 标签/表情图片正确清洗 |
| `test_metadata_extraction` | 转发/评论/点赞数正确提取 |
| `test_card_type_filtering` | card_type=11 (非微博) 被过滤 |

---

## 10. 行业价值分析与主流实现方式

### 10.1 功能实际价值评估

| 维度 | 评估 | 说明 |
|:---|:---|:---|
| **数据独特性** | ⭐⭐⭐⭐ | 微博是中文互联网最大的公共舆论场，实时性极强，突发事件和热点话题的第一信息源 |
| **对 Deep Research 的贡献** | ⭐⭐⭐ | 在"热点事件"、"公众舆论"、"行业动态"类查询中提供实时视角，但内容深度有限（140 字限制） |
| **ROI（投入产出比）** | ⭐⭐⭐⭐ | Mobile API 方案轻量高效，Cookie 管理复杂度低于知乎 |
| **时效性价值** | ⭐⭐⭐⭐⭐ | 微博内容分钟级更新，是 Deep Research 获取"正在发生"的事件的关键数据源 |

**核心价值**: 微博提供其他渠道难以获得的**实时公共舆论数据**，特别是在突发事件、行业动态、政策解读等场景。但需注意微博内容的碎片化特点——它更适合作为"信号源"而非"深度内容源"。

### 10.2 业界主流实现方式

#### 双策略架构 — 行业验证

2025-2026 年业界在微博数据采集上的共识：

| 方案 | 行业采用度 | 定位 |
|:---|:---|:---|
| **Mobile API (m.weibo.cn)** | ⭐⭐⭐⭐⭐ 行业首选 | 高效、结构化 JSON、资源消耗低 |
| **Playwright 桌面渲染** | ⭐⭐⭐ 备用方案 | API 失败时的降级路径 |
| **托管爬虫服务 (Apify等)** | ⭐⭐⭐⭐ 上升趋势 | 企业用户外包反爬对抗复杂性 |
| **微博开放平台 API** | ⭐⭐ 有限 | 搜索功能受限，需要企业资质认证 |

> [!TIP]
> **行业验证**: 我们的"Mobile API 首选 + Playwright 备选 + SearXNG 兜底"三级降级架构与行业最佳实践高度一致。行业领先的社交媒体数据采集方案（如 Apify 的 Weibo Scraper）也采用类似的分层策略。

#### 关键技术挑战 — 行业经验

1. **Sina Visitor System (SVS)**: 微博要求管理临时 visitor cookie 来访问 API，这些 cookie 过期快、需要自动刷新。行业实践是实现事件驱动的重新认证机制（遇到 401/403 时自动触发刷新）
2. **API 稳定性**: 微博频繁更新 API 路径、payload 结构和签名算法。行业经验表明使用内部 API 的脚本需要**持续维护**
3. **Geetest 验证码**: 微博使用 Geetest 验证码系统，业界主要通过严格限速来规避触发，而非尝试破解
4. **法律合规**: 2025 年中国的数据爬取法律环境显著趋严，行业实践强调尊重 robots.txt、避免收集 PII、不干扰平台正常运行

#### 竞品对比

| 产品 | 微博数据获取 | 方式 |
|:---|:---|:---|
| **Perplexity AI** | ❌ 不支持 | 不索引中文社交媒体 |
| **ChatGPT Search** | ❌ 不支持 | Bing 对微博索引有限 |
| **秘塔 AI 搜索** | ✅ 部分支持 | 具体方案未知 |
| **Kimi Chat** | ✅ 部分支持 | 可能通过搜索引擎间接获取 |
| **我们** | ✅ 计划支持 | Mobile API + Playwright 双策略 |

### 10.3 设计方案评审 — 与行业最佳实践的差距

| 项 | 当前设计 | 行业最佳实践 | 差距 | 建议 |
|:---|:---|:---|:---|:---|
| API 路径管理 | 硬编码 containerid | 可配置 + 版本监控 | 中 | 加配置项，便于 API 变更时快速调整 |
| Cookie 刷新 | 手动刷新 | 事件驱动自动重认证 | 中 | Phase 3 实现自动 Cookie 刷新 |
| 行为模拟 | 未设计 | 移动端行为模拟（User-Agent 轮换） | 低 | 已在 header 中配置移动端 UA |
| 数据过滤 | card_type 过滤 | 内容质量评分 + 去重 | 中 | 可在 Agent 侧通过 LLM 评估内容质量 |
| 合规性 | 未明确 | robots.txt 尊重 + PII 过滤 | 低 | 需补充合规策略文档 |

---

## 11. 待确认事项

> [!NOTE]

1. **Mobile API 当前可用性**: `m.weibo.cn/api/container/getIndex` 的 `containerid` 格式可能已变化。需要在实现前用 curl 验证一下实际请求格式。

2. **Cookie 跨端兼容性**: Mobile Cookie 和 Desktop Cookie 是否仍然共用 `SUB`？需要在实际设备上验证。

3. **time_scope 传递**: Mobile API 是否支持时间过滤？目前设计中 time_scope 只在 Playwright 路径生效。如果 Mobile API 也支持，需要找到对应参数名。

4. **搜索结果分页**: 当前设计只取第一页。微博搜索结果每页约 10 条，是否需要支持分页以获取更多结果？

5. **Geetest 验证码**: 微博使用 Geetest 验证码，如果被触发，当前设计仅记录错误并降级。是否需要集成验证码解决服务（如 2Captcha）？
