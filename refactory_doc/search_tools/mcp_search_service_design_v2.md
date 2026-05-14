# 自建 MCP 搜索服务 — 系统设计文档 (v2)

> **目标**：为 deep_research_agent 构建一个自托管的 MCP (Model Context Protocol) 搜索服务，支持百度/搜狗/知乎/微博等中国大陆平台搜索，替代 Tavily 作为核心搜索后端。

> **v2 变更说明**：本版本在 v1 基础上增加了行业调研与架构定位（§0）、设计决策的深度论证（§1-2 重写）、搜索策略细化（§5-6 增强）、实施路线量化验收标准（§10-11 增强）。

---

# §0. 行业背景与架构定位

> **核心命题**：AI Agent 的搜索系统应该怎么设计？行业里主流的解决方式是什么？

---

## 0.1 双阶段模型：Discovery → Extraction

行业共识是将搜索拆为两阶段：

1. **Discovery（发现）**：通过搜索 API/索引快速定位候选 URL（100-800ms，HTTP 轻量调用）
2. **Extraction（提取）**：对高价值 URL 用浏览器深度提取内容（1-8s/页，重量级渲染）

**为什么不全部用 Playwright 爬虫？** — 5 个维度的量化对比：

| 维度 | SearXNG (Discovery) | Playwright (Extraction) | 全 Playwright |
|:---|:---|:---|:---|
| **延迟/查询** | 200-800ms | 2-8s/页 | 2-8s × N 页 |
| **内存/并发** | ~0 MB (HTTP) | ~150-300 MB/上下文 | 150-300 MB × N |
| **可靠性** | 高 (社区维护) | 中 (泄漏/崩溃) | 低 |
| **维护成本** | 几乎为零 | 需跟踪 DOM 变更 | 极高 |
| **合规风险** | 低 (搜索引擎) | 高 (直接爬取) | 高 |

> Discovery 是高频低成本操作，Extraction 是低频高成本操作。分离后 80% 查询只做 Discovery，仅 20% 追加 Extraction — 经典 Pareto 优化。

---

## 0.2 行业产品架构光谱

```
自研索引 ←─────────────────────────────────────→ 纯 API
Perplexity    ChatGPT     Cursor/开源Agent    简单Wrapper
(自建索引     (Bing API   (SearXNG+           (Tavily
 +RAG引擎)    +Fetcher)    Playwright)         直调)
                              ▲
                              我们在这里
```

### Perplexity AI
- Discovery: 自建索引(百亿网页) + BM25/Dense 混合检索
- Extraction: 自建提取 + 三层 ML Reranking
- 启示: 不可复制，但验证了"搜索质量=Agent质量"

### ChatGPT Search
- Discovery: Bing API
- Extraction: 自建 Fetcher (OAI-SearchBot)
- 启示: 即使 OpenAI 也不自建索引，复用成熟搜索引擎

### Cursor / Windsurf
- Discovery: MCP 接入 SearXNG 或商业 API
- Extraction: Playwright MCP Server（提取 Accessibility Tree）
- 启示: **MCP + SearXNG + Playwright 已是 AI Agent 社区事实标准**

---

## 0.3 我们的架构定位

| 约束 | 影响 |
|:---|:---|
| 中国大陆网络 | Google/Bing/DDG/Brave 全部不可用 |
| 小团队 (1-2人) | 不可能自建搜索索引 |
| 需要垂直平台内容 | 知乎/微博/小红书无法通过通用引擎覆盖 |

**定位**：开源 Agent 光谱的中国大陆特化版本：
- **Discovery**: **博查 API（主力）** + SearXNG（百度/搜狗/360，fallback）
- **Extraction**: Playwright（JS 渲染、登录态、垂直平台爬取）

> **v2.1 更新（2026-05-13）**：基于 [模块价值分析](../../search_service/bocha_integration_analysis.md) 和 [行业调研（§9）](captcha_mitigation_strategy.md)，将博查 API 从"备选"提升为"主力 Discovery 后端"，SearXNG 降级为 fallback。这与 DeepSeek 的搜索架构路径一致 — DeepSeek 同样使用博查 API 作为中文搜索后端。详见 [§12. 博查 API 集成设计](#12-博查-api-集成设计)。

---


---

## 1. 设计决策与约束

### 1.1 核心约束

| 约束 | 说明 |
|:---|:---|
| 网络环境 | 中国大陆服务器，Google/DuckDuckGo/Brave 等被墙 |
| 成本目标 | 尽可能免费，仅服务器成本 |
| 技术栈 | Python（与 deep_research_agent 一致） |
| 协议 | MCP (Model Context Protocol)，使用 FastMCP Python SDK |
| 传输层 | Streamable HTTP（生产）+ stdio（开发） |

### 1.2 关键设计决策

#### 决策 1：Python 而非 TypeScript

*理由*：与 deep_research_agent 技术栈一致，复用 Playwright async API，无需维护双语言栈。OneSearch MCP 虽为 TypeScript，但我们借鉴其接口设计，用 Python 重建。

#### 决策 2：双层架构 = SearXNG (Discovery) + Playwright (Extraction)

这是本文档最关键的设计决策，需要正面回答：**为什么不全部用 Playwright？**

**核心论点：SearXNG 和 Playwright 解决的是根本不同的问题。**

- **SearXNG 解决 "在哪找"**：它利用百度/搜狗/360 的**已有索引**（数百亿网页），用 HTTP API 在 200-800ms 内返回排序后的候选 URL 列表。这是我们不可能自建的能力。
- **Playwright 解决 "怎么取"**：对于需要 JS 渲染、登录态、或 SearXNG 无法覆盖的垂直平台，Playwright 提供深度内容提取。

**纯 Playwright 方案不可行的 5 个原因：**

| # | 原因 | 详细说明 |
|:---|:---|:---|
| 1 | **没有索引** | Playwright 只能访问你给它的 URL。它不知道"哪些网页和 query 相关" — 这需要搜索引擎的倒排索引，构建成本是数亿美元级别的。 |
| 2 | **性能灾难** | 单次页面渲染 2-8 秒，若每次搜索需爬取 10+ 页面 = 20-80 秒/查询，完全不可接受。SearXNG 的 JSON API 在 200-800ms 内完成。 |
| 3 | **资源消耗** | 每个 Playwright 上下文 ~150-300MB 内存。3 并发 = 450-900MB。10 并发 = 1.5-3GB。而 SearXNG 作为 HTTP 客户端内存消耗可忽略。 |
| 4 | **反爬对抗** | 百度/搜狗对直接爬取有严格反爬（验证码、IP 封禁）。SearXNG 社区持续维护引擎适配器，帮我们打赢了反爬战争。 |
| 5 | **合规风险** | 直接爬取搜索引擎 SERP 违反其 ToS。SearXNG 作为开源元搜索引擎，是行业公认的合规方式。 |

**决策结论**：SearXNG 和 Playwright 不是替代关系，而是互补关系。SearXNG 做不了深度提取，Playwright 做不了快速发现。两者组合才是完整方案。

#### 决策 3：为什么不用 Firecrawl / Jina Reader 替代 Playwright？

| 方案 | 中国可用 | 说明 |
|:---|:---|:---|
| Firecrawl | ❌ 不可用 | 商业服务，服务器在海外，中国大陆无法访问 |
| Jina Reader | ❌ 不可用 | 德国公司 (s.jina.ai)，未在中国备案 |
| 自建 Playwright | ✅ 完全可控 | 唯一可行的 Extraction 方案 |

在中国大陆网络约束下，Playwright 自建是 Extraction 层的**唯一选择**。

#### 决策 4：优先级排序 = 百度 > 知乎 > 微博 > 小红书

*理由*：百度通过 SearXNG 零成本获得；知乎内容质量最高且反爬相对可控；微博适合时效性搜索；小红书反爬极强（Native 层加密），ROI 最低。

---

## 2. 系统架构

### 2.1 双阶段 Pipeline 架构图

```
+-----------------------------------------------------------------+
|                     Deep Research Agent                          |
|  Supervisor -> Worker -> web_search() / scrape_url()            |
+---------------------------+-------------------------------------+
                            | MCP Protocol (HTTP / stdio)
                            v
+-----------------------------------------------------------------+
|                      MCP Search Server                          |
|                                                                  |
|  +----------------------------------------------------------+   |
|  |                   Tool Registry                           |   |
|  |                                                           |   |
|  |  +- Discovery Tools ---------------------------------+   |   |
|  |  |  web_search -> SearchRouter (Fallback Chain)       |   |   |
|  |  |                 +- BochaClient  (博查API, 主力)    |   |   |
|  |  |                 +- SearXNGClient (百度/搜狗, 备选) |   |   |
|  |  +----------------------------------------------------+   |   |
|  |                                                           |   |
|  |  +- Extraction Tools ---------------------------------+   |   |
|  |  |  scrape_url    -> Playwright (通用页面提取)         |   |   |
|  |  |  zhihu_search  -> site: prefix via SearchRouter     |   |   |
|  |  |  weibo_search  -> site: prefix via SearchRouter     |   |   |
|  |  |  weixin_search -> site: prefix via SearchRouter     |   |   |
|  |  +----------------------------------------------------+   |   |
|  |                                                           |   |
|  |  +- Specialized Tools --------------------------------+   |   |
|  |  |  github_search -> GitHubClient (REST API)          |   |   |
|  |  +----------------------------------------------------+   |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  +----------+ +----------+ +-----------+ +------------------+    |
|  | Cache    | | Bocha    | | SearXNG   | | Browser Pool     |    |
|  | Layer    | | Client   | | Client    | | (Playwright)     |    |
|  |(NullCache| |(httpx    | |(httpx     | |                  |    |
|  | -> Redis)| | async)   | | async)    | |                  |    |
|  +----------+ +----------+ +-----------+ +------------------+    |
|                                                                   |
|  +-----------------------------------------------------------+   |
|  |  SearchResultFilter -> Dedup + Noise Removal + Relevance  |   |
|  +-----------------------------------------------------------+   |
+-----------------------------------------------------------------+
         |              |                       |
         v              v                       v
  +--------------+ +----------+      +------------------+
  |   SearXNG    | | Bocha    |      |   Target Sites   |
  |  (Docker)    | | API      |      |  知乎/微博/百度   |
  +--------------+ +----------+      +------------------+
```

### 2.2 关键设计：scrape_url 的职责

`scrape_url` 是连接 Discovery 和 Extraction 的桥梁：

1. Agent Worker 调用 `web_search`（Discovery）获得 URL 列表
2. Worker 阅读摘要后，对高价值 URL 调用 `scrape_url`（Extraction）获取完整内容
3. 这个两步模式让 Agent 自主决定"哪些内容值得深挖"

这与 ChatGPT Search 的模式完全一致：Bing API 返回候选 → Fetcher 提取内容 → LLM 综合。

### 2.3 组件职责

| 组件 | 职责 | 阶段 | 技术选型 |
|:---|:---|:---|:---|
| **MCP Server** | 暴露标准 MCP 工具接口 | — | `mcp[cli]` (FastMCP) |
| **Cache Layer** | 拦截重复请求，降低成本和封禁风险 | 前置 | `redis.asyncio` / 内存 LRU |
| **SearXNG Client** | 通用搜索 JSON API | Discovery | `httpx` async |
| **Browser Pool** | 管理 Playwright 浏览器实例 | Extraction | `playwright`, `asyncio.Semaphore` |
| **Platform Scrapers** | 各垂直平台搜索逻辑 | Extraction | Playwright + BeautifulSoup |
| **Result Normalizer** | 统一结果为 Pydantic schema | 后置 | `pydantic` |
| **SearXNG Instance** | 元搜索引擎 (Docker) | Discovery | Docker compose |

---

## 3. 数据模型

```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class SearchEngine(str, Enum):
    BAIDU = "baidu"
    SOGOU = "sogou"
    ZHIHU = "zhihu"
    WEIBO = "weibo"
    XIAOHONGSHU = "xiaohongshu"
    SEARXNG = "searxng"  # 多引擎聚合

class SearchResultItem(BaseModel):
    """单条搜索结果，对标 Tavily 的返回格式。"""
    title: str
    url: str
    content: str = Field(description="摘要或正文片段")
    source_engine: SearchEngine
    published_date: Optional[str] = None
    score: Optional[float] = None
    raw_content: Optional[str] = Field(
        default=None, description="完整页面内容（可选）"
    )
    metadata: dict = Field(
        default_factory=dict, 
        description="平台特定扩展元数据（如作者、点赞数、评论数），确保接口向后兼容的同时提供强扩展性"
    )

class SearchResponse(BaseModel):
    """搜索工具的统一返回格式。"""
    query: str
    results: list[SearchResultItem]
    result_count: int
    search_time_ms: int
    engines_used: list[str]

class ScrapeResponse(BaseModel):
    """页面抓取的返回格式。"""
    url: str
    title: str
    content: str  # Markdown 格式
    content_length: int
```

> [!IMPORTANT]
> `SearchResultItem` 的字段设计刻意对标 Tavily 的返回格式（title, url, content），确保上游 `_make_internet_search()` 替换后，Worker prompt 无需修改。


---

## 4. MCP 工具定义

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("deep-research-search")

@mcp.tool()
async def web_search(
    query: str,
    max_results: int = 10,
    engines: str = "baidu,sogou",
    time_range: str = "",
) -> dict:
    """通用网页搜索。通过 SearXNG 聚合多个搜索引擎。

    Args:
        query: 搜索关键词。
        max_results: 最大返回结果数 (1-20)。
        engines: 逗号分隔的引擎列表 (baidu,sogou,360,bing)。
        time_range: 时间范围过滤 (day, week, month, year)。
    """
    ...

@mcp.tool()
async def zhihu_search(
    query: str,
    max_results: int = 10,
    search_type: str = "general",
) -> dict:
    """知乎站内搜索。获取高质量问答和专栏内容。

    Args:
        query: 搜索关键词。
        max_results: 最大返回结果数 (1-10)。
        search_type: 搜索类型 (general, question, article, user)。
    """
    ...

@mcp.tool()
async def weibo_search(
    query: str,
    max_results: int = 10,
    search_type: str = "general",
) -> dict:
    """微博搜索。获取实时热点和舆情信息。

    Args:
        query: 搜索关键词。
        max_results: 最大返回结果数 (1-10)。
        search_type: 搜索类型 (general, realtime, hot)。
    """
    ...

@mcp.tool()
async def scrape_url(url: str) -> dict:
    """抓取指定 URL 的页面内容，转换为 Markdown 格式。

    Args:
        url: 要抓取的网页 URL。
    """
    ...
```

---


---

## 5. 各平台搜索策略深度分析

### 5.1 请求路由决策树

Agent Worker 收到搜索任务后，MCP Server 内部的路由逻辑：

```
Query 进入
    │
    ├─ 通用搜索 (web_search)
    │   └─ SearXNG Client → 百度/搜狗/360 聚合
    │       └─ 返回 URL + 摘要 (Discovery)
    │
    ├─ 知乎搜索 (zhihu_search)
    │   └─ Playwright → zhihu.com 站内搜索
    │       └─ 返回问答内容 (Discovery + Extraction)
    │
    ├─ 微博搜索 (weibo_search)
    │   └─ Playwright → m.weibo.cn 搜索
    │       └─ 返回微博内容 (Discovery + Extraction)
    │
    └─ 页面抓取 (scrape_url)
        └─ Playwright → 目标 URL
            └─ 返回 Markdown 内容 (Extraction)
```

> 垂直平台搜索 (zhihu/weibo) 是 Discovery+Extraction 的混合操作：Playwright 同时完成"搜索"和"内容提取"，因为这些平台没有可用的搜索 API。

### 5.2 百度/搜狗/360（通过 SearXNG）

| 维度 | 分析 |
|:---|:---|
| **实现方式** | 完全委托 SearXNG，不写任何爬虫代码 |
| **反爬难度** | SearXNG 社区持续维护引擎适配器 |
| **稳定性** | ⭐⭐⭐⭐ 高（社区驱动） |
| **维护成本** | 几乎为零，跟随 Docker 镜像更新 |
| **搜索质量** | 百度中文搜索质量高，搜狗/360 作为补充 |

**SearXNG settings.yml 关键配置：**
```yaml
search:
  formats: [html, json]
  default_lang: zh-CN

engines:
  - name: baidu
    engine: baidu
    disabled: false
    timeout: 6.0
    weight: 2.0          # 百度权重最高
  - name: sogou
    engine: sogou
    disabled: false
    timeout: 5.0
  - name: 360search
    engine: xpath
    disabled: false
    timeout: 5.0
  - name: google
    disabled: true
  - name: duckduckgo
    disabled: true

outgoing:
  request_timeout: 8.0
  max_request_timeout: 12.0
```

### 5.3 知乎搜索

| 维度 | 分析 |
|:---|:---|
| **实现方式** | Playwright 浏览器自动化 |
| **核心难点** | `x-zse-96` 动态加密签名 + 登录态校验 |
| **反爬等级** | ⭐⭐⭐⭐ 高 |
| **推荐方案** | Playwright + stealth.js + 预登录 Cookie |

**Cookie 生命周期管理：**
1. **初始化**：手动登录知乎，通过 `context.storage_state()` 导出 `zhihu_storage.json`
2. **使用**：启动 BrowserContext 时加载此文件，实现免登录
3. **过期检测**：每次请求后检查响应状态，若返回登录页 → 标记 Cookie 过期
4. **告警**：Cookie 过期时通过 logging.error + 可选 webhook 通知运维手动刷新
5. **降级**：Cookie 过期期间，知乎搜索自动降级为 `web_search` + `site:zhihu.com`

**搜索实现要点：**
- URL: `https://www.zhihu.com/search?type=content&q={query}`
- 随机延迟 2-5 秒/请求
- 提取 DOM: `.SearchResult-Card` 内的标题、摘要、作者、赞数
- 对每个结果页面可选深度提取完整答案内容

### 5.4 微博搜索

| 维度 | 分析 |
|:---|:---|
| **实现方式** | Playwright（移动端 m.weibo.cn） |
| **核心难点** | 频率限制 + 短 Cookie 有效期 |
| **反爬等级** | ⭐⭐⭐ 中高 |
| **推荐方案** | 移动端 UA + Cookie 注入 |

**实现要点：**
- 移动端 URL: `https://m.weibo.cn/search?containerid=100103type%3D1%26q%3D{query}`
- 移动端反爬比 PC 端宽松
- 解析 XHR 返回的 JSON 数据
- Cookie 有效期 ~24-48 小时，需定期刷新

### 5.5 小红书（Phase 3 可选，高风险）

| 维度 | 分析 |
|:---|:---|
| **反爬等级** | ⭐⭐⭐⭐⭐ 极高 |
| **核心难点** | `X-Sign`/`a-bogus` 签名在 Native .so 层 |
| **推荐** | **暂不实现**，用 `site:xiaohongshu.com` 间接搜索 |

> [!WARNING]
> 小红书反爬是所有平台中最强的。Phase 1/2 不做，仅在强需求时通过百度 `site:` 搜索替代。

---

## 6. Browser Pool 生产级设计

```python
class BrowserPool:
    """管理 Playwright 浏览器实例和并发上下文。
    
    生产加固要点:
    1. 进程回收: 每处理 MAX_REQUESTS_PER_BROWSER 次后强制重启 Chromium
    2. 域名级限流: 对每个目标域名独立限流 (Token Bucket)
    3. 内存监控: 定期检查进程内存，超阈值强制重启
    """

    MAX_REQUESTS_PER_BROWSER = 100  # 防止 Chromium 内存泄漏
    DOMAIN_RATE_LIMITS = {
        "zhihu.com": {"requests_per_minute": 12, "min_interval_seconds": 3},
        "m.weibo.cn": {"requests_per_minute": 10, "min_interval_seconds": 4},
        "default": {"requests_per_minute": 20, "min_interval_seconds": 1},
    }
```

### 6.1 并发控制

- **全局并发**: `asyncio.Semaphore(max_concurrency=3)` — 限制同时运行的浏览器上下文数
- **域名并发**: 每个域名独立的令牌桶，防止对单一平台产生流量峰值
- **请求间隔**: 每个域名的最小请求间隔 (见 DOMAIN_RATE_LIMITS)

### 6.2 生命周期管理

```
Browser 启动 → 处理请求 → 计数器++ → 达到阈值? → 重启 Browser
                                         ↓ 否
                                    继续处理
```

### 6.3 故障恢复

- Context 异常 → 自动关闭并重试一次
- Browser 崩溃 → 自动重启 Playwright 进程
- 内存超限 → 强制 kill 并重建

---

## 7. 与 Deep Research Agent 的集成

### 7.1 方式 A：直接函数调用（推荐 Phase 1）

最简方案：不启用 MCP 传输层，直接将搜索逻辑封装为 Python 函数，替换现有的 `_make_internet_search()`。

```python
# agents/tools.py 和 agents/agent.py 修改

def _make_internet_search(settings: dict):
    """创建搜索函数，使用自建搜索后端。"""
    from search_service.searxng_client import SearXNGClient

    client = SearXNGClient(base_url=settings["searxng_url"])

    async def internet_search(
        query: str,
        max_results: int = 10,
        topic: str = "general",
    ) -> dict:
        """搜索互联网获取最新信息。"""
        result = await client.search(query, max_results=max_results)
        # 转换为 Tavily 兼容格式
        return {
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "content": r.content,
                }
                for r in result.results
            ]
        }

    return internet_search
```

### 7.2 方式 B：通过 MCP 协议（Phase 2+）

使用 `langchain-mcp-adapters` 连接 MCP Server：

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_client = MultiServerMCPClient(servers=[
    {
        "command": "python",
        "args": ["-m", "search_service.server"],
        "transport": "stdio",  # 本地开发
    }
])

# 获取工具列表，直接传入 create_deep_agent
mcp_tools = mcp_client.get_tools()
```

---

## 8. 项目结构

```
deep_research_agent/
├── search_service/              # 新增：MCP 搜索服务
│   ├── __init__.py
│   ├── server.py                # FastMCP Server 入口
│   ├── config.py                # SearchServiceConfig (Pydantic Settings)
│   ├── models.py                # SearchResultItem, SearchResponse 等
│   ├── exceptions.py            # SearchProviderError 等自定义异常
│   │
│   ├── backends/                # 搜索后端实现
│   │   ├── __init__.py
│   │   ├── base.py              # SearchBackend Protocol
│   │   ├── searxng_client.py    # SearXNG HTTP 客户端
│   │   ├── zhihu_scraper.py     # 知乎搜索 (Playwright)
│   │   ├── weibo_scraper.py     # 微博搜索 (Playwright)
│   │   └── page_scraper.py      # 通用页面抓取
│   │
│   ├── browser/                 # 浏览器管理
│   │   ├── __init__.py
│   │   ├── pool.py              # BrowserPool
│   │   └── stealth.py           # Anti-detection 配置
│   │
│   └── resources/               # 静态资源
│       └── stealth.min.js       # Puppeteer stealth 脚本
│
├── docker/                      # Docker 编排
│   ├── docker-compose.yml       # SearXNG + MCP Server
│   └── searxng/
│       └── settings.yml         # SearXNG 引擎配置
│
└── tests/
    └── test_search_service/
        ├── test_searxng_client.py
        ├── test_zhihu_scraper.py
        └── test_models.py
```

---

## 9. Docker 部署方案（增强版）

```yaml
# docker/docker-compose.yml
version: "3.8"

services:
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    ports:
      - "8080:8080"
    volumes:
      - ./searxng/settings.yml:/etc/searxng/settings.yml:ro
    environment:
      - SEARXNG_BASE_URL=http://localhost:8080
    restart: unless-stopped
    mem_limit: 512m
    cpus: "0.5"
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:8080/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s

  # Phase 2: Redis 缓存
  # redis:
  #   image: redis:7-alpine
  #   ports: ["6379:6379"]
  #   mem_limit: 256m

  # Phase 2: MCP Server
  # mcp-search:
  #   build: ../
  #   depends_on: [searxng, redis]
  #   environment:
  #     - SEARXNG_URL=http://searxng:8080
  #     - REDIS_URL=redis://redis:6379/0
  #   ports: ["8090:8090"]
  #   mem_limit: 2g
```

**服务器配置需求：**

| 组件 | CPU | 内存 | 磁盘 |
|:---|:---|:---|:---|
| SearXNG | 0.5 核 | 256-512 MB | 1 GB |
| MCP Server (无 Playwright) | 0.5 核 | 256 MB | 500 MB |
| MCP Server (含 Playwright) | 1-2 核 | 1-2 GB | 2 GB |
| Redis (可选) | 0.25 核 | 128-256 MB | 500 MB |
| **总计（推荐）** | **2-3 核** | **2-4 GB** | **5 GB** |

---

## 10. 分阶段实施路线（优化版）

### Phase 1：SearXNG + 直接函数调用（3-5 天）

**验收标准**：用 `run_deep_agent.py` 进行 3 个中文 query 的端到端测试，搜索结果数量 ≥5 且延迟 <3s。

- [ ] Docker 部署 SearXNG，配置百度/搜狗引擎
- [ ] 实现 `search_service/models.py`（Pydantic 数据模型）
- [ ] 实现基础缓存层（内存 LRU，语义缓存 Phase 2 再做）
- [ ] 实现 `search_service/backends/searxng_client.py`（httpx + 指数退避重试）
- [ ] 实现 `search_service/backends/base.py`（SearchBackend Protocol）
- [ ] 修改 `_make_internet_search()` 使用 SearXNG
- [ ] 端到端测试 + 与 Tavily 结果质量对比

### Phase 2：知乎/微博 + MCP 标准化（1-2 周）

**验收标准**：知乎/微博搜索可用率 ≥90%（10 次查询 ≥9 次成功）。

- [ ] 实现 `browser/pool.py`（BrowserPool + 域名限流）
- [ ] 实现 `backends/zhihu_scraper.py`
- [ ] 实现 `backends/weibo_scraper.py`
- [ ] 实现 `server.py`（FastMCP 入口）
- [ ] 实现 `backends/page_scraper.py`（scrape_url）
- [ ] 接入 `langchain-mcp-adapters`
- [ ] Cookie 过期检测 + 告警

### Phase 3：生产加固（2-4 周）

**验收标准**：连续 7 天搜索服务可用率 ≥95%。

- [ ] Redis 缓存层（替代内存 LRU）
- [ ] 健康检查 `/health` + Prometheus 指标
- [ ] Cookie 自动刷新（半自动：检测过期→通知→手动刷新→自动加载）
- [ ] Browser Pool 进程回收 + 内存监控
- [ ] 搜索质量持续评估（见下方）

---

## 10.1 搜索质量持续评估机制

**问题**：替换搜索后端后，如何确保 Agent 的研究质量不下降？

**方案**：旁路监控 + LLM-as-Judge

```
正常流量 ──────────────────────→ Agent 正常工作
     │
     └── 每 100 个 query 抽样 1 个
              │
              ├── 发送到新后端 (SearXNG) → 结果 A
              ├── 发送到旧后端 (Tavily)  → 结果 B (若可用)
              │
              └── LLM-as-Judge 评分
                  - 召回相关性 (0-10)
                  - 内容覆盖度 (0-10)
                  - 时效性 (0-10)
                  → 写入评估日志
```

---

## 11. 风险评估（增强版）

| 风险 | 概率 | 影响 | 风险分 | 缓解措施 | SLA 目标 |
|:---|:---|:---|:---|:---|:---|
| SearXNG 百度适配器失效 | 中 | 高 | 🟡 | 社区跟进；备选直接爬取 | 48h 内恢复 |
| 知乎 Cookie 过期 | 高 | 中 | 🟡 | 过期检测+告警+site:降级 | 4h 内刷新 |
| 微博 IP 封禁 | 中 | 低 | 🟢 | 降低频率；代理 IP | 24h 内恢复 |
| Playwright 内存泄漏 | 中 | 中 | 🟡 | 进程回收+内存监控 | 自动恢复 |
| 平台 ToS 违规 | 低 | 极高 | 🔴 | 控制频率；仅搜索公开内容 | 法律评估 |
| 搜索质量退化 | 中 | 高 | 🟡 | LLM-as-Judge 旁路监控 | 周报审查 |

---

## Verification Plan

### Automated Tests
```bash
# 单元测试
python -m pytest tests/test_search_service/ -v

# SearXNG 连通性测试
curl -s "http://localhost:8080/search?q=test&format=json" | python -m json.tool

# 端到端测试
python examples/run_deep_agent.py "What is the current state of AI in China?"
```

### Manual Verification
- 对比 Tavily 和 SearXNG 对同一 query 的搜索结果质量
- 验证知乎/微博搜索的结果完整性和准确性
- 在中国大陆服务器上验证全链路延迟

---

## 12. 博查 API 集成设计

> **新增日期**: 2026-05-13
> **决策依据**: [search_service 模块价值分析](../../search_service/bocha_integration_analysis.md)
> **行业调研**: [头部 LLM 搜索架构深度拆解 (captcha_mitigation_strategy.md §9)](captcha_mitigation_strategy.md)
> **搜索备选方案对比**: [search_alternatives_analysis.md](search_alternatives_analysis.md)
> **详细实施设计**: [bocha_integration_detail_design.md](bocha_integration_detail_design.md) — 含 API 技术规格、字段映射、非代码工作清单、BochaClient 完整实现规格

### 12.1 设计目标

将博查 (Bocha) Search API 作为 `SearchRouter` 的一级 Backend 接入 `search_service`，实现：

1. **博查优先，SearXNG 兜底** — 博查 API 稳定性和结果质量优于 SearXNG 自建，但存在费用和额度限制
2. **Agent 层零改动** — `MCPSearchClient` 发现的 MCP 工具集不变，仅 `web_search` 背后的 Backend 优先级调整
3. **成本可控** — 通过 Cache-aside 避免相同 Query 重复计费

### 12.2 架构决策：为什么不直接给 Agent 一个博查 Tool？

> 详细分析见 [bocha_integration_analysis.md §2](../../search_service/bocha_integration_analysis.md)

我们评估了三种方案，选择了 **方案 B（博查作为 SearchRouter 的新 Backend）**：

| 决策因素 | 方案 A（直连 Tool） | **方案 B（MCP 内接入）** |
|:---|:---|:---|
| 生产可靠性 | 单点故障 | 双 Backend fallback |
| 成本控制 | 每次查询都计费 | Cache-aside 节省重复查询 |
| 结果质量 | 裸结果 | 去重 + 噪声过滤 |
| 改动范围 | Agent 层 + 新 Tool | 仅 search_service 内部 |
| 向后兼容 | 破坏 MCP 架构 | Agent 层零改动 |

> [!IMPORTANT]
> **行业验证**：DeepSeek 使用博查 API 时也有结果后处理层（小模型打分过滤），ChatGPT 用 Bing API 时也有 OAI-SearchBot 做二次抓取。**没有任何生产级 Agent 把搜索 API 裸露给模型**。我们的 `SearchRouter` + `SearchResultFilter` 正是这个中间层。

### 12.3 博查 API 概览

博查提供两类搜索接口：

| 接口 | 端点 | 用途 |
|:---|:---|:---|
| **Web Search** | `POST /v1/web-search` | 全网网页搜索，返回 title/URL/snippet |
| **AI Search** | `POST /v1/ai-search` | 语义增强搜索，额外返回结构化"模态卡"（天气/百科/股票等） |

**关键参数**：

```json
{
  "query": "AI Agent 框架对比 2026",
  "count": 10,
  "freshness": "oneWeek"
}
```

**认证方式**：Bearer Token
```
Authorization: Bearer {BOCHA_API_KEY}
```

### 12.4 新增文件：`search_service/backends/bocha_client.py`

实现 `SearchBackend` 协议，约 80 行：

```python
class BochaClient:
    """Bocha Search API async client.

    Searches via the Bocha Web Search API and converts results to
    :class:`SearchResponse`.

    Args:
        config: SearchServiceConfig instance.
    """

    API_BASE = "https://api.bochaai.com"

    def __init__(self, config: SearchServiceConfig) -> None:
        self._api_key = config.bocha_api_key
        self._base_url = config.bocha_base_url or self.API_BASE
        self._timeout = config.bocha_timeout_seconds  # 默认 10s
        self._client: Optional[httpx.AsyncClient] = None
        self._logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "bocha"

    async def search(
        self, query: str, max_results: int = 10, **kwargs: Any,
    ) -> SearchResponse:
        """Execute a search query against Bocha API."""
        client = await self._ensure_client()
        payload = {
            "query": query,
            "count": min(max_results, 20),
        }
        # 可选: freshness 参数透传
        if freshness := kwargs.get("time_range"):
            payload["freshness"] = self._map_time_range(freshness)

        try:
            response = await client.post(
                "/v1/web-search", json=payload,
            )
            response.raise_for_status()
            return self._parse_response(query, response.json())
        except httpx.HTTPStatusError as exc:
            raise SearchProviderError("bocha", str(exc), exc)
        except Exception as exc:
            raise SearchProviderError("bocha", str(exc), exc)

    async def health_check(self) -> bool:
        """Check Bocha API availability."""
        try:
            client = await self._ensure_client()
            resp = await client.post(
                "/v1/web-search",
                json={"query": "test", "count": 1},
                timeout=5.0,
            )
            return resp.status_code == 200
        except Exception:
            return False
```

### 12.5 修改清单

| 文件 | 变更类型 | 改动内容 |
|:---|:---|:---|
| `search_service/backends/bocha_client.py` | **新增** | `BochaClient` 类，实现 `SearchBackend` 协议 |
| `search_service/config.py` | 修改 | 新增 `bocha_api_key`, `bocha_base_url`, `bocha_timeout_seconds` 字段 |
| `search_service/models.py` | 修改 | `SearchEngine` 枚举新增 `BOCHA = "bocha"` |
| `search_service/server.py` | 修改 | `startup()` 中根据 `config.bocha_api_key` 条件性地将 `BochaClient` 插入 `SearchRouter.backends` 首位 |
| Agent 层 (`agents/`) | **无改动** | `MCPSearchClient` 动态发现的 MCP 工具集不变 |

### 12.6 Backend 优先级与 Fallback 策略

```
SearchRouter.backends 优先级（从高到低）:

1. BochaClient     <-- 仅当 BOCHA_API_KEY 配置时启用
2. SearXNGClient   <-- 始终启用，作为免费 fallback
```

**Fallback 触发条件**：

| BochaClient 错误 | 行为 |
|:---|:---|
| HTTP 401/403（API Key 无效/余额不足） | Fallback -> SearXNG |
| HTTP 429（限流） | Fallback -> SearXNG |
| HTTP 5xx（服务端错误） | Fallback -> SearXNG |
| 网络超时 | Fallback -> SearXNG |
| 响应解析失败 | Fallback -> SearXNG |

所有 Fallback 均由现有 `SearchRouter.search()` 的 `try/except SearchProviderError` 逻辑自动处理，无需额外代码。

### 12.7 搜索流时序图

```
Agent                MCPSearchClient       SearchRouter        BochaClient      SearXNGClient
  |                        |                    |                   |                  |
  |-- web_search(query) -->|                    |                   |                  |
  |                        |-- MCP call ------->|                   |                  |
  |                        |                    |-- cache check --->|                  |
  |                        |                    |   (miss)          |                  |
  |                        |                    |-- search() ------>|                  |
  |                        |                    |                   |-- POST /v1/ ---->|
  |                        |                    |                   |<-- results ------|
  |                        |                    |<-- SearchResponse-|                  |
  |                        |                    |                   |                  |
  |                        |                    |-- filter() ------>|                  |
  |                        |                    |-- cache set ------|                  |
  |                        |<-- response -------|                   |                  |
  |<-- results ------------|                    |                   |                  |
  |                        |                    |                   |                  |

  -- Fallback 场景 --
  |                        |                    |-- search() ------>|                  |
  |                        |                    |                   |-- ERROR ---------|
  |                        |                    |-- search() -------------------------------->|
  |                        |                    |                                      |
  |                        |                    |<-- SearchResponse -----------------------|
```

### 12.8 配置示例

```python
# search_service/config.py 新增字段
class SearchServiceConfig(BaseSettings):
    # ... 现有字段 ...

    # Bocha API configuration
    bocha_api_key: Optional[str] = Field(
        default=None,
        description="Bocha Search API key. When set, BochaClient is"
                    " registered as the primary search backend.",
    )
    bocha_base_url: str = Field(
        default="https://api.bochaai.com",
        description="Bocha API base URL.",
    )
    bocha_timeout_seconds: float = Field(
        default=10.0,
        description="Bocha API request timeout.",
    )
```

```bash
# .env 配置
BOCHA_API_KEY=sk-xxxx-your-key-here
# BOCHA_BASE_URL=https://api.bochaai.com  # 可选，默认即可
```

### 12.9 server.py startup() 修改

```python
async def startup(self) -> None:
    """Initialize all backends."""
    cache = NullCache()
    backends: list[SearchBackend] = []

    # 博查优先（仅当 API Key 配置时启用）
    if self._config.bocha_api_key:
        from search_service.backends.bocha_client import BochaClient
        backends.append(BochaClient(self._config))
        self._logger.info("bocha_backend_enabled")

    # SearXNG 始终作为 fallback
    backends.append(SearXNGClient(self._config))

    self._router = SearchRouter(backends=backends, cache=cache)
    # ... 其余不变 ...
```

### 12.10 预估工时与验收标准

**预估工时**: 0.5-1 天

**验收标准**:

| 验收项 | 标准 |
|:---|:---|
| BochaClient 单元测试 | Mock 博查 API，验证请求格式、响应解析、异常处理 |
| Fallback 集成测试 | Mock BochaClient 抛出异常 -> 自动 fallback 到 SearXNG |
| 缓存命中测试 | 相同 Query 二次调用 -> 走 Cache，不调用博查 API |
| 端到端测试 | `run_deep_agent.py` 执行 3 个中文 Query，结果数量 >= 5 |
| 无 API Key 降级 | 不配置 `BOCHA_API_KEY` -> 仅使用 SearXNG，行为与当前完全一致 |

---

## 相关文档索引

| 文档 | 说明 |
|:---|:---|
| [bocha_integration_analysis.md](../../search_service/bocha_integration_analysis.md) | `search_service` 模块价值分析：三种集成架构的详细对比与推荐 |
| [bocha_integration_detail_design.md](bocha_integration_detail_design.md) | 博查 API 集成详细设计：API 规格、字段映射、非代码清单、BochaClient 实现 |
| [captcha_mitigation_strategy.md](captcha_mitigation_strategy.md) | CAPTCHA 缓解策略全文，含 §9 头部 LLM 搜索架构深度拆解 |
| [search_alternatives_analysis.md](search_alternatives_analysis.md) | 搜索备选方案对比（Tavily / 博查 / SearXNG / MCP Scraper） |
| [implementation_progress.md](implementation_progress.md) | Phase 1 实施进度追踪 |
| [01_data_models_and_config.md](01_data_models_and_config.md) | 数据模型与配置详细设计 |
| [02_searxng_client.md](02_searxng_client.md) | SearXNG 客户端详细设计 |

