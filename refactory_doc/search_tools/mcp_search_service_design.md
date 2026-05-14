# 自建 MCP 搜索服务 — 系统设计文档

> **目标**：为 deep_research_agent 构建一个自托管的 MCP (Model Context Protocol) 搜索服务，支持百度/搜狗/知乎/小红书/微博等中国大陆平台搜索，替代 Tavily 作为核心搜索后端。

---

## 1. 设计决策与约束

### 1.1 核心约束
| 约束 | 说明 |
|:---|:---|
| 网络环境 | 中国大陆服务器，无法访问 Google/DuckDuckGo/Brave 等被墙服务 |
| 成本目标 | 尽可能免费，仅服务器成本 |
| 技术栈 | Python 生态（与现有 deep_research_agent 一致） |
| 协议 | MCP (Model Context Protocol)，使用 FastMCP Python SDK |
| 传输层 | Streamable HTTP（生产部署）+ stdio（本地开发） |

### 1.2 关键设计决策

**决策 1：MCP Server 用 Python 而非 TypeScript**
- *理由*：与现有 deep_research_agent 技术栈一致，复用 Playwright async API，团队无需维护两种语言。OneSearch MCP 虽是 TypeScript，但我们不 fork 它，而是用 Python 重建核心逻辑。

**决策 2：双层架构 = SearXNG（通用搜索骨干）+ Playwright 爬虫（垂直平台）**
- *理由*：SearXNG 已处理好百度/搜狗的 SERP 解析和多引擎聚合，自己重写毫无意义。垂直平台（知乎/小红书/微博）SearXNG 无法覆盖，才需要 Playwright。

**决策 3：优先级排序 = 百度 > 知乎 > 微博 > 小红书**
- *理由*：百度通过 SearXNG 零成本获得；知乎内容质量最高且反爬相对可控；微博适合时效性搜索；小红书反爬最强（动态签名 + Native 层加密），ROI 最低。

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Deep Research Agent                       │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐   │
│  │Supervisor│───▶│ Worker   │───▶│ internet_search()    │   │
│  └──────────┘    └──────────┘    │ (LangChain Tool)     │   │
│                                  └──────────┬───────────┘   │
└─────────────────────────────────────────────┤───────────────┘
                                              │ HTTP / stdio
                                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    MCP Search Server                        │
│                  (FastMCP Python SDK)                        │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Tool Registry (MCP Tools)                │   │
│  │  ┌────────────┐ ┌────────────┐ ┌──────────────────┐  │   │
│  │  │ web_search │ │zhihu_search│ │xiaohongshu_search│  │   │
│  │  └─────┬──────┘ └─────┬──────┘ └────────┬─────────┘  │   │
│  │  ┌─────┴──────┐ ┌─────┴──────┐ ┌────────┴─────────┐  │   │
│  │  │weibo_search│ │ scrape_url │ │  health_check     │  │   │
│  │  └─────┬──────┘ └─────┬──────┘ └──────────────────┘  │   │
│  └────────┼──────────────┼───────────────────────────────┘   │
│           │              │                                   │
│  ┌────────▼──────────────▼───────────────────────────────┐   │
│  │              Search Backend Layer                      │   │
│  │                                                        │   │
│  │  ┌──────────────┐    ┌─────────────────────────────┐   │   │
│  │  │SearXNG Client│    │  Playwright Browser Pool    │   │   │
│  │  │(HTTP JSON)   │    │  (asyncio.Semaphore=3)      │   │   │
│  │  │              │    │  ┌───────┐ ┌───────┐        │   │   │
│  │  │ 百度/搜狗/   │    │  │Context│ │Context│ ...    │   │   │
│  │  │ 360/Bing     │    │  │ Pool  │ │ Pool  │        │   │   │
│  │  └──────┬───────┘    │  └───────┘ └───────┘        │   │   │
│  │         │            └──────────────┬──────────────┘   │   │
│  └─────────┼───────────────────────────┼──────────────────┘   │
│            │                           │                     │
│  ┌─────────▼───────────────────────────▼──────────────────┐   │
│  │              Result Normalizer                          │   │
│  │  Raw HTML/JSON  →  SearchResult(Pydantic)               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                    │                          │
                    ▼                          ▼
            ┌──────────────┐         ┌──────────────────┐
            │   SearXNG    │         │   Target Sites   │
            │  (Docker)    │         │  知乎/微博/小红书 │
            │  Port: 8080  │         │                  │
            └──────────────┘         └──────────────────┘
```

### 2.2 组件职责

| 组件 | 职责 | 技术选型 |
|:---|:---|:---|
| **MCP Server** | 暴露标准 MCP 工具接口，管理生命周期 | `mcp[cli]` (FastMCP) |
| **Cache Layer** | **[架构建议]** 拦截重复请求，大幅降低 API 调用和被封锁风险 | `redis.asyncio` / `aiocache` |
| **SearXNG Client** | 通用搜索的 JSON API 调用 | `httpx` async client |
| **Browser Pool** | 管理 Playwright 浏览器实例，控制并发 | `playwright`, `asyncio.Semaphore` |
| **Platform Scrapers** | 各平台搜索逻辑（知乎/微博/小红书） | Playwright + BeautifulSoup |
| **Result Normalizer** | 统一各来源结果为标准 Pydantic schema | `pydantic` |
| **SearXNG Instance** | 元搜索引擎（百度/搜狗/360） | Docker compose |

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

## 5. 各平台搜索策略分析

### 5.1 百度 / 搜狗 / 360（通过 SearXNG）

| 维度 | 分析 |
|:---|:---|
| **实现方式** | 完全委托 SearXNG，不写任何爬虫代码 |
| **反爬难度** | SearXNG 社区持续维护引擎适配器 |
| **稳定性** | ⭐⭐⭐⭐ 高（社区驱动） |
| **维护成本** | 几乎为零，跟随 SearXNG Docker 镜像更新即可 |
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
  # 禁用所有被墙引擎
  - name: google
    disabled: true
  - name: duckduckgo
    disabled: true

outgoing:
  request_timeout: 8.0
  max_request_timeout: 12.0
```

### 5.2 知乎搜索

| 维度 | 分析 |
|:---|:---|
| **实现方式** | Playwright 浏览器自动化 |
| **核心难点** | `x-zse-96` 动态加密签名 + 登录态校验 |
| **策略** | 使用持久化 `storage_state`（Cookie JSON），避免每次登录 |
| **反爬等级** | ⭐⭐⭐⭐ 高 |
| **推荐方案** | Playwright + stealth.js + 预登录 Cookie |

**实现要点：**
1. 首次手动登录知乎，导出 `storage_state.json`
2. 后续启动浏览器上下文时加载此文件，实现免登录
3. 搜索 URL: `https://www.zhihu.com/search?type=content&q={query}`
4. 解析搜索结果列表的 DOM 元素
5. 每次请求间隔 2-5 秒随机延迟

### 5.3 微博搜索

| 维度 | 分析 |
|:---|:---|
| **实现方式** | Playwright 浏览器自动化（移动端 m.weibo.cn） |
| **核心难点** | 频率限制 + 登录校验 |
| **策略** | 优先使用移动端接口（反爬较轻） |
| **反爬等级** | ⭐⭐⭐ 中高 |
| **推荐方案** | 移动端 UA + Cookie 注入 + 代理 IP |

**实现要点：**
1. 使用移动端 URL: `https://m.weibo.cn/search?containerid=100103type%3D1%26q%3D{query}`
2. 移动端反爬相比 PC 端更宽松
3. 解析 XHR 返回的 JSON 数据
4. Cookie 有效期较短，需定期刷新机制

### 5.4 小红书搜索（Phase 3 可选，高风险）

| 维度 | 分析 |
|:---|:---|
| **实现方式** | Playwright + 复杂签名逆向 |
| **核心难点** | `X-Sign` / `a-bogus` 动态签名在 Native 层 (.so)，逆向极难 |
| **反爬等级** | ⭐⭐⭐⭐⭐ 极高 |
| **维护成本** | 极高，签名算法频繁更新 |
| **推荐** | **暂不实现**，等待社区成熟方案或官方 API |

> [!WARNING]
> 小红书的反爬是所有平台中最强的。动态签名逻辑位于 App 的 Native 层，纯 Python 逆向几乎不可能长期维护。建议 Phase 1/2 不做小红书，仅在用户强需求时通过百度 `site:xiaohongshu.com` 间接搜索。

---

## 6. Browser Pool 设计

```python
import asyncio
import logging
from contextlib import asynccontextmanager
from playwright.async_api import async_playwright, Browser, BrowserContext

logger = logging.getLogger(__name__)

class BrowserPool:
    """管理 Playwright 浏览器实例和并发上下文。"""

    STEALTH_JS_PATH = "resources/stealth.min.js"

    def __init__(self, max_concurrency: int = 3):
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._browser: Browser | None = None
        self._playwright = None

    async def start(self) -> None:
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        logger.info("Browser pool started (max_concurrency=%d)",
                     self._semaphore._value)

    async def stop(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser pool stopped")

    @asynccontextmanager
    async def acquire_context(
        self,
        storage_state: str | None = None,
        mobile: bool = False,
    ):
        """获取一个浏览器上下文，自动控制并发。
        
        [Architect's Note - 生产级加固]: 
        在高可用环境中，Playwright 存在隐性的内存泄漏风险。
        1. 必须在此实现生命周期管理：记录总请求数，每处理 N 个请求强制重启底层 Chromium 进程。
        2. 建议引入令牌桶限流 (Token Bucket)，针对每个域名（如 zhihu.com）限制出站并发和频率（如 1次/3秒），防止瞬间峰值导致代理 IP 或账号被永久封停。
        """
        async with self._semaphore:
            ctx_kwargs = {}
            if storage_state:
                ctx_kwargs["storage_state"] = storage_state
            if mobile:
                ctx_kwargs["user_agent"] = (
                    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                    "AppleWebKit/605.1.15"
                )
                ctx_kwargs["viewport"] = {"width": 390, "height": 844}

            context = await self._browser.new_context(**ctx_kwargs)

            # 注入 stealth.js 防检测
            await context.add_init_script(path=self.STEALTH_JS_PATH)

            try:
                yield context
            finally:
                await context.close()
```

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

## 9. Docker 部署方案

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

  # Phase 2: MCP Server 作为独立服务
  # mcp-search:
  #   build: ../
  #   depends_on: [searxng]
  #   environment:
  #     - SEARXNG_URL=http://searxng:8080
  #   ports:
  #     - "8090:8090"
```

**服务器配置需求：**

| 组件 | CPU | 内存 | 磁盘 |
|:---|:---|:---|:---|
| SearXNG | 0.5 核 | 256-512 MB | 1 GB |
| MCP Server (无浏览器) | 0.5 核 | 256 MB | 500 MB |
| MCP Server (含 Playwright) | 1-2 核 | 1-2 GB | 2 GB |
| **总计（推荐）** | **2 核** | **2-4 GB** | **5 GB** |

---

## 10. 分阶段实施路线

### Phase 1：SearXNG + 直接函数调用（3-5 天）

- [ ] Docker 部署 SearXNG，配置百度/搜狗引擎
- [ ] 实现 `search_service/models.py`（Pydantic 数据模型，预留 metadata 扩展）
- [ ] **[架构建议]** 实现基础语义缓存层 (Redis / 内存 LRU)，作为降本和防封禁的核心防御机制
- [ ] 实现 `search_service/backends/searxng_client.py`（async httpx）并加入指数退避重试
- [ ] 实现 `search_service/backends/base.py`（SearchBackend Protocol）
- [ ] 修改 `_make_internet_search()` 使用 SearXNG 后端
- [ ] 修改 Settings 新增 `searxng_url` 及缓存相关配置项
- [ ] 端到端测试：用 `run_deep_agent.py` 验证搜索功能

### Phase 2：知乎 + 微博搜索 + MCP 标准化（1-2 周）

- [ ] 实现 `search_service/browser/pool.py`（BrowserPool）
- [ ] 实现 `search_service/backends/zhihu_scraper.py`
- [ ] 实现 `search_service/backends/weibo_scraper.py`
- [ ] 实现 `search_service/server.py`（FastMCP 入口）
- [ ] 实现 `search_service/backends/page_scraper.py`（通用页面抓取）
- [ ] 接入 `langchain-mcp-adapters` 连接 MCP Server
- [ ] 集成测试全部搜索工具

### Phase 3：生产加固 + 可选平台（2-4 周）

- [ ] **[已移至 Phase 1] 缓存层实现** (越早实现收益越大)
- [ ] 健康检查 `/health` 端点与 Prometheus 指标暴露
- [ ] Cookie 自动刷新机制
- [ ] 小红书搜索（视需求决定）
- [ ] 负载测试与稳定性验证

---

## 11. 风险评估

| 风险 | 级别 | 影响 | 缓解措施 |
|:---|:---|:---|:---|
| SearXNG 百度引擎适配器失效 | 中 | 通用搜索不可用 | 跟进社区更新；备选直接 HTTP 爬取 |
| 知乎 Cookie 过期 | 高 | 知乎搜索失败 | 自动检测过期 + 告警通知手动刷新 |
| 微博 IP 封禁 | 中 | 微博搜索失败 | 代理 IP 轮换；降低请求频率 |
| Playwright 内存泄漏 | 中 | 服务器 OOM | BrowserPool 定时重启；内存监控 |
| 平台 ToS 违规 | 高 | 法律风险 | 控制请求频率；仅搜索公开内容；法律评估 |
| 小红书签名逆向失败 | 极高 | 小红书搜索无法实现 | 不在 Phase 1/2 实现；用 `site:` 搜索替代 |

---

## User Review Required

> [!IMPORTANT]
> **以下问题需要你的确认后才能开始实施：**

1. **Phase 1 的搜索后端选择**：
   - 方案 A：纯 SearXNG（零额外成本，但需要一台服务器部署 Docker）
   - 方案 B：博查 API 作为 Phase 1 + SearXNG 作为 Phase 2 兜底
   - 你倾向哪个？

2. **服务器资源**：你是否已有可部署 Docker 的云服务器？规格如何？

3. **知乎/微博的需求优先级**：Phase 2 是否需要尽快启动？还是 Phase 1（通用搜索）够用后再说？

4. **MCP 传输层**：Phase 1 你接受直接函数调用（最简单）还是希望一步到位用 MCP 协议？

5. **小红书**：确认暂不实现？还是有强需求？

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
