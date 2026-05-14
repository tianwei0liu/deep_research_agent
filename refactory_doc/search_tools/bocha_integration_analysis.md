# `search_service` 模块价值分析：接入博查 API 后的架构决策

> **分析日期**: 2026-05-13  
> **分析目标**: 评估接入博查 API 后 `search_service` 模块的剩余价值，并推荐最优集成路径

---

## 1. `search_service` 模块现状能力盘点

先拆解这个模块到底提供了什么：

| 组件 | 文件 | 核心能力 | 博查能替代？ |
|:---|:---|:---|:---:|
| **SearchRouter** | `backends/base.py` | Fallback 链 + Cache-aside + 结果后处理 | ❌ 不能 |
| **SearXNGClient** | `backends/searxng_client.py` | SearXNG JSON API 对接 + 指数退避重试 | ✅ 可替代（搜索源） |
| **SearchResultFilter** | `backends/result_filter.py` | URL 去重 + 空内容移除 + n-gram 相关性过滤（中英双语） | ❌ 不能 |
| **PageScraper** | `backends/page_scraper.py` | Playwright 渲染 + HTML→Markdown 提取 | ❌ 不能 |
| **GitHubClient** | `backends/github_client.py` | GitHub Search API 对接 + Rate Limit 处理 | ❌ 不能 |
| **BrowserPool** | `browser/pool.py` + `stealth.py` | Playwright 浏览器池 + Stealth 指纹伪装 | ❌ 不能 |
| **CacheLayer** | `cache.py` | 缓存协议 + TTL 策略 + Key 生成 | ❌ 不能 |
| **MCP Server** | `server.py` | 6 个 MCP 工具注册 + 生命周期管理 | ⚠️ 部分替代 |
| **Data Models** | `models.py` | Pydantic 统一数据契约（SearchResponse/SearchResultItem） | ❌ 不能 |
| **Exceptions** | `exceptions.py` | 7 种自定义异常分类 | ❌ 不能 |

### 关键发现

**博查 API 只替代了一个东西：搜索源（SearXNG → 博查）。** 模块中约 70% 的代码提供的是搜索源之外的基础设施能力：路由/容错、结果质量过滤、页面抓取、缓存、数据标准化。这些能力博查 API 不提供，也不应该由博查提供。

---

## 2. 三种集成架构对比

### 方案 A：直接将博查 API 作为 LangChain Tool 给 Agent

```
Agent (LangGraph)
  ├─ bocha_web_search (LangChain @tool)    ← 直接 HTTP 调用博查
  ├─ github_search (LangChain @tool)        ← 直接调用 GitHub API  
  └─ scrape_url (LangChain @tool)           ← 需要自己实现或用 Firecrawl
```

```python
# 实现示例（约 30 行）
@tool
async def bocha_web_search(query: str, count: int = 10) -> dict:
    """搜索互联网获取实时信息。"""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.bochaai.com/v1/web-search",
            headers={"Authorization": f"Bearer {BOCHA_API_KEY}"},
            json={"query": query, "count": count},
        )
        return resp.json()
```

| 维度 | 评估 |
|:---|:---|
| **实现成本** | ⭐ 极低（30 行代码，0.5 天） |
| **Fallback 容错** | ❌ 无。博查挂了 = 搜索全挂 |
| **结果质量过滤** | ❌ 无。依赖博查原始排序，无去重/噪声过滤 |
| **缓存** | ❌ 无。相同 Query 重复计费 |
| **Scrape URL** | ❌ 丢失。需要另外实现或引入 Firecrawl |
| **GitHub 搜索** | ❌ 丢失。需要另外实现 |
| **垂直平台搜索（知乎/微博/微信）** | ❌ 丢失。无 site: 前缀路由 |
| **MCP 标准化** | ❌ 丢失。工具硬编码在 Agent 中 |
| **search_service 模块命运** | 🗑️ 废弃 |

> [!WARNING]
> **方案 A 的根本问题**：它是一个原型级方案。你会失去 `search_service` 中所有的生产级基础设施（容错、缓存、过滤、scrape），并且把搜索的可用性完全绑定在一个外部 API 上。任何头部玩家都不会这么做——ChatGPT 有 Bing API，但它还有 OAI-SearchBot 做二次抓取；Claude 有 Brave API，但它有 Dynamic Filtering 做结果预处理。

---

### 方案 B：通过现有 MCP 服务接入（博查作为 MCP Server 的新 Backend）

```
Agent (LangGraph)
  └─ MCPSearchClient (stdio)
       └─ SearchMCPServer (server.py)
            ├─ web_search → SearchRouter
            │                 ├─ Backend 1: BochaClient (新增，优先)
            │                 └─ Backend 2: SearXNGClient (fallback)
            ├─ github_search → GitHubClient
            ├─ scrape_url → PageScraper + BrowserPool
            ├─ zhihu_search → site: prefix via SearchRouter
            ├─ weibo_search → site: prefix via SearchRouter
            └─ weixin_search → site: prefix via SearchRouter
```

```python
# 新增 BochaClient，实现 SearchBackend 协议（约 80 行）
class BochaClient:
    @property
    def name(self) -> str:
        return "bocha"

    async def search(self, query: str, max_results: int = 10, **kwargs) -> SearchResponse:
        # HTTP 调用博查 API，转换为统一 SearchResponse
        ...
```

| 维度 | 评估 |
|:---|:---|
| **实现成本** | ⭐⭐ 低（新增 `BochaClient` 约 80 行，修改 `server.py` startup 约 5 行） |
| **Fallback 容错** | ✅ 博查优先 → SearXNG fallback，`AllProvidersExhaustedError` 兜底 |
| **结果质量过滤** | ✅ 保留 `SearchResultFilter`（去重 + 噪声过滤） |
| **缓存** | ✅ 保留 Cache-aside，相同 Query 不重复计费 |
| **Scrape URL** | ✅ 保留 PageScraper + BrowserPool |
| **GitHub 搜索** | ✅ 保留 GitHubClient |
| **垂直平台搜索** | ✅ 保留 site: 前缀路由（V1），未来可扩展为真实 Scraper |
| **MCP 标准化** | ✅ 保留。Agent 端零改动 |
| **search_service 模块命运** | ✅ 增强 — 博查成为新的一级 Backend |

> [!TIP]
> **方案 B 的核心优势**：Agent 层完全不需要改动。`MCPSearchClient` 发现的工具不变（`web_search`, `scrape_url`, `github_search`, ...），只是 `web_search` 背后的 `SearchRouter` 多了一个更可靠的 Backend。这是最符合"开闭原则"的方案。

---

### 方案 C：混合架构 — 博查直连 + search_service 保留非搜索能力（不推荐）

```
Agent (LangGraph)
  ├─ bocha_web_search (直连 LangChain Tool)  ← 搜索走博查直连
  └─ MCPSearchClient (stdio)
       └─ SearchMCPServer
            ├─ scrape_url → PageScraper
            ├─ github_search → GitHubClient
            ├─ zhihu_search → site: via SearXNG (仍需 SearXNG)
            └─ weibo_search / weixin_search → SearXNG
```

| 维度 | 评估 |
|:---|:---|
| **实现成本** | ⭐⭐⭐ 中等（两套集成路径，维护复杂） |
| **Fallback 容错** | ⚠️ 部分。web_search 无 fallback，其他工具有 |
| **架构一致性** | ❌ 差。两种 Tool 提供方式混合（直连 + MCP），增加认知负担 |
| **search_service 模块命运** | ⚠️ 半废弃 — SearchRouter 和缓存被绕过，模块沦为 scrape+GitHub 的壳 |

> 不推荐。混合两种 Tool 提供路径增加了不必要的复杂度，且绕过了 SearchRouter 的所有基础设施能力。

---

## 3. 业界如何接入博查 API

### 3.1 博查官方提供的集成方式

| 方式 | 技术栈 | 适合场景 |
|:---|:---|:---|
| **REST API 直调** | `httpx.post("https://api.bochaai.com/v1/web-search")` | 最基础的集成 |
| **LangChain @tool** | 自定义 `@tool` 装饰器 | 快速原型验证 |
| **官方 MCP Server** | `bocha-search-mcp`（GitHub: BochaAI/bocha-search-mcp） | 任何 MCP 客户端（Claude Desktop, Cursor） |
| **Dify 插件** | Dify 市场预置插件 | 低代码平台 |
| **Open WebUI 插件** | 内置集成 | 聊天界面 |

### 3.2 DeepSeek 的集成方式（最值得参考）

DeepSeek 是博查的标杆客户。它的集成方式是：
- **平台层直调**：在 DeepSeek 的后端服务中，通过 HTTP 直接调用博查 API
- **不经过 MCP**：DeepSeek 有自己的 Tool Calling 协议，不使用 MCP
- **有结果后处理**：搜索结果返回后，DeepSeek 用小模型做相关性打分和过滤

### 3.3 LangGraph Agent 的典型集成模式

在 LangGraph 生态中，接入博查的主流方式有两种：

**模式 1：langchain-mcp-adapters（推荐）**
```python
from langchain_mcp_adapters.tools import load_mcp_tools
# 通过 MCP 协议发现博查工具，与我们现有架构完全一致
```

**模式 2：直接 @tool 封装**
```python
@tool
async def bocha_search(query: str) -> dict:
    """..."""
    # 适合快速原型，不适合生产
```

> [!IMPORTANT]
> **业界共识**：生产级 Agent 不会把搜索 API 裸露给模型。它们都有一个中间层做：(1) 结果过滤/排序，(2) 缓存，(3) 容错/重试，(4) 可观测性。这正是我们 `SearchRouter` 的职责。

---

## 4. 推荐方案：方案 B（博查作为 SearchRouter 的一级 Backend）

### 4.1 为什么选方案 B

| 决策因素 | 方案 A（直连） | **方案 B（MCP 内接入）** |
|:---|:---|:---|
| 生产可靠性 | ❌ 单点故障 | ✅ 双 Backend fallback |
| 成本控制 | ❌ 每次查询都计费 | ✅ Cache-aside 节省重复查询费用 |
| 结果质量 | ❌ 裸结果 | ✅ 去重 + 噪声过滤 |
| 改动范围 | Agent 层 + 新 Tool | 仅 search_service 内部 |
| 向后兼容 | ❌ 破坏 MCP 架构 | ✅ Agent 层零改动 |
| scrape_url 保留 | ❌ 丢失 | ✅ 保留 |

### 4.2 具体实施路径

```
Step 1: 新增 search_service/backends/bocha_client.py（~80 行）
  - 实现 SearchBackend 协议
  - HTTP 调用博查 API
  - 转换响应为 SearchResponse

Step 2: 修改 search_service/server.py startup()（~5 行改动）
  - BochaClient 加入 SearchRouter.backends，排在 SearXNG 之前
  - 仅当 BOCHA_API_KEY 配置时才启用

Step 3: 修改 search_service/config.py（~3 行）
  - 新增 bocha_api_key 和 bocha_base_url 配置项

Step 4: 修改 search_service/models.py（~1 行）
  - SearchEngine 枚举新增 BOCHA = "bocha"

预估总工时: 0.5-1 天
```

### 4.3 博查接入后的搜索流

```
Agent 调用 web_search("AI Agent 框架对比 2026")
    ↓
MCPSearchClient → SearchMCPServer → SearchRouter.search()
    ↓
1. Cache-aside 检查 → 命中则直接返回（省钱）
    ↓ miss
2. 尝试 BochaClient.search()
    ↓ 成功
3. SearchResultFilter.filter() → 去重 + 噪声过滤
    ↓
4. 写入缓存 → 返回结果
    ↓ BochaClient 失败（余额不足/网络异常）
5. Fallback → SearXNGClient.search()
    ↓ 也失败
6. raise AllProvidersExhaustedError
```

---

## 5. `search_service` 模块在博查接入后的价值总结

| 能力 | 价值 | 说明 |
|:---|:---:|:---|
| **SearchRouter（路由/容错）** | 🔴 核心 | 博查 → SearXNG fallback，单点故障 → 高可用 |
| **SearchResultFilter（结果过滤）** | 🔴 核心 | 博查返回的结果也需要去重和噪声过滤 |
| **CacheLayer（缓存）** | 🟡 重要 | 相同 Query 不重复调用博查，直接省钱 |
| **PageScraper（页面抓取）** | 🔴 核心 | 博查只返回 snippet，深度阅读需要 scrape_url |
| **GitHubClient** | 🟡 重要 | 博查不支持 GitHub 搜索 |
| **BrowserPool + Stealth** | 🟡 重要 | scrape_url 和未来垂直 Scraper 的基础 |
| **MCP Server 协议层** | 🟢 有用 | 保持 Agent 与搜索实现解耦 |
| **统一数据模型** | 🔴 核心 | 博查/SearXNG/GitHub 结果统一为 `SearchResponse` |
| **异常分类** | 🟢 有用 | `AllProvidersExhaustedError` 等精确错误处理 |

**一句话结论**：`search_service` 模块的价值不在于"它连接了 SearXNG"，而在于"它提供了搜索源之上的生产级基础设施"。接入博查后，SearXNG 从主力降级为 fallback，但 SearchRouter、Filter、Cache、Scraper 的价值不减反增。
