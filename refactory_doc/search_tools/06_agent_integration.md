# S06: Agent Integration

> **Phase**: 1 (直接函数调用) + Phase 2 (MCP 协议)  
> **预估工时**: 1 天  
> **修改文件**: `agents/agent.py`, `agents/tools.py`, `agents/prompts.py`, `config/settings.py`  
> **依赖**: S01 (models), S02 (SearXNG Client)  
> **下游**: 无 (终端集成层)

---

## 1. 目标

- Phase 1: 替换 `_make_internet_search()` 从 Tavily → SearXNG（直接函数调用）
- Phase 2: 切换到 MCP 协议 (`langchain-mcp-adapters`)
- 更新 Worker prompt 增加 `site:` 搜索策略指导
- 保持 tool name/description 在两个 Phase 之间一致

---

## 2. Phase 1: 直接函数调用

### 2.1 修改 `_make_internet_search()` (位于 `agents/tools.py`)

**Before** (当前 Tavily 实现):
```python
def _make_internet_search(api_key: str):
    from tavily import TavilyClient
    client = TavilyClient(api_key=api_key)
    def internet_search(query, max_results=10, topic="general", ...) -> dict:
        return client.search(query, ...)
    return internet_search
```

**After** (SearXNG 实现):
```python
async def _make_internet_search(config: SearchServiceConfig) -> Callable:
    """创建搜索工具函数，使用 SearXNG 后端。

    返回的 async 函数签名与 Tavily 兼容，确保 Worker prompt 无需修改。
    LangGraph ToolNode 自动处理 async Callable → BaseTool 转换。
    """
    from search_service.backends.searxng_client import SearXNGClient
    from search_service.backends.base import SearchRouter
    from search_service.cache import NullCache

    client = SearXNGClient(config)
    router = SearchRouter(backends=[client], cache=NullCache())

    async def internet_search(
        query: str,
        max_results: int = 10,
        topic: str = "general",
    ) -> dict:
        """Search the internet for current, factual information.

        Use this tool to find up-to-date information about any topic.
        For targeted searches, use site: prefix (e.g. "site:arxiv.org LLM agents").

        Args:
            query: The search query. Supports site: operator for targeted search.
            max_results: Maximum number of results (1-20).
            topic: Search topic hint — general, news, or finance.

        Returns:
            Search results with titles, urls, and content snippets.
        """
        # 根据 topic 调整引擎选择
        engines = None
        if topic == "news":
            engines = ["bing"]  # Bing 新闻覆盖好

        response = await router.search(
            query, max_results=max_results, engines=engines
        )

        # 转换为 Tavily 兼容格式 (Worker prompt 依赖此结构)
        return {
            "query": response.query,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "content": r.content,
                }
                for r in response.results
            ],
        }

    return internet_search
```

**关键变更**：
1. 参数从 `api_key: str` 改为 `config: SearchServiceConfig`
2. 返回 `async def` 而非 `def`（LangGraph ToolNode 原生支持）
3. 函数名保持 `internet_search`（ToolNode 用此名生成 tool.name）
4. 返回格式保持 Tavily 兼容 `{"query": ..., "results": [{"title", "url", "content"}]}`

### 2.2 修改 `_load_settings()` (位于 `agents/tools.py`)

```python
def _load_settings() -> dict[str, Any]:
    load_dotenv()
    try:
        from deep_research_agent.config import Settings
        s = Settings.load()
        from search_service.config import SearchServiceConfig
        search_config = SearchServiceConfig()
        return {
            "search_config": search_config,
            "planner_model": s.planner_model,
            "worker_model": s.worker_model,
        }
    except Exception:
        from search_service.config import SearchServiceConfig
        return {
            "search_config": SearchServiceConfig(),
            "planner_model": os.environ.get("PLANNER_MODEL", "deepseek-chat"),
            "worker_model": os.environ.get("WORKER_MODEL", "deepseek-chat"),
        }
```

### 2.3 修改 `build_deep_agent()` (位于 `agents/agent.py`)

```python
async def build_deep_agent(*, checkpointer=None, **overrides):
    cfg = _load_settings()
    search_tool = await _make_internet_search(cfg["search_config"])
    # ... 其余不变
```

> [!WARNING]
> `build_deep_agent` 从 `def` 变为 `async def`，因为 `_make_internet_search` 现在可能需要异步初始化（如检查 SearXNG 连通性）。调用方需要相应调整。
>
> 如果不想改变 `build_deep_agent` 的签名，可将 `_make_internet_search` 改为同步工厂，内部的 `internet_search` 仍为 async。这样 `SearXNGClient` 在首次搜索时懒初始化。

**简化方案（推荐）**：

```python
def _make_internet_search(config: SearchServiceConfig):
    """同步工厂函数，返回 async 搜索工具。Client 懒初始化。"""
    client = SearXNGClient(config)  # 不在此处建立连接
    router = SearchRouter(backends=[client], cache=NullCache())

    async def internet_search(query: str, max_results: int = 10, topic: str = "general") -> dict:
        # ... 同上
        ...

    return internet_search
```

这样 `build_deep_agent` 保持同步，无需改变上游调用方式。

---

## 3. Worker Prompt 更新

在 `agents/prompts.py` 中的 `DeepAgentPrompts.WORKER` 中增加 `site:` 搜索策略指导：

```python
# 在 WORKER prompt 的 "## Research Strategy" 部分追加：

"""
## Search Optimization Techniques

### Targeted Site Search
When researching specific domains, use the `site:` operator to focus your search:

- **Technical content**: `site:csdn.net` or `site:juejin.cn` for Chinese tech articles
- **Academic papers**: `site:arxiv.org` for research papers
- **Industry analysis**: `site:36kr.com` or `site:huxiu.com` for business insights
- **Financial data**: `site:xueqiu.com` for investment analysis
- **Product reviews**: `site:sspai.com` for digital product reviews
- **News**: `site:thepaper.cn` for in-depth news reporting
- **Stack Overflow**: `site:stackoverflow.com` for programming Q&A
- **Wikipedia**: `site:zh.wikipedia.org` for encyclopedic knowledge

Example: `site:arxiv.org multi-agent deep research system 2026`

### Multi-Site Search
Combine multiple sites with OR for broader coverage:
`site:csdn.net OR site:juejin.cn LangGraph 状态管理`

### Language Awareness
- For Chinese queries: results will primarily come from Baidu/Sogou
- For English queries: results will come from Bing (cn.bing.com)
- Mix languages if the topic spans both: use English for technical terms
"""
```

---

## 4. Phase 2: MCP 协议集成

### 4.1 使用 `langchain-mcp-adapters`

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

async def _make_mcp_tools():
    """通过 MCP 协议获取搜索工具。"""
    client = MultiServerMCPClient(servers=[
        {
            "command": "python",
            "args": ["-m", "search_service.server"],
            "transport": "stdio",
        }
    ])
    return await client.get_tools()
```

### 4.2 Tool Name 一致性

| Phase | Tool Name | 来源 |
|:---|:---|:---|
| Phase 1 | `internet_search` | 函数名自动推导 |
| Phase 2 | `web_search` | MCP Server 定义 |

> [!IMPORTANT]
> Phase 2 的 MCP tool 名为 `web_search`（遵循 MCP 设计文档），与 Phase 1 的 `internet_search` 不同。需要同步更新 Worker prompt 中对工具名的引用。

**解决方案**：Phase 1 就将函数名改为 `web_search`，提前与 Phase 2 对齐。

```python
async def web_search(query: str, max_results: int = 10, topic: str = "general") -> dict:
    """Search the web for current, factual information. ..."""
```

---

## 5. Config 扩展 (`config/settings.py`)

```python
# 新增字段到现有 Settings dataclass
@dataclass(frozen=True)
class Settings:
    # ... 现有字段 ...

    # Search Service (新增)
    searxng_base_url: str  # 默认 http://localhost:8080

    @classmethod
    def load(cls) -> Settings:
        yaml_data = cls._load_yaml_defaults()
        return cls(
            # ... 现有字段 ...
            searxng_base_url=cls._env(
                "SEARCH_SEARXNG_BASE_URL",
                yaml_data.get("searxng_base_url", "http://localhost:8080"),
            ),
        )
```

---

## 6. 验收标准

### Phase 1

```bash
# 1. 确保 SearXNG Docker 运行中
docker compose -f docker/docker-compose.yml ps

# 2. 端到端测试
python examples/run_deep_agent.py "中国AI产业最新发展趋势"
python examples/run_deep_agent.py "What is LangGraph and how does it work?"
python examples/run_deep_agent.py "对比 RAG 和 Fine-tuning 的优缺点"

# 3. 验证搜索结果来源
# 在日志中确认 backend=searxng，而非 tavily
```

### 验收指标

| 指标 | 目标 |
|:---|:---|
| 中文查询端到端成功 | 3/3 |
| 英文查询端到端成功 | 1/1 |
| 搜索结果数 (每个查询) | ≥5 |
| Agent 生成报告质量 | 与 Tavily 版本可比 |
| 无 Tavily 依赖 | `pip show tavily` 后可安全移除 |
