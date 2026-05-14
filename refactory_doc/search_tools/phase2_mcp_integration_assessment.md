# Phase 2: langchain-mcp-adapters 集成现状评估与补充设计

> **模块**: `agents/mcp_client.py`, `agents/agent.py`  
> **依赖**: `langchain-mcp-adapters>=0.1.0`  
> **日期**: 2026-05-05

---

## 1. 现状评估

### 1.1 已完成的工作

根据 [walkthrough](file:///home/tianwei/.gemini/antigravity/brain/5f2e02d0-f0ca-473c-9d21-f4ca1c183de7/walkthrough.md)，以下工作已完成:

| 完成项 | 详情 |
|:---|:---|
| ✅ `MCPSearchClient` | Async context manager, stdio subprocess spawn |
| ✅ 动态工具发现 | `load_mcp_tools(session)` → LangChain `BaseTool` |
| ✅ 工具名过滤 | `get_tools(names=[...])` → Supervisor/Worker 不同工具集 |
| ✅ `build_deep_agent` 异步化 | `async def build_deep_agent(mcp_client=...)` |
| ✅ MCP 会话生命周期 | `stream_deep_research` 内 `async with MCPSearchClient()` |
| ✅ `language` 参数 | `web_search` MCP tool 新增 `language` 参数 |
| ✅ Prompt 更新 | `internet_search` → `web_search`，平台工具文档 |
| ✅ 测试 | 245 passed, 15 new/updated tests |

### 1.2 Remaining Work 中的 MCP 项

原始 `implementation_progress.md` Phase 2 列出:
```
- 使用 langchain-mcp-adapters 的 MultiServerMCPClient 接入 MCP 协议
- Tool name 从 internet_search 统一为 web_search
```

### 1.3 差距分析

| 原始需求 | 当前状态 | 差距 |
|:---|:---|:---|
| MultiServerMCPClient | 使用了单 server `MCPSearchClient` | ⚠️ 见下文分析 |
| Tool name 统一为 web_search | ✅ 已完成 | 无 |

---

## 2. MultiServerMCPClient 需求分析

### 2.1 原始设计意图

`langchain-mcp-adapters` 提供两种客户端:
- **`load_mcp_tools`** — 单 MCP server 连接（当前使用）
- **`MultiServerMCPClient`** — 同时连接多个 MCP server

原始计划中使用 `MultiServerMCPClient` 的意图是为了支持**多个独立的 MCP 工具源**。

### 2.2 当前架构是否已满足？

**结论: 当前单 server 架构已满足 Phase 2 需求，不需要迁移到 MultiServerMCPClient。**

理由:

```
当前架构:
Agent → MCPSearchClient (单连接) → SearchMCPServer (6 tools)

MultiServerMCPClient 适用场景:
Agent → MultiServerMCPClient
         ├─ Server A: search-service (6 tools)
         ├─ Server B: code-executor (3 tools)
         └─ Server C: database-query (2 tools)
```

我们当前所有搜索工具（web_search, zhihu_search, weibo_search, weixin_search, github_search, scrape_url）都运行在**同一个 SearchMCPServer** 进程内。这是正确的设计，因为:

1. **共享资源**: 所有搜索工具共享 `BrowserPool`、`SearchRouter`、`SearXNGClient`
2. **统一生命周期**: 单进程便于管理 startup/shutdown
3. **低开销**: 一个 subprocess 比多个 subprocess 更高效

### 2.3 何时需要 MultiServerMCPClient？

如果未来引入**非搜索类的 MCP 工具源**（如代码执行器、数据库查询、文件系统工具），则需要 `MultiServerMCPClient`:

```python
# 未来可能的用法
async with MultiServerMCPClient({
    "search": {
        "command": sys.executable,
        "args": ["-m", "search_service"],
        "transport": "stdio",
    },
    "code_executor": {
        "command": sys.executable,
        "args": ["-m", "code_executor"],
        "transport": "stdio",
    },
}) as client:
    search_tools = client.get_tools("search")
    code_tools = client.get_tools("code_executor")
```

**但这是 Phase 3+ 或更远期的需求，当前不需要。**

---

## 3. 当前实现的改进点

虽然 MCP 集成已基本完成，但以下改进应在 Phase 2 一并完成:

### 3.1 Tool name 统一 ✅ 已完成

`internet_search` → `web_search` 已在 prompt 和 server 中统一。无需额外工作。

### 3.2 Cookie 过期告警传递

当 Phase 2 平台 scraper 上线后，`CookieExpiredError` 需要通过 MCP 工具的返回值传递给 Agent:

```python
# server.py — zhihu_search tool
async def zhihu_search(query: str, max_results: int = 10) -> dict:
    try:
        ...
    except CookieExpiredError:
        # 不只是 fallback，还要在返回值中标记降级状态
        response = await self._router.search(f"site:zhihu.com {query}", ...)
        result = response.model_dump()
        result["_degraded"] = True
        result["_degraded_reason"] = "cookie_expired"
        return result
```

### 3.3 MCP Server 启动参数传递

当前 `MCPSearchClient` 硬编码了 server 启动命令。Phase 2 的 scraper 可能需要额外的环境变量（如 `SEARCH_COOKIE_STORAGE_DIR`）:

```python
class MCPSearchClient:
    def __init__(
        self,
        *,
        server_command: Optional[str] = None,
        server_args: Optional[list[str]] = None,
        env: Optional[dict[str, str]] = None,  # 新增: 传递环境变量
    ) -> None:
        self._server_params = StdioServerParameters(
            command=server_command or sys.executable,
            args=server_args or ["-m", "search_service"],
            env=env,  # 传递给 subprocess
        )
```

### 3.4 MCP 重连机制

当前实现中，MCP subprocess 如果意外退出（OOM、crash），整个研究会话会失败。需要增加:

```python
async def _reconnect(self) -> None:
    """尝试重新建立 MCP 连接。

    用于 subprocess 意外退出后的恢复。
    最多重试 3 次，每次等待 2 秒。
    """
    for attempt in range(3):
        try:
            await self.__aexit__(None, None, None)
            await self.__aenter__()
            self._logger.info(
                "mcp_reconnected", extra={"attempt": attempt + 1},
            )
            return
        except Exception as exc:
            self._logger.warning(
                "mcp_reconnect_failed",
                extra={"attempt": attempt + 1, "error": str(exc)},
            )
            await asyncio.sleep(2)

    raise RuntimeError("MCP server reconnection failed after 3 attempts")
```

> [!WARNING]
> 重连后 MCP 工具引用会失效（旧的 `BaseTool` 对象绑定了已关闭的 session）。
> 重连时需要重新调用 `load_mcp_tools()` 并更新 Agent 的工具引用。
> 但在 LangGraph 中动态替换已编译 graph 的工具是复杂的操作。
> 
> **实际方案**: 不在 session 中间重连，而是让 `stream_deep_research` 捕获 MCP 错误，
> 生成 partial report 后退出。用户可以通过 multi-turn 重新发起。

---

## 4. 结论

### Phase 2 MCP 相关工作项

| 项目 | 状态 | 需要的工作 |
|:---|:---:|:---|
| `langchain-mcp-adapters` 集成 | ✅ 已完成 | 无 |
| `MultiServerMCPClient` | ⏭ 不需要 | 当前单 server 架构正确 |
| Tool name 统一 | ✅ 已完成 | 无 |
| Cookie 告警传递 | 🆕 新增 | 在 MCP tool 返回值中添加 `_degraded` 标记 |
| 环境变量传递 | 🆕 新增 | `MCPSearchClient.__init__` 添加 `env` 参数 |
| MCP 错误处理 | ⚠️ 增强 | `stream_deep_research` 中 catch MCP subprocess 错误 |

### Phase 3+ 可能的 MCP 演进

| 场景 | 时间 | 方案 |
|:---|:---|:---|
| 引入非搜索 MCP Server | Phase 3+ | 迁移到 MultiServerMCPClient |
| MCP SSE transport | Phase 3+ | 支持远程 MCP server (非 stdio) |
| MCP server 池化 | Phase 3+ | 多个 SearchMCPServer 进程负载均衡 |

---

## 5. 行业价值分析与主流实现方式

### 5.1 功能实际价值评估

| 维度 | 评估 | 说明 |
|:---|:---|:---|
| **架构前瞻性** | ⭐⭐⭐⭐⭐ | MCP 是 2025-2026 年 AI Agent 工具集成的行业标准协议 |
| **扩展性** | ⭐⭐⭐⭐ | 单 Server 当前够用，MultiServer 为未来非搜索工具预留扩展 |
| **生态兼容** | ⭐⭐⭐⭐ | 与 LangChain/LangGraph 生态深度集成 |

### 5.2 业界主流实现方式

**MCP 在 2025-2026 年已成为连接 AI Agent 与外部工具的标准协议**。行业实践要点：

1. **单 Server vs Multi-Server**: 行业共识是工具按"共享资源边界"分组——共享 BrowserPool/Cache 的工具放在同一 Server。我们的单 Server 架构正确
2. **stdio vs SSE/HTTP**: stdio 适合本地进程，SSE/HTTP 适合远程部署。行业趋势是 Phase 1 用 stdio，生产环境迁移到 HTTP transport
3. **MCP 可观测性挑战**: 标准 MCP 网关隐藏了单个工具的性能指标。行业最佳实践是在 Server 内部实现自定义中间件拦截工具调用并记录 per-tool metrics
4. **错误传播**: MCP 的 JSON-RPC 错误通道需要在 tool 返回值中嵌入降级状态标记（如 `_degraded`），这与我们的设计一致

> [!TIP]
> 当前单 Server + stdio 架构与行业 MVP 阶段实践完全一致。`_degraded` 标记传递降级状态是行业推荐做法。

---

## 6. 待确认事项

> [!NOTE]

1. **降级状态传递**: 是否需要在 MCP tool 返回值中添加 `_degraded` 标记，让 Agent 知道某个平台搜索降级了？还是透明降级即可（Agent 不需要知道）？

2. **MCP subprocess 崩溃恢复**: 如果 MCP server 进程 OOM，当前会话直接失败。是否需要实现自动重启？我的建议是 Phase 2 先接受 fail-fast 策略，Phase 3 再增强。

3. **StdioServerParameters.env**: `langchain-mcp-adapters` 的 `StdioServerParameters` 是否支持传递 `env` 参数？需要确认 API 兼容性。如果不支持，可以通过在 `MCPSearchClient` 外部设置 `os.environ` 来传递。
