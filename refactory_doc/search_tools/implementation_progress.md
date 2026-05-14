# Search Tools — Implementation Progress

> **Last Updated**: 2026-05-04
> **Session**: Phase 1 Complete — All Tasks Done
> **Total Tests**: 213 (all passing)

---

## Execution Summary

Phase 1 实现全部完成（10/10 任务），采用严格 TDD 流程。整个 `search_service` 包从零搭建，涵盖数据模型、配置、缓存、限流、搜索路由、SearXNG 客户端、浏览器池、页面抓取、GitHub 客户端、MCP Server 和 Agent 集成。Agent 已从 Tavily 迁移至自建 SearXNG 搜索后端，Worker 新增 `scrape_url` 深度提取工具和 `site:` 搜索策略指引。

---

## Task Completion Matrix

| Task | Spec | Module | Tests | Status |
|:-----|:-----|:-------|------:|:------:|
| 1 | S01 | models, config, exceptions, cache, rate_limiter | 41 | ✅ |
| 2 | S02 | SearchBackend Protocol + SearchRouter | 8 | ✅ |
| 3 | S02 | SearXNG async client | 13 | ✅ |
| 4 | S02 | Docker (docker-compose + settings.yml) | — | ✅ |
| 5 | S03 | BrowserPool + StealthInjector | 8 | ✅ |
| 6 | S04a/e | PageScraper + GitHubClient | 15 | ✅ |
| 7 | S04b/c/d | Zhihu, Weibo, Weixin Scrapers | — | ⏳ Phase 2 |
| 8 | S05 | MCP Server (6 tools) | 8 | ✅ |
| 9 | S06 | Agent Integration (Tavily → SearXNG) | 12 | ✅ |
| 10 | — | E2E Verification (full test suite) | — | ✅ |

---

## Task 9 Details: Agent Integration

### Code Changes

#### `agents/tools.py` — Complete rewrite (Tavily → SearXNG)

| Change | Detail |
|:---|:---|
| `make_internet_search(config)` | Synchronous factory → async `internet_search` inner function. Uses `SearXNGClient` + `SearchRouter` + `NullCache`. Output dict is Tavily-compatible: `{"query", "results": [{"title", "url", "content"}]}` |
| `make_scrape_url(config)` | New factory → async `scrape_url` inner function. Uses `BrowserPool` (lazy start) + `PageScraper`. Returns `ScrapeResponse.model_dump()` |
| `load_settings()` | Returns `search_config: SearchServiceConfig` instead of `tavily_api_key: str`. No longer depends on `TAVILY_API_KEY` env var |
| Tavily removal | `from tavily import TavilyClient` completely removed |

#### `agents/agent.py` — Wiring changes

| Change | Detail |
|:---|:---|
| Import | Added `make_scrape_url` import |
| `_make_scrape_url` alias | Backward-compat alias for test patching |
| `build_deep_agent()` | `_make_internet_search(cfg["search_config"])` replaces `_make_internet_search(cfg["tavily_api_key"])` |
| Worker tools | `[search_tool, scrape_tool]` instead of `[search_tool]` |

#### `agents/prompts.py` — Worker prompt enhancement

Added `## Search Optimization Techniques` section to `WORKER` prompt:
- `site:` operator guidance for targeted domain search (arxiv, CSDN, 36kr, etc.)
- Multi-site OR operator for broader coverage
- `scrape_url` deep extraction guidance (selective, high-value URLs only)
- Language awareness (Chinese → Baidu/Sogou, English → Bing)

#### `config/settings.py` — Settings dataclass

| Change | Detail |
|:---|:---|
| New field | `searxng_base_url: str` |
| `load()` | Loads from `SEARCH_SEARXNG_BASE_URL` env or `settings.yaml` key `searxng_base_url` (default: `http://localhost:8080`) |
| Tavily fields | Retained for backward compatibility (Settings is also used by BenchmarkGrader), marked deprecated in YAML |

#### `config/settings.yaml` — Configuration

| Change | Detail |
|:---|:---|
| New key | `searxng_base_url: "http://localhost:8080"` |
| Tavily | Fields retained with `DEPRECATED` comment |

#### `requirements.txt` — Dependency changes

| Removed | Added |
|:---|:---|
| `tavily-python` | `pydantic-settings>=2.0` |
| | `playwright>=1.40.0` |
| | `markdownify>=0.11.0` |
| | `mcp[cli]>=1.20.0` |

#### Test Updates

| File | Change |
|:---|:---|
| `tests/test_search_service/test_agent_integration.py` | **[NEW]** 12 tests covering `make_internet_search`, `make_scrape_url`, `load_settings` |
| `tests/test_citation/test_agent_integration.py` | Updated mock patches: `tavily_api_key` → `search_config`, added `_make_scrape_url` patch |
| `tests/test_grader.py` | Added `searxng_base_url` to `_make_settings()` fixture |

---

## Files Created / Modified

### Search Service (Tasks 1-8, previously completed)

```
search_service/
├── __init__.py
├── __main__.py              # python -m search_service 入口
├── models.py                # SearchEngine, SearchResultItem, SearchResponse, ScrapeResponse
├── config.py                # SearchServiceConfig (pydantic-settings, SEARCH_ prefix)
├── exceptions.py            # SearchServiceError 异常层次结构 (7 types)
├── cache.py                 # CacheLayer Protocol, NullCache, generate_cache_key
├── rate_limiter.py          # AsyncRateLimiter (token-bucket)
├── server.py                # SearchMCPServer (FastMCP, 6 tools)
├── backends/
│   ├── __init__.py
│   ├── base.py              # SearchBackend Protocol + SearchRouter
│   ├── searxng_client.py    # SearXNG HTTP async client
│   ├── page_scraper.py      # URL → Markdown (Playwright)
│   └── github_client.py     # GitHub REST API client
├── browser/
│   ├── __init__.py
│   ├── pool.py              # BrowserPool (semaphore concurrency)
│   └── stealth.py           # StealthInjector (stealth.min.js loader)
└── resources/               # stealth.min.js placeholder
```

### Agent Integration (Task 9, this session)

```
agents/
├── tools.py                 # [MODIFIED] Tavily → SearXNG + scrape_url
├── agent.py                 # [MODIFIED] Wired new tool factories
└── prompts.py               # [MODIFIED] Added site: search strategy

config/
├── settings.py              # [MODIFIED] Added searxng_base_url field
└── settings.yaml            # [MODIFIED] Added SearXNG config, deprecated Tavily

requirements.txt             # [MODIFIED] tavily-python removed, search deps added
```

### Docker (Task 4, previously completed)

```
docker/
├── docker-compose.yml       # SearXNG container (port 8080)
└── searxng/
    └── settings.yml         # baidu(weight=2) + sogou + bing; GFW-blocked disabled
```

### Tests (all sessions combined)

```
tests/test_search_service/
├── __init__.py
├── conftest.py              # autouse fixture for StealthInjector reset
├── test_models.py           # 15 tests
├── test_config.py           # 4 tests
├── test_exceptions.py       # 13 tests
├── test_cache.py            # 9 tests
├── test_rate_limiter.py     # 3 tests (timing-tolerant)
├── test_search_router.py    # 8 tests
├── test_searxng_client.py   # 13 tests
├── test_browser_pool.py     # 8 tests
├── test_page_scraper.py     # 8 tests
├── test_github_client.py    # 7 tests
├── test_server.py           # 8 tests
└── test_agent_integration.py # 12 tests [NEW]

tests/test_citation/
└── test_agent_integration.py # 4 tests [MODIFIED]

tests/
└── test_grader.py            # 7 tests [MODIFIED]
```

---

## Dependencies

| Package | Version | Purpose |
|:--------|:--------|:--------|
| `pydantic-settings` | ≥2.0 | `SearchServiceConfig` env-based configuration |
| `playwright` | ≥1.40.0 | BrowserPool / PageScraper |
| `markdownify` | ≥0.11.0 | HTML → Markdown conversion |
| `mcp[cli]` | ≥1.20.0 | FastMCP server SDK |

> **Removed**: `tavily-python` — no longer used at runtime
> **Note**: `httpx` is present as a transitive dependency

---

## Key Design Decisions

### 1. Synchronous Factory, Async Inner Function
`make_internet_search()` and `make_scrape_url()` are synchronous factories that return async callables. This preserves `build_deep_agent()`'s synchronous signature — no change to the upstream call chain. The SearXNG client is lazily initialized on first HTTP call; the BrowserPool is lazily started on first `scrape_url` invocation.

### 2. Tavily-Compatible Output Format
The `internet_search` function returns `{"query": str, "results": [{"title", "url", "content"}]}` — the same structure as the former Tavily implementation. This ensures Worker prompts work without modification and enables a clean rollback if needed.

### 3. Tool Naming Strategy
Phase 1 uses function name `internet_search` (LangGraph ToolNode derives `tool.name` from `__name__`). Phase 2 MCP will register the tool as `web_search`. Worker prompt references both names.

### 4. Lazy BrowserPool Initialization
`scrape_url` lazily calls `BrowserPool.start()` on first invocation. This avoids spawning Chromium processes unless actually needed, reducing resource usage for queries that only need Discovery (SearXNG search).

### 5. V1 Platform Fallback Strategy
Zhihu/Weibo/Weixin searches in V1 use `site:` prefix + SearXNG — no Cookie, no Playwright scraper needed:
```
zhihu_search("AI agent") → router.search("site:zhihu.com AI agent")
weibo_search("热搜话题")  → router.search("site:weibo.com 热搜话题")
weixin_search("行业分析") → router.search("site:mp.weixin.qq.com 行业分析")
```

### 6. Config Isolation
- `SearchServiceConfig` (pydantic-settings, `SEARCH_` prefix) is independent from project-level `Settings` dataclass
- Connection point: `searxng_base_url` exists in both configs
- `extra="ignore"` prevents `.env` pollution

### 7. Tavily Graceful Deprecation
Tavily fields retained in `Settings` dataclass and `settings.yaml` (with `DEPRECATED` comment) because `BenchmarkGrader` and other test fixtures reference them. Full removal deferred to a follow-up cleanup PR.

---

## Tech Debt

| Item | Description | Priority |
|:-----|:------------|:--------:|
| AsyncRateLimiter | Simple token-bucket, should upgrade to sliding window + Redis-backed | Medium |
| stealth.min.js | Needs manual `npx extract-stealth-evasions` generation | Low |
| NullCache | V1 has no actual caching — high-frequency queries produce duplicate requests | Medium |
| Platform Scrapers | Zhihu/Weibo/Weixin use `site:` fallback; deep scraping deferred | High (Phase 2) |
| Tavily Cleanup | `tavily_api_key`, `tavily_search_url`, `tavily_max_result_chars` fields in Settings | Low |
| BrowserPool Shutdown | `make_scrape_url` creates a BrowserPool with no explicit shutdown hook at agent teardown | Medium |

---

## Remaining Work

### Phase 2: Vertical Platform Scrapers + MCP Protocol
- 实现 `backends/zhihu_scraper.py` (Playwright + Cookie 注入 + x-zse-96 绕过)
- 实现 `backends/weibo_scraper.py` (移动端 m.weibo.cn + XHR 解析)
- 实现 `backends/weixin_scraper.py` (搜狗微信搜索 + 验证码降级)
- 使用 `langchain-mcp-adapters` 的 `MultiServerMCPClient` 接入 MCP 协议
- Tool name 从 `internet_search` 统一为 `web_search`
- Cookie 过期检测 + 告警机制

### Phase 3: Production Hardening
- Redis 缓存层替换 NullCache
- `/health` 端点 + Prometheus metrics
- Browser Pool 进程回收 + 内存监控
- 搜索质量旁路监控 (LLM-as-Judge)
- Cookie 半自动刷新流程
- BrowserPool 生命周期与 Agent 进程绑定

### Cleanup
- 完全移除 `tavily-python` 残留（Settings 字段 + YAML 键）
- 移除 `require_tavily_api_key()` 方法
- 统一 `.env.example` 中的变量说明
