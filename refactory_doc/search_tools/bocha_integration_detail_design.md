# 博查 API 集成详细设计文档

> **创建日期**: 2026-05-13
> **前置文档**: [MCP 搜索服务设计 v2](mcp_search_service_design_v2.md) §12、[search_service 模块价值分析](bocha_integration_analysis.md)
> **设计目标**: 将 `bocha_integration_analysis.md` 中确定的**方案 B（博查作为 SearchRouter 一级 Backend）** 细化为可直接开发的技术规格

---

## 1. 非代码工作清单（API Key 及其他准备事项）

> [!IMPORTANT]
> 以下事项必须在代码开发**之前**完成，否则无法进行任何集成测试。

### 1.1 必须完成

| # | 事项 | 负责人 | 预计耗时 | 状态 |
|---|------|--------|----------|------|
| 1 | **注册博查开放平台账号** — 访问 [open.bochaai.com](https://open.bochaai.com)，微信扫码注册 | 项目负责人 | 5 分钟 | `[ ]` |
| 2 | **创建 API Key** — 登录后台 → "API Key 管理" → 新建 Key → 妥善保存 | 项目负责人 | 2 分钟 | `[ ]` |
| 3 | **账户充值** — 预付费模式，建议首次充值 ￥50（约 1,400 次调用）用于开发调试 | 项目负责人 | 5 分钟 | `[ ]` |
| 4 | **配置 `.env` 文件** — 添加 `SEARCH_BOCHA_API_KEY=sk-xxxx` | 开发人员 | 1 分钟 | `[ ]` |
| 5 | **验证 API 连通性** — `curl` 测试确认 Key 有效且网络可达（见 §1.3） | 开发人员 | 5 分钟 | `[ ]` |

### 1.2 建议完成

| # | 事项 | 说明 | 优先级 |
|---|------|------|--------|
| 6 | **加入博查开发者交流群** | 获取技术支持和 API 变更通知（官网有微信群二维码） | 中 |
| 7 | **阅读完整 API 文档** | [飞书文档](https://aq6ky2b8nql.feishu.cn/wiki/HmtOw1z6vik14Fkdu5uc9VaInBb) — 了解全部参数和错误码 | 高 |
| 8 | **了解计费规则** | Web Search ≈ ￥0.036/次；AI Search 费用更高；确认是否有新人免费额度 | 高 |
| 9 | **评估月度预算上限** | 根据 Agent 调用频率估算月成本（见 §7 成本模型），设定告警阈值 | 中 |
| 10 | **配置余额监控** | 博查提供 `GET /v1/fund/remaining` 接口，可接入监控或定期检查 | 低（Phase 2） |

### 1.3 API 连通性验证脚本

```bash
# 替换 YOUR_KEY 后执行
curl -s -X POST https://api.bochaai.com/v1/web-search \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "count": 1}' \
  | python -m json.tool

# 预期: HTTP 200，返回 {"code": 200, "msg": "success", "data": {"webPages": {...}}}
# 失败排查:
#   - 401: API Key 无效或未激活
#   - 403: 账户余额不足
#   - 429: 限流（QPS 超限）
#   - 连接超时: 检查网络/DNS
```

---

## 2. 博查 API 技术规格

### 2.1 接口总览

| 接口 | 端点 | 方法 | 我们使用？ | 说明 |
|------|------|------|:---:|------|
| **Web Search** | `/v1/web-search` | POST | ✅ 主力 | 全网网页搜索，返回 title/URL/snippet |
| **AI Search** | `/v1/ai-search` | POST | ❌ 暂不 | 费用更高，模态卡与我们场景不匹配 |
| **Reranker** | `/v1/rerank` | POST | ❌ 暂不 | 语义重排；我们已有 `SearchResultFilter` |
| **余额查询** | `/v1/fund/remaining` | GET | 🔜 Phase 2 | 余额监控告警 |

> [!TIP]
> **为什么只用 Web Search 不用 AI Search？** AI Search 返回的模态卡（天气/股票/百科）对 Deep Research Agent 无意义——Agent 需要的是网页 URL + 摘要文本作为研究素材，不是结构化卡片。且 AI Search 单次调用成本更高。

### 2.2 Web Search API 请求规格

```
POST https://api.bochaai.com/v1/web-search
Authorization: Bearer {BOCHA_API_KEY}
Content-Type: application/json
```

**请求体 (JSON)**:

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|:---:|--------|------|
| `query` | string | ✅ | — | 搜索关键词 |
| `count` | int | ❌ | 10 | 返回结果数，上限 50 |
| `freshness` | string | ❌ | `"noLimit"` | 时间范围：`noLimit`, `oneDay`, `oneWeek`, `oneMonth`, `oneYear` |
| `summary` | bool | ❌ | `false` | 是否返回 AI 生成的长文本摘要（增加延迟约 1-3s） |

### 2.3 Web Search API 响应规格

```json
{
  "code": 200,
  "msg": "success",
  "data": {
    "webPages": {
      "value": [
        {
          "name": "网页标题",
          "url": "https://example.com/article",
          "snippet": "简短的搜索摘要片段...",
          "summary": "AI 生成的长文本摘要（仅 summary=true 时）",
          "siteName": "来源网站名称",
          "siteIcon": "https://example.com/favicon.ico",
          "publicationTime": "2026-05-10T08:00:00Z"
        }
      ]
    },
    "images": {
      "value": []
    }
  }
}
```

**字段映射（Bocha → SearchResultItem）**:

| Bocha 字段 | SearchResultItem 字段 | 转换逻辑 |
|------------|----------------------|----------|
| `name` | `title` | 直接映射 |
| `url` | `url` | 直接映射 |
| `snippet` | `content` | 直接映射 |
| `summary` | `raw_content` | 仅当 `summary=true` 时有值 |
| `publicationTime` | `published_date` | 直接映射 |
| `siteName` | `metadata["site_name"]` | 放入 metadata |
| `siteIcon` | `metadata["site_icon"]` | 放入 metadata |
| — | `source_engine` | 硬编码 `SearchEngine.BOCHA` |
| — | `score` | `None`（博查不返回相关性分数） |

### 2.4 错误码与处理策略

| HTTP 状态码 | 含义 | 处理策略 |
|-------------|------|----------|
| 200 | 成功 | 解析并返回 |
| 401 | API Key 无效 | `raise SearchProviderError` → fallback SearXNG |
| 403 | 余额不足 / 权限不足 | `raise SearchProviderError` → fallback + logging.error 告警 |
| 429 | 限流（QPS 超限） | `raise SearchProviderError` → fallback SearXNG |
| 500/502/503 | 博查服务端异常 | `raise SearchProviderError` → fallback SearXNG |
| 网络超时 | 连接 / 读取超时 | `raise SearchProviderError` → fallback SearXNG |

> 所有错误均抛出 `SearchProviderError`，由 `SearchRouter` 的现有 fallback 逻辑自动降级到 SearXNG，**无需额外 fallback 代码**。

### 2.5 `freshness` 参数映射

我们的 `web_search` MCP 工具接受 `time_range` 参数（`day`, `week`, `month`, `year`），需映射到博查的 `freshness` 参数：

| MCP `time_range` | 博查 `freshness` |
|-------------------|-------------------|
| `""` (空) | `"noLimit"` |
| `"day"` | `"oneDay"` |
| `"week"` | `"oneWeek"` |
| `"month"` | `"oneMonth"` |
| `"year"` | `"oneYear"` |

---

## 3. 实现规格：`BochaClient`

### 3.1 类设计

```python
"""Bocha Web Search API async client.

Queries the Bocha Web Search API and converts results into the unified
:class:`SearchResponse` model. Follows the same pattern as SearXNGClient.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx

from search_service.config import SearchServiceConfig
from search_service.exceptions import SearchProviderError
from search_service.models import SearchEngine, SearchResponse, SearchResultItem


_FRESHNESS_MAP: dict[str, str] = {
    "": "noLimit",
    "day": "oneDay",
    "week": "oneWeek",
    "month": "oneMonth",
    "year": "oneYear",
}


class BochaClient:
    """Bocha Web Search API async client.

    Implements the SearchBackend protocol. Uses httpx for async HTTP
    with lazy-initialized connection pooling.

    Args:
        config: SearchServiceConfig instance with bocha_* fields.
    """

    API_BASE = "https://api.bochaai.com"

    def __init__(self, config: SearchServiceConfig) -> None:
        self._api_key: str = config.bocha_api_key  # type: ignore[assignment]
        self._base_url = (config.bocha_base_url or self.API_BASE).rstrip("/")
        self._timeout = config.bocha_timeout_seconds
        self._summary_enabled = config.bocha_summary_enabled
        self._client: Optional[httpx.AsyncClient] = None
        self._logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "bocha"

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Lazy-initialize httpx client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self._timeout),
                follow_redirects=True,
            )
        return self._client

    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> SearchResponse:
        """Execute a search query against Bocha Web Search API.

        Args:
            query: Search keywords.
            max_results: Maximum results (1-50).
            **kwargs: Supports 'time_range' (str).

        Returns:
            Unified SearchResponse.

        Raises:
            SearchProviderError: On any Bocha API failure.
        """
        start = time.monotonic()
        client = await self._ensure_client()

        payload: dict[str, Any] = {
            "query": query,
            "count": min(max_results, 50),
            "summary": self._summary_enabled,
        }

        time_range = kwargs.get("time_range", "")
        payload["freshness"] = _FRESHNESS_MAP.get(time_range, "noLimit")

        try:
            response = await client.post("/v1/web-search", json=payload)
            response.raise_for_status()
            data = response.json()
            elapsed_ms = int((time.monotonic() - start) * 1000)
            return self._parse_response(query, data, elapsed_ms)
        except httpx.HTTPStatusError as exc:
            self._logger.warning(
                "bocha_http_error",
                extra={"status": exc.response.status_code, "query": query},
            )
            raise SearchProviderError("bocha", str(exc), exc)
        except httpx.TimeoutException as exc:
            raise SearchProviderError("bocha", f"Timeout: {exc}", exc)
        except Exception as exc:
            raise SearchProviderError("bocha", str(exc), exc)

    def _parse_response(
        self, query: str, data: dict, elapsed_ms: int,
    ) -> SearchResponse:
        """Convert Bocha JSON to SearchResponse.

        Args:
            query: Original query.
            data: Raw JSON from Bocha API.
            elapsed_ms: Request latency.

        Returns:
            Parsed SearchResponse.
        """
        # Bocha wraps results in data.webPages.value
        code = data.get("code", 0)
        if code != 200:
            msg = data.get("msg", "Unknown Bocha error")
            raise SearchProviderError("bocha", f"API error {code}: {msg}")

        web_pages = data.get("data", {}).get("webPages", {})
        raw_results = web_pages.get("value", [])

        items: list[SearchResultItem] = []
        for raw in raw_results:
            items.append(
                SearchResultItem(
                    title=raw.get("name", ""),
                    url=raw.get("url", ""),
                    content=raw.get("snippet", ""),
                    source_engine=SearchEngine.BOCHA,
                    published_date=raw.get("publicationTime"),
                    score=None,  # Bocha does not return relevance scores
                    raw_content=raw.get("summary"),  # AI summary if enabled
                    metadata={
                        k: v
                        for k, v in {
                            "site_name": raw.get("siteName"),
                            "site_icon": raw.get("siteIcon"),
                        }.items()
                        if v is not None
                    },
                )
            )

        return SearchResponse(
            query=query,
            results=items,
            result_count=len(items),
            search_time_ms=elapsed_ms,
            engines_used=["bocha"],
        )

    async def health_check(self) -> bool:
        """Check Bocha API availability with a minimal query.

        Returns:
            True if Bocha responds with HTTP 200.
        """
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

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
```

### 3.2 与 SearXNGClient 的设计一致性

| 维度 | SearXNGClient | BochaClient |
|------|---------------|-------------|
| 初始化 | `config` 注入 | `config` 注入 |
| HTTP 客户端 | `httpx.AsyncClient` 懒初始化 | 同 |
| `name` 属性 | `"searxng"` | `"bocha"` |
| `search()` 签名 | `(query, max_results, **kwargs)` | 同 |
| 错误处理 | `raise SearchProviderError` | 同 |
| `health_check()` | 调用 `/healthz` | 调用 `/v1/web-search` (count=1) |
| `close()` | 关闭 httpx client | 同 |

---

## 4. 改动清单

### 4.1 新增文件

| 文件 | 说明 | 预估行数 |
|------|------|----------|
| `search_service/backends/bocha_client.py` | BochaClient 实现 | ~140 行 |
| `tests/test_search_service/test_bocha_client.py` | 单元测试 | ~120 行 |

### 4.2 修改文件

| 文件 | 改动内容 | 预估改动 |
|------|----------|----------|
| `search_service/config.py` | 新增 4 个 `bocha_*` 配置字段 | +12 行 |
| `search_service/models.py` | `SearchEngine` 枚举新增 `BOCHA = "bocha"` | +1 行 |
| `search_service/server.py` | `startup()` 中条件性注册 BochaClient 为首位 Backend | +7 行 |
| `.env.example` | 新增 `SEARCH_BOCHA_API_KEY` 配置示例 | +3 行 |

### 4.3 不改动

| 范围 | 说明 |
|------|------|
| Agent 层 (`agents/`) | `MCPSearchClient` 动态发现的工具集不变 |
| `SearchRouter` (`backends/base.py`) | 现有 fallback 逻辑完全兼容，零改动 |
| `SearchResultFilter` (`backends/result_filter.py`) | 博查结果也走去重 + 噪声过滤 |
| `CacheLayer` (`cache.py`) | Cache-aside 逻辑不变，自动缓存博查结果 |

---

## 5. 配置变更详细规格

### 5.1 `config.py` 新增字段

```python
class SearchServiceConfig(BaseSettings):
    # ... 现有字段 ...

    # Bocha API
    bocha_api_key: Optional[str] = Field(
        default=None,
        description="Bocha Search API key. When set, BochaClient is "
                    "registered as the primary search backend.",
    )
    bocha_base_url: str = Field(
        default="https://api.bochaai.com",
        description="Bocha API base URL.",
    )
    bocha_timeout_seconds: float = Field(
        default=10.0,
        description="Bocha API request timeout in seconds.",
    )
    bocha_summary_enabled: bool = Field(
        default=False,
        description="Request AI-generated summaries from Bocha. "
                    "Increases latency by 1-3s per call.",
    )
```

> **环境变量映射**（`env_prefix="SEARCH_"` 生效）:
> - `SEARCH_BOCHA_API_KEY` → `bocha_api_key`
> - `SEARCH_BOCHA_BASE_URL` → `bocha_base_url`
> - `SEARCH_BOCHA_TIMEOUT_SECONDS` → `bocha_timeout_seconds`
> - `SEARCH_BOCHA_SUMMARY_ENABLED` → `bocha_summary_enabled`

### 5.2 `server.py` startup() 修改

```python
async def startup(self) -> None:
    """Initialize all backends."""
    cache = NullCache()
    backends: list[SearchBackend] = []

    # Bocha as primary backend (when API key is configured)
    if self._config.bocha_api_key:
        from search_service.backends.bocha_client import BochaClient
        backends.append(BochaClient(self._config))
        self._logger.info("bocha_backend_enabled")

    # SearXNG as fallback (always enabled)
    backends.append(SearXNGClient(self._config))

    self._router = SearchRouter(backends=backends, cache=cache)

    self._browser_pool = BrowserPool(self._config)
    await self._browser_pool.start()
    self._logger.info("search_service_started")
```

### 5.3 `models.py` 枚举变更

```python
class SearchEngine(str, Enum):
    # ... 现有成员 ...
    BOCHA = "bocha"  # 新增
```

---

## 6. 测试策略

### 6.1 单元测试 (`test_bocha_client.py`)

| 测试用例 | 验证内容 |
|----------|----------|
| `test_search_success` | Mock 200 响应 → 正确解析为 `SearchResponse` |
| `test_parse_response_mapping` | 验证 Bocha 字段 → `SearchResultItem` 的映射正确性 |
| `test_freshness_mapping` | `time_range="week"` → Bocha `freshness="oneWeek"` |
| `test_http_401_raises_provider_error` | 401 → `SearchProviderError("bocha", ...)` |
| `test_http_429_raises_provider_error` | 429 → `SearchProviderError` |
| `test_timeout_raises_provider_error` | 超时 → `SearchProviderError` |
| `test_api_code_non_200` | Bocha 返回 `{"code": 500, ...}` → `SearchProviderError` |
| `test_empty_results` | 空 `webPages.value` → `SearchResponse(result_count=0)` |
| `test_summary_field` | `summary=true` 时 `raw_content` 有值 |
| `test_health_check_success` | Mock 200 → `True` |
| `test_health_check_failure` | Mock 异常 → `False` |

### 6.2 集成测试

| 测试用例 | 验证内容 |
|----------|----------|
| `test_fallback_bocha_to_searxng` | Mock BochaClient 抛异常 → 自动 fallback 到 SearXNG |
| `test_cache_prevents_duplicate_calls` | 相同 query 二次调用 → 走缓存，不调用博查 |
| `test_no_api_key_uses_searxng_only` | 不配置 `BOCHA_API_KEY` → 仅 SearXNG，行为不变 |

### 6.3 端到端测试

```bash
# 配置 SEARCH_BOCHA_API_KEY 后执行
python examples/run_deep_agent.py "2026年AI Agent框架对比分析"
python examples/run_deep_agent.py "What is retrieval augmented generation?"
python examples/run_deep_agent.py "LangGraph vs CrewAI vs AutoGen"
```

**验收标准**: 每个 query 返回 ≥5 条结果，延迟 <3s。

---

## 7. 成本模型与监控

### 7.1 成本估算

| 场景 | 日查询量 | 月查询量 | 月成本 (￥) | 说明 |
|------|----------|----------|------------|------|
| 开发调试 | ~30 | ~900 | ￥32 | 初始充值 ￥50 可用约 6 周 |
| 轻度使用 | ~50 | ~1,500 | ￥54 | 个人研究 |
| 中度使用 | ~200 | ~6,000 | ￥216 | 小团队日常 |
| 重度使用 | ~500 | ~15,000 | ￥540 | 产品化高频调用 |

> 缓存命中率预估 20-30%（技术研究 query 重复率不高），实际成本可下浮 20-30%。

### 7.2 成本控制措施

| 措施 | 实现层 | 效果 |
|------|--------|------|
| Cache-aside（已有） | `SearchRouter` | 相同 query 不重复计费 |
| SearXNG fallback（已有） | `SearchRouter` | 博查余额耗尽时自动免费降级 |
| 余额监控（Phase 2） | 定时任务 | 调用 `/v1/fund/remaining`，低于阈值告警 |
| `summary=false`（默认） | `BochaClient` | 不启用 AI 摘要，降低单次成本 |

---

## 8. 风险评估（博查特有）

| 风险 | 概率 | 影响 | 缓解措施 |
|------|:---:|:---:|----------|
| API Key 泄露 | 低 | 高 | `.env` 不入 Git；部署时用环境变量注入 |
| 账户余额耗尽 | 中 | 中 | SearXNG 自动 fallback；Phase 2 余额告警 |
| 博查 API 大规模故障 | 低 | 中 | SearXNG fallback 覆盖（已有） |
| API 接口变更 / Breaking Change | 低 | 中 | 加入开发者交流群获取提前通知 |
| 博查公司经营风险 | 低 | 高 | SearXNG 始终可用；模块化设计便于替换 |
| 定价上调 | 中 | 低 | 监控月成本；必要时降低博查调用比例 |

---

## 9. 实施时间线

```
Day 0 (准备):
  [x] 注册博查平台 + 获取 API Key + 充值
  [x] 验证 API 连通性 (curl)
  [x] 配置 .env

Day 1 (开发 + 测试):
  [ ] models.py: 新增 BOCHA 枚举
  [ ] config.py: 新增 bocha_* 配置字段
  [ ] backends/bocha_client.py: 完整实现
  [ ] server.py: startup() 注册 BochaClient
  [ ] 单元测试: test_bocha_client.py
  [ ] 集成测试: fallback + cache

Day 2 (验证 + 收尾):
  [ ] 端到端测试: run_deep_agent.py 3 个 query
  [ ] 搜索质量对比: Bocha vs SearXNG 同 query 结果
  [ ] 更新 .env.example
  [ ] 更新 implementation_progress.md
```

**预估总工时**: 1-1.5 天（不含 Day 0 准备工作）

---

## 相关文档索引

| 文档 | 说明 |
|------|------|
| [mcp_search_service_design_v2.md §12](mcp_search_service_design_v2.md) | 博查集成在整体架构中的定位 |
| [bocha_integration_analysis.md](bocha_integration_analysis.md) | 三种集成架构对比，确定方案 B |
| [search_alternatives_analysis.md](search_alternatives_analysis.md) | 搜索备选方案全景对比 |
| [博查 API 开发文档 (飞书)](https://aq6ky2b8nql.feishu.cn/wiki/HmtOw1z6vik14Fkdu5uc9VaInBb) | 官方接口文档（参数/响应/错误码完整说明） |
| [博查开放平台](https://open.bochaai.com) | 注册、API Key 管理、余额充值 |
