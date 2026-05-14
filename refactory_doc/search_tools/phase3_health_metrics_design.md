# Phase 3: Health Endpoint + Prometheus Metrics 设计

> **模块**: `search_service/observability.py` (新建)  
> **依赖**: `prometheus_client`, `aiohttp`  
> **日期**: 2026-05-05

---

## 1. 设计目标

为 MCP Search Service 添加可观测性:
- `/health` — 服务健康状态 (供 Docker / K8s 探针使用)
- `/metrics` — Prometheus 指标 (供 Grafana 仪表盘使用)
- 核心指标: 搜索延迟、错误率、缓存命中率、浏览器池使用率

---

## 2. 挑战: MCP stdio + HTTP 共存

当前 `SearchMCPServer` 使用 `stdio` transport（stdin/stdout），无法直接暴露 HTTP 端点。

**方案**: 在 MCP server 进程内启动一个轻量 HTTP 旁路服务器（aiohttp），与 MCP stdio 并行运行。

```
┌─ SearchMCPServer Process ────────────────────────┐
│                                                   │
│  Task 1: FastMCP.run_async(transport="stdio")     │
│           ← Agent 通过 stdin/stdout 通信           │
│                                                   │
│  Task 2: ObservabilityServer (aiohttp)            │
│           ← HTTP :9090/health, :9090/metrics      │
│                                                   │
│  (asyncio.gather 并行运行两个 task)                │
└───────────────────────────────────────────────────┘
```

---

## 3. 类设计

### 3.1 MetricsCollector

```python
class MetricsCollector:
    """Prometheus metrics for the search service.

    Collects:
    - Search request latency per backend/tool
    - Request count per backend/status
    - Error breakdown by type
    - Cache hit/miss ratio
    - Browser pool utilization

    Usage:
        metrics = MetricsCollector()
        with metrics.track_search("web_search", "searxng"):
            result = await searxng.search(query)
        metrics.record_cache_hit("web_search")
    """

    def __init__(self) -> None:
        from prometheus_client import Counter, Histogram, Gauge

        self.search_duration = Histogram(
            "search_request_duration_seconds",
            "Search request latency",
            labelnames=["tool", "backend"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        )

        self.search_total = Counter(
            "search_requests_total",
            "Total search requests",
            labelnames=["tool", "backend", "status"],
        )

        self.search_errors = Counter(
            "search_errors_total",
            "Search errors by type",
            labelnames=["tool", "error_type"],
        )

        self.cache_operations = Counter(
            "search_cache_operations_total",
            "Cache operations",
            labelnames=["operation"],  # hit, miss, set, delete
        )

        self.browser_pool_active = Gauge(
            "search_browser_pool_active",
            "Active browser contexts",
        )

        self.browser_pool_queue = Gauge(
            "search_browser_pool_waiting",
            "Requests waiting for browser context",
        )

        self.scrape_content_length = Histogram(
            "search_scrape_content_bytes",
            "Scraped content size in bytes",
            labelnames=["domain"],
            buckets=[100, 1000, 5000, 10000, 50000],
        )

        self.cookie_ttl_hours = Gauge(
            "search_cookie_ttl_hours",
            "Hours until platform cookie expires",
            labelnames=["platform"],
        )

    @contextmanager
    def track_search(self, tool: str, backend: str):
        """Context manager to track search latency and status."""
        start = time.monotonic()
        try:
            yield
            elapsed = time.monotonic() - start
            self.search_duration.labels(tool=tool, backend=backend).observe(elapsed)
            self.search_total.labels(tool=tool, backend=backend, status="success").inc()
        except Exception as exc:
            elapsed = time.monotonic() - start
            self.search_duration.labels(tool=tool, backend=backend).observe(elapsed)
            self.search_total.labels(tool=tool, backend=backend, status="error").inc()
            error_type = type(exc).__name__
            self.search_errors.labels(tool=tool, error_type=error_type).inc()
            raise

    def record_cache_hit(self, tool: str) -> None:
        self.cache_operations.labels(operation="hit").inc()

    def record_cache_miss(self, tool: str) -> None:
        self.cache_operations.labels(operation="miss").inc()
```

### 3.2 ObservabilityServer

```python
class ObservabilityServer:
    """Lightweight HTTP server for /health and /metrics endpoints.

    Runs alongside the MCP stdio server on a separate port.

    Args:
        port: HTTP port (default 9090).
        health_checkers: Dict of component name → async health check callable.
    """

    def __init__(
        self,
        port: int = 9090,
        health_checkers: Optional[dict[str, Callable]] = None,
    ) -> None:
        self._port = port
        self._health_checkers = health_checkers or {}

    async def start(self) -> None:
        """Start the HTTP server."""
        from aiohttp import web
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

        app = web.Application()

        async def health_handler(request: web.Request) -> web.Response:
            results = {}
            overall_healthy = True
            for name, checker in self._health_checkers.items():
                try:
                    healthy = await asyncio.wait_for(checker(), timeout=5.0)
                    results[name] = "ok" if healthy else "degraded"
                    if not healthy:
                        overall_healthy = False
                except Exception:
                    results[name] = "error"
                    overall_healthy = False

            status = 200 if overall_healthy else 503
            return web.json_response(
                {"status": "healthy" if overall_healthy else "degraded", "components": results},
                status=status,
            )

        async def metrics_handler(request: web.Request) -> web.Response:
            metrics = generate_latest()
            return web.Response(body=metrics, content_type=CONTENT_TYPE_LATEST)

        app.router.add_get("/health", health_handler)
        app.router.add_get("/metrics", metrics_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self._port)
        await site.start()
```

---

## 4. SearchMCPServer 集成

```python
class SearchMCPServer:
    async def _run_async(self) -> None:
        await self.startup()

        # 初始化可观测性
        obs_server = ObservabilityServer(
            port=9090,
            health_checkers={
                "searxng": self._router._backends["searxng"].health_check,
                "browser_pool": self._browser_pool_health,
                "redis": self._cache_health,
            },
        )

        try:
            await asyncio.gather(
                self._mcp.run_async(transport="stdio"),
                obs_server.start(),
            )
        finally:
            await self.shutdown()
```

---

## 5. 推荐 Grafana Dashboard Panels

| Panel | 类型 | 指标 |
|:---|:---|:---|
| Search Success Rate | Stat | `search_requests_total{status="success"}` / total |
| Latency P50/P95/P99 | Heatmap | `search_request_duration_seconds` |
| Error Breakdown | Bar | `search_errors_total` by `error_type` |
| Cache Hit Ratio | Time Series | hit / (hit + miss) |
| Browser Pool | Gauge | `search_browser_pool_active` |
| Cookie TTL | Table | `search_cookie_ttl_hours` by platform |

---

## 6. 配置

```python
# SearchServiceConfig 新增
observability_port: int = 9090
observability_enabled: bool = True
```

---

## 7. 行业价值分析与主流实现方式

### 7.1 功能实际价值评估

| 维度 | 评估 | 说明 |
|:---|:---|:---|
| **生产必要性** | ⭐⭐⭐⭐⭐ | 无可观测性的搜索服务在生产环境中是"黑盒"，故障排查依赖猜测 |
| **运维价值** | ⭐⭐⭐⭐⭐ | P50/P95 延迟、错误率、缓存命中率是容量规划和 SLA 的基础 |
| **投入产出比** | ⭐⭐⭐⭐ | Prometheus + Grafana 是成熟生态，实现成本低 |

### 7.2 业界主流实现方式

**Prometheus + Grafana 在 2025-2026 年仍是搜索服务可观测性的行业标准**。

针对 MCP 服务的行业特殊挑战和最佳实践：

1. **MCP 可观测性难题**: 标准 MCP 网关通过单一端点路由所有工具调用，标准 API 监控只看到聚合流量。行业方案是**在 Server 内部实现自定义中间件**拦截工具调用，记录 per-tool 指标
2. **结构化健康端点**: 超越简单 up/down 检查，返回详细 JSON（各依赖组件状态）。区分 Liveness Probe（决定是否重启）和 Readiness Probe（决定是否接收流量）
3. **stdio + HTTP 并行**: 我们的"MCP stdio + aiohttp 旁路"方案是 MCP 生态的标准做法——stdio 不支持 HTTP，需要旁路 server
4. **OpenTelemetry 融合**: 2025 年趋势是将 Prometheus metrics 与 OpenTelemetry tracing 统一，实现从基础设施到 AI 工具调用的端到端追踪

> [!TIP]
> 我们的设计（stdio + aiohttp 旁路、per-tool 指标、组件级健康检查）与行业最佳实践完全一致。建议 Phase 3+ 考虑 OpenTelemetry 集成以实现跨服务追踪。

---

## 8. 待确认事项

1. **端口选择**: 9090 是否合适？是否与其他服务冲突？
2. **aiohttp vs uvicorn**: aiohttp 更轻量，uvicorn+prometheus_client ASGI 更标准。你偏好哪个？
3. **Docker healthcheck**: 是否将 Docker healthcheck 从 SearXNG `/healthz` 改为指向 SearchMCPServer 的 `/health`？
4. **Grafana 部署**: 是否需要在 docker-compose 中增加 Grafana + Prometheus 容器？
