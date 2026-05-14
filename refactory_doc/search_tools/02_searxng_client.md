# S02: SearXNG Client + Docker 部署

> **Phase**: 1 | **预估工时**: 1-2 天  
> **产出文件**: `search_service/backends/searxng_client.py`, `search_service/backends/base.py`, `docker/`  
> **依赖**: S01 (models, config, exceptions)  
> **下游**: S05 (MCP Server), S06 (Agent Integration)

---

## 1. 目标

- 实现 `SearchBackend` Protocol 和 `SearchRouter` 策略路由层
- 实现 SearXNG HTTP 异步客户端（httpx + 指数退避重试）
- Docker Compose 部署 SearXNG（百度 + 搜狗 + 360 + Bing cn）
- 验证英文搜索通道（cn.bing.com）可用性

---

## 2. SearchBackend Protocol (`backends/base.py`)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class SearchBackend(Protocol):
    """搜索后端统一接口。所有后端（SearXNG、Scrapers）实现此协议。"""

    @property
    def name(self) -> str:
        """后端标识名，用于日志和路由。"""
        ...

    async def search(self, query: str, max_results: int = 10, **kwargs) -> SearchResponse:
        """执行搜索。"""
        ...

    async def health_check(self) -> bool:
        """健康检查。返回 True 表示后端可用。"""
        ...
```

## 3. SearchRouter (`backends/base.py`)

```python
class SearchRouter:
    """策略路由层：按 fallback 顺序调用后端。

    V1: 只有 SearXNG 一个后端，直接调用。
    V2+: 支持 fallback chain + 熔断 + 缓存。

    Args:
        backends: 搜索后端列表，按优先级排列。
        cache: 缓存层实例 (V1 为 NullCache)。
    """

    def __init__(self, backends: list[SearchBackend], cache: CacheLayer):
        self._backends = {b.name: b for b in backends}
        self._fallback_order = [b.name for b in backends]
        self._cache = cache
        self._logger = logging.getLogger(__name__)

    async def search(self, query: str, **kwargs) -> SearchResponse:
        """执行搜索，支持缓存和 fallback。"""
        cache_key = generate_cache_key("web_search", query, **kwargs)

        # Cache-Aside: check cache first
        cached = await self._cache.get(cache_key)
        if cached is not None:
            self._logger.info("search_cache_hit", extra={"query": query})
            return cached

        # Try backends in fallback order
        errors: dict[str, str] = {}
        for name in self._fallback_order:
            try:
                start = time.monotonic()
                result = await self._backends[name].search(query, **kwargs)
                elapsed_ms = int((time.monotonic() - start) * 1000)

                self._logger.info("search_request", extra={
                    "tool": "web_search", "query": query,
                    "latency_ms": elapsed_ms, "status": "success",
                    "result_count": result.result_count, "backend": name,
                })

                # Store in cache
                ttl = CACHE_TTL.get("web_search", 3600)
                await self._cache.set(cache_key, result, ttl_seconds=ttl)
                return result

            except SearchProviderError as e:
                errors[name] = str(e)
                self._logger.warning("search_backend_failed", extra={
                    "backend": name, "error": str(e),
                })

        raise AllProvidersExhaustedError(errors)
```

---

## 4. SearXNG Client (`backends/searxng_client.py`)

```python
class SearXNGClient:
    """SearXNG HTTP 异步客户端。

    通过 SearXNG JSON API 进行搜索，将结果转换为 SearchResponse。

    Args:
        config: SearchServiceConfig 实例。
    """

    def __init__(self, config: SearchServiceConfig):
        self._base_url = config.searxng_base_url.rstrip("/")
        self._timeout = config.searxng_timeout_seconds
        self._max_retries = config.searxng_max_retries
        self._client: httpx.AsyncClient | None = None
        self._logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "searxng"

    async def _ensure_client(self) -> httpx.AsyncClient:
        """懒初始化 httpx 客户端（连接池复用）。"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(self._timeout),
                follow_redirects=True,
            )
        return self._client

    async def search(
        self,
        query: str,
        max_results: int = 10,
        engines: list[str] | None = None,
        time_range: str = "",
        language: str = "auto",
    ) -> SearchResponse:
        """执行搜索。

        Args:
            query: 搜索关键词。
            max_results: 最大返回结果数 (1-20)。
            engines: 引擎列表，None 时使用 SearXNG 默认配置。
            time_range: 时间范围 ("day", "week", "month", "year", "")。
            language: 搜索语言 ("auto", "zh-CN", "en-US")。

        Returns:
            SearchResponse

        Raises:
            SearchProviderError: SearXNG 请求失败。
        """
        params = {
            "q": query,
            "format": "json",
            "pageno": 1,
        }
        if engines:
            params["engines"] = ",".join(engines)
        if time_range:
            params["time_range"] = time_range
        if language != "auto":
            params["language"] = language

        # 指数退避重试
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                client = await self._ensure_client()
                response = await client.get("/search", params=params)
                response.raise_for_status()
                data = response.json()
                return self._parse_response(query, data, max_results)
            except (httpx.HTTPStatusError, httpx.RequestError, Exception) as e:
                last_error = e
                wait = min(2 ** attempt, 10)  # 1s, 2s, 4s, max 10s
                self._logger.warning(
                    "searxng_retry", extra={
                        "attempt": attempt + 1, "wait_seconds": wait,
                        "error": str(e),
                    }
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(wait)

        raise SearchProviderError("searxng", f"Failed after {self._max_retries} retries", last_error)

    def _parse_response(self, query: str, data: dict, max_results: int) -> SearchResponse:
        """将 SearXNG JSON 转换为 SearchResponse。"""
        raw_results = data.get("results", [])[:max_results]
        items = []
        for r in raw_results:
            engine = r.get("engine", "searxng")
            try:
                source = SearchEngine(engine)
            except ValueError:
                source = SearchEngine.SEARXNG

            items.append(SearchResultItem(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
                source_engine=source,
                published_date=r.get("publishedDate"),
                score=self._normalize_score(r.get("score")),
            ))

        engines_used = list({r.get("engine", "searxng") for r in raw_results})
        search_time = int(data.get("search_time", 0) * 1000)  # SearXNG 返回秒

        return SearchResponse(
            query=query,
            results=items,
            result_count=len(items),
            search_time_ms=search_time,
            engines_used=engines_used,
        )

    @staticmethod
    def _normalize_score(raw_score: float | None) -> float | None:
        """将 SearXNG 的 score 归一化到 [0, 1]。"""
        if raw_score is None:
            return None
        return max(0.0, min(1.0, raw_score))

    async def health_check(self) -> bool:
        try:
            client = await self._ensure_client()
            resp = await client.get("/healthz", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
```

---

## 5. Docker 部署

### 5.1 docker-compose.yml

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
```

### 5.2 SearXNG settings.yml

```yaml
# docker/searxng/settings.yml
use_default_settings: true

general:
  instance_name: "Deep Research Search"
  debug: false

search:
  formats:
    - html
    - json
  default_lang: "auto"    # 自动检测语言
  autocomplete: false

server:
  port: 8080
  bind_address: "0.0.0.0"
  secret_key: "change-me-in-production"

engines:
  # === 中文搜索主力 ===
  - name: baidu
    engine: baidu
    disabled: false
    timeout: 6.0
    weight: 2.0              # 百度权重最高

  - name: sogou
    engine: sogou
    disabled: false
    timeout: 5.0
    weight: 1.0

  - name: 360search
    engine: xpath
    disabled: false
    timeout: 5.0
    weight: 0.8

  # === 英文搜索通道 ===
  - name: bing
    engine: bing
    disabled: false
    timeout: 6.0
    weight: 1.5
    send_accept_language_header: true

  # === 禁用被墙引擎 ===
  - name: google
    disabled: true
  - name: duckduckgo
    disabled: true
  - name: brave
    disabled: true
  - name: yahoo
    disabled: true

outgoing:
  request_timeout: 8.0
  max_request_timeout: 12.0
  # 可选：代理配置
  # proxies:
  #   all://:
  #     - http://proxy:8080
```

---

## 6. 验收标准

### 自动化测试

```bash
# 1. 启动 SearXNG
cd docker && docker compose up -d
# 等待健康检查通过
docker compose ps

# 2. 连通性测试
curl -s "http://localhost:8080/search?q=test&format=json" | python -m json.tool

# 3. 中文搜索测试
curl -s "http://localhost:8080/search?q=人工智能最新进展&format=json" | python -m json.tool

# 4. 英文搜索测试 (Bing)
curl -s "http://localhost:8080/search?q=LangGraph+tutorial&format=json&engines=bing" | python -m json.tool

# 5. 单元测试
python -m pytest tests/test_search_service/test_searxng_client.py -v
```

### 验收指标

| 指标 | 目标 |
|:---|:---|
| 中文搜索结果数 | ≥5 (query: "人工智能") |
| 英文搜索结果数 (Bing) | ≥3 (query: "LangGraph") |
| 搜索延迟 p95 | <3s |
| SearXNG 健康检查 | 通过 |
| 重试逻辑 | SearXNG 503 时自动重试 |
