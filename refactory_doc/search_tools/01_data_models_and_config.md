# S01: Data Models, Config & Exceptions

> **Phase**: 1 | **预估工时**: 0.5 天  
> **产出文件**: `search_service/models.py`, `config.py`, `exceptions.py`, `cache.py`  
> **依赖**: 无 | **下游**: S02, S03, S04a-e, S05, S06

---

## 1. 目标

定义搜索服务的基础数据契约：统一数据模型、配置、异常层次、缓存抽象。

---

## 2. 数据模型 (`models.py`)

### 2.1 SearchEngine 枚举

```python
class SearchEngine(str, Enum):
    BAIDU = "baidu"
    SOGOU = "sogou"
    SO360 = "360search"
    BING = "bing"
    SEARXNG = "searxng"
    ZHIHU = "zhihu"
    WEIBO = "weibo"
    WEIXIN = "weixin"
    GITHUB = "github"
```

`str, Enum` 混入确保 JSON 序列化直接输出字符串值。

### 2.2 SearchResultItem

```python
class SearchResultItem(BaseModel):
    """单条搜索结果，对标 Tavily 返回格式。"""
    title: str
    url: str
    content: str = Field(description="摘要或正文片段")
    source_engine: SearchEngine
    published_date: Optional[str] = None
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    raw_content: Optional[str] = Field(default=None, description="Markdown 全文")
    metadata: dict = Field(default_factory=dict, description="平台扩展元数据")
```

### 2.3 SearchResponse

```python
class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
    result_count: int
    search_time_ms: int
    engines_used: list[str]

    @classmethod
    def empty(cls, query: str, engines: list[str]) -> "SearchResponse":
        return cls(query=query, results=[], result_count=0, search_time_ms=0, engines_used=engines)
```

### 2.4 ScrapeResponse

```python
class ScrapeResponse(BaseModel):
    url: str
    title: str
    content: str  # Markdown
    content_length: int
    metadata: dict = Field(default_factory=dict)
```

---

## 3. 配置 (`config.py`)

```python
class SearchServiceConfig(BaseSettings):
    # SearXNG
    searxng_base_url: str = "http://localhost:8080"
    searxng_timeout_seconds: float = 8.0
    searxng_max_retries: int = 3

    # Browser Pool
    browser_max_concurrency: int = 3
    browser_max_requests_per_instance: int = 100
    browser_memory_limit_mb: int = 512

    # Cache (V1: null)
    cache_backend: Literal["null", "memory", "redis"] = "null"
    redis_url: Optional[str] = None
    cache_default_ttl_seconds: int = 3600

    # Cookie / GitHub / Rate Limits
    cookie_storage_dir: Path = Field(default=Path("./data/cookies"))
    github_token: Optional[str] = None
    zhihu_rpm: int = 12
    weibo_rpm: int = 10
    weixin_rpm: int = 8

    model_config = SettingsConfigDict(env_prefix="SEARCH_", env_file=".env")
```

与项目级 `Settings` 独立。两者通过 `searxng_base_url` 连接。

---

## 4. 异常体系 (`exceptions.py`)

```python
class SearchServiceError(Exception): ...           # Base
class SearchProviderError(SearchServiceError):      # 后端错误 (HTTP 5xx/超时)
    def __init__(self, provider_name, message, original_error=None): ...
class AllProvidersExhaustedError(SearchServiceError): # fallback 全失败
    def __init__(self, errors: dict[str, str]): ...
class CookieExpiredError(SearchServiceError):       # Cookie 过期
    def __init__(self, platform: str): ...
class RateLimitedError(SearchServiceError):         # 429/403
    def __init__(self, platform, retry_after_seconds=None): ...
class BrowserPoolExhaustedError(SearchServiceError): # 并发耗尽
    def __init__(self, max_concurrency, wait_timeout_seconds): ...
class ContentExtractionError(SearchServiceError):   # DOM 提取失败
    def __init__(self, url, reason, selector=None): ...
```

### 异常流向

| 异常 | 抛出方 | 捕获方 | 处理 |
|:---|:---|:---|:---|
| `SearchProviderError` | Client/Scraper | SearchRouter | Fallback |
| `AllProvidersExhaustedError` | SearchRouter | MCP Tool | 返回错误 |
| `CookieExpiredError` | Scraper | Scraper 内部 | `site:` 降级 + 告警 |
| `RateLimitedError` | Scraper | SearchRouter | 降频/换代理 |
| `BrowserPoolExhaustedError` | BrowserPool | MCP Tool | 返回错误 |
| `ContentExtractionError` | page_scraper | MCP Tool | 返回部分结果 |

---

## 5. 缓存抽象 (`cache.py`)

### CacheLayer Protocol

```python
class CacheLayer(Protocol):
    async def get(self, key: str) -> Optional[SearchResponse]: ...
    async def set(self, key: str, value: SearchResponse, ttl_seconds: int) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def clear(self) -> None: ...
```

### NullCache (V1)

```python
class NullCache:
    """透传：不缓存，所有操作为空。V2 可无缝替换。"""
    async def get(self, key): return None
    async def set(self, key, value, ttl_seconds): pass
    async def delete(self, key): pass
    async def clear(self): pass
```

### Cache Key 生成

```python
def generate_cache_key(tool_name: str, query: str, **params) -> str:
    normalized = " ".join(query.lower().strip().split())
    sorted_params = sorted((str(k), str(v)) for k, v in params.items())
    raw = f"v1:{tool_name}:{normalized}:{sorted_params}"
    return hashlib.sha256(raw.encode()).hexdigest()
```

### TTL 常量

| 工具 | TTL | 理由 |
|:---|:---|:---|
| `web_search` | 1h | 中等变化频率 |
| `zhihu_search` | 4h | 内容稳定 |
| `weibo_search` | 15min | 时效性强 |
| `weixin_search` | 2h | 公众号更新频率中等 |
| `github_search` | 30min | 代码仓库变化快 |
| `scrape_url` | 24h | 页面内容稳定 |

---

## 6. 验收标准

```bash
python -m pytest tests/test_search_service/test_models.py -v
```

| 测试用例 | 说明 |
|:---|:---|
| `test_search_result_item_valid` | 正常数据通过校验 |
| `test_score_range_validation` | score 超 [0,1] → ValidationError |
| `test_search_response_empty` | 工厂方法返回空结果 |
| `test_engine_json_serialization` | BAIDU 序列化为 "baidu" |
| `test_config_env_prefix` | SEARCH_SEARXNG_BASE_URL 正确注入 |
| `test_config_defaults` | 默认值正确 |
| `test_null_cache_always_miss` | get() 返回 None |
| `test_cache_key_normalization` | "AI Agent" == "ai agent" |
| `test_exception_hierarchy` | 所有异常继承 SearchServiceError |
