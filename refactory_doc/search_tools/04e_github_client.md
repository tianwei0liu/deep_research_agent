# S04e: GitHub Client (REST API)

> **Phase**: 2 | **预估工时**: 0.5 天  
> **产出文件**: `search_service/backends/github_client.py`  
> **依赖**: S01 (models, config, exceptions)  
> **下游**: S05 (MCP Server 暴露 `github_search` 工具)

---

## 1. 目标

- 通过 GitHub REST API v3 实现仓库搜索和代码搜索
- 不需要 Playwright（GitHub API 无反爬）
- 支持 Token 认证（提升 rate limit: 10→30 次/分钟）
- 实现 SearchBackend Protocol

---

## 2. 核心接口

```python
class GitHubClient:
    """GitHub REST API 客户端。

    使用 GitHub Search API 搜索仓库和代码。
    不需要 Playwright，纯 HTTP 请求。

    API 文档: https://docs.github.com/en/rest/search

    Rate Limits:
      - 无 Token: 10 次/分钟
      - 有 Token: 30 次/分钟

    Args:
        config: SearchServiceConfig (使用 github_token 字段)。
    """

    API_BASE = "https://api.github.com"

    def __init__(self, config: SearchServiceConfig):
        self._token = config.github_token
        self._client: httpx.AsyncClient | None = None
        self._logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        return "github"

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
            if self._token:
                headers["Authorization"] = f"Bearer {self._token}"
            self._client = httpx.AsyncClient(
                base_url=self.API_BASE,
                headers=headers,
                timeout=httpx.Timeout(10.0),
            )
        return self._client

    async def search(
        self,
        query: str,
        max_results: int = 10,
        search_type: str = "repositories",
    ) -> SearchResponse:
        """搜索 GitHub。

        Args:
            query: 搜索关键词。支持 GitHub 搜索语法。
            max_results: 最大返回结果数 (1-30)。
            search_type: "repositories" 或 "code"。

        Returns:
            SearchResponse

        Raises:
            SearchProviderError: API 请求失败。
            RateLimitedError: GitHub API 限流。
        """
        endpoint = f"/search/{search_type}"
        params = {"q": query, "per_page": min(max_results, 30)}

        try:
            client = await self._ensure_client()
            response = await client.get(endpoint, params=params)

            # Rate limit 检查
            remaining = int(response.headers.get("X-RateLimit-Remaining", "999"))
            if response.status_code == 403 and remaining == 0:
                reset = int(response.headers.get("X-RateLimit-Reset", "0"))
                retry_after = max(0, reset - int(time.time()))
                raise RateLimitedError("github", retry_after)

            response.raise_for_status()
            data = response.json()
            return self._parse_response(query, data, search_type)

        except (RateLimitedError, SearchProviderError):
            raise
        except Exception as e:
            raise SearchProviderError("github", str(e), e)

    def _parse_response(
        self, query: str, data: dict, search_type: str
    ) -> SearchResponse:
        """解析 GitHub API 响应。"""
        items = data.get("items", [])
        results = []

        for item in items:
            if search_type == "repositories":
                results.append(SearchResultItem(
                    title=item.get("full_name", ""),
                    url=item.get("html_url", ""),
                    content=item.get("description", "") or "",
                    source_engine=SearchEngine.GITHUB,
                    metadata={
                        "stars": item.get("stargazers_count", 0),
                        "language": item.get("language"),
                        "updated_at": item.get("updated_at"),
                        "forks": item.get("forks_count", 0),
                    },
                ))
            elif search_type == "code":
                repo = item.get("repository", {})
                results.append(SearchResultItem(
                    title=f"{repo.get('full_name', '')} / {item.get('name', '')}",
                    url=item.get("html_url", ""),
                    content=item.get("path", ""),
                    source_engine=SearchEngine.GITHUB,
                    metadata={
                        "repo": repo.get("full_name"),
                        "path": item.get("path"),
                        "sha": item.get("sha"),
                    },
                ))

        return SearchResponse(
            query=query,
            results=results,
            result_count=len(results),
            search_time_ms=0,  # GitHub API 不返回搜索耗时
            engines_used=["github"],
        )

    async def health_check(self) -> bool:
        try:
            client = await self._ensure_client()
            resp = await client.get("/rate_limit", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
```

---

## 3. GitHub Search 语法示例

Agent 可通过 `github_search` 工具使用以下语法：

```
# 按语言搜索仓库
"LangGraph agent language:python stars:>100"

# 搜索特定代码
"SearchBackend Protocol language:python"

# 搜索用户仓库
"user:langchain-ai agent"
```

---

## 4. Rate Limit 管理

| 场景 | Rate Limit | 策略 |
|:---|:---|:---|
| 无 Token | 10 次/分钟 | 足够低频使用 |
| 有 Token | 30 次/分钟 | 建议配置 |
| 触发限流 | X-RateLimit-Remaining=0 | 抛 RateLimitedError，附带 retry_after |

---

## 5. 验收标准

```bash
python -m pytest tests/test_search_service/test_github_client.py -v
```

| 测试用例 | 说明 |
|:---|:---|
| `test_search_repositories` | 仓库搜索返回 SearchResponse |
| `test_search_code` | 代码搜索返回正确元数据 |
| `test_rate_limit_handling` | 403 + Remaining=0 → RateLimitedError |
| `test_auth_header` | Token 正确设置到 Authorization |
| `test_health_check` | /rate_limit 端点可达 |
| `test_no_token_fallback` | 无 Token 时仍可搜索（低限额） |
