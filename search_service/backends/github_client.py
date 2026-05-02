"""GitHub REST API client for repository and code search.

Uses the GitHub Search API v3. No Playwright dependency — pure HTTP.
Supports optional token authentication to raise rate limits from
10 to 30 requests per minute.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import httpx

from search_service.config import SearchServiceConfig
from search_service.exceptions import RateLimitedError, SearchProviderError
from search_service.models import SearchEngine, SearchResponse, SearchResultItem


class GitHubClient:
    """GitHub REST API search client.

    Searches repositories and code via the GitHub Search API.

    Rate Limits:
        - Without token: 10 requests/minute
        - With token: 30 requests/minute

    Args:
        config: SearchServiceConfig (reads ``github_token`` field).
    """

    API_BASE = "https://api.github.com"

    def __init__(self, config: SearchServiceConfig) -> None:
        self._token = config.github_token
        self._client: Optional[httpx.AsyncClient] = None
        self._logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "github"

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Lazy-initialize httpx client with GitHub headers."""
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {
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
        **kwargs: Any,
    ) -> SearchResponse:
        """Search GitHub repositories or code.

        Args:
            query: Search keywords (supports GitHub search syntax).
            max_results: Max results (1-30).
            search_type: ``"repositories"`` or ``"code"``.

        Returns:
            SearchResponse with GitHub results.

        Raises:
            RateLimitedError: When GitHub returns 403 with remaining=0.
            SearchProviderError: On other API failures.
        """
        endpoint = f"/search/{search_type}"
        params = {"q": query, "per_page": min(max_results, 30)}

        try:
            client = await self._ensure_client()
            response = await client.get(endpoint, params=params)

            # Rate limit check
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
        except Exception as exc:
            raise SearchProviderError("github", str(exc), exc)

    def _parse_response(
        self, query: str, data: dict, search_type: str,
    ) -> SearchResponse:
        """Parse GitHub API JSON response into SearchResponse."""
        items = data.get("items", [])
        results: list[SearchResultItem] = []

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
            search_time_ms=0,  # GitHub API doesn't report search time
            engines_used=["github"],
        )

    async def health_check(self) -> bool:
        """Check GitHub API availability via /rate_limit endpoint."""
        try:
            client = await self._ensure_client()
            resp = await client.get("/rate_limit", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
