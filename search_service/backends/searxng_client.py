"""SearXNG HTTP async client.

Queries a self-hosted SearXNG instance via its JSON API and converts
results into the unified :class:`SearchResponse` model.

Features:
- Exponential backoff retry on transient failures.
- Lazy-initialized ``httpx.AsyncClient`` with connection pooling.
- Score normalization to [0, 1].
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import httpx

from search_service.config import SearchServiceConfig
from search_service.exceptions import SearchProviderError
from search_service.models import SearchEngine, SearchResponse, SearchResultItem


class SearXNGClient:
    """SearXNG HTTP async client.

    Searches via the SearXNG JSON API and converts results to
    :class:`SearchResponse`.

    Args:
        config: SearchServiceConfig instance.
    """

    def __init__(self, config: SearchServiceConfig) -> None:
        self._base_url = config.searxng_base_url.rstrip("/")
        self._timeout = config.searxng_timeout_seconds
        self._max_retries = config.searxng_max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "searxng"

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Lazy-initialize httpx client (connection pool reuse)."""
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
        engines: Optional[list[str]] = None,
        time_range: str = "",
        language: str = "auto",
        **kwargs: Any,
    ) -> SearchResponse:
        """Execute a search query against SearXNG.

        Args:
            query: Search keywords.
            max_results: Maximum results (1-20).
            engines: Engine list; None uses SearXNG defaults.
            time_range: Time filter ("day", "week", "month", "year", "").
            language: Search language ("auto", "zh-CN", "en-US").

        Returns:
            SearchResponse with normalized results.

        Raises:
            SearchProviderError: After all retries are exhausted.
        """
        params: dict[str, Any] = {
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

        # Exponential backoff retry
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                client = await self._ensure_client()
                response = await client.get("/search", params=params)
                response.raise_for_status()
                data = response.json()
                return self._parse_response(query, data, max_results)
            except Exception as exc:
                last_error = exc
                wait = min(2 ** attempt, 10)  # 1s, 2s, 4s, max 10s
                self._logger.warning(
                    "searxng_retry",
                    extra={
                        "attempt": attempt + 1,
                        "wait_seconds": wait,
                        "error": str(exc),
                    },
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(wait)

        raise SearchProviderError(
            "searxng",
            f"Failed after {self._max_retries} retries",
            last_error if isinstance(last_error, Exception) else None,
        )

    def _parse_response(
        self, query: str, data: dict, max_results: int,
    ) -> SearchResponse:
        """Convert SearXNG JSON to SearchResponse.

        Args:
            query: Original query string.
            data: Raw JSON dict from SearXNG.
            max_results: Truncate results to this count.

        Returns:
            Parsed SearchResponse.
        """
        raw_results = data.get("results", [])[:max_results]
        items: list[SearchResultItem] = []

        for raw in raw_results:
            engine_name = raw.get("engine", "searxng")
            try:
                source = SearchEngine(engine_name)
            except ValueError:
                source = SearchEngine.SEARXNG

            items.append(
                SearchResultItem(
                    title=raw.get("title", ""),
                    url=raw.get("url", ""),
                    content=raw.get("content", ""),
                    source_engine=source,
                    published_date=raw.get("publishedDate"),
                    score=self._normalize_score(raw.get("score")),
                )
            )

        engines_used = list({r.get("engine", "searxng") for r in raw_results})
        search_time_ms = int(data.get("search_time", 0) * 1000)

        return SearchResponse(
            query=query,
            results=items,
            result_count=len(items),
            search_time_ms=search_time_ms,
            engines_used=engines_used,
        )

    @staticmethod
    def _normalize_score(raw_score: Optional[float]) -> Optional[float]:
        """Clamp score to [0, 1] range.

        Args:
            raw_score: Raw score from SearXNG (may exceed 1.0).

        Returns:
            Normalized score, or None if input is None.
        """
        if raw_score is None:
            return None
        return max(0.0, min(1.0, raw_score))

    async def health_check(self) -> bool:
        """Check SearXNG availability via /healthz endpoint.

        Returns:
            True if SearXNG responds with HTTP 200.
        """
        try:
            client = await self._ensure_client()
            resp = await client.get("/healthz", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
