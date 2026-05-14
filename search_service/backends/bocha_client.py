"""Bocha Web Search API async client.

Queries the Bocha Web Search API and converts results into the unified
:class:`SearchResponse` model. Follows the same structural pattern as
:class:`SearXNGClient` for consistency.

Features:
- Lazy-initialized ``httpx.AsyncClient`` with connection pooling.
- ``freshness`` parameter mapping from MCP time_range convention.
- Wraps all failures as ``SearchProviderError`` for SearchRouter fallback.
"""

from __future__ import annotations

import asyncio
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

    Implements the :class:`SearchBackend` protocol. Uses ``httpx`` for
    async HTTP with lazy-initialized connection pooling.

    Args:
        config: SearchServiceConfig instance with ``bocha_*`` fields.
    """

    API_BASE = "https://api.bochaai.com"
    _MAX_CONCURRENT = 3
    _MIN_INTERVAL_SECONDS = 0.3

    def __init__(self, config: SearchServiceConfig) -> None:
        if not config.bocha_api_key:
            raise ValueError("bocha_api_key must be set to use BochaClient")
        self._api_key: str = config.bocha_api_key
        self._base_url = (config.bocha_base_url or self.API_BASE).rstrip("/")
        self._timeout = config.bocha_timeout_seconds
        self._summary_enabled = config.bocha_summary_enabled
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(self._MAX_CONCURRENT)
        self._last_request_time: float = 0.0
        self._rate_lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        """Backend identifier used for logging and routing."""
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
            **kwargs: Supports ``time_range`` (str) for freshness filter.
                Other kwargs (``engines``, ``language``) are ignored since
                Bocha does not support them.

        Returns:
            Unified SearchResponse.

        Raises:
            SearchProviderError: On any Bocha API failure.
        """
        async with self._semaphore:
            await self._throttle()
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
                    extra={
                        "status": exc.response.status_code,
                        "query": query,
                    },
                )
                raise SearchProviderError("bocha", str(exc), exc)
            except httpx.TimeoutException as exc:
                raise SearchProviderError("bocha", f"Timeout: {exc}", exc)
            except SearchProviderError:
                raise
            except Exception as exc:
                raise SearchProviderError("bocha", str(exc), exc)

    async def _throttle(self) -> None:
        """Enforce minimum interval between Bocha API requests."""
        async with self._rate_lock:
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < self._MIN_INTERVAL_SECONDS:
                await asyncio.sleep(self._MIN_INTERVAL_SECONDS - elapsed)
            self._last_request_time = time.monotonic()

    def _parse_response(
        self, query: str, data: dict, elapsed_ms: int,
    ) -> SearchResponse:
        """Convert Bocha JSON to SearchResponse.

        Bocha wraps results in ``data.webPages.value``. Each item has
        ``name``, ``url``, ``snippet``, and optionally ``summary``,
        ``siteName``, ``siteIcon``, ``publicationTime``.

        Args:
            query: Original query string.
            data: Raw JSON dict from Bocha API.
            elapsed_ms: Request latency in milliseconds.

        Returns:
            Parsed SearchResponse.

        Raises:
            SearchProviderError: If Bocha returns a non-200 business code.
        """
        code = data.get("code", 0)
        if code != 200:
            msg = data.get("msg", "Unknown Bocha error")
            raise SearchProviderError("bocha", f"API error {code}: {msg}")

        web_pages = data.get("data", {}).get("webPages", {})
        raw_results = web_pages.get("value") or []

        items: list[SearchResultItem] = []
        for raw in raw_results:
            metadata: dict[str, Any] = {}
            if raw.get("siteName"):
                metadata["site_name"] = raw["siteName"]
            if raw.get("siteIcon"):
                metadata["site_icon"] = raw["siteIcon"]

            items.append(
                SearchResultItem(
                    title=raw.get("name", ""),
                    url=raw.get("url", ""),
                    content=raw.get("snippet", ""),
                    source_engine=SearchEngine.BOCHA,
                    published_date=raw.get("publicationTime"),
                    score=None,
                    raw_content=raw.get("summary"),
                    metadata=metadata,
                ),
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
