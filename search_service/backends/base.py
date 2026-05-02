"""SearchBackend protocol and SearchRouter strategy layer.

:class:`SearchBackend` defines the uniform interface that all search
backends (SearXNG, scrapers, GitHub API) must implement.

:class:`SearchRouter` orchestrates backends in a fallback chain with
cache-aside caching.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol, runtime_checkable

from search_service.cache import CACHE_TTL, CacheLayer, generate_cache_key
from search_service.exceptions import AllProvidersExhaustedError, SearchProviderError
from search_service.models import SearchResponse


@runtime_checkable
class SearchBackend(Protocol):
    """Unified interface for all search backends.

    Every backend (SearXNG, scrapers, API clients) must implement
    ``name``, ``search``, and ``health_check``.
    """

    @property
    def name(self) -> str:
        """Backend identifier used for logging and routing."""
        ...

    async def search(
        self, query: str, max_results: int = 10, **kwargs: Any,
    ) -> SearchResponse:
        """Execute a search query.

        Args:
            query: Search keywords.
            max_results: Maximum number of results to return.
            **kwargs: Backend-specific parameters.

        Returns:
            Unified SearchResponse.

        Raises:
            SearchProviderError: On backend failure.
        """
        ...

    async def health_check(self) -> bool:
        """Return True if the backend is operational."""
        ...


class SearchRouter:
    """Strategy routing layer: call backends in fallback order.

    V1: Single SearXNG backend, direct call.
    V2+: Fallback chain + circuit breaker + cache.

    Args:
        backends: Search backends ordered by priority (first = preferred).
        cache: Cache layer instance (V1 uses NullCache).
    """

    def __init__(
        self,
        backends: list[SearchBackend],
        cache: CacheLayer,
    ) -> None:
        self._backends = {b.name: b for b in backends}
        self._fallback_order = [b.name for b in backends]
        self._cache = cache
        self._logger = logging.getLogger(__name__)

    async def search(self, query: str, **kwargs: Any) -> SearchResponse:
        """Execute a search with cache-aside lookup and fallback.

        Args:
            query: Search keywords.
            **kwargs: Forwarded to the backend ``search()`` method.

        Returns:
            SearchResponse from the first successful backend.

        Raises:
            AllProvidersExhaustedError: All backends failed.
        """
        cache_key = generate_cache_key("web_search", query, **kwargs)

        # Cache-aside: check cache first
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

                self._logger.info(
                    "search_request",
                    extra={
                        "tool": "web_search",
                        "query": query,
                        "latency_ms": elapsed_ms,
                        "status": "success",
                        "result_count": result.result_count,
                        "backend": name,
                    },
                )

                # Store in cache
                ttl = CACHE_TTL.get("web_search", 3600)
                await self._cache.set(cache_key, result, ttl_seconds=ttl)
                return result

            except SearchProviderError as exc:
                errors[name] = str(exc)
                self._logger.warning(
                    "search_backend_failed",
                    extra={"backend": name, "error": str(exc)},
                )

        raise AllProvidersExhaustedError(errors)
