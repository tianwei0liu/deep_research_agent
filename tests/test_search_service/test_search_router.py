"""Tests for search_service.backends.base — SearchBackend Protocol + SearchRouter."""

from __future__ import annotations

from typing import Optional
from unittest.mock import AsyncMock

import pytest

from search_service.backends.base import SearchBackend, SearchRouter
from search_service.cache import NullCache
from search_service.exceptions import AllProvidersExhaustedError, SearchProviderError
from search_service.models import SearchEngine, SearchResponse, SearchResultItem


# ---------------------------------------------------------------------------
# Helpers: Fake backends for testing
# ---------------------------------------------------------------------------


class FakeBackend:
    """Configurable fake SearchBackend for unit tests."""

    def __init__(
        self,
        name: str,
        response: Optional[SearchResponse] = None,
        error: Optional[Exception] = None,
    ) -> None:
        self._name = name
        self._response = response
        self._error = error

    @property
    def name(self) -> str:
        return self._name

    async def search(self, query: str, max_results: int = 10, **kwargs) -> SearchResponse:
        if self._error:
            raise self._error
        assert self._response is not None
        return self._response

    async def health_check(self) -> bool:
        return self._error is None


def _make_response(query: str = "test", n: int = 1) -> SearchResponse:
    """Helper to create a SearchResponse with N dummy results."""
    items = [
        SearchResultItem(
            title=f"Result {i}",
            url=f"https://example.com/{i}",
            content=f"Content {i}",
            source_engine=SearchEngine.SEARXNG,
        )
        for i in range(n)
    ]
    return SearchResponse(
        query=query, results=items, result_count=n,
        search_time_ms=100, engines_used=["searxng"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSearchBackendProtocol:
    """SearchBackend is a runtime-checkable Protocol."""

    def test_fake_backend_satisfies_protocol(self) -> None:
        backend = FakeBackend("test", response=_make_response())
        assert isinstance(backend, SearchBackend)

    def test_object_does_not_satisfy_protocol(self) -> None:
        assert not isinstance(object(), SearchBackend)


class TestSearchRouter:
    """SearchRouter: fallback chain, caching, error aggregation."""

    @pytest.mark.asyncio
    async def test_router_calls_first_backend(self) -> None:
        """First healthy backend is used."""
        resp = _make_response("hello", n=3)
        backend = FakeBackend("primary", response=resp)
        router = SearchRouter(backends=[backend], cache=NullCache())

        result = await router.search("hello")
        assert result.query == "hello"
        assert result.result_count == 3

    @pytest.mark.asyncio
    async def test_router_fallback_on_error(self) -> None:
        """When primary fails, router falls back to secondary."""
        failing = FakeBackend(
            "primary",
            error=SearchProviderError("primary", "timeout"),
        )
        fallback_resp = _make_response("test", n=2)
        healthy = FakeBackend("secondary", response=fallback_resp)
        router = SearchRouter(backends=[failing, healthy], cache=NullCache())

        result = await router.search("test")
        assert result.result_count == 2

    @pytest.mark.asyncio
    async def test_router_raises_all_exhausted(self) -> None:
        """When all backends fail, raises AllProvidersExhaustedError."""
        fail1 = FakeBackend("a", error=SearchProviderError("a", "err1"))
        fail2 = FakeBackend("b", error=SearchProviderError("b", "err2"))
        router = SearchRouter(backends=[fail1, fail2], cache=NullCache())

        with pytest.raises(AllProvidersExhaustedError) as exc_info:
            await router.search("test")
        assert "a" in exc_info.value.errors
        assert "b" in exc_info.value.errors

    @pytest.mark.asyncio
    async def test_router_cache_hit(self) -> None:
        """Cached response is returned without calling any backend."""
        resp = _make_response("cached_query")
        cache = AsyncMock()
        cache.get = AsyncMock(return_value=resp)
        cache.set = AsyncMock()

        backend = FakeBackend("never_called", response=_make_response())
        router = SearchRouter(backends=[backend], cache=cache)

        result = await router.search("cached_query")
        assert result.query == "cached_query"
        cache.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_router_cache_miss_stores(self) -> None:
        """On cache miss, result is stored after successful search."""
        cache = AsyncMock()
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()

        resp = _make_response("new_query", n=5)
        backend = FakeBackend("primary", response=resp)
        router = SearchRouter(backends=[backend], cache=cache)

        result = await router.search("new_query")
        assert result.result_count == 5
        cache.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_router_skips_non_provider_errors(self) -> None:
        """Non-SearchProviderError exceptions are NOT caught by fallback — they propagate."""
        fail = FakeBackend("buggy", error=RuntimeError("unexpected"))
        healthy = FakeBackend("backup", response=_make_response())
        router = SearchRouter(backends=[fail, healthy], cache=NullCache())

        with pytest.raises(RuntimeError, match="unexpected"):
            await router.search("test")
