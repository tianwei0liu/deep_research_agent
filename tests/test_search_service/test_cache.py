"""Tests for search_service.cache — CacheLayer protocol, NullCache, key generation."""

import pytest
import asyncio

from search_service.cache import NullCache, generate_cache_key


class TestNullCache:
    """NullCache always misses — transparent pass-through for V1."""

    @pytest.mark.asyncio
    async def test_get_always_returns_none(self) -> None:
        cache = NullCache()
        result = await cache.get("any_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_does_not_raise(self) -> None:
        cache = NullCache()
        # Should complete without error
        await cache.set("key", "value", ttl_seconds=60)

    @pytest.mark.asyncio
    async def test_delete_does_not_raise(self) -> None:
        cache = NullCache()
        await cache.delete("key")

    @pytest.mark.asyncio
    async def test_clear_does_not_raise(self) -> None:
        cache = NullCache()
        await cache.clear()

    @pytest.mark.asyncio
    async def test_set_then_get_still_none(self) -> None:
        """NullCache discards all writes — get always returns None."""
        cache = NullCache()
        await cache.set("key", "value", ttl_seconds=60)
        result = await cache.get("key")
        assert result is None


class TestGenerateCacheKey:
    """Cache key generation with normalization."""

    def test_basic_key(self) -> None:
        key = generate_cache_key("web_search", "AI Agent")
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hex digest

    def test_normalization_case_insensitive(self) -> None:
        """'AI Agent' and 'ai agent' produce the same key."""
        k1 = generate_cache_key("web_search", "AI Agent")
        k2 = generate_cache_key("web_search", "ai agent")
        assert k1 == k2

    def test_normalization_whitespace(self) -> None:
        """Extra whitespace is collapsed."""
        k1 = generate_cache_key("web_search", "AI Agent")
        k2 = generate_cache_key("web_search", "  AI   Agent  ")
        assert k1 == k2

    def test_different_tool_names_differ(self) -> None:
        """Different tool names produce different keys."""
        k1 = generate_cache_key("web_search", "test")
        k2 = generate_cache_key("zhihu_search", "test")
        assert k1 != k2

    def test_params_affect_key(self) -> None:
        """Additional params change the key."""
        k1 = generate_cache_key("web_search", "test")
        k2 = generate_cache_key("web_search", "test", max_results=5)
        assert k1 != k2

    def test_param_order_independent(self) -> None:
        """Param order doesn't matter (sorted internally)."""
        k1 = generate_cache_key("web_search", "test", a="1", b="2")
        k2 = generate_cache_key("web_search", "test", b="2", a="1")
        assert k1 == k2
