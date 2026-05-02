"""Cache abstraction layer for the search service.

V1 ships with :class:`NullCache` (transparent pass-through).
V2 can swap in Redis or in-memory LRU by implementing the
:class:`CacheLayer` protocol.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional, Protocol


class CacheLayer(Protocol):
    """Cache backend protocol.

    Any class implementing these four async methods can serve as
    the search service cache layer.
    """

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached value by key. Returns None on miss."""
        ...

    async def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Store a value with the given TTL."""
        ...

    async def delete(self, key: str) -> None:
        """Remove a specific key from the cache."""
        ...

    async def clear(self) -> None:
        """Flush the entire cache."""
        ...


class NullCache:
    """Transparent pass-through cache — all operations are no-ops.

    Used in V1 to satisfy the CacheLayer interface without any
    actual caching. V2 can seamlessly replace this with Redis.
    """

    async def get(self, key: str) -> None:
        """Always returns None (cache miss)."""
        return None

    async def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Discards the value (no storage)."""

    async def delete(self, key: str) -> None:
        """No-op."""

    async def clear(self) -> None:
        """No-op."""


# ---------------------------------------------------------------------------
# TTL constants per tool (seconds)
# ---------------------------------------------------------------------------

CACHE_TTL: dict[str, int] = {
    "web_search": 3600,       # 1h — moderate change frequency
    "zhihu_search": 14400,    # 4h — stable content
    "weibo_search": 900,      # 15min — highly time-sensitive
    "weixin_search": 7200,    # 2h — medium update frequency
    "github_search": 1800,    # 30min — fast-changing repos
    "scrape_url": 86400,      # 24h — page content is stable
}


def generate_cache_key(tool_name: str, query: str, **params: Any) -> str:
    """Generate a deterministic SHA-256 cache key.

    Normalization rules:
    - Query is lowercased and whitespace-collapsed.
    - Extra params are sorted by key for order independence.

    Args:
        tool_name: The MCP tool name (e.g. "web_search").
        query: The raw search query.
        **params: Additional parameters that affect the result.

    Returns:
        A 64-character hex digest string.
    """
    normalized = " ".join(query.lower().strip().split())
    sorted_params = sorted((str(k), str(v)) for k, v in params.items())
    raw = f"v1:{tool_name}:{normalized}:{sorted_params}"
    return hashlib.sha256(raw.encode()).hexdigest()
