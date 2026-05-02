"""Search service configuration via pydantic-settings.

All settings use the ``SEARCH_`` env var prefix.
Example: ``SEARCH_SEARXNG_BASE_URL=http://searxng:8080``

Kept independent from the project-level ``Settings`` dataclass;
the two connect through the ``searxng_base_url`` value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class SearchServiceConfig(BaseSettings):
    """Configuration for the MCP search service.

    Attributes:
        searxng_base_url: SearXNG instance URL.
        searxng_timeout_seconds: HTTP request timeout for SearXNG.
        searxng_max_retries: Max retry attempts on transient failures.
        browser_max_concurrency: Max concurrent Playwright BrowserContexts.
        browser_max_requests_per_instance: Recycle browser after N requests.
        browser_memory_limit_mb: JS heap memory limit per browser.
        cache_backend: Cache implementation — "null" (V1), "memory", or "redis".
        redis_url: Redis connection URL (required when cache_backend="redis").
        cache_default_ttl_seconds: Default cache TTL in seconds.
        cookie_storage_dir: Directory for platform cookie JSON files.
        github_token: GitHub personal access token (optional, raises rate limit).
        zhihu_rpm: Zhihu requests-per-minute limit.
        weibo_rpm: Weibo requests-per-minute limit.
        weixin_rpm: Weixin (Sogou) requests-per-minute limit.
    """

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
    cookie_storage_dir: Path = Path("./data/cookies")
    github_token: Optional[str] = None
    zhihu_rpm: int = 12
    weibo_rpm: int = 10
    weixin_rpm: int = 8

    model_config = SettingsConfigDict(
        env_prefix="SEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
