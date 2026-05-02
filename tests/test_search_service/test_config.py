"""Tests for search_service.config — SearchServiceConfig."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from search_service.config import SearchServiceConfig


class TestSearchServiceConfig:
    """SearchServiceConfig (pydantic-settings BaseSettings)."""

    def test_config_defaults(self) -> None:
        """All defaults are set correctly without any env vars."""
        with patch.dict(os.environ, {}, clear=True):
            cfg = SearchServiceConfig()
        assert cfg.searxng_base_url == "http://localhost:8080"
        assert cfg.searxng_timeout_seconds == 8.0
        assert cfg.searxng_max_retries == 3
        assert cfg.browser_max_concurrency == 3
        assert cfg.browser_max_requests_per_instance == 100
        assert cfg.browser_memory_limit_mb == 512
        assert cfg.cache_backend == "null"
        assert cfg.redis_url is None
        assert cfg.cache_default_ttl_seconds == 3600
        assert isinstance(cfg.cookie_storage_dir, Path)
        assert cfg.github_token is None
        assert cfg.zhihu_rpm == 12
        assert cfg.weibo_rpm == 10
        assert cfg.weixin_rpm == 8

    def test_config_env_prefix(self) -> None:
        """Env var with SEARCH_ prefix overrides defaults."""
        env = {"SEARCH_SEARXNG_BASE_URL": "http://searxng:9090"}
        with patch.dict(os.environ, env, clear=True):
            cfg = SearchServiceConfig()
        assert cfg.searxng_base_url == "http://searxng:9090"

    def test_config_multiple_env_overrides(self) -> None:
        """Multiple env vars override correctly."""
        env = {
            "SEARCH_BROWSER_MAX_CONCURRENCY": "5",
            "SEARCH_CACHE_BACKEND": "redis",
            "SEARCH_REDIS_URL": "redis://localhost:6379/0",
            "SEARCH_GITHUB_TOKEN": "ghp_test123",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = SearchServiceConfig()
        assert cfg.browser_max_concurrency == 5
        assert cfg.cache_backend == "redis"
        assert cfg.redis_url == "redis://localhost:6379/0"
        assert cfg.github_token == "ghp_test123"

    def test_config_cache_backend_literal(self) -> None:
        """cache_backend only accepts 'null', 'memory', or 'redis'."""
        env = {"SEARCH_CACHE_BACKEND": "invalid_backend"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(Exception):  # ValidationError
                SearchServiceConfig()
