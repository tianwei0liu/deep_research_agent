"""Tests for search_service.browser.pool — BrowserPool and StealthInjector."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from search_service.browser.pool import BrowserPool
from search_service.browser.stealth import StealthInjector
from search_service.config import SearchServiceConfig
from search_service.exceptions import BrowserPoolExhaustedError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> SearchServiceConfig:
    return SearchServiceConfig(
        browser_max_concurrency=2,
        browser_max_requests_per_instance=5,
        browser_memory_limit_mb=256,
    )


def _mock_context() -> AsyncMock:
    """Create a mock BrowserContext."""
    ctx = AsyncMock()
    ctx.close = AsyncMock()
    ctx.new_page = AsyncMock()
    ctx.add_init_script = AsyncMock()
    ctx.add_cookies = AsyncMock()
    return ctx


def _mock_browser() -> AsyncMock:
    """Create a mock Browser."""
    browser = AsyncMock()
    browser.new_context = AsyncMock(return_value=_mock_context())
    browser.close = AsyncMock()
    return browser


def _mock_playwright() -> AsyncMock:
    """Create a mock Playwright."""
    pw = AsyncMock()
    pw.chromium.launch = AsyncMock(return_value=_mock_browser())
    pw.stop = AsyncMock()
    return pw


# ---------------------------------------------------------------------------
# Tests — BrowserPool (all stealth patched at class level)
# ---------------------------------------------------------------------------

_STEALTH_PATCH = patch.object(
    StealthInjector, "load_script", return_value="// mock stealth",
)


@_STEALTH_PATCH
class TestBrowserPool:
    """BrowserPool lifecycle and concurrency management."""

    @pytest.mark.asyncio
    async def test_acquire_and_release(self, _mock, config: SearchServiceConfig) -> None:
        """Acquire a context, use it, and release the semaphore."""
        pool = BrowserPool(config)
        pool._playwright = _mock_playwright()
        pool._browser = _mock_browser()

        async with pool.acquire() as ctx:
            assert ctx is not None

        assert pool._semaphore._value == config.browser_max_concurrency

    @pytest.mark.asyncio
    async def test_concurrency_limit(self, _mock, config: SearchServiceConfig) -> None:
        """Third acquire blocks when max_concurrency=2."""
        pool = BrowserPool(config)
        pool._playwright = _mock_playwright()
        pool._browser = _mock_browser()

        acquired = []

        async def acquire_and_hold(hold_seconds: float) -> None:
            async with pool.acquire(timeout=5.0) as ctx:
                acquired.append(ctx)
                await asyncio.sleep(hold_seconds)

        t1 = asyncio.create_task(acquire_and_hold(0.5))
        t2 = asyncio.create_task(acquire_and_hold(0.5))
        await asyncio.sleep(0.05)

        assert len(acquired) == 2

        t3 = asyncio.create_task(acquire_and_hold(0.1))
        await asyncio.gather(t1, t2, t3)
        assert len(acquired) == 3

    @pytest.mark.asyncio
    async def test_timeout_raises_exhausted(self, _mock, config: SearchServiceConfig) -> None:
        """When all slots are occupied and timeout elapses, raise error."""
        pool = BrowserPool(config)
        pool._playwright = _mock_playwright()
        pool._browser = _mock_browser()

        for _ in range(config.browser_max_concurrency):
            await pool._semaphore.acquire()

        with pytest.raises(BrowserPoolExhaustedError):
            async with pool.acquire(timeout=0.1):
                pass  # pragma: no cover

        for _ in range(config.browser_max_concurrency):
            pool._semaphore.release()

    @pytest.mark.asyncio
    async def test_browser_recycle(self, _mock) -> None:
        """Browser recycles after max_requests_per_instance requests."""
        cfg = SearchServiceConfig(
            browser_max_concurrency=1,
            browser_max_requests_per_instance=2,
        )
        pool = BrowserPool(cfg)
        pool._playwright = _mock_playwright()
        pool._browser = _mock_browser()

        async with pool.acquire():
            pass
        async with pool.acquire():
            pass

        old_browser = pool._browser
        async with pool.acquire():
            pass

        old_browser.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cookie_injection(self, _mock, config: SearchServiceConfig) -> None:
        """Cookies are injected into the BrowserContext."""
        pool = BrowserPool(config)
        pool._playwright = _mock_playwright()
        mock_browser = _mock_browser()
        mock_ctx = _mock_context()
        mock_browser.new_context = AsyncMock(return_value=mock_ctx)
        pool._browser = mock_browser

        cookies = [{"name": "session", "value": "abc123", "domain": ".zhihu.com", "path": "/"}]

        async with pool.acquire(cookies=cookies) as ctx:
            ctx.add_cookies.assert_called_once_with(cookies)

    @pytest.mark.asyncio
    async def test_shutdown_cleanup(self, _mock, config: SearchServiceConfig) -> None:
        """shutdown() closes browser and stops Playwright."""
        pool = BrowserPool(config)
        pool._playwright = _mock_playwright()
        pool._browser = _mock_browser()

        await pool.shutdown()
        pool._browser.close.assert_called_once()
        pool._playwright.stop.assert_called_once()


# ---------------------------------------------------------------------------
# Tests — StealthInjector (no mock, test real behavior)
# ---------------------------------------------------------------------------


class TestStealthInjector:
    """StealthInjector: load_script caching and error handling."""

    def test_load_script_file_not_found(self, tmp_path) -> None:
        """Raises FileNotFoundError when stealth.min.js is missing."""
        original = StealthInjector._SCRIPT_PATH
        StealthInjector._SCRIPT_PATH = tmp_path / "nonexistent.js"
        try:
            with pytest.raises(FileNotFoundError, match="stealth.min.js not found"):
                StealthInjector.load_script()
        finally:
            StealthInjector._SCRIPT_PATH = original

    def test_load_script_caching(self, tmp_path) -> None:
        """Script is read once and cached."""
        script_file = tmp_path / "stealth.min.js"
        script_file.write_text("// stealth evasion script")

        original = StealthInjector._SCRIPT_PATH
        StealthInjector._SCRIPT_PATH = script_file

        try:
            result1 = StealthInjector.load_script()
            result2 = StealthInjector.load_script()
            assert result1 == "// stealth evasion script"
            assert result1 is result2  # Same object (cached)
        finally:
            StealthInjector._SCRIPT_PATH = original
