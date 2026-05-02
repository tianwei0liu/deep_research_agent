"""Playwright BrowserContext pool with concurrency control.

Manages browser instance lifecycle, semaphore-based concurrency limiting,
stealth injection, and automatic browser recycling after N requests.
"""

from __future__ import annotations

import asyncio
import logging
import random
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from playwright.async_api import BrowserContext, Browser, Playwright
from playwright.async_api import async_playwright

from search_service.browser.stealth import StealthInjector
from search_service.config import SearchServiceConfig
from search_service.exceptions import BrowserPoolExhaustedError


class BrowserPool:
    """Playwright BrowserContext pool.

    Controls concurrent browser contexts via ``asyncio.Semaphore``.
    Each context is isolated (cookies, cache, session) and has
    ``stealth.min.js`` injected for anti-detection.

    Args:
        config: Search service configuration.

    Usage::

        pool = BrowserPool(config)
        await pool.start()

        async with pool.acquire() as context:
            page = await context.new_page()
            await page.goto("https://example.com")
            content = await page.content()

        await pool.shutdown()
    """

    _USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) "
        "Gecko/20100101 Firefox/128.0",
    ]

    def __init__(self, config: SearchServiceConfig) -> None:
        self._max_concurrency = config.browser_max_concurrency
        self._max_requests = config.browser_max_requests_per_instance
        self._memory_limit_mb = config.browser_memory_limit_mb
        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        self._browser: Optional[Browser] = None
        self._playwright: Optional[Playwright] = None
        self._request_count = 0
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)

    async def start(self) -> None:
        """Start Playwright and launch the Chromium browser process."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                f"--js-flags=--max-old-space-size={self._memory_limit_mb}",
            ],
        )
        self._logger.info(
            "browser_pool_started",
            extra={"max_concurrency": self._max_concurrency},
        )

    @asynccontextmanager
    async def acquire(
        self,
        timeout: float = 10.0,
        cookies: Optional[list[dict]] = None,
    ) -> AsyncGenerator[BrowserContext, None]:
        """Acquire a BrowserContext from the pool.

        Args:
            timeout: Max seconds to wait for a semaphore slot.
            cookies: Pre-set cookies (for Zhihu/Weibo login state).

        Yields:
            A BrowserContext with stealth.js injected.

        Raises:
            BrowserPoolExhaustedError: If wait times out.
        """
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            raise BrowserPoolExhaustedError(self._max_concurrency, timeout)

        try:
            await self._maybe_recycle_browser()
            context = await self._create_context(cookies)
            try:
                yield context
            finally:
                await context.close()
                async with self._lock:
                    self._request_count += 1
        finally:
            self._semaphore.release()

    async def _create_context(
        self, cookies: Optional[list[dict]],
    ) -> BrowserContext:
        """Create a new BrowserContext with stealth injection.

        Args:
            cookies: Optional cookies to inject.

        Returns:
            Configured BrowserContext.
        """
        assert self._browser is not None, "BrowserPool.start() not called"
        context = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=random.choice(self._USER_AGENTS),
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
        )
        # Inject stealth.js into every new page in this context
        stealth_js = StealthInjector.load_script()
        await context.add_init_script(stealth_js)

        if cookies:
            await context.add_cookies(cookies)

        return context

    async def _maybe_recycle_browser(self) -> None:
        """Recycle browser if request count exceeds the limit."""
        async with self._lock:
            if self._request_count >= self._max_requests:
                self._logger.info(
                    "browser_recycle",
                    extra={"requests_served": self._request_count},
                )
                if self._browser:
                    await self._browser.close()
                assert self._playwright is not None
                self._browser = await self._playwright.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage"],
                )
                self._request_count = 0

    async def shutdown(self) -> None:
        """Close browser and stop Playwright."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._logger.info("browser_pool_shutdown")
