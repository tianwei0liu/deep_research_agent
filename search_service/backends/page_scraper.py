"""Generic URL → Markdown page scraper.

Uses Playwright (via :class:`BrowserPool`) to render pages, then extracts
the main content and converts it to clean Markdown.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from search_service.browser.pool import BrowserPool
from search_service.exceptions import ContentExtractionError
from search_service.models import ScrapeResponse


class PageScraper:
    """Generic page scraper: URL → Markdown.

    Renders pages with Playwright, removes noise (nav, sidebar, ads),
    extracts the main content area, and converts HTML to Markdown.

    Args:
        browser_pool: BrowserPool instance for acquiring contexts.
    """

    _NOISE_SELECTORS = [
        "nav", "header", "footer",
        "[role='navigation']", "[role='banner']",
        ".sidebar", ".ad", ".advertisement",
        ".comment", ".comments",
        "script", "style", "iframe",
    ]

    def __init__(self, browser_pool: BrowserPool) -> None:
        self._pool = browser_pool
        self._logger = logging.getLogger(__name__)

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "page_scraper"

    async def scrape(
        self,
        url: str,
        timeout_seconds: float = 15.0,
        max_content_length: int = 50000,
        wait_for_selector: Optional[str] = None,
    ) -> ScrapeResponse:
        """Scrape a URL and return Markdown content.

        Args:
            url: Target URL.
            timeout_seconds: Page load timeout.
            max_content_length: Max characters in output.
            wait_for_selector: Optional CSS selector to wait for.

        Returns:
            ScrapeResponse with Markdown content.

        Raises:
            ContentExtractionError: On extraction failure.
        """
        async with self._pool.acquire() as context:
            page = await context.new_page()
            try:
                await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=timeout_seconds * 1000,
                )

                if wait_for_selector:
                    await page.wait_for_selector(
                        wait_for_selector, timeout=5000,
                    )

                await self._remove_noise(page)
                main_html = await self._extract_main_content(page)
                content = self._html_to_markdown(main_html)
                content = content[:max_content_length]
                title = await page.title() or ""

                return ScrapeResponse(
                    url=url,
                    title=title,
                    content=content,
                    content_length=len(content),
                    metadata={"source": "playwright"},
                )
            except ContentExtractionError:
                raise
            except Exception as exc:
                raise ContentExtractionError(url, str(exc))
            finally:
                await page.close()

    async def _remove_noise(self, page) -> None:
        """Remove non-content elements (nav, sidebar, ads, etc.)."""
        for selector in self._NOISE_SELECTORS:
            await page.evaluate(f"""
                document.querySelectorAll('{selector}')
                    .forEach(el => el.remove());
            """)

    async def _extract_main_content(self, page) -> str:
        """Extract main content HTML.

        Priority: <article> → <main> → highest text-density div.
        """
        article = await page.query_selector("article")
        if article:
            return await article.inner_html()

        main = await page.query_selector("main")
        if main:
            return await main.inner_html()

        # Readability heuristic: find div with highest text density
        result = await page.evaluate("""
            () => {
                const divs = document.querySelectorAll('div, section');
                let best = null, bestScore = 0;
                for (const div of divs) {
                    const text = div.innerText || '';
                    const score = text.length - (div.querySelectorAll('a').length * 30);
                    if (score > bestScore) {
                        bestScore = score;
                        best = div;
                    }
                }
                return best ? best.innerHTML : document.body.innerHTML;
            }
        """)
        return result

    @staticmethod
    def _html_to_markdown(html: str) -> str:
        """Convert HTML to Markdown.

        Uses markdownify with ATX heading style, strips media elements.

        Args:
            html: Raw HTML string.

        Returns:
            Clean Markdown string.
        """
        if not html:
            return ""
        try:
            import markdownify
            return markdownify.markdownify(
                html,
                heading_style="ATX",
                strip=["img", "video", "audio"],
            ).strip()
        except ImportError:
            # Fallback: simple tag stripping
            text = re.sub(r"<[^>]+>", "", html)
            return re.sub(r"\s+", " ", text).strip()
