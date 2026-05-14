"""Generic URL → Markdown page scraper.

Uses Playwright (via :class:`BrowserPool`) to render pages, then extracts
the main content and converts it to clean Markdown.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse
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

    # File extensions and domains that are not scrapable HTML pages.
    _NON_HTML_EXTENSIONS = {
        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
        ".bmp", ".tiff", ".avif",
        ".mp4", ".webm", ".mov", ".avi", ".mkv",
        ".mp3", ".wav", ".ogg", ".flac",
        ".pdf", ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
        ".woff", ".woff2", ".ttf", ".otf", ".eot",
        ".bin", ".exe", ".dmg", ".iso",
    }
    _NON_HTML_DOMAINS = {
        "img.shields.io",  # SVG badges
    }

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
        self._validate_url(url)
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
            # Use backticks for JS string to avoid quote conflicts with selectors like [role='navigation']
            await page.evaluate(f"""
                document.querySelectorAll(`{selector}`)
                    .forEach(el => el.remove());
            """)

    def _validate_url(self, url: str) -> None:
        """Reject URLs that point to non-HTML resources.

        Args:
            url: Target URL to validate.

        Raises:
            ContentExtractionError: If the URL points to a known
                non-scrapable resource (image, video, binary, etc.).
        """
        parsed = urlparse(url)
        path_lower = parsed.path.lower()

        # Check domain blocklist (e.g. img.shields.io serves SVGs)
        if parsed.hostname and parsed.hostname in self._NON_HTML_DOMAINS:
            raise ContentExtractionError(
                url,
                f"Non-HTML domain: {parsed.hostname} serves non-scrapable "
                f"content (badges, images). Skip this URL.",
            )

        # Check file extension
        for ext in self._NON_HTML_EXTENSIONS:
            if path_lower.endswith(ext):
                raise ContentExtractionError(
                    url,
                    f"Non-HTML resource (extension: {ext}). "
                    f"Cannot extract text from binary/media files.",
                )

    async def _extract_main_content(self, page) -> str:
        """Extract main content HTML.

        Priority: <article> → <main> → highest text-density div → <body>.
        Falls back to empty string when document.body is absent
        (e.g. SVG or XML documents).
        """
        article = await page.query_selector("article")
        if article:
            return await article.inner_html()

        main = await page.query_selector("main")
        if main:
            return await main.inner_html()

        # Readability heuristic: find div with highest text density.
        # Guard against null document.body (SVG/XML responses).
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
                if (best) return best.innerHTML;
                if (document.body) return document.body.innerHTML;
                return '';
            }
        """)
        return result or ""

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
