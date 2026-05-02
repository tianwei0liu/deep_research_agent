"""Tests for search_service.backends.page_scraper — PageScraper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from search_service.backends.page_scraper import PageScraper
from search_service.exceptions import ContentExtractionError
from search_service.models import ScrapeResponse


class TestHtmlToMarkdown:
    """PageScraper._html_to_markdown static method."""

    def test_heading_conversion(self) -> None:
        html = "<h1>Title</h1><p>Body text</p>"
        md = PageScraper._html_to_markdown(html)
        assert "# Title" in md
        assert "Body text" in md

    def test_list_conversion(self) -> None:
        html = "<ul><li>Item 1</li><li>Item 2</li></ul>"
        md = PageScraper._html_to_markdown(html)
        assert "Item 1" in md
        assert "Item 2" in md

    def test_code_block_conversion(self) -> None:
        html = "<pre><code>print('hello')</code></pre>"
        md = PageScraper._html_to_markdown(html)
        assert "print('hello')" in md

    def test_strips_images(self) -> None:
        html = '<p>Text</p><img src="foo.png" alt="image"/><p>More</p>'
        md = PageScraper._html_to_markdown(html)
        assert "foo.png" not in md
        assert "Text" in md

    def test_empty_html(self) -> None:
        md = PageScraper._html_to_markdown("")
        assert md == ""


class TestPageScraper:
    """PageScraper.scrape with mocked BrowserPool."""

    @staticmethod
    def _make_pool_mock(mock_ctx):
        """Create a pool mock with a proper acquire() context manager."""
        from contextlib import asynccontextmanager

        mock_pool = MagicMock()

        @asynccontextmanager
        async def acquire(**kwargs):
            yield mock_ctx

        mock_pool.acquire = acquire
        return mock_pool

    @pytest.mark.asyncio
    async def test_scrape_returns_scrape_response(self) -> None:
        """Successful scrape returns ScrapeResponse with Markdown content."""
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="Test Page")
        mock_page.close = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.new_page = AsyncMock(return_value=mock_page)

        mock_pool = self._make_pool_mock(mock_ctx)
        scraper = PageScraper(mock_pool)

        with patch.object(scraper, "_remove_noise", AsyncMock()):
            with patch.object(
                scraper, "_extract_main_content",
                AsyncMock(return_value="<h1>Hello</h1><p>World</p>"),
            ):
                result = await scraper.scrape("https://example.com")

        assert isinstance(result, ScrapeResponse)
        assert result.url == "https://example.com"
        assert result.title == "Test Page"
        assert "Hello" in result.content

    @pytest.mark.asyncio
    async def test_content_length_limit(self) -> None:
        """Content is truncated to max_content_length."""
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="Title")
        mock_page.close = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.new_page = AsyncMock(return_value=mock_page)

        mock_pool = self._make_pool_mock(mock_ctx)
        scraper = PageScraper(mock_pool)

        long_content = "<p>" + "x" * 60000 + "</p>"
        with patch.object(scraper, "_remove_noise", AsyncMock()):
            with patch.object(
                scraper, "_extract_main_content",
                AsyncMock(return_value=long_content),
            ):
                result = await scraper.scrape(
                    "https://example.com", max_content_length=100,
                )

        assert result.content_length <= 100

    @pytest.mark.asyncio
    async def test_scrape_raises_extraction_error(self) -> None:
        """Page load failure raises ContentExtractionError."""
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=TimeoutError("page load timeout"))
        mock_page.close = AsyncMock()

        mock_ctx = AsyncMock()
        mock_ctx.new_page = AsyncMock(return_value=mock_page)

        mock_pool = self._make_pool_mock(mock_ctx)
        scraper = PageScraper(mock_pool)

        with pytest.raises(ContentExtractionError, match="page load timeout"):
            await scraper.scrape("https://unreachable.example.com")
