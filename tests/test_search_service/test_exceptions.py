"""Tests for search_service.exceptions — Exception hierarchy."""

import pytest

from search_service.exceptions import (
    SearchServiceError,
    SearchProviderError,
    AllProvidersExhaustedError,
    CookieExpiredError,
    RateLimitedError,
    BrowserPoolExhaustedError,
    ContentExtractionError,
)


class TestExceptionHierarchy:
    """All exceptions inherit from SearchServiceError."""

    def test_all_inherit_from_base(self) -> None:
        """Every custom exception is a subclass of SearchServiceError."""
        assert issubclass(SearchProviderError, SearchServiceError)
        assert issubclass(AllProvidersExhaustedError, SearchServiceError)
        assert issubclass(CookieExpiredError, SearchServiceError)
        assert issubclass(RateLimitedError, SearchServiceError)
        assert issubclass(BrowserPoolExhaustedError, SearchServiceError)
        assert issubclass(ContentExtractionError, SearchServiceError)

    def test_base_is_exception(self) -> None:
        assert issubclass(SearchServiceError, Exception)


class TestSearchProviderError:
    """SearchProviderError carries provider name and original error."""

    def test_attributes(self) -> None:
        original = ConnectionError("timeout")
        err = SearchProviderError("searxng", "connection failed", original)
        assert err.provider_name == "searxng"
        assert "connection failed" in str(err)
        assert err.original_error is original

    def test_without_original_error(self) -> None:
        err = SearchProviderError("baidu", "HTTP 503")
        assert err.original_error is None


class TestAllProvidersExhaustedError:
    """AllProvidersExhaustedError carries errors dict."""

    def test_attributes(self) -> None:
        errors = {"searxng": "timeout", "baidu": "503"}
        err = AllProvidersExhaustedError(errors)
        assert err.errors == errors
        assert "searxng" in str(err)


class TestCookieExpiredError:
    """CookieExpiredError carries platform name."""

    def test_attributes(self) -> None:
        err = CookieExpiredError("zhihu")
        assert err.platform == "zhihu"
        assert "zhihu" in str(err)


class TestRateLimitedError:
    """RateLimitedError carries platform and retry-after."""

    def test_with_retry_after(self) -> None:
        err = RateLimitedError("weixin_sogou", retry_after_seconds=300)
        assert err.platform == "weixin_sogou"
        assert err.retry_after_seconds == 300

    def test_without_retry_after(self) -> None:
        err = RateLimitedError("github")
        assert err.retry_after_seconds is None


class TestBrowserPoolExhaustedError:
    """BrowserPoolExhaustedError carries concurrency and timeout info."""

    def test_attributes(self) -> None:
        err = BrowserPoolExhaustedError(max_concurrency=3, wait_timeout_seconds=10.0)
        assert err.max_concurrency == 3
        assert err.wait_timeout_seconds == 10.0


class TestContentExtractionError:
    """ContentExtractionError carries URL and reason."""

    def test_attributes(self) -> None:
        err = ContentExtractionError("https://example.com", "DOM timeout", selector=".main")
        assert err.url == "https://example.com"
        assert err.reason == "DOM timeout"
        assert err.selector == ".main"

    def test_without_selector(self) -> None:
        err = ContentExtractionError("https://example.com", "empty body")
        assert err.selector is None
