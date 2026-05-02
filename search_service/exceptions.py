"""Custom exception hierarchy for the search service.

All exceptions inherit from :class:`SearchServiceError` so callers can
catch the entire family with a single ``except SearchServiceError``.
"""

from __future__ import annotations

from typing import Optional


class SearchServiceError(Exception):
    """Base exception for all search service errors."""


class SearchProviderError(SearchServiceError):
    """A specific search backend (SearXNG, scraper, API) failed.

    Raised on HTTP 5xx, timeouts, or unexpected backend errors.

    Args:
        provider_name: Identifier of the failing backend.
        message: Human-readable error description.
        original_error: The underlying exception, if any.
    """

    def __init__(
        self,
        provider_name: str,
        message: str,
        original_error: Optional[Exception] = None,
    ) -> None:
        self.provider_name = provider_name
        self.original_error = original_error
        super().__init__(f"[{provider_name}] {message}")


class AllProvidersExhaustedError(SearchServiceError):
    """All backends in the fallback chain failed.

    Args:
        errors: Mapping of ``{backend_name: error_message}``.
    """

    def __init__(self, errors: dict[str, str]) -> None:
        self.errors = errors
        details = "; ".join(f"{k}: {v}" for k, v in errors.items())
        super().__init__(f"All providers exhausted — {details}")


class CookieExpiredError(SearchServiceError):
    """Platform cookie has expired, requiring manual refresh.

    Args:
        platform: The platform name (e.g. "zhihu", "weibo").
    """

    def __init__(self, platform: str) -> None:
        self.platform = platform
        super().__init__(f"Cookie expired for platform: {platform}")


class RateLimitedError(SearchServiceError):
    """Request was rate-limited (HTTP 429/403) by the target platform.

    Args:
        platform: The platform that rejected the request.
        retry_after_seconds: Suggested wait time before retrying.
    """

    def __init__(
        self,
        platform: str,
        retry_after_seconds: Optional[float] = None,
    ) -> None:
        self.platform = platform
        self.retry_after_seconds = retry_after_seconds
        msg = f"Rate limited by {platform}"
        if retry_after_seconds is not None:
            msg += f" (retry after {retry_after_seconds}s)"
        super().__init__(msg)


class BrowserPoolExhaustedError(SearchServiceError):
    """No browser context available within the timeout window.

    Args:
        max_concurrency: Configured concurrency limit.
        wait_timeout_seconds: How long we waited before giving up.
    """

    def __init__(
        self,
        max_concurrency: int,
        wait_timeout_seconds: float,
    ) -> None:
        self.max_concurrency = max_concurrency
        self.wait_timeout_seconds = wait_timeout_seconds
        super().__init__(
            f"Browser pool exhausted (max_concurrency={max_concurrency}, "
            f"waited {wait_timeout_seconds}s)"
        )


class ContentExtractionError(SearchServiceError):
    """Failed to extract content from a page.

    Args:
        url: The URL that was being scraped.
        reason: Human-readable failure reason.
        selector: The CSS selector that failed, if applicable.
    """

    def __init__(
        self,
        url: str,
        reason: str,
        selector: Optional[str] = None,
    ) -> None:
        self.url = url
        self.reason = reason
        self.selector = selector
        msg = f"Content extraction failed for {url}: {reason}"
        if selector:
            msg += f" (selector: {selector})"
        super().__init__(msg)
