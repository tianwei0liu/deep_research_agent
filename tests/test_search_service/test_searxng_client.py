"""Tests for search_service.backends.searxng_client — SearXNG HTTP async client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from search_service.backends.searxng_client import SearXNGClient
from search_service.config import SearchServiceConfig
from search_service.exceptions import SearchProviderError
from search_service.models import SearchEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> SearchServiceConfig:
    """Minimal config for testing."""
    return SearchServiceConfig(
        searxng_base_url="http://localhost:8080",
        searxng_timeout_seconds=5.0,
        searxng_max_retries=2,
    )


@pytest.fixture
def client(config: SearchServiceConfig) -> SearXNGClient:
    return SearXNGClient(config)


# ---------------------------------------------------------------------------
# Fake SearXNG JSON responses
# ---------------------------------------------------------------------------

FAKE_SEARXNG_RESPONSE = {
    "results": [
        {
            "title": "AI Agent Guide",
            "url": "https://example.com/ai",
            "content": "A comprehensive guide to AI agents.",
            "engine": "baidu",
            "score": 0.8,
            "publishedDate": "2026-01-15",
        },
        {
            "title": "LangGraph Tutorial",
            "url": "https://example.com/langgraph",
            "content": "Learn LangGraph step by step.",
            "engine": "bing",
            "score": 1.5,  # will be clamped to 1.0
        },
    ],
    "search_time": 0.234,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSearXNGClientParseResponse:
    """_parse_response converts SearXNG JSON to SearchResponse."""

    def test_parse_response_basic(self, client: SearXNGClient) -> None:
        result = client._parse_response("AI", FAKE_SEARXNG_RESPONSE, max_results=10)
        assert result.query == "AI"
        assert result.result_count == 2
        assert result.results[0].title == "AI Agent Guide"
        assert result.results[0].source_engine == SearchEngine.BAIDU
        assert result.results[0].published_date == "2026-01-15"
        assert result.search_time_ms == 234  # 0.234s → 234ms

    def test_parse_response_max_results(self, client: SearXNGClient) -> None:
        """Only first max_results items are returned."""
        result = client._parse_response("AI", FAKE_SEARXNG_RESPONSE, max_results=1)
        assert result.result_count == 1

    def test_parse_response_unknown_engine(self, client: SearXNGClient) -> None:
        """Unknown engine name falls back to SearchEngine.SEARXNG."""
        data = {
            "results": [
                {"title": "X", "url": "https://x.com", "content": "X", "engine": "unknown_engine"},
            ],
            "search_time": 0.1,
        }
        result = client._parse_response("test", data, max_results=10)
        assert result.results[0].source_engine == SearchEngine.SEARXNG

    def test_parse_response_empty(self, client: SearXNGClient) -> None:
        """Empty results list."""
        data = {"results": [], "search_time": 0.05}
        result = client._parse_response("test", data, max_results=10)
        assert result.result_count == 0
        assert result.results == []


class TestNormalizeScore:
    """Score normalization to [0, 1]."""

    def test_none_stays_none(self) -> None:
        assert SearXNGClient._normalize_score(None) is None

    def test_normal_value(self) -> None:
        assert SearXNGClient._normalize_score(0.5) == 0.5

    def test_clamp_above_one(self) -> None:
        assert SearXNGClient._normalize_score(1.5) == 1.0

    def test_clamp_below_zero(self) -> None:
        assert SearXNGClient._normalize_score(-0.3) == 0.0


class TestSearXNGClientSearch:
    """Search method with retry logic."""

    @pytest.mark.asyncio
    async def test_search_success(self, client: SearXNGClient) -> None:
        """Successful search on first attempt."""
        import httpx

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = FAKE_SEARXNG_RESPONSE

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_response)
        mock_http_client.is_closed = False
        client._client = mock_http_client

        result = await client.search("AI Agent", max_results=5)
        assert result.query == "AI Agent"
        assert result.result_count == 2

    @pytest.mark.asyncio
    async def test_search_retry_then_success(self, client: SearXNGClient) -> None:
        """First attempt fails, second succeeds."""
        import httpx

        error_response = MagicMock()
        error_response.status_code = 503
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "503", request=MagicMock(), response=error_response,
        )

        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.raise_for_status = MagicMock()
        ok_response.json.return_value = FAKE_SEARXNG_RESPONSE

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(side_effect=[error_response, ok_response])
        mock_http_client.is_closed = False
        client._client = mock_http_client

        result = await client.search("test")
        assert result.result_count == 2

    @pytest.mark.asyncio
    async def test_search_raises_after_all_retries(self, client: SearXNGClient) -> None:
        """All retries fail → SearchProviderError."""
        import httpx

        error_response = MagicMock()
        error_response.status_code = 503
        error_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "503", request=MagicMock(), response=error_response,
        )

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=error_response)
        mock_http_client.is_closed = False
        client._client = mock_http_client

        with pytest.raises(SearchProviderError, match="Failed after 2 retries"):
            await client.search("fail")


class TestSearXNGClientHealthCheck:
    """Health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, client: SearXNGClient) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(return_value=mock_resp)
        mock_http_client.is_closed = False
        client._client = mock_http_client

        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client: SearXNGClient) -> None:
        mock_http_client = AsyncMock()
        mock_http_client.get = AsyncMock(side_effect=ConnectionError("refused"))
        mock_http_client.is_closed = False
        client._client = mock_http_client

        assert await client.health_check() is False
