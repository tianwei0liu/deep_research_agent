"""Unit tests for BochaClient.

Covers: successful search, response parsing, freshness mapping,
HTTP errors, timeouts, API business errors, empty results, summary
field, health_check, and the SearchBackend protocol contract.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from search_service.backends.bocha_client import BochaClient, _FRESHNESS_MAP
from search_service.config import SearchServiceConfig
from search_service.exceptions import SearchProviderError
from search_service.models import SearchEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(**overrides: Any) -> SearchServiceConfig:
    """Create a SearchServiceConfig with Bocha defaults for testing."""
    defaults = {
        "bocha_api_key": "sk-test-key",
        "bocha_base_url": "https://api.bochaai.com",
        "bocha_timeout_seconds": 5.0,
        "bocha_summary_enabled": False,
    }
    defaults.update(overrides)
    return SearchServiceConfig(**defaults)


def _bocha_response(
    results: list[dict] | None = None,
    code: int = 200,
    msg: str = "success",
) -> dict:
    """Build a mock Bocha API response body."""
    if results is None:
        results = [
            {
                "name": "Test Title",
                "url": "https://example.com/article",
                "snippet": "Test snippet content",
                "siteName": "Example",
                "siteIcon": "https://example.com/favicon.ico",
                "publicationTime": "2026-05-10T08:00:00Z",
            },
        ]
    return {
        "code": code,
        "msg": msg,
        "data": {
            "webPages": {"value": results},
            "images": {"value": []},
        },
    }


@pytest.fixture
def config() -> SearchServiceConfig:
    return _make_config()


@pytest.fixture
def client(config: SearchServiceConfig) -> BochaClient:
    return BochaClient(config)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestBochaClientInit:
    """Tests for BochaClient construction and protocol compliance."""

    def test_name_property(self, client: BochaClient) -> None:
        assert client.name == "bocha"

    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="bocha_api_key must be set"):
            BochaClient(_make_config(bocha_api_key=None))


# ---------------------------------------------------------------------------
# Search — success paths
# ---------------------------------------------------------------------------

class TestBochaSearch:
    """Tests for the search() method happy paths."""

    @pytest.mark.asyncio
    async def test_search_success(self, client: BochaClient) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _bocha_response()

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.is_closed = False
        mock_http.post.return_value = mock_resp
        client._client = mock_http

        result = await client.search("AI Agent 2026", max_results=5)

        assert result.query == "AI Agent 2026"
        assert result.result_count == 1
        assert result.engines_used == ["bocha"]
        assert result.results[0].title == "Test Title"
        assert result.results[0].url == "https://example.com/article"
        assert result.results[0].content == "Test snippet content"
        assert result.results[0].source_engine == SearchEngine.BOCHA

    @pytest.mark.asyncio
    async def test_parse_metadata_fields(self, client: BochaClient) -> None:
        """siteName and siteIcon should land in metadata dict."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _bocha_response()

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.is_closed = False
        mock_http.post.return_value = mock_resp
        client._client = mock_http

        result = await client.search("test")
        item = result.results[0]

        assert item.metadata["site_name"] == "Example"
        assert item.metadata["site_icon"] == "https://example.com/favicon.ico"
        assert item.published_date == "2026-05-10T08:00:00Z"

    @pytest.mark.asyncio
    async def test_summary_field_present(self) -> None:
        """When summary=true, raw_content should contain the summary."""
        cfg = _make_config(bocha_summary_enabled=True)
        c = BochaClient(cfg)

        results = [
            {
                "name": "Title",
                "url": "https://example.com",
                "snippet": "short",
                "summary": "Long AI-generated summary text here.",
            },
        ]
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _bocha_response(results=results)

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.is_closed = False
        mock_http.post.return_value = mock_resp
        c._client = mock_http

        result = await c.search("test")
        assert result.results[0].raw_content == "Long AI-generated summary text here."

    @pytest.mark.asyncio
    async def test_empty_results(self, client: BochaClient) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _bocha_response(results=[])

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.is_closed = False
        mock_http.post.return_value = mock_resp
        client._client = mock_http

        result = await client.search("obscure query xyz")
        assert result.result_count == 0
        assert result.results == []


# ---------------------------------------------------------------------------
# Freshness mapping
# ---------------------------------------------------------------------------

class TestFreshnessMapping:
    """Verify time_range → Bocha freshness parameter mapping."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "time_range,expected_freshness",
        [
            ("", "noLimit"),
            ("day", "oneDay"),
            ("week", "oneWeek"),
            ("month", "oneMonth"),
            ("year", "oneYear"),
            ("unknown", "noLimit"),  # fallback for unmapped values
        ],
    )
    async def test_freshness_param(
        self, client: BochaClient,
        time_range: str, expected_freshness: str,
    ) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = _bocha_response()

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.is_closed = False
        mock_http.post.return_value = mock_resp
        client._client = mock_http

        await client.search("test", time_range=time_range)

        call_kwargs = mock_http.post.call_args
        sent_payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert sent_payload["freshness"] == expected_freshness


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestBochaErrors:
    """Tests for error paths: HTTP errors, timeouts, API codes."""

    @pytest.mark.asyncio
    async def test_http_401_raises_provider_error(
        self, client: BochaClient,
    ) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 401
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=MagicMock(), response=mock_resp,
        )

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.is_closed = False
        mock_http.post.return_value = mock_resp
        client._client = mock_http

        with pytest.raises(SearchProviderError, match="bocha"):
            await client.search("test")

    @pytest.mark.asyncio
    async def test_http_429_raises_provider_error(
        self, client: BochaClient,
    ) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 429
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Too Many Requests", request=MagicMock(), response=mock_resp,
        )

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.is_closed = False
        mock_http.post.return_value = mock_resp
        client._client = mock_http

        with pytest.raises(SearchProviderError, match="bocha"):
            await client.search("test")

    @pytest.mark.asyncio
    async def test_timeout_raises_provider_error(
        self, client: BochaClient,
    ) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.is_closed = False
        mock_http.post.side_effect = httpx.ReadTimeout("read timed out")
        client._client = mock_http

        with pytest.raises(SearchProviderError, match="Timeout"):
            await client.search("test")

    @pytest.mark.asyncio
    async def test_api_code_non_200_raises_provider_error(
        self, client: BochaClient,
    ) -> None:
        """Bocha returns HTTP 200 but business code != 200."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "code": 500,
            "msg": "internal error",
            "data": {},
        }

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.is_closed = False
        mock_http.post.return_value = mock_resp
        client._client = mock_http

        with pytest.raises(SearchProviderError, match="API error 500"):
            await client.search("test")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestBochaHealthCheck:
    """Tests for health_check()."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, client: BochaClient) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.is_closed = False
        mock_http.post.return_value = mock_resp
        client._client = mock_http

        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client: BochaClient) -> None:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_http.is_closed = False
        mock_http.post.side_effect = httpx.ConnectError("connection refused")
        client._client = mock_http

        assert await client.health_check() is False
