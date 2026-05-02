"""Tests for search_service.backends.github_client — GitHubClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from search_service.backends.github_client import GitHubClient
from search_service.config import SearchServiceConfig
from search_service.exceptions import RateLimitedError, SearchProviderError
from search_service.models import SearchEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_with_token() -> SearchServiceConfig:
    return SearchServiceConfig(github_token="ghp_test123")


@pytest.fixture
def config_no_token() -> SearchServiceConfig:
    return SearchServiceConfig(github_token=None)


FAKE_REPO_RESPONSE = {
    "total_count": 2,
    "items": [
        {
            "full_name": "langchain-ai/langgraph",
            "html_url": "https://github.com/langchain-ai/langgraph",
            "description": "Build resilient agents with LangGraph",
            "stargazers_count": 12000,
            "language": "Python",
            "updated_at": "2026-01-15T10:00:00Z",
            "forks_count": 1200,
        },
        {
            "full_name": "microsoft/autogen",
            "html_url": "https://github.com/microsoft/autogen",
            "description": "Multi-agent framework",
            "stargazers_count": 25000,
            "language": "Python",
            "updated_at": "2026-01-14T08:00:00Z",
            "forks_count": 3000,
        },
    ],
}

FAKE_CODE_RESPONSE = {
    "total_count": 1,
    "items": [
        {
            "name": "base.py",
            "path": "search_service/backends/base.py",
            "sha": "abc123",
            "html_url": "https://github.com/user/repo/blob/main/base.py",
            "repository": {"full_name": "user/repo"},
        },
    ],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGitHubClientSearch:
    """GitHubClient.search for repositories and code."""

    @pytest.mark.asyncio
    async def test_search_repositories(self, config_with_token: SearchServiceConfig) -> None:
        client = GitHubClient(config_with_token)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"X-RateLimit-Remaining": "29"}
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = FAKE_REPO_RESPONSE

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.search("LangGraph agent", search_type="repositories")
        assert result.result_count == 2
        assert result.results[0].source_engine == SearchEngine.GITHUB
        assert result.results[0].metadata["stars"] == 12000

    @pytest.mark.asyncio
    async def test_search_code(self, config_with_token: SearchServiceConfig) -> None:
        client = GitHubClient(config_with_token)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"X-RateLimit-Remaining": "28"}
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = FAKE_CODE_RESPONSE

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.is_closed = False
        client._client = mock_http

        result = await client.search("SearchBackend Protocol", search_type="code")
        assert result.result_count == 1
        assert result.results[0].metadata["path"] == "search_service/backends/base.py"

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, config_with_token: SearchServiceConfig) -> None:
        """403 with Remaining=0 raises RateLimitedError."""
        client = GitHubClient(config_with_token)

        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.headers = {
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": "9999999999",
        }

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.is_closed = False
        client._client = mock_http

        with pytest.raises(RateLimitedError, match="github"):
            await client.search("test")

    @pytest.mark.asyncio
    async def test_auth_header(self, config_with_token: SearchServiceConfig) -> None:
        """Token is set in Authorization header."""
        client = GitHubClient(config_with_token)
        http_client = await client._ensure_client()
        assert "Authorization" in http_client.headers
        assert http_client.headers["Authorization"] == "Bearer ghp_test123"

    @pytest.mark.asyncio
    async def test_no_token_fallback(self, config_no_token: SearchServiceConfig) -> None:
        """Without token, client still works (no Authorization header)."""
        client = GitHubClient(config_no_token)
        http_client = await client._ensure_client()
        assert "Authorization" not in http_client.headers
        await client.close()


class TestGitHubClientHealthCheck:
    """GitHubClient.health_check."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, config_with_token: SearchServiceConfig) -> None:
        client = GitHubClient(config_with_token)

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.is_closed = False
        client._client = mock_http

        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, config_with_token: SearchServiceConfig) -> None:
        client = GitHubClient(config_with_token)

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=ConnectionError("refused"))
        mock_http.is_closed = False
        client._client = mock_http

        assert await client.health_check() is False
