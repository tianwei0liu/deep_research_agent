"""Live integration tests for search quality against a running SearXNG instance.

These tests are marked with ``@pytest.mark.integration`` and are skipped
by default. Run them explicitly with::

    pytest tests/test_search_service/test_search_quality.py -m integration -v

Requires a running SearXNG instance on ``http://localhost:8080``.
"""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from search_service.backends.searxng_client import SearXNGClient
from search_service.config import SearchServiceConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _searxng_available() -> bool:
    """Check if SearXNG is running locally."""
    try:
        resp = httpx.get(
            "http://localhost:8080/healthz", timeout=3.0,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _content_fill_rate(results: list) -> float:
    """Fraction of results with non-empty content."""
    if not results:
        return 0.0
    non_empty = sum(1 for r in results if r.content.strip())
    return non_empty / len(results)


def _relevance_p5(results: list, keywords: list[str]) -> float:
    """Precision@5: fraction of top-5 results matching any keyword."""
    top5 = results[:5]
    if not top5 or not keywords:
        return 1.0
    hits = 0
    for r in top5:
        text = (r.title + " " + r.content).lower()
        if any(kw.lower() in text for kw in keywords):
            hits += 1
    return hits / len(top5)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_skip_reason = "SearXNG not running on localhost:8080"
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _searxng_available(), reason=_skip_reason),
]


@pytest_asyncio.fixture
async def live_client() -> SearXNGClient:
    """SearXNG client connected to live instance."""
    config = SearchServiceConfig(
        searxng_base_url="http://localhost:8080",
        searxng_timeout_seconds=10.0,
        searxng_max_retries=2,
    )
    client = SearXNGClient(config)
    yield client
    await client.close()


# ---------------------------------------------------------------------------
# English Technical Queries
# ---------------------------------------------------------------------------


class TestEnglishSearchQuality:
    """English search quality validation."""

    @pytest.mark.asyncio
    async def test_langgraph_memory(self, live_client: SearXNGClient) -> None:
        """LangGraph memory query returns relevant results."""
        response = await live_client.search(
            "LangGraph memory management", max_results=10,
        )
        assert response.result_count >= 5, (
            f"Expected >=5 results, got {response.result_count}"
        )
        assert _content_fill_rate(response.results) >= 0.70
        assert _relevance_p5(
            response.results,
            ["langgraph", "memory", "checkpoint", "persistence"],
        ) >= 0.60

    @pytest.mark.asyncio
    async def test_rag_comparison(self, live_client: SearXNGClient) -> None:
        """RAG vs fine-tuning query returns relevant results."""
        response = await live_client.search(
            "RAG vs fine-tuning comparison 2025", max_results=10,
        )
        assert response.result_count >= 3
        assert _relevance_p5(
            response.results,
            ["rag", "fine-tun", "retrieval", "augment"],
        ) >= 0.40  # Relaxed — this is a broad query

    @pytest.mark.asyncio
    async def test_multi_agent(self, live_client: SearXNGClient) -> None:
        """Multi-agent query returns framework results."""
        response = await live_client.search(
            "multi-agent orchestration framework", max_results=10,
        )
        assert response.result_count >= 5


# ---------------------------------------------------------------------------
# Chinese Technical Queries
# ---------------------------------------------------------------------------


class TestChineseSearchQuality:
    """Chinese search quality validation."""

    @pytest.mark.asyncio
    async def test_ai_progress(self, live_client: SearXNGClient) -> None:
        """Chinese AI progress query returns relevant results."""
        response = await live_client.search(
            "人工智能 大模型 最新进展 2026", max_results=10,
        )
        assert response.result_count >= 5
        assert _relevance_p5(
            response.results,
            ["大模型", "AI", "人工智能", "模型"],
        ) >= 0.60

    @pytest.mark.asyncio
    async def test_langchain_rag(self, live_client: SearXNGClient) -> None:
        """Chinese LangChain RAG query returns results."""
        response = await live_client.search(
            "LangChain 向量数据库 RAG 教程", max_results=10,
        )
        assert response.result_count >= 3


# ---------------------------------------------------------------------------
# Site Operator
# ---------------------------------------------------------------------------


class TestSiteOperator:
    """site: operator precision validation."""

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="Sogou site: operator intermittently returns 0 results (D4)",
        strict=False,
    )
    async def test_site_zhihu(self, live_client: SearXNGClient) -> None:
        """site:zhihu.com returns only zhihu URLs.

        Note: This test is xfail because sogou (the only engine that
        reliably handles site: queries) has intermittent availability.
        See deficiency D4 in the quality audit.
        """
        response = await live_client.search(
            "site:zhihu.com AI Agent 框架", max_results=10,
        )
        assert response.result_count >= 3

        zhihu_hits = sum(
            1 for r in response.results if "zhihu.com" in r.url
        )
        hit_rate = zhihu_hits / max(1, response.result_count)
        assert hit_rate >= 0.80, (
            f"site: hit rate {hit_rate:.0%} < 80%"
        )


# ---------------------------------------------------------------------------
# Content Quality
# ---------------------------------------------------------------------------


class TestContentQuality:
    """Content fill rate and basic quality checks."""

    @pytest.mark.asyncio
    async def test_content_fill_rate_english(
        self, live_client: SearXNGClient,
    ) -> None:
        """English query achieves high content fill rate."""
        response = await live_client.search(
            "Python asyncio best practices", max_results=10,
        )
        rate = _content_fill_rate(response.results)
        assert rate >= 0.70, f"Content fill rate {rate:.0%} < 70%"

    @pytest.mark.asyncio
    async def test_content_fill_rate_chinese(
        self, live_client: SearXNGClient,
    ) -> None:
        """Chinese query achieves acceptable content fill rate."""
        response = await live_client.search(
            "Kubernetes 容器编排 最佳实践", max_results=10,
        )
        rate = _content_fill_rate(response.results)
        assert rate >= 0.60, f"Content fill rate {rate:.0%} < 60%"

    @pytest.mark.asyncio
    async def test_no_full_deduplication(
        self, live_client: SearXNGClient,
    ) -> None:
        """Results should have mostly unique URLs."""
        response = await live_client.search(
            "LangGraph tutorial", max_results=10,
        )
        urls = [
            r.url.split("?")[0].rstrip("/")
            for r in response.results
        ]
        unique_rate = len(set(urls)) / max(1, len(urls))
        assert unique_rate >= 0.90, (
            f"URL uniqueness {unique_rate:.0%} < 90%"
        )
