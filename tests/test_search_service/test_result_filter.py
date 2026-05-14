"""Tests for search_service.backends.result_filter — SearchResultFilter."""

from __future__ import annotations

import pytest

from search_service.backends.result_filter import SearchResultFilter
from search_service.models import SearchEngine, SearchResultItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _item(
    title: str = "Test",
    url: str = "https://example.com",
    content: str = "Test content",
    score: float = 0.5,
    engine: SearchEngine = SearchEngine.SEARXNG,
) -> SearchResultItem:
    """Factory for test SearchResultItem."""
    return SearchResultItem(
        title=title,
        url=url,
        content=content,
        source_engine=engine,
        score=score,
    )


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """URL deduplication with normalization."""

    def test_no_duplicates(self) -> None:
        """Items with unique URLs pass through unchanged."""
        f = SearchResultFilter()
        items = [
            _item(url="https://a.com/1"),
            _item(url="https://b.com/2"),
        ]
        result = f._deduplicate(items)
        assert len(result) == 2

    def test_removes_exact_duplicates(self) -> None:
        """Exact duplicate URLs are removed."""
        f = SearchResultFilter()
        items = [
            _item(title="First", url="https://a.com/page", score=0.3),
            _item(title="Second", url="https://a.com/page", score=0.8),
        ]
        result = f._deduplicate(items)
        assert len(result) == 1
        assert result[0].title == "Second"  # Higher score kept

    def test_normalizes_query_params(self) -> None:
        """URLs differing only in query params are treated as same."""
        f = SearchResultFilter()
        items = [
            _item(url="https://a.com/page?ref=twitter", score=0.9),
            _item(url="https://a.com/page?ref=google", score=0.1),
        ]
        result = f._deduplicate(items)
        assert len(result) == 1
        assert result[0].score == 0.9

    def test_normalizes_trailing_slash(self) -> None:
        """URLs differing only in trailing slash are deduplicated."""
        f = SearchResultFilter()
        items = [
            _item(url="https://a.com/page/"),
            _item(url="https://a.com/page"),
        ]
        result = f._deduplicate(items)
        assert len(result) == 1

    def test_normalizes_fragment(self) -> None:
        """URLs differing only in fragment are deduplicated."""
        f = SearchResultFilter()
        items = [
            _item(url="https://a.com/page#section1"),
            _item(url="https://a.com/page#section2"),
        ]
        result = f._deduplicate(items)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Empty content removal
# ---------------------------------------------------------------------------


class TestEmptyContentRemoval:
    """Removal of results with empty content."""

    def test_removes_empty_content(self) -> None:
        """Items with empty content are removed."""
        f = SearchResultFilter(min_keep=1)
        items = [
            _item(title="Good", content="Real content"),
            _item(title="Empty", content=""),
            _item(title="Whitespace", content="   "),
        ]
        result = f._remove_empty_content(items)
        assert len(result) == 1
        assert result[0].title == "Good"

    def test_preserves_min_keep(self) -> None:
        """When too few non-empty results, keeps min_keep items."""
        f = SearchResultFilter(min_keep=3)
        items = [
            _item(title="E1", content=""),
            _item(title="E2", content=""),
            _item(title="OK", content="content"),
        ]
        result = f._remove_empty_content(items)
        assert len(result) == 3  # min_keep honored

    def test_all_have_content(self) -> None:
        """When all have content, nothing is removed."""
        f = SearchResultFilter(min_keep=1)
        items = [_item(content="a"), _item(content="b")]
        result = f._remove_empty_content(items)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Relevance filtering (n-gram similarity)
# ---------------------------------------------------------------------------


class TestRelevanceFiltering:
    """N-gram similarity-based noise filtering."""

    def test_relevant_results_kept(self) -> None:
        """Results with high similarity to query are kept."""
        f = SearchResultFilter(min_keep=1, min_similarity=0.05)
        items = [
            _item(title="LangGraph Memory Guide", content="How to manage memory in LangGraph"),
            _item(title="Python Memory Pool", content="Python memory allocation internals"),
        ]
        result = f._filter_by_relevance(items, "LangGraph memory management")
        assert len(result) >= 1
        assert result[0].title == "LangGraph Memory Guide"

    def test_noise_removed(self) -> None:
        """Results with very low similarity are removed."""
        f = SearchResultFilter(min_keep=1, min_similarity=0.05)
        items = [
            _item(
                title="LangGraph Memory Guide",
                content="Manage memory in LangGraph checkpoints",
            ),
            _item(
                title="Win10蓝屏修复教程",
                content="如何修复Windows蓝屏 memory management错误代码",
            ),
            _item(
                title="烹饪技巧大全",
                content="如何制作美味的红烧肉",
            ),
        ]
        result = f._filter_by_relevance(items, "LangGraph memory management")
        # The cooking recipe should be filtered out
        titles = [r.title for r in result]
        assert "烹饪技巧大全" not in titles

    def test_site_operator_stripped(self) -> None:
        """site: operators in query don't affect relevance scoring."""
        f = SearchResultFilter(min_keep=1, min_similarity=0.01)
        items = [
            _item(title="AI Agent on Zhihu", content="AI Agent discussion"),
        ]
        result = f._filter_by_relevance(
            items, "site:zhihu.com AI Agent",
        )
        assert len(result) == 1

    def test_chinese_query_relevance(self) -> None:
        """Chinese queries use CJK unigrams for similarity."""
        f = SearchResultFilter(min_keep=1, min_similarity=0.05)
        items = [
            _item(title="人工智能最新进展", content="大模型技术突破"),
            _item(title="Cooking Tips", content="How to make pasta"),
        ]
        result = f._filter_by_relevance(items, "人工智能 大模型 最新进展")
        assert len(result) >= 1
        assert result[0].title == "人工智能最新进展"

    def test_min_keep_respected(self) -> None:
        """Even if all results are below threshold, min_keep are kept."""
        f = SearchResultFilter(min_keep=2, min_similarity=0.99)
        items = [
            _item(title="Foo", content="bar"),
            _item(title="Baz", content="qux"),
            _item(title="Hello", content="world"),
        ]
        result = f._filter_by_relevance(items, "completely unrelated query")
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# N-gram helpers
# ---------------------------------------------------------------------------


class TestNgramHelpers:
    """Character n-gram extraction and cosine similarity."""

    def test_char_ngrams_english(self) -> None:
        """English text produces character bigrams."""
        f = SearchResultFilter(ngram_size=2)
        ngrams = f._char_ngrams("hello")
        assert "he" in ngrams
        assert "el" in ngrams
        assert "ll" in ngrams
        assert "lo" in ngrams

    def test_char_ngrams_chinese(self) -> None:
        """Chinese characters produce unigrams."""
        f = SearchResultFilter(ngram_size=2)
        ngrams = f._char_ngrams("人工智能")
        # Individual CJK characters as unigrams
        assert "人" in ngrams
        assert "工" in ngrams
        assert "智" in ngrams
        assert "能" in ngrams

    def test_cosine_similarity_identical(self) -> None:
        """Identical inputs produce similarity of 1.0."""
        from collections import Counter
        a = Counter({"he": 1, "el": 1, "lo": 1})
        sim = SearchResultFilter._cosine_similarity(a, a)
        assert abs(sim - 1.0) < 0.001

    def test_cosine_similarity_disjoint(self) -> None:
        """Disjoint inputs produce similarity of 0.0."""
        from collections import Counter
        a = Counter({"ab": 1, "bc": 1})
        b = Counter({"xy": 1, "yz": 1})
        sim = SearchResultFilter._cosine_similarity(a, b)
        assert sim == 0.0

    def test_cosine_similarity_empty(self) -> None:
        """Empty counter produces 0.0."""
        from collections import Counter
        assert SearchResultFilter._cosine_similarity(Counter(), Counter({"a": 1})) == 0.0


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------


class TestUrlNormalization:
    """URL normalization for deduplication."""

    def test_strips_query_params(self) -> None:
        norm = SearchResultFilter._normalize_url(
            "https://example.com/page?ref=twitter&utm_source=google",
        )
        assert norm == "https://example.com/page"

    def test_strips_fragment(self) -> None:
        norm = SearchResultFilter._normalize_url(
            "https://example.com/page#section",
        )
        assert norm == "https://example.com/page"

    def test_strips_trailing_slash(self) -> None:
        norm = SearchResultFilter._normalize_url(
            "https://example.com/page/",
        )
        assert norm == "https://example.com/page"


# ---------------------------------------------------------------------------
# Full filter chain
# ---------------------------------------------------------------------------


class TestFullFilterChain:
    """End-to-end filter chain tests."""

    def test_filter_chain_integration(self) -> None:
        """Full chain: dedup → empty removal → relevance."""
        f = SearchResultFilter(min_keep=1, min_similarity=0.05)
        items = [
            _item(
                title="LangGraph Guide",
                url="https://a.com/guide",
                content="LangGraph memory management tutorial",
                score=0.9,
            ),
            _item(
                title="LangGraph Guide (dup)",
                url="https://a.com/guide?ref=bing",
                content="Same guide",
                score=0.3,
            ),
            _item(
                title="Empty Result",
                url="https://b.com",
                content="",
            ),
            _item(
                title="Unrelated Recipe",
                url="https://c.com/recipe",
                content="How to bake a chocolate cake with frosting",
            ),
        ]
        result = f.filter(items, "LangGraph memory management")
        # Dup removed, empty removed, recipe likely removed
        assert len(result) <= 3
        assert result[0].title == "LangGraph Guide"
        assert result[0].score == 0.9  # Higher score dup kept

    def test_strip_site_operator(self) -> None:
        """site: prefix removed from query for relevance scoring."""
        assert SearchResultFilter._strip_site_operator(
            "site:zhihu.com AI Agent",
        ) == "AI Agent"
        assert SearchResultFilter._strip_site_operator(
            "site:a.com OR site:b.com query",
        ) == "OR  query"
