"""Tests for search_service.models — Pydantic data models."""

import pytest
from pydantic import ValidationError

from search_service.models import (
    SearchEngine,
    SearchResultItem,
    SearchResponse,
    ScrapeResponse,
)


class TestSearchEngine:
    """SearchEngine enum serialization and behavior."""

    def test_engine_json_serialization(self) -> None:
        """SearchEngine.BAIDU serializes to plain string 'baidu'."""
        assert SearchEngine.BAIDU == "baidu"
        assert SearchEngine.BAIDU.value == "baidu"

    def test_all_engines_are_strings(self) -> None:
        """Every member is a valid string (str, Enum mixin)."""
        for engine in SearchEngine:
            assert isinstance(engine, str)
            assert isinstance(engine.value, str)

    def test_engine_from_value(self) -> None:
        """Construct engine from string value."""
        assert SearchEngine("zhihu") is SearchEngine.ZHIHU

    def test_invalid_engine_raises(self) -> None:
        """Unknown engine string raises ValueError."""
        with pytest.raises(ValueError):
            SearchEngine("nonexistent_engine")


class TestSearchResultItem:
    """SearchResultItem Pydantic model validation."""

    def test_search_result_item_valid(self) -> None:
        """Minimal valid item passes validation."""
        item = SearchResultItem(
            title="Test Result",
            url="https://example.com",
            content="This is a test snippet.",
            source_engine=SearchEngine.BAIDU,
        )
        assert item.title == "Test Result"
        assert item.url == "https://example.com"
        assert item.source_engine == SearchEngine.BAIDU
        assert item.published_date is None
        assert item.score is None
        assert item.raw_content is None
        assert item.metadata == {}

    def test_score_range_validation_above(self) -> None:
        """Score > 1.0 triggers ValidationError."""
        with pytest.raises(ValidationError):
            SearchResultItem(
                title="T", url="https://x.com", content="C",
                source_engine=SearchEngine.BING, score=1.5,
            )

    def test_score_range_validation_below(self) -> None:
        """Score < 0.0 triggers ValidationError."""
        with pytest.raises(ValidationError):
            SearchResultItem(
                title="T", url="https://x.com", content="C",
                source_engine=SearchEngine.BING, score=-0.1,
            )

    def test_score_boundary_values(self) -> None:
        """Score at exact boundaries (0.0 and 1.0) are valid."""
        item_zero = SearchResultItem(
            title="T", url="https://x.com", content="C",
            source_engine=SearchEngine.BING, score=0.0,
        )
        item_one = SearchResultItem(
            title="T", url="https://x.com", content="C",
            source_engine=SearchEngine.BING, score=1.0,
        )
        assert item_zero.score == 0.0
        assert item_one.score == 1.0

    def test_metadata_default_factory(self) -> None:
        """Each instance gets its own metadata dict (no shared reference)."""
        a = SearchResultItem(
            title="A", url="https://a.com", content="A",
            source_engine=SearchEngine.SOGOU,
        )
        b = SearchResultItem(
            title="B", url="https://b.com", content="B",
            source_engine=SearchEngine.SOGOU,
        )
        a.metadata["key"] = "value"
        assert "key" not in b.metadata


class TestSearchResponse:
    """SearchResponse model and factory method."""

    def test_search_response_construction(self) -> None:
        """Normal construction with results."""
        item = SearchResultItem(
            title="T", url="https://x.com", content="C",
            source_engine=SearchEngine.BAIDU,
        )
        resp = SearchResponse(
            query="test", results=[item],
            result_count=1, search_time_ms=150,
            engines_used=["baidu"],
        )
        assert resp.query == "test"
        assert resp.result_count == 1
        assert len(resp.results) == 1

    def test_search_response_empty_factory(self) -> None:
        """SearchResponse.empty() returns zero-result response."""
        resp = SearchResponse.empty("hello", ["baidu", "sogou"])
        assert resp.query == "hello"
        assert resp.results == []
        assert resp.result_count == 0
        assert resp.search_time_ms == 0
        assert resp.engines_used == ["baidu", "sogou"]


class TestScrapeResponse:
    """ScrapeResponse model."""

    def test_scrape_response_construction(self) -> None:
        resp = ScrapeResponse(
            url="https://example.com",
            title="Example",
            content="# Hello\nWorld",
            content_length=13,
        )
        assert resp.url == "https://example.com"
        assert resp.content_length == 13
        assert resp.metadata == {}
