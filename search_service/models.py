"""Pydantic data models for the search service.

Defines the unified data contract shared by all backends:
SearchEngine enum, SearchResultItem, SearchResponse, ScrapeResponse.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SearchEngine(str, Enum):
    """Supported search engine identifiers.

    ``str`` mixin ensures JSON serialization outputs plain string values
    (e.g. ``"baidu"`` instead of ``"SearchEngine.BAIDU"``).
    """

    BAIDU = "baidu"
    SOGOU = "sogou"
    SO360 = "360search"
    BING = "bing"
    SEARXNG = "searxng"
    ZHIHU = "zhihu"
    WEIBO = "weibo"
    WEIXIN = "weixin"
    GITHUB = "github"


class SearchResultItem(BaseModel):
    """Single search result, aligned with Tavily response format.

    Attributes:
        title: Result page title.
        url: Result page URL.
        content: Snippet or body excerpt.
        source_engine: Which engine produced this result.
        published_date: ISO date string if available.
        score: Relevance score normalized to [0, 1].
        raw_content: Full-page Markdown content (optional).
        metadata: Platform-specific extension metadata.
    """

    title: str
    url: str
    content: str = Field(description="Snippet or body excerpt")
    source_engine: SearchEngine
    published_date: Optional[str] = None
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    raw_content: Optional[str] = Field(
        default=None, description="Full-page Markdown content",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Platform-specific extension metadata",
    )


class SearchResponse(BaseModel):
    """Unified search response returned by all backends.

    Attributes:
        query: Original search query.
        results: List of search result items.
        result_count: Number of results returned.
        search_time_ms: Backend search latency in milliseconds.
        engines_used: List of engine names that contributed results.
    """

    query: str
    results: list[SearchResultItem]
    result_count: int
    search_time_ms: int
    engines_used: list[str]

    @classmethod
    def empty(cls, query: str, engines: list[str]) -> SearchResponse:
        """Factory for an empty response (no results found).

        Args:
            query: The original query.
            engines: Engine names that were tried.

        Returns:
            A SearchResponse with zero results.
        """
        return cls(
            query=query,
            results=[],
            result_count=0,
            search_time_ms=0,
            engines_used=engines,
        )


class ScrapeResponse(BaseModel):
    """Response from page content extraction (URL → Markdown).

    Attributes:
        url: The scraped URL.
        title: Page title.
        content: Extracted content in Markdown format.
        content_length: Character count of ``content``.
        metadata: Additional extraction metadata.
    """

    url: str
    title: str
    content: str
    content_length: int
    metadata: dict = Field(default_factory=dict)
