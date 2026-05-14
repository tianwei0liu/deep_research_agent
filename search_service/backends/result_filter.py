"""Search result post-processing filters.

Applies a filter chain to raw search results:
1. URL deduplication (normalize → keep highest score).
2. Empty content removal (preserve at least ``min_keep``).
3. Relevance filtering via character n-gram similarity.

The n-gram similarity approach is pure-Python, requires no external
dependencies (no numpy/sklearn/API keys), and handles both Chinese
and English text natively (CJK characters treated as unigrams).
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Optional
from urllib.parse import urlparse

from search_service.models import SearchResultItem


class SearchResultFilter:
    """Post-processing filter chain for search results.

    Applies deduplication, empty-content removal, and relevance-based
    noise filtering in a single ``filter()`` call.

    Args:
        min_keep: Minimum number of results to preserve even if
            filters would remove them.
        min_similarity: Minimum n-gram similarity threshold for
            a result to be considered "relevant". Results below
            this threshold are candidates for removal.
        ngram_size: Size of character n-grams for similarity.
    """

    def __init__(
        self,
        min_keep: int = 3,
        min_similarity: float = 0.05,
        ngram_size: int = 2,
    ) -> None:
        self._min_keep = min_keep
        self._min_similarity = min_similarity
        self._ngram_size = ngram_size
        self._logger = logging.getLogger(__name__)

    def filter(
        self,
        results: list[SearchResultItem],
        query: str,
    ) -> list[SearchResultItem]:
        """Apply the full filter chain.

        Order: dedup → empty removal → relevance filtering.

        Args:
            results: Raw search result items.
            query: Original search query (used for relevance scoring).

        Returns:
            Filtered list of SearchResultItem.
        """
        before = len(results)

        results = self._deduplicate(results)
        dedup_count = before - len(results)

        results = self._remove_empty_content(results)
        empty_removed = before - dedup_count - len(results)

        results = self._filter_by_relevance(results, query)
        noise_removed = (
            before - dedup_count - empty_removed - len(results)
        )

        if dedup_count or empty_removed or noise_removed:
            self._logger.info(
                "search_result_filter",
                extra={
                    "before": before,
                    "after": len(results),
                    "dedup_removed": dedup_count,
                    "empty_removed": empty_removed,
                    "noise_removed": noise_removed,
                },
            )

        return results

    # ------------------------------------------------------------------
    # Filter stages
    # ------------------------------------------------------------------

    def _deduplicate(
        self, results: list[SearchResultItem],
    ) -> list[SearchResultItem]:
        """Remove duplicate URLs, keeping the result with the highest score.

        URL normalization: strip query params, fragment, trailing slash,
        and lowercase the host.

        Args:
            results: Input result list.

        Returns:
            Deduplicated result list preserving original order.
        """
        seen: dict[str, int] = {}  # normalized_url → index in output
        output: list[SearchResultItem] = []

        for item in results:
            norm = self._normalize_url(item.url)
            if norm in seen:
                # Keep the one with higher score
                idx = seen[norm]
                existing_score = output[idx].score or 0.0
                new_score = item.score or 0.0
                if new_score > existing_score:
                    output[idx] = item
            else:
                seen[norm] = len(output)
                output.append(item)

        return output

    def _remove_empty_content(
        self, results: list[SearchResultItem],
    ) -> list[SearchResultItem]:
        """Remove results with empty content fields.

        Preserves at least ``min_keep`` results to avoid returning nothing.

        Args:
            results: Input result list.

        Returns:
            Filtered result list.
        """
        non_empty = [r for r in results if r.content.strip()]

        if len(non_empty) >= self._min_keep:
            return non_empty

        # Not enough non-empty results — keep all to meet min_keep
        return results[:max(self._min_keep, len(non_empty))]

    def _filter_by_relevance(
        self, results: list[SearchResultItem],
        query: str,
    ) -> list[SearchResultItem]:
        """Remove results with very low relevance to the query.

        Uses character n-gram cosine similarity between the query
        and each result's title+content. Results below
        ``min_similarity`` are removed, subject to ``min_keep``.

        The query is preprocessed to strip ``site:`` operators
        since they shouldn't factor into content relevance.

        Args:
            results: Input result list.
            query: Original search query.

        Returns:
            Relevance-filtered result list.
        """
        clean_query = self._strip_site_operator(query)
        if not clean_query.strip():
            return results

        query_ngrams = self._char_ngrams(clean_query)
        if not query_ngrams:
            return results

        scored: list[tuple[SearchResultItem, float]] = []
        for item in results:
            doc_text = f"{item.title} {item.content}"
            doc_ngrams = self._char_ngrams(doc_text)
            sim = self._cosine_similarity(query_ngrams, doc_ngrams)
            scored.append((item, sim))

        # Keep results above threshold
        relevant = [
            (item, sim)
            for item, sim in scored
            if sim >= self._min_similarity
        ]

        # Ensure min_keep
        if len(relevant) < self._min_keep:
            # Fall back to top-N by similarity
            scored.sort(key=lambda x: x[1], reverse=True)
            relevant = scored[:self._min_keep]

        # Log removed noise
        removed = [
            (item, sim) for item, sim in scored
            if sim < self._min_similarity
        ]
        for item, sim in removed:
            if (item, sim) not in relevant:
                self._logger.debug(
                    "noise_filtered",
                    extra={
                        "title": item.title[:60],
                        "similarity": f"{sim:.3f}",
                        "threshold": f"{self._min_similarity:.3f}",
                    },
                )

        # Return in original order
        relevant_items = {id(item) for item, _ in relevant}
        return [r for r in results if id(r) in relevant_items]

    # ------------------------------------------------------------------
    # Character n-gram similarity (pure Python, CJK-safe)
    # ------------------------------------------------------------------

    def _char_ngrams(self, text: str) -> Counter[str]:
        """Extract character n-grams from text.

        For CJK characters (Chinese/Japanese/Korean), each character
        is treated as a unigram in addition to n-grams, ensuring that
        single Chinese characters contribute to similarity.

        Args:
            text: Input text.

        Returns:
            Counter of n-gram frequencies.
        """
        text = text.lower().strip()
        ngrams: Counter[str] = Counter()

        # Add CJK unigrams
        for ch in text:
            if self._is_cjk(ch):
                ngrams[ch] += 1

        # Add character n-grams
        n = self._ngram_size
        for i in range(len(text) - n + 1):
            gram = text[i:i + n]
            if gram.strip():  # Skip whitespace-only grams
                ngrams[gram] += 1

        return ngrams

    @staticmethod
    def _cosine_similarity(
        a: Counter[str], b: Counter[str],
    ) -> float:
        """Compute cosine similarity between two n-gram counters.

        Args:
            a: First counter.
            b: Second counter.

        Returns:
            Cosine similarity in [0, 1].
        """
        if not a or not b:
            return 0.0

        # Dot product
        intersection = set(a.keys()) & set(b.keys())
        dot = sum(a[k] * b[k] for k in intersection)

        # Magnitudes
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot / (mag_a * mag_b)

    @staticmethod
    def _is_cjk(char: str) -> bool:
        """Check if a character is CJK (Chinese/Japanese/Korean).

        Args:
            char: Single character.

        Returns:
            True if CJK.
        """
        cp = ord(char)
        return (
            (0x4E00 <= cp <= 0x9FFF)      # CJK Unified
            or (0x3400 <= cp <= 0x4DBF)    # CJK Extension A
            or (0xF900 <= cp <= 0xFAFF)    # CJK Compatibility
            or (0x20000 <= cp <= 0x2A6DF)  # CJK Extension B
        )

    # ------------------------------------------------------------------
    # URL normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize a URL for deduplication.

        Strips query params, fragment, trailing slash, lowercases host.

        Args:
            url: Raw URL string.

        Returns:
            Normalized URL string.
        """
        try:
            parsed = urlparse(url)
            host = parsed.hostname or ""
            path = parsed.path.rstrip("/")
            return f"{parsed.scheme}://{host}{path}"
        except Exception:
            return url.split("?")[0].split("#")[0].rstrip("/")

    @staticmethod
    def _strip_site_operator(query: str) -> str:
        """Remove site: operators from a query.

        Args:
            query: Search query possibly containing site: prefixes.

        Returns:
            Query with site: operators removed.
        """
        return re.sub(r"site:\S+", "", query).strip()
