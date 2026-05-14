"""Search Quality Report — human-readable quality monitoring tool.

Runs a suite of search quality test cases against a live SearXNG instance
and produces a structured quality report with optional per-result detail
for manual inspection.

Usage:
    python examples/search_quality_report.py
    python examples/search_quality_report.py --show-results
    python examples/search_quality_report.py --query "custom query" -s
    python examples/search_quality_report.py --verbose
"""

from __future__ import annotations

import argparse
import asyncio

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx


# ---------------------------------------------------------------------------
# Quality test case definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchQualityCase:
    """Single search quality test case.

    Attributes:
        name: Human-readable case name.
        query: Search query to execute.
        language: Expected primary language ("en" or "zh").
        min_results: Minimum acceptable result count.
        relevance_keywords: Keywords to check in title+content for P@5.
        site_domain: If set, check that results contain this domain.
    """

    name: str
    query: str
    language: str
    min_results: int
    relevance_keywords: list[str] = field(default_factory=list)
    site_domain: Optional[str] = None


QUALITY_CASES: list[SearchQualityCase] = [
    # --- English Technical ---
    SearchQualityCase(
        name="en_langgraph_memory",
        query="LangGraph memory management",
        language="en", min_results=5,
        relevance_keywords=["langgraph", "memory", "checkpoint", "persistence"],
    ),
    SearchQualityCase(
        name="en_rag_finetuning",
        query="RAG vs fine-tuning comparison 2025",
        language="en", min_results=5,
        relevance_keywords=["rag", "fine-tun", "retrieval", "augment"],
    ),
    SearchQualityCase(
        name="en_multi_agent",
        query="multi-agent orchestration framework",
        language="en", min_results=5,
        relevance_keywords=["agent", "multi", "orchestrat", "framework"],
    ),
    # --- Chinese Technical ---
    SearchQualityCase(
        name="zh_ai_progress",
        query="人工智能 大模型 最新进展 2026",
        language="zh", min_results=5,
        relevance_keywords=["大模型", "AI", "人工智能", "模型"],
    ),
    SearchQualityCase(
        name="zh_langchain_rag",
        query="LangChain 向量数据库 RAG 教程",
        language="zh", min_results=3,
        relevance_keywords=["langchain", "向量", "rag", "数据库"],
    ),
    # --- site: operator ---
    SearchQualityCase(
        name="site_zhihu",
        query="site:zhihu.com AI Agent 框架",
        language="zh", min_results=3,
        relevance_keywords=["agent"],
        site_domain="zhihu.com",
    ),
    # --- Additional ---
    SearchQualityCase(
        name="en_python_asyncio",
        query="Python asyncio best practices 2025",
        language="en", min_results=5,
        relevance_keywords=["asyncio", "python", "async"],
    ),
    SearchQualityCase(
        name="zh_kubernetes",
        query="Kubernetes 容器编排 最佳实践",
        language="zh", min_results=3,
        relevance_keywords=["kubernetes", "容器", "k8s", "编排"],
    ),
]


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

@dataclass
class CaseResult:
    """Quality metrics for a single test case.

    Attributes:
        case: The test case definition.
        result_count: Number of results returned.
        content_fill_rate: Fraction of results with non-empty content.
        engine_count: Number of distinct engines that contributed.
        engines: Set of engine names.
        relevance_p5: Precision@5 — fraction of top-5 results matching keywords.
        site_hit_rate: For site: queries, fraction of URLs containing target domain.
        latency_ms: End-to-end search latency in milliseconds.
        passed: Whether the case met all quality thresholds.
        failures: List of threshold names that failed.
    """

    case: SearchQualityCase
    result_count: int = 0
    content_fill_rate: float = 0.0
    engine_count: int = 0
    engines: set[str] = field(default_factory=set)
    unresponsive_engines: list[str] = field(default_factory=list)
    relevance_p5: float = 0.0
    site_hit_rate: float = 0.0
    latency_ms: int = 0
    passed: bool = True
    failures: list[str] = field(default_factory=list)
    raw_results: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MIN_CONTENT_FILL_RATE = 0.85
MIN_RELEVANCE_P5 = 0.60
MIN_ENGINE_COUNT = 1  # Relaxed — some queries only hit 1 engine
MIN_SITE_HIT_RATE = 0.80
MAX_LATENCY_MS = 5000


# ---------------------------------------------------------------------------
# Quality evaluator
# ---------------------------------------------------------------------------

class SearchQualityEvaluator:
    """Evaluates search result quality against predefined thresholds.

    Args:
        searxng_url: Base URL of the SearXNG instance.
    """

    def __init__(self, searxng_url: str = "http://localhost:8080") -> None:
        self._base_url = searxng_url.rstrip("/")
        self._logger = logging.getLogger(__name__)

    async def run_all(
        self,
        cases: list[SearchQualityCase],
        custom_query: Optional[str] = None,
    ) -> list[CaseResult]:
        """Run all quality test cases.

        Args:
            cases: List of test cases to execute.
            custom_query: If provided, run only this single query.

        Returns:
            List of CaseResult with metrics.
        """
        if custom_query:
            cases = [
                SearchQualityCase(
                    name="custom",
                    query=custom_query,
                    language="auto",
                    min_results=3,
                    relevance_keywords=[],
                ),
            ]

        results: list[CaseResult] = []
        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(10.0),
        ) as client:
            for case in cases:
                result = await self._evaluate_case(client, case)
                results.append(result)

        return results

    async def _evaluate_case(
        self, client: httpx.AsyncClient, case: SearchQualityCase,
    ) -> CaseResult:
        """Evaluate a single test case.

        Args:
            client: HTTP client.
            case: Test case to evaluate.

        Returns:
            CaseResult with populated metrics.
        """
        cr = CaseResult(case=case)

        start = time.monotonic()
        try:
            resp = await client.get("/search", params={
                "q": case.query,
                "format": "json",
                "pageno": 1,
            })
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            self._logger.error(
                "search_quality_request_failed",
                extra={"case": case.name, "error": str(exc)},
            )
            cr.passed = False
            cr.failures.append(f"HTTP error: {exc}")
            return cr

        cr.latency_ms = int((time.monotonic() - start) * 1000)
        raw_results = data.get("results", [])
        cr.raw_results = raw_results
        cr.unresponsive_engines = [
            e[0] for e in data.get("unresponsive_engines", [])
        ]

        # --- Metrics ---
        cr.result_count = len(raw_results)
        cr.engines = {r.get("engine", "?") for r in raw_results}
        cr.engine_count = len(cr.engines)

        # Content fill rate
        non_empty = sum(
            1 for r in raw_results if r.get("content", "").strip()
        )
        cr.content_fill_rate = (
            non_empty / cr.result_count if cr.result_count else 0.0
        )

        # Relevance P@5
        if case.relevance_keywords:
            top5 = raw_results[:5]
            hits = 0
            for r in top5:
                text = (
                    r.get("title", "") + " " + r.get("content", "")
                ).lower()
                if any(kw.lower() in text for kw in case.relevance_keywords):
                    hits += 1
            cr.relevance_p5 = hits / max(1, len(top5))
        else:
            cr.relevance_p5 = 1.0  # No keywords = skip check

        # Site hit rate
        if case.site_domain:
            site_hits = sum(
                1 for r in raw_results
                if case.site_domain in r.get("url", "")
            )
            cr.site_hit_rate = (
                site_hits / cr.result_count if cr.result_count else 0.0
            )

        # --- Threshold checks ---
        if cr.result_count < case.min_results:
            cr.passed = False
            cr.failures.append(
                f"result_count {cr.result_count} < {case.min_results}",
            )

        if cr.content_fill_rate < MIN_CONTENT_FILL_RATE:
            cr.failures.append(
                f"content_fill_rate {cr.content_fill_rate:.0%}"
                f" < {MIN_CONTENT_FILL_RATE:.0%}",
            )
            # Warn but don't fail — sogou often returns empty content
            self._logger.warning(
                "low_content_fill_rate",
                extra={
                    "case": case.name,
                    "rate": f"{cr.content_fill_rate:.0%}",
                },
            )

        if case.relevance_keywords and cr.relevance_p5 < MIN_RELEVANCE_P5:
            cr.passed = False
            cr.failures.append(
                f"relevance_p5 {cr.relevance_p5:.0%}"
                f" < {MIN_RELEVANCE_P5:.0%}",
            )

        if (
            case.site_domain
            and cr.site_hit_rate < MIN_SITE_HIT_RATE
        ):
            cr.passed = False
            cr.failures.append(
                f"site_hit_rate {cr.site_hit_rate:.0%}"
                f" < {MIN_SITE_HIT_RATE:.0%}",
            )

        if cr.latency_ms > MAX_LATENCY_MS:
            cr.failures.append(f"latency {cr.latency_ms}ms > {MAX_LATENCY_MS}ms")

        self._logger.info(
            "search_quality_case",
            extra={
                "case": case.name,
                "results": cr.result_count,
                "fill_rate": f"{cr.content_fill_rate:.0%}",
                "engines": cr.engine_count,
                "p5": f"{cr.relevance_p5:.0%}",
                "latency_ms": cr.latency_ms,
                "passed": cr.passed,
            },
        )

        return cr


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def print_report(
    results: list[CaseResult],
    show_results: bool = False,
) -> None:
    """Print a human-readable quality report.

    Args:
        results: List of evaluated CaseResult objects.
        show_results: If True, print each individual search result
            for human review.
    """
    print("\n" + "=" * 70)
    print("  Search Quality Report")
    print("=" * 70)

    pass_count = 0
    warn_count = 0
    fail_count = 0

    for cr in results:
        status = "✅ PASS" if cr.passed else "❌ FAIL"
        if cr.passed and cr.failures:
            status = "⚠️  WARN"
            warn_count += 1
        elif cr.passed:
            pass_count += 1
        else:
            fail_count += 1

        print(f"\n  [{status}] {cr.case.name}")
        print(f"    Query: \"{cr.case.query}\"")
        print(
            f"    Results: {cr.result_count:>3}"
            f"  | Fill: {cr.content_fill_rate:>5.0%}"
            f"  | Engines: {sorted(cr.engines)}"
            f"  | Latency: {cr.latency_ms}ms",
        )

        if cr.unresponsive_engines:
            print(f"    ⚠ Unresponsive: {cr.unresponsive_engines}")

        if cr.case.relevance_keywords:
            print(f"    Relevance P@5: {cr.relevance_p5:.0%}")

        if cr.case.site_domain:
            print(f"    Site Hit Rate: {cr.site_hit_rate:.0%}")

        if cr.failures:
            for f in cr.failures:
                print(f"    ⚠ {f}")

        # --- Detailed per-result output for human review ---
        if show_results and cr.raw_results:
            print(f"    {'─' * 58}")
            for idx, r in enumerate(cr.raw_results[:10]):
                engine = r.get("engine", "?")
                score = r.get("score", "N/A")
                title = r.get("title", "")[:70]
                url = r.get("url", "")[:90]
                content = r.get("content", "").strip()
                snippet = content[:160] if content else "⚠️ EMPTY"

                # Mark relevance hit
                text = (r.get("title", "") + " " + content).lower()
                hit = any(
                    kw.lower() in text
                    for kw in cr.case.relevance_keywords
                ) if cr.case.relevance_keywords else True
                marker = "✓" if hit else "✗"

                print(f"    [{idx + 1:>2}] {marker} ({engine}, score={score})")
                print(f"         {title}")
                print(f"         {url}")
                print(f"         {snippet}")

    total = len(results)
    print("\n" + "-" * 70)
    print(
        f"  Overall: {pass_count} PASS, {warn_count} WARN,"
        f" {fail_count} FAIL / {total} total",
    )
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(log_level: str) -> None:
    """Configure logging to console + file.

    Args:
        log_level: Log level string (DEBUG, INFO, etc.).
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(
        "examples/search_quality_report.log", mode="w",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)


def _resolve_log_level(args: argparse.Namespace) -> str:
    """Determine log level from CLI args and env.

    Priority: --log-level > --verbose > env > default (INFO).

    Args:
        args: Parsed CLI arguments.

    Returns:
        Log level string.
    """
    if args.log_level:
        return args.log_level
    if args.verbose:
        return "DEBUG"
    return os.environ.get("RESEARCH_ASSISTANT_LOG_LEVEL", "INFO")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _main(args: argparse.Namespace) -> int:
    """Main async entrypoint.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 = all pass, 1 = failures).
    """
    evaluator = SearchQualityEvaluator(
        searxng_url=args.searxng_url,
    )
    results = await evaluator.run_all(
        QUALITY_CASES, custom_query=args.query,
    )
    print_report(results, show_results=args.show_results)

    failed = sum(1 for r in results if not r.passed)
    return 1 if failed else 0


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Search Quality Monitoring Script",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Run a single custom query instead of the full suite.",
    )
    parser.add_argument(
        "--show-results", "-s", action="store_true",
        help="Print each search result for manual review.",
    )
    parser.add_argument(
        "--searxng-url", type=str, default="http://localhost:8080",
        help="SearXNG instance URL.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )
    parser.add_argument(
        "--log-level", type=str, default=None,
        help="Log level (DEBUG, INFO, WARNING, ERROR).",
    )
    args = parser.parse_args()

    log_level = _resolve_log_level(args)
    _setup_logging(log_level)

    exit_code = asyncio.run(_main(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
