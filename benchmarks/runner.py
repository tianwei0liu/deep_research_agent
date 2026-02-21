"""Benchmark runner: executes benchmark cases through the orchestrator and collects results.

Loads cases from JSON, runs each through the full OrchestratorGraph pipeline,
captures operational metrics, and delegates scoring to the BenchmarkGrader.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from research_assistant.agents.orchestrator.graph import OrchestratorGraph
from research_assistant.agents.orchestrator.state import OrchestratorConfig
from research_assistant.agents.worker.schemas import Limits
from research_assistant.benchmarks.benchmark_case import BenchmarkCase, EvaluationResult
from research_assistant.benchmarks.grader import BenchmarkGrader
from research_assistant.config import Settings


# Default path to the core benchmark dataset, relative to the package directory.
_DEFAULT_DATASET_PATH = Path(__file__).parent / "datasets" / "core.json"


class BenchmarkRunner:
    """Runs benchmark cases through the research agent and evaluates results.

    Args:
        settings: Application settings (API keys, model config).
        dataset_path: Path to the benchmark JSON file. Defaults to datasets/core.json.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        dataset_path: Optional[Path] = None,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self._settings = settings or Settings.load()
        self._dataset_path = dataset_path or _DEFAULT_DATASET_PATH
        self._grader = BenchmarkGrader(self._settings)

    def load_cases(
        self,
        case_ids: Optional[List[str]] = None,
        category: Optional[str] = None,
    ) -> List[BenchmarkCase]:
        """Load benchmark cases from the dataset JSON.

        Args:
            case_ids: If provided, only load cases with these IDs.
            category: If provided, only load cases in this category.

        Returns:
            List of BenchmarkCase objects.

        Raises:
            FileNotFoundError: If the dataset file does not exist.
        """
        if not self._dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self._dataset_path}")

        with open(self._dataset_path, encoding="utf-8") as f:
            raw_cases = json.load(f)

        cases = [BenchmarkCase(**c) for c in raw_cases]

        if case_ids:
            cases = [c for c in cases if c.id in case_ids]
        if category:
            cases = [c for c in cases if c.category == category]

        self.logger.info(
            "Loaded %d benchmark case(s) (filter: ids=%s, category=%s)",
            len(cases),
            case_ids,
            category,
        )
        return cases

    async def run_case(self, case: BenchmarkCase) -> EvaluationResult:
        """Execute a single benchmark case end-to-end.

        Runs the orchestrator, captures metrics, grades the output.

        Args:
            case: The benchmark case to execute.

        Returns:
            EvaluationResult with scores and metadata.
        """
        self.logger.info("=== Running case: %s (%s) ===", case.id, case.category)

        # Build orchestrator
        orchestrator = OrchestratorGraph()
        app = orchestrator.compile()

        config = OrchestratorConfig(
            messages=[HumanMessage(content=case.query)],
            max_parallel_workers=self._settings.default_max_parallel_workers,
            recursion_limit=self._settings.default_recursion_limit,
            worker_limits=Limits(
                max_tool_calls=self._settings.default_worker_max_tool_calls,
                max_turns=self._settings.default_worker_max_turns,
                max_output_tokens=self._settings.default_worker_max_output_tokens,
            ),
        )
        initial_state = config.to_state()
        recursion_limit = initial_state.get("recursion_limit", self._settings.default_recursion_limit)

        # Execute and capture metrics
        start_time = time.monotonic()
        final_report = ""
        todos: list = []
        turn_count = 0
        tool_call_count = 0

        try:
            async for event in app.astream(
                initial_state,
                config={"recursion_limit": recursion_limit},
            ):
                for key, value in event.items():
                    turn_count += 1

                    if "messages" in value:
                        last_msg = value["messages"][-1]
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            tool_call_count += len(last_msg.tool_calls)

                    if "todos" in value:
                        todos = value["todos"]

                    if "final_report" in value and value["final_report"]:
                        final_report = value["final_report"]

        except Exception as exc:
            self.logger.exception("Orchestrator execution failed for case %s", case.id)
            final_report = f"EXECUTION ERROR: {exc}"

        elapsed = time.monotonic() - start_time
        task_objectives = [t.objective for t in todos]

        self.logger.info(
            "Case %s completed in %.1fs — %d turns, %d tool calls, %d tasks",
            case.id,
            elapsed,
            turn_count,
            tool_call_count,
            len(todos),
        )

        # Grade the report
        try:
            scores = await self._grader.grade(
                query=case.query,
                expected_facets=case.expected_facets,
                final_report=final_report,
                task_objectives=task_objectives,
                reference_answer=case.reference_answer,
            )
        except Exception as exc:
            self.logger.error("Grading failed for case %s: %s", case.id, exc)
            scores = {
                "correctness_score": 0.0,
                "completeness_score": 0.0,
                "citation_score": 0.0,
                "decomposition_score": 0.0,
                "reasoning": f"Grading error: {exc}",
            }

        return EvaluationResult(
            case_id=case.id,
            correctness_score=scores["correctness_score"],
            completeness_score=scores["completeness_score"],
            citation_score=scores["citation_score"],
            decomposition_score=scores["decomposition_score"],
            total_time_seconds=round(elapsed, 2),
            total_tool_calls=tool_call_count,
            total_turns=turn_count,
            task_count=len(todos),
            final_report=final_report,
            grader_reasoning=scores["reasoning"],
        )

    async def run_all(
        self,
        case_ids: Optional[List[str]] = None,
        category: Optional[str] = None,
    ) -> List[EvaluationResult]:
        """Run all matching benchmark cases sequentially.

        Args:
            case_ids: Optional filter by case IDs.
            category: Optional filter by category.

        Returns:
            List of EvaluationResult for each case.
        """
        cases = self.load_cases(case_ids=case_ids, category=category)
        results: List[EvaluationResult] = []

        for case in cases:
            result = await self.run_case(case)
            results.append(result)

        return results

    @staticmethod
    def save_results(
        results: List[EvaluationResult],
        output_path: Path,
    ) -> None:
        """Save evaluation results to a JSON file.

        Args:
            results: List of EvaluationResult objects.
            output_path: Destination file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.model_dump() for r in results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def summarize(results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compute aggregate statistics across benchmark results.

        Args:
            results: List of evaluation results.

        Returns:
            Dict with per-dimension averages, total time, and per-case summaries.
        """
        if not results:
            return {"error": "No results to summarize."}

        count = len(results)
        avg_correctness = sum(r.correctness_score for r in results) / count
        avg_completeness = sum(r.completeness_score for r in results) / count
        avg_citation = sum(r.citation_score for r in results) / count
        avg_decomposition = sum(r.decomposition_score for r in results) / count
        total_time = sum(r.total_time_seconds for r in results)

        # Composite score: equal-weighted average of all dimensions
        composite = (avg_correctness + avg_completeness + avg_citation + avg_decomposition) / 4.0

        per_case = [
            {
                "case_id": r.case_id,
                "correctness": r.correctness_score,
                "completeness": r.completeness_score,
                "citation": r.citation_score,
                "decomposition": r.decomposition_score,
                "time_seconds": r.total_time_seconds,
                "tool_calls": r.total_tool_calls,
                "tasks": r.task_count,
            }
            for r in results
        ]

        return {
            "cases_run": count,
            "composite_score": round(composite, 3),
            "avg_correctness": round(avg_correctness, 3),
            "avg_completeness": round(avg_completeness, 3),
            "avg_citation": round(avg_citation, 3),
            "avg_decomposition": round(avg_decomposition, 3),
            "total_time_seconds": round(total_time, 2),
            "per_case": per_case,
        }
