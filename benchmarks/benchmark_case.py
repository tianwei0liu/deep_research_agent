"""Data models for benchmark cases and evaluation results."""

from typing import List, Optional

from pydantic import BaseModel, Field


class BenchmarkCase(BaseModel):
    """A single benchmark test case for the research agent.

    Attributes:
        id: Unique identifier (e.g., "factual_001").
        category: Category tag (e.g., "factual", "comparative", "temporal", "multi_hop").
        query: The user query to send to the orchestrator.
        reference_answer: Optional gold-standard answer for correctness grading.
        expected_facets: Topics the final report MUST cover for completeness grading.
        difficulty: Difficulty tier ("easy", "medium", "hard").
    """

    id: str = Field(..., description="Unique case identifier.")
    category: str = Field(..., description="Category tag for grouping.")
    query: str = Field(..., description="The user query.")
    reference_answer: Optional[str] = Field(
        None, description="Optional gold-standard answer."
    )
    expected_facets: List[str] = Field(
        default_factory=list,
        description="Topics the report must cover.",
    )
    difficulty: str = Field(
        default="medium",
        description="Difficulty tier: easy, medium, or hard.",
    )


class EvaluationResult(BaseModel):
    """Structured evaluation result for a single benchmark run.

    Attributes:
        case_id: The benchmark case that was evaluated.
        correctness_score: Factual correctness (0.0–1.0).
        completeness_score: Coverage of expected facets (0.0–1.0).
        citation_score: Quality and presence of inline citations (0.0–1.0).
        decomposition_score: Quality of task decomposition by the supervisor (0.0–1.0).
        total_time_seconds: Wall-clock time for the full orchestrator run.
        total_tool_calls: Number of tool calls made by the supervisor.
        total_turns: Number of supervisor loop iterations.
        task_count: Number of research tasks created.
        final_report: The full text of the agent's final report.
        grader_reasoning: The LLM judge's explanation of the scores.
    """

    case_id: str
    correctness_score: float = Field(
        ..., ge=0.0, le=1.0, description="Factual correctness."
    )
    completeness_score: float = Field(
        ..., ge=0.0, le=1.0, description="Facet coverage."
    )
    citation_score: float = Field(
        ..., ge=0.0, le=1.0, description="Citation quality."
    )
    decomposition_score: float = Field(
        ..., ge=0.0, le=1.0, description="Task decomposition quality."
    )
    total_time_seconds: float
    total_tool_calls: int
    total_turns: int
    task_count: int
    final_report: str
    grader_reasoning: str
