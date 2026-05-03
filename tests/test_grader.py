"""Unit tests for the BenchmarkGrader."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research_agent.benchmarks.grader import BenchmarkGrader
from deep_research_agent.config import Settings


def _make_settings(**overrides) -> Settings:
    """Create a Settings instance with sensible test defaults."""
    defaults = dict(
        tavily_api_key="test-tavily",
        gemini_api_key="test-gemini",
        worker_model="gemini-2.5-flash",
        worker_temperature=0.0,
        planner_model="gemini-2.5-flash",
        planner_temperature=0.0,
        grader_model="gemini-2.5-flash",
        grader_temperature=0.0,
        grader_thinking_level="medium",
        grader_google_search_enabled=True,
        tavily_search_url="https://api.tavily.com/search",
        tavily_max_result_chars=1000,
        max_parallel_workers=10,
        supervisor_max_turns=35,
        supervisor_max_search_calls=10,
        worker_max_search_calls=60,
        worker_max_turns=10,
        worker_max_output_tokens=8192,
        citation_max_retries=5,
        research_timeout_seconds=600,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _make_response(parts_data: list) -> MagicMock:
    """Build a mock GenerateContentResponse with the given parts.

    Each item in parts_data is a dict with optional keys:
      - text: str
      - thought: bool (default False)
    """
    parts = []
    for pd in parts_data:
        p = MagicMock()
        p.text = pd.get("text")
        p.thought = pd.get("thought", False)
        parts.append(p)

    candidate = MagicMock()
    candidate.content.parts = parts
    response = MagicMock()
    response.candidates = [candidate]
    return response


# ---------------------------------------------------------------------------
# _extract_text tests
# ---------------------------------------------------------------------------


class TestExtractText:
    """Tests for BenchmarkGrader._extract_text."""

    def test_single_text_part(self):
        """Normal case: one text part, no thinking."""
        resp = _make_response([{"text": "hello", "thought": False}])
        assert BenchmarkGrader._extract_text(resp) == "hello"

    def test_skips_thinking_part(self):
        """When thinking is enabled, the first part is a thought — should be skipped."""
        resp = _make_response([
            {"text": "internal reasoning...", "thought": True},
            {"text": '{"correctness_score": 1.0}', "thought": False},
        ])
        assert BenchmarkGrader._extract_text(resp) == '{"correctness_score": 1.0}'

    def test_multiple_thinking_parts(self):
        """Multiple thinking parts before the text output."""
        resp = _make_response([
            {"text": "step 1", "thought": True},
            {"text": "step 2", "thought": True},
            {"text": "final answer", "thought": False},
        ])
        assert BenchmarkGrader._extract_text(resp) == "final answer"

    def test_fallback_when_all_are_thoughts(self):
        """Edge case: all parts are thoughts — fallback to parts[0]."""
        resp = _make_response([
            {"text": "only thought", "thought": True},
        ])
        assert BenchmarkGrader._extract_text(resp) == "only thought"


# ---------------------------------------------------------------------------
# _parse_grader_response tests
# ---------------------------------------------------------------------------


class TestParseGraderResponse:
    """Tests for BenchmarkGrader._parse_grader_response."""

    def test_valid_json(self):
        grader = BenchmarkGrader(settings=_make_settings())
        raw = json.dumps({
            "correctness_score": 0.9,
            "completeness_score": 0.8,
            "citation_score": 0.7,
            "decomposition_score": 0.6,
            "reasoning": "Good report.",
        })
        result = grader._parse_grader_response(raw)
        assert result["correctness_score"] == 0.9
        assert result["reasoning"] == "Good report."

    def test_strips_markdown_fence(self):
        grader = BenchmarkGrader(settings=_make_settings())
        raw = '```json\n{"correctness_score": 1.0, "completeness_score": 1.0, "citation_score": 1.0, "decomposition_score": 1.0, "reasoning": "ok"}\n```'
        result = grader._parse_grader_response(raw)
        assert result["correctness_score"] == 1.0

    def test_clamps_out_of_range_scores(self):
        grader = BenchmarkGrader(settings=_make_settings())
        raw = json.dumps({
            "correctness_score": 1.5,
            "completeness_score": -0.3,
            "citation_score": 0.5,
            "decomposition_score": 0.5,
            "reasoning": "out of range",
        })
        result = grader._parse_grader_response(raw)
        assert result["correctness_score"] == 1.0
        assert result["completeness_score"] == 0.0

    def test_fills_missing_keys(self):
        grader = BenchmarkGrader(settings=_make_settings())
        raw = json.dumps({"correctness_score": 0.8})
        result = grader._parse_grader_response(raw)
        assert result["completeness_score"] == 0.0
        assert result["reasoning"] == "No reasoning provided."


# ---------------------------------------------------------------------------
# grade() integration-level tests (mocked API)
# ---------------------------------------------------------------------------


class TestGradeConfig:
    """Verify that grade() passes the right config to the Gemini API."""

    @pytest.mark.asyncio
    async def test_includes_search_tool_when_enabled(self):
        """When grader_google_search_enabled=True, tools should include GoogleSearch."""
        settings = _make_settings(grader_google_search_enabled=True)
        grader = BenchmarkGrader(settings=settings)

        mock_response = _make_response([{
            "text": json.dumps({
                "correctness_score": 1.0,
                "completeness_score": 1.0,
                "citation_score": 1.0,
                "decomposition_score": 1.0,
                "reasoning": "ok",
            }),
            "thought": False,
        }])

        with patch("deep_research_agent.benchmarks.grader.genai") as mock_genai:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            mock_genai.Client.return_value = mock_client

            await grader.grade(
                query="test",
                expected_facets=["a"],
                final_report="report",
                task_objectives=["t1"],
            )

            call_kwargs = mock_client.aio.models.generate_content.call_args
            config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
            assert config.tools is not None
            assert len(config.tools) == 1

    @pytest.mark.asyncio
    async def test_excludes_search_tool_when_disabled(self):
        """When grader_google_search_enabled=False, tools should be None."""
        settings = _make_settings(grader_google_search_enabled=False)
        grader = BenchmarkGrader(settings=settings)

        mock_response = _make_response([{
            "text": json.dumps({
                "correctness_score": 1.0,
                "completeness_score": 1.0,
                "citation_score": 1.0,
                "decomposition_score": 1.0,
                "reasoning": "ok",
            }),
            "thought": False,
        }])

        with patch("deep_research_agent.benchmarks.grader.genai") as mock_genai:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            mock_genai.Client.return_value = mock_client

            await grader.grade(
                query="test",
                expected_facets=["a"],
                final_report="report",
                task_objectives=["t1"],
            )

            call_kwargs = mock_client.aio.models.generate_content.call_args
            config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
            assert config.tools is None

    @pytest.mark.asyncio
    async def test_system_prompt_includes_date(self):
        """The system instruction should contain today's date."""
        settings = _make_settings(grader_google_search_enabled=False)
        grader = BenchmarkGrader(settings=settings)

        mock_response = _make_response([{
            "text": json.dumps({
                "correctness_score": 1.0,
                "completeness_score": 1.0,
                "citation_score": 1.0,
                "decomposition_score": 1.0,
                "reasoning": "ok",
            }),
            "thought": False,
        }])

        with patch("deep_research_agent.benchmarks.grader.genai") as mock_genai:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            mock_genai.Client.return_value = mock_client

            await grader.grade(
                query="test",
                expected_facets=["a"],
                final_report="report",
                task_objectives=["t1"],
            )

            call_kwargs = mock_client.aio.models.generate_content.call_args
            config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            assert today in config.system_instruction
