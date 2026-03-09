"""LLM-as-Judge grader for evaluating research agent output.

Uses Gemini to score final reports on correctness, completeness,
citation quality, and decomposition quality.
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from deep_research_agent.config import Settings


class BenchmarkGrader:
    """Grades a research agent's final report using an LLM judge.

    Args:
        settings: Application settings for API keys and model configuration.
    """

    # Rubric prompt template for the grader. The LLM produces a structured JSON
    # evaluation based on the query, expected facets, and the agent's report.
    _GRADER_SYSTEM_PROMPT = """You are an expert evaluator for an AI research agent.
You will receive:
1. The original User Query.
2. A list of Expected Facets the report should cover.
3. An optional Reference Answer (gold standard).
4. The Agent's Final Report.
5. Metadata about the agent's task decomposition.

Evaluate the report on four dimensions. For each, give a score from 0.0 to 1.0.

## Scoring Rubric

### Correctness (0.0–1.0)
- 1.0: All stated facts are accurate and consistent with the reference answer (if provided).
- 0.7: Mostly accurate with minor errors or imprecisions.
- 0.4: Contains some correct information but also notable inaccuracies.
- 0.0: Largely incorrect or fabricated information.
If no reference answer is provided, use the google_search tool to verify key factual claims
before scoring. If a claim cannot be verified through search, mark it as "unverifiable" in
your reasoning rather than assuming it is a hallucination. Only penalize claims that are
demonstrably false based on search results.

### Completeness (0.0–1.0)
- 1.0: All expected facets are covered with meaningful depth.
- 0.7: Most facets covered, one or two minor omissions.
- 0.4: Several facets missing or only superficially mentioned.
- 0.0: Report is off-topic or covers almost none of the expected facets.

### Citation Quality (0.0–1.0)
- 1.0: All major claims have inline citations; sources are listed and appear credible.
- 0.7: Most claims cited; a few unsupported statements.
- 0.4: Sparse citations; many claims lack support.
- 0.0: No citations at all.

### Decomposition Quality (0.0–1.0)
Evaluate based on the task metadata provided.
- 1.0: Tasks are well-scoped, non-overlapping, and collectively cover the query.
- 0.7: Reasonable decomposition with minor redundancy or gaps.
- 0.4: Poor decomposition — overly broad tasks or significant overlap.
- 0.0: No meaningful decomposition (single task repeating the query, or zero tasks).

## Output Format
You MUST return ONLY valid JSON (no markdown fences, no extra text) with this schema:
{
  "correctness_score": <float>,
  "completeness_score": <float>,
  "citation_score": <float>,
  "decomposition_score": <float>,
  "reasoning": "<string explaining scores>"
}
"""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.logger = logging.getLogger(__name__)
        self._settings = settings or Settings.load()

    async def grade(
        self,
        query: str,
        expected_facets: List[str],
        final_report: str,
        task_objectives: List[str],
        reference_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Grade a final report against the benchmark criteria.

        Args:
            query: The original user query.
            expected_facets: List of topics the report must cover.
            final_report: The agent's generated report text.
            task_objectives: Objectives of the tasks the supervisor created.
            reference_answer: Optional gold-standard answer for correctness.

        Returns:
            Dict with keys: correctness_score, completeness_score,
            citation_score, decomposition_score, reasoning.

        Raises:
            ValueError: If the grader response cannot be parsed as JSON.
        """
        client = genai.Client(api_key=self._settings.require_gemini_api_key())
        model_id = self._settings.grader_model

        user_prompt = self._build_user_prompt(
            query, expected_facets, final_report, task_objectives, reference_answer
        )

        self.logger.info("Grading report for query: %s", query[:80])

        # Inject current date so the Judge knows its parametric knowledge may be stale.
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        system_prompt = (
            f"Today's date is {today}. Your training data may not include events "
            "after your knowledge cutoff. Use the google_search tool to verify "
            "any facts you are uncertain about before scoring.\n\n"
            + self._GRADER_SYSTEM_PROMPT
        )

        # Enable Google Search grounding so the Judge can verify facts in real time.
        tools = None
        if self._settings.grader_google_search_enabled:
            tools = [types.Tool(google_search=types.GoogleSearch())]

        response = await client.aio.models.generate_content(
            model=model_id,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=user_prompt)],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self._settings.grader_temperature,
                thinking_config=types.ThinkingConfig(thinking_level=self._settings.grader_thinking_level),
                tools=tools,
            ),
        )

        raw_text = self._extract_text(response)
        return self._parse_grader_response(raw_text)

    @staticmethod
    def _extract_text(response: types.GenerateContentResponse) -> str:
        """Extract the model's text output, skipping any thinking parts.

        When thinking is enabled, parts[0] is the thought and the actual
        text response comes in a subsequent part.  This method finds the
        first non-thought text part.

        Args:
            response: The raw Gemini API response.

        Returns:
            The text content of the model's response.
        """
        for part in response.candidates[0].content.parts:
            if part.thought:  # Skip thinking parts
                continue
            if part.text is not None:
                return part.text
        # Fallback: return whatever is in parts[0]
        return response.candidates[0].content.parts[0].text

    @staticmethod
    def _build_user_prompt(
        query: str,
        expected_facets: List[str],
        final_report: str,
        task_objectives: List[str],
        reference_answer: Optional[str],
    ) -> str:
        """Construct the user prompt for the grader LLM.

        Args:
            query: Original user query.
            expected_facets: Required topics.
            final_report: Agent's report.
            task_objectives: Supervisor's task decomposition.
            reference_answer: Optional gold standard.

        Returns:
            Formatted prompt string.
        """
        facets_str = "\n".join(f"- {f}" for f in expected_facets)
        tasks_str = "\n".join(f"- {obj}" for obj in task_objectives) if task_objectives else "(No tasks created)"

        ref_section = ""
        if reference_answer:
            ref_section = f"\n## Reference Answer\n{reference_answer}\n"

        return f"""## User Query
{query}

## Expected Facets
{facets_str}
{ref_section}
## Task Decomposition by Supervisor
{tasks_str}

## Agent's Final Report
{final_report}
"""

    def _parse_grader_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse the grader's JSON response, handling common LLM formatting issues.

        Args:
            raw_text: Raw text output from the grader LLM.

        Returns:
            Dict with score fields and reasoning.

        Raises:
            ValueError: If parsing fails after cleanup attempts.
        """
        # Strip markdown code fences if the LLM wraps its output
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            # Remove closing fence
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            self.logger.error("Failed to parse grader response: %s", raw_text[:200])
            raise ValueError(f"Grader returned invalid JSON: {exc}") from exc

        # Validate expected keys with safe defaults
        default_scores: Dict[str, Any] = {
            "correctness_score": 0.0,
            "completeness_score": 0.0,
            "citation_score": 0.0,
            "decomposition_score": 0.0,
            "reasoning": "No reasoning provided.",
        }

        for key, default in default_scores.items():
            if key not in data:
                self.logger.warning("Grader response missing key '%s', using default.", key)
                data[key] = default

        # Clamp scores to [0.0, 1.0]
        for score_key in ["correctness_score", "completeness_score", "citation_score", "decomposition_score"]:
            data[score_key] = max(0.0, min(1.0, float(data[score_key])))

        return data
