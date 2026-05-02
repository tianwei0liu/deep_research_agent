"""Tests for citation prompt templates.

Validates prompt content requirements from:
- 01_worker_output_design.md §4.1 (Worker prompt)
- 02_citation_annotation_design.md §2.2 (CitationSpecialist prompt)
- 02_citation_annotation_design.md §2.3 (Supervisor §6)
"""

from deep_research_agent.agents.deep_agent.deep_agent import DeepAgentPrompts


class TestWorkerPrompt:
    """Tests for the updated Worker prompt."""

    def test_prohibits_numbered_citations(self) -> None:
        """Worker prompt must explicitly prohibit [1], [2] citations."""
        assert "DO NOT use numbered citations" in DeepAgentPrompts.WORKER

    def test_mentions_structured_schema(self) -> None:
        """Worker prompt must reference structured schema output."""
        assert "structured schema" in DeepAgentPrompts.WORKER.lower() or \
               "structured output" in DeepAgentPrompts.WORKER.lower()

    def test_contains_extract_immediately(self) -> None:
        """Worker prompt must contain 'Extract Immediately' instruction."""
        assert "Extract Immediately" in DeepAgentPrompts.WORKER or \
               "extract immediately" in DeepAgentPrompts.WORKER.lower()

    def test_mentions_source_urls(self) -> None:
        """Worker prompt must mention source URL tracking."""
        assert "source URL" in DeepAgentPrompts.WORKER or \
               "source url" in DeepAgentPrompts.WORKER.lower()

    def test_mentions_evidence(self) -> None:
        """Worker prompt must mention evidence collection."""
        assert "evidence" in DeepAgentPrompts.WORKER.lower()

    def test_mentions_source_titles(self) -> None:
        """Worker prompt must mention source titles."""
        assert "source_titles" in DeepAgentPrompts.WORKER or \
               "title" in DeepAgentPrompts.WORKER.lower()


class TestCitationSpecialistPrompt:
    """Tests for the CitationSpecialist prompt."""

    def test_prompt_exists(self) -> None:
        """CITATION_SPECIALIST prompt attribute must exist."""
        assert hasattr(DeepAgentPrompts, "CITATION_SPECIALIST")
        assert len(DeepAgentPrompts.CITATION_SPECIALIST) > 100

    def test_mentions_numbered_citations(self) -> None:
        """Prompt must describe inline citation format [N]."""
        prompt = DeepAgentPrompts.CITATION_SPECIALIST
        assert "[N]" in prompt or "[1]" in prompt

    def test_mentions_sources_section(self) -> None:
        """Prompt must require a Sources section."""
        assert "Sources" in DeepAgentPrompts.CITATION_SPECIALIST

    def test_mentions_worker_findings(self) -> None:
        """Prompt must reference worker findings as citation source."""
        prompt = DeepAgentPrompts.CITATION_SPECIALIST.lower()
        assert "finding" in prompt or "worker" in prompt

    def test_mentions_dedup(self) -> None:
        """Prompt must address URL deduplication."""
        prompt = DeepAgentPrompts.CITATION_SPECIALIST.lower()
        assert "dedup" in prompt or "duplicate" in prompt or "same url" in prompt

    def test_mentions_sequential_numbering(self) -> None:
        """Prompt must require sequential numbering starting from 1."""
        prompt = DeepAgentPrompts.CITATION_SPECIALIST
        assert "sequential" in prompt.lower() or "1, 2, 3" in prompt

    def test_mentions_self_check(self) -> None:
        """Prompt must include self-verification instructions (L1 prompt-level)."""
        prompt = DeepAgentPrompts.CITATION_SPECIALIST.lower()
        assert "verify" in prompt or "check" in prompt

    def test_mentions_citation_needed_fallback(self) -> None:
        """Prompt must handle unmatched claims with [citation needed]."""
        prompt = DeepAgentPrompts.CITATION_SPECIALIST.lower()
        assert "citation needed" in prompt


class TestSupervisorPrompt:
    """Tests for the Supervisor prompt §6 citation workflow."""

    def test_contains_citation_workflow(self) -> None:
        """Supervisor prompt must contain §6 or 'citation' workflow instructions."""
        prompt = DeepAgentPrompts.SUPERVISOR.lower()
        assert "citation" in prompt

    def test_mentions_citation_specialist(self) -> None:
        """Supervisor prompt must reference the citation-specialist subagent."""
        assert "citation-specialist" in DeepAgentPrompts.SUPERVISOR or \
               "citation specialist" in DeepAgentPrompts.SUPERVISOR.lower()

    def test_mentions_fallback(self) -> None:
        """Supervisor prompt must describe fallback behavior (§6.1)."""
        prompt = DeepAgentPrompts.SUPERVISOR.lower()
        assert "fallback" in prompt or "self-citation" in prompt or \
               "if the citation" in prompt
