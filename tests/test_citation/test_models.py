"""Tests for citation data models (Finding, WorkerOutput, validation models).

Covers acceptance criteria AC-1 through AC-12 from 01_worker_output_design.md
and validation models from 03_citation_validation_design.md §2.2.
"""

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Finding model tests
# ---------------------------------------------------------------------------

class TestFinding:
    """Tests for the Finding Pydantic model."""

    def test_valid_finding_single_source(self) -> None:
        """AC-1: A valid finding with one source URL is accepted."""
        from deep_research_agent.agents.deep_agent.citation.models import Finding

        f = Finding(
            claim="LangGraph supports MemorySaver and PostgresSaver checkpointers",
            source_urls=["https://langchain-ai.github.io/langgraph/concepts/persistence/"],
            source_titles=["LangGraph Persistence Docs"],
            evidence='"LangGraph provides two built-in checkpointer backends: MemorySaver and PostgresSaver."',
        )
        assert f.claim.startswith("LangGraph")
        assert len(f.source_urls) == 1
        assert len(f.source_titles) == 1

    def test_valid_finding_multiple_sources(self) -> None:
        """AC-2: A finding can bind to multiple source URLs (1:N mapping)."""
        from deep_research_agent.agents.deep_agent.citation.models import Finding

        f = Finding(
            claim="Both LangGraph and CrewAI support parallel execution",
            source_urls=[
                "https://langchain-ai.github.io/langgraph/",
                "https://docs.crewai.com/",
            ],
            source_titles=["LangGraph Docs", "CrewAI Docs"],
            evidence="Both frameworks document concurrent task execution.",
        )
        assert len(f.source_urls) == 2

    def test_missing_source_url_rejected(self) -> None:
        """AC-3: A finding with empty source_urls is rejected (min_length=1)."""
        from deep_research_agent.agents.deep_agent.citation.models import Finding

        with pytest.raises(ValidationError, match="source_urls"):
            Finding(
                claim="Some claim",
                source_urls=[],
                evidence="Some evidence",
            )

    def test_numbered_citation_in_claim_rejected(self) -> None:
        """AC-4: Claim containing [1], [2] etc. is rejected by field_validator."""
        from deep_research_agent.agents.deep_agent.citation.models import Finding

        with pytest.raises(ValidationError, match="numbered citations"):
            Finding(
                claim="LangGraph supports checkpointing [1]",
                source_urls=["https://example.com"],
                evidence="Some evidence",
            )

    def test_numbered_citation_in_claim_double_digit(self) -> None:
        """AC-4b: Multi-digit numbered citations like [12] are also rejected."""
        from deep_research_agent.agents.deep_agent.citation.models import Finding

        with pytest.raises(ValidationError, match="numbered citations"):
            Finding(
                claim="See reference [12] for details",
                source_urls=["https://example.com"],
                evidence="Some evidence",
            )

    def test_bracket_non_numeric_allowed(self) -> None:
        """Brackets with non-numeric content like [RAG] are allowed."""
        from deep_research_agent.agents.deep_agent.citation.models import Finding

        f = Finding(
            claim="The [RAG] approach is widely adopted",
            source_urls=["https://example.com"],
            evidence="RAG is widely used",
        )
        assert "[RAG]" in f.claim

    def test_invalid_url_scheme_rejected(self) -> None:
        """AC-5: URLs without http:// or https:// are rejected."""
        from deep_research_agent.agents.deep_agent.citation.models import Finding

        with pytest.raises(ValidationError, match="http://"):
            Finding(
                claim="Some claim",
                source_urls=["ftp://example.com/file"],
                evidence="Some evidence",
            )

    def test_url_normalization_trailing_slash(self) -> None:
        """AC-6: Trailing slashes are stripped (except root /)."""
        from deep_research_agent.agents.deep_agent.citation.models import Finding

        f = Finding(
            claim="Some claim",
            source_urls=["https://example.com/path/"],
            evidence="evidence",
        )
        assert f.source_urls[0] == "https://example.com/path"

    def test_url_normalization_case(self) -> None:
        """AC-7: Hostname is lowercased."""
        from deep_research_agent.agents.deep_agent.citation.models import Finding

        f = Finding(
            claim="Some claim",
            source_urls=["https://EXAMPLE.COM/Path"],
            evidence="evidence",
        )
        assert f.source_urls[0] == "https://example.com/Path"

    def test_url_normalization_default_port(self) -> None:
        """AC-8: Default ports (:80 for http, :443 for https) are removed."""
        from deep_research_agent.agents.deep_agent.citation.models import Finding

        f = Finding(
            claim="Some claim",
            source_urls=["https://example.com:443/path"],
            evidence="evidence",
        )
        assert ":443" not in f.source_urls[0]

    def test_url_normalization_root_slash_preserved(self) -> None:
        """AC-9: Root path '/' is NOT stripped."""
        from deep_research_agent.agents.deep_agent.citation.models import Finding

        f = Finding(
            claim="Some claim",
            source_urls=["https://example.com/"],
            evidence="evidence",
        )
        assert f.source_urls[0] == "https://example.com/"

    def test_source_titles_default_empty_list(self) -> None:
        """AC-10: source_titles defaults to empty list when not provided."""
        from deep_research_agent.agents.deep_agent.citation.models import Finding

        f = Finding(
            claim="Some claim",
            source_urls=["https://example.com"],
            evidence="evidence",
        )
        assert f.source_titles == []


class TestNormalizeUrl:
    """Tests for Finding.normalize_url() static method."""

    def test_normalize_url_basic(self) -> None:
        from deep_research_agent.agents.deep_agent.citation.models import Finding

        result = Finding.normalize_url("https://EXAMPLE.COM/Path/")
        assert result == "https://example.com/Path"

    def test_normalize_url_empty_fragment_removed(self) -> None:
        from deep_research_agent.agents.deep_agent.citation.models import Finding

        result = Finding.normalize_url("https://example.com/page#")
        assert result == "https://example.com/page"

    def test_normalize_url_non_empty_fragment_preserved(self) -> None:
        from deep_research_agent.agents.deep_agent.citation.models import Finding

        result = Finding.normalize_url("https://example.com/page#section")
        assert result == "https://example.com/page#section"

    def test_normalize_url_non_default_port_preserved(self) -> None:
        from deep_research_agent.agents.deep_agent.citation.models import Finding

        result = Finding.normalize_url("https://example.com:8080/api")
        assert ":8080" in result

    def test_normalize_url_query_params_preserved(self) -> None:
        from deep_research_agent.agents.deep_agent.citation.models import Finding

        result = Finding.normalize_url("https://example.com/search?q=test&page=1")
        assert "q=test" in result
        assert "page=1" in result


# ---------------------------------------------------------------------------
# WorkerOutput model tests
# ---------------------------------------------------------------------------

class TestWorkerOutput:
    """Tests for the WorkerOutput Pydantic model."""

    def _make_finding(self, **overrides) -> dict:
        """Helper to create a valid finding dict."""
        base = {
            "claim": "Test claim",
            "source_urls": ["https://example.com"],
            "evidence": "Test evidence",
        }
        base.update(overrides)
        return base

    def test_valid_worker_output(self) -> None:
        """AC-11: A valid WorkerOutput with all fields is accepted."""
        from deep_research_agent.agents.deep_agent.citation.models import WorkerOutput

        wo = WorkerOutput(
            summary="Overview of findings",
            findings=[self._make_finding()],
            sources_consulted=["https://example.com"],
            caveats="Some pages were paywalled.",
        )
        assert len(wo.findings) == 1
        assert wo.summary == "Overview of findings"

    def test_empty_findings_rejected(self) -> None:
        """AC-12: WorkerOutput with zero findings is rejected (min_length=1)."""
        from deep_research_agent.agents.deep_agent.citation.models import WorkerOutput

        with pytest.raises(ValidationError, match="findings"):
            WorkerOutput(
                summary="Overview",
                findings=[],
            )

    def test_caveats_defaults_to_empty_string(self) -> None:
        """Caveats defaults to empty string when not provided."""
        from deep_research_agent.agents.deep_agent.citation.models import WorkerOutput

        wo = WorkerOutput(
            summary="Overview",
            findings=[self._make_finding()],
        )
        assert wo.caveats == ""

    def test_sources_consulted_defaults_to_empty_list(self) -> None:
        """sources_consulted defaults to empty list when not provided."""
        from deep_research_agent.agents.deep_agent.citation.models import WorkerOutput

        wo = WorkerOutput(
            summary="Overview",
            findings=[self._make_finding()],
        )
        assert wo.sources_consulted == []

    def test_serialization_round_trip(self) -> None:
        """JSON serialization and deserialization preserves all data."""
        from deep_research_agent.agents.deep_agent.citation.models import WorkerOutput

        wo = WorkerOutput(
            summary="Overview",
            findings=[self._make_finding(
                claim="Specific claim",
                source_urls=["https://example.com/a", "https://example.com/b"],
                source_titles=["Source A", "Source B"],
                evidence="Direct quote from source",
            )],
            sources_consulted=["https://example.com/c"],
            caveats="Limited access",
        )
        json_str = wo.model_dump_json()
        restored = WorkerOutput.model_validate_json(json_str)
        assert restored.summary == wo.summary
        assert len(restored.findings) == 1
        assert restored.findings[0].source_urls == wo.findings[0].source_urls
        assert restored.caveats == wo.caveats


# ---------------------------------------------------------------------------
# Validation models tests (Severity, ValidationIssue, L1ValidationResult)
# ---------------------------------------------------------------------------

class TestSeverity:
    """Tests for the Severity enum."""

    def test_severity_values(self) -> None:
        from deep_research_agent.agents.deep_agent.citation.models import Severity

        assert Severity.ERROR == "error"
        assert Severity.WARNING == "warning"

    def test_severity_is_string_enum(self) -> None:
        from deep_research_agent.agents.deep_agent.citation.models import Severity

        assert isinstance(Severity.ERROR, str)


class TestValidationIssue:
    """Tests for the ValidationIssue model."""

    def test_valid_issue(self) -> None:
        from deep_research_agent.agents.deep_agent.citation.models import (
            Severity,
            ValidationIssue,
        )

        issue = ValidationIssue(
            rule_id="L1-01",
            severity=Severity.ERROR,
            message="Citation [8] in body has no entry in Sources",
            context="... some text [8] ...",
        )
        assert issue.rule_id == "L1-01"
        assert issue.severity == Severity.ERROR

    def test_context_defaults_to_none(self) -> None:
        from deep_research_agent.agents.deep_agent.citation.models import (
            Severity,
            ValidationIssue,
        )

        issue = ValidationIssue(
            rule_id="L1-02",
            severity=Severity.WARNING,
            message="Source [5] is never cited",
        )
        assert issue.context is None


class TestL1ValidationResult:
    """Tests for the L1ValidationResult model."""

    def test_valid_result_no_issues(self) -> None:
        from deep_research_agent.agents.deep_agent.citation.models import L1ValidationResult

        result = L1ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.error_count == 0
        assert result.warning_count == 0

    def test_result_with_mixed_issues(self) -> None:
        from deep_research_agent.agents.deep_agent.citation.models import (
            L1ValidationResult,
            Severity,
            ValidationIssue,
        )

        result = L1ValidationResult(
            is_valid=False,
            issues=[
                ValidationIssue(rule_id="L1-01", severity=Severity.ERROR, message="Dangling [8]"),
                ValidationIssue(rule_id="L1-02", severity=Severity.WARNING, message="Orphan [5]"),
                ValidationIssue(rule_id="L1-01", severity=Severity.ERROR, message="Dangling [9]"),
            ],
            body_citations={1, 2, 8, 9},
            source_entries={1: "https://a.com", 2: "https://b.com", 5: "https://e.com"},
        )
        assert result.is_valid is False
        assert result.error_count == 2
        assert result.warning_count == 1

    def test_result_defaults(self) -> None:
        from deep_research_agent.agents.deep_agent.citation.models import L1ValidationResult

        result = L1ValidationResult(is_valid=True)
        assert result.issues == []
        assert result.body_citations == set()
        assert result.source_entries == {}
