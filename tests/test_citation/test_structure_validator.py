"""Tests for L1 CitationStructureValidator.

Covers acceptance criteria AC-L1-01 through AC-L1-11 from
03_citation_validation_design.md §2.
"""

import pytest

from deep_research_agent.agents.deep_agent.citation.models import Severity


class TestCitationStructureValidator:
    """Tests for CitationStructureValidator.validate()."""

    def _make_validator(self):
        from deep_research_agent.agents.deep_agent.citation.structure_validator import (
            CitationStructureValidator,
        )
        return CitationStructureValidator()

    # -- AC-L1-01: Perfect report passes --

    def test_perfect_report_passes(self) -> None:
        """A well-formed report with sequential citations passes L1."""
        v = self._make_validator()
        report = (
            "# Research Report\n\n"
            "LangGraph supports checkpointing [1]. "
            "It uses PostgresSaver for production [2].\n\n"
            "## Sources\n\n"
            "[1] https://langchain-ai.github.io/langgraph/\n"
            "[2] https://github.com/langchain-ai/langgraph\n"
        )
        result = v.validate(report)
        assert result.is_valid is True
        assert result.error_count == 0
        assert result.warning_count == 0

    # -- AC-L1-02: No citations at all → valid (nothing to check) --

    def test_no_citations_no_sources_valid(self) -> None:
        """A report with no citations at all is valid."""
        v = self._make_validator()
        report = "# Report\n\nThis is a simple report without citations.\n"
        result = v.validate(report)
        assert result.is_valid is True

    # -- AC-L1-03: L1-01 Dangling citation (ERROR) --

    def test_dangling_citation_error(self) -> None:
        """Citation [3] in body but not in Sources → L1-01 ERROR."""
        v = self._make_validator()
        report = (
            "This is supported by research [1] and [3].\n\n"
            "## Sources\n\n"
            "[1] https://example.com/a\n"
        )
        result = v.validate(report)
        assert result.is_valid is False
        assert result.error_count >= 1
        error_rules = [i.rule_id for i in result.issues if i.severity == Severity.ERROR]
        assert "L1-01" in error_rules

    # -- AC-L1-04: L1-02 Orphan source (WARNING) --

    def test_orphan_source_warning(self) -> None:
        """Source [2] exists in Sources but never cited → L1-02 WARNING."""
        v = self._make_validator()
        report = (
            "This is supported by research [1].\n\n"
            "## Sources\n\n"
            "[1] https://example.com/a\n"
            "[2] https://example.com/b\n"
        )
        result = v.validate(report)
        # Orphan is WARNING, not ERROR → still valid
        assert result.is_valid is True
        assert result.warning_count >= 1
        warning_rules = [i.rule_id for i in result.issues if i.severity == Severity.WARNING]
        assert "L1-02" in warning_rules

    # -- AC-L1-05: L1-03 Non-sequential numbering (WARNING) --

    def test_non_sequential_numbering_warning(self) -> None:
        """Numbers 1, 2, 5 with gap at 3, 4 → L1-03 WARNING."""
        v = self._make_validator()
        report = (
            "Fact A [1]. Fact B [2]. Fact C [5].\n\n"
            "## Sources\n\n"
            "[1] https://example.com/a\n"
            "[2] https://example.com/b\n"
            "[5] https://example.com/e\n"
        )
        result = v.validate(report)
        assert result.is_valid is True  # WARNING only, no ERROR
        warning_rules = [i.rule_id for i in result.issues if i.severity == Severity.WARNING]
        assert "L1-03" in warning_rules

    # -- AC-L1-06: L1-04 Duplicate URL (WARNING) --

    def test_duplicate_url_warning(self) -> None:
        """Same URL assigned to [1] and [2] → L1-04 WARNING."""
        v = self._make_validator()
        report = (
            "Fact A [1]. Fact B [2].\n\n"
            "## Sources\n\n"
            "[1] https://example.com/same\n"
            "[2] https://example.com/same\n"
        )
        result = v.validate(report)
        assert result.is_valid is True  # WARNING only
        warning_rules = [i.rule_id for i in result.issues if i.severity == Severity.WARNING]
        assert "L1-04" in warning_rules

    # -- AC-L1-07: L1-05 Missing Sources section (ERROR, fast-fail) --

    def test_missing_sources_section_error(self) -> None:
        """Citations exist but no '## Sources' section → L1-05 ERROR."""
        v = self._make_validator()
        report = (
            "# Report\n\n"
            "This has citations [1] and [2] but no Sources section.\n"
        )
        result = v.validate(report)
        assert result.is_valid is False
        error_rules = [i.rule_id for i in result.issues if i.severity == Severity.ERROR]
        assert "L1-05" in error_rules
        # Fast-fail: only L1-05 error, no L1-01 etc.
        assert len(error_rules) == 1

    # -- AC-L1-08: Code blocks are ignored --

    def test_code_block_citations_ignored(self) -> None:
        """Citations inside fenced code blocks should not be extracted."""
        v = self._make_validator()
        report = (
            "# Report\n\n"
            "Real citation [1].\n\n"
            "```python\n"
            "array[2] = 'value'\n"
            "```\n\n"
            "## Sources\n\n"
            "[1] https://example.com\n"
        )
        result = v.validate(report)
        assert result.is_valid is True
        # [2] in code block should NOT be treated as a citation
        assert 2 not in result.body_citations

    def test_inline_code_citations_ignored(self) -> None:
        """Citations inside inline code should not be extracted."""
        v = self._make_validator()
        report = (
            "# Report\n\n"
            "Use `items[0]` to access. Real citation [1].\n\n"
            "## Sources\n\n"
            "[1] https://example.com\n"
        )
        result = v.validate(report)
        assert result.is_valid is True
        assert 0 not in result.body_citations

    # -- AC-L1-09: Markdown link format in Sources --

    def test_markdown_link_in_sources(self) -> None:
        """Sources with Markdown link format [Title](URL) are parsed correctly."""
        v = self._make_validator()
        report = (
            "Fact [1].\n\n"
            "## Sources\n\n"
            "[1] [LangGraph Docs](https://langchain-ai.github.io/langgraph/)\n"
        )
        result = v.validate(report)
        assert result.is_valid is True
        assert result.source_entries[1] == "https://langchain-ai.github.io/langgraph/"

    # -- AC-L1-10: Dash-separated format in Sources --

    def test_dash_separated_url_in_sources(self) -> None:
        """Sources with 'Title - URL' format are parsed correctly."""
        v = self._make_validator()
        report = (
            "Fact [1].\n\n"
            "## Sources\n\n"
            "[1] LangGraph Documentation - https://langchain-ai.github.io/langgraph/\n"
        )
        result = v.validate(report)
        assert result.is_valid is True
        assert "langchain-ai.github.io" in result.source_entries[1]

    # -- AC-L1-11: Empty report is valid --

    def test_empty_report_valid(self) -> None:
        """An empty string report is valid (nothing to validate)."""
        v = self._make_validator()
        result = v.validate("")
        assert result.is_valid is True

    # -- Combined errors and warnings --

    def test_multiple_issues_combined(self) -> None:
        """Report with both dangling citation and orphan source."""
        v = self._make_validator()
        report = (
            "Fact [1]. Unknown [5].\n\n"
            "## Sources\n\n"
            "[1] https://example.com/a\n"
            "[3] https://example.com/c\n"
        )
        result = v.validate(report)
        assert result.is_valid is False  # [5] is dangling → ERROR
        rule_ids = [i.rule_id for i in result.issues]
        assert "L1-01" in rule_ids  # dangling [5]
        assert "L1-02" in rule_ids  # orphan [3]

    # -- Case-insensitive Sources header --

    def test_sources_header_case_insensitive(self) -> None:
        """'## SOURCES' and '## sources' are both recognized."""
        v = self._make_validator()
        report = (
            "Fact [1].\n\n"
            "## SOURCES\n\n"
            "[1] https://example.com\n"
        )
        result = v.validate(report)
        assert result.is_valid is True
