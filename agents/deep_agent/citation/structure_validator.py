"""L1 structural validation for citation integrity.

Validates that inline citation numbers ``[N]`` in a report body have
matching entries in the Sources section, and vice versa.

Design decisions:
- Code blocks (fenced and inline) are stripped before extraction to
  avoid false positives from array indices or code examples (R-01).
- Sources header is always English ``## Sources`` (enforced by
  CitationAgent prompt). Content language may vary.
- L1-05 triggers fast-fail — no further checks if Sources section
  is missing (R-03).
"""

from __future__ import annotations

import logging
import re

from deep_research_agent.agents.deep_agent.citation.models import (
    L1ValidationResult,
    Severity,
    ValidationIssue,
)

logger = logging.getLogger(__name__)


class CitationStructureValidator:
    """L1 structural validation for citation integrity.

    Validates five rules:
        - L1-01: Dangling citation (body ``[N]`` without Source entry) → ERROR
        - L1-02: Orphan source (Source entry never cited in body) → WARNING
        - L1-03: Non-sequential numbering (gaps in Source numbers) → WARNING
        - L1-04: Duplicate URL (same URL in multiple entries) → WARNING
        - L1-05: Missing Sources section (fast-fail) → ERROR
    """

    # Fixed English header — CitationAgent prompt enforces this format
    _SOURCES_HEADER_RE = re.compile(
        r"^##\s*Sources?\s*$", re.MULTILINE | re.IGNORECASE
    )
    _SOURCE_ENTRY_RE = re.compile(
        r"^\[(\d+)\]\s+(.+)$", re.MULTILINE
    )
    _CITATION_RE = re.compile(r"\[(\d+)\]")
    _FENCED_CODE_RE = re.compile(r"```[\s\S]*?```")
    _INLINE_CODE_RE = re.compile(r"`[^`]+`")

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    def validate(self, report: str) -> L1ValidationResult:
        """Validate citation structure of a complete report.

        Args:
            report: Full report text containing body and Sources section.

        Returns:
            L1ValidationResult with all issues found.
        """
        issues: list[ValidationIssue] = []

        # --- Split body vs Sources section ---
        body, source_entries = self._split_report(report)

        # --- Strip code blocks before extracting citations ---
        clean_body = self._strip_code_blocks(body)
        body_citations = {
            int(m) for m in self._CITATION_RE.findall(clean_body)
        }

        if not body_citations:
            self._logger.debug("No inline citations found in report body")
            return L1ValidationResult(is_valid=True)

        # --- L1-05: fast-fail if citations exist but no Sources section ---
        if not source_entries:
            issues.append(ValidationIssue(
                rule_id="L1-05",
                severity=Severity.ERROR,
                message="Report has inline citations but no Sources section",
            ))
            self._logger.warning("L1-05: citations found but no Sources section")
            return L1ValidationResult(
                is_valid=False,
                issues=issues,
                body_citations=body_citations,
                source_entries=source_entries,
            )

        # --- L1-01: dangling citations ---
        dangling = body_citations - set(source_entries.keys())
        for n in sorted(dangling):
            issues.append(ValidationIssue(
                rule_id="L1-01",
                severity=Severity.ERROR,
                message=f"Citation [{n}] in body has no entry in Sources",
            ))

        # --- L1-02: orphan sources ---
        orphans = set(source_entries.keys()) - body_citations
        for n in sorted(orphans):
            issues.append(ValidationIssue(
                rule_id="L1-02",
                severity=Severity.WARNING,
                message=f"Source [{n}] exists but is never cited in body",
            ))

        # --- L1-03: non-sequential numbering ---
        if source_entries:
            expected = set(range(1, max(source_entries.keys()) + 1))
            gaps = expected - set(source_entries.keys())
            if gaps:
                issues.append(ValidationIssue(
                    rule_id="L1-03",
                    severity=Severity.WARNING,
                    message=f"Non-sequential numbering, missing: {sorted(gaps)}",
                ))

        # --- L1-04: duplicate URLs ---
        url_to_numbers: dict[str, list[int]] = {}
        for num, url in source_entries.items():
            url_to_numbers.setdefault(url, []).append(num)
        for url, numbers in url_to_numbers.items():
            if len(numbers) > 1:
                issues.append(ValidationIssue(
                    rule_id="L1-04",
                    severity=Severity.WARNING,
                    message=f"URL {url} assigned multiple numbers: {numbers}",
                ))

        has_errors = any(i.severity == Severity.ERROR for i in issues)
        if issues:
            self._logger.info(
                "L1 validation complete: %d errors, %d warnings",
                sum(1 for i in issues if i.severity == Severity.ERROR),
                sum(1 for i in issues if i.severity == Severity.WARNING),
            )

        return L1ValidationResult(
            is_valid=not has_errors,
            issues=issues,
            body_citations=body_citations,
            source_entries=source_entries,
        )

    def _split_report(self, report: str) -> tuple[str, dict[int, str]]:
        """Split report into body text and parsed source entries.

        Args:
            report: Full report text.

        Returns:
            Tuple of (body_text, {source_number: extracted_url}).
        """
        sources_match = self._SOURCES_HEADER_RE.search(report)
        if not sources_match:
            return report, {}

        body = report[:sources_match.start()]
        sources_text = report[sources_match.start():]
        source_entries = {
            int(m.group(1)): self._extract_url(m.group(2).strip())
            for m in self._SOURCE_ENTRY_RE.finditer(sources_text)
        }
        return body, source_entries

    @staticmethod
    def _extract_url(text: str) -> str:
        """Extract URL from a Sources entry line.

        Handles formats:
        - ``'https://example.com'``
        - ``'Title - https://example.com'``
        - ``'[Title](https://example.com)'``

        Args:
            text: The text after ``[N]`` in a Sources entry.

        Returns:
            The extracted URL, or the original text if no URL found.
        """
        # Markdown link format
        md_match = re.search(r"\((https?://[^\s)]+)\)", text)
        if md_match:
            return md_match.group(1)
        # URL anywhere in text
        url_match = re.search(r"(https?://[^\s]+)", text)
        if url_match:
            return url_match.group(1)
        return text

    @staticmethod
    def _strip_code_blocks(text: str) -> str:
        """Remove fenced code blocks and inline code to avoid false matches.

        Args:
            text: Report body text.

        Returns:
            Text with code blocks removed.
        """
        text = CitationStructureValidator._FENCED_CODE_RE.sub("", text)
        text = CitationStructureValidator._INLINE_CODE_RE.sub("", text)
        return text
