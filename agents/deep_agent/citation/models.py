"""Pydantic data models for the citation system.

Defines:
- ``Finding``: A claim-source-evidence triple preserving provenance.
- ``WorkerOutput``: Structured output schema for research Worker subagents.
- ``Severity``, ``ValidationIssue``, ``L1ValidationResult``: L1 validation data.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional
from urllib.parse import urlparse, urlunparse

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Worker output models
# ---------------------------------------------------------------------------

class Finding(BaseModel):
    """A single claim-source-evidence triple preserving the provenance chain.

    Represents one factual finding extracted by a research Worker.

    Attributes:
        claim: A factual statement. Must not contain numbered citations.
        source_urls: URLs supporting this claim (min 1). Normalized on validation.
        source_titles: Human-readable titles corresponding to source_urls.
        evidence: Brief quote or paraphrase from the source(s).
    """

    claim: str = Field(
        description=(
            "A specific factual statement extracted from research. "
            "Must not contain numbered citations like [1], [2]."
        ),
    )
    source_urls: list[str] = Field(
        description=(
            "URLs that directly support this claim. "
            "Each must be a valid HTTP/HTTPS URL. "
            "Use the most authoritative source(s) available. "
            "For paywalled content, use the accessible secondary source URL."
        ),
        min_length=1,
    )
    source_titles: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable titles for each source URL, in the same "
            "order as source_urls. Extract from the search result's "
            "title field. If a title is unavailable, use the URL's "
            "domain as fallback (e.g., 'langchain.com'). Length should "
            "match source_urls when populated."
        ),
    )
    evidence: str = Field(
        description=(
            "A brief quote or close paraphrase from the source(s) "
            "that directly supports the claim. Use quotation marks "
            'for direct quotes (e.g., "exact words from the source"). '
            "Keep concise but sufficient for downstream verification."
        ),
    )

    @field_validator("claim")
    @classmethod
    def claim_must_not_contain_numbered_citations(cls, v: str) -> str:
        """Reject claims containing numbered citations like [1], [2].

        Citation numbering is handled downstream by CitationAgent.
        Workers must not pre-assign numbers to avoid cross-worker conflicts.
        """
        if re.search(r"\[\d+\]", v):
            raise ValueError(
                "Claim must not contain numbered citations like [1], [2]. "
                "Citation numbering is handled downstream by CitationAgent."
            )
        return v

    @field_validator("source_urls")
    @classmethod
    def urls_must_be_valid_and_normalized(cls, v: list[str]) -> list[str]:
        """Validate URL scheme and normalize to canonical form.

        Normalization rules (RFC 3986):
        1. Lowercase scheme and hostname
        2. Strip trailing slash from path (unless path is exactly '/')
        3. Remove default ports (:80 for http, :443 for https)
        4. Remove empty fragment (#)

        This ensures semantically identical URLs produce the same
        canonical string, enabling correct dedup in CitationAgent.
        """
        normalized: list[str] = []
        for url in v:
            if not url.startswith(("http://", "https://")):
                raise ValueError(
                    f"Source URL must start with http:// or https://, got: {url}"
                )
            normalized.append(Finding.normalize_url(url))
        return normalized

    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize a URL to its canonical form.

        Exposed as a static method so downstream consumers (e.g.
        CitationAgent dedup, L1-04 duplicate URL detection) can
        reuse the same normalization logic.

        Args:
            url: A valid HTTP/HTTPS URL string.

        Returns:
            The canonicalized URL string.
        """
        parsed = urlparse(url)

        # Lowercase scheme and host
        scheme = parsed.scheme.lower()
        host = parsed.hostname or ""
        port = parsed.port

        # Remove default ports
        if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            port = None

        netloc = host
        if port:
            netloc = f"{host}:{port}"
        if parsed.username:
            userinfo = parsed.username
            if parsed.password:
                userinfo += f":{parsed.password}"
            netloc = f"{userinfo}@{netloc}"

        # Strip trailing slash (unless path is root '/')
        path = parsed.path
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")

        # Remove empty fragment
        fragment = parsed.fragment if parsed.fragment else ""

        return urlunparse((scheme, netloc, path, parsed.params, parsed.query, fragment))


class WorkerOutput(BaseModel):
    """Structured output schema for a research Worker subagent.

    Used as ``response_format`` in the Worker SubAgent spec, ensuring
    the LLM produces well-structured, machine-parseable output while
    preserving full fact-to-source provenance.

    Attributes:
        summary: Brief overview of core findings.
        findings: Factual findings with provenance (min 1).
        sources_consulted: All URLs searched, including uncited ones.
        caveats: Information gaps or limitations encountered.
    """

    summary: str = Field(
        description="2-3 sentence overview of the core findings.",
    )
    findings: list[Finding] = Field(
        description=(
            "Key factual findings extracted from research. "
            "Each finding binds a claim to its source(s) and evidence. "
            "Aim for 3-10 findings for typical research tasks."
        ),
        min_length=1,
    )
    sources_consulted: list[str] = Field(
        default_factory=list,
        description=(
            "All URLs searched during research, including those "
            "not directly cited in findings. Each entry is a URL "
            "optionally followed by ' — ' and a brief description."
        ),
    )
    caveats: str = Field(
        default="",
        description=(
            "Information gaps, uncertainties, or search limitations "
            "encountered during research. Empty string if none."
        ),
    )


# ---------------------------------------------------------------------------
# L1 Validation models
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Validation issue severity level."""

    ERROR = "error"
    WARNING = "warning"


class ValidationIssue(BaseModel):
    """A single validation finding.

    Attributes:
        rule_id: Rule identifier, e.g. 'L1-01'.
        severity: ERROR or WARNING.
        message: Human-readable description of the issue.
        context: Relevant text snippet for debugging.
    """

    rule_id: str = Field(description="Rule identifier, e.g. 'L1-01'")
    severity: Severity
    message: str = Field(description="Human-readable description of the issue")
    context: Optional[str] = Field(
        default=None,
        description="Relevant text snippet from the report for debugging",
    )


class L1ValidationResult(BaseModel):
    """Result of L1 structural validation.

    Attributes:
        is_valid: True if no ERROR-level issues found.
        issues: All validation issues found.
        body_citations: Set of citation numbers found in report body.
        source_entries: Mapping of source number to URL from Sources section.
    """

    is_valid: bool = Field(description="True if no ERROR-level issues found")
    issues: list[ValidationIssue] = Field(default_factory=list)
    body_citations: set[int] = Field(default_factory=set)
    source_entries: dict[int, str] = Field(default_factory=dict)

    @property
    def error_count(self) -> int:
        """Count of ERROR-level issues."""
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of WARNING-level issues."""
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)
