"""Lightweight Skills catalog for Supervisor prompt injection.

Generates a compact metadata table of available persona skills for
the Supervisor to recommend during the Skills Discovery phase.
Does NOT expose full framework prompts — only IDs, names, descriptions,
and domain tags.

The catalog is intentionally separated from ``PersonaRegistry`` to
enforce the progressive-disclosure principle: Supervisor sees only
metadata; the full framework is loaded later by ``PersonaMiddleware``
upon activation.
"""

from __future__ import annotations

import logging
from typing import Optional

from deep_research_agent.personas.registry import PersonaConfig, PersonaRegistry

logger = logging.getLogger(__name__)


class SkillsCatalog:
    """Read-only index of available persona skills for prompt injection.

    Consumes a ``PersonaRegistry`` and produces compact representations
    suitable for embedding in the Supervisor system prompt (~2K tokens
    for 19 personas).

    Args:
        registry: The persona registry to derive the catalog from.
            If ``None``, an empty catalog is created (no skills available).
    """

    def __init__(self, registry: Optional[PersonaRegistry] = None) -> None:
        self._registry = registry
        self._personas: list[PersonaConfig] = (
            registry.list_personas() if registry else []
        )

    @property
    def count(self) -> int:
        """Number of available personas in the catalog."""
        return len(self._personas)

    def format_skills_table(self) -> str:
        """Generate a markdown table of available skills for prompt injection.

        The table includes only lightweight metadata — persona ID,
        display name, one-line description, and applicable domains.
        This keeps the context cost to ~2K tokens for 19 personas.

        Returns:
            Formatted markdown table string.  Returns an empty string
            if no personas are registered.
        """
        if not self._personas:
            return ""

        lines = [
            "| persona_id | display_name | description | domains |",
            "|------------|-------------|-------------|---------|",
        ]
        for p in self._personas:
            domains = ", ".join(p.applicable_domains) if p.applicable_domains else "-"
            lines.append(
                f"| {p.persona_id} | {p.display_name} | {p.description} | {domains} |"
            )
        return "\n".join(lines)

    def match_domains(self, domains: list[str]) -> list[PersonaConfig]:
        """Return personas whose applicable_domains intersect with the given domains.

        Args:
            domains: Domain tags to match against (e.g. ``["finance", "investment"]``).

        Returns:
            List of matching ``PersonaConfig`` instances, ordered by
            number of overlapping domains (most relevant first).
        """
        if not domains:
            return []

        query_set = set(d.lower() for d in domains)
        scored: list[tuple[int, PersonaConfig]] = []
        for p in self._personas:
            persona_domains = set(d.lower() for d in p.applicable_domains)
            overlap = len(query_set & persona_domains)
            if overlap > 0:
                scored.append((overlap, p))

        # Sort by overlap count descending, then by persona_id for stability
        scored.sort(key=lambda x: (-x[0], x[1].persona_id))
        return [p for _, p in scored]

    def get_persona_ids(self) -> list[str]:
        """Return all registered persona IDs.

        Returns:
            List of persona ID strings.
        """
        return [p.persona_id for p in self._personas]
