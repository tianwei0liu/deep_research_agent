"""Tests for SkillsCatalog and PersonaMiddleware runtime activation.

Covers:
- SkillsCatalog.format_skills_table() output format
- SkillsCatalog.match_domains() matching logic
- PersonaMiddleware.activate() / deactivate() state transitions
"""

from __future__ import annotations

from dataclasses import field
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from deep_research_agent.agents.skills_catalog import SkillsCatalog
from deep_research_agent.agents.persona_middleware import PersonaMiddleware
from deep_research_agent.personas.registry import PersonaConfig, PersonaRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_persona(
    persona_id: str,
    display_name: str = "Test",
    description: str = "desc",
    domains: Optional[list[str]] = None,
) -> PersonaConfig:
    """Create a PersonaConfig for testing."""
    return PersonaConfig(
        persona_id=persona_id,
        display_name=display_name,
        description=description,
        framework_prompt="test framework prompt",
        max_tokens=800,
        applicable_domains=domains or [],
    )


def _make_registry(personas: list[PersonaConfig]) -> PersonaRegistry:
    """Create a mock PersonaRegistry with the given personas."""
    registry = MagicMock(spec=PersonaRegistry)
    registry.list_personas.return_value = personas

    def get_side_effect(persona_id: str) -> Optional[PersonaConfig]:
        for p in personas:
            if p.persona_id == persona_id:
                return p
        return None

    registry.get.side_effect = get_side_effect
    return registry


# ---------------------------------------------------------------------------
# SkillsCatalog tests
# ---------------------------------------------------------------------------


class TestSkillsCatalog:
    """Tests for SkillsCatalog."""

    def test_empty_catalog_returns_empty_string(self) -> None:
        """format_skills_table returns empty string for empty catalog."""
        catalog = SkillsCatalog(registry=None)
        assert catalog.format_skills_table() == ""
        assert catalog.count == 0

    def test_format_skills_table_structure(self) -> None:
        """Table output has correct markdown format."""
        personas = [
            _make_persona("buffett", "巴菲特", "价值投资", ["finance", "investment"]),
            _make_persona("feynman", "费曼", "物理学", ["physics", "science"]),
        ]
        registry = _make_registry(personas)
        catalog = SkillsCatalog(registry=registry)

        table = catalog.format_skills_table()

        # Header row
        assert "persona_id" in table
        assert "display_name" in table
        assert "description" in table
        assert "domains" in table

        # Separator row
        assert "|---" in table

        # Data rows
        assert "buffett" in table
        assert "巴菲特" in table
        assert "feynman" in table
        assert "费曼" in table
        assert "finance, investment" in table

    def test_count_reflects_registry(self) -> None:
        """Count property reflects the number of personas."""
        personas = [
            _make_persona("a"),
            _make_persona("b"),
            _make_persona("c"),
        ]
        catalog = SkillsCatalog(registry=_make_registry(personas))
        assert catalog.count == 3

    def test_get_persona_ids(self) -> None:
        """get_persona_ids returns all IDs."""
        personas = [_make_persona("x"), _make_persona("y")]
        catalog = SkillsCatalog(registry=_make_registry(personas))
        assert catalog.get_persona_ids() == ["x", "y"]

    def test_match_domains_single_overlap(self) -> None:
        """Matches personas with overlapping domains."""
        personas = [
            _make_persona("buffett", domains=["finance", "investment"]),
            _make_persona("feynman", domains=["physics", "science"]),
            _make_persona("karpathy", domains=["ai", "technology"]),
        ]
        catalog = SkillsCatalog(registry=_make_registry(personas))

        matches = catalog.match_domains(["finance"])
        assert len(matches) == 1
        assert matches[0].persona_id == "buffett"

    def test_match_domains_multiple_overlap_sorted(self) -> None:
        """Personas with more overlapping domains rank higher."""
        personas = [
            _make_persona("munger", domains=["finance", "philosophy"]),
            _make_persona("buffett", domains=["finance", "investment", "business"]),
        ]
        catalog = SkillsCatalog(registry=_make_registry(personas))

        matches = catalog.match_domains(["finance", "investment"])
        assert len(matches) == 2
        # buffett has 2 overlaps, munger has 1
        assert matches[0].persona_id == "buffett"
        assert matches[1].persona_id == "munger"

    def test_match_domains_no_match(self) -> None:
        """Returns empty list when no domains overlap."""
        personas = [
            _make_persona("buffett", domains=["finance"]),
        ]
        catalog = SkillsCatalog(registry=_make_registry(personas))

        matches = catalog.match_domains(["hardware", "semiconductor"])
        assert matches == []

    def test_match_domains_empty_input(self) -> None:
        """Returns empty list for empty domain list."""
        personas = [_make_persona("buffett", domains=["finance"])]
        catalog = SkillsCatalog(registry=_make_registry(personas))
        assert catalog.match_domains([]) == []

    def test_match_domains_case_insensitive(self) -> None:
        """Domain matching is case-insensitive."""
        personas = [
            _make_persona("buffett", domains=["Finance", "Investment"]),
        ]
        catalog = SkillsCatalog(registry=_make_registry(personas))

        matches = catalog.match_domains(["finance"])
        assert len(matches) == 1

    def test_format_table_no_domains(self) -> None:
        """Personas without domains show '-' in the domains column."""
        personas = [_make_persona("test", domains=[])]
        catalog = SkillsCatalog(registry=_make_registry(personas))
        table = catalog.format_skills_table()
        assert "- |" in table


# ---------------------------------------------------------------------------
# PersonaMiddleware runtime activation tests
# ---------------------------------------------------------------------------


class TestPersonaMiddlewareActivation:
    """Tests for PersonaMiddleware.activate() / deactivate()."""

    def test_activate_success(self) -> None:
        """activate() returns True and sets current_persona."""
        persona = _make_persona("buffett", "巴菲特")
        registry = _make_registry([persona])

        mw = PersonaMiddleware(registry=registry)
        assert not mw.active

        result = mw.activate("buffett")
        assert result is True
        assert mw.active
        assert mw.current_persona is not None
        assert mw.current_persona.persona_id == "buffett"

    def test_activate_not_found(self) -> None:
        """activate() returns False for unknown persona_id."""
        registry = _make_registry([])

        mw = PersonaMiddleware(registry=registry)
        result = mw.activate("nonexistent")
        assert result is False
        assert not mw.active

    def test_activate_without_registry_raises(self) -> None:
        """activate() raises RuntimeError if no registry provided."""
        mw = PersonaMiddleware()
        with pytest.raises(RuntimeError, match="no PersonaRegistry"):
            mw.activate("buffett")

    def test_deactivate(self) -> None:
        """deactivate() resets to inactive state."""
        persona = _make_persona("buffett", "巴菲特")
        registry = _make_registry([persona])

        mw = PersonaMiddleware(persona=persona, registry=registry)
        assert mw.active

        mw.deactivate()
        assert not mw.active
        assert mw.current_persona is None

    def test_deactivate_when_already_inactive(self) -> None:
        """deactivate() is a no-op when already inactive."""
        mw = PersonaMiddleware()
        mw.deactivate()  # Should not raise
        assert not mw.active

    def test_build_time_activation_backward_compat(self) -> None:
        """Build-time persona activation still works."""
        persona = _make_persona("buffett", "巴菲特")
        mw = PersonaMiddleware(persona=persona)

        assert mw.active
        assert mw.current_persona is persona

    def test_activate_switch_persona(self) -> None:
        """Can switch from one persona to another at runtime."""
        p1 = _make_persona("buffett", "巴菲特")
        p2 = _make_persona("feynman", "费曼")
        registry = _make_registry([p1, p2])

        mw = PersonaMiddleware(persona=p1, registry=registry)
        assert mw.current_persona.persona_id == "buffett"

        mw.activate("feynman")
        assert mw.current_persona.persona_id == "feynman"

    def test_activate_rebuilds_persona_block(self) -> None:
        """activate() rebuilds the persona injection block."""
        persona = _make_persona("buffett", "巴菲特")
        registry = _make_registry([persona])

        mw = PersonaMiddleware(registry=registry)
        assert mw._persona_block == ""

        mw.activate("buffett")
        assert "巴菲特" in mw._persona_block
        assert "Active Analysis Framework" in mw._persona_block
