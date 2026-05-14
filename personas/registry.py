"""Persona registry — discover, load, and validate persona frameworks.

Reads ``registry.yaml`` alongside this module to map persona IDs to
their framework markdown files.  Each persona's ``framework_prompt``
is the raw upstream ``.skill`` content, loaded verbatim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_PERSONAS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PersonaConfig:
    """Validated persona configuration from ``registry.yaml``.

    Attributes:
        persona_id: Unique identifier (e.g. ``buffett``, ``zhangxuefeng``).
        display_name: Human-readable name shown in UI/logs.
        description: One-line description of the persona's focus areas.
        framework_prompt: Raw markdown content from the framework file.
        max_tokens: Soft token budget for the framework prompt.
        applicable_domains: Suggested domains where this persona adds value.
    """

    persona_id: str
    display_name: str
    description: str
    framework_prompt: str
    max_tokens: int
    applicable_domains: list[str] = field(default_factory=list)


class PersonaRegistry:
    """Discovers and loads persona frameworks from the ``personas/`` directory.

    Parses ``registry.yaml`` at construction time and eagerly loads
    every referenced framework markdown file into memory.

    Args:
        registry_path: Path to ``registry.yaml``.  Defaults to
            ``personas/registry.yaml`` alongside this module.
    """

    def __init__(
        self,
        registry_path: Optional[Path] = None,
    ) -> None:
        self._registry_path = registry_path or (_PERSONAS_DIR / "registry.yaml")
        self._personas: dict[str, PersonaConfig] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Parse ``registry.yaml`` and eagerly load each persona's framework."""
        if not self._registry_path.is_file():
            logger.warning("Persona registry not found: %s", self._registry_path)
            return

        with open(self._registry_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        for persona_id, meta in data.get("personas", {}).items():
            framework_file = _PERSONAS_DIR / meta["file"]
            if not framework_file.is_file():
                logger.warning(
                    "Persona framework file missing: %s (skipped)", framework_file
                )
                continue

            framework_text = framework_file.read_text(encoding="utf-8").strip()
            self._personas[persona_id] = PersonaConfig(
                persona_id=persona_id,
                display_name=meta.get("display_name", persona_id),
                description=meta.get("description", ""),
                framework_prompt=framework_text,
                max_tokens=int(meta.get("max_tokens", 800)),
                applicable_domains=meta.get("applicable_domains", []),
            )

        logger.info(
            "Loaded %d persona(s): %s",
            len(self._personas),
            list(self._personas),
        )

    def get(self, persona_id: str) -> Optional[PersonaConfig]:
        """Look up a persona by ID.

        Args:
            persona_id: The persona identifier (e.g. ``buffett``).

        Returns:
            The ``PersonaConfig`` if found, ``None`` otherwise.
        """
        return self._personas.get(persona_id)

    def list_personas(self) -> list[PersonaConfig]:
        """Return all registered personas.

        Returns:
            List of ``PersonaConfig`` instances in registration order.
        """
        return list(self._personas.values())
