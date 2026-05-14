"""Persona framework registry for the deep research agent.

Provides:
- ``PersonaRegistry`` — discover, load, and validate persona frameworks.
- ``PersonaConfig``   — validated persona configuration dataclass.
"""

from deep_research_agent.personas.registry import PersonaConfig, PersonaRegistry

__all__ = ["PersonaConfig", "PersonaRegistry"]
