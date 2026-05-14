
"""PersonaMiddleware -- inject a cognitive framework into Supervisor prompts.

When a persona is active, appends a ``## 🧠 Active Analysis Framework``
block to the Supervisor's system message before every LLM call.  This
influences how the Supervisor decomposes tasks and synthesizes reports,
while Workers and Citation Specialist remain unaffected (they run in
isolated subagent graphs with their own system prompts).

Supports three activation modes:

1. **Build-time**: Pass ``persona=PersonaConfig(...)`` at construction
   (backward-compatible with ``--persona`` CLI flag).
2. **Runtime (tool)**: The ``activate_persona`` tool calls
   ``activate(persona_id)`` during graph execution.
3. **State restoration**: On graph rebuild (multi-turn), reads
   ``active_persona_id`` from the persisted ``AgentState`` to
   restore the previously activated persona.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, NotRequired, Optional

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents.middleware._utils import append_to_system_message

from deep_research_agent.personas.registry import PersonaConfig, PersonaRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema — persisted by the checkpointer alongside messages
# ---------------------------------------------------------------------------

class PersonaMiddlewareState(AgentState):
    """Extended agent state with persona tracking.

    The ``active_persona_id`` field is persisted by the checkpointer,
    enabling the middleware to restore the active persona across graph
    rebuilds in multi-turn conversations.
    """

    active_persona_id: NotRequired[Annotated[str | None, PrivateStateAttr]]
    """The currently active persona ID, or ``None`` if no persona is active."""


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class PersonaMiddleware(AgentMiddleware):  # type: ignore[type-arg]
    """Injects a persona's cognitive framework into the Supervisor system prompt.

    The framework is appended as a structured block that instructs the
    Supervisor to decompose tasks and write reports through the lens of
    the selected persona.

    State Persistence:
        Defines ``PersonaMiddlewareState`` with an ``active_persona_id``
        field.  When a persona is activated via the ``activate_persona``
        tool, the middleware's ``awrap_tool_call`` hook returns a
        ``Command(update={"active_persona_id": ...})`` that persists the
        choice in the graph state.  On subsequent turns (after graph
        rebuild), ``awrap_model_call`` reads this field to restore the
        persona without requiring re-activation.

    Args:
        persona: Optional initial persona to activate at build time.
            If ``None``, the middleware starts inactive (passthrough).
        registry: Optional ``PersonaRegistry`` for runtime activation
            via ``activate(persona_id)``.  Required for the interactive
            Skills Discovery flow.
    """

    state_schema = PersonaMiddlewareState

    def __init__(
        self,
        *,
        persona: Optional[PersonaConfig] = None,
        registry: Optional[PersonaRegistry] = None,
    ) -> None:
        super().__init__()
        self._persona = persona
        self._registry = registry
        self._persona_block: str = self._build_persona_block()

    @property
    def active(self) -> bool:
        """Whether a persona is currently active."""
        return self._persona is not None

    @property
    def current_persona(self) -> Optional[PersonaConfig]:
        """The currently active persona config, or ``None``."""
        return self._persona

    # ------------------------------------------------------------------
    # Runtime activation / deactivation
    # ------------------------------------------------------------------

    def activate(self, persona_id: str) -> bool:
        """Activate a persona at runtime by ID.

        Looks up the persona in the registry and rebuilds the injection
        block.  If the persona is not found, logs a warning and returns
        ``False``.

        Args:
            persona_id: The persona identifier (e.g. ``"buffett"``).

        Returns:
            ``True`` if activation succeeded, ``False`` otherwise.

        Raises:
            RuntimeError: If no registry was provided at construction.
        """
        if self._registry is None:
            raise RuntimeError(
                "Cannot activate persona at runtime: no PersonaRegistry "
                "was provided to PersonaMiddleware."
            )

        config = self._registry.get(persona_id)
        if config is None:
            logger.warning(
                "Persona '%s' not found in registry, activation failed",
                persona_id,
            )
            return False

        self._persona = config
        self._persona_block = self._build_persona_block()
        logger.info(
            "Persona dynamically activated: %s (%s)",
            config.display_name,
            persona_id,
        )
        return True

    def deactivate(self) -> None:
        """Deactivate the current persona, resetting to objective mode."""
        if self._persona is not None:
            logger.info(
                "Persona deactivated: %s",
                self._persona.display_name,
            )
        self._persona = None
        self._persona_block = ""

    # ------------------------------------------------------------------
    # State-based persona restoration
    # ------------------------------------------------------------------

    def _restore_from_state(self, state: dict[str, Any]) -> None:
        """Restore persona from persisted state if not already active.

        Called by ``awrap_model_call`` on every LLM invocation to ensure
        the persona survives graph rebuilds in multi-turn conversations.

        Args:
            state: The current ``AgentState`` dict.
        """
        if self.active:
            # Already activated (build-time or earlier in this turn)
            return

        persisted_id = state.get("active_persona_id")
        if not persisted_id:
            return

        if self._registry is None:
            logger.warning(
                "State has active_persona_id='%s' but no registry to "
                "resolve it; persona cannot be restored",
                persisted_id,
            )
            return

        success = self.activate(persisted_id)
        if success:
            logger.info(
                "Persona restored from checkpointed state: %s",
                persisted_id,
            )
        else:
            logger.warning(
                "Failed to restore persona '%s' from state",
                persisted_id,
            )

    # ------------------------------------------------------------------
    # Persona block construction
    # ------------------------------------------------------------------

    def _build_persona_block(self) -> str:
        """Build the persona injection block for the system prompt.

        Returns:
            Formatted persona framework string, or empty string if inactive.
        """
        if self._persona is None:
            return ""

        return (
            f"## 🧠 Active Analysis Framework: "
            f"{self._persona.display_name}\n\n"
            f"The user has explicitly requested analysis through the "
            f"following cognitive framework. You MUST:\n"
            f"1. Decompose tasks using the mental models and analysis "
            f"dimensions defined below.\n"
            f"2. Instruct workers with task descriptions that reflect "
            f"this framework's priorities.\n"
            f"3. Write the final report in the voice and style "
            f"described below.\n"
            f"4. If a sub-topic falls outside this framework's scope, "
            f"acknowledge it explicitly rather than forcing a fit.\n\n"
            f"---\n\n"
            f"{self._persona.framework_prompt}\n\n"
            f"---\n\n"
            f"> ⚠️ DISCLAIMER: This analysis uses an AI-simulated "
            f"decision framework. It does not represent the views of "
            f"any real individual and does not constitute professional "
            f"advice."
        )

    # ------------------------------------------------------------------
    # Middleware hooks
    # ------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Sync: inject persona framework before LLM call."""
        self._restore_from_state(request.state)

        if not self.active:
            return handler(request)

        new_system = append_to_system_message(
            request.system_message, self._persona_block
        )
        logger.debug(
            "PersonaMiddleware injected '%s' framework",
            self._persona.display_name,  # type: ignore[union-attr]
        )
        return handler(request.override(system_message=new_system))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Async: inject persona framework before LLM call."""
        self._restore_from_state(request.state)

        if not self.active:
            return await handler(request)

        new_system = append_to_system_message(
            request.system_message, self._persona_block
        )
        logger.debug(
            "PersonaMiddleware injected '%s' framework",
            self._persona.display_name,  # type: ignore[union-attr]
        )
        return await handler(request.override(system_message=new_system))

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[..., Awaitable[Any]],
    ) -> Any:
        """Intercept activate_persona tool calls to persist state.

        After the tool executes successfully, returns a ``Command`` that
        updates ``active_persona_id`` in the graph state.  This ensures
        the persona choice survives checkpointing and graph rebuilds.

        For all other tools, delegates to the handler unchanged.

        Args:
            request: The tool call request.
            handler: The next handler in the middleware chain.

        Returns:
            ``ToolMessage`` for non-persona tools, or ``Command`` with
            state update for successful persona activations.
        """
        result = await handler(request)

        tool_name = request.tool_call.get("name", "")
        if tool_name != "activate_persona":
            return result

        # Only persist if the activation actually succeeded
        if not self.active:
            return result

        persona_id = request.tool_call.get("args", {}).get("persona_id", "")
        logger.info(
            "Persisting active_persona_id='%s' to graph state",
            persona_id,
        )

        # Return Command to update state.  The ToolMessage content is
        # preserved in the update's messages field so the Supervisor
        # sees the activation confirmation.
        if isinstance(result, ToolMessage):
            return Command(
                update={
                    "active_persona_id": persona_id,
                    "messages": [result],
                },
            )

        # Fallback: result is already a Command or unexpected type
        return result
