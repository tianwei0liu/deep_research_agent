"""BudgetTrackingMiddleware — dynamic per-turn budget injection.

Restores the ``build_dynamic_limits_prompt(current_step, max_steps)``
capability that was lost during the migration to ``create_deep_agent``.

Injects a ``## ⏱ Budget Status`` block at the end of the system message
before **every** Supervisor LLM call, telling the model exactly how many
turns it has consumed and how many remain.  When remaining turns ≤ 3, an
urgency WARNING is injected that instructs the model to stop immediately.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage

from deepagents.middleware._utils import append_to_system_message

logger = logging.getLogger(__name__)

# Urgency threshold: when remaining turns ≤ this value, inject CRITICAL.
_CRITICAL_THRESHOLD: int = 3


class BudgetTrackingMiddleware(AgentMiddleware[Any, Any, Any]):
    """Injects remaining-turn budget into each Supervisor LLM call.

    The middleware counts ``AIMessage`` instances in the conversation
    history to determine the current turn number, then appends a
    dynamic status block to the system message.

    Args:
        max_turns: Maximum number of Supervisor reasoning turns.
        critical_threshold: Remaining-turn count at which to inject
            the CRITICAL stop warning.  Defaults to 3.
    """

    def __init__(
        self,
        *,
        max_turns: int,
        critical_threshold: int = _CRITICAL_THRESHOLD,
    ) -> None:
        super().__init__()
        self._max_turns = max_turns
        self._critical_threshold = critical_threshold
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Budget message construction
    # ------------------------------------------------------------------

    def _build_budget_message(self, current_turn: int) -> str:
        """Build the dynamic budget status block.

        Args:
            current_turn: 1-indexed turn number (next turn about to start).

        Returns:
            Formatted budget status string.
        """
        remaining = max(self._max_turns - current_turn, 0)

        lines = [
            "## ⏱ Budget Status (auto-injected, do NOT ignore)",
            f"- Current turn: {current_turn} / {self._max_turns}",
            f"- Remaining turns: {remaining}",
        ]

        if remaining <= self._critical_threshold:
            lines.append(
                "- Status: 🚨 CRITICAL — You MUST stop ALL research "
                "immediately. Synthesize whatever partial findings you "
                "have and produce the final report NOW. Do NOT delegate "
                "any more tasks."
            )
            self._logger.warning(
                "Budget CRITICAL: turn %d/%d, remaining=%d",
                current_turn,
                self._max_turns,
                remaining,
            )
        else:
            lines.append("- Status: ✅ NORMAL — continue research as planned.")
            self._logger.debug(
                "Budget NORMAL: turn %d/%d, remaining=%d",
                current_turn,
                self._max_turns,
                remaining,
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Turn counting
    # ------------------------------------------------------------------

    @staticmethod
    def _count_ai_messages(messages: list[Any]) -> int:
        """Count AIMessage instances in the conversation history.

        Each AIMessage represents one completed Supervisor reasoning turn.
        The next call will be turn ``count + 1``.

        Args:
            messages: The conversation message list from ModelRequest.

        Returns:
            Number of AIMessage instances found.
        """
        return sum(1 for m in messages if isinstance(m, AIMessage))

    # ------------------------------------------------------------------
    # Middleware hooks
    # ------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Sync: inject budget status into system message before LLM call."""
        current_turn = self._count_ai_messages(request.messages) + 1
        budget_msg = self._build_budget_message(current_turn)
        new_system = append_to_system_message(
            request.system_message, budget_msg
        )
        return handler(request.override(system_message=new_system))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Async: inject budget status into system message before LLM call."""
        current_turn = self._count_ai_messages(request.messages) + 1
        budget_msg = self._build_budget_message(current_turn)
        new_system = append_to_system_message(
            request.system_message, budget_msg
        )
        return await handler(request.override(system_message=new_system))
