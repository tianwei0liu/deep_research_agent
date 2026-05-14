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
        """Build the dynamic budget status block with progressive urgency.

        Implements a 4-tier escalation system to condition the LLM into
        respecting budget constraints from the very first turn:

        - **NORMAL** (>50% budget remaining): Inform + warn about overrun consequences.
        - **ELEVATED** (25-50% budget remaining): Demand wrap-up planning.
        - **CRITICAL** (≤ threshold turns remaining): Demand immediate finalization.
        - **OVERRUN** (past budget): Maximum urgency — force report output.

        Args:
            current_turn: 1-indexed turn number (next turn about to start).

        Returns:
            Formatted budget status string.
        """
        remaining = max(self._max_turns - current_turn, 0)
        used_ratio = current_turn / self._max_turns if self._max_turns > 0 else 1.0

        lines = [
            "## ⏱ Budget Status (auto-injected — YOU MUST OBEY)",
            f"- Current turn: {current_turn} / {self._max_turns}",
            f"- Remaining turns: {remaining}",
        ]

        if current_turn > self._max_turns:
            # OVERRUN — past the hard limit
            lines.append(
                "- Status: 🛑 OVERRUN — You have EXCEEDED your turn budget! "
                "The system will forcibly terminate your session after this "
                "turn. You MUST output a final report RIGHT NOW using only "
                "the findings you already have. Do NOT call any research "
                "tools. Do NOT delegate to workers. Synthesize immediately "
                "and delegate to citation-specialist, or write the report "
                "yourself with citations. ANY other action is FORBIDDEN."
            )
            self._logger.error(
                "Budget OVERRUN: turn %d/%d — LLM exceeded hard limit",
                current_turn,
                self._max_turns,
            )
        elif remaining <= self._critical_threshold:
            # CRITICAL — about to hit the wall
            lines.append(
                f"- Status: 🚨 CRITICAL — Only {remaining} turn(s) left! "
                "You MUST stop ALL research immediately. Do NOT delegate "
                "any new research tasks. Synthesize whatever findings you "
                "have and produce the final report NOW. If you do not "
                "finish within your remaining turns, the system will "
                "forcibly terminate your session and the report will be "
                "INCOMPLETE and TRUNCATED. This is your LAST chance to "
                "produce a quality report."
            )
            self._logger.warning(
                "Budget CRITICAL: turn %d/%d, remaining=%d",
                current_turn,
                self._max_turns,
                remaining,
            )
        elif used_ratio >= 0.5:
            # ELEVATED — past the halfway point
            lines.append(
                f"- Status: ⚠️ ELEVATED — You have used {current_turn} of "
                f"{self._max_turns} turns ({used_ratio:.0%}). You are past "
                "the halfway point. You MUST begin wrapping up your research. "
                "Finish any in-progress tasks, but do NOT start new lines of "
                "investigation. Plan to write your final report within the "
                f"next {remaining} turns. If you exceed the budget, the "
                "system will forcibly terminate your session and produce an "
                "INCOMPLETE report. Prioritize depth over breadth from now on."
            )
            self._logger.info(
                "Budget ELEVATED: turn %d/%d, remaining=%d",
                current_turn,
                self._max_turns,
                remaining,
            )
        else:
            # NORMAL — still have budget, but always warn about consequences
            lines.append(
                f"- Status: ✅ NORMAL — You have {remaining} turns remaining. "
                "Plan your research to complete well within this budget. "
                f"Remember: if you exceed {self._max_turns} turns, the system "
                "will forcibly terminate your session and the report will be "
                "INCOMPLETE. Budget your turns wisely — reserve at least "
                f"{self._critical_threshold + 1} turns for report writing and "
                "citation."
            )
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
