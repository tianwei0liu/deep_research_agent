"""Patched ``ChatDeepSeek`` that preserves ``reasoning_content`` across turns.

DeepSeek V4 models default to "thinking mode", which returns a
``reasoning_content`` field alongside the standard ``content`` in assistant
messages.  When the model also issues tool calls, the API **requires** this
field to be echoed back in subsequent requests.

``langchain-deepseek==1.0.1`` saves the field on the inbound path
(``_create_chat_result``), but drops it on the outbound path
(``_get_request_payload`` → ``_convert_message_to_dict``).  This module
patches that gap by subclassing ``ChatDeepSeek`` and re-injecting the
field after the standard serialization pipeline runs.

See: https://github.com/langchain-ai/langchain/issues/34166

Remove this module once ``langchain-deepseek`` ships a fix upstream.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek

logger = logging.getLogger(__name__)


class PatchedChatDeepSeek(ChatDeepSeek):
    """``ChatDeepSeek`` with ``reasoning_content`` passback fix.

    Overrides ``_get_request_payload`` to re-inject
    ``reasoning_content`` from ``AIMessage.additional_kwargs`` into the
    serialized assistant message dicts before they are sent to the API.

    This is a transparent drop-in replacement for ``ChatDeepSeek``.
    """

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Build the API request payload, preserving ``reasoning_content``.

        Algorithm:
            1. Convert ``input_`` to ``list[BaseMessage]`` via the standard
               ``_convert_input`` pipeline and cache each ``AIMessage``'s
               ``reasoning_content`` keyed by its list index.
            2. Delegate to ``super()._get_request_payload()`` which runs the
               full serialization pipeline (and drops ``reasoning_content``).
            3. Walk through ``payload["messages"]`` and re-inject cached
               ``reasoning_content`` values at matching indices.

        Index alignment is guaranteed because ``_convert_input`` is a pure
        function that preserves message order across invocations.

        Args:
            input_: Raw model input (``list[BaseMessage]`` in agent context).
            stop: Optional stop sequences.
            **kwargs: Additional parameters forwarded to the base class.

        Returns:
            The complete API request payload dict with ``reasoning_content``
            correctly included in assistant messages.
        """
        # --- Step 1: Snapshot reasoning_content before serialization ---
        original_messages = self._convert_input(input_).to_messages()
        reasoning_by_index: dict[int, str] = {}
        for idx, msg in enumerate(original_messages):
            if isinstance(msg, AIMessage):
                rc = msg.additional_kwargs.get("reasoning_content")
                if rc is not None:
                    reasoning_by_index[idx] = rc

        # --- Step 2: Standard payload construction (drops reasoning_content) ---
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        # --- Step 3: Re-inject reasoning_content into serialized dicts ---
        if reasoning_by_index and "messages" in payload:
            payload_messages = payload["messages"]
            for idx, rc in reasoning_by_index.items():
                if idx < len(payload_messages):
                    msg_dict = payload_messages[idx]
                    if msg_dict.get("role") == "assistant":
                        msg_dict["reasoning_content"] = rc
                    else:
                        logger.warning(
                            "Index %d: expected role='assistant' but got '%s'; "
                            "skipping reasoning_content injection",
                            idx,
                            msg_dict.get("role"),
                        )

        return payload
