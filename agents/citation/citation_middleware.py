"""CitationDataMiddleware -- auto-inject Worker findings + L1 validation gate.

This middleware intercepts ``task(citation-specialist, ...)`` calls at
the Supervisor level and performs three functions:

1. **Data injection**: Extracts Worker findings from Supervisor state
   and appends them to the CitationAgent input description.
2. **L1 validation gate**: Validates the CitationAgent output against
   L1 structural rules (dangling citations, orphan sources, etc.).
3. **Self-correction retry**: On L1 ERROR, re-invokes the CitationAgent
   with correction instructions (up to MAX_RETRIES times).

The Supervisor never sees an L1-failed report.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deep_research_agent.agents.citation.models import (
    Severity,
    WorkerOutput,
)
from deep_research_agent.agents.citation.structure_validator import (
    CitationStructureValidator,
)

logger = logging.getLogger(__name__)


class CitationDataMiddleware(AgentMiddleware):
    """Intercepts citation-specialist task calls for auto-injection and L1 gating.

    Registered in ``create_deep_agent(middleware=[CitationDataMiddleware()])``
    on the Supervisor middleware stack. Only fires for
    ``task(subagent_type="citation-specialist")`` calls; all other tool
    calls pass through unmodified.
    """

    DEFAULT_MAX_RETRIES: int = 5

    def __init__(self, *, max_retries: int | None = None) -> None:
        self._max_retries = (
            max_retries if max_retries is not None else self.DEFAULT_MAX_RETRIES
        )
        self._logger = logging.getLogger(__name__)
        self._validator = CitationStructureValidator()

    # ------------------------------------------------------------------
    # Sync entry point
    # ------------------------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Sync interception of tool calls.

        For citation-specialist tasks: inject findings -> validate -> retry.
        For all other tools: pass through.
        """
        if not self._is_citation_specialist_task(request):
            return handler(request)

        # --- Phase 1: Inject Worker findings ---
        enriched_request = self._enrich_request(request)

        # --- Phase 2: Invoke + Validate + Retry ---
        return self._invoke_and_validate(enriched_request, handler)

    # ------------------------------------------------------------------
    # Async entry point
    # ------------------------------------------------------------------

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async interception of tool calls.

        Same logic as sync version but with ``await handler(...)``.
        """
        if not self._is_citation_specialist_task(request):
            return await handler(request)

        enriched_request = self._enrich_request(request)
        return await self._ainvoke_and_validate(enriched_request, handler)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_citation_specialist_task(request: ToolCallRequest) -> bool:
        """Check if this is a task call targeting citation-specialist."""
        tool_call = request.tool_call
        return (
            tool_call.get("name") == "task"
            and tool_call.get("args", {}).get("subagent_type") == "citation-specialist"
        )

    def _enrich_request(self, request: ToolCallRequest) -> ToolCallRequest:
        """Inject Worker findings into the task description."""
        worker_findings = self._extract_worker_findings(request.state)
        if not worker_findings:
            self._logger.debug("No Worker findings found in state")
            return request

        original_desc = request.tool_call["args"]["description"]
        enriched_desc = (
            f"{original_desc}\n\n"
            f"## WORKER FINDINGS (auto-injected)\n\n"
            f"{worker_findings}"
        )
        modified_call = {
            **request.tool_call,
            "args": {**request.tool_call["args"], "description": enriched_desc},
        }
        return request.override(tool_call=modified_call)

    def _invoke_and_validate(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Invoke handler, validate L1, retry on error (sync)."""
        result = handler(request)
        report = self._extract_report(result)
        l1_result = self._validator.validate(report)

        if l1_result.is_valid or l1_result.error_count == 0:
            self._logger.info("L1 validation passed on first attempt")
            return result

        # --- Retry loop ---
        worker_findings = self._extract_worker_findings(request.state)
        for attempt in range(self._max_retries):
            self._logger.warning(
                "L1 validation failed (attempt %d/%d), retrying: %d errors",
                attempt + 1,
                self._max_retries,
                l1_result.error_count,
            )
            correction_desc = self._build_correction_description(
                l1_result, report, worker_findings,
                attempt=attempt,
                max_retries=self._max_retries,
            )
            retry_call = {
                **request.tool_call,
                "args": {
                    **request.tool_call["args"],
                    "description": correction_desc,
                },
            }
            retry_request = request.override(tool_call=retry_call)
            result = handler(retry_request)
            report = self._extract_report(result)
            l1_result = self._validator.validate(report)

            if l1_result.is_valid or l1_result.error_count == 0:
                self._logger.info("L1 validation passed after retry %d", attempt + 1)
                return result

        # --- Retries exhausted ---
        self._logger.warning(
            "L1 retry exhausted after %d attempts, appending warning",
            self._max_retries,
        )
        return self._append_warning(result, l1_result)

    async def _ainvoke_and_validate(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Invoke handler, validate L1, retry on error (async)."""
        result = await handler(request)
        report = self._extract_report(result)
        l1_result = self._validator.validate(report)

        if l1_result.is_valid or l1_result.error_count == 0:
            self._logger.info("L1 validation passed on first attempt")
            return result

        worker_findings = self._extract_worker_findings(request.state)
        for attempt in range(self._max_retries):
            self._logger.warning(
                "L1 validation failed (attempt %d/%d), retrying: %d errors",
                attempt + 1,
                self._max_retries,
                l1_result.error_count,
            )
            correction_desc = self._build_correction_description(
                l1_result, report, worker_findings,
                attempt=attempt,
                max_retries=self._max_retries,
            )
            retry_call = {
                **request.tool_call,
                "args": {
                    **request.tool_call["args"],
                    "description": correction_desc,
                },
            }
            retry_request = request.override(tool_call=retry_call)
            result = await handler(retry_request)
            report = self._extract_report(result)
            l1_result = self._validator.validate(report)

            if l1_result.is_valid or l1_result.error_count == 0:
                self._logger.info("L1 validation passed after retry %d", attempt + 1)
                return result

        self._logger.warning(
            "L1 retry exhausted after %d attempts, appending warning",
            self._max_retries,
        )
        return self._append_warning(result, l1_result)

    @staticmethod
    def _extract_worker_findings(state: Any) -> str:
        """Extract all WorkerOutput JSONs from ToolMessages in state.

        Args:
            state: The Supervisor agent state dict.

        Returns:
            Concatenated JSON strings of Worker findings, separated by ``---``.
            Empty string if no valid WorkerOutput found.
        """
        messages = state.get("messages", []) if isinstance(state, dict) else []
        findings_parts: list[str] = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = msg.content
                if isinstance(content, str) and '"findings"' in content:
                    try:
                        WorkerOutput.model_validate_json(content)
                        findings_parts.append(content)
                    except Exception:
                        # Not a valid WorkerOutput, skip
                        pass
        return "\n---\n".join(findings_parts)

    @staticmethod
    def _extract_report(result: ToolMessage | Command[Any]) -> str:
        """Extract the report text from a handler result.

        Uses duck typing (``hasattr(result, 'update')``) instead of
        ``isinstance(result, Command)`` so that both real Command
        objects and test mocks are handled correctly.

        Args:
            result: The Command or ToolMessage returned by handler.

        Returns:
            The report text content.
        """
        # Command-like: has .update dict with 'messages' key
        if hasattr(result, "update") and isinstance(result.update, dict):
            messages = result.update.get("messages", [])
            if messages and hasattr(messages[0], "content"):
                return messages[0].content
        # ToolMessage-like: has .content directly
        if hasattr(result, "content"):
            return result.content
        return str(result)

    @staticmethod
    def _build_correction_description(
        l1_result: Any,
        original_report: str,
        worker_findings: str,
        *,
        attempt: int = 0,
        max_retries: int = 5,
    ) -> str:
        """Build a correction description for retry.

        Only includes ERROR-level issues (not warnings).  Appends a
        budget status line so the Citation Specialist knows how many
        retries remain.

        Args:
            l1_result: The L1ValidationResult with issues.
            original_report: The CitationAgent output that failed L1.
            worker_findings: Worker findings JSON string.
            attempt: Current retry attempt index (0-based).
            max_retries: Total retry budget.

        Returns:
            Formatted correction description for the retry request.
        """
        error_issues = [
            i for i in l1_result.issues if i.severity == Severity.ERROR
        ]
        issues_text = "\n".join(
            f"- {i.rule_id}: {i.message}" for i in error_issues
        )

        remaining = max_retries - attempt - 1

        parts = [
            "## CITATION CORRECTION REQUIRED\n",
            "The following citation issues were found in your output:\n",
            issues_text,
            "\nPlease fix these issues and output the corrected report.\n",
            f"## \u23f1 Citation Budget\n",
            f"- Retry attempt: {attempt + 1} / {max_retries}\n",
            f"- Remaining retries: {remaining}\n",
        ]

        if remaining <= 1:
            parts.append(
                "- \u26a0\ufe0f This is your LAST chance. Fix as many issues as "
                "possible in this attempt.\n"
            )

        parts.extend([
            "\n## ORIGINAL REPORT (with issues)\n",
            original_report,
        ])

        if worker_findings:
            parts.extend([
                "\n\n## WORKER FINDINGS (auto-injected)\n\n",
                worker_findings,
            ])

        return "\n".join(parts)

    @staticmethod
    def _append_warning(
        result: ToolMessage | Command[Any],
        l1_result: Any,
    ) -> ToolMessage | Command[Any]:
        """Append a citation warning to the report when retries are exhausted.

        Args:
            result: The handler result with the invalid report.
            l1_result: The L1ValidationResult with unresolved issues.

        Returns:
            Modified result with warning appended to content.
        """
        error_issues = [
            i for i in l1_result.issues if i.severity == Severity.ERROR
        ]
        issue_details = ", ".join(
            f"[{i.message.split('[')[1].split(']')[0]}]"
            if "[" in i.message else i.rule_id
            for i in error_issues
        )

        warning = (
            f"\n\n> ⚠️ **Citation Notice**: Some citation inconsistencies were "
            f"detected but could not be automatically resolved. "
            f"Citations {issue_details} may not have matching source entries."
        )

        # Command-like: has .update dict with 'messages' key
        if hasattr(result, "update") and isinstance(result.update, dict):
            messages = result.update.get("messages", [])
            if messages and hasattr(messages[0], "content"):
                original_content = messages[0].content
                tool_call_id = messages[0].tool_call_id if hasattr(messages[0], "tool_call_id") else "unknown"
                new_msg = ToolMessage(
                    content=original_content + warning,
                    tool_call_id=tool_call_id,
                )
                result.update["messages"] = [new_msg]
        elif hasattr(result, "content"):
            result.content += warning

        return result
