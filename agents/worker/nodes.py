"""
Worker Nodes: Encapsulated logic for the Worker graph nodes.
"""

import asyncio
import inspect
import json
import logging
import re
from typing import Any, Literal
import uuid

from google import genai
from google.genai import types
from langsmith import trace
from langsmith.run_trees import RunTree

from langchain_core.runnables import RunnableConfig
from deep_research_agent.agents.utils.tracing import Tracing

from deep_research_agent.agents.utils.cache_manager import get_process_level_cache

from deep_research_agent.config import Settings
from deep_research_agent.agents.worker.schemas import WorkerResult
from deep_research_agent.agents.orchestrator.schemas import TaskStatus
from deep_research_agent.agents.worker.prompts import WorkerPrompts
from deep_research_agent.agents.worker.state import WorkerState
from deep_research_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class WorkerNodes:
    """
    Encapsulates the logic for the Worker LangGraph nodes.
    """

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or Settings.load()

    def _get_client(self, existing_client: genai.Client | None = None) -> genai.Client:
        """Get Gemini client (injected or created from settings)."""
        if existing_client:
            return existing_client
        api_key = self._settings.require_gemini_api_key()
        return genai.Client(api_key=api_key)

    async def _generate_content_traced(
        self,
        client: genai.Client,
        model_id: str,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
        parent_run: RunTree | None = None,
    ) -> types.GenerateContentResponse:
        """Call Gemini generate_content async and record an LLM span in LangSmith."""
        with trace(
            "gemini_generate_content",
            run_type="llm",
            inputs={"model": model_id, "num_contents": len(contents)},
            parent=parent_run,
        ) as run:
            # Use the async client (aio)
            response = await client.aio.models.generate_content(
                model=model_id, contents=contents, config=config
            )
            run.end(
                outputs={
                    "has_candidates": bool(response.candidates),
                    "num_parts": (
                        len(response.candidates[0].content.parts)
                        if response.candidates
                        and getattr(response.candidates[0], "content", None)
                        and response.candidates[0].content.parts
                        else 0
                    ),
                },
            )
        return response

    @staticmethod
    def _get_output_tokens_from_response(response: Any) -> int:
        """Extract output (candidates) token count from Gemini response.usage_metadata."""
        if response is None:
            return 0
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return 0
        return int(getattr(usage, "candidates_token_count", 0) or 0)

    @staticmethod
    def _get_input_tokens_from_response(response: Any) -> int:
        """Extract input (prompt) token count from Gemini response.usage_metadata."""
        if response is None:
            return 0
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return 0
        return int(getattr(usage, "prompt_token_count", 0) or 0)

    @staticmethod
    def _last_model_text_from_contents(contents: list[types.Content]) -> str:
        """Get text from the last model message (for fallback when forced final answer is empty)."""
        text = ""
        for content in reversed(contents):
            if getattr(content, "role", None) != "model":
                continue
            if not getattr(content, "parts", None):
                continue
            for p in content.parts:
                if hasattr(p, "text") and p.text:
                    text += p.text
            if text:
                break
        return text

    def _parse_final_response(
        self,
        response: Any,
        tool_calls_used: int,
        turns: int,
        partial_reason: str | None = None,
        contents_for_fallback: list[types.Content] | None = None,
        output_tokens_used: int = 0,
        input_tokens_used: int = 0,
    ) -> WorkerResult:
        """Extract text from response and build WorkerResult (§4).

        If the LLM returned a JSON dict with ``brief_summary`` and
        ``full_findings`` keys, those are promoted to first-class fields.
        Otherwise the raw text becomes ``full_findings`` and
        ``brief_summary`` is left as ``None``.
        """
        text = ""
        if (
            response
            and response.candidates
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            for p in response.candidates[0].content.parts:
                if hasattr(p, "text") and p.text:
                    text += p.text

        if not text and partial_reason and contents_for_fallback:
            text = self._last_model_text_from_contents(contents_for_fallback)
            if not text:
                return WorkerResult(
                    status=TaskStatus.PARTIAL,
                    brief_summary=partial_reason,
                    metadata={
                        "tool_calls_used": tool_calls_used,
                        "turns": turns,
                        "output_tokens_used": output_tokens_used,
                        "input_tokens_used": input_tokens_used,
                        "caveats": partial_reason,
                    },
                )
            partial_reason = (
                partial_reason
                + "; synthesized from last model turn (forced round was empty)"
            )
        elif not text and partial_reason:
            return WorkerResult(
                status=TaskStatus.PARTIAL,
                brief_summary=partial_reason,
                metadata={
                    "tool_calls_used": tool_calls_used,
                    "turns": turns,
                    "output_tokens_used": output_tokens_used,
                    "input_tokens_used": input_tokens_used,
                    "caveats": partial_reason,
                },
            )

        # Attempt to extract structured JSON from ```json...``` blocks
        brief_summary: str | None = None
        full_findings: str | None = text
        metadata: dict[str, Any] = {
            "tool_calls_used": tool_calls_used,
            "turns": turns,
            "output_tokens_used": output_tokens_used,
            "input_tokens_used": input_tokens_used,
            "sources_used": [f"tavily_search ({tool_calls_used})"],
        }
        if partial_reason:
            metadata["caveats"] = partial_reason

        try:
            json_match = re.search(
                r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text
            )
            if json_match:
                parsed = json.loads(json_match.group(1))
                if isinstance(parsed, dict):
                    brief_summary = parsed.get("brief_summary")
                    full_findings = parsed.get("full_findings", text)
                elif isinstance(parsed, list):
                    metadata["findings_count"] = len(parsed)
        except json.JSONDecodeError:
            pass

        status = TaskStatus.PARTIAL if partial_reason else TaskStatus.COMPLETED
        logger.info(
            "run_worker finished status=%s tool_calls_used=%s turns=%s",
            status.value,
            tool_calls_used,
            turns,
        )
        return WorkerResult(
            status=status,
            brief_summary=brief_summary,
            full_findings=full_findings,
            metadata=metadata,
        )

    def _make_partial_or_failure(
        self,
        findings: Any,
        tool_calls_used: int,
        turns: int,
        last_error: str,
        output_tokens_used: int = 0,
        input_tokens_used: int = 0,
    ) -> WorkerResult:
        has_findings = findings is not None
        return WorkerResult(
            status=TaskStatus.FAILED if not has_findings else TaskStatus.PARTIAL,
            brief_summary=last_error,
            full_findings=str(findings) if has_findings else None,
            metadata={
                "tool_calls_used": tool_calls_used,
                "turns": turns,
                "output_tokens_used": output_tokens_used,
                "input_tokens_used": input_tokens_used,
                "last_error": last_error,
            },
        )

    async def validate_and_init(self, state: WorkerState, config: RunnableConfig) -> dict[str, Any]:
        """§12: Validate spawn input; on success init Gemini config and contents; on failure set result."""
        task = state.get("task")
        limits = state.get("limits")
        tool_names = state.get("tool_names") or []
        model = state.get("model")

        if not task:
            return {
                "result": WorkerResult(
                    status=TaskStatus.FAILED,
                    brief_summary="task input is required",
                    metadata={"reason": "task input is required"},
                )
            }
        if not limits:
            return {
                "result": WorkerResult(
                    status=TaskStatus.FAILED,
                    brief_summary="limits input is required",
                    metadata={"reason": "limits input is required"},
                )
            }

        if not (task.objective and task.output_format):
            logger.info(
                "run_worker validation failed: missing objective or output_format"
            )
            return {
                "result": WorkerResult(
                    status=TaskStatus.FAILED,
                    brief_summary="task.objective and task.output_format are required",
                    metadata={
                        "reason": "task.objective and task.output_format are required"
                    },
                ),
            }

        if not limits.max_tool_calls:
            logger.info("run_worker validation failed: max_tool_calls required")
            return {
                "result": WorkerResult(
                    status=TaskStatus.FAILED,
                    brief_summary="limits.max_tool_calls is required",
                    metadata={"reason": "limits.max_tool_calls is required"},
                ),
            }

        if not tool_names:
            logger.info("run_worker validation failed: tools list empty")
            return {
                "result": WorkerResult(
                    status=TaskStatus.FAILED,
                    brief_summary="tools list cannot be empty",
                    metadata={"reason": "tools list cannot be empty"},
                ),
            }

        declarations, impls = ToolRegistry.resolve(tool_names, self._settings)
        if not declarations:
            logger.info("run_worker validation failed: no valid tools for %s", tool_names)
            return {
                "result": WorkerResult(
                    status=TaskStatus.FAILED,
                    brief_summary="no valid tools resolved",
                    metadata={
                        "reason": "no valid tools resolved",
                        "requested": tool_names,
                    },
                ),
            }

        try:
            # We don't have a reliable way to get the existing client from state if it wasn't there
            # But the facade might have injected it.
            # In Graph execution, state is what we have.
            # If the client is not in state, we create one.
            client = self._get_client(state.get("client"))
        except Exception as e:
            logger.exception("Failed to create Gemini client")
            return {
                "result": WorkerResult(
                    status=TaskStatus.FAILED,
                    brief_summary=f"client init failed: {e}",
                    metadata={"reason": "client init failed", "last_error": str(e)},
                ),
            }

        model_id = model or self._settings.worker_model
        
        # 1. Static System Prompt (Cacheable)
        static_system_instruction = WorkerPrompts.build_static_system_prompt()
        
        # 2. Dynamic Task Instructions (User Context)
        task_instructions = WorkerPrompts.build_task_instructions(task, limits)

        gemini_tools = types.Tool(function_declarations=declarations)
        gemini_tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="AUTO")
        )

        # Attempt to use/create a shared process-level cache for this worker configuration
        cached_name = None
        if self._settings.worker_context_caching_enabled:
            cached_name = await get_process_level_cache(
                client=client,
                model=model_id,
                base_name="worker",
                system_instruction=static_system_instruction,
                tools=[gemini_tools],
                tool_config=gemini_tool_config,
                ttl_minutes=60,
            )

        if cached_name:
            logger.info("Worker using cached content: %s", cached_name)
            gemini_config = types.GenerateContentConfig(
                cached_content=cached_name,
                temperature=self._settings.worker_temperature,
                thinking_config=types.ThinkingConfig(thinking_level=self._settings.worker_thinking_level)
            )
        else:
            logger.info("Worker cache miss, using standard context.")
            gemini_config = types.GenerateContentConfig(
                system_instruction=static_system_instruction,
                tools=[gemini_tools],
                tool_config=gemini_tool_config,
                temperature=self._settings.worker_temperature,  # Deterministic for factual research workers (§6)
                thinking_config=types.ThinkingConfig(thinking_level=self._settings.worker_thinking_level)
            )

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=task_instructions),
                    types.Part.from_text(
                        text="\n\nBegin your research task. Use your internal thinking process to decide what to search, synthesize findings, or format the output. Then emit only the requested tool call or final JSON output. Conclude as soon as you have enough to fill the output; do not use all available turns."
                    )
                ],
            )
        ]

        logger.info(
            "run_worker starting model=%s tools=%s max_tool_calls=%s max_turns=%s max_output_tokens=%s",
            model_id,
            tool_names,
            limits.max_tool_calls,
            limits.max_turns,
            limits.max_output_tokens,
        )

        return {
            "task": task,
            "limits": limits,
            "declarations": declarations,
            "impls": impls,
            "gemini_config": gemini_config,
            "model_id": model_id,
            "contents": contents,
            "client": client,
            "tool_calls_used": 0,
            "turns": 0,
            "output_tokens_used": 0,
            "input_tokens_used": 0,
            "last_error": None,
            "last_response": None,
            "last_function_calls": [],
            "forced_final_round": False,
            "partial_reason": None,
        }

    async def reason_act(self, state: WorkerState, config: RunnableConfig) -> dict[str, Any]:
        """§5: One ReAct turn: optionally inject limit message, call Gemini, store response and function_calls."""
        # Bridge LangGraph config → langsmith parent so LLM spans nest under this node
        parent_run = Tracing.get_parent_run(config)
        contents = state["contents"]
        limits = state["limits"]
        tool_calls_used = state["tool_calls_used"]
        output_tokens_used = state.get("output_tokens_used", 0)
        input_tokens_used = state.get("input_tokens_used", 0)
        turns = state["turns"] + 1
        client = state["client"]
        model_id = state["model_id"]
        config = state["gemini_config"]

        if limits.max_turns is not None and turns > limits.max_turns:
            new_contents = list(contents)
            new_contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text="You have reached the turn limit. Output your final answer now in the requested format."
                        )
                    ],
                )
            )
            try:
                response = await self._generate_content_traced(
                    client, model_id, new_contents, config, parent_run=parent_run
                )
            except Exception as e:
                return {
                    "turns": turns,
                    "last_error": str(e),
                    "result": self._make_partial_or_failure(
                        None,
                        tool_calls_used,
                        turns,
                        str(e),
                        output_tokens_used,
                        input_tokens_used,
                    ),
                }
            output_tokens_used += self._get_output_tokens_from_response(response)
            input_tokens_used = self._get_input_tokens_from_response(response)
            return {
                "contents": new_contents,
                "turns": turns,
                "output_tokens_used": output_tokens_used,
                "input_tokens_used": input_tokens_used,
                "last_response": response,
                "last_function_calls": [],
                "forced_final_round": True,
                "partial_reason": "max_turns reached",
            }

        # Retry loop for generation to handle transient errors like MALFORMED_FUNCTION_CALL
        # Max retries = 3 (initial attempt + 2 retries)
        max_retries = 3
        response = None
        last_exception = None

        for attempt in range(max_retries):
            try:
                response = await self._generate_content_traced(
                    client, model_id, contents, config, parent_run=parent_run
                )
                
                # Check for empty content with specific finish reasons
                candidate = response.candidates[0] if response.candidates else None
                if not candidate or not candidate.content or not candidate.content.parts:
                    finish_reason = getattr(candidate, "finish_reason", "unknown")
                    # FinishReason enum values are integers, but string representation is useful for logging
                    # Common bad finish reasons: MALFORMED_FUNCTION_CALL (4), RECITATION (2), SAFETY (3)
                    # We retry on these. STOP (1) and MAX_TOKENS (2) are normal (though MAX_TOKENS logic is handled later).
                    
                    # Log the issue
                    logger.warning(
                        "Gemini response empty/blocked (attempt %d/%d). Finish reason: %s",
                        attempt + 1,
                        max_retries,
                        finish_reason
                    )
                    
                    # If it's the last attempt, we let it fall through to existing logic 
                    # which will likely produce a "partial" result or handle it.
                    if attempt < max_retries - 1:
                        # Optional: Add a small delay or slight config tweak if needed
                        await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff-ish
                        continue
                
                # If we got here, response seems valid enough to proceed
                break
            except Exception as e:
                logger.exception("Gemini generate_content failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
                last_exception = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                else:
                    return {
                        "turns": turns,
                        "last_error": str(e),
                        "result": WorkerResult(
                            status=TaskStatus.FAILED,
                            brief_summary=f"Gemini generation failed: {e}",
                            metadata={
                                "tool_calls_used": tool_calls_used,
                                "turns": turns,
                                "output_tokens_used": output_tokens_used,
                                "input_tokens_used": input_tokens_used,
                                "last_error": str(e),
                            },
                        ),
                    }



        if (
            limits.max_output_tokens is not None
            and output_tokens_used >= limits.max_output_tokens
        ):
            summary_contents = list(contents)
            candidate = response.candidates[0] if response.candidates else None
            if candidate and candidate.content:
                summary_contents.append(candidate.content)
            summary_contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text="You have reached the output token limit. Summarize your findings so far into the requested output format in one concise response. Do not call any tools; output only your final answer."
                        )
                    ],
                )
            )
            try:
                summary_response = await self._generate_content_traced(
                    client, model_id, summary_contents, config, parent_run=parent_run
                )
            except Exception as e:
                logger.warning(
                    "Token-limit summarization call failed, using last response: %s", e
                )
                summary_response = response
            else:
                output_tokens_used += self._get_output_tokens_from_response(
                    summary_response
                )
                input_tokens_used = self._get_input_tokens_from_response(
                    summary_response
                )
            return {
                "contents": summary_contents,
                "turns": turns,
                "output_tokens_used": output_tokens_used,
                "input_tokens_used": input_tokens_used,
                "last_response": summary_response,
                "last_function_calls": [],
                "forced_final_round": True,
                "partial_reason": "max_output_tokens reached",
            }

        candidate = response.candidates[0] if response.candidates else None
        if not candidate or not candidate.content or not candidate.content.parts:
            return {
                "turns": turns,
                "output_tokens_used": output_tokens_used,
                "last_response": response,
                "last_function_calls": [],
            }

        parts = candidate.content.parts
        function_calls = [p for p in parts if getattr(p, "function_call", None)]

        new_contents = list(contents)
        if function_calls:
            new_contents.append(candidate.content)
            logger.debug(
                "turn=%d executing %d tool call(s)",
                turns,
                min(len(function_calls), limits.max_tool_calls - tool_calls_used),
            )

        return {
            "contents": new_contents,
            "turns": turns,
            "output_tokens_used": output_tokens_used,
            "input_tokens_used": input_tokens_used,
            "last_response": response,
            "last_function_calls": function_calls,
            "forced_final_round": False,
            "partial_reason": None,
        }

    async def execute_tools(self, state: WorkerState, config: RunnableConfig) -> dict[str, Any]:
        """§5.1: Run tool batch, append results to contents; enforce remaining budget and optional forced final round."""
        # Bridge LangGraph config → langsmith parent so tool spans nest under this node
        parent_run = Tracing.get_parent_run(config)
        contents = state["contents"]
        limits = state["limits"]
        tool_calls_used = state["tool_calls_used"]
        output_tokens_used = state.get("output_tokens_used", 0)
        input_tokens_used = state.get("input_tokens_used", 0)
        last_function_calls = state["last_function_calls"]
        impls = state["impls"]

        remaining = limits.max_tool_calls - tool_calls_used
        to_run = last_function_calls[:remaining]
        response_parts = []
        last_error = state.get("last_error")
        used = tool_calls_used

        # Prepare tasks for parallel execution
        tasks = []
        for fc in to_run:
            name = fc.function_call.name
            args = dict(fc.function_call.args) if fc.function_call.args else {}
            logger.debug("Tool Call: %s(args=%s)", name, args)
            impl = impls.get(name)
            
            async def run_tool(name: str, args: dict[str, Any], impl: Any, parent: RunTree | None = None) -> Any:
                if impl is None:
                    logger.warning("Unknown tool requested: %s", name)
                    return {"error": f"Unknown tool: {name}"}
                try:
                    with trace(name, run_type="tool", inputs=args, parent=parent):
                        if inspect.iscoroutinefunction(impl):
                            return await impl(**args)
                        else:
                            return await asyncio.to_thread(impl, **args)
                except Exception as e:
                    logger.warning("Tool %s failed: %s", name, e)
                    return {"error": str(e)}

            tasks.append(run_tool(name, args, impl, parent=parent_run))
            used += 1

        # Execute all tools in parallel
        results = await asyncio.gather(*tasks)

        # Process results in order
        for i, fc in enumerate(to_run):
             name = fc.function_call.name
             result = results[i]
             if isinstance(result, dict) and "error" in result:
                 last_error = result["error"]
             
             response_parts.append(
                types.Part.from_function_response(name=name, response=result)
             )

        for fc in last_function_calls[remaining:]:
            response_parts.append(
                types.Part.from_function_response(
                    name=fc.function_call.name,
                    response={
                        "error": "Limit reached. Output your final answer now in the requested format.",
                        "result": None,
                    },
                )
            )

        new_contents = list(contents)
        new_contents.append(types.Content(role="user", parts=response_parts))

        need_forced_final = used >= limits.max_tool_calls
        if need_forced_final:
            client = state["client"]
            model_id = state["model_id"]
            config = state["gemini_config"]
            turns = state["turns"]
            try:
                response = await self._generate_content_traced(
                    client, model_id, new_contents, config, parent_run=parent_run
                )
            except Exception as e:
                return {
                    "contents": new_contents,
                    "tool_calls_used": used,
                    "last_error": str(e),
                    "result": self._make_partial_or_failure(
                        None,
                        used,
                        turns,
                        str(e),
                        output_tokens_used,
                        input_tokens_used,
                    ),
                }
            output_tokens_used += self._get_output_tokens_from_response(response)
            input_tokens_used = self._get_input_tokens_from_response(response)
            result = self._parse_final_response(
                response,
                used,
                turns,
                partial_reason="max_tool_calls reached",
                contents_for_fallback=new_contents,
                output_tokens_used=output_tokens_used,
                input_tokens_used=input_tokens_used,
            )
            return {
                "contents": new_contents,
                "tool_calls_used": used,
                "output_tokens_used": output_tokens_used,
                "input_tokens_used": input_tokens_used,
                "last_error": last_error,
                "last_response": response,
                "last_function_calls": [],
                "forced_final_round": True,
                "partial_reason": "max_tool_calls reached",
                "result": result,
            }

        return {
            "contents": new_contents,
            "tool_calls_used": used,
            "output_tokens_used": output_tokens_used,
            "input_tokens_used": input_tokens_used,
            "last_error": last_error,
            "last_function_calls": [],
        }

    async def parse_final(self, state: WorkerState, config: RunnableConfig) -> dict[str, Any]:
        """§4: Build WorkerResult from last_response / contents and set result."""
        response = state.get("last_response")
        tool_calls_used = state["tool_calls_used"] or 0
        turns = state["turns"] or 0
        output_tokens_used = state.get("output_tokens_used", 0)
        input_tokens_used = state.get("input_tokens_used", 0)
        partial_reason = state.get("partial_reason")
        contents = state.get("contents")

        result = self._parse_final_response(
            response,
            tool_calls_used,
            turns,
            partial_reason=partial_reason,
            contents_for_fallback=contents,
            output_tokens_used=output_tokens_used,
            input_tokens_used=input_tokens_used,
        )
        return {"result": result}
