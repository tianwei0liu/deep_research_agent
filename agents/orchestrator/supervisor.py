"""
Supervisor Node: The central brain of the dynamic architecture.
Refactored to use native `google.genai` SDK and fix message coalescing.
Added deep debug logging for contents and ID inspection.
Implements Map-Reduce pattern for Parallel Execution (v2.0).
Now encapsulated in Supervisor class.
"""

import logging
import asyncio
import time
from collections import defaultdict
import uuid
import inspect
from typing import List, Any, Optional, Dict, Tuple

from google import genai
from google.genai import types
from deep_research_agent.agents.utils.tracing import Tracing

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, BaseMessage

from deep_research_agent.config import Settings
from deep_research_agent.agents.orchestrator.schemas import TaskStatus
from deep_research_agent.agents.orchestrator.state import OrchestratorState
from deep_research_agent.agents.orchestrator.prompts import OrchestratorPrompts

# Tool Registry


class Supervisor:
    """
    The Supervisor Node (Native GenAI Implementation).
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.logger = logging.getLogger(__name__)
        # self.logger.setLevel(logging.DEBUG) # Let configuration handle levels
        self.settings = settings or Settings.load()

    def _convert_langchain_to_genai(self, messages: List[BaseMessage]) -> List[types.Content]:
        """Converts a list of LangChain messages to Google GenAI Content objects."""
        contents = []
        
        i = 0
        while i < len(messages):
            msg = messages[i]
            
            if isinstance(msg, SystemMessage):
                i += 1
                continue
                
            elif isinstance(msg, HumanMessage):
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=msg.content)]))
                i += 1
                
            elif isinstance(msg, AIMessage):
                parts = []
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))
                
                thought_signatures = msg.additional_kwargs.get("thought_signatures", {})
                
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        fc_id = tc.get("id")
                        self.logger.debug(f"Creating FunctionCall for {tc['name']} with ID: {fc_id}")
                        
                        fc_kwargs = {
                            "name": tc["name"],
                            "args": tc["args"],
                            "id": fc_id if fc_id else None
                        }
                        
                        part_kwargs = {"function_call": types.FunctionCall(**fc_kwargs)}
                        if fc_id and fc_id in thought_signatures:
                            part_kwargs["thought_signature"] = thought_signatures[fc_id]
                            
                        try:
                            part = types.Part(**part_kwargs)
                            parts.append(part)
                        except Exception as e:
                            self.logger.error(f"Failed to create FunctionCall with ID: {e}")
                            parts.append(types.Part.from_function_call(
                                name=tc["name"],
                                args=tc["args"]
                            ))

                contents.append(types.Content(role="model", parts=parts))
                i += 1
                
            elif isinstance(msg, ToolMessage):
                tool_parts = []
                while i < len(messages) and isinstance(messages[i], ToolMessage):
                    t_msg = messages[i]
                    
                    resp_id = t_msg.tool_call_id
                    self.logger.debug(f"Creating FunctionResponse for {t_msg.name} with ID: {resp_id}")

                    try:
                        part = types.Part(
                            function_response=types.FunctionResponse(
                                name=t_msg.name,
                                response={"result": t_msg.content},
                                id=resp_id if resp_id else None
                            )
                        )
                        tool_parts.append(part)
                    except Exception as e:
                        self.logger.error(f"Failed to create FunctionResponse with ID: {e}")
                        tool_parts.append(types.Part.from_function_response(
                            name=t_msg.name,
                            response={"result": t_msg.content}
                        ))
                    
                    i += 1
                
                contents.append(types.Content(role="user", parts=tool_parts))
                
            else:
                self.logger.warning(f"Unknown message type: {type(msg)}")
                i += 1

        return contents

    def _convert_genai_response_to_langchain(self, response: Any) -> AIMessage:
        """Converts a Gemini GenerateContentResponse to a LangChain AIMessage with tool_calls."""
        if not response.candidates:
            return AIMessage(content="")
        
        candidate = response.candidates[0]
        content_part = candidate.content
        text_content = ""
        tool_calls = []
        additional_kwargs = {}
        
        if content_part and content_part.parts:
            for part in content_part.parts:
                if part.text:
                    self.logger.debug(f"Raw Text Part: {part.text}")
                    text_content += part.text
                
                if hasattr(part, "function_call") and part.function_call:
                    self.logger.debug(f"Raw Function Call Part: {part.function_call.name} args={part.function_call.args}")
                    raw_args = part.function_call.args
                    args = dict(raw_args) if raw_args else {}
                    
                    id_val = getattr(part.function_call, "id", None)
                    if not id_val:
                         id_val = str(uuid.uuid4())
                    
                    tool_call_dict = {
                        "name": part.function_call.name,
                        "args": args,
                        "id": id_val
                    }
                        
                    if getattr(part, "thought_signature", None):
                        if "thought_signatures" not in additional_kwargs:
                            additional_kwargs["thought_signatures"] = {}
                        additional_kwargs["thought_signatures"][id_val] = part.thought_signature
                        
                    tool_calls.append(tool_call_dict)
                
        return AIMessage(content=text_content, tool_calls=tool_calls, additional_kwargs=additional_kwargs)

    @Tracing.trace(run_type="chain")
    async def run(self, state: OrchestratorState):
        """
        The Supervisor Node logic.
        """
        # Ensure client is available (Instantiate per run to avoid stale connection/event loop issues)
        try:
             client = genai.Client(api_key=self.settings.require_gemini_api_key())
        except Exception as e:
             self.logger.error(f"Cannot run supervisor: {e}")
             return {"messages": [AIMessage(content=f"Error: configuration missing {e}")]}

        try:
            model_id = self.settings.planner_model 
            
            # Resolve Tools dynamically
            from deep_research_agent.tools.registry import ToolRegistry
            tool_names = ["add_task", "update_task_status", "remove_task", "write_file", "delegate_research", "finish"]
            declarations, _ = ToolRegistry.resolve(tool_names, self.settings)
            
            # Convert declarations to types.Tool
            gemini_tools = types.Tool(function_declarations=declarations)

            messages = state.get("messages", [])
            todos = state.get("todos", [])
            
            self.logger.info(f"Supervisor Node: Processing {len(messages)} messages. Todos: {len(todos)}")

            
            # SYSTEM PROMPT
            # STATIC: Only the Role + Handbook (Cacheable)
            system_prompt_text = OrchestratorPrompts.build_supervisor_prompt()

            # DYNAMIC: The Todo List (Cannot be in cached system prompt)
            todo_list_str = "## Current Todo List:\n"
            if not todos:
                todo_list_str += "(Empty)\n"
            else:
                for t in todos:
                    todo_list_str += f"- [{t.status.value}] {t.id}: {t.objective}\n"
                    if t.brief_summary:
                        todo_list_str += f"  - Result Summary: {t.brief_summary}\n"
                    elif t.full_findings:
                        snippet = t.full_findings[:200] + "..." if len(t.full_findings) > 200 else t.full_findings
                        todo_list_str += f"  - Result Summary: {snippet}\n"

            # CONVERT MESSAGES
            contents = self._convert_langchain_to_genai(messages)
            
            # DYNAMIC: The Limits
            # Calculate current step by counting previous supervisor AIMessages
            current_step = sum(1 for m in messages if isinstance(m, AIMessage)) + 1
            max_steps = state.get("recursion_limit", self.settings.default_recursion_limit)
            dynamic_limits_str = OrchestratorPrompts.build_dynamic_limits_prompt(current_step, max_steps)

            # INJECT TODOS AND LIMITS
            state_text = f"--- DYNAMIC CONSTRAINTS ---\n{dynamic_limits_str}\n\n--- ORCHESTRATOR STATE ---\n{todo_list_str}\n--------------------------\nPlease proceed with the next step."
            state_message = types.Content(
                role="user", 
                parts=[types.Part.from_text(text=state_text)]
            )
            
            # Append to contents
            if contents:
                contents.append(state_message)
            else:
                contents = [state_message]
            
            # CHECK TURN ORDER (Fix for missing user message after tool response)
            if contents:
                last_item = contents[-1]
                if last_item.role == "model":
                    has_fc = any(p.function_call for p in last_item.parts)
                    if has_fc:
                        self.logger.error("CRITICAL: Last message is Model with FunctionCall, but no User(FunctionResponse) follows. State desync!")
                        contents.append(types.Content(role="user", parts=[
                            types.Part.from_text(text="Error: Previous tool call execution failed or returned no output. Please retry or proceed.")
                        ]))
                        self.logger.warning("Injected dummy user message to fix turn order.")
            
            # CACHE LOOKUP
            from deep_research_agent.agents.utils.cache_manager import get_process_level_cache
            
            cached_name = await get_process_level_cache(
                client,
                model_id,
                "supervisor", 
                system_prompt_text, 
                [gemini_tools],
                ttl_minutes=30
            )
            
            if cached_name:
                self.logger.info(f"Using Cached Content: {cached_name}")
                config = types.GenerateContentConfig(
                    cached_content=cached_name,
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(thinking_level=self.settings.supervisor_thinking_level)
                )
            else:
                self.logger.info("Cache miss or disabled. Using standard context.")
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt_text,
                    tools=[gemini_tools],
                    temperature=0.0,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    thinking_config=types.ThinkingConfig(thinking_level=self.settings.supervisor_thinking_level)
                )
            
            response = await client.aio.models.generate_content(
                model=model_id,
                contents=contents,
                config=config
            )

            # CONVERT BACK
            ai_msg = self._convert_genai_response_to_langchain(response)
            self.logger.info(f"Supervisor Response: {ai_msg.content[:50]}... Tool Calls: {len(ai_msg.tool_calls)}")
            
            return {"messages": [ai_msg]}

        except Exception as e:
            self.logger.exception("Supervisor Gemini call failed")
            return {"messages": [AIMessage(content=f"Error: {e}")]}

    @Tracing.trace(run_type="chain")
    async def execute_tools(self, state: OrchestratorState):
        """
        Executes tool calls from the Supervisor using Map-Reduce pattern.
        """
        messages = state.get("messages", [])
        if not messages:
            return {}
        
        last_msg = messages[-1]
        if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
            return {}
        
        todos = state.get("todos", [])
        
        # Resolve tool implementations
        from deep_research_agent.tools.registry import ToolRegistry
        tool_names = ["add_task", "update_task_status", "remove_task", "write_file", "delegate_research", "finish"]
        _, impls = ToolRegistry.resolve(tool_names, self.settings)
        
        self.logger.info(f"Executing {len(last_msg.tool_calls)} tools.")

        # 1. CLASSIFY TOOLS
        mutators = []
        parallel_workers = []
        others = []

        for tool_call in last_msg.tool_calls:
            name = tool_call["name"]
            if name in ["add_task", "update_task_status", "remove_task"]:
                mutators.append(tool_call)
            elif name == "delegate_research":
                parallel_workers.append(tool_call)
            else:
                others.append(tool_call)

        self.logger.info(
            f"Tool classification: {len(mutators)} mutators (sequential), "
            f"{len(parallel_workers)} workers (parallel), {len(others)} others"
        )

        tool_outputs_map = {} # call_id -> result string

        # 2. EXECUTE MUTATORS (Strictly Sequential)
        for tool_call in mutators:
            name = tool_call["name"]
            args = tool_call["args"]
            call_id = tool_call["id"]
            
            try:
                impl = impls.get(name) # This is apply_add_task etc.
                # Inject state manually as per design
                if name == "add_task":
                     todos = impl(current_todos=todos, **args)
                     added_task = todos[-1]
                     result_str = f"Successfully added task '{added_task.id}': {args.get('objective')}"
                elif name == "update_task_status":
                     todos = impl(current_todos=todos, **args)
                     result_str = f"Successfully executed update_task_status: {args.get('task_id')} -> {args.get('status')}"
                elif name == "remove_task":
                     todos = impl(current_todos=todos, **args)
                     result_str = f"Successfully executed remove_task: {args.get('task_id')}"
                else:
                     # Fallback for future mutators
                     todos = impl(current_todos=todos, **args)
                     result_str = f"Successfully executed {name}"

            except Exception as e:
                self.logger.exception(f"Error executing mutator {name}")
                result_str = f"Error: {str(e)}"
            
            # Create ToolMessage with tool_call_id
            tool_outputs_map[call_id] = ToolMessage(content=result_str, tool_call_id=call_id, name=name)

        # 3. MAP PHASE: Execute Parallel Workers
        worker_limits = state.get("worker_limits")

        async def execute_worker(tool_call: dict) -> Tuple[str, dict[str, Any], Optional[str], Optional[str]]:
            """Returns (call_id, result_dict, task_id_if_success, error_msg)"""
            name = tool_call["name"]
            args = tool_call["args"]
            call_id = tool_call["id"]
            
            try:
                from deep_research_agent.tools.delegation import DelegationTool
                dt = DelegationTool(self.settings)

                # Auto-inject context from completed dependencies
                context_str = None
                if name == "delegate_research":
                    target_task = next((t for t in todos if t.id == args["task_id"]), None)
                    if target_task and target_task.dependencies:
                        context_parts = []
                        for dep_id in target_task.dependencies:
                            dep_task = next((t for t in todos if t.id == dep_id), None)
                            if dep_task and dep_task.full_findings:
                                context_parts.append(
                                    f"### Result from '{dep_task.objective}':\n{dep_task.full_findings}"
                                )
                        if context_parts:
                            context_str = "\n\n".join(context_parts)
                            self.logger.info(
                                f"Injecting context from {len(context_parts)} "
                                f"dependencies for task {args['task_id']}"
                            )

                findings_dict = await dt.delegate_research(
                    task_id=args["task_id"],
                    objective=args["objective"],
                    instructions=args["instructions"],
                    context=context_str,
                    limits=worker_limits
                )

                return (call_id, findings_dict, args.get("task_id"), None)
            except Exception as e:
                self.logger.exception(f"Error in parallel worker {name}")
                return (call_id, {}, None, str(e))

        worker_results = []
        if parallel_workers:
            # Rate limit to max_parallel_workers (default 5)
            max_workers = state.get("max_parallel_workers", 10)
            semaphore = asyncio.Semaphore(max_workers)
            
            async def limited_execute(tc):
                async with semaphore:
                    return await execute_worker(tc)

            # Use asyncio.gather for compatibility (TaskGroup is 3.11+)
            self.logger.info(f"Launching {len(parallel_workers)} workers in parallel (max_parallel={max_workers})")
            start_time = time.monotonic()
            try:
                tasks = [limited_execute(tc) for tc in parallel_workers]
                worker_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Post-process results to handle exceptions from gather
                cleaned_results = []
                for i, res in enumerate(worker_results):
                    if isinstance(res, Exception):
                        call_id = parallel_workers[i]["id"]
                        self.logger.error(f"Worker task failed: {res}")
                        cleaned_results.append((call_id, {}, None, str(res)))
                    else:
                        cleaned_results.append(res)
                worker_results = cleaned_results

            except Exception as e:
                self.logger.error(f"Parallel execution failed: {e}")
                worker_results = []
            finally:
                elapsed = time.monotonic() - start_time
                self.logger.info(f"Parallel execution completed: {len(parallel_workers)} workers in {elapsed:.2f}s")

        # 4. REDUCE PHASE: Update State from Worker Results
        for call_id, findings_dict, task_id, error_msg in worker_results:
            # Create Tool Output
            if error_msg:
                 content = f"Error during delegation: {error_msg}"
            else:
                 content = (
                     f"Delegation complete. Summary: {findings_dict.get('brief_summary', 'No summary.')}" 
                 )
            
            tool_outputs_map[call_id] = ToolMessage(
                content=content,
                tool_call_id=call_id,
                name="delegate_research"
            )
            
            # Update State (Atomic Reduction)
            if task_id and not error_msg:
                # Map the worker's terminal status back to TaskStatus
                worker_status_str = findings_dict.get('status', 'completed')
                try:
                    worker_status = TaskStatus(worker_status_str)
                except ValueError:
                    worker_status = TaskStatus.COMPLETED
                updated_todos = []
                for t in todos:
                    if t.id == task_id:
                        t.status = worker_status
                        t.brief_summary = findings_dict.get('brief_summary')
                        t.full_findings = findings_dict.get('full_findings') # Full findings here
                    updated_todos.append(t)
                todos = updated_todos

        # 5. EXECUTE OTHERS (Sequential fallback)
        for tool_call in others:
            name = tool_call["name"]
            args = tool_call["args"]
            call_id = tool_call["id"]
            
            try:
                impl = impls.get(name)
                if not impl:
                    result_val = f"Unknown tool: {name}"
                elif inspect.iscoroutinefunction(impl):
                    result_val = await impl(**args)
                else:
                    result_val = impl(**args)
            except Exception as e:
                self.logger.exception(f"Error executing tool {name}")
                result_val = f"Error: {str(e)}"
            
            tool_outputs_map[call_id] = ToolMessage(content=str(result_val), tool_call_id=call_id, name=name)

        # Reconstruct outputs in original order to match tool_calls
        final_outputs = []
        for tool_call in last_msg.tool_calls:
            if tool_call["id"] in tool_outputs_map:
                final_outputs.append(tool_outputs_map[tool_call["id"]])
            else:
                 self.logger.error(f"Missing output for tool call {tool_call['id']}")
                 final_outputs.append(ToolMessage(content="Error: Tool execution failed", tool_call_id=tool_call["id"], name=tool_call["name"]))
            
        return {
            "messages": final_outputs,
            "todos": todos 
        }

