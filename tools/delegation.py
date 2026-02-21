"""
Delegation tools: Spawn worker agents for research tasks.
"""

import logging
from typing import Any, Callable
from pydantic import BaseModel, Field

from research_assistant.config import Settings
from research_assistant.agents.worker.schemas import SpawnTask, Limits, WorkerResult

logger = logging.getLogger(__name__)


class DelegationTool:
    """Tool for delegating work to sub-agents."""

    def __init__(self, settings: Settings):
        self.settings = settings

    # --- Schema ---
    class DelegateResearchInput(BaseModel):
        """Input for delegating a research task to a worker."""
        task_id: str = Field(..., description="ID of the task to execute.")
        objective: str = Field(..., description="Specific objective for the worker.")
        instructions: str = Field(..., description="Detailed research instructions/boundaries.")

    # --- Actions ---
    async def delegate_research(self, task_id: str, objective: str, instructions: str, context: str | None = None, limits: Limits = None) -> dict[str, Any]:
        """
        Executes a worker for the given task.
        Returns a dictionary containing the findings.

        Args:
            context: Optional context from completed dependency tasks,
                     injected by the Supervisor (not the LLM).
        """
        logger.info(f"Delegating research for task {task_id}: {objective}")
        
        # Lazy import to avoid circular dependency
        from research_assistant.agents.worker.worker import Worker
        
        # We spawn the worker with its own tools (e.g. tavily)
        worker = Worker(settings=self.settings)
        
        spawn_task = SpawnTask(
            objective=objective,
            output_format="""JSON dictionary with keys:
            - 'brief_summary': A 1-2 sentence executive summary of what was found.
            - 'full_findings': Comprehensive markdown with citations.""",
            sources_and_tools="Use available research tools.",
            task_boundaries=instructions,
            context_snippet=context
        )
        
        if not limits:
            limits = Limits(
                max_tool_calls=getattr(self.settings, "default_worker_max_tool_calls", 40),
                max_turns=getattr(self.settings, "default_worker_max_turns", 10),
                max_output_tokens=getattr(self.settings, "default_worker_max_output_tokens", 8192)
            )
        
        try:
            result: WorkerResult = await worker.run_async(
                task=spawn_task,
                limits=limits,
                tools=["tavily_search"] 
            )
            
            return {
                 "status": result.status.value,
                 "brief_summary": result.brief_summary or "Summary not available.",
                 "full_findings": result.full_findings or "",
                 "caveats": result.metadata.get("caveats")
            }
        except Exception as e:
            logger.error(f"Worker execution failed for {task_id}: {e}")
            return {
                 "status": "failed",
                 "brief_summary": f"Worker failed: {e}",
                 "full_findings": f"Worker execution failed: {e}",
                 "caveats": str(e)
            }

    # --- Interface ---
    @staticmethod
    def get_declaration() -> dict[str, Any]:
        return {
            "name": "delegate_research",
            "description": "Delegate a research task to a worker agent.",
            "parameters": DelegationTool.DelegateResearchInput.model_json_schema(),
        }

    @staticmethod
    def make_impl(settings: Settings) -> Callable:
        """Factory for delegate_research implementation (strict validation)."""
        tool = DelegationTool(settings)

        async def impl(**kwargs):
            # Strict validation
            input_data = DelegationTool.DelegateResearchInput(**kwargs)
            return await tool.delegate_research(
                task_id=input_data.task_id,
                objective=input_data.objective,
                instructions=input_data.instructions
            )
        return impl
