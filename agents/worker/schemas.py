"""
Worker-specific data schemas and contracts.
Moved from agent/states.py.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from deep_research_agent.agents.orchestrator.schemas import TaskStatus


# --- Spawn input (from lead) ---


class SpawnTask(BaseModel):
    """Task description the lead must supply when creating a worker (§3.1)."""
    
    objective: str = Field(..., description="The main objective for the worker.")
    output_format: str = Field(..., description="The required format for the output.")
    sources_and_tools: str = Field(..., description="Instructions on which sources and tools to use.")
    task_boundaries: str = Field(..., description="Constraints and boundaries for the task.")
    context_snippet: Optional[str] = Field(None, description="Optional context from the conversation.")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class Limits(BaseModel):
    """Effort and limits for the worker (§3.2)."""
    
    max_tool_calls: int = Field(..., gt=0, description="Maximum number of tool calls allowed.")
    max_turns: Optional[int] = Field(None, gt=0, description="Maximum number of turns allowed.")
    max_output_tokens: Optional[int] = Field(None, gt=0, description="Maximum number of output tokens allowed.")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


# --- Result envelope (to lead) (§4, §11.2) ---


class WorkerResult(BaseModel):
    """Structured return envelope from worker to supervisor.
    
    Uses terminal TaskStatus values only: COMPLETED, PARTIAL, FAILED.
    """

    status: TaskStatus = Field(..., description="Terminal execution status (completed, partial, failed).")
    brief_summary: Optional[str] = Field(None, description="A 3-5 sentence executive summary of the full_findings for the Supervisor.")
    full_findings: Optional[str] = Field(None, description="Comprehensive markdown with citations for the Composer.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata (turns, tokens, etc).")
    sources: List[Dict[str, str]] = Field(default_factory=list, description="List of sources used.")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)
