"""
State definitions for the Orchestrator graph (Dynamic Supervisor).
"""

from typing import List, Optional, TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from deep_research_agent.agents.worker.schemas import WorkerResult, Limits
from deep_research_agent.agents.orchestrator.schemas import ResearchTask, TaskStatus
from pydantic import BaseModel, Field

from deep_research_agent.config import Settings


class OrchestratorState(TypedDict):
    """Global state for the Orchestrator graph."""
    # Conversation history with the Supervisor (User + AI + Tool outputs)
    # CRITICAL FIX: Use add_messages to APPEND history, not overwrite!
    messages: Annotated[List[BaseMessage], add_messages]
    
    # The dynamic "Todo List"
    # We overwrite this one because the supervisor returns the FULL updated list
    todos: List[ResearchTask]
    
    # Final synthesized report
    final_report: Optional[str]
    
    # Configuration
    max_parallel_workers: int
    recursion_limit: int
    worker_limits: Optional["Limits"]


def _default_max_parallel_workers() -> int:
    return Settings.load().default_max_parallel_workers


def _default_recursion_limit() -> int:
    return Settings.load().default_recursion_limit


def _default_worker_limits() -> Limits:
    settings = Settings.load()
    return Limits(
        max_tool_calls=settings.default_worker_max_tool_calls,
        max_turns=settings.default_worker_max_turns,
        max_output_tokens=settings.default_worker_max_output_tokens,
    )


class OrchestratorConfig(BaseModel):
    """Configuration for the Orchestrator run."""
    messages: List[BaseMessage]
    max_parallel_workers: int = Field(default_factory=_default_max_parallel_workers)
    recursion_limit: int = Field(default_factory=_default_recursion_limit)
    worker_limits: Optional[Limits] = Field(default_factory=_default_worker_limits)

    def to_state(self) -> OrchestratorState:
        return {
            "messages": self.messages,
            "todos": [],
            "max_parallel_workers": self.max_parallel_workers,
            "recursion_limit": self.recursion_limit,
            "worker_limits": self.worker_limits,
            "final_report": None
        }
