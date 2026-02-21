"""
Orchestrator-specific data schemas.
Moved from agent/states.py.
"""

from typing import List, Optional
from enum import Enum
import uuid
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ResearchTask(BaseModel):
    """A single unit of work in the research plan (Todo Item)."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Task ID. Auto-generated if not specified.")
    objective: str = Field(..., description="The research objective for this task.")
    description: str = Field(..., description="Detailed description/instructions for the worker.")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current execution status.")
    brief_summary: Optional[str] = Field(None, description="A 3-5 sentence executive summary of what was found.")
    full_findings: Optional[str] = Field(None, description="Summary of the result from the worker execution.")
    dependencies: List[str] = Field(default_factory=list, description="IDs of tasks that must complete before this one starts.")

    def to_dict(self):
        return self.model_dump()
