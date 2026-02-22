"""
Planning tools: Manage the research todo list (state manipulation).
"""

import logging
from typing import Literal, Optional, List, Any, Callable
from pydantic import BaseModel, Field

from deep_research_agent.config import Settings
from deep_research_agent.agents.orchestrator.schemas import ResearchTask, TaskStatus

logger = logging.getLogger(__name__)


class PlanningTool:
    """Tool for managing the research plan."""

    def __init__(self, settings: Settings):
        self.settings = settings

    # --- Schemas ---
    class AddTaskInput(BaseModel):
        """Input for adding a new task to the research plan."""
        task_id: Optional[str] = Field(None, description="Optional task ID for same-turn dependency referencing.")
        objective: str = Field(..., description="The objective of the task.")
        description: str = Field(..., description="Detailed instructions for the task.")
        dependencies: List[str] = Field(default_factory=list, description="IDs of tasks this depends on.")

    class UpdateTaskStatusInput(BaseModel):
        """Input for updating the status of an existing task."""
        task_id: str = Field(..., description="ID of the task to update.")
        status: Literal["completed", "failed", "partial"] = Field(
            ...,
            description=(
                "The new terminal status of the task. "
                "'completed': finished successfully. "
                "'failed': encountered an error. "
                "'partial': finished but reached limits before fully completing its objective."
            ),
        )

    class RemoveTaskInput(BaseModel):
        """Input for removing a task from the research plan."""
        task_id: str = Field(..., description="ID of the task to remove.")

    # --- Actions ---
    def add_task(self, current_todos: List[ResearchTask], input_data: "PlanningTool.AddTaskInput") -> List[ResearchTask]:
        """Adds a task to the list using validated input.

        Uses model_dump(exclude_none=True) so new optional fields
        automatically thread through to ResearchTask.
        """
        new_todos = [t.model_copy() for t in current_todos]
        task_fields = input_data.model_dump(exclude_none=True)
        # Map 'task_id' -> 'id' for ResearchTask constructor
        if "task_id" in task_fields:
            task_fields["id"] = task_fields.pop("task_id")
        task_fields["status"] = TaskStatus.PENDING
        new_task = ResearchTask(**task_fields)
        new_todos.append(new_task)
        logger.info(f"Added task: {new_task.id} - {input_data.objective}")
        return new_todos

    def update_task_status(self, current_todos: List[ResearchTask], task_id: str, status: TaskStatus) -> List[ResearchTask]:
        """Updates the status of a task."""
        new_todos = [t.model_copy() for t in current_todos]
        found = False
        for t in new_todos:
            if t.id == task_id:
                t.status = status
                found = True
                logger.info(f"Updated task {task_id} to {status}")
                break
        if not found:
            logger.warning(f"Task {task_id} not found for update.")
        return new_todos

    def remove_task(self, current_todos: List[ResearchTask], task_id: str) -> List[ResearchTask]:
        """Removes a task from the list."""
        initial_len = len(current_todos)
        new_todos = [t.model_copy() for t in current_todos if t.id != task_id]
        if len(new_todos) < initial_len:
            logger.info(f"Removed task {task_id}")
        else:
            logger.warning(f"Task {task_id} not found for remove.")
        return new_todos

    # --- Interface ---
    @staticmethod
    def get_add_task_declaration() -> dict[str, Any]:
        return {
            "name": "add_task",
            "description": "Add a new task to the research todo list.",
            "parameters": PlanningTool.AddTaskInput.model_json_schema(),
        }

    @staticmethod
    def get_update_task_status_declaration() -> dict[str, Any]:
        return {
            "name": "update_task_status",
            "description": "Update the status of an existing task.",
            "parameters": PlanningTool.UpdateTaskStatusInput.model_json_schema(),
        }

    @staticmethod
    def get_remove_task_declaration() -> dict[str, Any]:
        return {
            "name": "remove_task",
            "description": "Remove a task from the research todo list.",
            "parameters": PlanningTool.RemoveTaskInput.model_json_schema(),
        }

    # --- Factories ---
    @staticmethod
    def make_add_impl(settings: Settings) -> Callable:
        tool = PlanningTool(settings)
        def impl(current_todos: List[ResearchTask], **kwargs):
            input_data = PlanningTool.AddTaskInput(**kwargs)
            return tool.add_task(current_todos, input_data)
        return impl

    @staticmethod
    def make_update_status_impl(settings: Settings) -> Callable:
        tool = PlanningTool(settings)
        def impl(current_todos: List[ResearchTask], **kwargs):
            input_data = PlanningTool.UpdateTaskStatusInput(**kwargs)
            status = TaskStatus(input_data.status)
            return tool.update_task_status(current_todos, input_data.task_id, status)
        return impl
    
    @staticmethod
    def make_remove_impl(settings: Settings) -> Callable:
        tool = PlanningTool(settings)
        def impl(current_todos: List[ResearchTask], **kwargs):
            input_data = PlanningTool.RemoveTaskInput(**kwargs)
            return tool.remove_task(current_todos, input_data.task_id)
        return impl
