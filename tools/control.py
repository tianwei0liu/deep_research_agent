"""
Control tools: Signals for loop termination.
"""

from typing import Any, Callable
from pydantic import BaseModel, Field
from research_assistant.config import Settings


class ControlTool:
    """Tool for controlling the agent loop (e.g. finishing)."""

    def __init__(self, settings: Settings):
        self.settings = settings

    # --- Schema ---
    class FinishInput(BaseModel):
        reason: str = Field(..., description="Reason for finishing.")

    # --- Actions ---
    def finish(self, reason: str) -> str:
        return reason

    # --- Interface ---
    @staticmethod
    def get_declaration() -> dict[str, Any]:
        return {
            "name": "finish",
            "description": "Signal that the research is complete.",
            "parameters": ControlTool.FinishInput.model_json_schema(),
        }

    @staticmethod
    def make_impl(settings: Settings) -> Callable:
        tool = ControlTool(settings)
        
        def impl(**kwargs):
            # Strict validation
            input_data = ControlTool.FinishInput(**kwargs)
            return tool.finish(input_data.reason) 
        return impl
