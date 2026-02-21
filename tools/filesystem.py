"""
Filesystem tools: safe file writing and management.
"""

import os
import logging
from typing import Any, Callable
from pydantic import BaseModel, Field

from deep_research_agent.config import Settings

logger = logging.getLogger(__name__)


class FilesystemTool:
    """Tool for safe file operations."""

    def __init__(self, settings: Settings):
        self.settings = settings

    # --- Schema ---
    class WriteFileInput(BaseModel):
        """Input for writing a file to the workspace."""
        path: str = Field(..., description="Relative path to write the file (e.g., 'research_plan.md').")
        content: str = Field(..., description="The content to write.")

    # --- Actions ---
    def write_file(self, path: str, content: str) -> str:
        """Writes content to the specified path (safe sandbox check could go here)."""
        try:
            # Prevent traversal
            clean_path = os.path.normpath(path)
            if clean_path.startswith("..") or clean_path.startswith("/"):
                 # For safety in this agent demo, forcing local relative path
                 clean_path = clean_path.lstrip("./\\")
            
            target_path = clean_path
            
            os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
            
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
                
            return f"Successfully wrote {len(content)} bytes to {target_path}."
        except Exception as e:
            return f"Error writing file: {e}"

    # --- Interface ---
    @staticmethod
    def get_declaration() -> dict[str, Any]:
        return {
            "name": "write_file",
            "description": "Write content to a file in the workspace.",
            "parameters": FilesystemTool.WriteFileInput.model_json_schema(),
        }

    @staticmethod
    def make_impl(settings: Settings) -> Callable:
        """Factory for write_file implementation (strict validation)."""
        tool = FilesystemTool(settings)
        
        def impl(**kwargs):
            # Strict validation
            input_data = FilesystemTool.WriteFileInput(**kwargs)
            return tool.write_file(input_data.path, input_data.content)
        return impl
