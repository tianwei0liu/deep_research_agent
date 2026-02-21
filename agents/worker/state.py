"""
Worker State definition.
"""

from typing import Any, TypedDict
from google import genai
from google.genai import types

from deep_research_agent.agents.worker.schemas import Limits, SpawnTask, WorkerResult

class WorkerState(TypedDict, total=False):
    """State for the worker graph. Input keys: task, limits, tool_names, model."""

    # Input (from spawn)
    task: SpawnTask
    limits: Limits
    tool_names: list[str]
    model: str | None

    # Set by validate_and_init or early failure
    result: WorkerResult

    # Resolved after validation (internal)
    declarations: list[dict[str, Any]]
    impls: dict[str, Any]
    gemini_config: types.GenerateContentConfig
    model_id: str
    contents: list[types.Content]
    client: genai.Client

    # Mutable counters
    tool_calls_used: int
    turns: int
    output_tokens_used: int
    input_tokens_used: int  # Max context used (latest turn's prompt tokens)
    last_error: str | None

    # From reason_act
    last_response: types.GenerateContentResponse
    last_function_calls: list[types.Part]
    forced_final_round: bool
    partial_reason: str | None
