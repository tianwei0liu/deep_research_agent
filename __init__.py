"""
Multi-agent research system: worker and tools.

Public API:
  - Worker: instantiate and call .run(task, limits, tools, model=..., run_id=...) -> WorkerResult
  - SpawnTask, Limits, WorkerResult (models)

Configuration: set env vars or load .env; use Settings.load() from research_assistant.config.
"""

from research_assistant.agents import Limits, SpawnTask, Worker, WorkerResult

__all__ = [
    "Worker",
    "SpawnTask",
    "Limits",
    "WorkerResult",
]
