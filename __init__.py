"""
Deep Research Agent — multi-agent research system.

Public API:
  - build_deep_agent: compile a LangGraph research agent.
  - run_deep_research: one-shot query → final report.
  - stream_deep_research: async generator for real-time streaming events.

Configuration: set env vars or load .env; use Settings.load() from deep_research_agent.config.
"""

from deep_research_agent.agents import (
    build_deep_agent,
    run_deep_research,
    stream_deep_research,
)

__all__ = [
    "build_deep_agent",
    "run_deep_research",
    "stream_deep_research",
]
