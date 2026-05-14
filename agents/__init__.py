"""Agent layer: deep research agent built on ``deepagents``."""

from deep_research_agent.agents.agent import (
    build_deep_agent,
    run_deep_research,
    stream_deep_research,
)
from deep_research_agent.agents.mcp_client import MCPSearchClient

__all__ = [
    "build_deep_agent",
    "run_deep_research",
    "stream_deep_research",
    "MCPSearchClient",
]
