"""
Example: run the worker with a single delegated task (§3.4 example payload).

Usage:
  cd research_assistant && pip install -e .
  Put sensitive API keys in .env (GOOGLE_GEMINI_API_KEY, TAVILY_API_KEY). Non-sensitive
  options are in config/settings.yaml; LangSmith options there are applied to os.environ.
  python3 examples/run_worker.py [--verbose]
"""

import argparse
import asyncio
import logging
import os
import sys

# Project root (parent of examples/); add parent of project root so "research_assistant" package resolves
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(_root))
os.chdir(_root)

from dotenv import load_dotenv

load_dotenv()

# Load config so LangSmith env vars (tracing, project, endpoint) are set before any
# runs.
from research_assistant.config import Settings
Settings.load()

from research_assistant import Limits, SpawnTask, Worker

logger = logging.getLogger(__name__)


def _configure_logging(level_name: str) -> None:
    try:
        level = getattr(logging, level_name)
    except AttributeError:
        level = logging.INFO

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler("examples/run_worker.log")
    
    handlers = [c_handler, f_handler]

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run worker example (top 5 AI agent companies 2025).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging (sets log level to DEBUG)")
    parser.add_argument("--log-level", type=str, help="Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    args = parser.parse_args()

    # Priority: --log-level > --verbose > RESEARCH_ASSISTANT_LOG_LEVEL > INFO
    if args.log_level:
        log_level = args.log_level.upper()
    elif args.verbose:
        log_level = "DEBUG"
    else:
        log_level = os.environ.get("RESEARCH_ASSISTANT_LOG_LEVEL", "INFO").upper()
    
    _configure_logging(log_level)

    task = SpawnTask(
        objective="Find the top 5 AI agent companies founded or heavily marketed in 2025, with name and official website.",
        output_format='JSON array of objects: {"name": string, "url": string, "founded_or_2025_focus": string}',
        sources_and_tools="Use only tavily_search. Prefer company websites and news from 2025.",
        task_boundaries="US and Europe only. Stop after 5 companies or 8 search calls. Do not include pure LLM APIs (e.g. OpenAI); focus on agentic/AI agent products.",
        context_snippet='User question: "Who are the main AI agent companies in 2025?" Your job is only the company list.',
    )
    limits = Limits(max_tool_calls=40, max_turns=10, max_output_tokens=8192)
    tools = ["tavily_search"]

    worker = Worker()
    # Run the worker asynchronously
    result = await worker.run_async(task, limits, tools)

    logger.info("--- Result ---")
    logger.info("Status: %s", result.status)
    logger.info("Findings: %s", result.findings)
    logger.info("Metadata: %s", result.metadata)
    if isinstance(result.findings, list):
        logger.info("Number of findings: %d", len(result.findings))
    if result.sources:
        logger.info("Sources: %s", result.sources)


if __name__ == "__main__":
    asyncio.run(main())
