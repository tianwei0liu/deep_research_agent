"""Run a deep research query using ``create_deep_agent`` with streaming output.

Usage:
    PYTHONPATH=.. python examples/run_deep_agent.py
    PYTHONPATH=.. python examples/run_deep_agent.py "What is the current state of AI Agents?"
    PYTHONPATH=.. python examples/run_deep_agent.py --thread-id my-session "Follow-up question"
    PYTHONPATH=.. python examples/run_deep_agent.py --verbose
"""

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Trigger LangSmith env setup from settings.yaml
from deep_research_agent.config import Settings
Settings.load()


DEFAULT_QUERY = "What is the current state of AI Agents in 2025 according to Anthropic?"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a deep research query.")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY, help="Research query.")
    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help="Thread ID for multi-turn conversation. Reuse to continue a session.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        help="Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--persona",
        type=str,
        default=None,
        help=(
            "Activate a persona analysis framework "
            "(e.g., 'buffett', 'zhangxuefeng', 'feynman')."
        ),
    )
    parser.add_argument(
        "--no-skills-discovery",
        action="store_true",
        help="Disable Skills Discovery (skip persona recommendations).",
    )
    return parser.parse_args()


def _resolve_log_level(args: argparse.Namespace) -> int:
    """Determine log level from CLI args and environment."""
    if args.log_level:
        level_name = args.log_level.upper()
    elif args.verbose:
        level_name = "DEBUG"
    else:
        level_name = os.environ.get("RESEARCH_ASSISTANT_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def _configure_logging(log_level: int) -> logging.Logger:
    """Set up logging to console and file."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("examples/run_deep_agent.log"),
        ],
    )
    return logging.getLogger(__name__)


async def _run_stream(
    args: argparse.Namespace,
    logger: logging.Logger,
    checkpointer: "MemorySaver",
) -> tuple[str, int]:
    """Run stream_deep_research, return (report, tool_calls).

    Args:
        args: CLI arguments.
        logger: Logger instance.
        checkpointer: MemorySaver for multi-turn support.

    Returns:
        Tuple of (report_text, tool_call_count).
    """
    from deep_research_agent.agents import stream_deep_research

    report = ""
    tool_calls = 0

    async for event in stream_deep_research(
        args.query,
        thread_id=args.thread_id,
        checkpointer=checkpointer,
        persona_id=args.persona,
        enable_skills_discovery=not args.no_skills_discovery,
    ):
        event_type = event["type"]

        if event_type == "status":
            logger.info("[STATUS] %s", event["data"])

        elif event_type == "run_id":
            logger.info("[TRACE] LangSmith Run ID: %s", event["data"])

        elif event_type == "tool_start":
            tool_calls += 1
            tool_name = event["data"]["name"]
            logger.info("[TOOL START] %s (#%d)", tool_name, tool_calls)
            logger.debug("  Input: %s", event["data"].get("input", ""))

        elif event_type == "tool_end":
            tool_name = event["data"]["name"]
            logger.info("[TOOL END] %s", tool_name)
            logger.debug("  Output: %s", event["data"].get("output", "")[:200])

        elif event_type == "token":
            # Print tokens directly for real-time output
            sys.stdout.write(event["data"])
            sys.stdout.flush()

        elif event_type == "final_report":
            report = event["data"]

    return report, tool_calls


async def main() -> None:
    """Run the deep research agent with streaming output."""
    args = _parse_args()
    log_level = _resolve_log_level(args)
    logger = _configure_logging(log_level)

    # Use MemorySaver for multi-turn support within this process
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()

    logger.info("Query: %s", args.query)
    if args.thread_id:
        logger.info("Thread ID: %s (multi-turn mode)", args.thread_id)
    if args.persona:
        logger.info("Persona: %s", args.persona)
    if not args.no_skills_discovery:
        logger.info("Skills Discovery: enabled (autonomous)")
    logger.info("Starting deep research agent (this may take a few minutes)...")

    report, tool_calls = await _run_stream(args, logger, checkpointer)

    logger.info("=" * 80)
    logger.info("Research complete — %d tool calls", tool_calls)

    # Save to file — strip any leading '---' that the LLM may emit,
    # which Markdown previewers misinterpret as YAML front matter.
    clean_report = report.lstrip("-").lstrip("\n") if report else ""
    report_path = os.path.join(os.path.dirname(__file__), "deep_agent_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(clean_report)
    logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    asyncio.run(main())

