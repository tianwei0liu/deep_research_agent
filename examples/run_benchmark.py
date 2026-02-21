"""Entry point for running the research agent benchmark suite.

Usage:
    # Run all benchmarks
    PYTHONPATH=.. python3 examples/run_benchmark.py

    # Run only comparative cases
    PYTHONPATH=.. python3 examples/run_benchmark.py --category comparative

    # Run a single case
    PYTHONPATH=.. python3 examples/run_benchmark.py --cases factual_001

    # Debug mode
    PYTHONPATH=.. python3 examples/run_benchmark.py --verbose
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark runner.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run the research agent benchmark suite."
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        type=str,
        help="Specific case IDs to run (e.g., factual_001 comparative_002).",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["factual", "temporal", "comparative", "multi_hop"],
        help="Run only cases in this category.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON. Default: benchmarks/results/run_<timestamp>.json",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        help="Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging (sets log level to DEBUG).",
    )
    return parser.parse_args()


def _resolve_log_level(args: argparse.Namespace) -> int:
    """Determine the log level from CLI args and environment.

    Priority: --log-level > --verbose > RESEARCH_ASSISTANT_LOG_LEVEL > INFO.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Logging level integer.
    """
    if args.log_level:
        level_name = args.log_level.upper()
    elif args.verbose:
        level_name = "DEBUG"
    else:
        level_name = os.environ.get("RESEARCH_ASSISTANT_LOG_LEVEL", "INFO").upper()

    return getattr(logging, level_name, logging.INFO)


def _configure_logging(log_level: int) -> logging.Logger:
    """Set up logging to both console and file.

    Args:
        log_level: The logging level to use.

    Returns:
        Configured logger instance.
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("examples/run_benchmark.log"),
        ],
    )
    return logging.getLogger(__name__)


async def _run(args: argparse.Namespace, logger: logging.Logger) -> None:
    """Execute the benchmark suite.

    Args:
        args: Parsed CLI arguments.
        logger: Configured logger.
    """
    # Load settings to enable LangSmith tracing (if configured)
    from deep_research_agent.config import Settings
    Settings.load()

    from deep_research_agent.benchmarks.runner import BenchmarkRunner

    runner = BenchmarkRunner()

    logger.info("=== Research Agent Benchmark Suite ===")
    logger.info(
        "Filters: cases=%s, category=%s",
        args.cases,
        args.category,
    )

    results = await runner.run_all(
        case_ids=args.cases,
        category=args.category,
    )

    # Summarize
    summary = BenchmarkRunner.summarize(results)
    logger.info("=== Benchmark Summary ===")
    logger.info(json.dumps(summary, indent=2))

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = Path("benchmarks") / "results" / f"run_{timestamp}.json"

    BenchmarkRunner.save_results(results, output_path)
    logger.info("Results saved to %s", output_path)

    # Print composite score prominently
    composite = summary.get("composite_score", 0.0)
    logger.info(
        "=== Composite Score: %.1f%% (%d cases) ===",
        composite * 100,
        summary.get("cases_run", 0),
    )


def main() -> None:
    """Entry point for the benchmark script."""
    args = _parse_args()
    log_level = _resolve_log_level(args)
    logger = _configure_logging(log_level)

    try:
        asyncio.run(_run(args, logger))
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user.")
    except Exception:
        logger.exception("Benchmark suite failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
