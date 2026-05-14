"""Interactive multi-turn deep research CLI.

Provides a chat-like REPL where users can:
1. Input research queries conversationally
2. Receive persona skill recommendations and approve/reject/edit them
3. Watch research progress with streaming output
4. Ask follow-up questions within the same research session
5. Save reports automatically after each research cycle

Usage:
    cd /home/tianwei/workspace/deep_research_agent
    source .venv/bin/activate
    python examples/interactive_research.py
    python examples/interactive_research.py --verbose
    python examples/interactive_research.py --persona buffett
    python examples/interactive_research.py --no-skills-discovery
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Trigger LangSmith env setup from settings.yaml
from deep_research_agent.config import Settings
Settings.load()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BANNER = """\
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🔬  Deep Research Agent — Interactive Mode                 ║
║                                                              ║
║   Multi-turn conversation with persona skills support.       ║
║                                                              ║
║   Commands:                                                  ║
║     /skills   — List available persona skills                ║
║     /persona <id>  — Activate a persona mid-session          ║
║     /reset    — Start a new research session                 ║
║     /save     — Save the latest report to file               ║
║     /help     — Show this help message                       ║
║     /quit     — Exit the program                             ║
║                                                              ║
║   Type your research query to begin.                         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

_HELP_TEXT = """\
Available commands:
  /skills          — Show all available persona analysis frameworks
  /persona <id>    — Pre-activate a persona (e.g. /persona buffett)
  /reset           — Clear conversation history and start fresh
  /save [filename] — Save the last report (default: deep_research_report.md)
  /help            — Show this help message
  /quit, /exit     — Exit the program

Or just type a research question to start/continue research.
"""

_QUIT_COMMANDS = {"/quit", "/exit", "exit", "quit", "bye", "q"}


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive multi-turn deep research CLI."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        help="Set log level (DEBUG, INFO, WARNING, ERROR).",
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
        help="Pre-activate a persona framework (e.g. 'buffett').",
    )
    parser.add_argument(
        "--no-skills-discovery",
        action="store_true",
        help="Disable interactive skills discovery.",
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


def _configure_logging(log_level: int) -> None:
    """Set up dual console+file logging."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler("examples/interactive_research.log"),
        ],
    )

# ---------------------------------------------------------------------------
# Skills list command
# ---------------------------------------------------------------------------

def _show_skills() -> None:
    """Print available persona skills in a formatted table."""
    from deep_research_agent.personas.registry import PersonaRegistry
    from deep_research_agent.agents.skills_catalog import SkillsCatalog

    registry = PersonaRegistry()
    catalog = SkillsCatalog(registry=registry)

    if catalog.count == 0:
        print("  No persona skills available.\n")
        return

    print(f"\n  📚 Available Persona Skills ({catalog.count} total)\n")
    print("  " + "─" * 72)
    print(f"  {'ID':<18} {'Name':<12} {'Description':<28} {'Domains'}")
    print("  " + "─" * 72)

    for persona in registry.list_personas():
        domains = ", ".join(persona.applicable_domains) if persona.applicable_domains else "-"
        print(
            f"  {persona.persona_id:<18} "
            f"{persona.display_name:<12} "
            f"{persona.description[:28]:<28} "
            f"{domains}"
        )
    print("  " + "─" * 72)
    print("  Usage: /persona <id>  or let the Supervisor auto-select.\n")


# ---------------------------------------------------------------------------
# Research stream handler
# ---------------------------------------------------------------------------

class InteractiveSession:
    """Manages the interactive research session state.

    Encapsulates the checkpointer, thread ID, and report history
    for the multi-turn research session.

    Args:
        persona_id: Optional pre-activated persona.
        enable_skills_discovery: Whether to enable autonomous persona selection.
    """

    # Canonical report path (matches run_deep_agent.py)
    _REPORT_PATH = os.path.join(
        os.path.dirname(__file__), "deep_agent_report.md"
    )

    def __init__(
        self,
        *,
        persona_id: Optional[str] = None,
        enable_skills_discovery: bool = True,
    ) -> None:
        from langgraph.checkpoint.memory import MemorySaver
        self._checkpointer = MemorySaver()
        self._thread_id = str(uuid.uuid4())
        self._persona_id = persona_id
        self._enable_skills_discovery = enable_skills_discovery
        self._turn_count = 0
        self._last_report = ""
        self._total_tool_calls = 0
        self._research_complete = False

    @property
    def thread_id(self) -> str:
        """Current conversation thread ID."""
        return self._thread_id

    @property
    def research_complete(self) -> bool:
        """Whether the last run_query produced a final report."""
        return self._research_complete

    def reset(self) -> None:
        """Reset session state for a new conversation."""
        from langgraph.checkpoint.memory import MemorySaver
        self._checkpointer = MemorySaver()
        self._thread_id = str(uuid.uuid4())
        self._turn_count = 0
        self._last_report = ""
        self._total_tool_calls = 0
        self._persona_id = None
        self._enable_skills_discovery = True
        self._research_complete = False
        logger.info("Session reset: new thread_id=%s", self._thread_id)

    def set_persona(self, persona_id: str) -> None:
        """Pre-set a persona for the next query."""
        self._persona_id = persona_id

    async def run_query(self, query: str) -> None:
        """Execute a research query with streaming output.

        Handles the full lifecycle: stream events → display tokens →
        save report.

        Args:
            query: The user's research query or follow-up.
        """
        self._turn_count += 1
        print(f"\n{'─' * 60}")
        print(f"  🔍 Turn {self._turn_count} | Thread: {self._thread_id[:8]}...")
        print(f"{'─' * 60}\n")

        await self._stream_research(query)

    async def _stream_research(self, query: str) -> None:
        """Stream a single research pass and process events.

        Args:
            query: The research query.
        """
        from deep_research_agent.agents import stream_deep_research

        tool_calls = 0

        async for event in stream_deep_research(
            query,
            thread_id=self._thread_id,
            checkpointer=self._checkpointer,
            persona_id=self._persona_id,
            enable_skills_discovery=self._enable_skills_discovery,
        ):
            event_type = event["type"]

            if event_type == "status":
                _print_status(event["data"])

            elif event_type == "run_id":
                logger.info("[TRACE] LangSmith Run ID: %s", event["data"])

            elif event_type == "tool_start":
                tool_calls += 1
                _print_tool_start(event["data"]["name"], tool_calls)

            elif event_type == "tool_end":
                _print_tool_end(event["data"]["name"])

            elif event_type == "token":
                sys.stdout.write(event["data"])
                sys.stdout.flush()

            elif event_type == "final_report":
                self._last_report = event["data"]

        self._total_tool_calls += tool_calls
        logger.debug(
            "Stream done: tool_calls=%d, has_report=%s",
            tool_calls, bool(self._last_report),
        )

        # Print report summary and auto-save.
        if self._last_report and tool_calls > 0:
            saved_path = self._auto_save_report()
            print(f"\n\n{'═' * 60}")
            print(f"  ✅ Research complete — {tool_calls} tool calls this turn")
            print(f"  📊 Total tool calls this session: {self._total_tool_calls}")
            report_len = len(self._last_report)
            print(f"  📝 Report length: {report_len:,} chars")
            if saved_path:
                print(f"  📁 Report saved to: {saved_path}")
            print(f"{'═' * 60}")
            print("  💡 输入新的研究问题开始下一轮研究，或输入 /quit 退出\n")
            # Reset for next research cycle (fresh context, no history)
            self.reset()

    def _auto_save_report(self) -> str:
        """Auto-save the report to the canonical path after research completes.

        Writes to both:
        - The canonical path (``examples/deep_agent_report.md``)
        - A timestamped copy for history preservation

        Returns:
            The canonical path the report was saved to.
        """
        if not self._last_report:
            return ""

        clean_report = self._last_report.lstrip("-").lstrip("\n")

        # Write canonical file
        with open(self._REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(clean_report)
        logger.info("Report auto-saved to %s", self._REPORT_PATH)

        # Write timestamped copy for history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ts_path = os.path.join(
            os.path.dirname(__file__),
            f"research_report_{timestamp}.md",
        )
        with open(ts_path, "w", encoding="utf-8") as f:
            f.write(clean_report)
        logger.info("Timestamped copy saved to %s", ts_path)

        return self._REPORT_PATH

    def save_report(self, filename: Optional[str] = None) -> str:
        """Save the last report to a custom file.

        Args:
            filename: Output path. Defaults to timestamped name.

        Returns:
            The path the report was saved to.
        """
        if not self._last_report:
            print("  ⚠️ No report to save yet. Run a research query first.\n")
            return ""

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"examples/research_report_{timestamp}.md"

        clean_report = self._last_report.lstrip("-").lstrip("\n")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(clean_report)
        print(f"  📁 Report saved to: {filename}\n")
        return filename


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_status(msg: str) -> None:
    """Print a styled status message."""
    print(f"  ⏳ {msg}")


def _print_tool_start(name: str, count: int) -> None:
    """Print a tool invocation start indicator."""
    # Compact format to avoid flooding the terminal
    sys.stdout.write(f"\r  🔧 [{count}] {name}...")
    sys.stdout.flush()


def _print_tool_end(name: str) -> None:
    """Clear the tool line after completion."""
    sys.stdout.write(f"\r  ✓ {name}" + " " * 30 + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main REPL loop
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run the interactive research REPL."""
    args = _parse_args()
    log_level = _resolve_log_level(args)
    _configure_logging(log_level)

    print(_BANNER)

    session = InteractiveSession(
        persona_id=args.persona,
        enable_skills_discovery=not args.no_skills_discovery,
    )

    if args.persona:
        print(f"  🧠 Persona pre-activated: {args.persona}")
    if not args.no_skills_discovery:
        print("  ✨ Skills Discovery: enabled (personas will be recommended)")
    print(f"  🆔 Session ID: {session.thread_id[:8]}...\n")

    while True:
        try:
            user_input = input("You >>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  👋 Goodbye!\n")
            break

        if not user_input:
            continue

        # --- Slash commands ---
        lower_input = user_input.lower()

        if lower_input in _QUIT_COMMANDS:
            print("\n  👋 Goodbye!\n")
            break

        if lower_input == "/help":
            print(_HELP_TEXT)
            continue

        if lower_input == "/skills":
            _show_skills()
            continue

        if lower_input.startswith("/persona"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                persona_id = parts[1].strip()
                session.set_persona(persona_id)
                print(f"  🧠 Persona set to '{persona_id}' for next query.\n")
            else:
                print("  Usage: /persona <id>  (e.g. /persona buffett)\n")
            continue

        if lower_input == "/reset":
            session.reset()
            print("  🔄 Session reset. Start a new research topic.\n")
            continue

        if lower_input.startswith("/save"):
            parts = user_input.split(maxsplit=1)
            filename = parts[1].strip() if len(parts) > 1 else None
            session.save_report(filename)
            continue

        # --- Research query ---
        try:
            await session.run_query(user_input)
        except Exception as exc:
            logger.error("Research failed: %s", exc, exc_info=True)
            print(f"\n  ❌ Error: {exc}")
            print("  💡 Try again or use /reset to start fresh.\n")
            continue




if __name__ == "__main__":
    asyncio.run(main())
