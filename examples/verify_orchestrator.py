"""
Verification script for the Dynamic Supervisor Orchestrator.
"""

import asyncio
import logging
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) # Try going up two levels just in case

# Configure logging
# Configure logging
import argparse

def get_log_level() -> str:
    parser = argparse.ArgumentParser(description="Verify Orchestrator")
    parser.add_argument("--log-level", type=str, help="Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging (sets log level to DEBUG)")
    args, _ = parser.parse_known_args()

    # Priority: --log-level > --verbose > RESEARCH_ASSISTANT_LOG_LEVEL > INFO
    if args.log_level:
        return args.log_level.upper()
    elif args.verbose:
        return "DEBUG"
    return os.environ.get("RESEARCH_ASSISTANT_LOG_LEVEL", "INFO").upper()

log_level_name = get_log_level()
try:
    log_level = getattr(logging, log_level_name)
except AttributeError:
    log_level = logging.INFO

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("examples/verify_orchestrator.log")
    ]
)
logger = logging.getLogger(__name__)

# Load settings to enable LangSmith tracing (if configured in config/settings.yaml/env)
from research_assistant.config import Settings
Settings.load()

# Set research_assistant logger to INFO (or remove this line to inherit from root)
# logging.getLogger("research_assistant.agents.orchestrator.supervisor").setLevel(logging.DEBUG)

from langchain_core.messages import HumanMessage
# from research_assistant.agents.orchestrator.graph import build_orchestrator_graph

async def main() -> None:
    """Run the orchestrator verification."""
    logger.info("Building Orchestrator Graph...")
    from research_assistant.agents.orchestrator.graph import OrchestratorGraph
    orchestrator = OrchestratorGraph()
    app = orchestrator.compile()
    
    query = "What is the current state of AI Agents in 2025 according to Anthropic?"
    logger.info("Running query: %s", query)
    
    from research_assistant.agents.orchestrator.state import OrchestratorConfig
    from research_assistant.agents.worker.schemas import Limits

    config = OrchestratorConfig(
        messages=[HumanMessage(content=query)],
        max_parallel_workers=10,
        recursion_limit=20,
        worker_limits=Limits(max_tool_calls=40, max_turns=10, max_output_tokens=8192)
    )
    
    initial_state = config.to_state()
    
    recursion_limit = initial_state.get("recursion_limit", 20)
    
    logger.info("--- Starting Execution (Recursion Limit: %d) ---", recursion_limit)
    async for event in app.astream(initial_state, config={"recursion_limit": recursion_limit}):
        for key, value in event.items():
            logger.info("[Node: %s]", key)
            # Log simple summary of updates
            if "messages" in value:
                last_msg = value["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        logger.info("  Supervisor Tool Call: %s", tc['name'])
                        if 'args' in tc:
                             logger.info("    Args: %s", tc['args'])
                elif hasattr(last_msg, "content"):
                    logger.info("  Message Content: %s", last_msg.content)
            
            if "todos" in value:
                logger.info("  Todos Count: %d", len(value['todos']))
                for t in value["todos"]:
                    result_len = len(t.result) if t.result else 0
                    logger.info("    - [%s] %s (Result len: %d)", t.status.value, t.objective, result_len)
            
            if "final_report" in value:
                logger.info("[Final Report]")
                logger.info("\n%s\n", value['final_report'])
                
                # Save final report to file
                report_path = os.path.join(os.path.dirname(__file__), "final_report.md")
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(value['final_report'])
                logger.info("Final report saved to %s", report_path)

if __name__ == "__main__":
    asyncio.run(main())
