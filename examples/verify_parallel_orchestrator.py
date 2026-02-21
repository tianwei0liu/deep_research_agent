"""
Verify Parallel Supervisor Execution (Mocked)

We mock the `ToolRegistry` to return async functions that sleep for 1 second.
We then run the `supervisor_node` and `tool_execution_node` to ensure:
1. Two tasks delegated = ~1s total time (Parallel) vs ~2s (Sequential).
2. State update is correct (both tasks marked COMPLETED).
3. Logging occurs to both console and `verification.log`.
"""

import asyncio
import time
import logging
import uuid
import sys
import os

# Adjust path to include root (deep_research_agent package parent)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_core.messages import AIMessage, ToolMessage
from deep_research_agent.agents.orchestrator.state import OrchestratorState, ResearchTask
from deep_research_agent.agents.orchestrator.schemas import TaskStatus
from deep_research_agent.agents.orchestrator.supervisor import Supervisor
from deep_research_agent.tools.registry import ToolRegistry

# Configure logging to both console and file
# Log to examples/ folder as per GEMINI.md
log_filename = "examples/verify_parallel_orchestrator.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_filename)
    ]
)
logger = logging.getLogger(__name__)

async def mock_delegate_research(task_id: str, objective: str, instructions: str):
    logger.info(f"STARTING Mock Worker for task {task_id}")
    await asyncio.sleep(1.0) # Simulate work
    logger.info(f"FINISHED Mock Worker for task {task_id}")
    return f"Mock findings for {objective}"

async def main():
    logger.info("=== Starting Parallel Verification ===")
    logger.info(f"Logging to {log_filename}")
    
    # 1. Setup State with 2 Pending Tasks
    task1 = ResearchTask(id="task-1", objective="Research A", description="Find details about A", status=TaskStatus.PENDING)
    task2 = ResearchTask(id="task-2", objective="Research B", description="Find details about B", status=TaskStatus.PENDING)
    
    # 2. Mock Supervisor Output (Tool Calls)
    # The Supervisor "decides" to delegate both tasks at once.
    ai_message = AIMessage(
        content="Delegating tasks...",
        tool_calls=[
            {
                "name": "delegate_research",
                "args": {"task_id": "task-1", "objective": "Research A", "instructions": "Find A"},
                "id": "call_1"
            },
            {
                "name": "delegate_research",
                "args": {"task_id": "task-2", "objective": "Research B", "instructions": "Find B"},
                "id": "call_2"
            }
        ]
    )
    
    state = OrchestratorState(
        messages=[ai_message],
        todos=[task1, task2] # Pass the objects
    )
    
    # 3. Inject Mock Implementation into Registry
    # We monkeypatch the registry resolution for this test
    original_resolve = ToolRegistry.resolve
    
    def mock_resolve(names, settings):
        return [], {
            "delegate_research": mock_delegate_research,
            "manage_todos": lambda **kwargs: state["todos"] # No-op for this test
        }
    
    ToolRegistry.resolve = mock_resolve
    
    try:
        # 4. Run Tool Execution Node
        start_time = time.perf_counter()
        supervisor = Supervisor()
        result = await supervisor.execute_tools(state)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        logger.info(f"Execution took {duration:.2f} seconds")
        
        # 5. Verify Timing (Should be ~1s, not 2s)
        if 1.0 <= duration < 1.5:
            logger.info("PASS: Parallel Execution confirmed (Time ~1s)")
        else:
            logger.error(f"FAIL: Execution time {duration:.2f}s indicates sequential or overhead issues.")
            
        # 6. Verify State Update
        updated_todos = result["todos"]
        completed_count = sum(1 for t in updated_todos if t.status == TaskStatus.COMPLETED)
        
        if completed_count == 2:
            logger.info("PASS: State Integrity confirmed (2/2 tasks completed)")
        else:
            logger.error(f"FAIL: State Integity issue. Completed: {completed_count}/2")
            for t in updated_todos:
                logger.info(f"Task {t.id}: {t.status}")

    finally:
        # Restore registry
        ToolRegistry.resolve = original_resolve

if __name__ == "__main__":
    asyncio.run(main())
