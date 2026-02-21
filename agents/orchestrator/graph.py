"""
Orchestrator Graph: Wiring the Dynamic Supervisor.
Refactored to use Class-based components.
"""

from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, START, END
import logging

from deep_research_agent.agents.orchestrator.state import OrchestratorState
from deep_research_agent.agents.orchestrator.supervisor import Supervisor
from deep_research_agent.agents.orchestrator.composer import Composer

class OrchestratorGraph:
    """
    Constructs the StateGraph for the Orchestrator.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.supervisor = Supervisor()
        self.composer = Composer()

    def route_supervisor(self, state: OrchestratorState) -> Literal["tool_execution_node", "compose_node", "__end__"]:
        """Route based on the supervisor's output.

        Routing rules (evaluated in order):
            1. No messages at all           -> END
            2. 'finish' tool called         -> compose_node
            3. Other tool calls present     -> tool_execution_node
            4. No tool calls, but completed
               research tasks exist         -> compose_node  (implicit finish)
            5. No tool calls, no tasks      -> END
        """
        messages = state.get("messages", [])
        if not messages:
            return END

        last_msg = messages[-1]
        has_tool_calls = hasattr(last_msg, "tool_calls") and last_msg.tool_calls

        if has_tool_calls:
            for tool_call in last_msg.tool_calls:
                if tool_call["name"] in ("finish", "FinishInput"):
                    return "compose_node"
            return "tool_execution_node"

        # No tool calls — check if research was performed (implicit finish).
        todos = state.get("todos", [])
        from deep_research_agent.agents.orchestrator.schemas import TaskStatus
        actionable_statuses = {TaskStatus.COMPLETED, TaskStatus.PARTIAL, TaskStatus.FAILED}
        has_findings = any(t.status in actionable_statuses for t in todos)

        if has_findings:
            self.logger.warning(
                "Implicit finish detected: supervisor returned text without "
                "calling finish(). Routing to compose_node. "
                "Text preview: %.100s",
                getattr(last_msg, "content", "") or "",
            )
            return "compose_node"

        return END

    def compile(self):
        """
        Builds and compiles the state graph for the dynamic orchestrator.
        """
        workflow = StateGraph(OrchestratorState)
        
        # Add Nodes
        workflow.add_node("supervisor", self.supervisor.run)
        workflow.add_node("tool_execution_node", self.supervisor.execute_tools)
        workflow.add_node("compose_node", self.composer.run)
        
        # Add Edges
        workflow.add_edge(START, "supervisor")
        
        # Conditional Edge from Supervisor
        workflow.add_conditional_edges(
            "supervisor",
            self.route_supervisor,
            {
                "tool_execution_node": "tool_execution_node",
                "compose_node": "compose_node",
                END: END
            }
        )
        
        # Edge from Tool Execution back to Supervisor (Loop)
        workflow.add_edge("tool_execution_node", "supervisor")
        
        # Edge from Composer to END
        workflow.add_edge("compose_node", END)
        
        return workflow.compile()
