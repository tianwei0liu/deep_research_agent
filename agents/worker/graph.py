"""
Worker Graph: Definition and construction of the Worker StateGraph.
"""

from typing import Any, Literal
from langgraph.graph import END, START, StateGraph

from deep_research_agent.agents.worker.state import WorkerState
from deep_research_agent.agents.worker.nodes import WorkerNodes


class WorkerGraph:
    """
    Encapsulates the construction of the Worker StateGraph.
    """

    def __init__(self, nodes: WorkerNodes):
        self.nodes = nodes

    @staticmethod
    def _route_after_validate(state: WorkerState) -> Literal["reason_act", "__end__"]:
        if state.get("result") is not None:
            return END
        return "reason_act"

    @staticmethod
    def _route_after_reason(
        state: WorkerState
    ) -> Literal["execute_tools", "parse_final", "__end__"]:
        if state.get("result") is not None:
            return END
        if state.get("forced_final_round"):
            return "parse_final"
        function_calls = state.get("last_function_calls") or []
        limits = state["limits"]
        tool_calls_used = state["tool_calls_used"] or 0
        if function_calls and tool_calls_used < limits.max_tool_calls:
            return "execute_tools"
        return "parse_final"

    @staticmethod
    def _route_after_execute_tools(state: WorkerState) -> Literal["reason_act", "__end__"]:
        if state.get("result") is not None:
            return END
        return "reason_act"

    def build(self) -> Any:
        """Build and compile the worker StateGraph (§5, §11)."""
        graph = StateGraph(WorkerState)

        graph.add_node("validate_and_init", self.nodes.validate_and_init)
        graph.add_node("reason_act", self.nodes.reason_act)
        graph.add_node("execute_tools", self.nodes.execute_tools)
        graph.add_node("parse_final", self.nodes.parse_final)

        graph.add_edge(START, "validate_and_init")
        graph.add_conditional_edges(
            "validate_and_init", self._route_after_validate, {"reason_act": "reason_act", END: END}
        )
        graph.add_conditional_edges(
            "reason_act",
            self._route_after_reason,
            {"execute_tools": "execute_tools", "parse_final": "parse_final", END: END},
        )
        graph.add_conditional_edges(
            "execute_tools", self._route_after_execute_tools, {"reason_act": "reason_act", END: END}
        )
        graph.add_edge("parse_final", END)

        return graph.compile()
