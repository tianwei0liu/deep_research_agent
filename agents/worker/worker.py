"""
Worker runner: ReAct loop with Gemini and tool execution (§5, §11).

Implemented as a LangGraph 1.0 StateGraph for easy integration with the lead
researcher (e.g. as a subgraph or tool). Run to completion (or limit), return
WorkerResult envelope.

We use the native Google Genai SDK (google.genai) rather than LangChain's
ChatGoogleGenerativeAI so we can keep a single LLM dependency and full control
over request/response shapes (Content, Part, function_call).
"""

from __future__ import annotations

from typing import Any

from google import genai
from research_assistant.agents.utils.tracing import Tracing

from research_assistant.config import Settings
from research_assistant.agents.worker.schemas import Limits, SpawnTask, WorkerResult
from research_assistant.agents.orchestrator.schemas import TaskStatus
from research_assistant.agents.worker.state import WorkerState
from research_assistant.agents.worker.nodes import WorkerNodes
from research_assistant.agents.worker.graph import WorkerGraph


class Worker:
    """
    ReAct worker: validates spawn input, runs Gemini with tools, returns WorkerResult.

    Encapsulates the LangGraph (validate → reason_act ⇄ execute_tools → parse_final).
    Use run() for a single task, or get_graph() to plug the worker into a lead graph.
    """

    def __init__(
        self,
        client: genai.Client | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._client = client
        self._settings = settings or Settings.load()
        self._nodes = WorkerNodes(self._settings)
        self._graph_builder = WorkerGraph(self._nodes)
        self._graph: Any = None

    def get_graph(self) -> Any:
        """Return the compiled worker graph. Compiles once and reuses (§11.4 integration)."""
        if self._graph is None:
            self._graph = self._graph_builder.build()
        return self._graph

    @Tracing.trace(name="Worker.run_async")
    async def run_async(
        self,
        task: SpawnTask,
        limits: Limits,
        tools: list[str],
        *,
        model: str | None = None,
        run_id: str | None = None,
    ) -> WorkerResult:
        """
        Execute the worker: ReAct loop until done or limit reached. Returns envelope (§11.2).

        Args:
            task: Task description from lead (§3.1).
            limits: max_tool_calls, optional max_turns (§3.2).
            tools: Tool names (e.g. ["tavily_search"]). Runner resolves to definitions (§3.3).
            model: Gemini model id (default: gemini-2.5-flash).
            run_id: Optional id for tracing.

        Returns:
            WorkerResult with status, findings, metadata (§4).
        """
        initial: WorkerState = {
            "task": task,
            "limits": limits,
            "tool_names": tools,
            "model": model,
            "client": self._client,  # Inject client if available
        }
        config: dict[str, Any] = {}
        if run_id:
            config["configurable"] = config.get("configurable") or {}
            config["configurable"]["thread_id"] = run_id

        graph = self.get_graph()
        # invoke is sync; ainvoke is async for LangGraph.
        final_state = await graph.ainvoke(initial, config=config if config else None)
        result = final_state.get("result")
        if result is None:
            return WorkerResult(
                status=TaskStatus.FAILED,
                brief_summary="graph finished without result",
                metadata={
                    "reason": "graph finished without result",
                    "final_keys": list(final_state.keys()),
                },
            )
        return result
