"""
Tracing utilities for the Deep Research Agent.

Provides a bridge between LangGraph's RunnableConfig-based tracing
and LangSmith's contextvars-based tracing.
"""

from typing import Optional, Any

from langsmith.run_trees import RunTree


class Tracing:
    """
    Namespace for tracing utilities.
    """

    @staticmethod
    def get_parent_run(config: Optional[dict] = None) -> Optional[RunTree]:
        """Extract the LangSmith RunTree from a LangGraph RunnableConfig.

        This bridges LangGraph's callback-based tracing to langsmith's
        contextvars-based tracing. Use the returned RunTree as the
        ``parent`` argument to ``langsmith.trace()`` so that spans
        nest under the current LangGraph node.

        Args:
            config: The RunnableConfig passed to a LangGraph node function.

        Returns:
            A RunTree if one can be extracted, else None.
        """
        if config is None:
            return None
        try:
            return RunTree.from_runnable_config(config)
        except Exception:
            return None
