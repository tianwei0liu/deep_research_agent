"""
Tracing utilities for the Research Assistant.
Wraps langsmith.traceable with project-specific defaults.
"""

import functools
from typing import Optional, Callable, Any
from langsmith import traceable as ls_traceable


class Tracing:
    """
    Namespace for tracing utilities.
    """

    @staticmethod
    def trace(
        name: Optional[str] = None,
        run_type: str = "chain",
        **kwargs: Any
    ) -> Callable:
        """
        Decorator to trace a function or method with LangSmith.
        
        If name is not provided, it defaults to the qualified name of the function
        (e.g., "ClassName.method_name" or "function_name").
        """
        def decorator(func: Callable) -> Callable:
            # Determine the name if not provided
            trace_name = name
            if not trace_name:
                # Try to get the qualified name
                if hasattr(func, "__qualname__"):
                    trace_name = func.__qualname__
                else:
                    trace_name = func.__name__
            
            # Apply the langsmith traceable decorator
            return ls_traceable(name=trace_name, run_type=run_type, **kwargs)(func)
        
        return decorator
