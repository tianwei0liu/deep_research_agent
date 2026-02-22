"""
Tool Registry: Central repository for all available tools.
"""

from typing import Any, Callable
from deep_research_agent.config import Settings



class ToolRegistry:
    """
    Registry for tools.
    Maps tool names to their (declaration, factory) tuples.
    """
    _registry: dict[str, tuple[dict[str, Any], Callable[[Settings], Callable[..., Any]]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        declaration: dict[str, Any],
        factory: Callable[[Settings], Callable[..., Any]],
    ) -> None:
        """Register a tool with its declaration and factory."""
        cls._registry[name] = (declaration, factory)

    @classmethod
    def resolve(
        cls, names: list[str], settings: Settings
    ) -> tuple[list[dict[str, Any]], dict[str, Callable[..., Any]]]:
        """
        Resolve a list of tool names to their declarations and implementations.
        Returns:
            (list of declarations, dict of name -> implementation)
        """
        declarations = []
        implementations = {}

        if not cls._registry:
            cls.register_all_tools()

        for name in names:
            if name not in cls._registry:
                raise ValueError(f"Tool '{name}' not found in registry.")
            
            decl, factory = cls._registry[name]
            declarations.append(decl)
            implementations[name] = factory(settings)
            
        return declarations, implementations

    @classmethod
    def register_all_tools(cls):
        """Register all default tools."""
        from deep_research_agent.tools import tavily
        from deep_research_agent.tools import filesystem
        from deep_research_agent.tools import planning
        from deep_research_agent.tools import delegation
        from deep_research_agent.tools import control

        # 1. Search
        cls.register(
            "tavily_search", 
            tavily.TavilySearchTool.get_declaration(), 
            tavily.TavilySearchTool.make_impl
        )

        # 2. Filesystem
        cls.register(
            "write_file", 
            filesystem.FilesystemTool.get_declaration(), 
            filesystem.FilesystemTool.make_impl
        )

        # 3. Planning
        cls.register(
            "add_task", 
            planning.PlanningTool.get_add_task_declaration(), 
            planning.PlanningTool.make_add_impl
        )
        cls.register(
            "remove_task", 
            planning.PlanningTool.get_remove_task_declaration(), 
            planning.PlanningTool.make_remove_impl
        )

        # 4. Delegation
        cls.register(
            "delegate_research", 
            delegation.DelegationTool.get_declaration(), 
            delegation.DelegationTool.make_impl
        )

        # 5. Control
        cls.register(
            "finish", 
            control.ControlTool.get_declaration(), 
            control.ControlTool.make_impl
        )
