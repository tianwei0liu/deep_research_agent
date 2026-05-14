"""MCP client for connecting to the search service.

Provides :class:`MCPSearchClient`, an async context manager that spawns
the ``search_service`` MCP server as a stdio subprocess, initializes
the MCP session, and dynamically discovers tools as LangChain
:class:`~langchain_core.tools.BaseTool` objects.

Usage::

    async with MCPSearchClient() as client:
        all_tools = client.get_tools()
        web_only = client.get_tools(names=["web_search"])
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)


class MCPSearchClient:
    """Async context manager for the MCP search service connection.

    Spawns ``python -m search_service`` as a stdio subprocess,
    initializes the MCP handshake, and loads all advertised tools
    as LangChain ``BaseTool`` instances via dynamic discovery.

    The MCP session stays alive for the lifetime of the context
    manager, so tools can make calls throughout the agent's run.

    Attributes:
        _tools: Cached list of discovered LangChain tools.
        _tool_map: Name→tool mapping for filtered access.
    """

    def __init__(
        self,
        *,
        server_command: Optional[str] = None,
        server_args: Optional[list[str]] = None,
    ) -> None:
        """Initialize MCP client configuration.

        Args:
            server_command: Python executable to use. Defaults to
                ``sys.executable`` (the current interpreter).
            server_args: Arguments for the server process. Defaults
                to ``["-m", "search_service"]``.
        """
        self._server_params = StdioServerParameters(
            command=server_command or sys.executable,
            args=server_args or ["-m", "search_service"],
        )
        self._tools: list[BaseTool] = []
        self._tool_map: dict[str, BaseTool] = {}
        self._session: Optional[ClientSession] = None
        # Context manager state for cleanup
        self._stdio_cm: Optional[object] = None
        self._session_cm: Optional[object] = None
        self._read = None
        self._write = None

    async def __aenter__(self) -> MCPSearchClient:
        """Spawn the MCP server, initialize session, and discover tools.

        Returns:
            Self with tools loaded and ready.

        Raises:
            RuntimeError: If the MCP server fails to start or
                tool discovery returns no tools.
        """
        # 1. Spawn stdio subprocess
        self._stdio_cm = stdio_client(self._server_params)
        self._read, self._write = await self._stdio_cm.__aenter__()

        # 2. Initialize MCP session
        self._session_cm = ClientSession(self._read, self._write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()

        # 3. Dynamically discover and convert tools
        self._tools = await load_mcp_tools(self._session)
        self._tool_map = {tool.name: tool for tool in self._tools}

        if not self._tools:
            logger.warning("mcp_no_tools_discovered from search service")
        else:
            tool_names = [t.name for t in self._tools]
            logger.info(
                "mcp_tools_discovered count=%d tools=%s",
                len(self._tools),
                tool_names,
            )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[type-arg]
        """Gracefully close the MCP session and terminate the subprocess."""
        errors: list[str] = []

        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as exc:
                errors.append(f"session_close: {exc}")
            self._session_cm = None

        if self._stdio_cm is not None:
            try:
                await self._stdio_cm.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as exc:
                errors.append(f"stdio_close: {exc}")
            self._stdio_cm = None

        self._session = None
        self._tools = []
        self._tool_map = {}

        if errors:
            logger.warning(
                "mcp_client_cleanup_errors: %s", "; ".join(errors),
            )
        else:
            logger.info("mcp_client_closed")

    def get_tools(
        self,
        *,
        names: Optional[list[str]] = None,
    ) -> list[BaseTool]:
        """Return discovered MCP tools, optionally filtered by name.

        Args:
            names: If provided, return only tools whose names are
                in this list. If ``None``, return all tools.

        Returns:
            List of LangChain ``BaseTool`` instances.

        Raises:
            ValueError: If a requested tool name was not found.
        """
        if names is None:
            return list(self._tools)

        result: list[BaseTool] = []
        for name in names:
            tool = self._tool_map.get(name)
            if tool is None:
                available = list(self._tool_map.keys())
                raise ValueError(
                    f"MCP tool '{name}' not found. "
                    f"Available tools: {available}"
                )
            result.append(tool)
        return result
