"""Integration tests for MCP client and tool loading.

Tests verify that:
- ``MCPSearchClient`` correctly manages MCP session lifecycle
- ``load_settings`` returns expected configuration
- MCP tool filtering works correctly
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deep_research_agent.agents.mcp_client import MCPSearchClient


# ---------------------------------------------------------------------------
# MCPSearchClient
# ---------------------------------------------------------------------------

class TestMCPSearchClient:
    """Tests for ``MCPSearchClient`` lifecycle and tool access."""

    def test_default_server_params(self) -> None:
        """Client uses python -m search_service by default."""
        import sys

        client = MCPSearchClient()
        assert client._server_params.command == sys.executable
        assert client._server_params.args == ["-m", "search_service"]

    def test_custom_server_params(self) -> None:
        """Client accepts custom command and args."""
        client = MCPSearchClient(
            server_command="/usr/bin/python3",
            server_args=["-m", "my_server"],
        )
        assert client._server_params.command == "/usr/bin/python3"
        assert client._server_params.args == ["-m", "my_server"]

    @pytest.mark.asyncio
    async def test_aenter_loads_tools(self) -> None:
        """__aenter__ spawns server and discovers tools."""
        fake_tools = [
            MagicMock(name="web_search"),
            MagicMock(name="scrape_url"),
        ]
        fake_tools[0].name = "web_search"
        fake_tools[1].name = "scrape_url"

        with patch(
            "deep_research_agent.agents.mcp_client.stdio_client",
        ) as mock_stdio, patch(
            "deep_research_agent.agents.mcp_client.ClientSession",
        ) as mock_session_cls, patch(
            "deep_research_agent.agents.mcp_client.load_mcp_tools",
            new_callable=AsyncMock,
            return_value=fake_tools,
        ):
            # Mock stdio context manager
            mock_stdio_cm = AsyncMock()
            mock_stdio_cm.__aenter__ = AsyncMock(
                return_value=(MagicMock(), MagicMock()),
            )
            mock_stdio_cm.__aexit__ = AsyncMock(return_value=None)
            mock_stdio.return_value = mock_stdio_cm

            # Mock session context manager
            mock_session = AsyncMock()
            mock_session.initialize = AsyncMock()
            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = mock_session_cm

            client = MCPSearchClient()
            result = await client.__aenter__()

            assert result is client
            assert len(client.get_tools()) == 2

    @pytest.mark.asyncio
    async def test_aexit_cleans_up(self) -> None:
        """__aexit__ closes session and stdio transport."""
        client = MCPSearchClient()

        # Simulate entered state
        mock_session_cm = AsyncMock()
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)
        client._session_cm = mock_session_cm

        mock_stdio_cm = AsyncMock()
        mock_stdio_cm.__aexit__ = AsyncMock(return_value=None)
        client._stdio_cm = mock_stdio_cm

        client._session = MagicMock()
        client._tools = [MagicMock()]
        client._tool_map = {"test": MagicMock()}

        await client.__aexit__(None, None, None)

        mock_session_cm.__aexit__.assert_called_once()
        mock_stdio_cm.__aexit__.assert_called_once()
        assert client._tools == []
        assert client._tool_map == {}
        assert client._session is None

    def test_get_tools_returns_all(self) -> None:
        """get_tools() without names returns all tools."""
        client = MCPSearchClient()
        tool_a = MagicMock()
        tool_a.name = "a"
        tool_b = MagicMock()
        tool_b.name = "b"
        client._tools = [tool_a, tool_b]
        client._tool_map = {"a": tool_a, "b": tool_b}

        result = client.get_tools()
        assert len(result) == 2

    def test_get_tools_filters_by_name(self) -> None:
        """get_tools(names=[...]) returns only matching tools."""
        client = MCPSearchClient()
        tool_a = MagicMock()
        tool_a.name = "web_search"
        tool_b = MagicMock()
        tool_b.name = "scrape_url"
        client._tools = [tool_a, tool_b]
        client._tool_map = {"web_search": tool_a, "scrape_url": tool_b}

        result = client.get_tools(names=["web_search"])
        assert len(result) == 1
        assert result[0].name == "web_search"

    def test_get_tools_raises_on_missing_name(self) -> None:
        """get_tools raises ValueError for unknown tool names."""
        client = MCPSearchClient()
        client._tools = []
        client._tool_map = {}

        with pytest.raises(ValueError, match="not found"):
            client.get_tools(names=["nonexistent"])


# ---------------------------------------------------------------------------
# load_settings
# ---------------------------------------------------------------------------

class TestLoadSettings:
    """Tests for ``load_settings`` integration."""

    def test_returns_model_config(self) -> None:
        """load_settings returns planner and worker model names."""
        with patch.dict("os.environ", {}, clear=False):
            from deep_research_agent.agents.tools import load_settings

            result = load_settings()

        assert "planner_model" in result
        assert "worker_model" in result

    def test_no_search_config_key(self) -> None:
        """load_settings no longer includes search_config (handled by MCP server)."""
        with patch.dict("os.environ", {}, clear=False):
            from deep_research_agent.agents.tools import load_settings

            result = load_settings()

        # search_config was removed — MCP server manages its own config
        assert "search_config" not in result
