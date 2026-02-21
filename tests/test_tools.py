"""Unit tests for Tavily tools."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from deep_research_agent.tools.tavily import TavilySearchTool
from deep_research_agent.config import Settings


@pytest.mark.asyncio
async def test_tavily_search_async():
    """Test TavilySearchTool.search_async calls httpx correctly."""
    settings = Settings(
        tavily_api_key="test-key",
        gemini_api_key="test-gemini-key",
        worker_model="gemini-2.5-flash",
        worker_temperature=0.0,
        planner_model="gemini-2.5-flash",
        planner_temperature=0.0,
        composer_model="gemini-2.5-flash",
        composer_temperature=1.0,
        grader_model="gemini-2.5-flash",
        grader_temperature=0.0,
        tavily_search_url="https://api.tavily.com/search",
        tavily_max_result_chars=1000,
        default_max_parallel_workers=10,
        default_recursion_limit=20,
        default_worker_max_tool_calls=40,
        default_worker_max_turns=10,
        default_worker_max_output_tokens=8192,
        worker_context_caching_enabled=False,
        supervisor_thinking_level="medium",
        worker_thinking_level="medium",
        composer_thinking_level="medium",
        grader_thinking_level="medium",
    )
    tool = TavilySearchTool(settings)

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {"title": "Result 1", "url": "http://example.com/1", "content": "Content 1"},
            {"title": "Result 2", "url": "http://example.com/2", "content": "Content 2"},
        ],
        "query": "test query"
    }
    mock_response.raise_for_status = MagicMock()

    # Mock httpx.AsyncClient
    with patch("httpx.AsyncClient") as MockClient:
        mock_client_instance = MockClient.return_value
        # Async context manager __aenter__ returns the client instance
        mock_client_instance.__aenter__.return_value = mock_client_instance
        # post is an async method
        mock_client_instance.post = AsyncMock(return_value=mock_response)

        result = await tool.search_async(query="test query")

        # Verify post called with correct args
        mock_client_instance.post.assert_awaited_once()
        call_args = mock_client_instance.post.call_args
        arg_url = call_args[0][0]
        arg_kwargs = call_args[1]
        
        assert "api.tavily.com/search" in str(arg_url) or "https://api.tavily.com/search" == str(arg_url) # check setting logic if needed
        assert arg_kwargs["json"]["query"] == "test query"
        assert arg_kwargs["headers"]["Authorization"] == "Bearer test-key"
        
        # Verify result format
        assert result["count"] == 2
        assert "Content 1" in result["results_summary"]
