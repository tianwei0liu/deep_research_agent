"""Tavily web search tool (§7.1)."""

import logging

from typing import Any, Callable, Literal, Optional
from pydantic import BaseModel, Field
import httpx


from deep_research_agent.config import Settings

logger = logging.getLogger(__name__)


class TavilySearchTool:
    """Tavily web search tool. Provides declaration and search implementation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # --- Schema ---
    class TavilySearchInput(BaseModel):
        """Input for Tavily web search."""
        query: str = Field(..., description="The search query.")
        max_results: int = Field(10, description="Maximum number of results (1-20). Prefer more results for more information to digest.")
        topic: Literal["general", "news", "finance"] = Field("general", description="general, news, or finance. Use news for recent events.")
        time_range: Optional[Literal["day", "week", "month", "year", "d", "w", "m", "y"]] = Field(None, description="Filter by recency.")


    @classmethod
    def get_declaration(cls) -> dict[str, Any]:
        """Return the Gemini function declaration for this tool."""
        return {
            "name": "tavily_search",
            "description": "Web search for current, factual information. Optimize for gathering comprehensive information in fewer steps.",
            "parameters": cls.TavilySearchInput.model_json_schema(),
        }

    def _search_impl(
        self,
        input_data: TavilySearchInput,
        search_depth: str = "basic",
    ) -> dict[str, Any]:
        """Run the search using Tavily API."""
        api_key = self.settings.tavily_api_key
        if not api_key:
            logger.warning("TAVILY_API_KEY is not set")
            return {"error": "TAVILY_API_KEY is not set"}

        body: dict[str, Any] = {
            "query": input_data.query,
            "search_depth": search_depth,
            "max_results": min(20, max(1, input_data.max_results)),
            "topic": input_data.topic,
        }
        if input_data.time_range:
            body["time_range"] = input_data.time_range
        body["include_answer"] = False

        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.post(
                    self.settings.tavily_search_url,
                    json=body,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPStatusError as e:
            logger.warning("Tavily API error: %s %s", e.response.status_code, e.response.text[:200])
            return {"error": f"Tavily API error: {e.response.status_code} {e.response.text[:200]}"}
        except Exception as e:
            logger.exception("Tavily request failed")
            return {"error": f"Tavily request failed: {e!s}"}

        results = data.get("results", [])
        max_chars = self.settings.tavily_max_result_chars
        parts = []
        for i, item in enumerate(results):
            if i >= 10:
                break
            title = item.get("title", "")
            url = item.get("url", "")
            content = (item.get("content") or "")[:500]
            parts.append(f"[{i+1}] {title}\nURL: {url}\n{content}")
        text = "\n\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[truncated]"
        logger.debug("tavily_search returned %d results for query=%r", len(results), input_data.query[:80])
        return {"results_summary": text, "count": len(results), "query": data.get("query", input_data.query)}

    async def _search_impl_async(
        self,
        input_data: TavilySearchInput,
        search_depth: str = "basic",
    ) -> dict[str, Any]:
        """Run the search using Tavily API (async)."""
        api_key = self.settings.tavily_api_key
        if not api_key:
            logger.warning("TAVILY_API_KEY is not set")
            return {"error": "TAVILY_API_KEY is not set"}

        body: dict[str, Any] = {
            "query": input_data.query,
            "search_depth": search_depth,
            "max_results": min(20, max(1, input_data.max_results)),
            "topic": input_data.topic,
        }
        if input_data.time_range:
            body["time_range"] = input_data.time_range
        body["include_answer"] = False

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(
                    self.settings.tavily_search_url,
                    json=body,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPStatusError as e:
            logger.warning("Tavily API error: %s %s", e.response.status_code, e.response.text[:200])
            return {"error": f"Tavily API error: {e.response.status_code} {e.response.text[:200]}"}
        except Exception as e:
            logger.exception("Tavily request failed")
            return {"error": f"Tavily request failed: {e!s}"}

        results = data.get("results", [])
        max_chars = self.settings.tavily_max_result_chars
        parts = []
        for i, item in enumerate(results):
            if i >= 10:
                break
            title = item.get("title", "")
            url = item.get("url", "")
            content = (item.get("content") or "")[:500]
            parts.append(f"[{i+1}] {title}\nURL: {url}\n{content}")
        text = "\n\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n[truncated]"
        logger.debug("tavily_search_async returned %d results for query=%r", len(results), input_data.query[:80])
        return {"results_summary": text, "count": len(results), "query": data.get("query", input_data.query)}

    def search(
        self,
        **kwargs
    ) -> dict[str, Any]:
        """Run Tavily search (traceable). Used as the tool implementation by the registry."""
        input_data = self.TavilySearchInput(**kwargs)
        return self._search_impl(input_data)

    async def search_async(
        self,
        **kwargs
    ) -> dict[str, Any]:
        """Run Tavily search (traceable, async). Used as the tool implementation by the registry."""
        input_data = self.TavilySearchInput(**kwargs)
        return await self._search_impl_async(input_data)


    @staticmethod
    def make_impl(settings: Settings) -> Any:
        """
        Return a callable that runs tavily search with the given settings.
        Used by the tool registry; settings are not read from environment here.
        """
        return TavilySearchTool(settings).search_async
