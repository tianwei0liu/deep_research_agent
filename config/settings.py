"""
Application configuration.

- Secrets (API keys): from environment only. Put only in .env:
  LANGCHAIN_API_KEY, DEEPSEEK_API_KEY, GOOGLE_GEMINI_API_KEY, TAVILY_API_KEY
- Non-sensitive options: from config/settings.yaml (package), overridden by env vars
  RESEARCH_ASSISTANT_*, TAVILY_*, etc. LangChain/LangSmith options in YAML are
  applied to os.environ at load time so the SDK still sees them.

Load .env (e.g. python-dotenv) in your entry point before first use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Immutable settings: secrets from env, options from YAML + env override."""

    # --- API Keys ---
    gemini_api_key: str
    tavily_api_key: str

    # --- Deep Agent Models ---
    planner_model: str
    planner_temperature: float
    worker_model: str
    worker_temperature: float

    # --- Grader (LLM-as-Judge, uses Gemini) ---
    grader_model: str
    grader_temperature: float
    grader_thinking_level: str
    grader_google_search_enabled: bool

    # --- Tavily ---
    tavily_search_url: str
    tavily_max_result_chars: int

    # --- Orchestration Limits (used by benchmark runner) ---
    default_max_parallel_workers: int
    default_recursion_limit: int
    default_worker_max_tool_calls: int
    default_worker_max_turns: int
    default_worker_max_output_tokens: int

    def require_gemini_api_key(self) -> str:
        if not self.gemini_api_key:
            raise ValueError("GOOGLE_GEMINI_API_KEY is not set")
        return self.gemini_api_key

    def require_tavily_api_key(self) -> str:
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is not set")
        return self.tavily_api_key

    @staticmethod
    def _env(key: str, default: str = "") -> str:
        value = os.environ.get(key, default).strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1].strip()
        return value

    @staticmethod
    def _load_yaml_defaults() -> dict:
        """Load non-sensitive defaults from settings.yaml (project root or RESEARCH_ASSISTANT_SETTINGS_PATH)."""
        try:
            import yaml
        except ImportError:
            return {}
        base = Path(__file__).resolve().parent
        path = os.environ.get("RESEARCH_ASSISTANT_SETTINGS_PATH") or str(base / "settings.yaml")
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return {}
        if not data:
            return {}
        # Apply LangChain/LangSmith options to environment so the SDK reads them.
        lc = data.get("langchain") or {}
        if isinstance(lc, dict):
            if lc.get("tracing_v2") is True:
                os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
                os.environ.setdefault("LANGSMITH_TRACING", "true")
            if lc.get("project"):
                proj = str(lc["project"])
                os.environ.setdefault("LANGCHAIN_PROJECT", proj)
                os.environ.setdefault("LANGSMITH_PROJECT", proj)
        if data.get("langsmith_endpoint"):
            os.environ.setdefault("LANGSMITH_ENDPOINT", str(data["langsmith_endpoint"]))
        api_key = os.environ.get("LANGCHAIN_API_KEY") or os.environ.get("LANGSMITH_API_KEY")
        if api_key:
            os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
            os.environ.setdefault("LANGSMITH_API_KEY", api_key)
        return data

    @classmethod
    def load(cls) -> Settings:
        """Build Settings from environment and settings.yaml (no cache)."""
        yaml_data = cls._load_yaml_defaults()
        return cls(
            gemini_api_key=cls._env("GOOGLE_GEMINI_API_KEY"),
            tavily_api_key=cls._env("TAVILY_API_KEY"),
            planner_model=yaml_data.get("planner_model", "deepseek-chat"),
            planner_temperature=float(yaml_data.get("planner_temperature", 0.0)),
            worker_model=yaml_data.get("worker_model", "deepseek-chat"),
            worker_temperature=float(yaml_data.get("worker_temperature", 0.0)),
            grader_model=yaml_data.get("grader_model", "deepseek-chat"),
            grader_temperature=float(yaml_data.get("grader_temperature", 0.0)),
            grader_thinking_level=yaml_data.get("grader_thinking_level", "medium"),
            grader_google_search_enabled=bool(yaml_data.get("grader_google_search_enabled", True)),
            tavily_search_url=yaml_data.get("tavily_search_url", "https://api.tavily.com/search"),
            tavily_max_result_chars=int(yaml_data.get("tavily_max_result_chars", 12000)),
            default_max_parallel_workers=int(yaml_data.get("default_max_parallel_workers", 10)),
            default_recursion_limit=int(yaml_data.get("default_recursion_limit", 25)),
            default_worker_max_tool_calls=int(yaml_data.get("default_worker_max_tool_calls", 40)),
            default_worker_max_turns=int(yaml_data.get("default_worker_max_turns", 10)),
            default_worker_max_output_tokens=int(yaml_data.get("default_worker_max_output_tokens", 8192)),
        )
