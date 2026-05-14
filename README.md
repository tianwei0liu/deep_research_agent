# Deep Research Agent

A multi-agent research system that autonomously conducts deep web research to answer complex questions. Inspired by [Anthropic's multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system).

## Architecture

The system uses a **Supervisor → Worker → Citation Specialist** pipeline built on [LangGraph](https://github.com/langchain-ai/langgraph) via the [`deepagents.create_deep_agent`](https://github.com/langchain-ai/deepagents) factory.

### Agent Topology

```
                      User Query
                          │
                          ▼
                 ┌─────────────────┐
                 │   Supervisor    │ ← BudgetTrackingMiddleware (per-turn budget injection)
                 │  (Orchestrator) │ ← CitationDataMiddleware (auto-inject findings + L1 gate)
                 └───────┬─────────┘
                         │
            ┌────────────┼─────────────┐
            │            │             │
            ▼            ▼             ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Worker 1 │  │ Worker 2 │  │ Worker N │   ← Parallel subagents
    └────┬─────┘  └────┬─────┘  └────┬─────┘
         │             │             │
         ▼             ▼             ▼
    ┌─────────────────────────────────────┐
    │        WorkerOutput (Pydantic)      │   ← Structured findings with provenance
    │  { findings: [Finding], sources }   │
    └─────────────────────────────────────┘
                         │
                         ▼
               ┌──────────────────┐
               │    Citation      │ ← CitationDataMiddleware auto-injects
               │   Specialist     │   worker findings + L1 validation gate
               └────────┬─────────┘
                         │
                         ▼
               ┌──────────────────┐
               │  Final Report    │   ← Markdown with inline [1][2] citations
               │  + ## Sources    │
               └──────────────────┘
```

### Agent Roles

| Agent | Model | Responsibility |
|-------|-------|----------------|
| **Supervisor** | DeepSeek V4 Flash (thinking mode) | Decomposes user queries into a DAG of research tasks, delegates to workers (parallel when possible), performs effort scaling (trivial → deep), writes draft reports, triggers citation |
| **Research Worker** | DeepSeek V4 Flash (thinking disabled) | Executes a single scoped research task via SearXNG search + Playwright URL scraping. Returns structured `WorkerOutput` (Pydantic) with `Finding` triples preserving fact-to-source provenance |
| **Citation Specialist** | DeepSeek V4 Flash (thinking disabled) | Adds sequential inline citations `[1]`, `[2]`... to the draft report based on worker findings. Output is validated by L1 structural rules before reaching the user |

### ID & Data Flow

```
1. User sends query
   │
2. stream_deep_research(query, thread_id="abc")
   │  ├── thread_id  → LangGraph checkpoint scope (multi-turn state isolation)
   │  └── run_id     → LangSmith trace ID (yielded as first event)
   │
3. Supervisor receives query as HumanMessage
   │  ├── Classifies complexity (trivial → deep)
   │  ├── For trivial: calls internet_search directly → responds
   │  └── For non-trivial: creates research tasks
   │
4. Supervisor delegates via tool call: task(subagent_type="research-worker", description="...")
   │  ├── Each task() → spawns a Worker subagent (react agent)
   │  ├── Workers share NO state — fully isolated subgraph execution
   │  └── Multiple independent task() calls in one turn → parallel execution
   │
5. Worker executes research
   │  ├── Tools: internet_search (SearXNG) + scrape_url (Playwright)
   │  ├── response_format=WorkerOutput enforces structured output
   │  └── Returns WorkerOutput JSON via ToolMessage to Supervisor
   │
6. Supervisor reviews findings, may:
   │  ├── Delegate more workers (iterative deepening)
   │  ├── Search directly (fact-check via internet_search)
   │  └── Write draft report → delegate to citation-specialist
   │
7. task(subagent_type="citation-specialist", description=<draft_report>)
   │  ├── CitationDataMiddleware intercepts this call:
   │  │   ├── Extracts all WorkerOutput from ToolMessages in Supervisor state
   │  │   └── Appends findings JSON to the task description
   │  ├── Citation Specialist adds [N] markers + ## Sources section
   │  └── L1 validation gate checks output:
   │      ├── PASS → return to Supervisor
   │      └── FAIL → auto-retry with correction instructions (up to N retries)
   │
8. Supervisor receives cited report → outputs as final response
   │
9. stream_deep_research yields:
      { type: "status" | "run_id" | "tool_start" | "tool_end" | "token" | "final_report" }
```

### Middleware Pipeline

Two middleware layers intercept Supervisor behavior at the LLM/tool-call boundary:

| Middleware | Hook | Purpose |
|-----------|------|---------|
| `BudgetTrackingMiddleware` | `wrap_model_call` | Counts `AIMessage` instances to track turns consumed. Appends `## ⏱ Budget Status` to system prompt before every LLM call. Injects CRITICAL stop warning when remaining turns ≤ 3 |
| `CitationDataMiddleware` | `wrap_tool_call` | Intercepts `task(subagent_type="citation-specialist")` calls. Auto-injects worker findings. Validates output against L1 structural rules. Auto-retries on validation failure (configurable max retries) |

### Citation Quality Assurance (L1 Validation)

The `CitationStructureValidator` enforces five rules on every cited report:

| Rule | Severity | Description |
|------|----------|-------------|
| L1-01 | ERROR | Dangling citation: body `[N]` without matching Sources entry |
| L1-02 | WARNING | Orphan source: Sources entry never cited in body |
| L1-03 | WARNING | Non-sequential numbering: gaps in source numbers |
| L1-04 | WARNING | Duplicate URL: same URL assigned multiple numbers |
| L1-05 | ERROR | Missing Sources section when citations exist (fast-fail) |

### Worker Output Schema (Provenance Chain)

Workers output structured `WorkerOutput` via Pydantic `response_format`:

```
WorkerOutput
├── summary: str              # 2-3 sentence overview
├── findings: list[Finding]   # Claim-source-evidence triples
│   └── Finding
│       ├── claim: str            # Factual statement (no [N] citations)
│       ├── source_urls: list[str] # Normalized URLs (RFC 3986)
│       ├── source_titles: list[str]
│       └── evidence: str          # Quote or paraphrase
├── sources_consulted: list[str]  # All URLs searched
└── caveats: str                  # Gaps or limitations
```

## Features

- **Effort-scaled task planning** — Supervisor classifies query complexity (trivial → deep) and adjusts worker delegation accordingly. Trivial queries are answered directly without delegation.
- **Parallel worker delegation** — Independent tasks are delegated to multiple workers simultaneously via concurrent `task()` tool calls.
- **Structured provenance** — Workers return `WorkerOutput` with `Finding` triples binding each claim to its source URLs and evidence.
- **Automated citation pipeline** — Citation Specialist adds inline `[N]` markers; L1 structural validation gates output quality with auto-retry.
- **Dynamic budget tracking** — Per-turn budget injection via middleware. Critical stop warning at ≤ 3 remaining turns.
- **Multi-turn conversations** — Checkpointer-based state persistence via `thread_id`. Reuse the same thread to continue research.
- **Streaming output** — `stream_deep_research()` yields real-time events (status, tool calls, tokens, final report).
- **Timeout recovery** — Configurable research timeout with checkpoint-resume finalization.
- **Self-hosted search** — SearXNG-backed search with Playwright URL scraping, replacing Tavily dependency.
- **LangSmith tracing** — Full observability via `astream_events`. Run ID yielded as first event for trace linking.
- **Benchmarking** — LLM-as-Judge benchmark suite with configurable grader.

## Prerequisites

- Python ≥ 3.10
- Docker (for SearXNG search service)
- API keys for:
  - [DeepSeek](https://platform.deepseek.com/) (Supervisor + Worker LLM)
  - [Google Gemini](https://ai.google.dev/) (Grader LLM, optional)
  - [LangSmith](https://smith.langchain.com/) (optional, for tracing)

## Installation

```bash
# Clone the repository
git clone https://github.com/tianwei0liu/deep_research_agent.git
cd deep_research_agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env and fill in your API keys (DEEPSEEK_API_KEY, etc.)

# Start the SearXNG search service
docker compose -f docker/docker-compose.yml up -d
```

## Quick Start

```python
import asyncio
from deep_research_agent import stream_deep_research
from langgraph.checkpoint.memory import MemorySaver

async def main():
    checkpointer = MemorySaver()
    async for event in stream_deep_research(
        "What is the current state of AI Agents in 2025?",
        thread_id="my-session",
        checkpointer=checkpointer,
    ):
        if event["type"] == "token":
            print(event["data"], end="", flush=True)
        elif event["type"] == "final_report":
            print("\n\n--- REPORT ---")
            print(event["data"])

asyncio.run(main())
```

## Configuration

### API Keys (`.env`)

Copy `.env.example` to `.env` and fill in your keys. See [.env.example](.env.example) for the required variables.

### System Settings (`config/settings.yaml`)

All non-sensitive settings live in [`config/settings.yaml`](config/settings.yaml):

#### Models

| Setting | Description | Default |
|---------|-------------|--------:|
| `planner_model` | LLM model for the Supervisor | `deepseek-v4-flash` |
| `worker_model` | LLM model for Workers + Citation Specialist | `deepseek-v4-flash` |
| `grader_model` | LLM model for the benchmark grader | `deepseek-v4-flash` |

#### Orchestration & Worker Limits

| Setting | Description | Default |
|---------|-------------|--------:|
| `supervisor_max_turns` | Max Supervisor reasoning turns before budget-critical warning | `35` |
| `supervisor_max_search_calls` | Max direct `internet_search` calls by Supervisor | `10` |
| `worker_max_search_calls` | Max search calls per Worker task | `60` |
| `worker_max_turns` | Max reasoning turns per Worker task | `20` |
| `worker_max_output_tokens` | Max output tokens per Worker response | `8192` |
| `max_parallel_workers` | Max concurrent Worker subagents | `10` |

#### Citation & Safety

| Setting | Description | Default |
|---------|-------------|--------:|
| `citation_max_retries` | Max L1 validation retries for Citation Specialist | `5` |
| `research_timeout_seconds` | Global research timeout (0 = no timeout) | `600` |

#### Search Service

| Setting | Description | Default |
|---------|-------------|--------:|
| `searxng_base_url` | SearXNG instance URL | `http://localhost:8080` |

## Project Structure

```
deep_research_agent/
├── __init__.py                  # Public API: build_deep_agent, run/stream_deep_research
├── config/
│   ├── settings.py              # Settings loader (YAML + env)
│   └── settings.yaml            # Default configuration
├── agents/                      # ★ Core agent system
│   ├── __init__.py              # Re-exports build_deep_agent, run/stream_deep_research
│   ├── agent.py                 # Agent factory (build_deep_agent) + streaming/runner
│   ├── prompts.py               # Supervisor / Worker / Citation Specialist prompts
│   ├── tools.py                 # Tool factories: internet_search (SearXNG), scrape_url (Playwright)
│   ├── budget_middleware.py     # BudgetTrackingMiddleware (per-turn budget injection)
│   ├── patched_deepseek.py      # DeepSeek V4 reasoning_content passback fix
│   └── citation/                # Citation subsystem
│       ├── models.py            # Finding, WorkerOutput, L1ValidationResult (Pydantic)
│       ├── citation_middleware.py  # CitationDataMiddleware (auto-inject + L1 gate)
│       └── structure_validator.py  # L1 structural validation rules
├── search_service/              # Self-hosted search infrastructure
│   ├── config.py                # SearchServiceConfig
│   ├── models.py                # SearchResponse, SearchResult
│   ├── server.py                # MCP server (FastMCP)
│   ├── cache.py                 # Search result caching
│   ├── rate_limiter.py          # Rate limiting
│   ├── exceptions.py            # Custom exceptions
│   ├── backends/
│   │   ├── base.py              # SearchRouter + abstract backend
│   │   ├── searxng_client.py    # SearXNG HTTP client
│   │   ├── page_scraper.py      # Playwright-based URL scraper
│   │   └── github_client.py     # GitHub search client
│   └── browser/                 # Headless browser pool (Playwright)
├── benchmarks/                  # LLM-as-Judge evaluation framework
│   ├── benchmark_case.py        # Benchmark case model
│   ├── datasets/core.json       # Benchmark cases
│   ├── runner.py                # Benchmark runner
│   └── grader.py                # LLM-based answer grading
├── docker/
│   ├── docker-compose.yml       # SearXNG + Redis service stack
│   └── searxng/                 # SearXNG configuration
├── examples/
│   ├── run_deep_agent.py        # Streaming CLI runner with multi-turn support
│   └── run_benchmark.py         # Benchmark suite runner
├── tests/
│   ├── test_budget_middleware.py # BudgetTrackingMiddleware unit tests
│   ├── test_citation/           # Citation system tests
│   ├── test_grader.py           # Grader tests
│   ├── test_patched_deepseek.py # DeepSeek patch tests
│   └── test_search_service/     # Search service tests
├── docs/                        # Design documents (historical + current)
├── refactory_doc/               # Iteration roadmap & design specs
│   ├── deep_research_agent_iteration_roadmap.md
│   ├── citation/                # Citation system design docs (4 phases)
│   └── search_tools/            # MCP search service design docs
├── scripts/                     # Utility scripts (trace analysis, etc.)
├── pyproject.toml
├── requirements.txt
└── LICENSE                      # MIT
```

## Running Examples

```bash
# Activate virtual environment
source .venv/bin/activate

# Run a deep research query (default query)
python examples/run_deep_agent.py

# Custom query
python examples/run_deep_agent.py "How does LangGraph implement checkpointing?"

# Multi-turn conversation (reuse thread-id)
python examples/run_deep_agent.py --thread-id my-session "What is RAG?"
python examples/run_deep_agent.py --thread-id my-session "How does it compare to fine-tuning?"

# Debug mode
python examples/run_deep_agent.py --verbose
python examples/run_deep_agent.py --log-level DEBUG

# Run the benchmark suite
python examples/run_benchmark.py
python examples/run_benchmark.py --category comparative
python examples/run_benchmark.py --cases factual_001
python examples/run_benchmark.py --verbose
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Documentation

### Design Documents

- [Architecture Information Flow](docs/architecture_information_flow.md) — Agent data flow and state management
- [Orchestrator Design](docs/orchestrator_design.md) — Supervisor architecture and routing
- [Worker Design](docs/worker_design.md) — Worker agent design and state machine
- [Research Handbook](docs/research_handbook.md) — Supervisor SOP and interaction patterns

### Refactoring & Iteration

- [Iteration Roadmap](refactory_doc/deep_research_agent_iteration_roadmap.md) — Phase 0–4 product roadmap
- [Citation System Design](refactory_doc/citation/) — 4-phase citation pipeline design
- [Search Service Design](refactory_doc/search_tools/) — MCP search service architecture

## License

[MIT](LICENSE) © tianwei0liu
