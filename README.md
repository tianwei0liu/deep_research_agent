# Deep Research Agent

A multi-agent research system that autonomously conducts deep web research to answer complex questions. Inspired by [Anthropic's multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system).

## Architecture

The system uses a **Supervisor–Worker** pattern built on [LangGraph](https://github.com/langchain-ai/langgraph):

```
User Query
    │
    ▼
┌──────────┐      ┌──────────┐
│Supervisor│─────▶│  Worker   │──▶ Tavily Search
│(Planner) │◀─────│ (Agent)  │──▶ File System
└──────────┘      └──────────┘
    │                  ▲
    │  parallel        │
    ▼  delegation      │
┌──────────┐           │
│  Worker  │───────────┘
│ (Agent)  │
└──────────┘
    │
    ▼
┌──────────┐
│ Composer │──▶ Final Report
└──────────┘
```

- **Supervisor** — Powered by Gemini. Decomposes the user query into a DAG of research tasks, delegates them to workers (in parallel when possible), and iterates until sufficient information is gathered.
- **Worker** — An autonomous research agent with access to web search (Tavily) and file system tools. Executes a single research task and returns structured findings.
- **Composer** — Synthesizes all worker findings into a comprehensive, cited final report.

## Features

- **Dynamic task planning** — The supervisor creates, updates, and removes research tasks on the fly based on intermediate results.
- **Parallel worker delegation** — Independent tasks are delegated to multiple workers simultaneously.
- **Structured findings** — Workers return quantitative metrics, key findings, and source citations.
- **Defense-in-depth** — Implicit finish detection ensures reports are generated even if the LLM skips the `finish` tool.
- **Configurable models & limits** — All model names, temperatures, token limits, and concurrency are configurable via `config/settings.yaml`.
- **LangSmith tracing** — Built-in integration for observability and debugging.
- **Benchmarking** — Automated benchmark suite with LLM-graded evaluation.

## Prerequisites

- Python ≥ 3.10
- API keys for:
  - [Google Gemini](https://ai.google.dev/) (worker and supervisor LLM)
  - [Tavily](https://tavily.com/) (web search)
  - [LangSmith](https://smith.langchain.com/) (optional, for tracing)

## Installation

```bash
# Clone the repository
git clone https://github.com/tianwei0liu/deep_research_agent.git
cd deep_research_agent

# Install in development mode
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env and fill in your API keys
```

## Quick Start

```python
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from deep_research_agent.config import Settings
from deep_research_agent.agents.orchestrator.graph import OrchestratorGraph
from deep_research_agent.agents.orchestrator.state import OrchestratorConfig
from deep_research_agent.agents.worker.schemas import Limits

Settings.load()

async def main():
    orchestrator = OrchestratorGraph()
    app = orchestrator.compile()

    config = OrchestratorConfig(
        messages=[HumanMessage(content="What is the current state of AI Agents in 2025?")],
        max_parallel_workers=10,
        recursion_limit=20,
        worker_limits=Limits(max_tool_calls=40, max_turns=10, max_output_tokens=8192),
    )

    initial_state = config.to_state()
    async for event in app.astream(initial_state, config={"recursion_limit": 20}):
        for key, value in event.items():
            if "final_report" in value:
                print(value["final_report"])

asyncio.run(main())
```

## Configuration

### API Keys (`.env`)

Copy `.env.example` to `.env` and fill in your keys. See [.env.example](.env.example) for the required variables.

### System Settings (`config/settings.yaml`)

All non-sensitive settings live in [`config/settings.yaml`](config/settings.yaml):

#### Models

| Setting | Description | Default |
|---------|-------------|---------:|
| `planner_model` | LLM model for the Supervisor (orchestrator) | `deepseek-v4-flash` |
| `worker_model` | LLM model for research workers | `deepseek-v4-flash` |
| `grader_model` | LLM model for the benchmark grader | `deepseek-v4-flash` |

#### Orchestrator & Worker Limits

| Setting | Description | Default |
|---------|-------------|---------:|
| `max_parallel_workers` | Max concurrent worker subagents | `10` |
| `supervisor_max_turns` | Max Supervisor reasoning turns before forced termination | `100` |
| `supervisor_max_search_calls` | Max direct `internet_search` calls by the Supervisor | `100` |
| `worker_max_tool_calls` | Max total tool invocations per worker task | `500` |
| `worker_max_turns` | Max reasoning turns per worker task | `100` |
| `worker_max_output_tokens` | Max output tokens per worker response | `8192` |

## Project Structure

```
deep_research_agent/
├── __init__.py                  # Public API
├── config/
│   ├── settings.py              # Settings loader (YAML + env)
│   └── settings.yaml            # Default configuration
├── agents/
│   ├── orchestrator/
│   │   ├── graph.py             # LangGraph state machine
│   │   ├── supervisor.py        # Supervisor node (plan + delegate)
│   │   ├── composer.py          # Report composer node
│   │   ├── prompts.py           # LLM prompt templates
│   │   ├── schemas.py           # ResearchTask, TaskStatus
│   │   └── state.py             # OrchestratorState TypedDict
│   ├── worker/
│   │   ├── graph.py             # Worker state machine
│   │   ├── worker.py            # Worker entrypoint
│   │   ├── nodes.py             # Worker graph nodes
│   │   ├── prompts.py           # Worker prompt templates
│   │   ├── schemas.py           # SpawnTask, Limits, WorkerResult
│   │   └── state.py             # WorkerState TypedDict
│   └── utils/
│       ├── cache_manager.py     # Context caching
│       └── tracing.py           # LangSmith tracing utilities
├── tools/
│   ├── registry.py              # Tool registry
│   ├── planning.py              # add_task, update_status, remove_task
│   ├── delegation.py            # delegate_research
│   ├── control.py               # finish
│   ├── tavily.py                # Web search via Tavily API
│   └── filesystem.py            # write_file
├── benchmarks/
│   ├── benchmark_case.py        # Benchmark case model
│   ├── datasets/core.json       # Benchmark cases
│   ├── runner.py                # Benchmark runner
│   └── grader.py                # LLM-based answer grading
├── examples/                    # Runnable example scripts
├── tests/                       # Pytest test suite
├── docs/                        # Design documents
├── pyproject.toml
├── requirements.txt
└── LICENSE                      # MIT
```

## Running Examples

```bash
# Install in dev mode first (if not already done)
pip install -e ".[dev]"

# Run a full orchestrator verification
python3 examples/verify_orchestrator.py

# Run the benchmark suite
python3 examples/run_benchmark.py

# Run a single worker
python3 examples/run_worker.py

# Run parallel orchestrator
python3 examples/verify_parallel_orchestrator.py
```

## Running Tests

```bash
pytest tests/ -v
```

## Documentation

Design documents are in the [`docs/`](docs/) directory:

- [Orchestrator Design](docs/orchestrator_design.md) — Supervisor architecture and routing
- [Worker Design](docs/worker_design.md) — Worker agent design and state machine
- [Parallel Execution Design](docs/parallel_execution_design.md) — Parallel worker delegation
- [Context Caching Design](docs/context_caching_design.md) — Gemini context caching strategy
- [Research Handbook](docs/research_handbook.md) — Supervisor SOP and interaction examples
- [Fix: Implicit Finish](docs/fix_implicit_finish.md) — Defense-in-depth for missed `finish` calls

## License

[MIT](LICENSE) © tianwei0liu
