# Contributing

Thank you for considering contributing to **Research Assistant**! This guide will help you get started.

## Prerequisites

- Python ≥ 3.10
- API keys for [Gemini](https://ai.google.dev/) and [Tavily](https://tavily.com/) (see `.env.example`)

## Development Setup

```bash
# Clone and install in development mode with test dependencies
git clone https://github.com/tianwei0liu/deep_research_agent.git
cd deep_research_agent
pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Running Tests

```bash
PYTHONPATH=.. python3 -m pytest tests/ -v
```

## Code Style

This project follows strict coding standards:

- **Type hints** on all function signatures (`typing` module).
- **No `print()`** — use `logging` (`self.logger = logging.getLogger(__name__)`).
- **Pydantic models** for all data boundaries (no raw dicts).
- **`async/await`** for I/O-bound operations (LLM calls, web requests).
- **Google-style docstrings** on every class and public method.
- **No global state** — pass configuration via dependency injection.

## Pull Request Guidelines

1. Fork the repository and create a feature branch from `main`.
2. Write or update tests for your changes.
3. Ensure `pytest tests/ -v` passes with no failures.
4. Keep commits focused and write clear commit messages.
5. Open a PR with a description of what and why.

## Reporting Issues

Please open a GitHub issue with:
- A clear description of the problem or feature request.
- Steps to reproduce (for bugs).
- Expected vs actual behavior.
