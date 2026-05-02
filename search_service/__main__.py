"""Entry point for running the search service as a module.

Usage::

    python -m search_service
"""

from search_service.server import SearchMCPServer


def main() -> None:
    """Create and run the MCP search server."""
    server = SearchMCPServer()
    server.run()


if __name__ == "__main__":
    main()
