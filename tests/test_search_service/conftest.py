"""Shared pytest fixtures for search_service tests."""

from __future__ import annotations

import pytest

from search_service.browser.stealth import StealthInjector


@pytest.fixture(autouse=True)
def _reset_stealth_cache() -> None:
    """Reset StealthInjector cached script between ALL tests."""
    StealthInjector._cached_script = None
    yield
    StealthInjector._cached_script = None
