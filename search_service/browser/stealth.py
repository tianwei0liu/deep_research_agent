"""Stealth script injection for Playwright browser contexts.

Loads and caches ``stealth.min.js`` which patches browser fingerprinting
APIs to evade headless Chrome detection.

stealth.js capabilities:
1. Overrides ``navigator.webdriver`` → ``undefined``
2. Masks Chrome DevTools detection fingerprints
3. Spoofs WebGL/Canvas renderer information
4. Intercepts Headless Chrome feature checks
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Optional


class StealthInjector:
    """Manages loading and caching of stealth.min.js.

    Usage::

        script = StealthInjector.load_script()
        await context.add_init_script(script)

    To generate stealth.min.js::

        npx extract-stealth-evasions
        mv stealth.min.js search_service/resources/stealth.min.js
    """

    _SCRIPT_PATH: ClassVar[Path] = (
        Path(__file__).parent.parent / "resources" / "stealth.min.js"
    )
    _cached_script: ClassVar[Optional[str]] = None

    _FALLBACK_SCRIPT: ClassVar[str] = (
        "// stealth.min.js not available — running without anti-detection\n"
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
    )

    @classmethod
    def load_script(cls) -> str:
        """Load stealth.js script content (cached after first read).

        Returns a minimal navigator.webdriver override if the full
        stealth.min.js is not present, rather than raising an error.

        Returns:
            The JavaScript source as a string.
        """
        if cls._cached_script is None:
            if not cls._SCRIPT_PATH.exists():
                import logging
                logging.getLogger(__name__).warning(
                    "stealth.min.js not found at %s — using minimal fallback. "
                    "Run 'npx extract-stealth-evasions' to generate it.",
                    cls._SCRIPT_PATH,
                )
                cls._cached_script = cls._FALLBACK_SCRIPT
            else:
                cls._cached_script = cls._SCRIPT_PATH.read_text(encoding="utf-8")
        return cls._cached_script
