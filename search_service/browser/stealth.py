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

    @classmethod
    def load_script(cls) -> str:
        """Load stealth.js script content (cached after first read).

        Returns:
            The JavaScript source as a string.

        Raises:
            FileNotFoundError: If ``stealth.min.js`` is not found at
                the expected path.
        """
        if cls._cached_script is None:
            if not cls._SCRIPT_PATH.exists():
                raise FileNotFoundError(
                    f"stealth.min.js not found at {cls._SCRIPT_PATH}. "
                    "Run 'npx extract-stealth-evasions' to generate it."
                )
            cls._cached_script = cls._SCRIPT_PATH.read_text(encoding="utf-8")
        return cls._cached_script
