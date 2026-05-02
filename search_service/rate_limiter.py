"""Simple async rate limiter using a token-bucket algorithm.

Tech debt: This is a minimal in-process implementation sufficient for V1.
Future improvements:
- Sliding window algorithm for smoother burst handling.
- Redis-backed distributed rate limiting for multi-instance deployment.
- Per-domain rate limiting (currently per-limiter instance).
"""

from __future__ import annotations

import asyncio
import time


class AsyncRateLimiter:
    """Token-bucket rate limiter for async code.

    Allows up to ``max_calls`` within each ``period`` (seconds).
    When tokens are exhausted, :meth:`acquire` blocks until a token
    becomes available.

    Args:
        max_calls: Maximum number of calls allowed per period.
        period: Time window in seconds.

    Example::

        limiter = AsyncRateLimiter(max_calls=12, period=60.0)
        await limiter.acquire()  # blocks if quota exhausted
    """

    def __init__(self, max_calls: int, period: float) -> None:
        self._max_calls = max_calls
        self._period = period
        self._tokens = float(max_calls)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a single token, blocking if none are available.

        Refills tokens proportionally based on elapsed time since
        the last refill. If no tokens are available after refill,
        sleeps until at least one token is replenished.
        """
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Calculate wait time for next token
                wait_seconds = (1.0 - self._tokens) * (
                    self._period / self._max_calls
                )
            await asyncio.sleep(wait_seconds)

    def _refill(self) -> None:
        """Add tokens based on elapsed time (called under lock)."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * (self._max_calls / self._period)
        self._tokens = min(self._max_calls, self._tokens + new_tokens)
        self._last_refill = now
