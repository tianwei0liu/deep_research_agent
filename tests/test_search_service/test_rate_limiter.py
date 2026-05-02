"""Tests for search_service.rate_limiter — AsyncRateLimiter."""

import asyncio
import time

import pytest

from search_service.rate_limiter import AsyncRateLimiter


class TestAsyncRateLimiter:
    """Simple token-bucket rate limiter."""

    @pytest.mark.asyncio
    async def test_allows_within_limit(self) -> None:
        """Calls within the limit complete immediately."""
        limiter = AsyncRateLimiter(max_calls=5, period=60.0)
        for _ in range(5):
            await limiter.acquire()
        # All 5 should succeed without blocking

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self) -> None:
        """Exceeding the limit forces a wait (we test with short period)."""
        limiter = AsyncRateLimiter(max_calls=2, period=0.5)

        # Exhaust the 2 tokens
        await limiter.acquire()
        await limiter.acquire()

        # Third call should take at least ~0.4s (waiting for token refill)
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.15, f"Expected wait >= 0.15s, got {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_refills_over_time(self) -> None:
        """Tokens refill after the period elapses."""
        limiter = AsyncRateLimiter(max_calls=1, period=0.3)
        await limiter.acquire()  # Use the single token
        await asyncio.sleep(0.35)  # Wait for refill
        # Should complete without meaningful delay
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.2
