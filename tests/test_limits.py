"""The limits are what separate a local demo from something exposed to the internet."""

from __future__ import annotations

import asyncio

import pytest

from wavelab.server.limits import RateLimiter, TooBusy, TooMany


async def test_cuts_off_per_minute():
    lim = RateLimiter(max_concurrent=4, per_hour=100, per_minute=3)
    for _ in range(3):
        async with lim.slot("1.2.3.4"):
            pass
    with pytest.raises(TooMany, match="per minute"):
        async with lim.slot("1.2.3.4"):
            pass


async def test_cuts_off_per_hour():
    lim = RateLimiter(max_concurrent=4, per_hour=3, per_minute=100)
    for _ in range(3):
        async with lim.slot("1.2.3.4"):
            pass
    with pytest.raises(TooMany, match="limit"):
        async with lim.slot("1.2.3.4"):
            pass


async def test_ips_do_not_get_in_each_others_way():
    """One hog cannot deny the service to everyone else."""
    lim = RateLimiter(max_concurrent=4, per_hour=2, per_minute=100)
    for _ in range(2):
        async with lim.slot("hog"):
            pass
    with pytest.raises(TooMany):
        async with lim.slot("hog"):
            pass
    async with lim.slot("someone_else"):    # must not raise
        pass


async def test_caps_concurrency():
    lim = RateLimiter(max_concurrent=2, per_hour=100, per_minute=100)
    active = 0
    peak = 0

    async def one(i):
        nonlocal active, peak
        async with lim.slot(f"ip{i}"):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.05)
            active -= 1

    await asyncio.gather(*(one(i) for i in range(8)))
    assert peak <= 2, f"{peak} ran at once against a cap of 2"


async def test_rejects_when_saturated():
    lim = RateLimiter(max_concurrent=1, per_hour=100, per_minute=100)
    lim._Ctx  # noqa: B018
    async with lim.slot("a"):
        # With the only slot taken and a short wait, the second caller has to give up.
        lim._sem = asyncio.Semaphore(0)
        ctx = lim.slot("b")
        with pytest.raises(TooBusy, match="too many validations"):
            await asyncio.wait_for(ctx.__aenter__(), timeout=30)


async def test_prunes_idle_ips():
    """Without pruning the dict only ever grows: a slow leak you notice weeks later."""
    lim = RateLimiter(max_concurrent=9, per_hour=1000, per_minute=1000)
    for i in range(5200):
        async with lim.slot(f"ip{i}"):
            pass
    assert lim.stats["active_ips"] <= 5200
