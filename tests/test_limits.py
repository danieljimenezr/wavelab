"""The limits are what separate a local demo from something exposed to the internet."""

from __future__ import annotations

import asyncio
import os

import pytest

from wavelab.server import limits as limits_module
from wavelab.server.limits import RateLimiter, TooBusy, TooMany

# Set BEFORE anything imports wavelab.server.app (one test below does, to read the limiter the
# service is actually served with): `APP = App()` runs at import time and builds a BarStore rooted
# at WAVELAB_DATA. Same value as tests/test_decide_route.py, whichever gets there first.
os.environ.setdefault("WAVELAB_DATA", "/nonexistent-wavelab-test-store")


class _Clock:
    """Stands in for the `time` module inside limits.py, which reads nothing but `monotonic()`.

    Both of the limiter's windows are durations — 3,600 s and 60 s — and every test in this file
    fires its calls inside the same millisecond, so until this existed not one of them could cross
    either boundary. Patching the real clock instead would move time for asyncio too.
    """

    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(limits_module, "time", c)
    return c


async def _call(lim: RateLimiter, ip: str = "1.2.3.4") -> None:
    async with lim.slot(ip):
        pass


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


async def test_the_queue_wait_is_the_declared_one():
    """The 25 s wait is a product decision, so it is pinned rather than left to the default.

    The test below overrides it to keep CI honest about time; this is what stops that override
    from also silently becoming the production value.
    """
    assert RateLimiter().acquire_timeout_s == 25.0


async def test_rejects_when_saturated():
    """The give-up path, without spending the whole timeout waiting for it.

    This test used to wall-clock the real 25 s wait: it was 25.00 s of a 28 s suite — 88% of every
    run on every push — to observe a `TimeoutError` being turned into a `TooBusy`. The behaviour
    under test is the conversion, not the duration, and the duration is pinned by the test above.
    """
    lim = RateLimiter(max_concurrent=1, per_hour=100, per_minute=100, acquire_timeout_s=0.05)
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
    assert lim.stats["active_ips"] == 5200, (
        f"{lim.stats['active_ips']} of 5,200 IPs survived a sweep in which every one of them had "
        "just called: `<= 5200` was true of any number at all, including the 0 that a sweep "
        "dropping live callers produces. See test_idle_ips_are_dropped_and_live_ones_are_not for "
        "the other half."
    )


async def test_the_hourly_quota_is_released_after_an_hour_and_not_before(clock):
    """The window is a duration, and 3600 sits four lines from a 60 that means something else.

    Read as minutes, the hourly quota — the one that says "30 validations per hour, each 5-10 s of
    CPU" — refills sixty times faster than the sentence the user is shown, and the node hands out
    an hour's worth of compute every minute to anyone running a loop. Nothing raises: the service
    simply stops being limited, and the first symptom is the bill or the neighbouring application.
    """
    lim = RateLimiter(max_concurrent=4, per_hour=3, per_minute=100)
    for _ in range(3):
        await _call(lim)

    clock.t = 3599.0
    with pytest.raises(TooMany, match="limit"):
        await _call(lim)

    clock.t = 3601.0
    await _call(lim)          # an hour after the first call, the first slot is back


async def test_the_burst_quota_is_released_after_a_minute_and_not_after_an_hour(clock):
    """The per-minute cap is the one that stops a burst; widened to an hour it stops being one.

    It would then be a second, redundant copy of the hourly cap — at a lower number, so the hourly
    one never fires and the message the user gets for the rest of the hour is "wait a few seconds".
    """
    lim = RateLimiter(max_concurrent=4, per_hour=100, per_minute=3)
    for _ in range(3):
        await _call(lim)

    clock.t = 59.0
    with pytest.raises(TooMany, match="per minute"):
        await _call(lim)

    clock.t = 61.0
    await _call(lim)


async def test_one_expired_call_ages_out_by_itself_and_takes_nothing_with_it(clock):
    """Eviction inspects the OLDEST entry and must remove that one — not whatever is on the end.

    Removing the newest instead does not trim an expired prefix, it empties the deque: the loop
    keeps re-reading the same old q[0] until nothing is left. A caller who has spent their whole
    hourly quota then gets all of it back the instant their FIRST call turns an hour old, rather
    than one slot at a time, which is roughly double the sustained throughput the limiter claims —
    and it is invisible, because the limiter goes on limiting, just not by as much as it says.
    """
    lim = RateLimiter(max_concurrent=4, per_hour=3, per_minute=100)
    for t in (0.0, 3000.0, 3100.0):
        clock.t = t
        await _call(lim)

    clock.t = 3601.0
    await _call(lim)          # the t=0 call has expired: exactly one slot has come free, and one

    with pytest.raises(TooMany, match="limit"):
        # The calls at t=3000 and t=3100 are still inside the hour, so with the one just made the
        # quota of three is spent again. Getting a fourth here means the sweep emptied the deque
        # rather than trimming it, and handed back two slots that had not expired.
        await _call(lim)


async def test_a_validation_that_raises_still_gives_its_slot_back():
    """Every route holding a slot catches TooBusy and TooMany and lets everything else through.

    A malformed CSV that gets past the parser, or a NumPy error inside the battery, therefore
    leaves the `async with` through an exception. If the release is conditional on a clean exit,
    that burns one of the two production slots for the lifetime of the process. After two the node
    answers "there are too many validations under way right now" to everybody, forever, while
    doing nothing at all — and /api/status reports free: 0 with no way to tell it from real load.
    """
    lim = RateLimiter(max_concurrent=1, per_hour=100, per_minute=100, acquire_timeout_s=0.05)

    class Blew(RuntimeError):
        """Whatever the battery raises that is neither TooBusy nor TooMany."""

    with pytest.raises(Blew):
        async with lim.slot("a"):
            raise Blew("nan in the equity curve")

    assert lim.stats["free"] == 1, (
        f"{lim.stats['free']} of 1 slot free after a validation raised: the slot was not returned"
    )
    async with lim.slot("b"):          # must not have to wait out acquire_timeout_s
        pass


async def test_the_served_limiter_carries_the_production_numbers():
    """Every other test here builds its own limiter, so none of them observes the shipped one.

    These four numbers are the whole exposure decision — at 5-10 s of CPU each, `per_minute` is
    what a single loop can take from the node, and the node shares a machine with applications that
    bring in money. A one-character edit to any of them changes what the service gives away and
    would otherwise be seen by nobody. Change them deliberately; this test is the record that it
    was deliberate.
    """
    from wavelab.server.app import LIMITS

    assert (LIMITS.max_concurrent, LIMITS.per_hour, LIMITS.per_minute) == (2, 30, 6)
    assert LIMITS.acquire_timeout_s == 25.0


async def test_idle_ips_are_dropped_and_live_ones_are_not(clock):
    """The sweep the dict's unbounded growth depends on, seen from both sides.

    `test_prunes_idle_ips` runs entirely inside one instant, so nothing there is ever idle and the
    whole pruning block can be deleted with the suite still green. The leak it guards against is
    measured in weeks of uptime, which is exactly the timescale no test reaches without a clock.

    Both directions are asserted because each alone is satisfied by a constant: never pruning
    passes the second assertion, pruning everything passes the first.
    """
    lim = RateLimiter(max_concurrent=9, per_hour=1000, per_minute=1000)
    for i in range(5100):
        await _call(lim, f"ip{i}")
    assert lim.stats["active_ips"] == 5100

    clock.t = 8000.0                   # 2h 13m later: the 5,100 are idle, this caller is not
    await _call(lim, "fresh")

    assert lim.stats["active_ips"] == 1, (
        f"{lim.stats['active_ips']} IPs still held after a sweep in which 5,100 of them had been "
        "silent for over two hours: the dict only grows, and that is the leak"
    )
    assert "fresh" in lim._hist, "the caller that triggered the sweep was swept away by it"
