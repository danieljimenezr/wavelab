"""The shadow recorders' entry point — the one module systemd runs that nothing tested.

`python -m wavelab.collect` is a unit on the production VPS (`wavelab-collect.service`) and had no
test of any kind. That is the wrong module to leave uncovered: it records liquidations and
derivatives metrics, and what it records **is the only data in this project that cannot be
recovered later at any price**. There is no historical feed to backfill it from. A day this process
is quietly broken is a day gone for good.

The failure mode it is built around deserves the same care: a WebSocket that accepts a subscription
and then sends nothing. The socket is open, the task is alive, `systemctl status` is green, and no
data arrives. That is why it watches SILENCE rather than connection state, and why it records from
two exchanges instead of one. Both of those are asserted here, because both are the kind of thing a
later simplification removes for looking redundant.
"""

from __future__ import annotations

import asyncio
import signal

import pytest

from wavelab import collect


class _FakeRecorder:
    """Stands in for a LiquidationRecorder: runs until cancelled, records nothing."""

    def __init__(self, source: str, *, silent: float = 0.0, n: int = 0) -> None:
        self.source, self.silent_seconds, self.n = source, silent, n
        self.ran = False

    async def run(self) -> None:
        self.ran = True
        await asyncio.Event().wait()          # forever, until cancelled


class _FakePoller(_FakeRecorder):
    pass


@pytest.fixture
def recorders(monkeypatch, tmp_path):
    made: dict[str, list] = {"liq": [], "deriv": []}

    def liq(path, source):
        r = _FakeRecorder(source)
        made["liq"].append((r, path))
        return r

    def deriv(path, symbol):
        r = _FakePoller(symbol)
        made["deriv"].append((r, path, symbol))
        return r

    monkeypatch.setattr(collect, "LiquidationRecorder", liq)
    monkeypatch.setattr(collect, "DerivativesPoller", deriv)
    monkeypatch.setattr(collect, "DATA", tmp_path)
    return made


async def _run_and_stop(after: float = 0.05) -> int:
    """Start main(), let it wire everything up, then ask it to stop the way systemd does."""
    task = asyncio.create_task(collect.main())
    await asyncio.sleep(after)
    # SIGTERM is what `systemctl stop` sends. add_signal_handler is registered on the running loop,
    # so raising the signal exercises the real path rather than a stand-in for it.
    signal.raise_signal(signal.SIGTERM)
    return await asyncio.wait_for(task, timeout=5)


class TestItRecordsFromMoreThanOneExchange:
    async def test_two_liquidation_sources_are_started_not_one(self, recorders):
        await _run_and_stop()
        sources = sorted(r.source for r, _ in recorders["liq"])
        assert sources == ["bybit", "okx"], (
            f"the collector started {sources}. Two sources is not redundancy to be tidied away: "
            "neither exchange sees the whole market, and when Binance went silent the second feed "
            "is what kept recording while anyone noticed"
        )

    async def test_every_recorder_is_actually_run(self, recorders):
        await _run_and_stop()
        started = [r for r, *_ in recorders["liq"]] + [r for r, *_ in recorders["deriv"]]
        assert len(started) == 3 and all(r.ran for r in started), (
            "a recorder was constructed and never run. It would appear in the startup log, hold a "
            "path, report zero, and record nothing — and the only sign would be a file that never "
            "grows"
        )

    async def test_the_recorders_write_under_the_configured_data_directory(self, recorders, tmp_path):
        await _run_and_stop()
        for _, path in recorders["liq"]:
            assert tmp_path in path.parents, (
                f"a recorder was pointed at {path}, outside WAVELAB_DATA. On the VPS everything is "
                "inside an 8 GB loopback image precisely so a runaway writer cannot touch the host"
            )


class TestItStopsWhenAskedTo:
    async def test_sigterm_stops_it_and_it_reports_a_clean_exit(self, recorders):
        """`systemctl stop` sends SIGTERM and then waits. A process that ignores it is killed after
        the timeout, mid-write, which is how a JSONL file acquires half a line."""
        assert await _run_and_stop() == 0

    async def test_no_task_is_left_running_afterwards(self, recorders):
        await _run_and_stop()
        alive = [t for t in asyncio.all_tasks()
                 if t is not asyncio.current_task() and not t.done()
                 and (t.get_name() or "").startswith(("liq-", "derivatives", "heartbeat"))]
        assert not alive, (
            f"{[t.get_name() for t in alive]} survived the shutdown. A recorder that outlives its "
            "own process holds the socket and the file handle, and the next start finds both taken"
        )
