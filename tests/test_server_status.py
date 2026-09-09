"""/api/status has to tell "starting" apart from "broken".

The endpoint's whole reason to exist is that distinction, and until the fix it could not make it:
`_start` is spawned with create_task and only gathered at shutdown, so a warm-up that raised parked
its exception in the task object, `ready` stayed False forever and the payload was byte for byte
the one a healthy slow start produces. The deploy then waited out its full 6-minute deadline and
rolled back with nothing in the log.

So the case exercised here is the CRASH, not the happy path. A test that only proves warm-up can
succeed would have passed against the broken code.
"""

from __future__ import annotations

import asyncio
import os

import pytest

# Must be set BEFORE the module is imported: `APP = App()` runs at import time and builds a
# BarStore rooted at WAVELAB_DATA. Pointing it away from the real store keeps a test run from ever
# touching data/ — two processes against the same directory have already corrupted a Parquet month.
os.environ.setdefault("WAVELAB_DATA", "/nonexistent-wavelab-test-store")

from wavelab.server import app as srv


@pytest.fixture
def clean_app():
    """APP is a module-level singleton, so state leaks between tests unless it is put back.

    `_tasks` is restored too, and it is not decoration: the happy-path test lets `_start` fill it
    with two fakes, and the crash test asserts it is EMPTY. Without this line the two only pass in
    file order — reverse them, or run one with `-k`, and the crash test fails on leftovers from a
    test that is not even in the selection. A fixture that restores three of the four fields it
    knows about is how an ordering dependency gets built in and then blamed on pytest.
    """
    before = (srv.APP.ready, srv.APP.startup_error,
              srv.APP._warm_n, srv.APP._warm_month, list(srv.APP._tasks))
    yield srv.APP
    (srv.APP.ready, srv.APP.startup_error,
     srv.APP._warm_n, srv.APP._warm_month, srv.APP._tasks) = before


async def test_status_reports_a_crashed_warmup(clean_app, monkeypatch, capsys):
    """warmup() raises -> the endpoint must SAY so, not keep claiming it is warming up."""
    def boom() -> int:
        raise RuntimeError("corrupt Parquet month 2019-04")

    monkeypatch.setattr(clean_app, "warmup", boom)
    clean_app.ready = False
    clean_app.startup_error = None
    clean_app._warm_n = 7
    clean_app._warm_month = "2019-04"

    # _start must not propagate: the process staying alive and answering IS the fix. A refused
    # connection is what we are trying not to look like.
    await srv._start(clean_app)

    resp = await srv.status()
    body = resp.body.decode()
    import json
    d = json.loads(body)

    assert d["ok"] is True, "the process is alive and serving; that is what `ok` means"
    assert d["ready"] is False
    assert d["failed"] is True, "without this a probe cannot tell 'broken' from 'starting'"
    assert "corrupt Parquet month 2019-04" in d["error"]
    assert d["error"].startswith("RuntimeError")
    # The bug in one assertion: a dead start must not go on reporting progress it is not making.
    assert d["warming_up"] is None

    # An operator reads journalctl, not a Python task object. The failure has to be in stdout.
    out = capsys.readouterr().out
    assert "WARM-UP FAILED" in out
    assert "corrupt Parquet month 2019-04" in out
    assert "in boom" in out, "the traceback, not just the message: we need the offending line"

    # No feed/health tasks may have been started against an engine that never warmed up.
    assert clean_app._tasks == []


async def test_status_while_warming_up_is_not_marked_failed(clean_app):
    """The state the crash case must stay distinguishable FROM."""
    clean_app.ready = False
    clean_app.startup_error = None
    clean_app._warm_n = 42
    clean_app._warm_month = "2021-11"

    import json
    d = json.loads((await srv.status()).body.decode())

    assert d["ok"] is True
    assert d["ready"] is False
    assert d["failed"] is False
    assert d["error"] is None
    assert d["warming_up"] == {"months": 42, "month": "2021-11"}


async def test_a_successful_warmup_clears_nothing_and_starts_the_feeds(clean_app, monkeypatch):
    """The happy path still has to set `ready` and leave `failed` false."""
    monkeypatch.setattr(clean_app, "warmup", lambda: 0)
    started: list[str] = []

    def fake_create_task(coro, name=None):
        coro.close()          # never actually run the feed in a test
        started.append(name)
        return object()

    monkeypatch.setattr(srv.asyncio, "create_task", fake_create_task)
    clean_app.ready = False
    clean_app.startup_error = None

    await srv._start(clean_app)

    import json
    d = json.loads((await srv.status()).body.decode())
    assert d["ready"] is True
    assert d["failed"] is False
    assert d["error"] is None
    assert d["warming_up"] is None
    assert started == ["feed-supervisor", "health"]


async def test_cancellation_is_not_swallowed_as_a_startup_error(clean_app, monkeypatch):
    """Shutdown cancels this task. A cancel is not a crash and must not be reported as one.

    CancelledError is a BaseException in 3.13, so `except Exception` lets it through — this pins
    that, because widening the catch to BaseException would make every clean shutdown log a
    fabricated warm-up failure.
    """
    def hang() -> int:
        raise asyncio.CancelledError

    monkeypatch.setattr(clean_app, "warmup", hang)
    clean_app.ready = False
    clean_app.startup_error = None

    with pytest.raises(asyncio.CancelledError):
        await srv._start(clean_app)

    assert clean_app.startup_error is None
    assert clean_app.ready is False
