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
import time

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


async def test_the_warm_up_asks_the_store_for_history_up_to_NOW_in_milliseconds(
    clean_app, monkeypatch, capsys
):
    """A seconds/milliseconds slip here empties the history and still reports a healthy start.

    `until` bounds the read: every bar in the store is stamped in epoch milliseconds, so an `until`
    built in seconds is ~1.7e9 — three weeks after 1970 — and `iter_months(symbol, 0, until)` finds
    nothing. The engine then warms with zero bars, `_start` sets `ready` anyway, and /api/status
    reports `{ok: true, ready: true}`: a service that looks perfectly healthy and answers "not
    enough data" on every route. That is worse than the crash this file was written for, because
    there `ready` is False and a probe can act on it.

    The fake store honours the range it is given, which is what makes this a test of the bound
    rather than of the literal expression that computes it: get the unit wrong and no bars come
    back, and the assertion that fails is the one about the bars.
    """
    import pandas as pd

    ts = [1_700_000_040_000 + i * 60_000 for i in range(3)]   # on the 1m UTC grid
    asked: list[tuple[int, int]] = []

    class FakeStore:
        def months(self, symbol):
            return ["2023-11"]

        def iter_months(self, symbol, lo, hi):
            asked.append((lo, hi))
            rows = [t for t in ts if lo <= t <= hi]
            if not rows:
                return
            yield "2023-11", pd.DataFrame(
                {"open": [1.0] * len(rows), "high": [2.0] * len(rows),
                 "low": [0.5] * len(rows), "close": [1.5] * len(rows),
                 "volume": [10.0] * len(rows)},
                index=pd.Index(rows, name="open_time_ms"),
            )

    warmed: list[int] = []

    class FakeEngine:
        def warmup(self, bars):
            warmed.extend(b.open_time_ms for b in bars)
            return len(warmed)

        def waves(self, tf):
            return {"n_confirmed": 0}

    monkeypatch.setattr(clean_app, "store", FakeStore())
    monkeypatch.setattr(clean_app, "engine", FakeEngine())

    n = clean_app.warmup()
    capsys.readouterr()

    assert warmed == ts, (
        f"the engine was warmed with {len(warmed)} of the store's 3 bars. An `until` that does not "
        "reach today's bars warms up on an empty history and reports success"
    )
    assert n == 3

    (lo, hi) = asked[-1]
    assert lo == 0, "warm-up reads from the beginning of the history, not from a rolling window"
    assert hi > ts[-1], (
        f"the store was asked for bars up to {hi}, which is before the newest bar it holds "
        f"({ts[-1]}): every bar after that upper bound is invisible to the warm-up"
    )
    assert abs(hi - time.time() * 1000) < 5_000, (
        f"the upper bound is {hi}; `now` in epoch milliseconds is {int(time.time() * 1000)}. A "
        "bound in seconds is an instant in January 1970 and selects no bar at all"
    )


class TestTheStaticCachePolicy:
    """A deploy must not be able to pair the new JavaScript with the old dictionary.

    StaticFiles sends an ETag and, by default, no `Cache-Control` — which leaves the browser free
    to reuse a file without asking. With one file that is harmless. With several it is not: a
    release that changes `assay.js` and `i18n.js` together can land a reader on the new script and
    the cached dictionary, and the page then renders `undefined · candles.body_flow_20` with
    nothing in the console. That happened here, to the person building the feature, on a machine
    where both files were already correct on disk.

    `no-cache` does not mean "do not cache" — it means "revalidate before use" — and the ETag is
    already there, so the cost is a conditional request answered 304 with no body.
    """

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        return TestClient(srv.app)

    @pytest.mark.parametrize("path", ["/i18n.js", "/assay.js", "/app.js", "/assay.html"])
    def test_the_app_s_own_files_are_revalidated_every_time(self, client, path):
        r = client.get(path)
        assert r.status_code == 200, f"{path} is not being served at all"
        assert r.headers.get("cache-control") == "no-cache", (
            f"{path} came back with cache-control={r.headers.get('cache-control')!r}. Anything "
            "that lets a browser skip revalidation can pair it with a stale sibling."
        )

    def test_vendor_is_cached_hard_because_its_url_carries_its_version(self, client):
        """The exception, and it has to stay one: the vendored chart bundle is version-pinned in
        its own filename, so the bytes behind a given URL never change. Revalidating 200 KB on
        every page load to be told it has not changed is the cost this exists to avoid."""
        r = client.get("/vendor/lightweight-charts.standalone.production.mjs")
        assert r.status_code == 200
        assert r.headers.get("cache-control") == "public, max-age=31536000, immutable"
