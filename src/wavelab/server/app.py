"""Server: FastAPI + WebSocket, listening ONLY on 127.0.0.1.

Zero new exposed ports, zero new TLS certificates, zero new authentication surface on a machine
that pays the bills. You reach it through an ssh tunnel:

    ssh -L 8000:localhost:8000 root@<node>

One process, one asyncio loop, one port, one browser tab.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from wavelab.config import load_config
from wavelab.core.timeframes import BY_NAME, TF_1M
from wavelab.core.types import Bar
from wavelab.engine.live import LiveEngine
from wavelab.feeds.binance_klines import KlineFeed
from wavelab.server.limits import RateLimiter, TooBusy, TooMany
from wavelab.store.bars import BarStore


def _ip(request) -> str:
    """The client's real IP. Behind Caddy, the connection's own address is always 127.0.0.1."""
    for h in ("x-forwarded-for", "x-real-ip"):
        v = request.headers.get(h)
        if v:
            return v.split(",")[0].strip()
    return request.client.host if request.client else "?"


def _rate_limited(e: Exception) -> JSONResponse:
    return JSONResponse({"error": str(e)}, status_code=429)


def _date_str(ms: int) -> str:
    import pandas as pd
    return str(pd.Timestamp(int(ms), unit="ms").date())


WEB = Path(__file__).resolve().parents[3] / "web"

#: In PUBLIC mode only Assay is served. The chart with the Elliott signals is the owner's private
#: tool and has no business being on the internet: less surface area, and fewer questions about
#: whether this is or is not investment advice.
#: The accepted truthy values keep their Spanish spellings on purpose — the one place in this
#: codebase that was not translated — and gain the English ones alongside them. This flag is read
#: from the environment of a unit file already running on the VPS, and a value not on the list does
#: not fail loudly: it just reads as false, which takes the gate off /api/decide, /api/history and
#: /ws and puts the owner's private chart on the open internet. Accepting both vocabularies costs
#: nothing; a `WAVELAB_PUBLIC=yes` that silently meant "no" would cost the whole point of the flag.
PUBLIC = os.environ.get("WAVELAB_PUBLIC", "").lower() in ("1", "true", "yes", "on", "si", "sí")
LIMITS = RateLimiter(max_concurrent=2, per_hour=30, per_minute=6)
DATA = Path(os.environ.get("WAVELAB_DATA", "data"))


def bar_json(b: Bar) -> dict:
    """Lightweight Charts expects the time in unix SECONDS, not milliseconds."""
    return {"time": b.open_time_ms // 1000, "open": b.open, "high": b.high,
            "low": b.low, "close": b.close, "volume": b.volume,
            "n_source_bars": b.n_source_bars, "is_gap": b.is_gap}


class Hub:
    """Fans out to the connected browsers. Never blocks the producer: one slow browser must not be
    able to stall the consumer of Binance's WebSocket."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def send(self, payload: dict) -> None:
        if not self.clients:
            return
        raw = json.dumps(payload, separators=(",", ":"))
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(raw)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


class App:
    def __init__(self) -> None:
        self.cfg = load_config(Path(__file__).resolve().parents[3] / "config")
        self.symbol = next(iter(self.cfg.assets), "BTCUSDT")
        asset = self.cfg.assets.get(self.symbol)
        tfs = asset.timeframes if asset else ["15m", "1h", "4h", "1d"]
        self.tfs = ["1m"] + [t for t in tfs if t != "1m"]
        self.store = BarStore(DATA / "bars")
        from wavelab.waves.pivots import ZigZagConfig
        zz = ZigZagConfig(k_atr=self.cfg.engine.zigzag_k_atr,
                          min_pct=self.cfg.engine.zigzag_min_pct,
                          atr_period=self.cfg.engine.atr_period,
                          on_close=(asset.r3_on == "close") if asset else False)
        self.engine = LiveEngine(self.symbol, self.tfs, self.cfg.engine.ring_capacity,
                                 self.cfg.trigger_tf, zigzag=zz)
        self.feed = KlineFeed(self.symbol)
        self.hub = Hub()
        self.started_ms = int(time.time() * 1000)
        self._pending: list[Bar] = []
        self._warm_n = 0
        self._warm_month: str | None = None
        self._feed_task = None
        self._tasks: list = []
        #: False while warming up. A probe has to be able to tell "starting" from "broken".
        self.ready = False
        #: Set when warmup() raised, and the ONLY thing that makes "broken" distinguishable from
        #: "starting": both leave `ready` False forever, so `ready` alone cannot tell them apart.
        #: `_start` runs in a task nobody awaits until shutdown, so without this the exception is
        #: parked in the task object and never printed — see the comment in `_start`.
        self.startup_error: str | None = None

    # ------------------------------------------------------------------ startup

    def _iter_bars(self, until_ms: int):
        """Yields bars, reading the store MONTH BY MONTH.

        Two distinct memory leaks, and both of them kill the service against its 768 MB cap:
        materialising 4.76M Bar objects (~950 MB), and loading the whole history into a single
        DataFrame (~340 MB plus the concatenation of 110 Parquet files). A generator solves the
        first one; the second is only solved by paginating the READ. That way the peak is one
        month: ~44,000 rows.
        """
        # Progress signal: on the VPS, with CPUQuota=50%, warming up nine years takes ~2 minutes.
        # A SILENT two-minute startup is indistinguishable from a hung one, and the first thing
        # anybody does about that is restart the service — so it never finishes starting.
        for _key, df in self.store.iter_months(self.symbol, 0, until_ms):
            self._warm_month = _key
            if self._warm_n % 20 == 0:
                print(f"[server] warming up… {_key} ({self._warm_n + 1} months)", flush=True)
            self._warm_n += 1
            for ts, r in zip(df.index, df.itertuples(index=False), strict=True):
                yield Bar(symbol=self.symbol, tf=TF_1M, open_time_ms=int(ts),
                          open=float(r.open), high=float(r.high), low=float(r.low),
                          close=float(r.close), volume=float(r.volume),
                          is_closed=True, n_source_bars=1)

    def warmup(self) -> int:
        """Warms up from the BEGINNING of the history, not from a rolling window.

        This is a correctness decision, not a completeness one. The pivot detector is path
        dependent: starting from a different point produces different pivots. With a rolling
        window, every restart of the service would change the structure shown to the user with no
        invalidation event whatsoever, and the live state would stop matching the one a full replay
        produces — which is the central promise of the design.

        Measured cost: ~570,000 bars/s, some 10-15 s for nine years. Paid once, at startup.
        """
        until = int(time.time() * 1000)
        months = self.store.months(self.symbol)
        if not months:
            print("[server] empty store: run `python -m wavelab.store.hydrate`", flush=True)
            return 0
        t0 = time.perf_counter()
        n = self.engine.warmup(self._iter_bars(until))
        dt = time.perf_counter() - t0
        print(f"[server] warmed up with {n:,} 1m bars since {months[0]} "
              f"in {dt:.1f}s ({n/dt:,.0f}/s, {len(months)} months)", flush=True)
        for tf in self.tfs:
            if tf != "1m":
                print(f"[server]   {tf}: {self.engine.waves(tf)['n_confirmed']:,} pivots",
                      flush=True)
        return n

    # ------------------------------------------------------------------ live loop

    async def run_feed(self) -> None:
        since = self.engine.state.health.last_closed_ms
        # Passed positionally on purpose: the feed owns the name of that parameter, and this call
        # should not break the day it is renamed.
        async for bar in self.feed.stream(since):
            self.engine.check_clock()
            closed = self.engine.on_bar_1m(bar)

            if bar.is_closed:
                self._pending.append(bar)
                if len(self._pending) >= 5:
                    self._flush()
                # A diff, not a snapshot: only what changed.
                await self.hub.send({"type": "bar", "tf": "1m", "bar": bar_json(bar)})
            else:
                await self.hub.send({"type": "tick", "tf": "1m", "bar": bar_json(bar)})

            for htf in closed:
                await self.hub.send({"type": "bar", "tf": htf.tf.name, "bar": bar_json(htf)})
                # Legs are recomputed only when a bar of that timeframe closes: the detector does
                # not advance in between, so resending them on every tick would be noise.
                ring = self.engine.state.rings[htf.tf.name]
                vis = ring.window(min(1500, len(ring)))
                await self.hub.send({"type": "waves", "tf": htf.tf.name,
                                     **self.engine.waves(htf.tf.name,
                                                         since_ms=int(vis.ts[0]))})

    def _flush(self) -> None:
        if not self._pending:
            return
        import pandas as pd
        df = pd.DataFrame(
            [{"open": b.open, "high": b.high, "low": b.low, "close": b.close,
              "volume": b.volume, "quote_volume": b.quote_volume, "trades": b.trades,
              "taker_buy_base": b.taker_buy_base, "taker_buy_quote": b.taker_buy_quote}
             for b in self._pending],
            index=pd.Index([b.open_time_ms for b in self._pending], name="open_time_ms"),
        )
        self.store.ingest(self.symbol, df)   # idempotent: the healing overlap does not duplicate
        self._pending.clear()

    async def supervise_feed(self) -> None:
        """Watches the feed task and resurrects it.

        systemd's `Restart=always` only fires if the PROCESS dies. An asyncio task that dies inside
        a living process does not trigger it: the HTTP server carries on happily serving stale
        data, and the health badge paints it red while nobody does anything about it. Observed
        live: 50 minutes "connected but MUTE" with Binance's stream working perfectly.

        Two resurrection conditions: the task finishes (with or without an exception), or it has
        spent `MAX_SILENCE` seconds connected without delivering a single message — which is a
        REAL, observed failure mode, not a hypothetical one.
        """
        MAX_SILENCE = 300
        attempt = 0
        while True:
            task = asyncio.create_task(self.run_feed(), name="feed")
            self._feed_task = task
            while not task.done():
                await asyncio.sleep(10)
                if self.feed.silent_seconds > MAX_SILENCE:
                    print(f"[server] feed MUTE for {self.feed.silent_seconds:.0f}s despite being "
                          "connected: restarting the task", flush=True)
                    task.cancel()
                    break
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            if task.cancelled() or task.exception() is not None:
                reason = "was cancelled" if task.cancelled() else f"{task.exception()!r}"
            else:
                reason = "finished on its own"
            attempt += 1
            delay = min(60, 2 ** min(attempt, 6))
            print(f"[server] the feed task {reason}; retry {attempt} in {delay}s",
                  flush=True)
            self.feed.connected = False
            self.feed.last_msg_ms = 0
            await asyncio.sleep(delay)

    async def run_health(self) -> None:
        while True:
            await asyncio.sleep(5)
            h = self.engine.update_health(
                self.feed.connected, self.feed.reconnects,
                self.feed.healed, self.feed.silent_seconds)
            await self.hub.send({"type": "health", **h.as_dict()})


APP = App()


async def _start(app_: App) -> None:
    """Warms up in the BACKGROUND and only then turns the feeds on.

    The warm-up used to live in the server's startup hook, and uvicorn does not accept connections
    until startup finishes. With nine years of history that is ~120 s during which the port is
    closed: `curl` says "connection refused", which is indistinguishable from a broken service.

    The continuous-deployment pipeline found this out itself, and in the worst possible way: the
    health check waited 120 s, the warm-up took 117.8, it timed out by two seconds, rolled back a
    deployment that was perfectly fine and declared a critical incident that did not exist.

    Raising the timeout would only have moved the boundary. The right answer is for the server to
    RESPOND from the very first second saying that it is warming up: that way a probe can tell
    "starting" from "broken", which is exactly what a probe is for.
    """
    import anyio
    try:
        await anyio.to_thread.run_sync(app_.warmup)
    except Exception as e:  # noqa: BLE001 — a crashed warm-up must never be silent
        # This task is created with create_task and only gathered at shutdown, so an exception
        # raised here used to sit in the task object with NOTHING in the log: `ready` stayed False
        # forever and /api/status kept answering {"ok":true,"ready":false,"warming_up":{...}} —
        # byte for byte what a healthy slow start looks like. The deploy then burned its whole
        # 6-minute deadline and rolled back with no clue why, and twenty minutes went into chasing
        # a "hung" process that had in fact already died. Record it, log it, keep serving.
        app_.startup_error = f"{type(e).__name__}: {e}"
        # One print, not print() plus traceback.print_exc(): those land on different streams and
        # journald interleaves them, which is how you get a traceback with no idea which line
        # introduced it.
        print(f"[server] WARM-UP FAILED, the service will NOT become ready: {app_.startup_error}\n"
              f"{traceback.format_exc()}", flush=True)
        # Deliberately NOT re-raised. The process staying alive and answering is the whole point:
        # a dead port is indistinguishable from a machine that never booted.
        return
    app_.ready = True
    app_._tasks = [asyncio.create_task(app_.supervise_feed(), name="feed-supervisor"),
                   asyncio.create_task(app_.run_health(), name="health")]


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = [asyncio.create_task(_start(APP), name="startup")]
    try:
        yield
    finally:
        APP._flush()
        for t in [*tasks, *getattr(APP, "_tasks", [])]:
            t.cancel()
        await asyncio.gather(*tasks, *getattr(APP, "_tasks", []), return_exceptions=True)


app = FastAPI(title="wavelab", lifespan=lifespan)


@app.get("/api/hypotheses")
async def list_hypotheses() -> JSONResponse:
    """Catalogue of registered strategies, with their reasoning and their falsification criterion."""
    from wavelab.hypotheses import load_all
    hs = load_all()
    return JSONResponse({"hypotheses": [
        {"name": h.name, "family": h.family, "rationale": h.rationale.strip(),
         "prior": h.prior.strip(), "params": {k: str(v) for k, v in h.params.items()},
         "timeframes": list(h.timeframes)}
        for h in sorted(hs.values(), key=lambda x: (x.family, x.name))]})


def _battery_own_series(ts_ms, close, sig, name: str, extra: dict | None = None):
    """Battery over the USER'S OWN price series.

    This is what makes the tool usable by someone who does not trade BTC on Binance: their rows ARE
    the bars, so there is nothing to align. The bar size is inferred from their own timestamps.
    """
    import numpy as np

    from wavelab.validation.battery import run_battery

    ts = np.asarray(ts_ms, dtype=np.int64)
    bar_ms = int(np.median(np.diff(ts))) if ts.size > 1 else 86_400_000
    r = run_battery(np.asarray(close, dtype=float), ts, np.asarray(sig, dtype=float),
                    name=name, bar_ms=max(bar_ms, 1), horizon_bars=1, n_random=250)
    step = max(1, len(r.equity) // 600)
    return {
        "name": r.name, "tf": f"{bar_ms // 60000} min between rows",
        "verdict": r.verdict, "summary": r.summary,
        "n_signals": r.n_signals, "n_effective": r.n_effective, "exposure": r.exposure,
        "cagr": r.cagr, "sharpe": r.sharpe, "max_dd": r.max_dd,
        "cagr_bh": r.cagr_bh, "sharpe_bh": r.sharpe_bh, "max_dd_bh": r.max_dd_bh,
        "equity_curve": [{"t": int(ts[1:][i]) // 1000, "e": float(r.equity[i]),
                          "b": float(r.equity_bh[i])} for i in range(0, len(r.equity), step)],
        "tests": [{"id": t.id, "title": t.title, "status": t.status, "value": t.value,
                   "reference": t.reference, "unit": t.unit,
                   "explanation": t.explanation, "detail": t.detail} for t in r.tests],
        **(extra or {}),
    }


def _series_and_battery(tf: str, sig, name: str, extra: dict | None = None):
    """The path shared by all three entry routes (catalogue, CSV and hand-written rule).

    Putting all three through the SAME battery is not code economy: it is so that a user can
    compare their own strategy against the catalogue's knowing both were measured the same way.
    """
    import numpy as np

    from wavelab.validation.battery import run_battery

    ring = APP.engine.state.rings[tf]
    w = ring.window(len(ring))
    r = run_battery(w.close, w.ts, np.asarray(sig, dtype=float),
                    name=name, bar_ms=BY_NAME[tf].ms, horizon_bars=1, n_random=250)
    step = max(1, len(r.equity) // 600)
    return {
        "name": r.name, "tf": tf,
        "verdict": r.verdict, "summary": r.summary,
        "n_signals": r.n_signals, "n_effective": r.n_effective, "exposure": r.exposure,
        "cagr": r.cagr, "sharpe": r.sharpe, "max_dd": r.max_dd,
        "cagr_bh": r.cagr_bh, "sharpe_bh": r.sharpe_bh, "max_dd_bh": r.max_dd_bh,
        "equity_curve": [{"t": int(w.ts[1:][i]) // 1000, "e": float(r.equity[i]),
                          "b": float(r.equity_bh[i])} for i in range(0, len(r.equity), step)],
        "tests": [{"id": t.id, "title": t.title, "status": t.status, "value": t.value,
                   "reference": t.reference, "unit": t.unit,
                   "explanation": t.explanation, "detail": t.detail} for t in r.tests],
        **(extra or {}),
    }


@app.post("/api/validate_csv")
async def validate_csv(request: Request, tf: str = "1d") -> JSONResponse:
    """Validates the user's strategy from their own CSV of signals."""

    try:
        async with LIMITS.slot(_ip(request)):
            return await _validate_csv(request, tf)
    except (TooBusy, TooMany) as e:
        return _rate_limited(e)


async def _validate_csv(request: Request, tf: str) -> JSONResponse:
    from wavelab.validation.csv_import import ImportError_, align_to_bars, parse_signals_csv

    body = await request.body()
    if not body:
        return JSONResponse({"error": "empty file"}, status_code=400)
    if len(body) > 12_000_000:
        return JSONResponse({"error": "the file is larger than 12 MB"}, status_code=400)
    ring = APP.engine.state.rings.get(tf)
    if not APP.ready:
        return JSONResponse({"error": "the service is loading nine years of history "
                             f"({APP._warm_n} months). Try again in a minute."},
                            status_code=503)
    if ring is None or len(ring) < 400:
        return JSONResponse({"error": f"not enough data on {tf}"}, status_code=400)
    try:
        imp = parse_signals_csv(body)
    except ImportError_ as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # If the user brings THEIR OWN prices, those are the ones used: they may be trading ETH,
    # equities or FX, and validating their strategy against the price of BTC would produce a
    # report that means nothing.
    if imp.has_prices:
        if len(imp.ts_ms) < 120:
            return JSONResponse({"error":
                f"with your own prices we need at least 120 rows and there are {len(imp.ts_ms)}. "
                "With fewer, none of the five tests can conclude anything."}, status_code=400)
        report = imp.report + [
            f"{imp.n_long} long, {imp.n_short} short, {imp.n_flat} flat",
            (f"validated on YOUR series of {len(imp.ts_ms):,} rows, between "
             f"{_date_str(imp.ts_ms.min())} and {_date_str(imp.ts_ms.max())}"),
        ]
        return JSONResponse(_battery_own_series(imp.ts_ms, imp.price, imp.signal,
                                                "your strategy (CSV with prices)",
                                                {"report": report}))

    w = ring.window(len(ring))
    sig = align_to_bars(imp, w.ts, BY_NAME[tf].ms)
    covered = int((sig != 0).sum())
    if covered < 30:
        return JSONResponse({"error":
            f"after aligning your CSV with our {tf} bars only {covered} bars are left holding a "
            "position. Check that the dates fall inside 2017-2026 and that the timeframe you "
            "picked is the one you meant."}, status_code=400)

    inside = (w.ts >= imp.ts_ms.min()) & (w.ts <= imp.ts_ms.max())
    report = imp.report + [
        ("your CSV has no price column: it is validated against OUR BTCUSDT series. "
         "If you trade a different asset, add a `price` column."),
        f"{imp.n_long} long, {imp.n_short} short, {imp.n_flat} flat in your file",
        (f"aligned to {int(inside.sum()):,} {tf} bars between "
         f"{_date_str(imp.ts_ms.min())} and {_date_str(imp.ts_ms.max())}"),
    ]
    return JSONResponse(_series_and_battery(tf, sig, "your strategy (CSV)", {"report": report}))


@app.post("/api/validate_rule")
async def validate_rule(request: Request) -> JSONResponse:
    """Validates a rule the user wrote in the editor."""

    try:
        async with LIMITS.slot(_ip(request)):
            return await _validate_rule(request)
    except (TooBusy, TooMany) as e:
        return _rate_limited(e)


async def _validate_rule(request: Request) -> JSONResponse:
    from wavelab.validation.expr import ExprError, build_series, evaluate_rule

    body = await request.json()
    tf = body.get("tf", "1d")
    ring = APP.engine.state.rings.get(tf)
    if not APP.ready:
        return JSONResponse({"error": "the service is loading nine years of history "
                             f"({APP._warm_n} months). Try again in a minute."},
                            status_code=503)
    if ring is None or len(ring) < 400:
        return JSONResponse({"error": f"not enough data on {tf}"}, status_code=400)
    w = ring.window(len(ring))
    ser = build_series(w.open, w.high, w.low, w.close, w.volume)
    try:
        r = evaluate_rule(ser, body.get("long", ""), body.get("short", ""))
    except ExprError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if r.n_long + r.n_short < 30:
        return JSONResponse({"error":
            f"the rule only holds on {r.n_long + r.n_short} bars out of {len(w.close):,}. "
            "With that few, nothing can be concluded."}, status_code=400)
    return JSONResponse(_series_and_battery(
        tf, r.signal, "your rule",
        {"report": [f"{r.n_long} long bars, {r.n_short} short out of {len(w.close):,}"]}))


@app.get("/api/rule_help")
async def rule_help() -> JSONResponse:
    from wavelab.validation.expr import FUNC_DOCS, SERIES_DOCS
    return JSONResponse({"series": SERIES_DOCS, "functions": FUNC_DOCS})


@app.get("/api/validate")
async def validate(request: Request, hyp: str, tf: str = "1d") -> JSONResponse:
    """Puts a strategy through the battery of five tests. THIS is the product."""
    try:
        async with LIMITS.slot(_ip(request)):
            return await _validate_catalog(hyp, tf)
    except (TooBusy, TooMany) as e:
        return _rate_limited(e)


async def _validate_catalog(hyp: str, tf: str) -> JSONResponse:

    from wavelab.hypotheses import load_all
    from wavelab.hypotheses.base import Series
    from wavelab.validation.battery import run_battery

    hs = load_all()
    h = hs.get(hyp)
    if h is None:
        return JSONResponse({"error": f"unknown hypothesis: {hyp}"}, status_code=400)
    ring = APP.engine.state.rings.get(tf)
    if not APP.ready:
        return JSONResponse({"error": "the service is loading nine years of history "
                             f"({APP._warm_n} months). Try again in a minute."},
                            status_code=503)
    if ring is None or len(ring) < 400:
        return JSONResponse({"error": f"not enough data on {tf}"}, status_code=400)

    w = ring.window(len(ring))
    series = Series(tf, w.ts, w.open, w.high, w.low, w.close, w.volume)
    sig = h.signals(series).astype(float)
    r = run_battery(w.close, w.ts, sig, name=h.name,
                    bar_ms=BY_NAME[tf].ms, horizon_bars=1, n_random=250)

    # The curve is subsampled so the browser does not receive 20,000 points per series.
    step = max(1, len(r.equity) // 600)
    return JSONResponse({
        "name": r.name, "tf": tf, "family": h.family,
        "rationale": h.rationale.strip(), "prior": h.prior.strip(),
        "verdict": r.verdict, "summary": r.summary,
        "n_signals": r.n_signals, "n_effective": r.n_effective, "exposure": r.exposure,
        "cagr": r.cagr, "sharpe": r.sharpe, "max_dd": r.max_dd,
        "cagr_bh": r.cagr_bh, "sharpe_bh": r.sharpe_bh, "max_dd_bh": r.max_dd_bh,
        "equity_curve": [{"t": int(w.ts[1:][i]) // 1000, "e": float(r.equity[i]),
                          "b": float(r.equity_bh[i])} for i in range(0, len(r.equity), step)],
        "tests": [{"id": t.id, "title": t.title, "status": t.status, "value": t.value,
                   "reference": t.reference, "unit": t.unit,
                   "explanation": t.explanation, "detail": t.detail} for t in r.tests],
    })


@app.get("/api/decide")
async def decide(tf: str = "4h") -> JSONResponse:
    """Ranked hypotheses and their plan. This is the decision card."""
    if PUBLIC:
        return JSONResponse({"error": "not available"}, status_code=404)
    ring = APP.engine.state.rings.get(tf)
    if ring is None or not len(ring):
        return JSONResponse({"error": f"no data for {tf}"}, status_code=400)
    price = float(ring.window(1).close[0])
    return JSONResponse(APP.engine.decide(tf, price))


@app.get("/api/history")
async def history(tf: str = "15m", limit: int = 1500) -> JSONResponse:
    if PUBLIC:
        return JSONResponse({"error": "not available"}, status_code=404)
    if tf not in APP.engine.state.rings:
        return JSONResponse({"error": f"timeframe {tf} is not configured",
                             "available": list(APP.engine.state.rings)}, status_code=400)
    ring = APP.engine.state.rings[tf]
    if not len(ring):
        return JSONResponse({"tf": tf, "bars": [], "health": APP.engine.state.health.as_dict()})
    w = ring.window(min(limit, len(ring)))
    bars = [{"time": int(t) // 1000, "open": float(o), "high": float(h), "low": float(l),
             "close": float(c), "volume": float(v), "is_gap": bool(g)}
            for t, o, h, l, c, v, g in zip(w.ts, w.open, w.high, w.low, w.close, w.volume,
                                           w.is_gap, strict=True)]
    return JSONResponse({
        # The legs are clipped to the SAME window as the bars. Without this, Lightweight Charts
        # stretches the time axis to span every historical pivot and squeezes the candles until
        # they are unreadable: the chart becomes a zigzag line drawn over nothing.
        "tf": tf, "symbol": APP.symbol, "bars": bars,
        "waves": APP.engine.waves(tf, since_ms=int(w.ts[0])),
        "gaps": w.n_gaps, "health": APP.engine.state.health.as_dict(),
        "timeframes": list(APP.engine.state.rings),
        "roles": {t: BY_NAME[t].role.value for t in APP.engine.state.rings},
    })


@app.websocket("/ws")
async def ws(socket: WebSocket) -> None:
    if PUBLIC:
        await socket.close(code=1008)
        return
    await socket.accept()
    APP.hub.clients.add(socket)
    await socket.send_text(json.dumps({"type": "health", **APP.engine.state.health.as_dict()}))
    try:
        while True:
            await socket.receive_text()      # keeps the connection alive
    except WebSocketDisconnect:
        pass
    finally:
        APP.hub.clients.discard(socket)


@app.get("/api/status")
async def status() -> JSONResponse:
    """Service health. Answers FROM THE VERY FIRST SECOND, warming up included.

    `ok` means "the process is alive and serving". `ready` means "it can actually do work now".
    Keeping the two apart is what lets a deployment wait without mistaking a slow start for a
    breakdown.

    `failed`/`error` are the third state, and they are what make that promise true. Warm-up
    crashing leaves `ready` False exactly like warm-up still running does, so the two states used
    to be identical on the wire and a probe had no way to stop waiting. `failed` is the signal to
    stop: it will never become ready.

    `ok` stays True when `failed` is True, on purpose — the process IS alive and serving, and that
    is precisely the fact that distinguishes this from a refused connection. Read `ready` and
    `failed`, never `ok` alone. `warming_up` goes None once it has failed so that a dead start
    cannot keep reporting progress it is not making.
    """
    return JSONResponse({
        "ok": True,
        "ready": APP.ready,
        "failed": APP.startup_error is not None,
        "error": APP.startup_error,
        "public": PUBLIC,
        "warming_up": None if (APP.ready or APP.startup_error) else
                      {"months": APP._warm_n, "month": APP._warm_month},
        "limits": LIMITS.stats,
        "mode": APP.engine.state.health.mode.value,
    })


if WEB.exists():
    if PUBLIC:
        # In public mode Assay IS the front page. Nobody should have to know a URL by heart.
        @app.get("/")
        async def front_page():
            from fastapi.responses import FileResponse
            return FileResponse(WEB / "assay.html")

    app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")


def main() -> None:
    import uvicorn
    uvicorn.run("wavelab.server.app:app", host="127.0.0.1",
                port=int(os.environ.get("WAVELAB_PORT", "8000")), log_level="warning")


if __name__ == "__main__":
    main()
