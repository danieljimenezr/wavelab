"""Shadow recorders: liquidations and derivatives metrics.

**Nothing consumes this yet, and it is still the first thing that gets started.** The reason is not
technical:

- ``@forceOrder`` (liquidations) **has no free historical source** and cannot be backfilled. Every
  hour not recorded is an hour lost forever.
- ``/futures/data/*`` (open interest, long/short ratios, taker aggression) keeps **only ~30 days**
  in the API, and for earlier dates it returns error -1130, not an empty list.

CAVEAT THAT HAS TO BE WRITTEN NOW, WHILE THE WHY IS STILL FRESH: ``@forceOrder`` is **sampled**.
Binance rate-limits how often it emits per symbol, so the file is a SAMPLE of the liquidations,
biased precisely against the clustered cascades -- which are the whole reason anyone would want this
data. Recording it is still the right call because there is no alternative, but any v2 feature that
treats it as a complete record will underestimate the size of the cascades exactly in the tail where
it matters. The real historical proxy for cascades is a sharp drop in sumOpenInterest in the metrics
archive coinciding with a large taker imbalance.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from wavelab.feeds.binance_rest import BinanceREST, RateLimitCircuitOpen
from wavelab.feeds.binance_ws import FAPI_WS, stream_json

__all__ = ["SAMPLING_CAVEAT", "DerivativesPoller", "LiquidationRecorder"]

SAMPLING_CAVEAT = (
    "@forceOrder is SAMPLED by Binance (it rate-limits how often it emits per symbol). "
    "This file is a sample, biased against clustered cascades. Do not treat it as a complete "
    "record of liquidations."
)


class LiquidationRecorder:
    """Dumps ``@forceOrder`` to JSONL, one file per UTC day. Append-only, never deleted."""

    def __init__(self, root: Path | str, symbols: list[str]) -> None:
        self.root = Path(root)
        self.symbols = [s.lower() for s in symbols]
        self.root.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self.reconnects = 0
        self._fh = None
        self._day: str | None = None
        # The caveat lives next to the data, not only in the code: a year from now, whoever reads
        # these files is not going to open this module.
        (self.root / "README.txt").write_text(SAMPLING_CAVEAT + "\n", encoding="utf-8")

    def _file_for(self, ts_ms: int):
        day = datetime.fromtimestamp(ts_ms / 1000, UTC).strftime("%Y-%m-%d")
        if day != self._day:
            if self._fh:
                self._fh.close()
            self._day = day
            self._fh = (self.root / f"liquidations-{day}.jsonl").open("a", encoding="utf-8")
        return self._fh

    async def run(self) -> None:
        streams = [f"{s}@forceOrder" for s in self.symbols]
        def _up() -> None:
            print(f"[liq] connected to {len(streams)} stream(s)", flush=True)
        def _down(reason: str) -> None:
            self.reconnects += 1
            print(f"[liq] disconnected ({reason}); reconnecting", flush=True)

        async for msg in stream_json(FAPI_WS, streams, on_connect=_up, on_disconnect=_down):
            fh = self._file_for(msg["_ts_ingest_ms"])
            fh.write(json.dumps(msg, separators=(",", ":")) + "\n")
            self.n += 1
            if self.n % 20 == 0:
                fh.flush()   # durability against cost: 20 lines is a reasonable compromise


class DerivativesPoller:
    """Polls fapi on its OWN, deliberately narrow weight budget.

    Kept apart from the bar feed on purpose: if the derivatives polling runs away, it must not be
    able to eat the budget the live chart depends on.
    """

    def __init__(self, root: Path | str, symbol: str = "BTCUSDT", every_s: int = 300) -> None:
        self.root = Path(root)
        self.symbol = symbol.upper()
        self.every_s = every_s
        self.root.mkdir(parents=True, exist_ok=True)
        self.n = 0

    def _append(self, kind: str, rows: list[dict]) -> None:
        if not rows:
            return
        day = datetime.now(UTC).strftime("%Y-%m")
        p = self.root / f"{kind}-{self.symbol}-{day}.jsonl.gz"
        with gzip.open(p, "at", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({**r, "_ts_ingest_ms": int(time.time() * 1000)},
                                    separators=(",", ":")) + "\n")

    async def run(self) -> None:
        async with BinanceREST() as c:
            while True:
                try:
                    self._append("funding", await c.funding_rate(self.symbol, limit=10))
                    self._append("open_interest", await c.open_interest_hist(self.symbol, "5m", 12))
                    self._append("long_short", await c.long_short_ratio(self.symbol, "5m", 12))
                    self._append("taker", await c.taker_ratio(self.symbol, "5m", 12))
                    self.n += 1
                    if self.n % 12 == 0:
                        print(f"[deriv] {self.n} polls, weight headroom {c.gov.headroom:.0%}",
                              flush=True)
                except RateLimitCircuitOpen as e:
                    print(f"[deriv] circuit open: {e}", flush=True)
                    await asyncio.sleep(3600)
                except Exception as e:  # noqa: BLE001
                    print(f"[deriv] error {type(e).__name__}: {e}", flush=True)
                    await asyncio.sleep(60)
                await asyncio.sleep(self.every_s)
