"""Liquidation recorder, with OKX as the primary source.

WHY NOT BINANCE, which was the original plan. Verified on 2026-09-08 from TWO independent machines
(a Telefónica residential line in Sitges and a Clouding datacenter in Barcelona):

    wss://fstream.binance.com  handshake 101 OK, SUBSCRIBE answered with {"result":null,"id":1},
                               and ZERO data frames in 90 s — including markPrice@1s and aggTrade,
                               which push several times per second.
    wss://data-stream.binance.vision (spot)   44-64 frames in 10 s. Works.
    https://fapi.binance.com (futures REST) works: funding, open interest, ratios.

That is: Binance accepts the connection and accepts the subscription to the futures stream, and then
sends nothing. The perfect silent failure — nothing raises, nothing warns, and a naive recorder would
write an empty file for months convinced it was working. If it ever reopens, this module notices it
through the message counter, not through the state of the connection.

CHOSEN ALTERNATIVE: OKX `liquidation-orders` over instType=SWAP (every swap), verified delivering
real data. Bybit `allLiquidation` stays as the backup.

CAVEAT, written now that the reason is understood: any single-exchange liquidation feed is a partial
sample of the market. Neither OKX nor Bybit see Binance's liquidations, and Binance is the largest
perpetuals venue. It serves as a stress and cascade signal, not as a census. And clustered cascades
are under-represented in any rate-limited feed.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import websockets

__all__ = ["CAVEAT", "SOURCES", "LiquidationRecorder"]

CAVEAT = (
    "Liquidations from OKX (primary) and Bybit (secondary). Binance is NOT included: its futures "
    "WebSocket accepts the subscription and sends no data from Spain (verified 2026-09-08 from a "
    "residential and a datacenter IP). Any single-exchange feed is a partial sample of the market, "
    "not a census: use it as a stress signal, not as an absolute magnitude."
)

SOURCES = {
    "okx": {
        "url": "wss://ws.okx.com:8443/ws/v5/public",
        "subscribe": {"op": "subscribe",
                      "args": [{"channel": "liquidation-orders", "instType": "SWAP"}]},
        "is_data": lambda d: bool(d.get("data")) and d.get("arg", {}).get("channel") == "liquidation-orders",
    },
    "bybit": {
        "url": "wss://stream.bybit.com/v5/public/linear",
        "subscribe": {"op": "subscribe", "args": ["allLiquidation.BTCUSDT", "allLiquidation.ETHUSDT"]},
        "is_data": lambda d: str(d.get("topic", "")).startswith("allLiquidation"),
    },
}


class LiquidationRecorder:
    """One JSONL file per day and source. Append-only: this is NEVER deleted or rewritten."""

    def __init__(self, root: Path | str, source: str = "okx") -> None:
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}; options: {list(SOURCES)}")
        self.root = Path(root)
        self.source = source
        self.cfg = SOURCES[source]
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "README.txt").write_text(CAVEAT + "\n", encoding="utf-8")
        self.n = 0
        self.reconnects = 0
        self.last_msg_ms = 0
        self._fh = None
        self._day: str | None = None

    def _file(self, ts_ms: int):
        day = datetime.fromtimestamp(ts_ms / 1000, UTC).strftime("%Y-%m-%d")
        if day != self._day:
            if self._fh:
                self._fh.close()
            self._day = day
            self._fh = (self.root / f"{self.source}-{day}.jsonl").open("a", encoding="utf-8")
        return self._fh

    @property
    def silent_seconds(self) -> float:
        """Seconds without a single message. This is the metric that matters: an OPEN connection
        that delivers nothing is exactly the Binance failure, and the socket state does not show
        it."""
        return (time.time() * 1000 - self.last_msg_ms) / 1000 if self.last_msg_ms else float("inf")

    async def run(self) -> None:
        attempt = 0
        while True:
            try:
                async with websockets.connect(self.cfg["url"], ping_interval=20,
                                              ping_timeout=20, close_timeout=5) as ws:
                    await ws.send(json.dumps(self.cfg["subscribe"]))
                    attempt = 0
                    print(f"[liq:{self.source}] connected", flush=True)
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=3600)
                        d = json.loads(raw)
                        if not self.cfg["is_data"](d):
                            continue
                        ts = int(time.time() * 1000)
                        d["_ts_ingest_ms"] = ts
                        d["_source"] = self.source
                        self.last_msg_ms = ts
                        fh = self._file(ts)
                        fh.write(json.dumps(d, separators=(",", ":")) + "\n")
                        fh.flush()      # liquidations are sparse: durability > throughput
                        self.n += 1
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self.reconnects += 1
                print(f"[liq:{self.source}] down ({type(e).__name__}); reconnecting", flush=True)
            attempt += 1
            await asyncio.sleep(min(60.0, 1.5 ** min(attempt, 10)))
