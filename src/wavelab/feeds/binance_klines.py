"""Live 1m bars, with gap healing on every reconnect.

The only series stored is 1m; everything else is resampled locally. If each timeframe were requested
separately, their boundaries and their gaps would not line up and the cross-timeframe agreement logic
would be measuring alignment noise instead of market structure.

Gap healing runs on EVERY reconnect, not only after the 24 h cut Binance makes by design. A service
restart, a deploy or a network outage leave exactly the same hole, and an unhealed hole raises no
error at all: it simply makes a window of N bars span more time than it claims to.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable

from wavelab.core.timeframes import TF_1M, Timeframe
from wavelab.core.types import Bar
from wavelab.feeds.binance_rest import BinanceREST
from wavelab.feeds.binance_ws import SPOT_WS, stream_json

__all__ = ["KlineFeed"]


def _to_bar(k: dict, symbol: str, tf: Timeframe) -> Bar:
    return Bar(
        symbol=symbol, tf=tf, open_time_ms=int(k["t"]),
        open=float(k["o"]), high=float(k["h"]), low=float(k["l"]), close=float(k["c"]),
        volume=float(k["v"]), quote_volume=float(k["q"]), trades=int(k["n"]),
        taker_buy_base=float(k["V"]), taker_buy_quote=float(k["Q"]),
        is_closed=bool(k["x"]),
        n_source_bars=tf.expected_source_bars if k["x"] else 0,
    )


class KlineFeed:
    """Emits 1m bars: the closed ones and, in between, the one still forming."""

    def __init__(self, symbol: str = "BTCUSDT") -> None:
        self.symbol = symbol.upper()
        self.connected = False
        self.reconnects = 0
        self.healed = 0
        self.last_closed_ms: int | None = None
        self.last_msg_ms = 0

    @property
    def silent_seconds(self) -> float:
        """An OPEN connection that delivers nothing is a real, observed failure mode (the Binance
        futures WebSocket does exactly that). So it is the silence that gets watched."""
        return (time.time() * 1000 - self.last_msg_ms) / 1000 if self.last_msg_ms else float("inf")

    async def heal(self, start_ms: int, end_ms: int) -> list[Bar]:
        """Fill the gap between the last known bar and now over REST."""
        if end_ms <= start_ms:
            return []
        async with BinanceREST() as c:
            bars = await c.heal_gap(self.symbol, TF_1M, start_ms, end_ms)
        self.healed += len(bars)
        return bars

    async def stream(
        self,
        start_ms: int | None = None,
        on_heal: Callable[[list[Bar]], None] | None = None,
    ) -> AsyncIterator[Bar]:
        """Live bars. Before the first one, and after every reconnect, the gap is healed."""
        self.last_closed_ms = start_ms
        heal_pending = asyncio.Event()
        heal_pending.set()

        def _up() -> None:
            self.connected = True
            heal_pending.set()

        def _down(reason: str) -> None:
            self.connected = False
            self.reconnects += 1
            print(f"[klines] down ({reason}); will heal the gap on the way back", flush=True)

        stream = f"{self.symbol.lower()}@kline_1m"
        async for msg in stream_json(SPOT_WS, [stream], on_connect=_up, on_disconnect=_down):
            self.last_msg_ms = msg["_ts_ingest_ms"]

            if heal_pending.is_set() and self.last_closed_ms is not None:
                heal_pending.clear()
                # `- TF_1M.ms` to overlap by one bar: the store deduplicates, and overlapping is
                # the only way to guarantee the boundary bar is not lost.
                bars = await self.heal(self.last_closed_ms - TF_1M.ms,
                                       int(msg["_ts_ingest_ms"]))
                fresh = [b for b in bars
                         if self.last_closed_ms is None or b.open_time_ms > self.last_closed_ms]
                if fresh:
                    print(f"[klines] gap healed: {len(fresh)} bars", flush=True)
                    if on_heal:
                        on_heal(fresh)
                    for b in fresh:
                        self.last_closed_ms = b.open_time_ms
                        yield b
            elif heal_pending.is_set():
                heal_pending.clear()

            bar = _to_bar(msg["k"], self.symbol, TF_1M)
            if bar.is_closed:
                if self.last_closed_ms is not None and bar.open_time_ms <= self.last_closed_ms:
                    continue          # we already had it from the healing
                self.last_closed_ms = bar.open_time_ms
            yield bar
