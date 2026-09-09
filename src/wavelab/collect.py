"""Entry point for the shadow recorders.

    python -m wavelab.collect

Starts BEFORE the rest of the system, and deliberately so: it consumes nothing of what it records,
but what it records is the only thing in the project that cannot be recovered later at any price.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path

from wavelab.feeds.binance_derivs import DerivativesPoller
from wavelab.feeds.liquidations import LiquidationRecorder

DATA = Path(os.environ.get("WAVELAB_DATA", "data"))
SYMBOLS = os.environ.get("WAVELAB_SYMBOLS", "BTCUSDT").split(",")


async def main() -> int:
    raw = DATA / "raw"
    # Two liquidation sources on purpose: neither one sees the whole market, and if one goes
    # silent — as Binance did — the other keeps recording while we find out.
    liqs = [LiquidationRecorder(raw / "liquidations", s) for s in ("okx", "bybit")]
    deriv = DerivativesPoller(raw / "derivatives", SYMBOLS[0])

    print(f"[collect] data in {DATA.resolve()} | symbols {SYMBOLS}", flush=True)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with __import__("contextlib").suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    tasks = [asyncio.create_task(l.run(), name=f"liq-{l.source}") for l in liqs]
    tasks.append(asyncio.create_task(deriv.run(), name="derivatives"))

    async def heartbeat() -> None:
        while not stop.is_set():
            await asyncio.sleep(600)
            parts = " ".join(f"{l.source}={l.n}(silent {l.silent_seconds/60:.0f}m)" for l in liqs)
            print(f"[collect] liquidations: {parts} | polls={deriv.n}", flush=True)
            # An OPEN connection that delivers nothing is the Binance failure mode. We watch the
            # silence, not the state of the socket.
            for l in liqs:
                if l.silent_seconds > 6 * 3600:
                    print(f"[collect] WARNING: {l.source} has gone {l.silent_seconds/3600:.1f} h "
                          "without a single message despite being connected", flush=True)

    tasks.append(asyncio.create_task(heartbeat(), name="heartbeat"))
    await stop.wait()
    print("[collect] stopping…", flush=True)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    print(f"[collect] total: {sum(l.n for l in liqs)} liquidations, "
          f"{deriv.n} polls", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
