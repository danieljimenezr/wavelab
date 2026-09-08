"""Punto de entrada de los grabadores en sombra.

    python -m wavelab.collect

Arranca ANTES que el resto del sistema y a propósito: no consume nada de lo que graba, pero lo que
graba es lo único del proyecto que no se puede recuperar más tarde a ningún precio.
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
    # Dos fuentes de liquidaciones a propósito: ninguna ve el mercado entero, y si una se
    # queda muda —como hizo Binance— la otra sigue grabando mientras nos enteramos.
    liqs = [LiquidationRecorder(raw / "liquidations", s) for s in ("okx", "bybit")]
    deriv = DerivativesPoller(raw / "derivatives", SYMBOLS[0])

    print(f"[collect] datos en {DATA.resolve()} | símbolos {SYMBOLS}", flush=True)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with __import__("contextlib").suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    tareas = [asyncio.create_task(l.run(), name=f"liq-{l.source}") for l in liqs]
    tareas.append(asyncio.create_task(deriv.run(), name="derivatives"))

    async def latido() -> None:
        while not stop.is_set():
            await asyncio.sleep(600)
            partes = " ".join(f"{l.source}={l.n}(mudo {l.silent_seconds/60:.0f}m)" for l in liqs)
            print(f"[collect] liquidaciones: {partes} | sondeos={deriv.n}", flush=True)
            # Una conexión ABIERTA que no entrega nada es el fallo de Binance. Vigilamos el
            # silencio, no el estado del socket.
            for l in liqs:
                if l.silent_seconds > 6 * 3600:
                    print(f"[collect] AVISO: {l.source} lleva {l.silent_seconds/3600:.1f} h "
                          "sin un solo mensaje pese a estar conectado", flush=True)

    tareas.append(asyncio.create_task(latido(), name="heartbeat"))
    await stop.wait()
    print("[collect] parando…", flush=True)
    for t in tareas:
        t.cancel()
    await asyncio.gather(*tareas, return_exceptions=True)
    print(f"[collect] total: {sum(l.n for l in liqs)} liquidaciones, "
          f"{deriv.n} sondeos", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
