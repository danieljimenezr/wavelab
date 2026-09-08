"""Velas de 1m en vivo, con curado de huecos en cada reconexión.

La única serie que se almacena es 1m; todo lo demás se resamplea en local. Si cada timeframe
se pidiese por separado, sus fronteras y sus huecos no coincidirían y la lógica de acuerdo entre
timeframes mediría ruido de alineación en vez de estructura de mercado.

El curado de huecos se ejecuta en CADA reconexión, no solo tras el corte de 24 h que Binance hace
por diseño. Un reinicio del servicio, un despliegue o un corte de red dejan exactamente el mismo
agujero, y un agujero sin curar no lanza ningún error: simplemente hace que una ventana de N velas
abarque más tiempo del que dice.
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
    """Emite velas de 1m: cerradas y, entre medias, la que está en curso."""

    def __init__(self, symbol: str = "BTCUSDT") -> None:
        self.symbol = symbol.upper()
        self.connected = False
        self.reconnects = 0
        self.healed = 0
        self.last_closed_ms: int | None = None
        self.last_msg_ms = 0

    @property
    def silent_seconds(self) -> float:
        """Una conexión ABIERTA que no entrega nada es un modo de fallo real y observado
        (el WebSocket de futuros de Binance hace exactamente eso). Se vigila el silencio."""
        return (time.time() * 1000 - self.last_msg_ms) / 1000 if self.last_msg_ms else float("inf")

    async def heal(self, desde_ms: int, hasta_ms: int) -> list[Bar]:
        """Rellena por REST el hueco entre la última vela conocida y ahora."""
        if hasta_ms <= desde_ms:
            return []
        async with BinanceREST() as c:
            velas = await c.heal_gap(self.symbol, TF_1M, desde_ms, hasta_ms)
        self.healed += len(velas)
        return velas

    async def stream(
        self,
        desde_ms: int | None = None,
        on_heal: Callable[[list[Bar]], None] | None = None,
    ) -> AsyncIterator[Bar]:
        """Velas en vivo. Antes de la primera y tras cada reconexión, cura el hueco."""
        self.last_closed_ms = desde_ms
        pendiente_curar = asyncio.Event()
        pendiente_curar.set()

        def _up() -> None:
            self.connected = True
            pendiente_curar.set()

        def _down(motivo: str) -> None:
            self.connected = False
            self.reconnects += 1
            print(f"[klines] caído ({motivo}); curará el hueco al volver", flush=True)

        stream = f"{self.symbol.lower()}@kline_1m"
        async for msg in stream_json(SPOT_WS, [stream], on_connect=_up, on_disconnect=_down):
            self.last_msg_ms = msg["_ts_ingest_ms"]

            if pendiente_curar.is_set() and self.last_closed_ms is not None:
                pendiente_curar.clear()
                # `- TF_1M.ms` para solapar una vela: el almacén deduplica, y solapar es la
                # única forma de garantizar que no se pierde la vela de la frontera.
                velas = await self.heal(self.last_closed_ms - TF_1M.ms,
                                        int(msg["_ts_ingest_ms"]))
                nuevas = [b for b in velas
                          if self.last_closed_ms is None or b.open_time_ms > self.last_closed_ms]
                if nuevas:
                    print(f"[klines] hueco curado: {len(nuevas)} velas", flush=True)
                    if on_heal:
                        on_heal(nuevas)
                    for b in nuevas:
                        self.last_closed_ms = b.open_time_ms
                        yield b
            elif pendiente_curar.is_set():
                pendiente_curar.clear()

            bar = _to_bar(msg["k"], self.symbol, TF_1M)
            if bar.is_closed:
                if self.last_closed_ms is not None and bar.open_time_ms <= self.last_closed_ms:
                    continue          # ya la teníamos por el curado
                self.last_closed_ms = bar.open_time_ms
            yield bar
