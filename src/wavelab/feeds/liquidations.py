"""Grabador de liquidaciones, con OKX como fuente primaria.

POR QUÉ NO BINANCE, que era el plan original. Verificado el 2026-09-08 desde DOS máquinas
independientes (residencial de Telefónica en Sitges y datacenter de Clouding en Barcelona):

    wss://fstream.binance.com  handshake 101 OK, SUBSCRIBE respondido con {"result":null,"id":1},
                               y CERO frames de datos en 90 s — incluidos markPrice@1s y aggTrade,
                               que empujan varias veces por segundo.
    wss://data-stream.binance.vision (spot)   44-64 frames en 10 s. Funciona.
    https://fapi.binance.com (REST de futuros) funciona: funding, open interest, ratios.

Es decir: Binance acepta la conexión y la suscripción al stream de futuros, y luego no envía nada.
El fallo silencioso perfecto — nada lanza, nada avisa, y un grabador ingenuo escribiría un fichero
vacío durante meses convencido de estar funcionando. Si algún día se reabre, este módulo lo detecta
por el contador de mensajes, no por el estado de la conexión.

ALTERNATIVA ELEGIDA: OKX `liquidation-orders` sobre instType=SWAP (todos los swaps), verificado
entregando datos reales. Bybit `allLiquidation` queda como secundaria.

CAVEAT, escrito ahora que se entiende: cualquier feed de liquidaciones de un solo exchange es una
muestra parcial del mercado. Ni OKX ni Bybit ven las liquidaciones de Binance, que es el mayor
mercado de perpetuos. Sirve como señal de estrés y de cascada, no como censo. Y las cascadas
agrupadas están infrarrepresentadas en cualquier feed con limitación de frecuencia.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import websockets

__all__ = ["LiquidationRecorder", "SOURCES", "CAVEAT"]

CAVEAT = (
    "Liquidaciones de OKX (primaria) y Bybit (secundaria). NO incluye Binance: su WebSocket de "
    "futuros acepta la suscripción y no envía datos desde España (verificado 2026-09-08 desde IP "
    "residencial y de datacenter). Cualquier feed de un solo exchange es una muestra parcial del "
    "mercado, no un censo: úsalo como señal de estrés, no como magnitud absoluta."
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
    """Un fichero JSONL por día y fuente. Append-only: esto NUNCA se borra ni se reescribe."""

    def __init__(self, root: Path | str, source: str = "okx") -> None:
        if source not in SOURCES:
            raise ValueError(f"fuente desconocida {source!r}; opciones: {list(SOURCES)}")
        self.root = Path(root)
        self.source = source
        self.cfg = SOURCES[source]
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "LEEME.txt").write_text(CAVEAT + "\n", encoding="utf-8")
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
        """Segundos sin un solo mensaje. Es la métrica que importa: una conexión ABIERTA que no
        entrega nada es exactamente el fallo de Binance, y el estado del socket no lo delata."""
        return (time.time() * 1000 - self.last_msg_ms) / 1000 if self.last_msg_ms else float("inf")

    async def run(self) -> None:
        intento = 0
        while True:
            try:
                async with websockets.connect(self.cfg["url"], ping_interval=20,
                                              ping_timeout=20, close_timeout=5) as ws:
                    await ws.send(json.dumps(self.cfg["subscribe"]))
                    intento = 0
                    print(f"[liq:{self.source}] conectado", flush=True)
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
                        fh.flush()      # las liquidaciones son escasas: durabilidad > rendimiento
                        self.n += 1
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self.reconnects += 1
                print(f"[liq:{self.source}] caído ({type(e).__name__}); reconectando", flush=True)
            intento += 1
            await asyncio.sleep(min(60.0, 1.5 ** min(intento, 10)))
