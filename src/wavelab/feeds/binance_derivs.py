"""Grabadores en sombra: liquidaciones y métricas de derivados.

**Nada consume esto todavía, y aun así es lo primero que se arranca.** El motivo no es técnico:

- ``@forceOrder`` (liquidaciones) **no tiene ninguna fuente histórica gratuita** y no se puede
  rellenar hacia atrás. Cada hora sin grabar es una hora perdida para siempre.
- ``/futures/data/*`` (open interest, ratios long/short, agresión de takers) retiene **solo ~30
  días** en la API, y para fechas anteriores devuelve el error -1130, no una lista vacía.

CAVEAT QUE HAY QUE ESCRIBIR AHORA, MIENTRAS SE ENTIENDE EL PORQUÉ: ``@forceOrder`` está
**muestreado**. Binance limita la frecuencia de emisión por símbolo, así que el fichero es una
MUESTRA de las liquidaciones, sesgada precisamente contra las cascadas agrupadas — que son el
motivo por el que uno querría este dato. Grabarlo sigue siendo correcto porque no hay alternativa,
pero cualquier feature de v2 que lo trate como registro completo subestimará la magnitud de las
cascadas justo en la cola donde importa. El proxy histórico real para cascadas es una caída brusca
de sumOpenInterest en el archivo de métricas coincidiendo con un desequilibrio grande de takers.
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

__all__ = ["LiquidationRecorder", "DerivativesPoller", "SAMPLING_CAVEAT"]

SAMPLING_CAVEAT = (
    "@forceOrder está MUESTREADO por Binance (limita la frecuencia de emisión por símbolo). "
    "Este fichero es una muestra, sesgada contra las cascadas agrupadas. No lo trates como un "
    "registro completo de liquidaciones."
)


class LiquidationRecorder:
    """Vuelca ``@forceOrder`` a JSONL, un fichero por día UTC. Append-only, nunca se borra."""

    def __init__(self, root: Path | str, symbols: list[str]) -> None:
        self.root = Path(root)
        self.symbols = [s.lower() for s in symbols]
        self.root.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self.reconnects = 0
        self._fh = None
        self._day: str | None = None
        # El caveat vive junto a los datos, no solo en el código: dentro de un año, quien lea
        # estos ficheros no va a abrir este módulo.
        (self.root / "LEEME.txt").write_text(SAMPLING_CAVEAT + "\n", encoding="utf-8")

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
            print(f"[liq] conectado a {len(streams)} stream(s)", flush=True)
        def _down(motivo: str) -> None:
            self.reconnects += 1
            print(f"[liq] desconectado ({motivo}); reconectando", flush=True)

        async for msg in stream_json(FAPI_WS, streams, on_connect=_up, on_disconnect=_down):
            fh = self._file_for(msg["_ts_ingest_ms"])
            fh.write(json.dumps(msg, separators=(",", ":")) + "\n")
            self.n += 1
            if self.n % 20 == 0:
                fh.flush()   # durabilidad frente a coste: 20 líneas es un compromiso razonable


class DerivativesPoller:
    """Sondea fapi con presupuesto de peso PROPIO y estrecho.

    Deliberadamente separado del feed de velas: si el sondeo de derivados se desmadra, no puede
    consumir el presupuesto del que depende el gráfico en vivo.
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
                        print(f"[deriv] {self.n} sondeos, margen de peso {c.gov.headroom:.0%}",
                              flush=True)
                except RateLimitCircuitOpen as e:
                    print(f"[deriv] circuito abierto: {e}", flush=True)
                    await asyncio.sleep(3600)
                except Exception as e:  # noqa: BLE001
                    print(f"[deriv] error {type(e).__name__}: {e}", flush=True)
                    await asyncio.sleep(60)
                await asyncio.sleep(self.every_s)
