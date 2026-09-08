"""Servidor: FastAPI + WebSocket, escuchando SOLO en 127.0.0.1.

Cero puertos nuevos expuestos, cero certificado TLS nuevo, cero superficie de autenticación nueva
en una máquina que factura. Se accede por túnel ssh:

    ssh -L 8000:localhost:8000 root@<nodo>

Un proceso, un bucle asyncio, un puerto, una pestaña.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from wavelab.config import load_config
from wavelab.core.timeframes import BY_NAME, TF_1M
from wavelab.core.types import Bar
from wavelab.engine.live import LiveEngine, Mode
from wavelab.feeds.binance_klines import KlineFeed
from wavelab.store.bars import BarStore

WEB = Path(__file__).resolve().parents[3] / "web"
DATA = Path(os.environ.get("WAVELAB_DATA", "data"))


def bar_json(b: Bar) -> dict:
    """Lightweight Charts espera el tiempo en SEGUNDOS unix, no en milisegundos."""
    return {"time": b.open_time_ms // 1000, "open": b.open, "high": b.high,
            "low": b.low, "close": b.close, "volume": b.volume,
            "n_source_bars": b.n_source_bars, "is_gap": b.is_gap}


class Hub:
    """Reparte a los navegadores conectados. Nunca bloquea al productor: un navegador lento
    no puede parar el consumidor del WebSocket de Binance."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def send(self, payload: dict) -> None:
        if not self.clients:
            return
        raw = json.dumps(payload, separators=(",", ":"))
        muertos = []
        for ws in list(self.clients):
            try:
                await ws.send_text(raw)
            except Exception:  # noqa: BLE001
                muertos.append(ws)
        for ws in muertos:
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

    # ------------------------------------------------------------------ arranque

    def warmup(self) -> int:
        """Carga desde el almacén. Sin suficiente historia el motor no arranca: prefiere no
        mostrar nada a mostrar indicadores calculados sobre una ventana incompleta."""
        tf_max = max((BY_NAME[t] for t in self.tfs), key=lambda t: t.ms)
        necesarias = max(self.cfg.engine.window_bars, tf_max.expected_source_bars * 300)
        hasta = int(time.time() * 1000)
        desde = hasta - necesarias * TF_1M.ms
        df = self.store.read(self.symbol, desde, hasta, fill_grid=False)
        if df.empty:
            print("[server] almacén vacío: ejecuta `python -m wavelab.store.hydrate`", flush=True)
            return 0
        bars = [
            Bar(symbol=self.symbol, tf=TF_1M, open_time_ms=int(ts),
                open=float(r.open), high=float(r.high), low=float(r.low), close=float(r.close),
                volume=float(r.volume), is_closed=True, n_source_bars=1)
            for ts, r in zip(df.index, df.itertuples(index=False), strict=True)
        ]
        n = self.engine.warmup(bars)
        print(f"[server] calentado con {n:,} velas de 1m "
              f"(desde {df.index[0]} hasta {df.index[-1]})", flush=True)
        return n

    # ------------------------------------------------------------------ bucle vivo

    async def run_feed(self) -> None:
        desde = self.engine.state.health.last_closed_ms
        async for bar in self.feed.stream(desde_ms=desde):
            self.engine.check_clock()
            cerradas = self.engine.on_bar_1m(bar)

            if bar.is_closed:
                self._pending.append(bar)
                if len(self._pending) >= 5:
                    self._flush()
                # Diff, no instantánea: solo lo que cambió.
                await self.hub.send({"type": "bar", "tf": "1m", "bar": bar_json(bar)})
            else:
                await self.hub.send({"type": "tick", "tf": "1m", "bar": bar_json(bar)})

            for htf in cerradas:
                await self.hub.send({"type": "bar", "tf": htf.tf.name, "bar": bar_json(htf)})
                # Los tramos se recalculan solo cuando cierra una vela de ese timeframe: el
                # detector no avanza entre medias, así que reenviar en cada tick sería ruido.
                anillo = self.engine.state.rings[htf.tf.name]
                vis = anillo.window(min(1500, len(anillo)))
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
        self.store.ingest(self.symbol, df)   # idempotente: el solape del curado no duplica
        self._pending.clear()

    async def run_health(self) -> None:
        while True:
            await asyncio.sleep(5)
            h = self.engine.update_health(
                self.feed.connected, self.feed.reconnects,
                self.feed.healed, self.feed.silent_seconds)
            await self.hub.send({"type": "health", **h.as_dict()})


APP = App()


@asynccontextmanager
async def lifespan(app: FastAPI):
    APP.warmup()
    tareas = [asyncio.create_task(APP.run_feed(), name="feed"),
              asyncio.create_task(APP.run_health(), name="health")]
    try:
        yield
    finally:
        APP._flush()
        for t in tareas:
            t.cancel()
        await asyncio.gather(*tareas, return_exceptions=True)


app = FastAPI(title="wavelab", lifespan=lifespan)


@app.get("/api/history")
async def history(tf: str = "15m", limit: int = 1500) -> JSONResponse:
    if tf not in APP.engine.state.rings:
        return JSONResponse({"error": f"timeframe {tf} no configurado",
                             "available": list(APP.engine.state.rings)}, status_code=400)
    anillo = APP.engine.state.rings[tf]
    if not len(anillo):
        return JSONResponse({"tf": tf, "bars": [], "health": APP.engine.state.health.as_dict()})
    w = anillo.window(min(limit, len(anillo)))
    bars = [{"time": int(t) // 1000, "open": float(o), "high": float(h), "low": float(l),
             "close": float(c), "volume": float(v), "is_gap": bool(g)}
            for t, o, h, l, c, v, g in zip(w.ts, w.open, w.high, w.low, w.close, w.volume,
                                           w.is_gap, strict=True)]
    return JSONResponse({
        # Los tramos se acotan a la MISMA ventana que las velas. Sin esto, Lightweight Charts
        # estira el eje temporal para abarcar todos los pivotes históricos y comprime las velas
        # hasta hacerlas ilegibles: el gráfico se vuelve una línea de zigzag sobre nada.
        "tf": tf, "symbol": APP.symbol, "bars": bars,
        "waves": APP.engine.waves(tf, since_ms=int(w.ts[0])),
        "gaps": w.n_gaps, "health": APP.engine.state.health.as_dict(),
        "timeframes": list(APP.engine.state.rings),
        "roles": {t: BY_NAME[t].role.value for t in APP.engine.state.rings},
    })


@app.websocket("/ws")
async def ws(socket: WebSocket) -> None:
    await socket.accept()
    APP.hub.clients.add(socket)
    await socket.send_text(json.dumps({"type": "health", **APP.engine.state.health.as_dict()}))
    try:
        while True:
            await socket.receive_text()      # mantiene viva la conexión
    except WebSocketDisconnect:
        pass
    finally:
        APP.hub.clients.discard(socket)


if WEB.exists():
    app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")


def main() -> None:
    import uvicorn
    uvicorn.run("wavelab.server.app:app", host="127.0.0.1",
                port=int(os.environ.get("WAVELAB_PORT", 8000)), log_level="warning")


if __name__ == "__main__":
    main()
