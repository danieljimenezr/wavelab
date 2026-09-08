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
        self._warm_n = 0
        self._warm_month: str | None = None

    # ------------------------------------------------------------------ arranque

    def _iter_bars(self, hasta_ms: int):
        """Genera velas leyendo el almacén MES A MES.

        Dos fugas de memoria distintas, y las dos matan al servicio contra su tope de 768 MB:
        materializar 4,76 M de objetos Bar (~950 MB) y cargar el histórico entero en un solo
        DataFrame (~340 MB más la concatenación de 110 ficheros Parquet). La primera se resuelve
        con un generador; la segunda solo se resuelve paginando la LECTURA. Así el pico es de un
        mes: ~44.000 filas.
        """
        # Señal de progreso: en el VPS, con CPUQuota=50%, calentar nueve años tarda ~2 minutos.
        # Un arranque MUDO de dos minutos es indistinguible de uno colgado, y lo primero que hace
        # cualquiera ante eso es reiniciar el servicio — con lo que nunca termina de arrancar.
        for _key, df in self.store.iter_months(self.symbol, 0, hasta_ms):
            self._warm_month = _key
            if self._warm_n % 20 == 0:
                print(f"[server] calentando… {_key} ({self._warm_n + 1} meses)", flush=True)
            self._warm_n += 1
            for ts, r in zip(df.index, df.itertuples(index=False), strict=True):
                yield Bar(symbol=self.symbol, tf=TF_1M, open_time_ms=int(ts),
                          open=float(r.open), high=float(r.high), low=float(r.low),
                          close=float(r.close), volume=float(r.volume),
                          is_closed=True, n_source_bars=1)

    def warmup(self) -> int:
        """Calienta desde el PRINCIPIO del histórico, no desde una ventana móvil.

        Es una decisión de correctitud, no de exhaustividad. El detector de pivotes es dependiente
        del camino: arrancar desde un punto distinto produce pivotes distintos. Con una ventana
        móvil, cada reinicio del servicio cambiaría la estructura mostrada al usuario sin ningún
        evento de invalidación, y además el estado en vivo dejaría de coincidir con el que produce
        un replay completo — que es la promesa central del diseño.

        Coste medido: ~570.000 velas/s, unos 10-15 s para nueve años. Se paga una vez al arrancar.
        """
        hasta = int(time.time() * 1000)
        meses = self.store.months(self.symbol)
        if not meses:
            print("[server] almacén vacío: ejecuta `python -m wavelab.store.hydrate`", flush=True)
            return 0
        t0 = time.perf_counter()
        n = self.engine.warmup(self._iter_bars(hasta))
        dt = time.perf_counter() - t0
        print(f"[server] calentado con {n:,} velas de 1m desde {meses[0]} "
              f"en {dt:.1f}s ({n/dt:,.0f}/s, {len(meses)} meses)", flush=True)
        for tf in self.tfs:
            if tf != "1m":
                print(f"[server]   {tf}: {self.engine.waves(tf)['n_confirmed']:,} pivotes",
                      flush=True)
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


@app.get("/api/hipotesis")
async def listar_hipotesis() -> JSONResponse:
    """Catálogo de estrategias registradas, con su razonamiento y su criterio de falsación."""
    from wavelab.hypotheses import load_all
    hs = load_all()
    return JSONResponse({"hipotesis": [
        {"name": h.name, "family": h.family, "rationale": h.rationale.strip(),
         "prior": h.prior.strip(), "params": {k: str(v) for k, v in h.params.items()},
         "timeframes": list(h.timeframes)}
        for h in sorted(hs.values(), key=lambda x: (x.family, x.name))]})


@app.get("/api/validar")
async def validar(hyp: str, tf: str = "1d") -> JSONResponse:
    """Somete una estrategia a la batería de cinco pruebas. ESTE es el producto."""
    import numpy as np
    from wavelab.hypotheses import load_all
    from wavelab.hypotheses.base import Series
    from wavelab.validation.battery import run_battery

    hs = load_all()
    h = hs.get(hyp)
    if h is None:
        return JSONResponse({"error": f"hipótesis desconocida: {hyp}"}, status_code=400)
    anillo = APP.engine.state.rings.get(tf)
    if anillo is None or len(anillo) < 400:
        return JSONResponse({"error": f"sin datos suficientes en {tf}"}, status_code=400)

    w = anillo.window(len(anillo))
    serie = Series(tf, w.ts, w.open, w.high, w.low, w.close, w.volume)
    sig = h.signals(serie).astype(float)
    r = run_battery(w.close, w.ts, sig, nombre=h.name,
                    bar_ms=BY_NAME[tf].ms, horizon_bars=1, n_random=250)

    # Se submuestrea la curva para que el navegador no reciba 20.000 puntos por serie.
    paso = max(1, len(r.equity) // 600)
    return JSONResponse({
        "nombre": r.nombre, "tf": tf, "family": h.family,
        "rationale": h.rationale.strip(), "prior": h.prior.strip(),
        "veredicto": r.veredicto, "resumen": r.resumen,
        "n_signals": r.n_signals, "n_effective": r.n_effective, "exposure": r.exposure,
        "cagr": r.cagr, "sharpe": r.sharpe, "max_dd": r.max_dd,
        "cagr_bh": r.cagr_bh, "sharpe_bh": r.sharpe_bh, "max_dd_bh": r.max_dd_bh,
        "curva": [{"t": int(w.ts[1:][i]) // 1000, "e": float(r.equity[i]),
                   "b": float(r.equity_bh[i])} for i in range(0, len(r.equity), paso)],
        "tests": [{"id": t.id, "titulo": t.titulo, "estado": t.estado, "valor": t.valor,
                   "referencia": t.referencia, "unidad": t.unidad,
                   "explicacion": t.explicacion, "detalle": t.detalle} for t in r.tests],
    })


@app.get("/api/decide")
async def decide(tf: str = "4h") -> JSONResponse:
    """Hipótesis ordenadas y su plan. Es la tarjeta de decisión."""
    anillo = APP.engine.state.rings.get(tf)
    if anillo is None or not len(anillo):
        return JSONResponse({"error": f"sin datos para {tf}"}, status_code=400)
    precio = float(anillo.window(1).close[0])
    return JSONResponse(APP.engine.decide(tf, precio))


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
