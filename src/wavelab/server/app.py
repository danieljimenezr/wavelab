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

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from wavelab.config import load_config
from wavelab.core.timeframes import BY_NAME, TF_1M
from wavelab.core.types import Bar
from wavelab.engine.live import LiveEngine, Mode
from wavelab.server.limits import RateLimiter, TooBusy, TooMany
from wavelab.feeds.binance_klines import KlineFeed
from wavelab.store.bars import BarStore

def _ip(request) -> str:
    """IP real del cliente. Detrás de Caddy, la de la conexión es siempre 127.0.0.1."""
    for h in ("x-forwarded-for", "x-real-ip"):
        v = request.headers.get(h)
        if v:
            return v.split(",")[0].strip()
    return request.client.host if request.client else "?"


def _limite(e: Exception) -> JSONResponse:
    return JSONResponse({"error": str(e)}, status_code=429)


def pd_fecha(ms: int) -> str:
    import pandas as pd
    return str(pd.Timestamp(int(ms), unit="ms").date())


WEB = Path(__file__).resolve().parents[3] / "web"

#: En modo PÚBLICO solo se sirve el validador. El gráfico con las señales de Elliott es la
#: herramienta privada del dueño y no tiene por qué estar en internet: menos superficie y menos
#: dudas sobre si esto es o no una recomendación de inversión.
PUBLICO = os.environ.get("WAVELAB_PUBLIC", "").lower() in ("1", "true", "si", "sí")
LIMITES = RateLimiter(max_concurrent=2, per_hour=30, per_minute=6)
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
        self._feed_task = None

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

    async def supervise_feed(self) -> None:
        """Vigila la tarea del feed y la resucita.

        `Restart=always` de systemd solo actúa si muere el PROCESO. Una tarea asyncio que muere
        dentro de un proceso vivo no lo dispara: el servidor HTTP sigue sirviendo tan ricamente
        datos rancios, y la insignia de salud lo pinta en rojo mientras nadie hace nada.
        Observado en vivo: 50 minutos "conectado pero MUDO" con el stream de Binance funcionando
        perfectamente.

        Dos condiciones de resurrección: la tarea termina (con o sin excepción), o lleva
        `SILENCIO_MAX` segundos conectada sin entregar un solo mensaje — que es un modo de fallo
        REAL y observado, no hipotético.
        """
        SILENCIO_MAX = 300
        intento = 0
        while True:
            tarea = asyncio.create_task(self.run_feed(), name="feed")
            self._feed_task = tarea
            while not tarea.done():
                await asyncio.sleep(10)
                if self.feed.silent_seconds > SILENCIO_MAX:
                    print(f"[server] feed MUDO {self.feed.silent_seconds:.0f}s pese a estar "
                          "conectado: reiniciando la tarea", flush=True)
                    tarea.cancel()
                    break
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await tarea
            if tarea.cancelled() or tarea.exception() is not None:
                motivo = "cancelada" if tarea.cancelled() else f"{tarea.exception()!r}"
            else:
                motivo = "terminó sola"
            intento += 1
            espera = min(60, 2 ** min(intento, 6))
            print(f"[server] la tarea del feed {motivo}; reintento {intento} en {espera}s",
                  flush=True)
            self.feed.connected = False
            self.feed.last_msg_ms = 0
            await asyncio.sleep(espera)

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
    tareas = [asyncio.create_task(APP.supervise_feed(), name="feed-supervisor"),
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


def _bateria_propia(ts_ms, close, sig, nombre: str, extra: dict | None = None):
    """Batería sobre la serie de precios DEL USUARIO.

    Es lo que hace la herramienta utilizable por alguien que no opera BTC en Binance: sus filas
    SON las velas, así que no hay que alinear nada. El tamaño de barra se deduce de sus propias
    marcas de tiempo.
    """
    import numpy as np
    from wavelab.validation.battery import run_battery

    ts = np.asarray(ts_ms, dtype=np.int64)
    bar_ms = int(np.median(np.diff(ts))) if ts.size > 1 else 86_400_000
    r = run_battery(np.asarray(close, dtype=float), ts, np.asarray(sig, dtype=float),
                    nombre=nombre, bar_ms=max(bar_ms, 1), horizon_bars=1, n_random=250)
    paso = max(1, len(r.equity) // 600)
    return {
        "nombre": r.nombre, "tf": f"{bar_ms // 60000} min entre filas",
        "veredicto": r.veredicto, "resumen": r.resumen,
        "n_signals": r.n_signals, "n_effective": r.n_effective, "exposure": r.exposure,
        "cagr": r.cagr, "sharpe": r.sharpe, "max_dd": r.max_dd,
        "cagr_bh": r.cagr_bh, "sharpe_bh": r.sharpe_bh, "max_dd_bh": r.max_dd_bh,
        "curva": [{"t": int(ts[1:][i]) // 1000, "e": float(r.equity[i]),
                   "b": float(r.equity_bh[i])} for i in range(0, len(r.equity), paso)],
        "tests": [{"id": t.id, "titulo": t.titulo, "estado": t.estado, "valor": t.valor,
                   "referencia": t.referencia, "unidad": t.unidad,
                   "explicacion": t.explicacion, "detalle": t.detalle} for t in r.tests],
        **(extra or {}),
    }


def _serie_y_bateria(tf: str, sig, nombre: str, extra: dict | None = None):
    """Camino común a las tres vías de entrada (catálogo, CSV y regla escrita).

    Que las tres pasen por la MISMA batería no es economía de código: es que un usuario tiene que
    poder comparar su estrategia con las del catálogo sabiendo que se han medido igual.
    """
    import numpy as np
    from wavelab.validation.battery import run_battery

    anillo = APP.engine.state.rings[tf]
    w = anillo.window(len(anillo))
    r = run_battery(w.close, w.ts, np.asarray(sig, dtype=float),
                    nombre=nombre, bar_ms=BY_NAME[tf].ms, horizon_bars=1, n_random=250)
    paso = max(1, len(r.equity) // 600)
    return {
        "nombre": r.nombre, "tf": tf,
        "veredicto": r.veredicto, "resumen": r.resumen,
        "n_signals": r.n_signals, "n_effective": r.n_effective, "exposure": r.exposure,
        "cagr": r.cagr, "sharpe": r.sharpe, "max_dd": r.max_dd,
        "cagr_bh": r.cagr_bh, "sharpe_bh": r.sharpe_bh, "max_dd_bh": r.max_dd_bh,
        "curva": [{"t": int(w.ts[1:][i]) // 1000, "e": float(r.equity[i]),
                   "b": float(r.equity_bh[i])} for i in range(0, len(r.equity), paso)],
        "tests": [{"id": t.id, "titulo": t.titulo, "estado": t.estado, "valor": t.valor,
                   "referencia": t.referencia, "unidad": t.unidad,
                   "explicacion": t.explicacion, "detalle": t.detalle} for t in r.tests],
        **(extra or {}),
    }


@app.post("/api/validar_csv")
async def validar_csv(request: Request, tf: str = "1d") -> JSONResponse:
    """Valida la estrategia del usuario a partir de su propio CSV de señales."""
    from wavelab.validation.csv_import import ImportError_, align_to_bars, parse_signals_csv

    try:
        async with LIMITES.slot(_ip(request)):
            return await _validar_csv(request, tf)
    except (TooBusy, TooMany) as e:
        return _limite(e)


async def _validar_csv(request: Request, tf: str) -> JSONResponse:
    from wavelab.validation.csv_import import ImportError_, align_to_bars, parse_signals_csv

    cuerpo = await request.body()
    if not cuerpo:
        return JSONResponse({"error": "fichero vacío"}, status_code=400)
    if len(cuerpo) > 12_000_000:
        return JSONResponse({"error": "el fichero supera los 12 MB"}, status_code=400)
    anillo = APP.engine.state.rings.get(tf)
    if anillo is None or len(anillo) < 400:
        return JSONResponse({"error": f"sin datos suficientes en {tf}"}, status_code=400)
    try:
        imp = parse_signals_csv(cuerpo)
    except ImportError_ as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    # Si el usuario trae SUS precios, se usan los suyos: puede estar operando ETH, acciones o
    # divisas, y validarle su estrategia contra el precio de BTC daría un informe sin sentido.
    if imp.tiene_precios:
        if len(imp.ts_ms) < 120:
            return JSONResponse({"error":
                f"con precios propios hacen falta al menos 120 filas y hay {len(imp.ts_ms)}. "
                "Con menos, ninguna de las cinco pruebas puede concluir nada."}, status_code=400)
        informe = imp.informe + [
            f"{imp.n_largo} largos, {imp.n_corto} cortos, {imp.n_fuera} fuera",
            f"validado sobre TU serie de {len(imp.ts_ms):,} filas, entre "
            f"{pd_fecha(imp.ts_ms.min())} y {pd_fecha(imp.ts_ms.max())}",
        ]
        return JSONResponse(_bateria_propia(imp.ts_ms, imp.precio, imp.signal,
                                            "tu estrategia (CSV con precios)",
                                            {"informe": informe}))

    w = anillo.window(len(anillo))
    sig = align_to_bars(imp, w.ts, BY_NAME[tf].ms)
    cubiertas = int((sig != 0).sum())
    if cubiertas < 30:
        return JSONResponse({"error":
            f"tras alinear tu CSV con nuestras velas de {tf} solo quedan {cubiertas} barras con "
            "posición. Comprueba que las fechas caen dentro de 2017-2026 y que el timeframe "
            "elegido es el tuyo."}, status_code=400)

    dentro = (w.ts >= imp.ts_ms.min()) & (w.ts <= imp.ts_ms.max())
    informe = imp.informe + [
        "tu CSV no trae columna de precio: se valida contra NUESTRA serie de BTCUSDT. "
        "Si operas otro activo, añade una columna `precio`.",
        f"{imp.n_largo} largos, {imp.n_corto} cortos, {imp.n_fuera} fuera en tu fichero",
        f"alineado a {int(dentro.sum()):,} velas de {tf} entre "
        f"{pd_fecha(imp.ts_ms.min())} y {pd_fecha(imp.ts_ms.max())}",
    ]
    return JSONResponse(_serie_y_bateria(tf, sig, "tu estrategia (CSV)", {"informe": informe}))


@app.post("/api/validar_regla")
async def validar_regla(request: Request) -> JSONResponse:
    """Valida una regla escrita por el usuario en el editor."""
    from wavelab.validation.expr import ExprError, build_series, evaluate_rule

    try:
        async with LIMITES.slot(_ip(request)):
            return await _validar_regla(request)
    except (TooBusy, TooMany) as e:
        return _limite(e)


async def _validar_regla(request: Request) -> JSONResponse:
    from wavelab.validation.expr import ExprError, build_series, evaluate_rule

    body = await request.json()
    tf = body.get("tf", "1d")
    anillo = APP.engine.state.rings.get(tf)
    if anillo is None or len(anillo) < 400:
        return JSONResponse({"error": f"sin datos suficientes en {tf}"}, status_code=400)
    w = anillo.window(len(anillo))
    ser = build_series(w.open, w.high, w.low, w.close, w.volume)
    try:
        r = evaluate_rule(ser, body.get("largo", ""), body.get("corto", ""))
    except ExprError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if r.n_largo + r.n_corto < 30:
        return JSONResponse({"error":
            f"la regla solo se cumple en {r.n_largo + r.n_corto} barras de {len(w.close):,}. "
            "Con tan pocas no se puede concluir nada."}, status_code=400)
    return JSONResponse(_serie_y_bateria(
        tf, r.signal, "tu regla",
        {"informe": [f"{r.n_largo} barras largas, {r.n_corto} cortas de {len(w.close):,}"]}))


@app.get("/api/ayuda_regla")
async def ayuda_regla() -> JSONResponse:
    from wavelab.validation.expr import FUNC_DOCS, SERIE_DOCS
    return JSONResponse({"series": SERIE_DOCS, "funciones": FUNC_DOCS})


@app.get("/api/validar")
async def validar(request: Request, hyp: str, tf: str = "1d") -> JSONResponse:
    """Somete una estrategia a la batería de cinco pruebas. ESTE es el producto."""
    try:
        async with LIMITES.slot(_ip(request)):
            return await _validar_catalogo(hyp, tf)
    except (TooBusy, TooMany) as e:
        return _limite(e)


async def _validar_catalogo(hyp: str, tf: str) -> JSONResponse:
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
    if PUBLICO:
        return JSONResponse({"error": "no disponible"}, status_code=404)
    anillo = APP.engine.state.rings.get(tf)
    if anillo is None or not len(anillo):
        return JSONResponse({"error": f"sin datos para {tf}"}, status_code=400)
    precio = float(anillo.window(1).close[0])
    return JSONResponse(APP.engine.decide(tf, precio))


@app.get("/api/history")
async def history(tf: str = "15m", limit: int = 1500) -> JSONResponse:
    if PUBLICO:
        return JSONResponse({"error": "no disponible"}, status_code=404)
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
    if PUBLICO:
        await socket.close(code=1008)
        return
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


@app.get("/api/estado")
async def estado() -> JSONResponse:
    """Salud del servicio, para vigilarlo desde fuera."""
    return JSONResponse({"ok": True, "publico": PUBLICO, "limites": LIMITES.stats,
                         "modo": APP.engine.state.health.mode.value})


if WEB.exists():
    if PUBLICO:
        # En público, el validador ES la portada. Nadie tiene que saberse una URL.
        @app.get("/")
        async def portada():
            from fastapi.responses import FileResponse
            return FileResponse(WEB / "validador.html")

    app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")


def main() -> None:
    import uvicorn
    uvicorn.run("wavelab.server.app:app", host="127.0.0.1",
                port=int(os.environ.get("WAVELAB_PORT", 8000)), log_level="warning")


if __name__ == "__main__":
    main()
