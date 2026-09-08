"""Motor en vivo: mantiene los anillos por timeframe desde la única serie de 1m.

Tres modos, y el usuario los ve:

  WARMUP    cargando histórico; no se emite nada
  CATCH_UP  reproduciendo un hueco tras un corte; el estado se actualiza pero la EMISIÓN DE
            DECISIONES ESTÁ SUPRIMIDA
  LIVE      al día

El modo CATCH_UP existe por un motivo concreto. Como ``on_bar`` es deliberadamente idéntico en vivo
y en replay, tras despertar de un corte producirá encantado una decisión ACCIONABLE con una zona de
entrada en un precio que pasó hace cuarenta minutos. Es la forma más probable de que la herramienta
pierda la confianza del usuario en la primera semana, y no se arregla con un aviso: se arregla no
emitiendo.

El tipo ``Decision`` ya lo impide estructuralmente (lanza si se construye ACCIONABLE con
``catching_up=True``), pero eso es la última red. Esta es la primera.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

from wavelab.core.ring import Ring
from wavelab.core.timeframes import BY_NAME, MIN_SOURCE_COVERAGE, TF_1M, Timeframe
from wavelab.core.types import Bar, Direction, MaturityLevel, Verdict
from wavelab.waves.matcher import MatcherConfig, match_impulses
from wavelab.waves.pivots import ZigZag, ZigZagConfig
from wavelab.waves.projection import PlanConfig, build_plan

__all__ = ["EngineState", "Health", "LiveEngine", "Mode"]


class Mode(StrEnum):
    WARMUP = "warmup"
    CATCH_UP = "catch_up"
    LIVE = "live"


@dataclass(slots=True)
class Health:
    """Lo que se pinta en la insignia de salud de datos. Si algo aquí no está bien,
    la interfaz está mintiendo sobre su frescura y el usuario tiene derecho a saberlo."""

    mode: Mode = Mode.WARMUP
    connected: bool = False
    reconnects: int = 0
    healed_bars: int = 0
    silent_seconds: float = 0.0
    lag_bars: float = 0.0
    gaps_in_window: int = 0
    last_closed_ms: int | None = None

    def as_dict(self) -> dict:
        return {
            "mode": self.mode.value, "connected": self.connected,
            "reconnects": self.reconnects, "healed_bars": self.healed_bars,
            "silent_seconds": round(self.silent_seconds, 1),
            "lag_bars": round(self.lag_bars, 2), "gaps_in_window": self.gaps_in_window,
            "last_closed_ms": self.last_closed_ms,
        }


@dataclass(slots=True)
class EngineState:
    """Estado mutable del motor. Se muta SOLO en el hilo del bucle de eventos.

    Cuando llegue el análisis pesado (M4+), se le pasará una instantánea inmutable y devolverá un
    estado nuevo que se aplica aquí. Nunca al revés: `asyncio.wait_for` alrededor de
    `asyncio.to_thread` NO cancela el hilo, así que un análisis abandonado seguiría mutando este
    objeto mientras empieza el siguiente. Sería una carrera de datos dentro de la única función
    que todo el diseño promete determinista.
    """

    symbol: str
    rings: dict[str, Ring] = field(default_factory=dict)
    #: Un detector por timeframe de ANÁLISIS. 1m no lleva: es la serie fuente, no un timeframe
    #: de análisis, y detectar pivotes en 1m sería medir ruido de microestructura.
    detectors: dict[str, ZigZag] = field(default_factory=dict)
    provisional: Bar | None = None
    health: Health = field(default_factory=Health)
    n_bars_1m: int = 0

    def ring(self, tf: Timeframe) -> Ring:
        return self.rings[tf.name]


class LiveEngine:
    """Consume velas de 1m y mantiene los anillos de todos los timeframes."""

    def __init__(self, symbol: str, timeframes: list[str], ring_capacity: int = 8192,
                 trigger_tf: str = "15m", zigzag: ZigZagConfig | None = None) -> None:
        self.symbol = symbol.upper()
        self.tfs = [BY_NAME[t] for t in timeframes]
        self.trigger = BY_NAME[trigger_tf]
        self.zz_cfg = zigzag or ZigZagConfig()
        self.matcher_cfg = MatcherConfig()
        self.plan_cfg = PlanConfig()
        #: La interfaz solo muestra largos, pero el motor evalúa AMBAS direcciones: así se acumula
        #: el doble de evidencia desde el día 1 y activar cortos es una línea de configuración.
        self.directions: list[Direction] = [Direction.LONG, Direction.SHORT]
        self.state = EngineState(symbol=self.symbol)
        self.state.rings[TF_1M.name] = Ring(self.symbol, TF_1M, ring_capacity)
        for tf in self.tfs:
            if tf is not TF_1M:
                self.state.rings[tf.name] = Ring(self.symbol, tf, ring_capacity)
                self.state.detectors[tf.name] = ZigZag(self.zz_cfg)
        self._last_wall = time.time()

    # ------------------------------------------------------------------ resampleo

    def _close_htf(self, tf: Timeframe, htf_open_ms: int) -> Bar | None:
        """Construye la vela de timeframe superior que acaba de cerrar, desde el anillo de 1m.

        Se reconstruye desde la serie fuente en vez de acumular incrementalmente: un acumulador
        que se desincroniza no lo delata nada, mientras que reconstruir siempre da el mismo
        resultado que daría el backtest sobre los mismos datos.
        """
        r1 = self.state.rings[TF_1M.name]
        n = tf.expected_source_bars
        if len(r1) == 0:
            return None
        w = r1.window(min(n + 5, len(r1)))
        sel = (w.ts >= htf_open_ms) & (w.ts < htf_open_ms + tf.ms)
        if not sel.any():
            return None
        ts, o, h, l, c, v = w.ts[sel], w.open[sel], w.high[sel], w.low[sel], w.close[sel], w.volume[sel]
        n_src = int(ts.size)
        return Bar(
            symbol=self.symbol, tf=tf, open_time_ms=htf_open_ms,
            open=float(o[0]), high=float(h.max()), low=float(l.min()), close=float(c[-1]),
            volume=float(v.sum()), is_closed=True, n_source_bars=n_src,
            # Una vela de 1h construida con 43 minutos no es una vela de 1h. Viaja marcada.
            is_gap=n_src < int(MIN_SOURCE_COVERAGE * n),
        )

    # ------------------------------------------------------------------ ingesta

    def on_bar_1m(self, bar: Bar) -> list[Bar]:
        """Añade una vela de 1m y devuelve las de timeframe superior que hayan cerrado con ella."""
        if not bar.is_closed:
            self.state.provisional = bar
            self.state.rings[TF_1M.name].set_provisional(bar)
            return []

        r1 = self.state.rings[TF_1M.name]
        ultimo = r1.last_closed_ts_ms
        if ultimo is not None and bar.open_time_ms <= ultimo:
            return []                                   # duplicada del curado de huecos
        r1.append(bar)
        self.state.n_bars_1m += 1
        self.state.provisional = None
        self.state.health.last_closed_ms = bar.open_time_ms

        cerradas: list[Bar] = []
        fin = bar.open_time_ms + TF_1M.ms      # instante en que termina esta vela de 1m
        for tf in self.tfs:
            if tf is TF_1M or fin % tf.ms != 0:
                continue
            htf = self._close_htf(tf, fin - tf.ms)
            if htf is None:
                continue
            anillo = self.state.rings[tf.name]
            ult = anillo.last_closed_ts_ms
            if ult is None or htf.open_time_ms > ult:
                anillo.append(htf)
                # El detector se alimenta con la MISMA vela que acaba de entrar al anillo, vela a
                # vela y en orden. Nunca se le pasa un array completo: llamarlo una vez sobre todo
                # el histórico y luego trocear el resultado infla cada entrada en ~el umbral entero
                # (1,2-2,5 ATR), que es más que cualquier ventaja real.
                det = self.state.detectors.get(tf.name)
                if det is not None:
                    det.update(htf.open_time_ms, htf.high, htf.low, htf.close)
                cerradas.append(htf)
        return cerradas

    def decide(self, tf_name: str, price: float) -> dict:
        """Hipótesis ordenadas y su plan. Es lo que se pinta en la tarjeta de decisión.

        El verdict SIEMPRE se topa en WATCH mientras el nivel de madurez sea PRIOR: sin evidencia
        propia no se puede marcar nada como ACCIONABLE. El constructor de `Decision` lo impone
        además estructuralmente, así que aquí es la primera red y allí la última.
        """
        det = self.state.detectors.get(tf_name)
        anillo = self.state.rings.get(tf_name)
        if det is None or anillo is None or not len(anillo) or det.atr is None:
            return {"verdict": Verdict.NO_TRADE.value, "maturity": int(MaturityLevel.PRIOR),
                    "hypotheses": [], "reasons": ["sin estructura suficiente todavía"]}

        pivs = det.store.as_of(anillo.last_closed_ts_ms or 0)
        hips = match_impulses(pivs, self.matcher_cfg,
                              directions=tuple(self.directions))
        atr = det.atr
        out, mejor_en_zona = [], False
        for h in hips:
            r = build_plan(h, price, atr, self.plan_cfg)
            fila = {
                "id": h.id, "state": h.state.value, "direction": h.direction.name,
                "label": h.terminal_label, "score": round(h.score, 3),
                "fit": {k: round(v, 3) for k, v in h.fit.items()},
                "archetype": h.archetype, "truncated": h.truncated,
                "points": [round(x, 2) for x in h.points],
                "pivot_ts": [p.ts_ms for p in h.pivots],
                "invalidation_price": round(h.invalidation_price, 2),
                "invalidation_rule": h.invalidation_rule,
                "viable": r.viable, "in_zone": r.in_zone,
                "reasons": list(r.reasons),
            }
            if r.viable:
                fila |= {
                    "entry_lo": round(r.plan.entry_lo, 2), "entry_hi": round(r.plan.entry_hi, 2),
                    "stop": round(r.plan.stop, 2),
                    "targets": [round(t, 2) for t in r.plan.targets],
                    "rr_t2": round(r.rr_t2, 2), "cost_r": round(r.cost_r, 4),
                    "p_required": round(r.p_required, 4), "stop_atr": round(r.stop_atr, 2),
                    "size_factor": round(r.size_factor, 3),
                }
                mejor_en_zona |= r.in_zone
            out.append(fila)

        # Nivel PRIOR: la tabla de expectativas está escrita a mano y no hay ni una operación
        # resuelta. Se puede VIGILAR, nunca marcar como accionable.
        verdict = Verdict.WATCH if (out and mejor_en_zona) else (
            Verdict.WATCH if out else Verdict.NO_TRADE)
        razones = []
        if not out:
            razones.append("ninguna estructura cumple las reglas duras ahora mismo")
        elif not mejor_en_zona:
            razones.append("hay estructura, pero el precio no está en ninguna zona de entrada")
        razones.append("nivel PRIOR (n=0): expectativas de tabla experta, sin validar. "
                       "El verdict no puede pasar de WATCH.")
        return {"verdict": verdict.value, "maturity": int(MaturityLevel.PRIOR),
                "hypotheses": out, "reasons": razones, "atr": round(atr, 2),
                "price": round(price, 2)}

    def waves(self, tf_name: str, now_ms: int | None = None,
              since_ms: int | None = None) -> dict:
        """Tramos y precio de confirmación para dibujar. Nunca lanza si el timeframe no existe."""
        det = self.state.detectors.get(tf_name)
        if det is None:
            return {"legs": [], "confirm_price": None, "n_confirmed": 0, "atr": None}
        anillo = self.state.rings.get(tf_name)
        t = now_ms if now_ms is not None else (anillo.last_closed_ts_ms if anillo else 0) or 0
        return {
            "legs": det.legs_as_of(t, since_ms=since_ms),
            "confirm_price": det.confirm_price(),
            "n_confirmed": det.n_confirmed,
            "atr": det.atr,
        }

    def warmup(self, bars_1m) -> int:
        """Carga histórico. Reproduce vela a vela, igual que la ruta viva: si el calentamiento
        usase un camino distinto, el estado inicial diferiría del que produciría el replay y la
        promesa de «una sola función» sería falsa desde el primer segundo."""
        self.state.health.mode = Mode.WARMUP
        n = 0
        for b in bars_1m:
            self.on_bar_1m(b)
            n += 1
        return n

    # ------------------------------------------------------------------ modo

    def check_clock(self, trigger_tf_ms: int | None = None) -> Mode:
        """Detecta un salto de reloj (suspensión, reinicio, corte largo) y entra en CATCH_UP."""
        tf_ms = trigger_tf_ms or self.trigger.ms
        ahora = time.time()
        salto = ahora - self._last_wall
        self._last_wall = ahora
        h = self.state.health
        if h.mode is not Mode.WARMUP and salto * 1000 > 2 * tf_ms:
            h.mode = Mode.CATCH_UP
        return h.mode

    def update_health(self, connected: bool, reconnects: int, healed: int,
                      silent_seconds: float) -> Health:
        h = self.state.health
        h.connected = connected
        h.reconnects = reconnects
        h.healed_bars = healed
        h.silent_seconds = silent_seconds
        if h.last_closed_ms:
            h.lag_bars = (time.time() * 1000 - h.last_closed_ms) / self.trigger.ms
        anillo = self.state.rings.get(self.trigger.name)
        if anillo is not None and len(anillo):
            h.gaps_in_window = anillo.window(min(500, len(anillo))).n_gaps
        # Se sale de CATCH_UP cuando el retraso vuelve a estar dentro de una vela de disparo.
        if h.mode is Mode.CATCH_UP and h.lag_bars <= 1.0:
            h.mode = Mode.LIVE
        elif h.mode is Mode.WARMUP and h.last_closed_ms:
            h.mode = Mode.LIVE if h.lag_bars <= 1.0 else Mode.CATCH_UP
        return h

    @property
    def emitting(self) -> bool:
        """Si esto es False, NO se emite ninguna decisión. Es la primera red; el constructor de
        `Decision` es la última."""
        return self.state.health.mode is Mode.LIVE
