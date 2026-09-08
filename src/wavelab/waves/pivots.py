"""Detección causal de pivotes: ATR-ZigZag con umbral CONGELADO.

LA MECÁNICA QUE CASI TODO EL MUNDO SE SALTA. Un pivote se LOCALIZA en la vela τ donde ocurrió el
extremo, pero solo se CONFIRMA en ``c(τ) = min{t > τ : |E_τ − precio_t| >= thr}``. Hasta entonces un
nuevo máximo simplemente lo desplaza: el pivote todavía no existe. Son dos marcas de tiempo, e
``idx`` es dónde lo DIBUJAS mientras ``confirmed_idx`` es la primera vela en que estaba PERMITIDO
saberlo.

El retardo entre ambas es un tiempo de primer paso a una barrera: mediana de pocas velas, cola
derecha muy pesada, cientos de velas en una tendencia fuerte. **Nunca se puede asumir un retardo
fijo**, y por eso no basta con «desplazar N barras».

EL UMBRAL SE CONGELA EN LA VELA DEL EXTREMO. Esto es lo que hace la confirmación monótona, y de esa
monotonía cuelgan cuatro afirmaciones arquitectónicas: el histórico de conteos es genuinamente
append-only, la instantánea «como estaba en la vela t» sale gratis, el arnés de replay es O(n) en vez
de O(n²), y —lo importante— un conteo que el usuario ya vio no puede desaparecer sin evento de
invalidación.

Si el umbral se recalculase con el ATR de hoy, un pivote confirmado ayer con volatilidad baja podría
dejar de cumplir la desigualdad mañana con volatilidad expandida. Nada lanzaría. El gráfico
simplemente cambiaría de opinión sobre el pasado, que es exactamente la deshonestidad que este
diseño existe para eliminar.

Coste medido: 1,19 ms sobre 5.000 velas en Python puro (M5, ver docs/BENCHMARKS.md). No hace falta
numba, y eso está medido, no supuesto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelab.core.types import Pivot, PivotKind
from wavelab.waves.store import PivotStore

__all__ = ["WilderATR", "ZigZag", "ZigZagConfig"]


class WilderATR:
    """ATR de Wilder incremental. Estrictamente causal: solo ve velas cerradas ya entregadas."""

    __slots__ = ("_atr", "_n", "_prev_close", "_sum", "period")

    def __init__(self, period: int = 14) -> None:
        self.period = period
        self._atr: float | None = None
        self._prev_close: float | None = None
        self._n = 0
        self._sum = 0.0

    @property
    def value(self) -> float | None:
        return self._atr

    @property
    def ready(self) -> bool:
        return self._atr is not None

    def update(self, high: float, low: float, close: float) -> float | None:
        # La PRIMERA vela no tiene cierre anterior, así que su "rango verdadero" no es verdadero:
        # es solo high-low. TA-Lib la descarta y empieza a acumular en la segunda, de modo que el
        # primer ATR(14) sale en el índice 14 y no en el 13. Incluirla desviaba nuestro ATR un ~2%
        # de forma permanente (la semilla arrastra), y con él el umbral del ZigZag y por tanto qué
        # pivotes se confirman. Lo cazó el test contra TA-Lib como oráculo independiente.
        if self._prev_close is None:
            self._prev_close = close
            return None
        tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        self._prev_close = close
        if self._atr is None:
            self._n += 1
            self._sum += tr
            if self._n >= self.period:
                self._atr = self._sum / self.period   # primera media simple, como Wilder
        else:
            self._atr = (self._atr * (self.period - 1) + tr) / self.period
        return self._atr


@dataclass(frozen=True, slots=True)
class ZigZagConfig:
    k_atr: float = 1.5
    min_pct: float = 0.005
    atr_period: int = 14
    #: Confirmar por CIERRE y no por mecha. En BTC la caza de liquidaciones es endémica y usar
    #: mechas invalida una fracción enorme de estructuras por lo demás válidas. Expuesto porque
    #: es una decisión, no una constante universal — e idéntico en la ruta viva y en el replay.
    on_close: bool = False


class ZigZag:
    """Detector incremental. Alimenta un ``PivotStore`` y expone el pivote provisional.

    Se alimenta vela a vela y NUNCA se llama sobre el array completo. Llamarlo una vez sobre todo
    el histórico y luego trocear el resultado es el bug que infla cada entrada en aproximadamente
    el umbral entero (1,2-2,5 ATR), que es más grande que cualquier ventaja real, y produce un
    backtest espléndido que opera fatal.
    """

    __slots__ = (
        "_atr",
        "_confirmed_n",
        "_ext_i",
        "_ext_px",
        "_ext_thr",
        "_ext_ts",
        "_i",
        "_up",
        "cfg",
        "store",
    )

    def __init__(self, cfg: ZigZagConfig | None = None, store: PivotStore | None = None) -> None:
        self.cfg = cfg or ZigZagConfig()
        self.store = store or PivotStore()
        self._atr = WilderATR(self.cfg.atr_period)
        self._i = -1
        self._up: bool | None = None      # None = aún sin dirección
        self._ext_i = 0
        self._ext_ts = 0
        self._ext_px = 0.0
        self._ext_thr = 0.0
        self._confirmed_n = 0

    # ------------------------------------------------------------------ interno

    def _thr(self, close: float) -> float:
        """Umbral absoluto y escalado por volatilidad.

        Que sea absoluto es lo que hace `k` adimensional: el mismo k=1.5 significa lo mismo en BTC,
        en EURUSD y en AAPL. Ahí está el requisito multi-activo hecho real en vez de aspiracional.
        """
        atr = self._atr.value or 0.0
        return max(self.cfg.k_atr * atr, self.cfg.min_pct * close)

    def _set_extreme(self, i: int, ts: int, px: float, close: float) -> None:
        self._ext_i, self._ext_ts, self._ext_px = i, ts, px
        # CONGELADO aquí. No se vuelve a tocar hasta que haya un extremo nuevo.
        self._ext_thr = self._thr(close)

    # ------------------------------------------------------------------ público

    def update(self, ts_ms: int, high: float, low: float, close: float) -> Pivot | None:
        """Procesa una vela CERRADA. Devuelve el pivote recién confirmado, si lo hay."""
        self._i += 1
        self._atr.update(high, low, close)
        if not self._atr.ready:
            return None

        i, ts = self._i, ts_ms
        arriba_px = close if self.cfg.on_close else high
        abajo_px = close if self.cfg.on_close else low

        if self._up is None:
            self._up = True
            self._set_extreme(i, ts, arriba_px, close)
            self._publish_provisional()
            return None

        confirmado: Pivot | None = None

        if self._up:
            if arriba_px > self._ext_px:
                self._set_extreme(i, ts, arriba_px, close)
            elif self._ext_px - abajo_px >= self._ext_thr:
                confirmado = Pivot(self._ext_i, self._ext_ts, self._ext_px, PivotKind.HIGH,
                                   self._ext_thr).confirmed_at(i, ts)
                self.store.append_confirmed(confirmado)
                self._confirmed_n += 1
                self._up = False
                self._set_extreme(i, ts, abajo_px, close)
        else:
            if abajo_px < self._ext_px:
                self._set_extreme(i, ts, abajo_px, close)
            elif arriba_px - self._ext_px >= self._ext_thr:
                confirmado = Pivot(self._ext_i, self._ext_ts, self._ext_px, PivotKind.LOW,
                                   self._ext_thr).confirmed_at(i, ts)
                self.store.append_confirmed(confirmado)
                self._confirmed_n += 1
                self._up = True
                self._set_extreme(i, ts, arriba_px, close)

        self._publish_provisional()
        return confirmado

    def _publish_provisional(self) -> None:
        if self._up is None:
            return
        self.store.set_provisional(Pivot(
            self._ext_i, self._ext_ts, self._ext_px,
            PivotKind.HIGH if self._up else PivotKind.LOW, self._ext_thr,
        ))

    # ------------------------------------------------------------------ lectura

    @property
    def n_confirmed(self) -> int:
        return self._confirmed_n

    @property
    def atr(self) -> float | None:
        return self._atr.value

    def confirm_price(self) -> float | None:
        """El precio al que el pivote provisional quedaría confirmado.

        ★ Esta es la línea gris del gráfico: «el conteo confirma por debajo de 108.240». Casi nadie
        la implementa, y convierte la debilidad de repintado de Elliott en la línea más accionable
        de la pantalla: el usuario deja de ver «esto podría ser el techo» y pasa a ver el precio
        exacto en que deja de ser un quizá.
        """
        if self._up is None:
            return None
        return (self._ext_px - self._ext_thr) if self._up else (self._ext_px + self._ext_thr)

    def legs_as_of(self, now_ms: int, include_provisional: bool = True,
                   since_ms: int | None = None) -> list[dict]:
        """Tramos para dibujar. Los confirmados van SÓLIDOS; el provisional, DISCONTINUO.

        La separación visual no es estética: el tramo provisional cambia legítimamente según se
        mueve el precio, y presentarlo igual que uno confirmado es afirmar una certeza que no se
        tiene.
        """
        pivs = self.store.as_of(now_ms)
        if since_ms is not None:
            # Un tramo extra hacia atrás: sin él, el primer tramo visible quedaría suelto,
            # empezando en la nada en el borde izquierdo del gráfico.
            i = next((j for j, p in enumerate(pivs) if p.ts_ms >= since_ms), len(pivs))
            pivs = pivs[max(0, i - 1):]
        out = [{"ts": p.ts_ms, "price": p.price, "kind": int(p.kind),
                "confirmed_ts": p.confirmed_ts_ms, "tentative": False} for p in pivs]
        if include_provisional:
            prov = self.store.provisional_as_of(now_ms)
            if prov is not None and (not out or prov.ts_ms > out[-1]["ts"]):
                out.append({"ts": prov.ts_ms, "price": prov.price, "kind": int(prov.kind),
                            "confirmed_ts": None, "tentative": True})
        return out


def detect_batch(ts: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 cfg: ZigZagConfig | None = None) -> ZigZag:
    """Reproduce un array vela a vela. NO es una ruta vectorizada: es el mismo bucle.

    Existe para el calentamiento y los tests, no para ir más rápido. Si hubiese una versión
    vectorizada distinta, su divergencia con la ruta viva reintroduciría lookahead en silencio.
    """
    z = ZigZag(cfg)
    for i in range(ts.size):
        z.update(int(ts[i]), float(high[i]), float(low[i]), float(close[i]))
    return z
