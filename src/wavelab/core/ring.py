"""Buffer circular con marcas de tiempo, y las ventanas que salen de él.

El error que este módulo existe para impedir: un buffer indexado por POSICIÓN sobre un almacén
indexado por TIEMPO. Las series de 1m de Binance tienen agujeros (paradas, reconexiones, incidencias).
Con indexación posicional, ``i-20`` para un ER(20) atraviesa un hueco de horas **en silencio** y
devuelve un número seguro y equivocado. No lanza, no avisa, y el valor es un float perfectamente
válido.

Por eso el ring guarda los timestamps JUNTO a los precios y cada ventana **asevera el paso** en vez de
fiarse de la posición.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelab.core.timeframes import Timeframe, close_time_for
from wavelab.core.types import Bar

__all__ = ["GapError", "ProvisionalWindow", "Ring", "Window"]


class GapError(ValueError):
    """La ventana pedida contiene un hueco que invalida el cálculo solicitado."""


@dataclass(frozen=True, slots=True)
class Window:
    """Vista sobre velas CERRADAS, con la discontinuidad explícita.

    ``end_closed_ts_ms`` se fija exclusivamente desde la última vela cerrada, nunca desde el cursor de
    escritura del ring. Es la diferencia entre una guarda que funciona y una que se salta sola: si el
    final de la ventana se calculase desde el cursor, incluiría la vela en curso y el ATR se movería
    intra-vela → el umbral del ZigZag se movería intra-vela → la confirmación de pivotes se movería
    intra-vela. Y nada lanzaría, porque los valores son float64 en ambos casos.
    """

    symbol: str
    tf: Timeframe
    ts: np.ndarray            # open_time_ms de cada vela
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    n_source_bars: np.ndarray
    is_gap: np.ndarray        # True si falta la vela ANTERIOR o su cobertura es insuficiente
    end_closed_ts_ms: int

    def __len__(self) -> int:
        return int(self.ts.size)

    @property
    def complete(self) -> bool:
        """Sin discontinuidades ni velas mal cubiertas en toda la ventana."""
        return not bool(self.is_gap.any())

    @property
    def n_gaps(self) -> int:
        return int(self.is_gap.sum())

    def require_complete(self, what: str) -> Window:
        """Para cálculos que no toleran huecos. Falla ruidosamente en vez de mentir en silencio."""
        if not self.complete:
            first = int(np.flatnonzero(self.is_gap)[0])
            raise GapError(
                f"{what}: la ventana de {len(self)} velas de {self.tf} tiene {self.n_gaps} "
                f"discontinuidad(es); la primera en ts={int(self.ts[first])}. "
                "Rehúsa calcular en vez de devolver un número confiado y equivocado."
            )
        return self


@dataclass(frozen=True, slots=True)
class ProvisionalWindow:
    """La misma forma que ``Window`` pero un tipo DISTINTO, y a propósito.

    Incluye la vela en curso. ``@causal`` lo rechaza SIEMPRE, sin mirar fechas, así que el canal
    provisional es estructuralmente incapaz de alimentar features causales, señales, journal o
    estadísticas. Solo puede producir anotaciones tentativas: trazo discontinuo, etiqueta hueca «?».
    """

    __wavelab_provisional__ = True

    symbol: str
    tf: Timeframe
    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    n_source_bars: np.ndarray
    is_gap: np.ndarray
    last_ts_ms: int           # deliberadamente NO se llama end_closed_ts_ms

    def __len__(self) -> int:
        return int(self.ts.size)


_FIELDS = ("open", "high", "low", "close", "volume")


class Ring:
    """Buffer circular de velas cerradas de un símbolo y timeframe, más la vela en curso."""

    __slots__ = ("_cap", "_cols", "_gap", "_n", "_nsrc", "_prov", "_ts", "_w", "symbol", "tf")

    def __init__(self, symbol: str, tf: Timeframe, capacity: int = 8192) -> None:
        if capacity < 2:
            raise ValueError("capacity debe ser >= 2")
        self.symbol = symbol
        self.tf = tf
        self._cap = int(capacity)
        self._n = 0          # cuántas velas se han escrito en total
        self._w = 0          # cursor de escritura
        self._ts = np.zeros(capacity, dtype=np.int64)
        self._cols = {f: np.zeros(capacity, dtype=np.float64) for f in _FIELDS}
        self._nsrc = np.zeros(capacity, dtype=np.int32)
        self._gap = np.zeros(capacity, dtype=bool)
        self._prov: Bar | None = None

    # ---------------------------------------------------------------- escritura

    def append(self, bar: Bar) -> None:
        """Añade una vela CERRADA. Rechaza velas en curso, fuera de rejilla o fuera de orden."""
        if not bar.is_closed:
            raise ValueError(
                f"Ring.append: {bar.symbol} {bar.tf} en {bar.open_time_ms} no está cerrada. "
                "Usa set_provisional() para la vela en curso."
            )
        if bar.tf is not self.tf or bar.symbol != self.symbol:
            raise ValueError(
                f"Ring.append: esperaba {self.symbol} {self.tf}, recibido {bar.symbol} {bar.tf}"
            )
        if self._n:
            last = int(self._ts[(self._w - 1) % self._cap])
            if bar.open_time_ms <= last:
                raise ValueError(
                    f"Ring.append: vela fuera de orden o duplicada "
                    f"(open_time_ms={bar.open_time_ms} <= última={last}). "
                    "La deduplicación es responsabilidad del almacén, no del ring."
                )
            # Discontinuidad: falta al menos una vela entre la anterior y ésta.
            discontinuous = (bar.open_time_ms - last) != self.tf.ms
        else:
            discontinuous = False

        i = self._w
        self._ts[i] = bar.open_time_ms
        self._cols["open"][i] = bar.open
        self._cols["high"][i] = bar.high
        self._cols["low"][i] = bar.low
        self._cols["close"][i] = bar.close
        self._cols["volume"][i] = bar.volume
        self._nsrc[i] = bar.n_source_bars
        self._gap[i] = discontinuous or bar.is_gap
        self._w = (i + 1) % self._cap
        self._n += 1
        self._prov = None  # la vela en curso queda absorbida por su versión cerrada

    def set_provisional(self, bar: Bar) -> None:
        if bar.is_closed:
            raise ValueError("set_provisional espera la vela EN CURSO (is_closed=False)")
        self._prov = bar

    # ---------------------------------------------------------------- lectura

    def __len__(self) -> int:
        return min(self._n, self._cap)

    @property
    def last_closed_ts_ms(self) -> int | None:
        if not self._n:
            return None
        return int(self._ts[(self._w - 1) % self._cap])

    def _take(self, n: int) -> np.ndarray:
        """Índices absolutos de las últimas ``n`` velas, resolviendo el envoltorio circular."""
        avail = len(self)
        n = min(n, avail)
        start = (self._w - n) % self._cap
        return (start + np.arange(n)) % self._cap

    def window(self, n: int) -> Window:
        """Ventana de las últimas ``n`` velas CERRADAS.

        Recalcula la máscara de huecos desde los timestamps reales en cada construcción: no se fía
        de lo que se marcó al escribir, porque el ring puede haber dado la vuelta.
        """
        if not self._n:
            raise ValueError("Ring vacío: no hay ninguna vela cerrada")
        idx = self._take(n)
        ts = self._ts[idx].copy()

        d = np.diff(ts)
        if (d <= 0).any():
            bad = int(np.flatnonzero(d <= 0)[0])
            raise ValueError(
                f"Ring.window: timestamps no crecientes en la posición {bad} "
                f"({int(ts[bad])} -> {int(ts[bad+1])}). El ring está corrupto."
            )
        gap = self._gap[idx].copy()
        gap[1:] |= d != self.tf.ms  # el paso REAL manda sobre lo que se anotó al escribir

        end_ts = int(ts[-1])
        return Window(
            symbol=self.symbol, tf=self.tf, ts=ts,
            open=self._cols["open"][idx].copy(),
            high=self._cols["high"][idx].copy(),
            low=self._cols["low"][idx].copy(),
            close=self._cols["close"][idx].copy(),
            volume=self._cols["volume"][idx].copy(),
            n_source_bars=self._nsrc[idx].copy(),
            is_gap=gap,
            end_closed_ts_ms=close_time_for(end_ts, self.tf),
        )

    def provisional_window(self, n: int) -> ProvisionalWindow:
        """Ventana que INCLUYE la vela en curso. Solo para anotación tentativa."""
        if self._prov is None:
            raise ValueError("no hay vela en curso: llama antes a set_provisional()")
        w = self.window(max(0, n - 1))
        p = self._prov
        cat = lambda a, v: np.concatenate([a, np.array([v], dtype=a.dtype)])
        return ProvisionalWindow(
            symbol=self.symbol, tf=self.tf,
            ts=cat(w.ts, p.open_time_ms),
            open=cat(w.open, p.open), high=cat(w.high, p.high),
            low=cat(w.low, p.low), close=cat(w.close, p.close),
            volume=cat(w.volume, p.volume),
            n_source_bars=cat(w.n_source_bars, p.n_source_bars),
            is_gap=cat(w.is_gap, p.is_gap),
            last_ts_ms=p.open_time_ms,
        )
