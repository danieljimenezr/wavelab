"""Timeframes, sus roles, y el resampleo desde la única serie fuente (1m).

Dos invariantes duros viven aquí:

**Separación de roles.** 1d/4h son un VETO direccional que nunca emite una señal con marca de tiempo;
1h clasifica el régimen; 15m es el ÚNICO timeframe con permiso para dibujar entradas y salidas. Sin
esto, dos timeframes pueden emitir señales que se contradicen y el usuario no tiene forma de saber a
cuál hacer caso.

**Una sola serie fuente.** Se guarda 1m y todo lo demás se resamplea en local. Si cada timeframe se
pidiese a la API por separado, sus fronteras y sus huecos no coincidirían y la lógica de acuerdo entre
timeframes mediría ruido de alineación en vez de estructura de mercado.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pandas solo en la frontera de E/S; jamás dentro de un bucle
    import pandas as pd

__all__ = [
    "Timeframe", "TfRole", "TF_1M", "TF_5M", "TF_15M", "TF_1H", "TF_4H", "TF_1D",
    "BY_NAME", "MIN_SOURCE_COVERAGE", "resample_from_1m", "close_time_for",
]

MINUTE_MS = 60_000

#: Cobertura mínima de velas de 1m para que una vela resampleada pueda emitir señal.
#: Una vela de 1h construida con 43 minutos no es una vela de 1h; es una mentira con forma de vela.
MIN_SOURCE_COVERAGE = 0.9


class TfRole(StrEnum):
    SOURCE = "source"    # 1m: solo se almacena, nunca se analiza directamente
    TRIGGER = "trigger"  # el único que dibuja entradas/salidas
    REGIME = "regime"    # clasifica el régimen de mercado
    GATE = "gate"        # veto direccional; NUNCA emite una señal con marca de tiempo


@dataclass(frozen=True, slots=True, order=True)
class Timeframe:
    ms: int
    name: str
    role: TfRole

    @property
    def minutes(self) -> int:
        return self.ms // MINUTE_MS

    @property
    def expected_source_bars(self) -> int:
        """Cuántas velas de 1m debería contener una vela completa de este timeframe."""
        return self.ms // MINUTE_MS

    @property
    def can_emit_signals(self) -> bool:
        """Solo el timeframe de disparo dibuja entradas. El resto informa o veta."""
        return self.role is TfRole.TRIGGER

    def floor_ms(self, ts_ms: int) -> int:
        """Inicio de la vela de este timeframe que contiene ``ts_ms`` (rejilla UTC)."""
        return ts_ms - (ts_ms % self.ms)

    def __str__(self) -> str:
        return self.name


TF_1M = Timeframe(1 * MINUTE_MS, "1m", TfRole.SOURCE)
TF_5M = Timeframe(5 * MINUTE_MS, "5m", TfRole.SOURCE)
TF_15M = Timeframe(15 * MINUTE_MS, "15m", TfRole.TRIGGER)
TF_1H = Timeframe(60 * MINUTE_MS, "1h", TfRole.REGIME)
TF_4H = Timeframe(240 * MINUTE_MS, "4h", TfRole.GATE)
TF_1D = Timeframe(1440 * MINUTE_MS, "1d", TfRole.GATE)

BY_NAME: dict[str, Timeframe] = {tf.name: tf for tf in (TF_1M, TF_5M, TF_15M, TF_1H, TF_4H, TF_1D)}


def close_time_for(open_time_ms: int, tf: Timeframe) -> int:
    """El contrato de `close_time`, en un solo sitio.

    Binance devuelve ``open + duración - 1 ms``; OKX, Bybit y Coinbase solo devuelven ``open_time``.
    Si cada adaptador normalizase a su manera, un resampleo que funciona hoy por un accidente de 1 ms
    se rompería el día que alguien redondease al alza. Aquí se DERIVA siempre de ``open_time``, que es
    inequívoco, y los adaptadores tienen prohibido inventarse otro convenio.
    """
    return open_time_ms + tf.ms - 1


def resample_from_1m(df_1m: "pd.DataFrame", tf: Timeframe) -> "pd.DataFrame":
    """Resamplea 1m al timeframe pedido, contando las velas fuente reales.

    ``df_1m`` debe estar indexado por ``open_time_ms`` (int64, UTC), sin duplicados y ordenado.
    NO tiene que estar completo: los huecos son normales y su cuenta es justamente lo que hay que
    propagar.

    El resampleo se hace sobre **open_time**, no sobre close_time. El plan original decía
    "indexado por close_time", pero eso solo funciona porque el ``-1 ms`` de Binance cae dentro del
    intervalo correcto; con un proveedor que redondease distinto, cada vela se asignaría al periodo
    siguiente en silencio. Sobre open_time no hay ambigüedad posible.

    Devuelve columnas OHLCV más:
      ``n_source_bars``  velas de 1m que compusieron la vela
      ``is_gap``         True si la cobertura es insuficiente para operar sobre ella
    """
    import pandas as pd

    if df_1m.empty:
        return df_1m.iloc[0:0].assign(n_source_bars=pd.Series(dtype="int32"),
                                      is_gap=pd.Series(dtype="bool"))

    idx = df_1m.index.to_numpy()
    if (idx[1:] <= idx[:-1]).any():
        raise ValueError("resample_from_1m: el índice debe ser estrictamente creciente y sin duplicados")

    bucket = idx - (idx % tf.ms)
    g = df_1m.groupby(bucket, sort=True)

    agg = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum",
    }
    for optional in ("quote_volume", "trades", "taker_buy_base", "taker_buy_quote"):
        if optional in df_1m.columns:
            agg[optional] = "sum"

    out = g.agg(agg)
    out["n_source_bars"] = g.size().astype("int32")
    out.index.name = "open_time_ms"

    expected = tf.expected_source_bars
    out["is_gap"] = out["n_source_bars"] < int(MIN_SOURCE_COVERAGE * expected)
    out["close_time_ms"] = out.index.to_numpy() + tf.ms - 1
    return out
