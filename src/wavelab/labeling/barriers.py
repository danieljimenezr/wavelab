"""Triple barrera: beneficio, pérdida y tiempo. La única forma honesta de etiquetar una operación.

★ LA GEOMETRÍA DE LA ETIQUETA ES LA DEL PLAN MOSTRADO. Es la corrección de un fallo sutil y letal:
si las etiquetas se calculan con un stop de "1,5 ATR" mientras la interfaz enseña un stop en la
invalidación de Elliott —que puede estar a 0,3 ATR o a 4—, entonces la tasa de acierto aprendida
describe una operación DISTINTA de la que el usuario va a hacer. Todas las probabilidades saldrían
descalibradas y ningún test lo detectaría, porque el identificador de la plantilla sí coincide.
Aquí la distancia al stop entra COMO PARÁMETRO, tomada del propio TradePlan.

AMBIGÜEDAD INTRA-VELA. Si en una misma vela el precio toca el objetivo Y el stop, no se sabe cuál
llegó antes. Se resuelve bajando a la serie de 1m, y cuando ni así se puede, se marca `ambiguous` y
se resuelve de forma PESIMISTA (stop). La TASA de ambigüedad se registra: por encima del 5% significa
que las barreras están demasiado juntas y las etiquetas no describen operaciones reales.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Outcome", "resolve_triple_barrier"]


@dataclass(frozen=True, slots=True)
class Outcome:
    barrier: str          # "tp" | "sl" | "vertical"
    r: float              # resultado en múltiplos de R (riesgo = |entrada - stop|)
    mae_r: float          # máxima excursión adversa, en R
    mfe_r: float          # máxima excursión favorable, en R
    bars: int
    ambiguous: bool = False

    @property
    def win(self) -> bool:
        return self.r > 0


def resolve_triple_barrier(
    entry: float,
    stop: float,
    target: float,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    *,
    max_bars: int,
    long: bool = True,
    fine_highs: list[np.ndarray] | None = None,
    fine_lows: list[np.ndarray] | None = None,
) -> Outcome:
    """Resuelve una operación recorriendo las velas POSTERIORES a la entrada.

    ``fine_highs``/``fine_lows``, si se aportan, son las velas de 1m dentro de cada vela del
    timeframe: permiten desempatar cuando objetivo y stop caen en la misma vela.
    """
    riesgo = abs(entry - stop)
    if riesgo <= 0:
        raise ValueError("riesgo nulo: la entrada coincide con el stop")
    s = 1.0 if long else -1.0
    n = min(len(highs), max_bars)
    mae = mfe = 0.0

    for i in range(n):
        hi, lo = float(highs[i]), float(lows[i])
        fav = s * (hi - entry) if long else s * (entry - lo)
        adv = s * (entry - lo) if long else s * (hi - entry)
        mfe = max(mfe, fav / riesgo)
        mae = max(mae, adv / riesgo)

        toca_tp = hi >= target if long else lo <= target
        toca_sl = lo <= stop if long else hi >= stop

        if toca_tp and toca_sl:
            # Ambas en la misma vela: bajar a 1m para saber cuál llegó primero.
            orden = _desempatar(entry, stop, target, long,
                               fine_highs[i] if fine_highs and i < len(fine_highs) else None,
                               fine_lows[i] if fine_lows and i < len(fine_lows) else None)
            if orden == "tp":
                return Outcome("tp", abs(target - entry) / riesgo, mae, mfe, i + 1, False)
            if orden == "sl":
                return Outcome("sl", -1.0, mae, mfe, i + 1, False)
            # Sin datos finos: PESIMISTA. Suponer el beneficio inflaría cada estadística.
            return Outcome("sl", -1.0, mae, mfe, i + 1, True)
        if toca_tp:
            return Outcome("tp", abs(target - entry) / riesgo, mae, mfe, i + 1, False)
        if toca_sl:
            return Outcome("sl", -1.0, mae, mfe, i + 1, False)

    # Barrera vertical: se cierra a mercado. Una fracción grande de las operaciones acaba así, y
    # es justo por lo que aproximar esto con un Bernoulli de dos resultados está mal.
    if n == 0:
        return Outcome("vertical", 0.0, 0.0, 0.0, 0, False)
    salida = float(closes[n - 1])
    return Outcome("vertical", s * (salida - entry) / riesgo, mae, mfe, n, False)


def _desempatar(entry: float, stop: float, target: float, long: bool,
                fh: np.ndarray | None, fl: np.ndarray | None) -> str | None:
    """Cuál se tocó antes, mirando dentro de la vela."""
    if fh is None or fl is None or len(fh) == 0:
        return None
    for j in range(len(fh)):
        hi, lo = float(fh[j]), float(fl[j])
        tp = hi >= target if long else lo <= target
        sl = lo <= stop if long else hi >= stop
        if tp and sl:
            return None          # ni con 1m se puede: sigue ambigua
        if tp:
            return "tp"
        if sl:
            return "sl"
    return None
