"""Evalúa TODAS las hipótesis registradas con disciplina fuera de muestra.

Cuatro controles, y ninguno es opcional:

1. **Walk-forward con purga.** Se parte el histórico en tramos y solo se reporta lo que cae FUERA
   de la muestra de ajuste. Entre tramos se PURGA una ventana igual al horizonte de tenencia: sin
   purgar, una operación abierta al final del tramo de entrenamiento se resuelve dentro del de
   prueba y filtra información entre ambos.

2. **Tasa base.** No se mide "¿acierta?" sino "¿acierta MÁS que estar dentro sin criterio?". En un
   activo que subió un 1.748%, cualquier estrategia larga acierta mucho, y eso no es una ventaja:
   es la deriva del mercado.

3. **n efectivo.** Señales que se solapan dentro del horizonte son UNA observación, no varias.

4. **Reality Check de White** sobre el conjunto completo, incluidas las hipótesis fracasadas.
   Ocultar las fallidas es lo que convierte un estudio en un folleto.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wavelab.hypotheses.base import Hypothesis, Series
from wavelab.validation.reality_check import RealityCheckResult, effective_n, reality_check

__all__ = ["HypResult", "evaluate_all", "forward_returns"]


def forward_returns(close: np.ndarray, horizon: int) -> np.ndarray:
    """Retorno logarítmico a ``horizon`` velas vista. NaN donde no hay futuro suficiente."""
    fwd = np.full(close.size, np.nan)
    if close.size > horizon:
        fwd[:-horizon] = np.log(close[horizon:] / close[:-horizon])
    return fwd


@dataclass(slots=True)
class HypResult:
    name: str
    family: str
    tf: str
    n_signals: int
    n_effective: int
    hit_rate: float
    base_rate: float
    mean_ret: float
    base_ret: float
    edge: float                # exceso de retorno sobre la tasa base
    oos_edge: float            # el MISMO exceso, pero solo fuera de muestra
    exposure: float            # fracción del tiempo dentro de mercado
    per_bar: np.ndarray = field(repr=False, default=None)

    @property
    def key(self) -> str:
        return f"{self.name}@{self.tf}"


def _walk_forward_mask(n: int, folds: int, horizon: int) -> np.ndarray:
    """True en las posiciones FUERA de muestra, con purga en las fronteras.

    Se reservan los primeros 1/folds como entrenamiento inicial y todo lo posterior es OOS por
    tramos, purgando `horizon` velas en cada frontera para que ninguna operación abierta cruce.
    """
    m = np.zeros(n, dtype=bool)
    paso = n // folds
    for k in range(1, folds):
        ini, fin = k * paso, min((k + 1) * paso, n)
        m[ini + horizon: fin] = True     # purga al principio de cada tramo OOS
    return m


def evaluate_all(
    hyps: dict[str, Hypothesis],
    series: dict[str, Series],
    *,
    horizon_bars: dict[str, int] | None = None,
    folds: int = 5,
    n_boot: int = 2000,
    min_signals: int = 60,
) -> tuple[list[HypResult], RealityCheckResult | None]:
    horizon_bars = horizon_bars or {"15m": 32, "1h": 24, "4h": 12, "1d": 5}
    resultados: list[HypResult] = []
    series_ret: dict[str, np.ndarray] = {}

    for name, h in sorted(hyps.items()):
        for tf in h.timeframes:
            s = series.get(tf)
            if s is None or len(s) < 500:
                continue
            H = horizon_bars.get(tf, 12)
            fwd = forward_returns(s.close, H)
            val = ~np.isnan(fwd)
            try:
                sig = h.signals(s)
            except Exception as e:  # noqa: BLE001
                print(f"  [!] {name}@{tf}: {type(e).__name__}: {e}")
                continue

            activo = val & (sig != 0)
            if activo.sum() < min_signals:
                continue

            base_ret = float(fwd[val].mean())

            # ★ RETORNO EN EXCESO DE LA DERIVA, no retorno bruto.
            #
            # Si el Reality Check se alimenta con `sig * fwd`, la estrategia ganadora será siempre
            # la que MÁS TIEMPO pase larga, porque BTC subió un 1.748% en la muestra. Eso no es
            # habilidad de temporización: es beta, y comprarla cuesta cero.
            #
            # Restando la deriva media (`fwd - base_ret`) la pregunta pasa a ser la correcta:
            # "¿acertó ESTE momento mejor que un momento cualquiera?". Un largo solo puntúa si el
            # retorno superó a la media, y un corto solo si quedó por debajo — que es justo el
            # coste de ponerse corto en un activo alcista.
            per_bar = np.zeros(s.close.size)
            per_bar[activo] = sig[activo] * (fwd[activo] - base_ret)
            base_hit = float((fwd[val] > 0).mean())
            r = fwd[activo] * sig[activo]
            oos = _walk_forward_mask(s.close.size, folds, H)
            oos_act = activo & oos
            oos_edge = (float((fwd[oos_act] * sig[oos_act]).mean()
                              - fwd[val & oos].mean()) if oos_act.sum() >= 20 else float("nan"))

            bar_ms = int(np.median(np.diff(s.ts))) if s.ts.size > 1 else 1
            resultados.append(HypResult(
                name=name, family=h.family, tf=tf,
                n_signals=int(activo.sum()),
                n_effective=effective_n(s.ts[activo], H, bar_ms),
                hit_rate=float((r > 0).mean()), base_rate=base_hit,
                mean_ret=float(r.mean()), base_ret=base_ret,
                edge=float(r.mean() - base_ret), oos_edge=oos_edge,
                exposure=float(activo.sum() / val.sum()), per_bar=per_bar,
            ))
            series_ret[f"{name}@{tf}"] = per_bar

    # El Reality Check exige series ALINEADAS en el tiempo, así que solo tiene sentido dentro de
    # un mismo timeframe: series de 1h y de 1d no se pueden apilar ni comparar vela a vela.
    # Con varios timeframes, el llamante lo ejecuta por separado para cada uno.
    largos = {len(v) for v in series_ret.values()}
    rc = (reality_check(series_ret, n_boot=n_boot)
          if n_boot > 0 and len(series_ret) >= 2 and len(largos) == 1 else None)
    return resultados, rc
