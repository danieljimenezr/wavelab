"""Backtest: reproduce el histórico por el MISMO camino que la ruta viva.

No hay ruta vectorizada, y no la habrá. Si backtest y live fuesen dos implementaciones, su
divergencia reintroduciría lookahead en silencio — y en un proyecto que etiqueta ondas a partir de
pivotes que repintan, esa divergencia es la forma más probable de que todo falle sin que nadie se
entere.

La DECISIÓN se toma con datos pasados (lo garantiza el motor causal). La RESOLUCIÓN usa datos
futuros, que es legítimo y necesario: saber cómo acabó una operación exige mirar después.

BRAZO NULO. Cada señal genera una operación gemela con entrada en un instante ALEATORIO cercano,
mismo stop en R, mismo objetivo en R y misma barrera temporal. El delta emparejado contra ese brazo
es interpretable muchísimo antes que la tasa de acierto absoluta: responde a "¿aporta algo el
conteo?" en vez de a "¿sube BTC?", que es lo que mide un backtest sin control.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from wavelab.core.timeframes import BY_NAME
from wavelab.core.types import Direction
from wavelab.engine.live import LiveEngine
from wavelab.labeling.barriers import Outcome, resolve_triple_barrier

__all__ = ["BacktestResult", "Signal", "run_backtest"]


@dataclass(slots=True)
class Signal:
    ts_ms: int
    archetype: str
    direction: Direction
    entry: float
    stop: float
    target: float
    rr: float
    score: float
    cost_r: float
    size_factor: float
    invalidation: float
    outcome: Outcome | None = None
    null_outcome: Outcome | None = None

    @property
    def net_r(self) -> float:
        """R neto de comisiones. El bruto es una cifra de folleto."""
        return (self.outcome.r - self.cost_r) if self.outcome else 0.0

    @property
    def null_net_r(self) -> float:
        return (self.null_outcome.r - self.cost_r) if self.null_outcome else 0.0


@dataclass(slots=True)
class BacktestResult:
    signals: list[Signal] = field(default_factory=list)
    bars_processed: int = 0
    decisions_evaluated: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    ambiguity_rate: float = 0.0

    def stats(self, cost_mult: float = 1.0) -> dict:
        res = [s for s in self.signals if s.outcome]
        if not res:
            return {"n": 0}
        r = np.array([s.outcome.r - s.cost_r * cost_mult for s in res])
        nulo = np.array([s.null_outcome.r - s.cost_r * cost_mult
                         for s in res if s.null_outcome])
        wins = r > 0
        perdidas = -r[~wins].sum()
        return {
            "n": len(res),
            "expectancy_r": float(r.mean()),
            "hit_rate": float(wins.mean()),
            "profit_factor": float(r[wins].sum() / perdidas) if perdidas > 0 else float("inf"),
            "r_total": float(r.sum()),
            "r_std": float(r.std()),
            "best": float(r.max()), "worst": float(r.min()),
            "null_expectancy_r": float(nulo.mean()) if len(nulo) else None,
            "edge_vs_null": float(r.mean() - nulo.mean()) if len(nulo) else None,
            "barriers": {b: sum(1 for s in res if s.outcome.barrier == b)
                         for b in ("tp", "sl", "vertical")},
        }


def run_backtest(
    bars_1m,
    symbol: str = "BTCUSDT",
    trigger_tf: str = "4h",
    timeframes: tuple[str, ...] = ("1m", "15m", "1h", "4h", "1d"),
    *,
    max_bars_hold: int = 48,
    cooldown_bars: int = 6,
    min_score: float = 0.0,
    seed: int = 17,
    progress_every: int = 500_000,
) -> BacktestResult:
    """Reproduce y recoge señales. La resolución se hace al final, con las velas posteriores."""
    rng = random.Random(seed)
    tf = BY_NAME[trigger_tf]
    eng = LiveEngine(symbol, list(timeframes), 8192, trigger_tf)
    res = BacktestResult()

    htf_ts: list[int] = []
    htf_h: list[float] = []
    htf_l: list[float] = []
    htf_c: list[float] = []
    ultimo_ts_por_arq: dict[str, int] = {}

    for bar in bars_1m:
        res.bars_processed += 1
        if res.bars_processed % progress_every == 0:
            print(f"  … {res.bars_processed:,} velas, {len(res.signals)} señales", flush=True)
        cerradas = eng.on_bar_1m(bar)
        for htf in cerradas:
            if htf.tf is not tf:
                continue
            htf_ts.append(htf.open_time_ms)
            htf_h.append(htf.high); htf_l.append(htf.low); htf_c.append(htf.close)

            res.decisions_evaluated += 1
            d = eng.decide(trigger_tf, htf.close)
            for h in d["hypotheses"]:
                if not h["viable"] or not h["in_zone"] or h["score"] < min_score:
                    if not h["viable"]:
                        motivo = h["reasons"][0].split(":")[0][:48] if h["reasons"] else "?"
                        res.rejected[motivo] = res.rejected.get(motivo, 0) + 1
                    continue
                arq = h["archetype"] or "?"
                idx = len(htf_ts) - 1
                if idx - ultimo_ts_por_arq.get(arq, -10**9) < cooldown_bars:
                    continue        # antirrebote: no reentrar en el mismo conteo cada vela
                ultimo_ts_por_arq[arq] = idx
                largo = h["direction"] == "LONG"
                res.signals.append(Signal(
                    ts_ms=htf.open_time_ms, archetype=f"{arq}_{'long' if largo else 'short'}",
                    direction=Direction.LONG if largo else Direction.SHORT,
                    entry=htf.close, stop=h["stop"], target=h["targets"][1],
                    rr=h["rr_t2"], score=h["score"], cost_r=h["cost_r"],
                    size_factor=h["size_factor"], invalidation=h["invalidation_price"],
                ))

    # ---------------------------------------------------------------- resolución
    H, L, C = np.array(htf_h), np.array(htf_l), np.array(htf_c)
    pos = {t: i for i, t in enumerate(htf_ts)}
    ambiguas = 0
    for s in res.signals:
        i = pos.get(s.ts_ms)
        if i is None or i + 1 >= len(H):
            continue
        largo = s.direction is Direction.LONG
        s.outcome = resolve_triple_barrier(
            s.entry, s.stop, s.target, H[i + 1:], L[i + 1:], C[i + 1:],
            max_bars=max_bars_hold, long=largo)
        ambiguas += int(s.outcome.ambiguous)

        # Brazo nulo: misma geometría en R, entrada desplazada al azar. Aísla "¿aporta el conteo?"
        # de "¿subió BTC en ese periodo?", que es lo que mide un backtest sin control.
        j = min(len(H) - 2, i + rng.randint(1, 20))
        e = float(C[j]); riesgo = abs(s.entry - s.stop); rr = abs(s.target - s.entry) / riesgo
        st = e - riesgo if largo else e + riesgo
        tg = e + riesgo * rr if largo else e - riesgo * rr
        s.null_outcome = resolve_triple_barrier(
            e, st, tg, H[j + 1:], L[j + 1:], C[j + 1:], max_bars=max_bars_hold, long=largo)

    n = sum(1 for s in res.signals if s.outcome)
    res.ambiguity_rate = ambiguas / n if n else 0.0
    return res
