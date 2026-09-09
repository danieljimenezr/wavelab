"""Backtest: replays history down the SAME path as the live route.

There is no vectorised path, and there never will be. If backtest and live were two separate
implementations, the divergence between them would silently reintroduce lookahead — and in a project
that labels waves from repainting pivots, that divergence is the most likely way for the whole thing
to fail without anybody noticing.

The DECISION is taken with past data (the causal engine guarantees that). The RESOLUTION uses future
data, which is both legitimate and necessary: knowing how a trade ended requires looking afterwards.

NULL ARM. Every signal spawns a twin trade entered at a RANDOM nearby instant, with the same stop in
R, the same target in R and the same time barrier. The paired delta against that arm becomes
interpretable far sooner than the absolute hit rate: it answers "does the count add anything?"
instead of "did BTC go up?", which is what an uncontrolled backtest measures.
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

    # The control arm's own geometry. It is computed inside the resolution loop and used to book
    # `null_outcome`, and until it was recorded here the twin was unobservable: `edge_vs_null` is
    # the project's headline number and the only thing that could be asserted about the arm it is
    # measured against was how many bars it ran for. Four separate faults in these four lines
    # survived the whole suite — a twin entering on the signal's own bar, one whose target lands
    # on the losing side, one whose stop is on the wrong side of its entry, one whose risk is
    # rescaled — and each of them moves `edge_vs_null` by more than its own magnitude, one of
    # them by 24x, including sign.
    null_ts_ms: int | None = None
    null_entry: float | None = None
    null_stop: float | None = None
    null_target: float | None = None

    @property
    def net_r(self) -> float:
        """R net of fees. The gross figure is a brochure number."""
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
        null = np.array([s.null_outcome.r - s.cost_r * cost_mult
                         for s in res if s.null_outcome])
        wins = r > 0
        losses = -r[~wins].sum()
        return {
            "n": len(res),
            "expectancy_r": float(r.mean()),
            "hit_rate": float(wins.mean()),
            "profit_factor": float(r[wins].sum() / losses) if losses > 0 else float("inf"),
            "r_total": float(r.sum()),
            "r_std": float(r.std()),
            "best": float(r.max()), "worst": float(r.min()),
            "null_expectancy_r": float(null.mean()) if len(null) else None,
            "edge_vs_null": float(r.mean() - null.mean()) if len(null) else None,
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
    """Replay and collect signals. Resolution happens at the end, with the later bars."""
    rng = random.Random(seed)
    tf = BY_NAME[trigger_tf]
    eng = LiveEngine(symbol, list(timeframes), 8192, trigger_tf)
    res = BacktestResult()

    htf_ts: list[int] = []
    htf_h: list[float] = []
    htf_l: list[float] = []
    htf_c: list[float] = []
    last_idx_by_archetype: dict[str, int] = {}

    for bar in bars_1m:
        res.bars_processed += 1
        if res.bars_processed % progress_every == 0:
            print(f"  … {res.bars_processed:,} bars, {len(res.signals)} signals", flush=True)
        closed = eng.on_bar_1m(bar)
        for htf in closed:
            if htf.tf is not tf:
                continue
            htf_ts.append(htf.open_time_ms)
            htf_h.append(htf.high); htf_l.append(htf.low); htf_c.append(htf.close)

            res.decisions_evaluated += 1
            d = eng.decide(trigger_tf, htf.close)
            for h in d["hypotheses"]:
                if not h["viable"] or not h["in_zone"] or h["score"] < min_score:
                    if not h["viable"]:
                        reason = h["reasons"][0].split(":")[0][:48] if h["reasons"] else "?"
                        res.rejected[reason] = res.rejected.get(reason, 0) + 1
                    continue
                arch = h["archetype"] or "?"
                idx = len(htf_ts) - 1
                if idx - last_idx_by_archetype.get(arch, -10**9) < cooldown_bars:
                    continue        # debounce: do not re-enter the same count on every bar
                last_idx_by_archetype[arch] = idx
                is_long = h["direction"] == "LONG"
                res.signals.append(Signal(
                    ts_ms=htf.open_time_ms, archetype=f"{arch}_{'long' if is_long else 'short'}",
                    direction=Direction.LONG if is_long else Direction.SHORT,
                    entry=htf.close, stop=h["stop"], target=h["targets"][1],
                    rr=h["rr_t2"], score=h["score"], cost_r=h["cost_r"],
                    size_factor=h["size_factor"], invalidation=h["invalidation_price"],
                ))

    # ---------------------------------------------------------------- resolution
    H, L, C = np.array(htf_h), np.array(htf_l), np.array(htf_c)
    pos = {t: i for i, t in enumerate(htf_ts)}
    ambiguous_n = 0
    for s in res.signals:
        i = pos.get(s.ts_ms)
        if i is None or i + 1 >= len(H):
            continue
        is_long = s.direction is Direction.LONG
        s.outcome = resolve_triple_barrier(
            s.entry, s.stop, s.target, H[i + 1:], L[i + 1:], C[i + 1:],
            max_bars=max_bars_hold, long=is_long)
        ambiguous_n += int(s.outcome.ambiguous)

        # Null arm: same geometry in R, entry shifted at random. Isolates "does the count add
        # anything?" from "did BTC go up over that stretch?", which is what an uncontrolled
        # backtest measures.
        j = min(len(H) - 2, i + rng.randint(1, 20))
        e = float(C[j]); risk = abs(s.entry - s.stop); rr = abs(s.target - s.entry) / risk
        st = e - risk if is_long else e + risk
        tg = e + risk * rr if is_long else e - risk * rr
        s.null_ts_ms, s.null_entry, s.null_stop, s.null_target = htf_ts[j], e, st, tg
        s.null_outcome = resolve_triple_barrier(
            e, st, tg, H[j + 1:], L[j + 1:], C[j + 1:], max_bars=max_bars_hold, long=is_long)

    n = sum(1 for s in res.signals if s.outcome)
    res.ambiguity_rate = ambiguous_n / n if n else 0.0
    return res
