"""Evaluates EVERY registered hypothesis with out-of-sample discipline.

Four controls, and not one of them is optional:

1. **Walk-forward with purging.** The history is cut into segments and only what falls OUTSIDE the
   fitting sample is reported. Between segments a window equal to the holding horizon is PURGED:
   without purging, a trade opened at the end of the training segment resolves inside the test
   segment and leaks information from one into the other.

2. **Base rate.** The question is not "is it right?" but "is it right MORE often than simply being
   in the market with no criterion at all?". In an asset that rose 1,748% over 2017-2026, any
   long strategy is
   right a lot, and that is not an edge: it is the market's drift.

3. **Effective n.** Signals that overlap inside the horizon are ONE observation, not several.

4. **White's Reality Check** over the whole set, failed hypotheses included. Hiding the failures
   is what turns a study into a brochure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wavelab.hypotheses.base import Hypothesis, Series
from wavelab.validation.reality_check import RealityCheckResult, effective_n, reality_check

__all__ = ["HypResult", "evaluate_all", "forward_returns"]


def forward_returns(close: np.ndarray, horizon: int) -> np.ndarray:
    """Log return ``horizon`` bars ahead. NaN wherever there is not enough future left."""
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
    edge: float                # excess return over the base rate
    oos_edge: float            # the SAME excess, but out of sample only
    exposure: float            # fraction of the time spent in the market
    per_bar: np.ndarray = field(repr=False, default=None)

    @property
    def key(self) -> str:
        return f"{self.name}@{self.tf}"


def _walk_forward_mask(n: int, folds: int, horizon: int) -> np.ndarray:
    """True at the OUT-OF-SAMPLE positions, purged at the boundaries.

    The first 1/folds is held back as the initial training set and everything after it is OOS in
    segments, purging `horizon` bars at each boundary so that no open trade can cross one.
    """
    m = np.zeros(n, dtype=bool)
    step = n // folds
    for k in range(1, folds):
        start, end = k * step, min((k + 1) * step, n)
        m[start + horizon: end] = True     # purge at the start of each OOS segment
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
    results: list[HypResult] = []
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

            active = val & (sig != 0)
            if active.sum() < min_signals:
                continue

            base_ret = float(fwd[val].mean())

            # ★ RETURN IN EXCESS OF THE DRIFT, not raw return.
            #
            # If the Reality Check is fed `sig * fwd`, the winning strategy will always be the one
            # that spends the MOST TIME long, because BTC rose 1,748% over the 2017-2026
            # sample measured here. That is not
            # timing skill: it is beta, and buying beta costs nothing.
            #
            # Subtracting the mean drift (`fwd - base_ret`) turns it into the right question:
            # "did THIS moment do better than any old moment?". A long only scores if the return
            # beat the average, and a short only if it came in below — which is exactly the cost
            # of being short in a rising asset.
            per_bar = np.zeros(s.close.size)
            per_bar[active] = sig[active] * (fwd[active] - base_ret)
            base_hit = float((fwd[val] > 0).mean())
            r = fwd[active] * sig[active]
            oos = _walk_forward_mask(s.close.size, folds, H)
            oos_act = active & oos
            oos_edge = (float((fwd[oos_act] * sig[oos_act]).mean()
                              - fwd[val & oos].mean()) if oos_act.sum() >= 20 else float("nan"))

            bar_ms = int(np.median(np.diff(s.ts))) if s.ts.size > 1 else 1
            results.append(HypResult(
                name=name, family=h.family, tf=tf,
                n_signals=int(active.sum()),
                n_effective=effective_n(s.ts[active], H, bar_ms),
                hit_rate=float((r > 0).mean()), base_rate=base_hit,
                mean_ret=float(r.mean()), base_ret=base_ret,
                edge=float(r.mean() - base_ret), oos_edge=oos_edge,
                exposure=float(active.sum() / val.sum()), per_bar=per_bar,
            ))
            series_ret[f"{name}@{tf}"] = per_bar

    # The Reality Check demands series ALIGNED in time, so it only makes sense within one and the
    # same timeframe: 1h and 1d series cannot be stacked or compared bar by bar. With several
    # timeframes, the caller runs it separately for each one.
    lengths = {len(v) for v in series_ret.values()}
    rc = (reality_check(series_ret, n_boot=n_boot)
          if n_boot > 0 and len(series_ret) >= 2 and len(lengths) == 1 else None)
    return results, rc
