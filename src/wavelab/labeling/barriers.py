"""Triple barrier: profit, loss and time. The only honest way to label a trade.

★ THE LABEL'S GEOMETRY IS THE GEOMETRY OF THE PLAN ON SCREEN. This is the fix for a subtle, lethal
bug: if the labels are computed with a "1.5 ATR" stop while the interface shows a stop at the
Elliott invalidation — which may sit 0.3 ATR away, or 4 — then the hit rate that gets learned
describes a DIFFERENT trade from the one the user is going to take. Every probability would come out
miscalibrated and no test would catch it, because the template id does match. Here the stop distance
comes in AS A PARAMETER, taken from the TradePlan itself.

INTRA-BAR AMBIGUITY. If price touches the target AND the stop within the same bar, there is no way
to know which came first. It is resolved by dropping down to the 1m series, and when even that is
not enough it is flagged `ambiguous` and resolved PESSIMISTICALLY (stop). The ambiguity RATE is
recorded: above 5% it means the barriers are too close together and the labels no longer describe
real trades.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Outcome", "resolve_triple_barrier"]


@dataclass(frozen=True, slots=True)
class Outcome:
    barrier: str          # "tp" | "sl" | "vertical"
    r: float              # result in R multiples (risk = |entry - stop|)
    mae_r: float          # maximum adverse excursion, in R
    mfe_r: float          # maximum favourable excursion, in R
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
    """Resolve a trade by walking the bars that come AFTER the entry.

    ``fine_highs``/``fine_lows``, when supplied, are the 1m bars inside each timeframe bar: they let
    us break the tie when target and stop land in the same bar.
    """
    risk = abs(entry - stop)
    if risk <= 0:
        raise ValueError("zero risk: the entry coincides with the stop")
    s = 1.0 if long else -1.0
    n = min(len(highs), max_bars)
    mae = mfe = 0.0

    for i in range(n):
        hi, lo = float(highs[i]), float(lows[i])
        fav = s * (hi - entry) if long else s * (entry - lo)
        adv = s * (entry - lo) if long else s * (hi - entry)
        mfe = max(mfe, fav / risk)
        mae = max(mae, adv / risk)

        hits_tp = hi >= target if long else lo <= target
        hits_sl = lo <= stop if long else hi >= stop

        if hits_tp and hits_sl:
            # Both inside the same bar: drop to 1m to find out which came first.
            first = _break_tie(entry, stop, target, long,
                               fine_highs[i] if fine_highs and i < len(fine_highs) else None,
                               fine_lows[i] if fine_lows and i < len(fine_lows) else None)
            if first == "tp":
                return Outcome("tp", abs(target - entry) / risk, mae, mfe, i + 1, False)
            if first == "sl":
                return Outcome("sl", -1.0, mae, mfe, i + 1, False)
            # No fine data: PESSIMISTIC. Assuming the profit would inflate every statistic.
            return Outcome("sl", -1.0, mae, mfe, i + 1, True)
        if hits_tp:
            return Outcome("tp", abs(target - entry) / risk, mae, mfe, i + 1, False)
        if hits_sl:
            return Outcome("sl", -1.0, mae, mfe, i + 1, False)

    # Vertical barrier: closed at market. A large fraction of trades ends this way, and it is
    # precisely why approximating this with a two-outcome Bernoulli is wrong.
    if n == 0:
        return Outcome("vertical", 0.0, 0.0, 0.0, 0, False)
    exit_px = float(closes[n - 1])
    return Outcome("vertical", s * (exit_px - entry) / risk, mae, mfe, n, False)


def _break_tie(entry: float, stop: float, target: float, long: bool,
               fh: np.ndarray | None, fl: np.ndarray | None) -> str | None:
    """Which one was touched first, looking inside the bar."""
    if fh is None or fl is None or len(fh) == 0:
        return None
    for j in range(len(fh)):
        hi, lo = float(fh[j]), float(fl[j])
        tp = hi >= target if long else lo <= target
        sl = lo <= stop if long else hi >= stop
        if tp and sl:
            return None          # not even 1m can tell: still ambiguous
        if tp:
            return "tp"
        if sl:
            return "sl"
    return None
