"""`volatility` family: compression and expansion of the range. PRE-REGISTRATION.

Written before running a single backtest. No parameter in this file was chosen by looking at
results: each one is the conventional value from the literature that defined it (Bollinger 20/2,
Keltner EMA-20 ± 2·ATR-10, TTM Squeeze 20/20/1.5, ATR-14, NATR-14, Chandelier 22/3, NR7, 252 as the
conventional annual window). That ATR-20, ATR-14, ATR-10 and ATR-22 appear in different hypotheses
is NOT a sweep over the ATR period: each one is fixed by the composite indicator that contains it,
and none of them moves.

The family's shared mechanism
-----------------------------
Volatility clustering (Mandelbrot 1963, Engle 1982): variance is autocorrelated even when the
return is not. In a 24/7 market with leveraged market makers, a quiet period compresses the
inventory of liquidity: MMs narrow their spreads, volatility sellers load up on short gamma and
everybody's stop orders pile up just outside an ever-narrower range. When the range breaks, the
hedging of that short gamma and the stop cascade push in the same direction: the big move is born
in the quiet period, not in the noisy one.

Realised volatility estimator
-----------------------------
Rogers-Satchell: RS_t = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O). Chosen over Garman-Klass because it is
unbiased under non-zero drift (BTC has strong drift and GK books it as variance), and over
Yang-Zhang because YZ devotes a term to the opening jump, which in a 24/7 asset with no session
does not exist: that term would be microstructure noise from the arbitrary candle boundary.

Deliberate contrasts (this is NOT a grid search)
-------------------------------------------------------
Three pairs registered on purpose, with the prediction of which one wins written in advance:

  1. `rs_vol_pct_low_carry` vs `bbw_pct_low_carry`: SAME direction rule, different compression
     estimator (RS over OHLC vs Bollinger width, which is the standard deviation of closes). If the
     "quiet precedes big" mechanism is real, the two should point the same way. If only one of them
     works, what is being measured is the estimator, not the mechanism.
  2. `rs_vol_pct_high_fade` vs `rs_vol_pct_high_carry`: exact complements. At most one of them can
     have an edge. Registering them together makes it impossible to choose the sign after looking.
  3. `natr_low_regime_trend` vs `natr_high_regime_trend`: a median split of the SAME trend signal.
     Together they reconstruct the unfiltered signal, so the comparison isolates the effect of the
     volatility regime and not that of the trend.

Causality
---------
Every indicator here uses only data up to and including i. Rolling windows are aligned with
`sliding_window_view(x, n)[j] == x[j : j+n]`, whose result is written at index `j+n-1`. There is no
negative shift, no `find_peaks`, and no statistic defined against the whole array. The stateful
constructions (`squeeze`, Keltner, Chandelier) are computed in forward loops, which are causal by
construction.
"""

from __future__ import annotations

import numpy as np
import talib
from numpy.lib.stride_tricks import sliding_window_view

from wavelab.hypotheses.base import Hypothesis, Series, register

# --------------------------------------------------------------------------------------------
# Utilities. None of them looks ahead.
# --------------------------------------------------------------------------------------------


def _ohlc(s: Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """talib requires contiguous float64."""
    f = lambda a: np.ascontiguousarray(a, dtype=np.float64)
    return f(s.open), f(s.high), f(s.low), f(s.close)


def _roll_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Mean of the last n observations, i's included. NaN if the window touches a NaN."""
    out = np.full(x.size, np.nan)
    if x.size >= n:
        out[n - 1:] = sliding_window_view(x, n).mean(axis=1)
    return out


def _rolling_rank(x: np.ndarray, n: int) -> np.ndarray:
    """Percentile of x[i] within the window x[i-n+1 : i+1]. Returns NaN during warm-up.

    The window ENDS at i, it never straddles it. A percentile computed against the whole array
    would be pure lookahead (it would know today what the maximum volatility of 2031 was), and that
    is exactly the mistake this project hunts.
    """
    out = np.full(x.size, np.nan)
    if x.size < n:
        return out
    win = sliding_window_view(x, n)                  # win[j] == x[j : j+n]  ->  index j+n-1
    # NaN count per window without propagating NaN (cumsum over integers).
    cs = np.concatenate(([0], np.cumsum(np.isnan(x).astype(np.int64))))
    valid = (cs[n:] - cs[:-n]) == 0
    last = win[:, -1]
    cnt = (win < last[:, None]).sum(axis=1)          # comparing with NaN gives False; dropped below
    out[n - 1:] = np.where(valid, cnt / (n - 1), np.nan)
    return out


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """Value from k candles ago. k > 0 ALWAYS: a negative k would be looking into the future."""
    if k <= 0:
        raise ValueError("_shift only looks backwards")
    out = np.full(x.size, np.nan)
    if k < x.size:
        out[k:] = x[: x.size - k]
    return out


def _rs_vol(s: Series, n: int) -> np.ndarray:
    """Rogers-Satchell realised volatility over the last n candles, i's included."""
    o, h, l, c = _ohlc(s)
    term = np.full(c.size, np.nan)
    pos = (o > 0) & (h > 0) & (l > 0) & (c > 0)
    if pos.any():
        oo, hh, ll, cc = o[pos], h[pos], l[pos], c[pos]
        term[pos] = np.log(hh / cc) * np.log(hh / oo) + np.log(ll / cc) * np.log(ll / oo)
    # RS is non-negative by construction (H >= max(O,C) and L <= min(O,C)); the clip only guards
    # against numerical noise. np.maximum propagates NaN, so the warm-up is preserved as NaN.
    return np.sqrt(_roll_mean(np.maximum(term, 0.0), n))


def _hold(entry: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """Holds the last entry until `flat` closes it. Forward loop: causal."""
    out = np.zeros(entry.size, dtype=np.int8)
    cur = 0
    for i in range(entry.size):
        if flat[i]:
            cur = 0
        e = int(entry[i])
        if e != 0:
            cur = e
        out[i] = cur
    return out


# --------------------------------------------------------------------------------------------
# 1. Bollinger squeeze inside Keltner (TTM Squeeze), traded on the release.
# --------------------------------------------------------------------------------------------


def _squeeze_release(s: Series) -> np.ndarray:
    o, h, l, c = _ohlc(s)
    up, mid, lo = talib.BBANDS(c, 20, 2.0, 2.0, 0)
    ema = talib.EMA(c, 20)
    atr = talib.ATR(h, l, c, 20)
    ok = ~(np.isnan(up) | np.isnan(lo) | np.isnan(mid) | np.isnan(ema) | np.isnan(atr))

    sq = np.zeros(c.size, dtype=bool)
    kc_up = ema + 1.5 * atr
    kc_lo = ema - 1.5 * atr
    sq[ok] = (up[ok] < kc_up[ok]) & (lo[ok] > kc_lo[ok])

    prev_sq = np.zeros(c.size, dtype=bool)
    prev_sq[1:] = sq[:-1]
    release = prev_sq & ~sq & ok

    entry = np.zeros(c.size, dtype=np.int8)
    entry[release & (c > mid)] = 1
    entry[release & (c < mid)] = -1

    out = np.zeros(c.size, dtype=np.int8)
    cur = 0
    for i in range(c.size):
        if cur == 1 and (sq[i] or (ok[i] and c[i] < mid[i])) or cur == -1 and (sq[i] or (ok[i] and c[i] > mid[i])):
            cur = 0
        e = int(entry[i])
        if e != 0:
            cur = e
        out[i] = cur
    return out


register(Hypothesis(
    name="volatility.squeeze_bb_kc_release",
    family="volatility",
    rationale="When the Bollinger bands (20, 2σ) fit inside the Keltner channel "
              "(EMA-20 ± 1.5·ATR-20), the dispersion of closes has sunk below the true range: the "
              "market is trading in a handkerchief while there is still intra-candle travel. That "
              "state is manufactured by market makers and volatility sellers who narrow spreads "
              "and accumulate short gamma, and by range traders who place stops just outside the "
              "handkerchief. When price leaves, those same participants have to hedge in the "
              "direction of the move and the stops execute at market, so the flow that follows the "
              "break is forced, not discretionary.",
    prior="We expect a positive edge in the direction of the break relative to the 20-period "
          "moving average, concentrated in the first candles after the release. We expect it to "
          "FAIL, with a negative edge, when the compression happens inside a wide higher-order "
          "range (a false break that reverts to the middle) and on 15m, where the 'squeeze' is "
          "often an overnight liquidity hole rather than an accumulation of positioning. If the "
          "edge turned out to be indistinguishable from zero on all three timeframes, the "
          "short-gamma mechanism is not operating in BTC at candle resolution.",
    fn=_squeeze_release,
    params={"bb_period": 20, "bb_sigma": 2.0, "kc_period": 20, "kc_atr": 20, "kc_mult": 1.5},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 2 and 3. Compression measured two different ways, SAME direction rule.
# --------------------------------------------------------------------------------------------


def _rs_vol_pct_low_carry(s: Series) -> np.ndarray:
    _, _, _, c = _ohlc(s)
    rank = _rolling_rank(_rs_vol(s, 24), 252)
    ema = talib.EMA(c, 55)
    out = np.zeros(c.size, dtype=np.int8)
    quiet = ~np.isnan(rank) & ~np.isnan(ema) & (rank < 0.20)
    out[quiet & (c > ema)] = 1
    out[quiet & (c < ema)] = -1
    return out


register(Hypothesis(
    name="volatility.rs_vol_pct_low_carry",
    family="volatility",
    rationale="Rogers-Satchell realised volatility over 24 candles, placed in its 252-period "
              "rolling percentile, measures whether the market is quiet RELATIVE TO ITSELF rather "
              "than relative to an absolute threshold that BTC's price inflation would render "
              "obsolete. With volatility in the bottom quintile, whoever is positioned is not "
              "being shaken out by noise: the cost of holding a trend position falls, forced "
              "liquidations stop and the drift accumulates without interruption. We give the "
              "direction with price against the EMA-55, deliberately trivial, because what is "
              "being tested here is the volatility GATE, not the trend detector.",
    prior="We expect a positive edge while volatility is in the low quintile, and for that edge to "
          "be GREATER than that of the same direction rule unfiltered. We expect it to fail in two "
          "specific places: (a) after a structural fall in the level of volatility (a prolonged "
          "range), where the percentile stays pinned at the bottom and the signal trades a range "
          "with no trend; (b) on 1d, where 252 candles are a whole year and the percentile drags "
          "an old regime along with it. If the edge were equal to or smaller than that of the "
          "unfiltered EMA-55, the volatility gate adds nothing. AUDIT NOTE (2026-09-08): "
          "that unfiltered arm was not registered, so neither this hypothesis's falsification "
          "criterion nor `bbw_pct_low_carry`'s could be run; it has been registered as "
          "`volatility.ema55_side_unfiltered` and that is what both must be measured against.",
    fn=_rs_vol_pct_low_carry,
    params={"rs_window": 24, "percentile_window": 252, "threshold": 0.20, "ema_direction": 55},
    timeframes=("1h", "4h", "1d"),
    min_warmup=300,
))


def _bbw_pct_low_carry(s: Series) -> np.ndarray:
    _, _, _, c = _ohlc(s)
    up, mid, lo = talib.BBANDS(c, 20, 2.0, 2.0, 0)
    bw = np.full(c.size, np.nan)
    good = ~(np.isnan(up) | np.isnan(lo) | np.isnan(mid)) & (mid != 0)
    bw[good] = (up[good] - lo[good]) / mid[good]
    rank = _rolling_rank(bw, 252)
    ema = talib.EMA(c, 55)
    out = np.zeros(c.size, dtype=np.int8)
    quiet = ~np.isnan(rank) & ~np.isnan(ema) & (rank < 0.20)
    out[quiet & (c > ema)] = 1
    out[quiet & (c < ema)] = -1
    return out


register(Hypothesis(
    name="volatility.bbw_pct_low_carry",
    family="volatility",
    rationale="A controlled contrast to the previous one: identical direction rule (price against "
              "the EMA-55), identical percentile window (252), identical threshold (bottom "
              "quintile), and the ONLY thing that changes is the compression estimator: Bollinger "
              "width, which is the standard deviation of CLOSES, against Rogers-Satchell, which "
              "uses the full OHLC range. The two can diverge a great deal: a candle with a long "
              "wick and a flat close (an absorbed liquidation) is quiet for Bollinger and noisy "
              "for RS. If the clustering mechanism is real, it should show up with both "
              "estimators.",
    prior="We expect the SAME sign as `rs_vol_pct_low_carry` and a somewhat smaller magnitude, "
          "because the close-based estimator throws away the information in the wicks. The strong "
          "falsifiable prediction is the joint one: if one of the two gives a clear edge and the "
          "other gives zero or the opposite sign, the correct conclusion is NOT 'compression "
          "works' but that we are measuring a quirk of the estimator, and neither of the two "
          "should survive the multiple-comparisons correction as evidence for the mechanism.",
    fn=_bbw_pct_low_carry,
    params={"bb_period": 20, "bb_sigma": 2.0, "percentile_window": 252, "threshold": 0.20,
            "ema_direction": 55},
    timeframes=("1h", "4h", "1d"),
    min_warmup=300,
))


# --------------------------------------------------------------------------------------------
# 4 and 5. High volatility: exact complements. At most one of them can have an edge.
# --------------------------------------------------------------------------------------------


def _rs_vol_pct_high_fade(s: Series) -> np.ndarray:
    _, _, _, c = _ohlc(s)
    rank = _rolling_rank(_rs_vol(s, 24), 252)
    prev = _shift(c, 3)
    out = np.zeros(c.size, dtype=np.int8)
    hot = ~np.isnan(rank) & ~np.isnan(prev) & (rank > 0.90)
    out[hot & (c > prev)] = -1
    out[hot & (c < prev)] = 1
    return out


register(Hypothesis(
    name="volatility.rs_vol_pct_high_fade",
    family="volatility",
    rationale="In the top decile of realised volatility, the move of the last 3 candles is "
              "dominated by forced liquidations: the exchange's risk engine sells (or buys) at "
              "market without looking at the price, and whoever provides liquidity against that "
              "cascade demands a premium. If that premium is the large part of the displacement, "
              "price comes back once the cascade exhausts the available collateral, and the "
              "liquidity provider pockets the retracement. This hypothesis bets that the excess is "
              "a liquidity premium, not information.",
    prior="We expect a positive edge from fading the 3-candle move. We expect it to FAIL, with a "
          "clearly negative edge, if high volatility in BTC is informative rather than mechanical: "
          "in March 2020 or in the May 2021 deleveraging the cascade ran for days. Since it is "
          "registered alongside its exact complement `rs_vol_pct_high_carry`, they cannot both "
          "win; if both come out indistinguishable from zero, the top volatility decile simply "
          "contains no directional signal and both must be published as failures.",
    fn=_rs_vol_pct_high_fade,
    params={"rs_window": 24, "percentile_window": 252, "threshold": 0.90, "impulse_bars": 3},
    timeframes=("1h", "4h", "1d"),
    min_warmup=300,
))


def _rs_vol_pct_high_carry(s: Series) -> np.ndarray:
    _, _, _, c = _ohlc(s)
    rank = _rolling_rank(_rs_vol(s, 24), 252)
    prev = _shift(c, 3)
    out = np.zeros(c.size, dtype=np.int8)
    hot = ~np.isnan(rank) & ~np.isnan(prev) & (rank > 0.90)
    out[hot & (c > prev)] = 1
    out[hot & (c < prev)] = -1
    return out


register(Hypothesis(
    name="volatility.rs_vol_pct_high_carry",
    family="volatility",
    rationale="The opposite reading of the same state: in the top decile of realised volatility, "
              "forced liquidation is reflexive. Each liquidation moves price towards the next "
              "cluster of collateral, which liquidates in turn; the book empties out in the "
              "direction of the move and market makers pull their quotes instead of absorbing. "
              "Under that mechanism, the excess volatility is itself the signal that the cascade "
              "is still alive, and continuation dominates the retracement over a horizon of a few "
              "candles.",
    prior="We expect a positive edge in the direction of the 3-candle move, but a SMALL one in net "
          "terms, because it trades exactly when spread and slippage "
          "are at their worst: it is a prediction that can be right in sign and still not survive "
          "costs, and that is how it must be evaluated. We expect it to fail at capitulation "
          "highs, where the last candle of the top decile is precisely the turn. It is the exact "
          "complement of `rs_vol_pct_high_fade`: registering them together makes it impossible to "
          "choose the sign after looking.",
    fn=_rs_vol_pct_high_carry,
    params={"rs_window": 24, "percentile_window": 252, "threshold": 0.90, "impulse_bars": 3},
    timeframes=("1h", "4h", "1d"),
    min_warmup=300,
))


# --------------------------------------------------------------------------------------------
# 6, 7, 8. Range contraction and expansion, at the candle level. Horizon of ONE candle: we do not
# introduce a position-duration parameter that would then have to be justified.
# --------------------------------------------------------------------------------------------


def _nr7_breakout(s: Series) -> np.ndarray:
    o, h, l, c = _ohlc(s)
    tr = talib.TRANGE(h, l, c)
    min7 = talib.MIN(tr, 7)
    is_nr7 = np.where(np.isnan(tr) | np.isnan(min7), np.nan, (tr <= min7).astype(float))
    nr7_prev = _shift(is_nr7, 1)
    hi_prev, lo_prev = _shift(h, 1), _shift(l, 1)
    out = np.zeros(c.size, dtype=np.int8)
    ok = ~(np.isnan(nr7_prev) | np.isnan(hi_prev) | np.isnan(lo_prev)) & (nr7_prev == 1.0)
    out[ok & (c > hi_prev)] = 1
    out[ok & (c < lo_prev)] = -1
    return out


register(Hypothesis(
    name="volatility.nr7_breakout",
    family="volatility",
    rationale="Toby Crabel's NR7: the previous candle had the narrowest true range of the last "
              "seven. A local range minimum means buyers and sellers agreed on the price for a "
              "whole period, which concentrates the resting orders (stops and limits) into a very "
              "thin band. The next candle closing outside that band consumes those orders in one "
              "go, and the resulting imbalance is mechanical. It is the minimal version of the "
              "family's mechanism: contraction measured with no estimators and no long windows, "
              "only the range over seven candles.",
    prior="We expect a positive edge in the direction of the break, with a horizon of a single "
          "candle. We expect it to be LARGER on 15m and 1h, where the book of resting orders is "
          "significant relative to typical size, and to degrade or disappear on 4h, where a candle "
          "aggregates too many hands for any exploitable imbalance to be left. If the edge grew "
          "with the timeframe we would be measuring trend, not range contraction, and the "
          "hypothesis would be refuted even if the number were good.",
    fn=_nr7_breakout,
    params={"nr_window": 7, "horizon_bars": 1},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


def _inside_bar_breakout(s: Series) -> np.ndarray:
    o, h, l, c = _ohlc(s)
    h1, l1 = _shift(h, 1), _shift(l, 1)
    h2, l2 = _shift(h, 2), _shift(l, 2)
    out = np.zeros(c.size, dtype=np.int8)
    ok = ~(np.isnan(h1) | np.isnan(l1) | np.isnan(h2) | np.isnan(l2))
    inside = ok & (h1 < h2) & (l1 > l2)
    out[inside & (c > h1)] = 1
    out[inside & (c < l1)] = -1
    return out


register(Hypothesis(
    name="volatility.inside_bar_breakout",
    family="volatility",
    rationale="Inside bar: the range of the previous candle is STRICTLY contained within that of "
              "the one before it. It is not the same thing as NR7, and that is why it is "
              "registered separately: NR7 is narrowness relative to a sample of seven, the inside "
              "bar is a condition of absolute containment relative to ONE specific candle, the "
              "mother. Containment means no participant managed to impose a price outside the "
              "range that had already been explored and rejected: equilibrium has reasserted "
              "itself within a known range, with the orders anchored to the extremes of the inside "
              "candle. Breaking them on a close invalidates that reassertion.",
    prior="We expect a positive edge, of the same sign as `nr7_breakout` and of similar magnitude. "
          "The comparison between the two is informative in itself: if absolute containment works "
          "and relative narrowness does not (or the other way round), what we have is a quirk of "
          "the criterion and not the contraction mechanism. We expect it to fail in markets with "
          "strong sustained drift, where the inside bar is only a continuation pause and the break "
          "already arrives late, and on 15m during the lowest-volume hours, where the containment "
          "reflects an absence of participants rather than agreement among them. "
          "AUDIT NOTE (2026-09-08): `candles.inside_bar_break` registers the same two-candle "
          "pattern but breaking the extremes of the MOTHER candle. Since the inside candle's high "
          "is lower than the mother's, every event of the former is also an event of the latter: "
          "they are the same pattern with two different thresholds. Both are kept because the "
          "level is the claim and these are different levels, but they are DEPENDENT trials and "
          "must be counted as such. (A third registration of the same pattern, "
          "`structure.inside_bar_break`, was an exact duplicate of the `candles` one and has been "
          "removed in the audit.)",
    fn=_inside_bar_breakout,
    params={"horizon_bars": 1},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


def _wide_range_thrust(s: Series) -> np.ndarray:
    o, h, l, c = _ohlc(s)
    tr = talib.TRANGE(h, l, c)
    atr = talib.ATR(h, l, c, 14)
    atr_prev = _shift(atr, 1)
    rank_prev = _shift(_rolling_rank(atr, 100), 1)
    out = np.zeros(c.size, dtype=np.int8)
    ok = ~(np.isnan(tr) | np.isnan(atr_prev) | np.isnan(rank_prev))
    thrust = ok & (rank_prev < 0.50) & (tr > 2.0 * atr_prev)
    out[thrust & (c > o)] = 1
    out[thrust & (c < o)] = -1
    return out


register(Hypothesis(
    name="volatility.wide_range_thrust",
    family="volatility",
    rationale="Range expansion AFTER contraction, which is the direct formulation of 'big moves "
              "are born in quiet periods': we require the previous candle's ATR-14 to have been "
              "below its 100-candle median (contraction) and the current true range to exceed "
              "twice that ATR (expansion). A candle like that, coming out of a quiet base, is not "
              "produced by dispersed retail flow: it is produced by a large participant who needs "
              "to execute and accepts paying the range, or by a cluster of stops being triggered. "
              "The direction is given by the candle's body, close against open, because that is "
              "what reveals who won the exchange.",
    prior="We expect a positive edge in the direction of the body, with a horizon of one candle. "
          "We expect it to FAIL clearly when the thrust is a reaction to a one-off news item that "
          "is entirely retraced over the following candles (a typical pattern in BTC with "
          "regulatory announcements), and on 1d, where an expansion candle is usually already the "
          "end of the move and not its beginning. If the edge were consistently negative, the "
          "correct mechanism would be exhaustion and not the start of an expansion.",
    fn=_wide_range_thrust,
    params={"atr_period": 14, "expansion_mult": 2.0, "median_window": 100, "horizon_bars": 1},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 9 and 10. Volatility-scaled structures: the threshold moves with the ATR, not with the price.
# --------------------------------------------------------------------------------------------


def _keltner_breakout(s: Series) -> np.ndarray:
    o, h, l, c = _ohlc(s)
    ema = talib.EMA(c, 20)
    atr = talib.ATR(h, l, c, 10)
    ok = ~(np.isnan(ema) | np.isnan(atr))
    up, lo = ema + 2.0 * atr, ema - 2.0 * atr
    entry = np.zeros(c.size, dtype=np.int8)
    entry[ok & (c > up)] = 1
    entry[ok & (c < lo)] = -1
    out = np.zeros(c.size, dtype=np.int8)
    cur = 0
    for i in range(c.size):
        if cur == 1 and ok[i] and c[i] < ema[i] or cur == -1 and ok[i] and c[i] > ema[i]:
            cur = 0
        e = int(entry[i])
        if e != 0:
            cur = e
        out[i] = cur
    return out


register(Hypothesis(
    name="volatility.keltner_breakout",
    family="volatility",
    rationale="A volatility-normalised breakout with no compression component: a close above "
              "EMA-20 + 2·ATR-10, or below EMA-20 − 2·ATR-10. Under a random walk with the CURRENT "
              "volatility, getting two ATR away from the mean is rare; that it happens is evidence "
              "that flow has arrived which the prevailing volatility level does not explain — that "
              "is, evidence of a change in the drift and not of a fluctuation. The threshold "
              "rescales itself, so the same rule is equally demanding at 20,000 as at 100,000 "
              "dollars. It is held until the close crosses back over the EMA-20, which is the "
              "minimal definition of 'the flow is over'.",
    prior="We expect a positive edge and, above all, an ASYMMETRY with respect to the squeeze: if "
          "`squeeze_bb_kc_release` beats this hypothesis, the prior compression adds information "
          "over and above the break alone, which is the family's central claim. If this one wins "
          "instead, the compression is decorative and the only thing that works is the "
          "volatility-normalised break. "
          "We expect it to fail in a mid-volatility range, where price crosses the two ATR in both "
          "directions repeatedly and each crossing costs a spread.",
    fn=_keltner_breakout,
    params={"ema_period": 20, "atr_period": 10, "mult": 2.0},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


def _chandelier_trail(s: Series) -> np.ndarray:
    o, h, l, c = _ohlc(s)
    atr = talib.ATR(h, l, c, 22)
    hh = talib.MAX(h, 22)
    ll = talib.MIN(l, 22)
    ok = ~(np.isnan(atr) | np.isnan(hh) | np.isnan(ll))
    long_stop = hh - 3.0 * atr
    short_stop = ll + 3.0 * atr

    # FIXED IN THE AUDIT (2026-09-08). This used to be:
    #     entry[ok & (c > short_stop)] = 1
    #     entry[ok & (c < long_stop)] = -1
    # and the two assignments trampled each other. When the 22-candle range is wide (hh - ll >
    # 6·ATR) the long threshold ends up ABOVE the short one, so a close in the zone between them
    # satisfied BOTH conditions at once; since the -1 was written second, the short always won.
    # Measured over a series containing all three regimes, that affected 2.7% of candles and forced
    # every one of them short.
    #
    # Why it mattered, and why it is not a detail. The comment on this very function declared the
    # opposite of what the code did ("when the close falls between the two thresholds there is no
    # new information: the state is held"), and the bias was not random: the ambiguous zone appears
    # precisely in wide ranges, that is, in the regime where this hypothesis's prior predicts
    # LOSSES. An arbitrary tie-break that systematically goes short exactly in the regime you are
    # trying to measure contaminates the sign of the result and makes it uninterpretable: you could
    # not tell whether what was measured was the ATR scaling or the tie-break. The ambiguous zone
    # now emits 0 —no new entry— and `_hold` carries the previous state forward, which is the
    # Chandelier rule as declared in the `rationale` and what `_events` does in the `structure`
    # family.
    entry = np.zeros(c.size, dtype=np.int8)
    long_side = ok & (c > short_stop)
    short_side = ok & (c < long_stop)
    ambiguous = long_side & short_side
    entry[long_side & ~ambiguous] = 1
    entry[short_side & ~ambiguous] = -1
    return _hold(entry, np.zeros(c.size, dtype=bool))


register(Hypothesis(
    name="volatility.chandelier_atr_trail",
    family="volatility",
    rationale="The Chandelier exit (Chuck LeBeau, 22/3): long while the close is above the "
              "22-candle low plus 3·ATR-22, short while it is below the 22-candle high minus "
              "3·ATR-22. The volatility content lies in the invalidation distance being scaled by "
              "the ATR: the position survives exactly the noise the current regime produces and "
              "not a bit more. It is the operational translation of volatility clustering: if "
              "variance is autocorrelated, tomorrow's noise is well estimated by today's, and a "
              "threshold fixed as a percentage would be too close in an agitated regime and too "
              "far in a quiet one.",
    prior="We expect a positive edge in a trending regime and a NEGATIVE one in a range: the "
          "signal is always in the market, so in a range it pays for every turn. The falsifiable "
          "claim specific to this family is not 'trend following works' —that is what the trend "
          "family tests— but that SCALING the threshold by the ATR beats the same structure with a "
          "fixed threshold — an arm that was NOT registered and that the 2026-09-08 audit has "
          "added as `volatility.chandelier_fixed_pct`, without which this criterion could not be "
          "run. If the net result across the whole history were indistinguishable from zero, the "
          "correct conclusion is that in BTC the gain from the trends exactly offsets the cost of "
          "the ranges, and there is no edge attributable to volatility scaling.",
    fn=_chandelier_trail,
    params={"window": 22, "atr_period": 22, "mult": 3.0},
    timeframes=("4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 11 and 12. Median split on NATR over the SAME trend signal.
# --------------------------------------------------------------------------------------------


def _natr_regime_trend(s: Series, high_vol: bool) -> np.ndarray:
    _, h, l, c = _ohlc(s)
    natr = talib.NATR(h, l, c, 14)
    rank = _rolling_rank(natr, 252)
    e21, e55 = talib.EMA(c, 21), talib.EMA(c, 55)
    out = np.zeros(c.size, dtype=np.int8)
    ok = ~(np.isnan(rank) | np.isnan(e21) | np.isnan(e55))
    gate = ok & ((rank >= 0.50) if high_vol else (rank < 0.50))
    out[gate & (e21 > e55)] = 1
    out[gate & (e21 < e55)] = -1
    return out


def _natr_low_regime_trend(s: Series) -> np.ndarray:
    return _natr_regime_trend(s, high_vol=False)


def _natr_high_regime_trend(s: Series) -> np.ndarray:
    return _natr_regime_trend(s, high_vol=True)


register(Hypothesis(
    name="volatility.natr_low_regime_trend",
    family="volatility",
    rationale="The normalised ATR (NATR-14, the ATR as a percentage of price) placed in its "
              "252-period rolling percentile classifies the regime without depending on the price "
              "level. Below its median, the typical travel per period is small relative to the "
              "position size anyone can hold: there are no stop sweeps and no liquidation chains, "
              "so whoever is on the right side of the EMA-21/EMA-55 is not shaken out before the "
              "drift materialises. The trend signal is a fixed instrument here; what is being "
              "tested is the regime.",
    prior="We expect a positive edge, and a greater one than that of the same unfiltered "
          "EMA-21/55 signal. We expect it to fail if low volatility in BTC corresponds mainly to "
          "long accumulation ranges rather than to smooth trends: in that case the moving-average "
          "cross will give strings of false signals and the edge will be negative. It is the low "
          "half of the median split: together with `natr_high_regime_trend` it reconstructs the "
          "unfiltered signal, so the two cannot both be better than it.",
    fn=_natr_low_regime_trend,
    params={"natr_period": 14, "percentile_window": 252, "cutoff": 0.50, "ema_fast": 21,
            "ema_slow": 55},
    timeframes=("1h", "4h", "1d"),
    min_warmup=300,
))


register(Hypothesis(
    name="volatility.natr_high_regime_trend",
    family="volatility",
    rationale="The complementary half, registered so that the previous claim is falsifiable and "
              "not a selection made after the fact. Above the NATR-14 median, the same trend "
              "signal trades when the typical travel is large: a single candle can cover the "
              "distance between the two averages several times over, so the cross happens and "
              "unwinds through amplitude noise, and on top of that spread and slippage are at "
              "their worst precisely there.",
    prior="We expect a NIL OR NEGATIVE edge here, and a positive one in `natr_low_regime_trend`. "
          "That is the contrast, and we are committed to that sign in advance. If it came out the "
          "other way round —trend paying in high volatility and not in low— the mechanism we have "
          "described for this entire family would be wrong: it would mean that in BTC high "
          "volatility accompanies directional trend instead of destroying it, and the compression "
          "hypotheses in this file would have to be read with far more suspicion even if one of "
          "them gave a favourable number.",
    fn=_natr_high_regime_trend,
    params={"natr_period": 14, "percentile_window": 252, "cutoff": 0.50, "ema_fast": 21,
            "ema_slow": 55},
    timeframes=("1h", "4h", "1d"),
    min_warmup=300,
))


# --------------------------------------------------------------------------------------------
# 13 and 14. CONTROLS ADDED IN THE AUDIT (2026-09-08).
#
# What was wrong. Three hypotheses in this file declared their falsification criterion against a
# control that was NOT registered anywhere in the pre-registration:
#
#   - `rs_vol_pct_low_carry` and `bbw_pct_low_carry` say their edge must be "GREATER than that of
#     the same direction rule unfiltered" and that "if the edge were equal to or smaller than that
#     of the unfiltered EMA-55, the volatility gate adds nothing". Price against the EMA-55 with no
#     filter did not exist as a registered hypothesis.
#   - `chandelier_atr_trail` says explicitly that "the falsifiable claim specific to this family
#     [...] is that SCALING the threshold by the ATR beats the same structure with a fixed
#     threshold". The structure with a fixed threshold did not exist.
#
# Why it mattered. A prior whose failure criterion names an object that does not exist cannot be
# run, and what cannot be run cannot fail: in practice those three hypotheses could only be judged
# on their absolute profitability, which is exactly the judgement the pre-registration exists to
# forbid (in an asset that went up twentyfold, any predominantly long rule "wins"). The two missing
# controls are registered here, with their own priors and counting as two more trials in the
# multiple-comparisons correction. Widening the number of trials —which is the conservative
# direction— is preferred to watering down falsification criteria that were already written, which
# would amount to rewriting the registry after closing it.
# --------------------------------------------------------------------------------------------


def _ema55_side_unfiltered(s: Series) -> np.ndarray:
    _, _, _, c = _ohlc(s)
    ema = talib.EMA(c, 55)
    out = np.zeros(c.size, dtype=np.int8)
    ok = ~np.isnan(ema)
    out[ok & (c > ema)] = 1
    out[ok & (c < ema)] = -1
    return out


register(Hypothesis(
    name="volatility.ema55_side_unfiltered",
    family="volatility",
    rationale="Unconditional control for `rs_vol_pct_low_carry` and `bbw_pct_low_carry`: exactly "
              "their same direction rule —close above or below the EMA-55— with no volatility gate "
              "at all. It claims no mechanism of its own, and that is the point: the two "
              "compression hypotheses do not bet that following the EMA-55 wins, but that it wins "
              "MORE when realised volatility is in the bottom quintile. That is a claim about a "
              "difference, and a difference cannot be measured with a single arm. It is registered "
              "in `volatility` and not in `trend` because its only reason to exist is to serve as "
              "the denominator for this family's volatility gate.",
    prior="We expect a small positive edge, dominated by BTC's secular drift, and we expect it to "
          "be SMALLER than `rs_vol_pct_low_carry`'s. That comparison is the whole content: if this "
          "control matches or beats the two filtered versions, the volatility gate adds nothing "
          "and both are refuted even if their absolute number is good; if the two filtered ones "
          "beat it, the volatility-clustering claim holds. On its own, a positive result for this "
          "control is NOT a finding: it is the asset's drift split by which side of a moving "
          "average you are on, and it must be published as such.",
    fn=_ema55_side_unfiltered,
    params={"ema_direction": 55, "volatility_gate": "none"},
    timeframes=("1h", "4h", "1d"),
    min_warmup=300,
))


def _chandelier_fixed_pct(s: Series) -> np.ndarray:
    """Identical to `_chandelier_trail` except that the buffer is a fixed 6% of the level instead
    of 3·ATR-22. The 6% was not searched for: it is the value that matches 3·ATR-22 in a typical
    BTC volatility regime on 4h, chosen in advance so that the control differs from the hypothesis
    in the SCALING and not in the average aggressiveness of the threshold."""
    _, h, low, c = _ohlc(s)
    hh = talib.MAX(h, 22)
    ll = talib.MIN(low, 22)
    ok = ~(np.isnan(hh) | np.isnan(ll))
    long_stop = hh * (1.0 - 0.06)
    short_stop = ll * (1.0 + 0.06)
    entry = np.zeros(c.size, dtype=np.int8)
    long_side = ok & (c > short_stop)
    short_side = ok & (c < long_stop)
    ambiguous = long_side & short_side    # same tie-break rule as the hypothesis it controls
    entry[long_side & ~ambiguous] = 1
    entry[short_side & ~ambiguous] = -1
    return _hold(entry, np.zeros(c.size, dtype=bool))


register(Hypothesis(
    name="volatility.chandelier_fixed_pct",
    family="volatility",
    rationale="Control for `chandelier_atr_trail`. Exactly the same structure —long while the "
              "close is above the 22-candle low plus a buffer, short while it is below the "
              "22-candle high minus that buffer— with a single difference: the buffer is a FIXED "
              "percentage of the level instead of 3·ATR-22. It is the arm that makes the family's "
              "central claim testable, which is not 'trend following works' but that measuring the "
              "invalidation distance in units of the PREVAILING volatility beats measuring it in "
              "units of price. If volatility clustering is real, the scaled threshold should be "
              "too close in an agitated regime and too far in a quiet one exactly when the fixed "
              "one gets it wrong, and not before.",
    prior="We expect a positive edge and a SMALLER one than `chandelier_atr_trail`'s, with the "
          "difference concentrated in the volatility regime changes —the months after March 2020 "
          "and May 2021— and practically nil over the rest of the sample. If the two turn out "
          "indistinguishable, the ATR scaling buys nothing and `chandelier_atr_trail` is refuted "
          "in its own claim even if it makes money; if this control turns out BETTER, volatility "
          "clustering operates in the opposite direction to the one described in this family's "
          "header and every compression hypothesis in this file must be reread with suspicion. The "
          "6% is frozen: if it fails, no other value is tried.",
    fn=_chandelier_fixed_pct,
    params={"window": 22, "buffer_pct": 0.06, "scaling": "none"},
    timeframes=("4h", "1d"),
    min_warmup=200,
))
