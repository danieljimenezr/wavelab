"""MOMENTUM family: time-series momentum on BTC.

PRE-REGISTERED. Written on 2026-09-08, before running a single backtest and without having looked
at any result. The parameters are the conventional ones from the literature and they are FROZEN: if
a hypothesis fails it is not retried with another value. All twelve count towards the
multiple-comparisons correction, including —above all— the ones we expect to fail.

Mechanisms postulated across the whole family, in order of importance:

  1. Underreaction. Information diffuses in layers (OTC desks and treasuries → derivatives →
     retail) over days or months. The price adjusts with a lag and that adjustment has a persistent
     sign.
  2. Herding. The arrival of new participants is self-reinforcing for as long as there is marginal
     money left to come in, and it stops dead when there is not.
  3. Systematic following. Trend funds and momentum-indexed products rebalance over known, public
     windows, with orders that are large, slow and directional.

Causality note: every statistic in this file is computed backwards. The indices of previous
extremes (`_prior_extreme`) are bounded by construction at `i - _DIV_SEPARATION`, and the shifts
(`_back`) reject k <= 0 with an exception rather than with a comment.
"""

from __future__ import annotations

import numpy as np
import talib
from numpy.lib.stride_tricks import sliding_window_view

from wavelab.hypotheses.base import Hypothesis, Series, register

# --------------------------------------------------------------------------------------------
# Parameters. ONE value per concept, fixed a priori. Not searched over.
# --------------------------------------------------------------------------------------------
_TSMOM_LONG = 360       # ~12 months in daily candles
_TSMOM_SKIP = 30        # ~1 month, the one that gets skipped
_TSMOM_SHORT = 30       # 1 month  (multi-horizon consensus)
_TSMOM_MID = 90         # 3 months
_TSMOM_SEMI = 180       # 6 months
_ROC_SHORT = 10
_ROC_LONG = 20
_AO_FAST = 5            # Bill Williams' canonical Awesome Oscillator
_AO_SLOW = 34
_DIV_WINDOW = 100       # window in which the previous extreme is looked for
_DIV_SEPARATION = 10    # minimum separation between the previous extreme and the current bar
_LINREG = 20
_PERSIST_WINDOW = 20
_PERSIST_HIGH = 14      # 70 % of the window's candles with the same sign
_ER_PERIOD = 10         # canonical window of Kaufman's efficiency ratio
_ER_THRESHOLD = 0.30    # "efficient path" threshold. Fixed a priori, NOT tuned.
_KAMA_ER = 10           # TA-Lib fixes fast=2 and slow=30 internally
_VOL_SHORT = 5
_VOL_LONG = 20


# --------------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------------
def _f(x: np.ndarray) -> np.ndarray:
    """Contiguous float64, the only thing TA-Lib accepts."""
    return np.ascontiguousarray(x, dtype=np.float64)


def _base(s: Series, minimum: int) -> tuple[np.ndarray | None, np.ndarray]:
    """(closes, output). `closes` is None if the series is too short or unusable."""
    out = np.zeros(len(s), dtype=np.int8)
    c = _f(s.close)
    if c.size < minimum or not np.isfinite(c).any():
        return None, out
    return c, out


def _back(x: np.ndarray, k: int) -> np.ndarray:
    """x[i-k], with NaN in the first k positions. k>0 always: looking ahead is a bug."""
    if k <= 0:
        raise ValueError(f"_back requires k>0 (causality); got {k}")
    out = np.full(x.size, np.nan, dtype=np.float64)
    if k < x.size:
        out[k:] = x[:-k]
    return out


def _ao(s: Series) -> np.ndarray:
    """Awesome Oscillator: SMA(5) - SMA(34) of the median price (H+L)/2."""
    hl = (_f(s.high) + _f(s.low)) / 2.0
    return talib.SMA(hl, _AO_FAST) - talib.SMA(hl, _AO_SLOW)


def _prior_extreme(x: np.ndarray, find_max: bool) -> np.ndarray:
    """Index of x's extreme within [i-_DIV_WINDOW, i-_DIV_SEPARATION].

    Strictly causal: the index returned for bar i never exceeds i-_DIV_SEPARATION.
    -1 where there is no complete window.
    """
    n = x.size
    j = np.full(n, -1, dtype=np.int64)
    width = _DIV_WINDOW - _DIV_SEPARATION + 1
    if n < _DIV_WINDOW + 1:
        return j
    v = sliding_window_view(x, width)            # v[k] = x[k : k+width]
    rel = np.argmax(v, axis=1) if find_max else np.argmin(v, axis=1)
    i = np.arange(_DIV_WINDOW, n)
    k = i - _DIV_WINDOW                          # window = x[i-100 : i-9]
    j[i] = k + rel[k]
    return j


# --------------------------------------------------------------------------------------------
# 1. 12-1 momentum
# --------------------------------------------------------------------------------------------
def _tsmom_12_1(s: Series) -> np.ndarray:
    c, out = _base(s, _TSMOM_LONG + _TSMOM_SKIP + 1)
    if c is None:
        return out
    recent = _back(c, _TSMOM_SKIP)               # close of ~1 month ago
    old = _back(c, _TSMOM_LONG)                  # close of ~12 months ago
    ok = np.isfinite(recent) & np.isfinite(old) & (old > 0)
    ret = np.full(c.size, np.nan)
    ret[ok] = recent[ok] / old[ok] - 1.0
    out[np.isfinite(ret) & (ret > 0)] = 1
    out[np.isfinite(ret) & (ret < 0)] = -1
    return out


register(Hypothesis(
    name="momentum.tsmom_12_1",
    family="momentum",
    rationale="Systematic trend funds and momentum-indexed products rebuild their portfolios over a "
              "12-month window; their rebalancing orders are large, slow and directional, and they "
              "push the price in the direction of the past return. The last month is excluded "
              "because at that horizon what dominates is the reversion that follows liquidation "
              "cascades, together with market-maker inventory, which has the opposite sign and "
              "contaminates the measurement. In BTC the same behaviour arrives via corporate "
              "treasuries, spot ETF flow and retail chasing the 'twelve-month high' headline.",
    prior="We expect a small, persistent positive edge on the daily, concentrated in 2017, 2020-21 "
          "and 2023-24, and a NEGATIVE edge in the 3-6 months following a cycle top (2018, 2022), "
          "when the signal stays long throughout the fall. It is falsified if the average edge is "
          "≤0 within the trending regime itself: that would mean that not even when the mechanism "
          "ought to act does it act. "
          "AUDIT NOTE (2026-09-08): `trend.tsmom_365d` registers the sign of the 365-day return on "
          "1d, that is, this very same bet WITHOUT skipping the last month. Neither of the two "
          "files declared the other. Neither is removed —the one-month skip is the substantive part "
          "of the 12-1 hypothesis and the two signal series differ enough for the comparison to "
          "have content— but let it be written down that they are DEPENDENT trials over the same "
          "horizon and the same timeframe: if both come out positive that is ONE result, not two, "
          "and the informative contrast is whether skipping the last month adds anything.",
    fn=_tsmom_12_1,
    params={"lookback": _TSMOM_LONG, "skip": _TSMOM_SKIP},
    timeframes=("1d",),
    min_warmup=420,
))


# --------------------------------------------------------------------------------------------
# 2. Multi-horizon consensus
# --------------------------------------------------------------------------------------------
def _tsmom_consensus(s: Series) -> np.ndarray:
    c, out = _base(s, _TSMOM_SEMI + 1)
    if c is None:
        return out
    r1 = talib.ROC(c, _TSMOM_SHORT)
    r3 = talib.ROC(c, _TSMOM_MID)
    r6 = talib.ROC(c, _TSMOM_SEMI)
    ok = np.isfinite(r1) & np.isfinite(r3) & np.isfinite(r6)
    out[ok & (r1 > 0) & (r3 > 0) & (r6 > 0)] = 1
    out[ok & (r1 < 0) & (r3 < 0) & (r6 < 0)] = -1
    return out


register(Hypothesis(
    name="momentum.tsmom_consensus",
    family="momentum",
    rationale="Information diffuses at different speeds: retail reacts in days, treasuries and ETFs "
              "in months. Requiring the 1-, 3- and 6-month horizons to agree in sign selects the "
              "stretches in which the three cohorts push at once, which is when the order-book "
              "imbalance is one-directional and there is no cohort selling to the other. It is not "
              "a grid: it is a single fixed configuration, the canonical time-series momentum "
              "triple, shortened from 1/3/12 to 1/3/6 because the crypto cycle lasts half as long.",
    prior="We expect less time in the market and a HIGHER edge per bar than tsmom_12_1, at the cost "
          "of arriving late to the 2019 and 2023 bottoms and of staying out through the first "
          "bounces. It is falsified if the edge per bar does not beat tsmom_12_1's: the consensus "
          "would be adding nothing and would merely be shrinking the sample.",
    fn=_tsmom_consensus,
    params={"short": _TSMOM_SHORT, "mid": _TSMOM_MID, "semi": _TSMOM_SEMI},
    timeframes=("1d",),
    min_warmup=240,
))


# --------------------------------------------------------------------------------------------
# 3. Short ROC
# --------------------------------------------------------------------------------------------
def _roc_short(s: Series) -> np.ndarray:
    c, out = _base(s, _ROC_SHORT + 1)
    if c is None:
        return out
    roc = talib.ROC(c, _ROC_SHORT)
    ok = np.isfinite(roc)
    out[ok & (roc > 0)] = 1
    out[ok & (roc < 0)] = -1
    return out


register(Hypothesis(
    name="momentum.roc10",
    family="momentum",
    rationale="Pure underreaction. When relevant information appears (a macro print, a move by a "
              "large holder, a regulatory headline) not every participant processes it at the same "
              "time: the adjustment is spread across several candles because the big desks slice "
              "their orders so as not to move the market. The sign of the last ten candles' return "
              "is the most direct measure of that adjustment in progress, before the positioning "
              "saturates.",
    prior="We expect a positive edge on 4h and 1d and a NULL OR NEGATIVE edge on 1h, where "
          "microstructure noise and the cost per turnover eat the effect. It is falsified if the "
          "sign of the ROC separates future returns on no timeframe; and if the edge turned out to "
          "be LARGER on 1h than on 1d, the mechanism we postulate is false: we would be measuring "
          "microstructure autocorrelation, not information diffusion.",
    fn=_roc_short,
    params={"period": _ROC_SHORT},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 4. Term structure of the ROC (acceleration)
# --------------------------------------------------------------------------------------------
def _roc_term_structure(s: Series) -> np.ndarray:
    c, out = _base(s, _ROC_LONG + 1)
    if c is None:
        return out
    # Rate PER CANDLE, so that the two horizons are comparable in scale.
    short_rate = talib.ROC(c, _ROC_SHORT) / _ROC_SHORT
    long_rate = talib.ROC(c, _ROC_LONG) / _ROC_LONG
    ok = np.isfinite(short_rate) & np.isfinite(long_rate)
    out[ok & (long_rate > 0) & (short_rate > long_rate)] = 1
    out[ok & (long_rate < 0) & (short_rate < long_rate)] = -1
    return out


register(Hypothesis(
    name="momentum.roc_term_structure",
    family="momentum",
    rationale="Second-order herding effect. A trend that is ACCELERATING indicates that new "
              "participants are still coming in —rising leverage, rising perpetual funding, FOMO "
              "buying— whereas one that is decelerating indicates that the marginal buyer has run "
              "out and all that is left is rotation among those already inside. We compare the "
              "per-candle rate over 10 candles against the one over 20 so that the comparison is on "
              "a homogeneous scale; this is not a variant of roc10, it is a condition on its "
              "derivative.",
    prior="We expect a positive edge, higher per bar than roc10 while the trend accelerates, and we "
          "expect it to FAIL exactly at blow-off tops, where the acceleration peaks on the eve of "
          "the turn (December 2017, April and November 2021). It is falsified if its edge is "
          "indistinguishable from roc10's: the acceleration would be adding no information over the "
          "level.",
    fn=_roc_term_structure,
    params={"short": _ROC_SHORT, "long": _ROC_LONG},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 5. Sign of the Awesome Oscillator
# --------------------------------------------------------------------------------------------
def _ao_sign(s: Series) -> np.ndarray:
    c, out = _base(s, _AO_SLOW + 1)
    if c is None:
        return out
    ao = _ao(s)
    ok = np.isfinite(ao)
    out[ok & (ao > 0)] = 1
    out[ok & (ao < 0)] = -1
    return out


register(Hypothesis(
    name="momentum.ao_sign",
    family="momentum",
    rationale="The AO subtracts the median price of the last 34 candles from that of the last 5, "
              "that is, it compares the average cost of the recent buying cohort with that of the "
              "previous one. If it is positive, those who bought recently are in profit and are in "
              "no hurry to sell, so the supply into each rise is thin; if it is negative, every "
              "bounce runs into people getting out at break-even, which is dense, continuously "
              "renewed supply. It uses (H+L)/2 instead of the close, which reduces the weight of "
              "the closing-auction price relative to the range actually traded.",
    prior="We expect a positive edge in a trend and around zero or negative in a range, where the "
          "crossover of the two averages happens with a systematic lag. We expect it NOT to beat an "
          "equivalent moving-average crossover from the trend family: if it does, the median price "
          "contributes something the close does not, and that would be a result in itself. It is "
          "falsified if the sign of the AO separates future returns not even in a trending regime.",
    fn=_ao_sign,
    params={"fast": _AO_FAST, "slow": _AO_SLOW},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 6. AO that CONFIRMS the new extreme (folklore: wave 3)
# --------------------------------------------------------------------------------------------
def _ao_confirms(s: Series) -> np.ndarray:
    c, out = _base(s, _DIV_WINDOW + _AO_SLOW + 1)
    if c is None:
        return out
    ao = _ao(s)
    jmax = _prior_extreme(c, find_max=True)
    jmin = _prior_extreme(c, find_max=False)

    has_max = jmax >= 0
    jm = np.where(has_max, jmax, 0)
    ok_max = has_max & np.isfinite(c) & np.isfinite(c[jm]) & np.isfinite(ao) & np.isfinite(ao[jm])
    out[ok_max & (c > c[jm]) & (ao > ao[jm]) & (ao > 0)] = 1

    has_min = jmin >= 0
    jn = np.where(has_min, jmin, 0)
    ok_min = has_min & np.isfinite(c) & np.isfinite(c[jn]) & np.isfinite(ao) & np.isfinite(ao[jn])
    out[ok_min & (c < c[jn]) & (ao < ao[jn]) & (ao < 0)] = -1
    return out


register(Hypothesis(
    name="momentum.ao_confirms_extreme",
    family="momentum",
    rationale="A new price high accompanied by an AO HIGHER than at the window's previous high "
              "means that the displacement of the average cost is bigger than it was last time: the "
              "cohort doing the buying is more numerous, not the same people rotating the same "
              "capital. That is the pattern Elliott practitioners call a third wave, the one with "
              "the most participation and the least resistance. The real mechanism is herding: the "
              "confirmed break draws in the systematic follower and retail at the same time, and "
              "both buy on the same side of the book.",
    prior="We expect a positive edge but a very concentrated one: the signal should be flat most of "
          "the time and active on only a few bars per year. We expect it to be WORSE than ao_sign "
          "in total edge (by being in the market far less) and BETTER in edge per bar. It is "
          "falsified if its edge per bar does not beat ao_divergence_extreme's: in that case the "
          "wave 3 / wave 5 distinction of Elliott folklore has no empirical content.",
    fn=_ao_confirms,
    params={"window": _DIV_WINDOW, "separation": _DIV_SEPARATION,
            "fast": _AO_FAST, "slow": _AO_SLOW},
    timeframes=("4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 7. AO that DIVERGES from the new extreme (folklore: wave 5)
# --------------------------------------------------------------------------------------------
def _ao_divergence(s: Series) -> np.ndarray:
    c, out = _base(s, _DIV_WINDOW + _AO_SLOW + 1)
    if c is None:
        return out
    ao = _ao(s)
    jmax = _prior_extreme(c, find_max=True)
    jmin = _prior_extreme(c, find_max=False)

    has_max = jmax >= 0
    jm = np.where(has_max, jmax, 0)
    ok_max = has_max & np.isfinite(c) & np.isfinite(c[jm]) & np.isfinite(ao) & np.isfinite(ao[jm])
    # Bill Williams' bearish "twin peaks": both peaks above zero.
    out[ok_max & (c > c[jm]) & (ao < ao[jm]) & (ao > 0) & (ao[jm] > 0)] = -1

    has_min = jmin >= 0
    jn = np.where(has_min, jmin, 0)
    ok_min = has_min & np.isfinite(c) & np.isfinite(c[jn]) & np.isfinite(ao) & np.isfinite(ao[jn])
    out[ok_min & (c < c[jn]) & (ao > ao[jn]) & (ao < 0) & (ao[jn] < 0)] = 1
    return out


register(Hypothesis(
    name="momentum.ao_divergence_extreme",
    family="momentum",
    rationale="A new price high with an AO LOWER than at the previous high says that the price was "
              "made with less displacement of the average cost: the high is produced by short "
              "covering and late retail, with a smaller aggregate size, while the cohort that drove "
              "the previous leg is already distributing into them. Both peaks are required to be "
              "above zero (Bill Williams' twin peaks) so as not to confuse exhaustion with a bounce "
              "inside a fall. We emit the signal against the price.",
    prior="Our prior is UNFAVOURABLE and we register it all the same. We expect a slightly negative "
          "edge over the history as a whole and a clearly negative one inside the long trends of "
          "2017-Q4 and 2020-Q4, where the divergence appears and the price goes on rising for "
          "weeks; we only expect a positive edge in the immediate neighbourhood of cycle tops, "
          "which are four events and therefore an insufficient n. It is falsified as a trading tool "
          "if its overall edge is not positive; it is registered in order to leave the expected "
          "failure of a very popular artefact on the record, not to endorse it.",
    fn=_ao_divergence,
    params={"window": _DIV_WINDOW, "separation": _DIV_SEPARATION,
            "fast": _AO_FAST, "slow": _AO_SLOW},
    timeframes=("4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 8. Acceleration by regression slope
# --------------------------------------------------------------------------------------------
def _linreg_acceleration(s: Series) -> np.ndarray:
    c, out = _base(s, 2 * _LINREG + 2)
    if c is None:
        return out
    safe = np.where(np.isfinite(c) & (c > 0), c, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_c = np.log(safe)
    if not np.isfinite(log_c).any():
        return out
    slope = talib.LINEARREG_SLOPE(_f(log_c), _LINREG)
    delta = slope - _back(slope, _LINREG)         # change over a non-overlapping window
    ok = np.isfinite(slope) & np.isfinite(delta)
    out[ok & (slope > 0) & (delta > 0)] = 1
    out[ok & (slope < 0) & (delta < 0)] = -1
    return out


register(Hypothesis(
    name="momentum.linreg_acceleration",
    family="momentum",
    rationale="The least-squares slope of twenty log prices uses all twenty observations, not just "
              "the endpoints, so a single cascading-liquidation wick —which is forced supply, not "
              "informed demand— does not dominate it. Its change over twenty candles measures "
              "whether the force sustaining the trend is growing or running out. The concrete "
              "hypothesis is that momentum measured with an estimator robust to liquidation noise "
              "survives where momentum measured from endpoints does not.",
    prior="It overlaps with roc_term_structure on purpose and is registered all the same: if "
          "linreg_acceleration's edge is materially LARGER, the difference is down to the noise of "
          "wicks and liquidations and not to the concept of acceleration; if the two are "
          "indistinguishable, the robust estimator buys nothing in BTC. We expect a small "
          "difference. It is falsified if neither version has an edge, and we expect both to fail "
          "at trend restarts after capitulation, where the slope is still negative once the bottom "
          "is already in.",
    fn=_linreg_acceleration,
    params={"period": _LINREG},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 9. Persistence of the sign of the return
# --------------------------------------------------------------------------------------------
def _sign_persistence(s: Series) -> np.ndarray:
    c, out = _base(s, _PERSIST_WINDOW + 2)
    if c is None:
        return out
    r = np.diff(c, prepend=np.nan)
    up = (r > 0).astype(np.float64)               # NaN -> False -> 0.0, no invented signal
    down = (r < 0).astype(np.float64)
    n_up = talib.SUM(_f(up), _PERSIST_WINDOW)
    n_down = talib.SUM(_f(down), _PERSIST_WINDOW)
    ok = np.isfinite(n_up) & np.isfinite(n_down)
    out[ok & (n_up >= _PERSIST_HIGH)] = 1
    out[ok & (n_down >= _PERSIST_HIGH)] = -1
    return out


register(Hypothesis(
    name="momentum.sign_persistence",
    family="momentum",
    rationale="It counts how many of the last twenty candles closed up and discards magnitude "
              "entirely. The reason is that the two mechanisms leave different footprints: herding "
              "and systematic following produce a succession of small, repeated purchases —drift "
              "with many candles of the same sign— whereas a headline or a liquidation produces an "
              "enormous magnitude in a single candle and no persistence at all. Separating "
              "frequency from size is exactly what this hypothesis tests, and it makes it immune to "
              "BTC's fat tails into the bargain.",
    prior="We expect a positive edge SMALLER than roc10's but more stable across regimes, it being "
          "a count statistic. We expect it to fail explicitly in stretches of slow bullish drift "
          "followed by vertical falls (the second half of 2021): there the frequency of green "
          "candles is high and the cumulative return negative, and the signal will be long the "
          "whole way. It is falsified if the count of signs separates future returns on no "
          "timeframe.",
    fn=_sign_persistence,
    params={"window": _PERSIST_WINDOW, "threshold": _PERSIST_HIGH},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 10. Momentum filtered by Kaufman's efficiency ratio
# --------------------------------------------------------------------------------------------
def _er_filtered(s: Series) -> np.ndarray:
    c, out = _base(s, _ER_PERIOD + 2)
    if c is None:
        return out
    net = c - _back(c, _ER_PERIOD)
    path = talib.SUM(_f(np.abs(np.diff(c, prepend=np.nan))), _ER_PERIOD)
    ok = np.isfinite(net) & np.isfinite(path) & (path > 0)
    er = np.full(c.size, np.nan)
    er[ok] = np.abs(net[ok]) / path[ok]
    efficient = np.isfinite(er) & (er >= _ER_THRESHOLD)
    out[efficient & (net > 0)] = 1
    out[efficient & (net < 0)] = -1
    return out


register(Hypothesis(
    name="momentum.er_kaufman_filtered",
    family="momentum",
    rationale="The efficiency ratio divides the net displacement by the total path travelled, so it "
              "is high when the price goes from A to B in almost a straight line. A straight path "
              "is the signature of a sustained imbalance of the book —somebody large executing in "
              "one direction for days— and a zigzag path is the signature of buyers and sellers "
              "alternating against a liquidity provider. The hypothesis is that momentum only pays "
              "in the first case, because in the second the one chasing the trend buys expensively "
              "from the one providing it and pays the spread over and over.",
    prior="We expect the edge conditioned on ER≥0.30 to be clearly superior to roc10's and the "
          "filter to greatly reduce the number of bars in position. It is falsified if the filtered "
          "edge does NOT beat roc10's: that would mean the straightness of the path says nothing "
          "about future persistence, which is precisely the postulated mechanism. The 0.30 "
          "threshold is fixed a priori; if it fails, no other threshold is tried, the hypothesis is "
          "taken as refuted.",
    fn=_er_filtered,
    params={"period": _ER_PERIOD, "threshold": _ER_THRESHOLD},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 11. KAMA crossover
# --------------------------------------------------------------------------------------------
def _kama_cross(s: Series) -> np.ndarray:
    c, out = _base(s, _KAMA_ER + 32)
    if c is None:
        return out
    kama = talib.KAMA(c, _KAMA_ER)
    ok = np.isfinite(kama) & np.isfinite(c)
    out[ok & (c > kama)] = 1
    out[ok & (c < kama)] = -1
    return out


register(Hypothesis(
    name="momentum.kama_cross",
    family="momentum",
    rationale="The KAMA uses the same efficiency ratio to interpolate between a 2-period average "
              "and a 30-period one: it sticks to the price when the path is efficient and flattens "
              "out when it is noisy. The participant this is meant to avoid being is a specific "
              "one: the trend follower who buys every false break in a range and hands the spread "
              "to the market maker candle after candle. Against er_kaufman_filtered, which decides "
              "whether to trade at all, here efficiency decides the SPEED of the filter and never "
              "stops having an opinion.",
    prior="We expect fewer signal changes and a better edge per bar than a fixed-average crossover, "
          "and we expect it to fail at the start of explosive trends, where the adaptation lags "
          "behind and it enters late. It is falsified if its edge does not beat that of the fixed "
          "EMA crossover from the trend family: in that case adapting the speed buys nothing and "
          "only adds a moving part.",
    fn=_kama_cross,
    params={"er_period": _KAMA_ER, "fast": 2, "slow": 30},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 12. Momentum confirmed by participation
# --------------------------------------------------------------------------------------------
def _momentum_with_volume(s: Series) -> np.ndarray:
    c, out = _base(s, _ROC_LONG + _VOL_LONG + 2)
    if c is None:
        return out
    v = _f(s.volume)
    if not np.isfinite(v).any():
        return out
    roc = talib.ROC(c, _ROC_LONG)
    v_short = talib.SMA(v, _VOL_SHORT)
    v_long = talib.SMA(v, _VOL_LONG)
    ok = np.isfinite(roc) & np.isfinite(v_short) & np.isfinite(v_long) & (v_long > 0)
    participating = ok & (v_short > v_long)
    out[participating & (roc > 0)] = 1
    out[participating & (roc < 0)] = -1
    return out


register(Hypothesis(
    name="momentum.momentum_with_volume",
    family="momentum",
    rationale="The herd leaves its footprint in the volume: a leg with growing participation "
              "implies new money coming in, whereas one with falling volume is the same capital "
              "rotating among those already inside, and that one runs out on its own. We require "
              "the 5-candle volume average to exceed the 20-candle one at the same time as the "
              "20-candle return has a definite sign, so that the signal is only active when "
              "direction and participation coincide. The two volume windows are not two variants of "
              "one parameter: both are needed to define 'growing', and they are the conventional "
              "relative-volume pair.",
    prior="We expect an edge superior to the plain ROC's only on 1d and 4h, and FAILURE on 1h "
          "because of the intraday seasonality of volume (the Asian and US sessions), which fires "
          "the filter by hour of day and not by conviction. This is also the first hypothesis in "
          "the family that ought to break if the data comes from an exchange with inflated or "
          "washed volume. It is falsified if the volume filter does not improve on the unfiltered "
          "20-bar ROC. "
          "AUDIT NOTE (2026-09-08): that control —the unfiltered 20-bar ROC— was not registered "
          "anywhere, so the falsification criterion above could not be executed; it has been "
          "registered as `flow.roc20_unfiltered` and that is what this hypothesis must be measured "
          "against. It is further declared that `flow.trend_with_participation` is the same "
          "construction (ROC-20 conditioned on an expansion of volume) with a different pair of "
          "volume averages, and that the two are dependent trials even though they share few "
          "signals.",
    fn=_momentum_with_volume,
    params={"roc": _ROC_LONG, "volume_short": _VOL_SHORT, "volume_long": _VOL_LONG},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))
