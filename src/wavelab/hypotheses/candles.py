"""Family `candles`: candle microstructure. PRE-REGISTERED, written without looking at one number.

Starting stance: sceptical. Most of the canonical candlestick patterns have no demonstrated
out-of-sample edge, and much of their semantics ("indecision", "rejection") imports the assumptions
of a market with a session open and a session close, which BTC — trading 24/7 — does not have.
Several hypotheses in this file predict ZERO on purpose: a prior of "zero" is falsifiable by any
significant effect in either direction, and a family in which everything predicts success is not a
family of hypotheses, it is a brochure.

Conventions, fixed BEFORE running anything:

- PULSE versus STATE. A pattern is an event: it emits ±1 on the candle where it completes and goes
  back to 0. Exit management is imposed by the engine, not by the hypothesis. Only two hypotheses
  extend the signal over several candles, and they do so because their declared mechanism lasts
  several candles (an institutional order worked in slices, the drift after a shock); it is said so
  in their rationale. Two hypotheses are state and not event (`close_in_range`, `body_flow_20`):
  they describe a sustained condition of the candle or of the window, not a one-off occurrence.

- PARAMETERS. One value per parameter, taken by convention from the literature (RSI 14 with 30/70
  thresholds, ADX 14 with a threshold of 25, ATR 14, EMA 200, window 20) or, where the literature
  offers none, a round number fixed in advance (0.8/0.2 for the close's position in the range, 2x
  the body and 0.6 of the range for the wick, 2x ATR for the shock, 0.30 of body flow). None was
  chosen by looking at data, and none will be re-tuned afterwards. Changing one would be a
  different hypothesis and a different trial.

- CAUSALITY. Everything read at position i comes from i or earlier. Lags go through `_lag` (fills
  NaN at the start, never wraps), pulse propagation uses `np.maximum.accumulate`, which only looks
  at the prefix, and nothing is used that is defined against the whole array (find_peaks with
  prominence, global minima, normalisations by the total). TA-Lib's CDL* functions are causal:
  their internal thresholds are backward-looking moving averages.

- NaN. TA-Lib returns NaN during warm-up. It is checked explicitly with np.isnan (`_no_nan`)
  instead of trusting that a comparison against NaN yields False, because that implicit protection
  is lost the moment a condition appears negated.

- COSTS. Candle patterns are one-candle signals and they fire a lot. Gross edge means nothing here:
  what gets recorded as success is the delta against the null arm NET of fees, and for several of
  these hypotheses the prior is precisely that costs eat it.
"""

from __future__ import annotations

import numpy as np
import talib

from wavelab.hypotheses.base import Hypothesis, Series, register

# No hypothesis in this family emits a signal on less history than this: it covers the longest
# warm-up (EMA 200) with room to spare and avoids handing TA-Lib series shorter than its lookback.
_MIN_BARS = 320


def _f(a: np.ndarray) -> np.ndarray:
    """TA-Lib demands contiguous float64. No copy if it already is."""
    return np.ascontiguousarray(a, dtype=np.float64)


def _no_nan(*arrays: np.ndarray) -> np.ndarray:
    """Mask that is True where NONE of the arrays is NaN.

    Made explicit rather than delegated to `NaN > x` being False: that protection disappears the
    moment the condition appears negated (`~(x > y)` IS True with NaN) and would come back as an
    invented signal right in the warm-up, which is where nobody is looking.
    """
    m = np.ones(np.shape(arrays[0]), dtype=bool)
    for a in arrays:
        m &= ~np.isnan(np.asarray(a, dtype=np.float64))
    return m


def _lag(a: np.ndarray, k: int = 1) -> np.ndarray:
    """The value at position i-k, placed at i. NaN at the start. Only ever looks backwards."""
    x = np.asarray(a, dtype=np.float64)
    if k <= 0:
        return x.copy()
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if k < x.size:
        out[k:] = x[: x.size - k]
    return out


def _hold(pulse: np.ndarray, n: int) -> np.ndarray:
    """Extends each ±1 pulse over n candles, counting the signal candle itself.

    Causal: `maximum.accumulate` only depends on the prefix, so position i knows nothing beyond the
    last signal that occurred at i or before. A new pulse replaces the previous one.
    """
    idx = np.arange(pulse.size)
    last = np.maximum.accumulate(np.where(pulse != 0, idx, -1))
    alive = (last >= 0) & ((idx - last) < n)
    return np.where(alive, pulse[np.maximum(last, 0)], 0).astype(np.int8)


def _ohlc(s: Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _f(s.open), _f(s.high), _f(s.low), _f(s.close)


# ---------------------------------------------------------------------------
# 1. Unfiltered engulfing: the folklore version, registered so that it can be refuted.
# ---------------------------------------------------------------------------
def _engulfing_raw(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_BARS:
        return out
    o, h, l, c = _ohlc(s)
    p = np.asarray(talib.CDLENGULFING(o, h, l, c), dtype=np.float64)
    ok = _no_nan(p)
    out[ok & (p > 0)] = 1
    out[ok & (p < 0)] = -1
    return out


register(Hypothesis(
    name="candles.engulfing_raw",
    family="candles",
    rationale="An engulfing candle means that the entire price range the previous candle considered "
              "acceptable has been travelled end to end and the close has ended up on the other "
              "side of its open: whoever took a position during the engulfed candle is underwater "
              "as a block, and their stops sit just beyond the far extreme of the engulfing candle, "
              "which supplies forced flow in the same direction. That is the mechanism attributed "
              "to it, and it is registered with no filter at all, exactly as the canon states it, "
              "so that the conditioned version (candles.engulfing_trend) has something to be "
              "measured against.",
    prior="We predict NO edge: we expect an expectancy indistinguishable from zero gross and "
          "negative once fees are deducted, because without context an engulfing candle is "
          "mechanically 'a big candle that closed strong', which in BTC turns up more often at the "
          "end of a liquidation cascade than at the start of a move. It is falsified if the net "
          "delta against the null arm is clearly positive on any of the three timeframes; it is "
          "falsified too, in the other direction, if it is clearly negative, which would mean the "
          "pattern works inverted.",
    fn=_engulfing_raw,
    params={"pattern": "CDLENGULFING", "hold": 1},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 2. Engulfing aligned with the underlying trend.
# ---------------------------------------------------------------------------
def _engulfing_with_trend(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_BARS:
        return out
    o, h, l, c = _ohlc(s)
    p = np.asarray(talib.CDLENGULFING(o, h, l, c), dtype=np.float64)
    ema = talib.EMA(c, 200)
    ok = _no_nan(p, ema)
    out[ok & (p > 0) & (c > ema)] = 1
    out[ok & (p < 0) & (c < ema)] = -1
    return out


register(Hypothesis(
    name="candles.engulfing_trend",
    family="candles",
    rationale="The same engulfing candle, but only when it points in the direction of the "
              "underlying trend. The stop mechanism can only work if there is one side trapped and "
              "another with the capacity to push: in a rising market a bullish engulfing marks the "
              "sellers of the pullback as the trapped block, whereas a bullish engulfing inside a "
              "decline marks buyers who will keep running into supply above. This is not a "
              "parameter variant of the previous hypothesis: the previous one is the canon's "
              "unconditional claim and this one is the claim that the effect is CONDITIONAL on the "
              "regime; the contrast between the two is the informative quantity, and the EMA 200 is "
              "fixed at its conventional value and not searched over.",
    prior="We expect a positive edge, larger than candles.engulfing_raw's, and we expect the "
          "difference between the two to be larger than the absolute edge of either. It fails if "
          "the filter improves nothing (the pattern is not context-sensitive and its declared "
          "mechanism is false) and it fails just the same if it improves but with a negative net "
          "expectancy, which would be nothing but BTC's structural bullish bias, filtered — not the "
          "pattern.",
    fn=_engulfing_with_trend,
    params={"pattern": "CDLENGULFING", "ema": 200, "hold": 1},
    timeframes=("1h", "4h", "1d"),
    min_warmup=300,
))


# ---------------------------------------------------------------------------
# 3. Hammer after a fall (RSI 14 oversold).
# ---------------------------------------------------------------------------
def _hammer_oversold(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_BARS:
        return out
    o, h, l, c = _ohlc(s)
    p = np.asarray(talib.CDLHAMMER(o, h, l, c), dtype=np.float64)
    rsi = talib.RSI(c, 14)
    ok = _no_nan(p, rsi)
    out[ok & (p > 0) & (rsi < 30.0)] = 1
    return out


register(Hypothesis(
    name="candles.hammer_oversold",
    family="candles",
    rationale="A hammer is a candle whose price fell a long way and came back: the lower wick is "
              "the visible footprint of passive demand that absorbed all the aggressive selling and "
              "handed the price back. That only says something after a fall, when the aggressive "
              "selling is forced liquidation and not orderly placement: the forced seller is spent "
              "within hours, the buyer who absorbed is still there, and the imbalance resolves "
              "upwards. Without the fall beforehand, the same shape is just another candle inside a "
              "range, which is why the oversold condition is part of the hypothesis and not an "
              "add-on.",
    prior="We expect a small positive edge and only on the long side. We expect it to fail exactly "
          "where the selling is not forced but informed: in sustained downtrends every absorption "
          "gets retested and the hammer turns into a falling knife, so we anticipate a fat left "
          "tail and a high hit rate with mediocre expectancy. Falsified if the sign is negative, or "
          "if the net delta against the null arm cannot be told apart from zero.",
    fn=_hammer_oversold,
    params={"pattern": "CDLHAMMER", "rsi": 14, "rsi_threshold": 30, "hold": 1},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 4. Shooting star after a rise (RSI 14 overbought). The hammer's mirror image.
# ---------------------------------------------------------------------------
def _shooting_star_overbought(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_BARS:
        return out
    o, h, l, c = _ohlc(s)
    p = np.asarray(talib.CDLSHOOTINGSTAR(o, h, l, c), dtype=np.float64)
    rsi = talib.RSI(c, 14)
    ok = _no_nan(p, rsi)
    out[ok & (p < 0) & (rsi > 70.0)] = -1
    return out


register(Hypothesis(
    name="candles.shooting_star_overbought",
    family="candles",
    rationale="Exact mirror of the hammer: a long upper wick after a rise, that is, passive supply "
              "that absorbed all the aggressive buying and handed the price back. It is registered "
              "separately rather than as the short side of the same hypothesis because the "
              "mechanism is NOT symmetric: aggressive buying at highs is discretionary and can walk "
              "away at no cost, whereas aggressive selling at lows is usually forced liquidation on "
              "a deadline. If the canon's symmetry held, both hypotheses ought to give a similar "
              "edge; the comparison between the two is the content of this pair.",
    prior="We expect an effect clearly SMALLER than the hammer's and we accept a negative "
          "expectancy as a likely outcome: shorting into strength in BTC fights both the asset's "
          "positive drift and the market's long bias at once, and the funding cost of the short is "
          "not recovered within one candle. Falsified if its edge matches or beats the hammer's, "
          "which would mean absorption is symmetric and our forced-flow explanation is superfluous.",
    fn=_shooting_star_overbought,
    params={"pattern": "CDLSHOOTINGSTAR", "rsi": 14, "rsi_threshold": 70, "hold": 1},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 5. Position of the close within the candle's range (state, not event).
# ---------------------------------------------------------------------------
def _close_in_range(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_BARS:
        return out
    _o, h, l, c = _ohlc(s)
    rng = h - l
    ok = _no_nan(h, l, c) & (rng > 0.0)
    pos = np.zeros(len(s), dtype=np.float64)
    np.divide(c - l, rng, out=pos, where=ok)
    out[ok & (pos >= 0.80)] = 1
    out[ok & (pos <= 0.20)] = -1
    return out


register(Hypothesis(
    name="candles.close_in_range",
    family="candles",
    rationale="The close is the only price both sides agreed to carry a position at from one candle "
              "into the next; the rest of the range is prices somebody rejected. A close pinned to "
              "the extreme of the range means the winning side still had unfilled demand when the "
              "candle ended: that order does not vanish at the candle boundary, it keeps executing. "
              "It is the most stripped-down reading of candle shape there is —it needs no pattern "
              "name and no context— and that is why it serves as the minimum benchmark against "
              "which to judge the named patterns of this same family.",
    prior="We expect continuation: a small positive edge on 1h and 4h. We expect it to cancel out "
          "or flip sign on 1d, where the horizon gives reversion time to take over, and on 15m to "
          "be positive gross but negative net, because the signal fires on almost one candle in "
          "three and the cost per turnover eats it. Falsified if the sign is negative on 1h and 4h, "
          "which would be evidence that a close at the extreme marks exhaustion and not residual "
          "demand.",
    fn=_close_in_range,
    params={"upper_threshold": 0.80, "lower_threshold": 0.20},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 6. Closing marubozu: a session won from end to end.
# ---------------------------------------------------------------------------
def _marubozu_continuation(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_BARS:
        return out
    o, h, l, c = _ohlc(s)
    p = np.asarray(talib.CDLCLOSINGMARUBOZU(o, h, l, c), dtype=np.float64)
    ok = _no_nan(p)
    out[ok & (p > 0)] = 1
    out[ok & (p < 0)] = -1
    return out


register(Hypothesis(
    name="candles.marubozu_continuation",
    family="candles",
    rationale="A closing marubozu is a large body with no wick on the close side: throughout the "
              "whole candle there was not one moment in which the losing side managed to hand the "
              "price back, not even at the end. It is the signature of a flow imbalance that met no "
              "passive opposition, and an imbalance like that does not expire exactly at the "
              "arbitrary boundary of the candle. The closing variant is used and not the strict "
              "marubozu because the informative extreme is the close; also demanding no wick at the "
              "open only adds rarity without adding mechanism.",
    prior="We expect continuation in the direction of the body, with a small positive edge. We "
          "expect it to fail at range extremes and at the end of long runs, where the same candle "
          "is capitulation and not impulse; as we do not tell the two cases apart ex ante, the "
          "average of both should come out weak. Falsified if the net sign is negative, which would "
          "support the exhaustion reading over the continuation one.",
    fn=_marubozu_continuation,
    params={"pattern": "CDLCLOSINGMARUBOZU", "hold": 1},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 7. Doji after a directional thrust (ADX 14 > 25). A hypothesis that predicts ZERO.
# ---------------------------------------------------------------------------
def _doji_after_thrust(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_BARS:
        return out
    o, h, l, c = _ohlc(s)
    p = np.asarray(talib.CDLDOJI(o, h, l, c), dtype=np.float64)
    adx = talib.ADX(h, l, c, 14)
    di_plus = talib.PLUS_DI(h, l, c, 14)
    di_minus = talib.MINUS_DI(h, l, c, 14)
    ok = _no_nan(p, adx, di_plus, di_minus)
    thrust = ok & (p != 0.0) & (adx > 25.0)
    out[thrust & (di_plus > di_minus)] = -1
    out[thrust & (di_minus > di_plus)] = 1
    return out


register(Hypothesis(
    name="candles.doji_after_thrust",
    family="candles",
    rationale="A doji is the first candle of a leg in which the dominant side finds enough "
              "counterparty to end up where it started; inside a mature directional move it would "
              "mark the point at which the marginal buyer (or seller) runs out, and that is why the "
              "canon reads it as a warning of a turn. The direction of the thrust is taken from "
              "Wilder's directional system with its conventional parameter, so as not to introduce "
              "a window of our own. We register it because it is the most-cited candle hypothesis "
              "we are able to state with an explicit mechanism, not because we believe in it.",
    prior="We predict ZERO. In a 24/7 market there is no session open and no session close, so a "
          "doji is not a collective verdict but simply a small-bodied candle, and they turn up in "
          "handfuls during the low-volume hours; the semantics of 'indecision' imports a session "
          "structure that does not exist here. A prior of zero is falsifiable by any significant "
          "effect in either direction: if fading the thrust gives a positive edge, our objection "
          "about the 24/7 market is wrong; if it gives a negative one, the doji is continuation and "
          "the canon has it inverted.",
    fn=_doji_after_thrust,
    params={"pattern": "CDLDOJI", "adx": 14, "adx_threshold": 25, "hold": 1},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 8. Outsized wick, with no folklore attached: pure symmetric geometry.
# ---------------------------------------------------------------------------
def _outsized_wick(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_BARS:
        return out
    o, h, l, c = _ohlc(s)
    ok = _no_nan(o, h, l, c)
    rng = h - l
    ok &= rng > 0.0
    body = np.abs(c - o)
    lower_wick = np.minimum(o, c) - l
    upper_wick = h - np.maximum(o, c)
    long_side = ok & (lower_wick >= 2.0 * body) & (lower_wick >= 0.60 * rng)
    short_side = ok & (upper_wick >= 2.0 * body) & (upper_wick >= 0.60 * rng)
    out[long_side] = 1
    out[short_side] = -1
    return out


register(Hypothesis(
    name="candles.outsized_wick",
    family="candles",
    rationale="A long wick is the footprint of a level the price reached and came back from: proof "
              "that there really was passive liquidity there, which absorbed the aggressive flow "
              "and was left partly unfilled. Whoever has a large order at that price puts it back, "
              "so the level acts as a soft floor or ceiling over the following candles. It is "
              "defined by geometry alone —wick greater than twice the body and more than 60% of the "
              "range, symmetric above and below— and without requiring a prior trend or a body "
              "position, precisely in order to separate the absorption effect from the vocabulary "
              "of hammer and shooting star, which add context conditions.",
    prior="We expect a small positive edge in the direction opposite to the wick, larger on 4h than "
          "on 15m. We expect it to fail in strong trends, where a long lower wick inside a fall is "
          "not absorption but a partial retracement before continuing, and we expect that on 15m "
          "the gross edge will exist but will disappear once fees are deducted. Falsified if the "
          "sign is negative, or if it beats the null arm on no timeframe.",
    fn=_outsized_wick,
    params={"wick_vs_body": 2.0, "wick_vs_range": 0.60},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 9. Range expansion against the ATR: informational shock or forced flow.
# ---------------------------------------------------------------------------
def _range_expansion(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_BARS:
        return out
    o, h, l, c = _ohlc(s)
    tr = talib.TRANGE(h, l, c)
    atr_prev = _lag(talib.ATR(h, l, c, 14), 1)
    ok = _no_nan(tr, atr_prev, o, c) & (atr_prev > 0.0)
    shock = ok & (tr >= 2.0 * atr_prev)
    pulse = np.zeros(len(s), dtype=np.int8)
    pulse[shock & (c > o)] = 1
    pulse[shock & (c < o)] = -1
    return _hold(pulse, 3)


register(Hypothesis(
    name="candles.range_expansion",
    family="candles",
    rationale="A candle whose true range doubles the ATR of the previous 14 is not noise: it is a "
              "repricing in which the book was traded straight through. Our mechanism is "
              "informational: new information has come in, volatility clusters, and whoever has to "
              "reposition a large portfolio does not do it inside a single candle, so the drift "
              "carries on in the direction the body closed. The ATR is compared lagged by one "
              "candle so that the reference is the norm BEFORE the shock and not a norm already "
              "contaminated by it. The signal is held for three candles because the declared "
              "mechanism —sliced repositioning— lasts more than one, not because three works "
              "better.",
    prior="We expect continuation: a positive edge in the direction of the shock candle's body. "
          "There is an explicit and credible rival mechanism —that the large range is a liquidation "
          "cascade, that is, forced flow carrying no information, which market makers revert as "
          "they requote— and that mechanism predicts the OPPOSITE sign. That is why the sign of the "
          "result is informative whatever happens: positive supports the informational reading, "
          "negative supports the forced-flow one, and zero says the two cancel out because we do "
          "not tell the two cases apart ex ante, which is what we consider most likely on 15m. "
          "AUDIT NOTE (2026-09-08): `volatility.wide_range_thrust` applies this very same rule "
          "(true range greater than twice the previous ATR, direction given by the body) with one "
          "added condition —that the previous candle's ATR was below its 100-bar median— and "
          "without holding the signal for three candles. Measured on data, its events are an almost "
          "exact SUBSET of this hypothesis's (263 of 267 match candle for candle). Neither is "
          "removed, because the unfiltered / filtered pair is informative and that contrast was "
          "declared in neither of the two files, but let it be said now: they are not two "
          "independent trials and they must not be read as cross-family confirmation if both come "
          "out positive. The contrast that matters is whether the compression gate improves on this "
          "one.",
    fn=_range_expansion,
    params={"atr": 14, "multiple": 2.0, "hold": 3},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 10. Inside candle and break of the mother candle.
# ---------------------------------------------------------------------------
def _inside_bar_break(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_BARS:
        return out
    _o, h, l, c = _ohlc(s)
    h1, l1 = _lag(h, 1), _lag(l, 1)
    h2, l2 = _lag(h, 2), _lag(l, 2)
    ok = _no_nan(c, h1, l1, h2, l2)
    inside = ok & (h1 <= h2) & (l1 >= l2)
    out[inside & (c > h2)] = 1
    out[inside & (c < l2)] = -1
    return out


register(Hypothesis(
    name="candles.inside_bar_break",
    family="candles",
    rationale="A candle contained within the range of the previous one is a contraction of the "
              "traded range: both sides accept the same price interval and uncertainty drops. "
              "Protective orders pile up there on both sides of the mother candle's extremes. The "
              "next candle closing outside that range means the passive liquidity on one side has "
              "been consumed, and the path of least resistance becomes the cluster of stops on the "
              "other. The break is judged with the CLOSE of the candle in progress against extremes "
              "of candles already closed, never with the intrabar high, so that the decision "
              "belongs to candle i and not to data that is only known afterwards.",
    prior="We expect continuation with a small positive edge on 4h and 1d. With low confidence: "
          "there is an equally plausible opposing mechanism in BTC, namely that the stop cluster is "
          "the TARGET of the aggressive flow and not its fuel, in which case the break is false by "
          "construction and the sign comes out negative. We also expect it to get worse as the "
          "timeframe drops, because range contraction on 1h is mostly low-volume hours and not "
          "agreement between participants. Falsified by a stable negative sign. "
          "AUDIT NOTE (2026-09-08): this hypothesis was registered TWICE. "
          "`structure.inside_bar_break` was the same rule —same containment, same mother-candle "
          "extremes as the level, same confirmation by close— and its 244 events were a strict "
          "subset of this one's 293; it has been removed there and this is the surviving "
          "registration. It is further declared that `volatility.inside_bar_breakout` operates on "
          "the SAME two-candle pattern but breaking the extremes of the INSIDE candle rather than "
          "those of the mother: since the inside candle's high is the lower of the two, every event "
          "of this hypothesis is also an event of that one. They are different levels and different "
          "mechanical claims, so that one is kept, but the two are DEPENDENT trials and the "
          "multiple-comparisons correction must treat them as such.",
    fn=_inside_bar_break,
    params={"mother_bars": 1, "confirmation": "close"},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 11. A range gap between consecutive candles in a 24/7 market.
# ---------------------------------------------------------------------------
def _range_gap(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_BARS:
        return out
    _o, h, l, _c = _ohlc(s)
    h1, l1 = _lag(h, 1), _lag(l, 1)
    ok = _no_nan(h, l, h1, l1)
    out[ok & (l > h1)] = -1
    out[ok & (h < l1)] = 1
    return out


register(Hypothesis(
    name="candles.range_gap",
    family="candles",
    rationale="BTC trades without interruption, so a gap cannot be the result of information piling "
              "up while the market was closed: if a candle's entire range sits above the previous "
              "candle's high, the book has been emptied and the price has travelled with no trading "
              "in between. That is forced flow —chained liquidations, or a stretch of very thin "
              "liquidity— and not a consented repricing: as soon as the market makers quote again, "
              "the gap fills. Warning registered in advance: a hole in the provider's data produces "
              "exactly this same shape, so a positive result obliges us to check the continuity of "
              "the timestamps before believing it.",
    prior="We expect reversion: a positive edge trading AGAINST the gap, larger on 15m than on 4h. "
          "We expect it to fail when the gap answers to real news (a legitimate repricing that does "
          "not fill), a case we cannot tell apart ex ante, and we expect very few signals on 4h, "
          "where a full-range gap implies a change of regime and ought to continue rather than "
          "revert. Falsified if the sign is one of continuation on 15m and 1h.",
    fn=_range_gap,
    params={"type": "full_range_gap"},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 12. Three soldiers / three crows: the footprint of an order worked in slices.
# ---------------------------------------------------------------------------
def _three_directional_bars(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < _MIN_BARS:
        return out
    o, h, l, c = _ohlc(s)
    soldiers = np.asarray(talib.CDL3WHITESOLDIERS(o, h, l, c), dtype=np.float64)
    crows = np.asarray(talib.CDL3BLACKCROWS(o, h, l, c), dtype=np.float64)
    ok = _no_nan(soldiers, crows)
    pulse = np.zeros(n, dtype=np.int8)
    pulse[ok & (soldiers > 0)] = 1
    pulse[ok & (crows < 0)] = -1
    return _hold(pulse, 3)


register(Hypothesis(
    name="candles.three_directional_bars",
    family="candles",
    rationale="Three candles in a row with wide bodies, each closing near its high and above the "
              "previous one, is not an impulse: it is the footprint of a large order being executed "
              "in slices over time. A speculative impulse burns out in one candle; an execution "
              "programme follows a schedule and that is why it leaves three identical candles. If "
              "the reading is right, the unexecuted part of the programme goes on buying after the "
              "third candle, and that is why the signal is held for three candles: it is the "
              "horizon of the declared mechanism, not a value chosen for performance.",
    prior="We expect weak continuation. Weak because the entry is late by construction: by the time "
          "the pattern completes, three candles of the move have already happened and we are buying "
          "against the executor itself. We register in advance that both patterns are rare and that "
          "the number of signals will be small, so the interval will be wide and the "
          "multiple-comparisons correction must be applied all the same: a spectacular result with "
          "a small n counts as inconclusive, not as a finding. Falsified by a negative sign, which "
          "would support the exhaustion reading.",
    fn=_three_directional_bars,
    params={"patterns": "CDL3WHITESOLDIERS/CDL3BLACKCROWS", "hold": 3},
    timeframes=("4h", "1d"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 13. Body flow over 20 candles: how much of the range turns into progress (state).
# ---------------------------------------------------------------------------
def _body_flow(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < _MIN_BARS:
        return out
    o, h, l, c = _ohlc(s)
    body_sum = talib.SUM(c - o, 20)
    range_sum = talib.SUM(h - l, 20)
    ok = _no_nan(body_sum, range_sum) & (range_sum > 0.0)
    flow = np.zeros(n, dtype=np.float64)
    np.divide(body_sum, range_sum, out=flow, where=ok)
    out[ok & (flow >= 0.30)] = 1
    out[ok & (flow <= -0.30)] = -1
    return out


register(Hypothesis(
    name="candles.body_flow_20",
    family="candles",
    rationale="Aggregates candle shape instead of hunting for a pattern: it measures what "
              "proportion of all the range traded over 20 candles has turned into net open-to-close "
              "progress. When that proportion is high, the same side is systematically winning the "
              "sessions and the range is being travelled, not round-tripped: there are directional "
              "participants dominating the market makers. When it is low, price moves but is handed "
              "back within each candle, which is the signature of market-maker inventory. It is not "
              "a moving average: it ignores the jumps between candles and weights by the range "
              "traded, so two series with identical price travel can give opposite values.",
    prior="We expect a positive edge in a trending regime and none or a negative one in a range, "
          "and we expect the two together to come out barely above zero. The strong, falsifiable "
          "claim is one of incremental value: if candle shape adds nothing over the plain direction "
          "of price, this hypothesis should be indistinguishable from an ordinary trend filter over "
          "the same candles; that it contributes something above that is what would make it "
          "interesting, and we do not take it for granted.",
    fn=_body_flow,
    params={"window": 20, "threshold": 0.30},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))
