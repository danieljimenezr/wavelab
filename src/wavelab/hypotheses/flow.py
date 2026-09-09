"""Family `flow`: volume and order flow.

PRE-REGISTERED. Everything in this file was written BEFORE running a single backtest and without
looking at a single result. The parameters are the conventional values from the literature, fixed
once and for all; not one of them was chosen by comparing returns.

The family's thesis: price says what happened, volume says how much conviction was behind it. An
advance that moves twice the usual number of contracts leaves inventory in hands that will have to
defend it or close it; an advance on thin volume leaves nothing and obliges nobody to do anything
afterwards. Every hypothesis below is a variation on that idea, and several of them contradict each
other on purpose: if the market rewarded continuation on volume and fading on volume at the same
time, the whole family would be measuring noise.

Limitation accepted and declared: `Series` only exposes o/h/l/c/v, so there is NO `taker_buy_base`
and no real volume delta. Where the sign of the aggression is needed we use the Close Location
Value (where the candle closes inside its own range), which is the best proxy that can be built
from OHLCV and is exactly what the Accumulation/Distribution line assumes. It is a biased proxy: on
a candle with a long wick caused by liquidations it attributes aggressive buying where there was
only a stop run. That weakness is part of what is being put to the test, not an oversight.
"""

from __future__ import annotations

import numpy as np
import talib

from wavelab.hypotheses.base import Hypothesis, Series, register

# Repo convention (`Bar.open_time_ms`, `Timeframe.ms`): timestamps are epoch in MILLISECONDS on the
# UTC grid. The VWAP's session anchor depends on that.
_DAY_MS = 86_400_000


# --------------------------------------------------------------------------------------------
# Utilities. All strictly causal: position i only looks at positions j <= i.
# --------------------------------------------------------------------------------------------

def _f(a: np.ndarray) -> np.ndarray:
    """talib demands contiguous float64."""
    return np.ascontiguousarray(a, dtype=np.float64)


def _finite(*arrays: np.ndarray) -> np.ndarray:
    """True where ALL the arrays hold a real number.

    talib returns NaN during warm-up, and divisions by a zero range produce inf. It is checked
    explicitly instead of trusting that `NaN > x` yields False, and above all instead of using
    nan_to_num, which would turn the warm-up into invented signals.
    """
    ok = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        ok &= ~np.isnan(a)
        ok &= ~np.isinf(a)
    return ok


def _clv(s: Series) -> np.ndarray:
    """Close Location Value in [-1, +1]: a proxy for the sign of the aggression.

    +1 = closes at the high (the buyer took the candle), -1 = closes at the low.
    Zero-range candles end up at 0, not at NaN and not at inf.
    """
    h, l, c = _f(s.high), _f(s.low), _f(s.close)
    rng = h - l
    clv = np.zeros(rng.shape, dtype=np.float64)
    np.divide((c - l) - (h - c), rng, out=clv, where=rng > 0)
    return clv


def _hold(ev: np.ndarray, k: int) -> np.ndarray:
    """Propagates each non-zero event over k bars.

    Causal by construction: at position i only an event whose index j satisfies j <= i can appear.
    `np.maximum.accumulate` is a forward sweep, never a backward one.
    """
    n = ev.size
    i = np.arange(n)
    idx = np.where(ev != 0, i, -1)
    last = np.maximum.accumulate(idx)
    alive = (last >= 0) & ((i - last) < k)
    out = np.zeros(n, dtype=np.int8)
    out[alive] = ev[last[alive]]
    return out


def _session_vwap(s: Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """VWAP anchored to the UTC day, its volume-weighted standard deviation, and how many candles
    we are into the anchor.

    Accumulates ONLY from the start of the current day up to and including i. The subtraction of
    cumulative sums uses the index before the anchor, clamped at 0, so that we never index with -1
    (which in numpy would return the LAST element of the array, that is, the entire future).
    """
    n = len(s)
    tp = talib.TYPPRICE(_f(s.high), _f(s.low), _f(s.close))
    v = _f(s.volume)
    day = np.asarray(s.ts, dtype=np.int64) // _DAY_MS

    is_new = np.ones(n, dtype=bool)
    is_new[1:] = day[1:] != day[:-1]
    i = np.arange(n)
    start = np.maximum.accumulate(np.where(is_new, i, 0))
    has_prev = start > 0
    prev = np.maximum(start - 1, 0)

    acc_v = np.cumsum(v)
    acc_pv = np.cumsum(tp * v)
    acc_pv2 = np.cumsum(tp * tp * v)
    vol = acc_v - np.where(has_prev, acc_v[prev], 0.0)
    pv = acc_pv - np.where(has_prev, acc_pv[prev], 0.0)
    pv2 = acc_pv2 - np.where(has_prev, acc_pv2[prev], 0.0)

    vwap = np.full(n, np.nan)
    np.divide(pv, vol, out=vwap, where=vol > 0)
    m2 = np.full(n, np.nan)
    np.divide(pv2, vol, out=m2, where=vol > 0)
    var = m2 - vwap * vwap
    var[~np.isnan(var) & (var < 0.0)] = 0.0   # floating-point noise, not negative variance
    return vwap, np.sqrt(var), (i - start + 1)


# --------------------------------------------------------------------------------------------
# 1. OBV against its own average
# --------------------------------------------------------------------------------------------

def _obv_ema21(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 60:
        return out
    obv = talib.OBV(_f(s.close), _f(s.volume))
    ema = talib.EMA(obv, 21)
    ok = _finite(obv, ema)
    out[ok & (obv > ema)] = 1
    out[ok & (obv < ema)] = -1
    return out


register(Hypothesis(
    name="flow.obv_ema21",
    family="flow",
    rationale="OBV adds up each candle's entire volume with the sign of its close, so its slope "
              "measures whether the recent contracts have moved more on up candles or on down "
              "ones. The concrete mechanism: whoever buys into the rise ends up with inventory that "
              "can only be unwound by selling, and until it is unwound they hold up the bid; "
              "whoever sold has to buy back. When the cumulative line rises above its own 21-period "
              "average there is more new inventory in long hands than has been let go, and that "
              "imbalance in positioning should push the price before it resolves.",
    prior="We expect a small positive edge in a trending regime (4h/1d) and no edge or a negative "
          "one in a range, where OBV crosses its average constantly without any real movement of "
          "inventory. Strong falsification: if the edge is the SAME in a trend and in a range, the "
          "declared mechanism is false and this is nothing more than a slow trend follower dressed "
          "up as volume. We also expect it to fail in periods dominated by liquidation cascades, "
          "where OBV attributes accumulation to what was forced liquidation.",
    fn=_obv_ema21,
    params={"ema": 21},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 2. Price-OBV divergence: price makes a new extreme and volume does not follow
# --------------------------------------------------------------------------------------------

def _obv_divergence(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 60:
        return out
    c = _f(s.close)
    obv = talib.OBV(c, _f(s.volume))
    max_c, min_c = talib.MAX(c, 20), talib.MIN(c, 20)
    max_o, min_o = talib.MAX(obv, 20), talib.MIN(obv, 20)
    ok = _finite(c, obv, max_c, min_c, max_o, min_o)
    ev = np.zeros(n, dtype=np.int8)
    # talib's MAX/MIN include the current candle: `c >= max_c` is exactly "new 20-bar high".
    ev[ok & (c >= max_c) & (obv < max_o)] = -1
    ev[ok & (c <= min_c) & (obv > min_o)] = 1
    return _hold(ev, 5)


register(Hypothesis(
    name="flow.obv_divergence20",
    family="flow",
    rationale="A price high NOT accompanied by an OBV high means the last leg was signed by fewer "
              "contracts than the one before: price rises because supply has withdrawn, not because "
              "new demand has come in. A book that rises through the absence of sellers is thin, "
              "and one order of size is enough to go through it the other way. The symmetric "
              "reading holds at lows: a lower price with an OBV that no longer follows it indicates "
              "that the selling has run out of ammunition.",
    prior="We expect a positive edge at a short horizon (the event is held for 5 candles) and that "
          "it concentrates at range extremes, not in established trends. We expect it to FAIL, and "
          "to lose money, in strong and persistent trends: there the divergence shows up dozens of "
          "times in a row while price goes on rising, and this is precisely the indicator that "
          "ruins whoever tries to call tops. If the edge turned out to be as good in a trend as in "
          "a range, we would suspect the proxy before we suspected the market.",
    fn=_obv_divergence,
    params={"window": 20, "hold_bars": 5},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 3. The other diagonal: OBV makes a new extreme and price has not yet
# --------------------------------------------------------------------------------------------

def _obv_leads(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 60:
        return out
    c = _f(s.close)
    obv = talib.OBV(c, _f(s.volume))
    max_c, min_c = talib.MAX(c, 20), talib.MIN(c, 20)
    max_o, min_o = talib.MAX(obv, 20), talib.MIN(obv, 20)
    ok = _finite(c, obv, max_c, min_c, max_o, min_o)
    ev = np.zeros(n, dtype=np.int8)
    ev[ok & (obv >= max_o) & (c < max_c)] = 1
    ev[ok & (obv <= min_o) & (c > min_c)] = -1
    return _hold(ev, 5)


register(Hypothesis(
    name="flow.obv_leads20",
    family="flow",
    rationale="It is the opposite cell of the same 2x2 table as `obv_divergence20`, and it is "
              "registered separately because it asserts a different mechanism and trades in the "
              "opposite direction: here the cumulative volume marks a new high while price has not "
              "yet. That is the signature of a large buyer slicing their order against the "
              "available supply: they absorb everything that comes out without letting the price "
              "rise, because pushing it up would make their own execution more expensive. When they "
              "exhaust that supply, the price jumps with no resistance. Registering the two cells "
              "separately is what allows one to work and the other not; merging them into a single "
              "hypothesis would hide that result.",
    prior="We expect a small positive edge, less frequent than the classic divergence. We expect it "
          "to fail in markets where volume is dominated by market makers recycling inventory in "
          "seconds: there the OBV shoots up without any directional accumulator existing at all. If "
          "`obv_leads20` and `obv_divergence20` both came out positive with a similar "
          "magnitude, we would read that not as two findings but as evidence that what wins is the "
          "plain '20-candle extreme' event and not the volume.",
    fn=_obv_leads,
    params={"window": 20, "hold_bars": 5},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 4. Money flow index at Wilder's extremes
# --------------------------------------------------------------------------------------------

def _mfi14(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 40:
        return out
    mfi = talib.MFI(_f(s.high), _f(s.low), _f(s.close), _f(s.volume), 14)
    ok = _finite(mfi)
    out[ok & (mfi < 20.0)] = 1
    out[ok & (mfi > 80.0)] = -1
    return out


register(Hypothesis(
    name="flow.mfi14_extremes",
    family="flow",
    rationale="The MFI is an RSI in which each candle is weighted by its volume in money terms, so "
              "it only reaches 80 when the rise has been made on growing volume. That state is one "
              "in which almost everybody who wanted to buy has already bought, and has done so with "
              "size on top of that: the incoming flow runs out for want of new participants, and "
              "the marginal position is leveraged and recent, which is to say fragile. Below 20 the "
              "symmetric thing happens with forced selling.",
    prior="We expect a positive edge in a ranging regime and a NEGATIVE one in a strong trend, "
          "because buying below 20 during a cascading fall is buying in front of a train. As the "
          "signal is a state and not an event, we also expect long runs of consecutive losses in "
          "the capitulations of March 2020 or May 2021. If the overall edge came out positive but "
          "only thanks to a handful of enormous bounces, we will count it as inconclusive and not "
          "as a success.",
    fn=_mfi14,
    params={"period": 14, "oversold": 20, "overbought": 80},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 5. Side of the session VWAP
# --------------------------------------------------------------------------------------------

def _vwap_side(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 10:
        return out
    vwap, _sd, _k = _session_vwap(s)
    c = _f(s.close)
    ok = _finite(vwap, c)
    out[ok & (c > vwap)] = 1
    out[ok & (c < vwap)] = -1
    return out


register(Hypothesis(
    name="flow.vwap_session_side",
    family="flow",
    rationale="The session VWAP is not just one more indicator: it is the price against which "
              "institutional execution is measured and settled, so there are real orders "
              "conditioned on it. An execution algorithm with a buy order slows down if the price "
              "is above the VWAP and speeds up if it is below, and the desks judge the trader by "
              "that difference. The mechanical consequence is that the side of the VWAP separates "
              "two different regimes of supply and demand within the same day, not merely two price "
              "ranges.",
    prior="We expect a small positive edge on 15m and 1h, degrading as the timeframe rises, because "
          "the daily anchor stops meaning anything once each candle is a third of the session. We "
          "expect it to fail on days with no direction, where the price crosses the VWAP ten times "
          "and only generates cost. BTC-specific falsification: crypto trades 24/7 and has no real "
          "institutional session, so if the mechanism is true the edge ought to be SMALLER than the "
          "one this same indicator shows in equities; if it came out enormous, we would suspect we "
          "are only measuring intraday momentum.",
    fn=_vwap_side,
    params={"anchor": "utc_day"},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 6. The VWAP's 2-sigma band: stretch against the volume-weighted average price
# --------------------------------------------------------------------------------------------

def _vwap_band(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 40:
        return out
    vwap, sd, k = _session_vwap(s)
    c = _f(s.close)
    ok = _finite(vwap, sd, c) & (sd > 0.0) & (k >= 12)
    out[ok & (c > vwap + 2.0 * sd)] = -1
    out[ok & (c < vwap - 2.0 * sd)] = 1
    return out


register(Hypothesis(
    name="flow.vwap_band2sigma",
    family="flow",
    rationale="It uses the same indicator as `vwap_session_side` but asserts the OPPOSITE mechanism, "
              "and that is why it is a separate hypothesis and not a variant: here what matters is "
              "not which side the price is on but how far away it has moved, measured in standard "
              "deviations of the session's own volume distribution. Two sigma above means the "
              "current price has been paid by a minuscule fraction of the day's volume: almost "
              "nobody holds a position there, and the inventory of the market maker who absorbed "
              "that rise is short and needs the price back in the zone where the bulk of the paper "
              "changed hands. The 2-sigma threshold is Bollinger's conventional one, not a searched "
              "value.",
    prior="We expect a positive edge on 15m and 1h within sessions with no news, and clear losses "
          "on breakout days, on which the price crosses the band in the morning and never comes "
          "back. Since fading wins a little many times and loses a lot a few times, we require the "
          "edge to survive scrutiny of the tail: if the result depends on not having lived through "
          "a 12 March 2020, it is a failure, not a success. We require at least 12 session candles "
          "before emitting a signal because with fewer the sigma is a three-point estimate; it is a "
          "statistical minimum, not a tuned threshold.",
    fn=_vwap_band,
    params={"anchor": "utc_day", "sigmas": 2.0, "min_session_bars": 12},
    timeframes=("15m", "1h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 7. Thrust: high effort WITH result (double volume and an expanding range)
# --------------------------------------------------------------------------------------------

def _volume_thrust(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 60:
        return out
    h, l, c, v = _f(s.high), _f(s.low), _f(s.close), _f(s.volume)
    v_mean = talib.SMA(v, 20)
    atr = talib.ATR(h, l, c, 14)
    tr = talib.TRANGE(h, l, c)
    clv = _clv(s)
    rel = np.full(n, np.nan)
    np.divide(v, v_mean, out=rel, where=~np.isnan(v_mean) & (v_mean > 0.0))
    ok = _finite(rel, atr, tr) & (atr > 0.0)
    strong = ok & (rel > 2.0) & (tr > atr)
    ev = np.zeros(n, dtype=np.int8)
    ev[strong & (clv > 0.5)] = 1
    ev[strong & (clv < -0.5)] = -1
    return _hold(ev, 5)


register(Hypothesis(
    name="flow.volume_thrust",
    family="flow",
    rationale="A candle with twice the average volume, a range greater than the ATR and a close in "
              "the top quarter of that range only happens when somebody crosses the spread over and "
              "over and takes out several levels of the book. Whoever does that is paying for "
              "immediacy, which reveals that their information or their urgency is worth more than "
              "the cost, and on top of that it leaves the book emptied on that side: the next "
              "levels have to be replenished higher up. The stops of the shorts trapped along the "
              "way supply additional forced buying over the following candles.",
    prior="We expect a positive edge over 5 candles and expect it to be LARGER on 15m than on 4h, "
          "because an emptied book is replenished in minutes or hours, not days. We expect it to "
          "fail, or to invert, when the thrust candle is the last of a long leg (exhaustion) and in "
          "the minutes after a macro release, where the expanded range is instantaneous repricing "
          "with no continuation. If the edge were the same on 4h and on 15m, the "
          "book-replenishment mechanism we assert would be false.",
    fn=_volume_thrust,
    params={"relative_volume": 2.0, "volume_mean": 20, "atr": 14, "clv": 0.5,
            "hold_bars": 5},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 8. Climax: 3-sigma volume at a 20-candle extreme, traded against the grain
# --------------------------------------------------------------------------------------------

def _volume_climax(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 140:
        return out
    c, v = _f(s.close), _f(s.volume)
    lv = np.log1p(v)   # log1p and not log: volume can be exactly zero
    mean = talib.SMA(lv, 100)
    sigma = talib.STDDEV(lv, 100)
    z = np.full(n, np.nan)
    np.divide(lv - mean, sigma, out=z, where=~np.isnan(sigma) & (sigma > 0.0))
    max_c, min_c = talib.MAX(c, 20), talib.MIN(c, 20)
    ok = _finite(z, max_c, min_c, c)
    extreme = ok & (z > 3.0)
    ev = np.zeros(n, dtype=np.int8)
    ev[extreme & (c >= max_c)] = -1
    ev[extreme & (c <= min_c)] = 1
    return _hold(ev, 5)


register(Hypothesis(
    name="flow.volume_climax",
    family="flow",
    rationale="Volume more than three sigma from its 100-period moving average, happening right at "
              "a 20-candle extreme, is not participation: it is transfer. In crypto that spike is "
              "almost always a liquidation cascade, in which the exchange's engine sends market "
              "orders that represent nobody who wants to trade at that price. When the liquidation "
              "queue empties, all that involuntary supply disappears at once, and the price goes "
              "back to the level where the real book was. The mean and the sigma are rolling over "
              "100 candles, never over the full history.",
    prior="We expect a positive edge over 5 candles, concentrated in a very small number of events "
          "(perhaps 30-80 across the whole history per timeframe), which caps the statistical power "
          "from the outset: if it comes out positive but with n < 30 we will declare it "
          "inconclusive. We expect it to fail when the volume spike is the START of a regime "
          "expansion and not its end —a range break with news behind it— where fading means "
          "stepping in front of the entire move. We also expect it to get worse from 2021 onwards, "
          "as the exchanges have gradually introduced partial-liquidation engines that smooth the "
          "cascades.",
    fn=_volume_climax,
    params={"volume_z": 3.0, "z_window": 100, "extreme_window": 20, "hold_bars": 5},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 9. Chaikin oscillator: momentum of the approximated volume delta
# --------------------------------------------------------------------------------------------

def _chaikin(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 40:
        return out
    osc = talib.ADOSC(_f(s.high), _f(s.low), _f(s.close), _f(s.volume),
                      fastperiod=3, slowperiod=10)
    ok = _finite(osc)
    out[ok & (osc > 0.0)] = 1
    out[ok & (osc < 0.0)] = -1
    return out


register(Hypothesis(
    name="flow.chaikin_osc_3_10",
    family="flow",
    rationale="The Accumulation/Distribution line is the cumulative volume delta that can be "
              "approximated without aggressor data: it splits each candle's volume between buying "
              "and selling according to where it closes inside the range, which is exactly what "
              "`taker_buy_base` would measure if `Series` exposed it. The Chaikin oscillator is its "
              "momentum (EMA 3 minus EMA 10), that is, whether the cumulative delta is "
              "accelerating. An accelerating delta means the imbalance between buy-side and "
              "sell-side aggression is growing, and that imbalance is what moves the price in the "
              "short run.",
    prior="We expect a small positive edge, smaller than `obv_ema21`'s, because splitting by the "
          "position of the close penalises exactly the candles with wicks, which in BTC are the "
          "informative ones. We expect it to fail systematically on candles with long wicks caused "
          "by stop runs, where the close comes back to the middle of the range and the indicator "
          "records 'indecision' at the very moment of maximum directional flow. If it came out "
          "better than OBV, the CLV aggression proxy would be more informative than we think and "
          "the whole family would need revisiting.",
    fn=_chaikin,
    params={"fast": 3, "slow": 10},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 10. Trend ONLY when participation expands
# --------------------------------------------------------------------------------------------

def _trend_with_participation(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 140:
        return out
    c, v = _f(s.close), _f(s.volume)
    roc = talib.ROC(c, 20)
    v20, v100 = talib.SMA(v, 20), talib.SMA(v, 100)
    ok = _finite(roc, v20, v100)
    expanding = ok & (v20 > v100)
    out[expanding & (roc > 0.0)] = 1
    out[expanding & (roc < 0.0)] = -1
    return out


register(Hypothesis(
    name="flow.trend_with_participation",
    family="flow",
    rationale="It asserts that volume does not give direction but does give permission. A sustained "
              "trend needs a continuous flow of new participants buying higher than the previous "
              "one did; if the 20-period average volume falls below the 100-period one, those still "
              "trading are the ones already inside, rotating among themselves, and there is nobody "
              "left to absorb the first wave of profit taking. The signal is the sign of the "
              "20-candle return conditioned on participation expanding, and zero when it does not.",
    prior="We expect a positive edge, but the real test is NOT that it be positive: it is that it "
          "be LARGER than plain 20-candle momentum with no filter. AUDIT NOTE (2026-09-08): that "
          "control was not registered in any family —the `trend` family does not register a 20-bar "
          "ROC— so this falsification criterion could not be executed; it has been registered as "
          "`flow.roc20_unfiltered` and that is what this must be measured against. "
          "If the volume filter adds nothing over that control, this hypothesis is refuted even if "
          "it makes money, because what it would be earning is the momentum and not the flow. We "
          "further expect the filter to hurt at bear-market bottoms, where volume dries up for "
          "months and the filter leaves out precisely the start of the next bull cycle.",
    fn=_trend_with_participation,
    params={"roc": 20, "volume_short": 20, "volume_long": 100},
    timeframes=("4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 11. A move on thin volume: fade it
# --------------------------------------------------------------------------------------------

def _move_without_volume(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 40:
        return out
    c, v = _f(s.close), _f(s.volume)
    v5, v20 = talib.SMA(v, 5), talib.SMA(v, 20)
    roc = talib.ROC(c, 5)
    ok = _finite(v5, v20, roc) & (v20 > 0.0)
    dry = ok & (v5 < v20)
    out[dry & (roc > 0.0)] = -1
    out[dry & (roc < 0.0)] = 1
    return out


register(Hypothesis(
    name="flow.move_without_volume",
    family="flow",
    rationale="The exact reverse of `trend_with_participation`, and that is why it is "
              "registered: a five-candle move made on volume below its own 20-period average has "
              "transferred no inventory, it has only walked the price through an empty book. Nobody "
              "has had to accept a large position on the wrong side, so there is no participant "
              "obliged to defend the new level, and the first order of size that turns up will put "
              "it back where it was. The threshold is 'below the average', with no free constant to "
              "tune.",
    prior="We expect a small positive edge on 15m/1h, above all at weekends and during the Asian "
          "hours, when BTC's book is thinnest. We expect it to fail during the slow starts of a "
          "trend, in which the price rises for weeks on decreasing volume and fading loses "
          "continuously; that is the failure mode that worries us most because it is persistent and "
          "not noisy. If this hypothesis and `trend_with_participation` both came out positive "
          "in the same regime, we would conclude that the volume filter separates nothing and that "
          "they win on the sign of the momentum, not on the flow.",
    fn=_move_without_volume,
    params={"volume_short": 5, "volume_long": 20, "roc": 5},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 12. Absorption: high effort WITHOUT result (double volume and a narrow range)
# --------------------------------------------------------------------------------------------

def _absorption(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 60:
        return out
    h, l, c, v = _f(s.high), _f(s.low), _f(s.close), _f(s.volume)
    v_mean = talib.SMA(v, 20)
    atr = talib.ATR(h, l, c, 14)
    tr = talib.TRANGE(h, l, c)
    clv = _clv(s)
    rel = np.full(n, np.nan)
    np.divide(v, v_mean, out=rel, where=~np.isnan(v_mean) & (v_mean > 0.0))
    ok = _finite(rel, atr, tr) & (atr > 0.0)
    absorbing = ok & (rel > 2.0) & (tr < atr)
    ev = np.zeros(n, dtype=np.int8)
    ev[absorbing & (clv > 0.0)] = 1
    ev[absorbing & (clv < 0.0)] = -1
    return _hold(ev, 5)


register(Hypothesis(
    name="flow.absorption_narrow_range",
    family="flow",
    rationale="It is the complementary, disjoint cell to `volume_thrust` in Wyckoff's "
              "effort/result table: the same double volume, but a range SMALLER than the ATR. Twice "
              "the usual number of contracts changing hands while the price does not move has only "
              "one mechanical explanation: there is a large passive order on one side of the book "
              "replenishing itself as fast as it is eaten. Whoever has the size to do that knows "
              "their target price and is not finished; when the aggressor tires, the price moves "
              "towards the absorber's side, which is the side the candle closed on. The direction "
              "is given by the sign of the close within the range, with no free threshold.",
    prior="We expect a small positive edge over 5 candles, and a lower frequency than the thrust. "
          "We expect it to fail in areas of accepted value —the middle of a long range— where high "
          "volume and a narrow range is simply the market's normal equilibrium and there is no "
          "absorber at all. Being disjoint from `volume_thrust` by construction, the comparison "
          "between the two is clean: if both come out positive, high relative volume is worth "
          "something on its own and the range adds nothing; if only one does, the effort/result "
          "axis is real.",
    fn=_absorption,
    params={"relative_volume": 2.0, "volume_mean": 20, "atr": 14, "hold_bars": 5},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 13. Negative Volume Index (Fosback): what price does on the thin-volume days
# --------------------------------------------------------------------------------------------

def _nvi_fosback(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 400:
        return out
    c, v = _f(s.close), _f(s.volume)
    ret = np.zeros(n, dtype=np.float64)
    prev = c[:-1]
    np.divide(c[1:] - prev, prev, out=ret[1:], where=prev > 0.0)
    lower_volume = np.zeros(n, dtype=bool)
    lower_volume[1:] = v[1:] < v[:-1]
    factor = np.where(lower_volume, 1.0 + ret, 1.0)
    factor = np.clip(factor, 0.01, None)   # no candle may drive the index to zero or below
    nvi = 1000.0 * np.cumprod(factor)
    ema = talib.EMA(nvi, 255)
    ok = _finite(nvi, ema)
    out[ok & (nvi > ema)] = 1
    return out


register(Hypothesis(
    name="flow.nvi_fosback",
    family="flow",
    rationale="The NVI accumulates the return ONLY of the candles whose volume falls relative to "
              "the previous one, isolating what price does when the public is not trading. "
              "Fosback's premise is that the retail crowd turns up with the volume and informed "
              "money does not need it: if the price rises on the quiet days, somebody is building a "
              "position without wanting to draw attention. The signal is asymmetric on purpose, as "
              "in the original: +1 when the NVI is above its 255-period EMA (a year of sessions) "
              "and 0 —not short— below it, because Fosback never claimed that the opposite "
              "indicated a bear market, only an absence of information.",
    prior="We expect a modest positive edge on 1d and a high exposure rate (it should be in the "
          "market for most of the bull run). The expected failure mode is twofold: the "
          "retail/informed distinction was built on exchanges with opening hours and press "
          "conferences, and BTC trades 24/7 with volume dominated by market makers and "
          "inter-exchange arbitrage, so the partition by volume may separate nobody. Besides, with "
          "only 1d and ~600 warm-up candles we are left with on the order of 2,500 observations and "
          "very few complete cycles: if the edge comes out positive but depends on having been in "
          "the market in 2020-2021, that is an observation, not evidence.",
    fn=_nvi_fosback,
    params={"ema": 255},
    timeframes=("1d",),
    min_warmup=600,
))


# --------------------------------------------------------------------------------------------
# 14. CONTROL ADDED DURING THE AUDIT (2026-09-08).
#
# What was wrong. Two hypotheses declared their falsification criterion against a control that did
# not exist: `flow.trend_with_participation` says that "the real test is NOT that it be positive:
# it is that it be LARGER than plain 20-candle momentum with no filter, which the `trend` family
# registers", and `momentum.momentum_with_volume` says that it "is falsified if the volume filter
# does not improve on the unfiltered 20-bar ROC". Neither `trend` nor `momentum` registered an
# unfiltered 20-bar ROC: `momentum` registers `roc10`, which is a different horizon and therefore a
# different trial.
#
# Why it mattered. The two hypotheses assert the same thing —that volume gives PERMISSION even
# though it gives no direction— and both condition the same statistic (the sign of the 20-candle
# return) on an expansion of participation. That is a claim about an increment, and without the
# unfiltered arm all that could be judged was absolute profitability, which in an asset with a
# positive secular drift confirms almost any mostly-long rule. With the control registered, the two
# become testable and all three count towards the multiple-comparisons correction.
#
# It is also declared here, for whoever runs the correction: `flow.trend_with_participation`
# (ROC-20 with volume SMA-20 > SMA-100) and `momentum.momentum_with_volume` (ROC-20 with volume
# SMA-5 > SMA-20) are the SAME construction with a different pair of volume averages. Measured on
# data they share little (Jaccard index ≈ 0.15 over the events), so they are not duplicates and
# neither is removed, but they are not independent trials either: they are two readings of the same
# mechanism and must be counted as dependent trials, not as two cross-family confirmations if both
# come out positive.
# --------------------------------------------------------------------------------------------

def _roc20_unfiltered(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 40:
        return out
    roc = talib.ROC(_f(s.close), 20)
    ok = _finite(roc)
    out[ok & (roc > 0.0)] = 1
    out[ok & (roc < 0.0)] = -1
    return out


register(Hypothesis(
    name="flow.roc20_unfiltered",
    family="flow",
    rationale="Unconditional control for `trend_with_participation` and for "
              "`momentum.momentum_with_volume`: the sign of the 20-candle return, with no condition "
              "on volume whatsoever. It asserts no flow mechanism —it cannot, because it does not "
              "look at volume— and that is exactly its role. The two hypotheses it controls do not "
              "bet that 20-candle momentum wins, but that it wins MORE when participation expands; "
              "that is a claim about a difference, and a difference needs both arms. It is "
              "registered in `flow` because its only reason to exist is to serve as the denominator "
              "for this family's volume filter.",
    prior="We expect a small positive edge, and we expect it to be SMALLER than "
          "`trend_with_participation`'s on 4h and 1d. All the information is in the comparison: "
          "if this control matches or beats the filtered versions, volume gives no permission, the "
          "two filtered hypotheses are refuted even if they make money, and this family's thesis "
          "—that volume measures conviction— loses its main support. On its own, a positive result "
          "from this control is NOT a finding: it is BTC's secular drift read through the sign of a "
          "past return, and it must be published as such.",
    fn=_roc20_unfiltered,
    params={"roc": 20, "volume_filter": "none"},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))
