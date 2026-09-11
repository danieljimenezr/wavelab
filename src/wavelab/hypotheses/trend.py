"""`trend` family: trend following. PRE-REGISTRATION — written before looking at a single number.

None of these twelve hypotheses has been run. The parameters are the conventional ones from the
literature (EMA 21/55/200, MACD 12/26/9, ADX/DMI 14, ATR 14, Donchian 55/20 from Turtle System 2,
Ichimoku 9/26/52, Aroon 25, SAR 0.02/0.2). There are no variants of the same parameter: where two
numbers appear (Donchian's 55/20, Supertrend's 3xATR) they are a pair published as a unit, and the
`rationale` says why they are one hypothesis and not a grid.

The family shares a single question: **why would a trend persist?** Each hypothesis proposes a
DIFFERENT mechanism (order slicing, agreement across horizons, a reflexive focal point, flow
acceleration, the resolution of disagreement, a stop cascade, stop distance, renewal of extremes,
drift over noise, the allocation horizon). Several of them predict results that CONTRADICT one
another (e.g. `supertrend` vs `psar` on whether to give the position room or tighten the stop).
That is deliberate: if they could all win at once, none of them would be falsifiable.

Causality. Every indicator here is a backward-looking window function. The only shift used is
`_lag(a, k)` with k >= 1, which brings the value from k bars ago to the current index; there is no
negative shift, no `np.roll`, and no extreme detector defined against the whole array. Two places
deserve to be spelled out:

  - Donchian: `talib.MAX(high, 55)[i]` INCLUDES bar i, so comparing `close[i]` against it would not
    break — but it fails to break for the wrong reason (`high[i] >= close[i]` always). The
    reference channel is lagged by one bar so that it is the maximum of the 55 PRECEDING bars.
  - Ichimoku: the cloud is displaced by +26. That means TODAY's level was computed from data 26
    bars old. It is past information drawn forward, never future information.

NaN. talib returns NaN during warm-up. `nan_to_num` is used nowhere here: an explicit mask is built
with `np.isfinite` and signals are written only where ALL the inputs are finite. Filling a NaN with
0 or with 1e18 invents signals at the start of the series.
"""

from __future__ import annotations

import numpy as np
import talib

from wavelab.hypotheses.base import Hypothesis, Series, register

# --------------------------------------------------------------------------------------------
# utilities
# --------------------------------------------------------------------------------------------


def _f64(a: np.ndarray) -> np.ndarray:
    """talib requires contiguous float64."""
    return np.ascontiguousarray(a, dtype=np.float64)


def _lag(a: np.ndarray, k: int) -> np.ndarray:
    """The value from `k` bars ago, read at the current bar. k >= 1, ALWAYS into the past.

    The head is left as NaN on purpose: there is no past value to bring forward, and any fill
    would manufacture signal where there is no information.
    """
    if k < 1:
        raise ValueError("_lag only shifts into the past (k >= 1)")
    out = np.full(a.shape, np.nan, dtype=np.float64)
    if k < a.size:
        out[k:] = a[: a.size - k]
    return out


def _ok(*arrays: np.ndarray) -> np.ndarray:
    """Mask of the positions where every input is finite."""
    m = np.isfinite(arrays[0])
    for a in arrays[1:]:
        m &= np.isfinite(a)
    return m


# --------------------------------------------------------------------------------------------
# 1. two-moving-average cross: order slicing
# --------------------------------------------------------------------------------------------


def _ema_cross_21_55(s: Series) -> np.ndarray:
    c = _f64(s.close)
    out = np.zeros(c.size, dtype=np.int8)
    fast = talib.EMA(c, 21)
    slow = talib.EMA(c, 55)
    ok = _ok(fast, slow)
    out[ok & (fast > slow)] = 1
    out[ok & (fast < slow)] = -1
    return out


register(Hypothesis(
    name="trend.ema_cross_21_55",
    family="trend",
    title="Side with the faster of two averages",
    rationale="New information is not incorporated within one candle because the large participant "
              "cannot execute within one candle: a mandate of any size is sliced over days so as "
              "not to move the price against itself, so today's buyer is also tomorrow's buyer. "
              "That slicing leaves positive autocorrelation in the drift at the scale of weeks. "
              "The EMA21/EMA55 cross predicts nothing: it merely declares that the flow imbalance "
              "is still live, and it gets paid for as long as the same participant keeps "
              "executing.",
    prior="We expect a positive edge in a trending regime and a NEGATIVE one in a range, with the "
          "loss concentrated in a few weeks of whipsaw. If the edge were positive in ranges too, "
          "the slicing mechanism would be false and we would be measuring something else. "
          "Falsifiable cross-prediction: the effect must degrade as the timeframe drops, because "
          "sliced institutional execution does not live at the 15m scale.",
    fn=_ema_cross_21_55,
    params={"fast": 21, "slow": 55},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=250,
))


# --------------------------------------------------------------------------------------------
# 2. three horizons aligned: the opposing liquidity withdraws
# --------------------------------------------------------------------------------------------


def _ema_stack_21_55_200(s: Series) -> np.ndarray:
    c = _f64(s.close)
    out = np.zeros(c.size, dtype=np.int8)
    e21 = talib.EMA(c, 21)
    e55 = talib.EMA(c, 55)
    e200 = talib.EMA(c, 200)
    ok = _ok(e21, e55, e200)
    out[ok & (e21 > e55) & (e55 > e200)] = 1
    out[ok & (e21 < e55) & (e55 < e200)] = -1
    return out


register(Hypothesis(
    name="trend.ema_stack_21_55_200",
    family="trend",
    title="Only when all three horizons agree",
    rationale="Each horizon belongs to a different participant: intraday, swing and allocator. "
              "While they disagree, whoever is positioned against the move provides the liquidity "
              "that absorbs whoever is positioned with it, and price reverts. When all three "
              "agree, nobody is left structurally obliged to sell into the advance, depth on the "
              "opposite side withdraws and the SAME flow moves more price. The hypothesis is not "
              "about moving averages but about the absence of a natural counterparty.",
    prior="We expect far fewer bars in the market than trend.ema_cross_21_55 and better "
          "performance PER BAR. Falsifiable in two ways: if time in market does not fall clearly, "
          "the agreement condition is filtering nothing; and if per-bar performance does not beat "
          "the plain cross, the 'absence of counterparty' adds nothing. We expect it to go flat "
          "and give up the whole opening move at every regime change, because the EMA200 aligns "
          "late.",
    fn=_ema_stack_21_55_200,
    params={"short": 21, "medium": 55, "long": 200},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=600,
))


# --------------------------------------------------------------------------------------------
# 3. the 200 average: a reflexive focal point
# --------------------------------------------------------------------------------------------


def _ema200_filter(s: Series) -> np.ndarray:
    c = _f64(s.close)
    out = np.zeros(c.size, dtype=np.int8)
    e200 = talib.EMA(c, 200)
    ok = _ok(c, e200)
    out[ok & (c > e200)] = 1
    out[ok & (c < e200)] = -1
    return out


register(Hypothesis(
    name="trend.ema200_filter",
    family="trend",
    title="Long above the EMA 200, short below",
    rationale="The 200 average measures no physical property of the market: it is a public focal "
              "point that risk desks, the financial press and allocation bots quote daily. Its "
              "power, if it has any, is entirely reflexive — there are real mandates that cut "
              "exposure 'below the 200', and that cut is genuine flow executed at the level. We "
              "register the bare level and not the slope because for an EMA200 the slope is very "
              "nearly a function of the cross itself, and adding it would be the same trial run "
              "twice.",
    prior="The prediction that matters is a cross-prediction, not one about profit: if the "
          "mechanism is reflexive, the effect must be LARGER on 1d, which is where the level is "
          "published and watched, and close to nil on 15m and 1h. An edge equal to or greater on "
          "15m would falsify the explanation even if the number came out profitable, and in that "
          "case the hypothesis must be declared failed despite making money. We expect a nil or "
          "negative edge in periods without a secular trend.",
    fn=_ema200_filter,
    params={"period": 200},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=600,
))


# --------------------------------------------------------------------------------------------
# 4. MACD: acceleration of the imbalance
# --------------------------------------------------------------------------------------------


def _macd_12_26_9(s: Series) -> np.ndarray:
    c = _f64(s.close)
    out = np.zeros(c.size, dtype=np.int8)
    macd, signal, _ = talib.MACD(c, 12, 26, 9)
    ok = _ok(macd, signal)
    out[ok & (macd > signal)] = 1
    out[ok & (macd < signal)] = -1
    return out


register(Hypothesis(
    name="trend.macd_12_26_9",
    family="trend",
    title="Trade MACD acceleration, not the level",
    rationale="The MACD against its signal line measures the ACCELERATION of the flow imbalance, "
              "not its level. Systematic momentum programmes scale size continuously as a "
              "function of the signal rather than in binary fashion: when the drift accelerates "
              "they add to the position, and that addition is literally tomorrow's drift. The bet "
              "is that this feedback loop shows up in the second derivative before it shows up in "
              "price.",
    prior="We expect more position changes and a lower hit rate per trade than "
          "trend.ema_cross_21_55, and a similar GROSS edge that may disappear net of costs. "
          "Directly falsifiable: if the MACD does not beat the moving-average cross net, "
          "'anticipating through acceleration' has no content and it is an expensive moving "
          "average. We expect a clearly negative edge in a low-volatility range, where "
          "acceleration is only noise.",
    fn=_macd_12_26_9,
    params={"fast": 12, "slow": 26, "signal": 9},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=250,
))


# --------------------------------------------------------------------------------------------
# 5. ADX/DMI: the dispersion of opinion has resolved
# --------------------------------------------------------------------------------------------


def _adx_dmi_14(s: Series) -> np.ndarray:
    h, l, c = _f64(s.high), _f64(s.low), _f64(s.close)
    out = np.zeros(c.size, dtype=np.int8)
    adx = talib.ADX(h, l, c, 14)
    pdi = talib.PLUS_DI(h, l, c, 14)
    mdi = talib.MINUS_DI(h, l, c, 14)
    ok = _ok(adx, pdi, mdi)
    strong = ok & (adx >= 25.0)
    out[strong & (pdi > mdi)] = 1
    out[strong & (mdi > pdi)] = -1
    return out


register(Hypothesis(
    name="trend.adx_dmi_14",
    family="trend",
    title="Trade direction only above ADX 25",
    rationale="Separates two questions that moving averages run together: is there a trend? (ADX) "
              "and in which direction? (DMI). The proposed mechanism is that persistence only "
              "appears once the dispersion of opinion has already resolved: if most of the "
              "aggressive volume arrives on the same side, the market maker accumulates "
              "directional inventory and hedges IN THE SAME DIRECTION, amplifying the very move "
              "that is hurting him. Below ADX 25 there are two camps of similar size and the flow "
              "cancels out without leaving any drift.",
    prior="We expect the edge to come from AVOIDING the range, not from getting the direction "
          "right more often. Explicit falsifiable test: per-bar performance INSIDE the filter must "
          "beat that of the bars the filter discards; if it does not, the ADX carries no "
          "information and the hypothesis dies even if the overall result is positive. We expect "
          "it to enter late and give up a large part of the start of every trend, because the ADX "
          "needs a move that has already happened in order to climb above 25.",
    fn=_adx_dmi_14,
    params={"period": 14, "adx_threshold": 25},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=250,
))


# --------------------------------------------------------------------------------------------
# 6. Donchian 55/20: stop cascade
# --------------------------------------------------------------------------------------------


def _donchian_turtle_55_20(s: Series) -> np.ndarray:
    h, l, c = _f64(s.high), _f64(s.low), _f64(s.close)
    n = c.size
    out = np.zeros(n, dtype=np.int8)
    # +1 bar of lag: channel of the 55 (or 20) PRECEDING bars, excluding the current one.
    hh_entry = _lag(talib.MAX(h, 55), 1)
    ll_entry = _lag(talib.MIN(l, 55), 1)
    hh_exit = _lag(talib.MAX(h, 20), 1)
    ll_exit = _lag(talib.MIN(l, 20), 1)
    ok = _ok(hh_entry, ll_entry, hh_exit, ll_exit, c)
    pos = 0
    for i in range(n):
        if not ok[i]:
            pos = 0
            out[i] = 0
            continue
        if pos == 0:
            if c[i] > hh_entry[i]:
                pos = 1
            elif c[i] < ll_entry[i]:
                pos = -1
        elif pos == 1:
            if c[i] < ll_exit[i]:
                pos = -1 if c[i] < ll_entry[i] else 0
        else:
            if c[i] > hh_exit[i]:
                pos = 1 if c[i] > hh_entry[i] else 0
        out[i] = pos
    return out


register(Hypothesis(
    name="trend.donchian_turtle_55_20",
    family="trend",
    title="Break the 55-bar high, exit at 20",
    rationale="Above a 55-bar high, short sellers' stops and breakout buy orders pile up. Their "
              "execution is a mechanical demand shock, insensitive to price, and the book is thin "
              "right there precisely because nobody has been willing to sell at that level yet; "
              "every stop that fills pushes into the next one. The 55/20 pair is Turtle System 2 "
              "published as a unit, not a grid: the shorter exit exists so that the position dies "
              "before the capital does, and separating the two numbers would be a different "
              "hypothesis.",
    prior="We expect a very asymmetric distribution: mostly small losing trades and a right tail "
          "that carries the whole result — that is, a positive MEAN with a negative MEDIAN. If the "
          "median comes out positive, what is at work is not the stop cascade but something else, "
          "and the declared mechanism is refuted. We expect clear failure in high-volatility "
          "markets without direction, where the false breakout is the norm.",
    fn=_donchian_turtle_55_20,
    params={"entry": 55, "exit": 20},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=250,
))


# --------------------------------------------------------------------------------------------
# 7. Supertrend: the stop where a risk desk would put it
# --------------------------------------------------------------------------------------------


def _supertrend_atr14_x3(s: Series) -> np.ndarray:
    h, l, c = _f64(s.high), _f64(s.low), _f64(s.close)
    n = c.size
    out = np.zeros(n, dtype=np.int8)
    atr = talib.ATR(h, l, c, 14)
    hl2 = (h + l) / 2.0
    upper_band = hl2 + 3.0 * atr
    lower_band = hl2 - 3.0 * atr
    ok = _ok(upper_band, lower_band, c)
    upper = np.nan
    lower = np.nan
    pos = 0
    for i in range(1, n):
        if not ok[i]:
            continue
        if not np.isfinite(upper):
            upper, lower = upper_band[i], lower_band[i]
            continue
        # ratchet: the band only tightens, and is released only when price breaks through it.
        if upper_band[i] < upper or c[i - 1] > upper:
            upper = upper_band[i]
        if lower_band[i] > lower or c[i - 1] < lower:
            lower = lower_band[i]
        if c[i] > upper:
            pos = 1
        elif c[i] < lower:
            pos = -1
        out[i] = pos
    return out


register(Hypothesis(
    name="trend.supertrend_atr14_x3",
    family="trend",
    title="Trail a wide stop, three ATR out",
    rationale="An invalidation threshold fixed as a percentage is incoherent across volatility "
              "regimes: the same 2% is noise in one regime and a signal in another. Supertrend "
              "puts the flip 3 ATR away from the midpoint, which is roughly where a risk desk "
              "places the stop — far enough that ordinary noise does not touch it. If real stops "
              "cluster at that distance, the level stops being a drawn line and starts having flow "
              "behind it. The band's ratchet is the hypothesis that it pays to GIVE the position "
              "room.",
    prior="We expect the same sign as the moving-average cross in a trend, but far fewer position "
          "changes thanks to the band's hysteresis, and that ALL the edge over the cross comes "
          "from that reduction and not from better accuracy. Falsifiable: if the number of "
          "position changes does not fall clearly relative to trend.ema_cross_21_55, the "
          "hysteresis is doing nothing. We expect it to give back a lot of profit on sharp "
          "reversals, precisely because of how far away the level sits; we predict it loses to "
          "trend.psar_002_020 in markets that turn in a V.",
    fn=_supertrend_atr14_x3,
    params={"atr": 14, "multiplier": 3.0},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=250,
))


# --------------------------------------------------------------------------------------------
# 8. parabolic SAR: the opposing hypothesis — tighten the stop
# --------------------------------------------------------------------------------------------


def _psar_002_020(s: Series) -> np.ndarray:
    h, l, c = _f64(s.high), _f64(s.low), _f64(s.close)
    out = np.zeros(c.size, dtype=np.int8)
    sar = talib.SAR(h, l, acceleration=0.02, maximum=0.2)
    ok = _ok(sar, c)
    out[ok & (c > sar)] = 1
    out[ok & (c < sar)] = -1
    return out


register(Hypothesis(
    name="trend.psar_002_020",
    family="trend",
    title="Tighten the stop with every new high",
    rationale="The SAR tightens the stop as the trend matures, accelerating with every new "
              "extreme. It reproduces an observable behaviour: the manager raises the stop after "
              "each new high to protect unrealised profit, and that staircase of stops is latent "
              "supply which, once touched, executes as a cascade. It is registered explicitly as "
              "the OPPOSITE hypothesis to trend.supertrend_atr14_x3 on the same question: whether "
              "to tighten the stop or to give it room.",
    prior="We predict that SAR and Supertrend disagree systematically about how long the position "
          "is held and that ONE of the two is clearly worse; we bet on the SAR losing net through "
          "excess trading. The result that would refute the whole question is that both perform "
          "the same: that would mean stop distance is not a relevant variable and that both "
          "rationalisations are noise. The SAR is always in the market, so we expect severe damage "
          "in a range.",
    fn=_psar_002_020,
    params={"acceleration": 0.02, "maximum": 0.2},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=250,
))


# --------------------------------------------------------------------------------------------
# 9. regression slope over ATR: drift measured in units of noise
# --------------------------------------------------------------------------------------------


def _linreg_slope_55_atr14(s: Series) -> np.ndarray:
    h, l, c = _f64(s.high), _f64(s.low), _f64(s.close)
    out = np.zeros(c.size, dtype=np.int8)
    slope = talib.LINEARREG_SLOPE(c, 55)
    atr = talib.ATR(h, l, c, 14)
    ok = _ok(slope, atr) & (atr > 0.0)
    # cumulative displacement of the window, in ATR: requiring > 1 is requiring that the drift
    # dominate the noise in the noise's own unit. It is not a tuned threshold: it is a change of
    # unit.
    ratio = np.full(c.size, np.nan, dtype=np.float64)
    ratio[ok] = (slope[ok] * 55.0) / atr[ok]
    out[ok & (ratio > 1.0)] = 1
    out[ok & (ratio < -1.0)] = -1
    return out


register(Hypothesis(
    name="trend.linreg_slope_55_atr14",
    family="trend",
    title="Only when the drift outruns the noise",
    rationale="Drift is only exploitable relative to the noise you have to wade through to collect "
              "it; a market that grinds higher in the middle of a gale is not tradeable. The "
              "55-bar regression slope estimates the drift and the ATR14 the noise per bar, and "
              "requiring the window's cumulative displacement to exceed 1 ATR is a change of unit, "
              "not a threshold fitted to data. It is the most literal formulation of 'trend' in "
              "the whole family: no crosses, no focal points and no memory of extremes.",
    prior="We expect this to be the detector with the FEWEST trades and, above all, the most "
          "stable one across timeframes, because the threshold is dimensionless. That is the "
          "falsifiable test: if its edge does not survive a change of timeframe better than "
          "trend.ema_cross_21_55's does, normalising by volatility adds nothing. We expect it to "
          "fail exactly at volatility jumps, because the ATR reacts late and the threshold "
          "loosens precisely when it should be tightening.",
    fn=_linreg_slope_55_atr14,
    params={"window": 55, "atr": 14, "threshold_in_atr": 1.0},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=250,
))


# --------------------------------------------------------------------------------------------
# 10. Ichimoku: the only popular construction with THICKNESS of equilibrium
# --------------------------------------------------------------------------------------------


def _ichimoku_kumo(s: Series) -> np.ndarray:
    h, l, c = _f64(s.high), _f64(s.low), _f64(s.close)
    out = np.zeros(c.size, dtype=np.int8)
    tenkan = (talib.MAX(h, 9) + talib.MIN(l, 9)) / 2.0
    kijun = (talib.MAX(h, 26) + talib.MIN(l, 26)) / 2.0
    # +26 displacement: TODAY's level comes out of data from 26 bars ago. Past, not future.
    span_a = _lag((tenkan + kijun) / 2.0, 26)
    span_b = _lag((talib.MAX(h, 52) + talib.MIN(l, 52)) / 2.0, 26)
    ok = _ok(tenkan, kijun, span_a, span_b, c)
    cloud_top = np.maximum(span_a, span_b)
    cloud_bottom = np.minimum(span_a, span_b)
    out[ok & (c > cloud_top) & (tenkan > kijun)] = 1
    out[ok & (c < cloud_bottom) & (tenkan < kijun)] = -1
    return out


register(Hypothesis(
    name="trend.ichimoku_kumo",
    family="trend",
    title="Above the Ichimoku cloud, flat inside it",
    rationale="The cloud is the only popular construction that declares the equilibrium zone with "
              "THICKNESS: between Senkou A and B there is no signal, and the thickness grows "
              "exactly when the recent extremes disagree with each other. It encodes the memory of "
              "the 9-, 26- and 52-bar ranges, the review horizons of a Japanese desk, and its "
              "value — if it has any — is reflexive, given how enormously the system is adopted in "
              "Asia. The +26 displacement is what makes it interesting and also what makes it easy "
              "to implement wrong: it is information from 26 bars ago drawn forward, never future "
              "information.",
    prior="We expect an edge similar to the moving-average cross but with considerably MORE time "
          "out of the market, because the cloud declares equilibrium explicitly. Falsifiable: if "
          "time out of the market does not increase clearly relative to trend.ema_cross_21_55, the "
          "cloud filters nothing and is a slow moving average with two coats of paint. We expect "
          "it to fail on trends that start from compression, where the cloud is thin and stops "
          "nothing.",
    fn=_ichimoku_kumo,
    params={"tenkan": 9, "kijun": 26, "senkou_b": 52, "displacement": 26},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=250,
))


# --------------------------------------------------------------------------------------------
# 11. Aroon: the trend as a renewal process
# --------------------------------------------------------------------------------------------


def _aroon_25(s: Series) -> np.ndarray:
    h, l = _f64(s.high), _f64(s.low)
    out = np.zeros(h.size, dtype=np.int8)
    down, up = talib.AROON(h, l, 25)
    ok = _ok(up, down)
    out[ok & (up > 70.0) & (down < 30.0)] = 1
    out[ok & (down > 70.0) & (up < 30.0)] = -1
    return out


register(Hypothesis(
    name="trend.aroon_25",
    family="trend",
    title="Time since the last high, not price",
    rationale="Aroon measures not price but TIME since the last extreme, and that is a different "
              "variable from everything else in this family. The thesis is that a trend is a "
              "renewal process: as long as new highs keep appearing frequently, there are buyers "
              "willing to pay prices that have never been paid, which only happens if information "
              "is still being incorporated. When the rate of new extremes falls, the flow has "
              "exhausted itself even if price has not turned yet.",
    prior="We expect it to exit BEFORE the moving averages at tops — giving back less profit on "
          "the turn — and to pay for that insurance with clearly worse entries. Falsifiable: if "
          "its pattern of results is indistinguishable from trend.ema_cross_21_55's across every "
          "regime, the time information is redundant with the price information and the hypothesis "
          "adds nothing new to the family. We expect a nil or negative edge in a range with "
          "alternating extremes, where both arms stay high and the signal switches off.",
    fn=_aroon_25,
    params={"period": 25, "high_threshold": 70, "low_threshold": 30},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=250,
))


# --------------------------------------------------------------------------------------------
# 12. twelve-month time-series momentum: the allocation horizon
# --------------------------------------------------------------------------------------------


def _tsmom_365d(s: Series) -> np.ndarray:
    c = _f64(s.close)
    out = np.zeros(c.size, dtype=np.int8)
    a_year_ago = _lag(c, 365)
    ok = _ok(c, a_year_ago) & (a_year_ago > 0.0)
    out[ok & (c > a_year_ago)] = 1
    out[ok & (c < a_year_ago)] = -1
    return out


register(Hypothesis(
    name="trend.tsmom_365d",
    family="trend",
    title="Above where it traded a year ago",
    rationale="This is the trend hypothesis with the broadest cross-asset academic support "
              "(Moskowitz, Ooi and Pedersen): the sign of the last twelve months' return predicts "
              "the sign of the following period in almost every asset class and across more than a "
              "century of data. The proposed mechanism is initial underreaction through anchoring "
              "and slow diffusion, followed by overreaction driven by trend-follower flow. In BTC "
              "the allocation cycle adds to it: a committee approves mandates quarters after the "
              "decision that motivated them, and that lag is exactly the persistence. It is "
              "registered on 1d only because 'twelve months' is a scale of the allocation "
              "calendar, not a number of bars: over 365 1h candles the declared mechanism does not "
              "exist.",
    prior="We expect the smallest edge PER BAR in the whole family and, at the same time, the most "
          "stable one, positive in the out-of-sample years as well. Falsifiable test against a "
          "sibling: if the 365-day sign adds nothing on top of what trend.linreg_slope_55_atr14 "
          "already gives, the allocation-horizon argument is false and all we are measuring is "
          "drift at any scale whatsoever. We expect a very large drawdown at every cycle turn, "
          "because it takes months to change sign.",
    fn=_tsmom_365d,
    params={"lag_days": 365},
    timeframes=("1d",),
    min_warmup=420,
))
