"""Family `mean_reversion`: price returns to an anchor after moving too far from it.

PRE-REGISTERED. Everything in this file was written BEFORE running a single backtest and before
looking at a single number. The parameters are the conventional ones from the literature and they
are FIXED: there are no variants of the same parameter "just to see". The 12 hypotheses are
registered together and all 12 must be counted in the multiple-comparisons correction, the ones
that fail included.

The market mechanism the family shares —what would have to be true for this to work— has three
parts:

  1. Liquidity provision. When a large, impatient order sweeps the book, the price moves further
     than the information that order carries justifies. Whoever is on the other side charges for
     absorbing it. That charge is the reversion premium, and it is the only "real" income this
     family can capture.
  2. Overreaction. Traders extrapolate the last leg and overweight what is recent; the price moves
     away from consensus further than the flow of information justifies and then corrects.
  3. Forced closing. In a market with high leverage (BTC perpetuals), cascading liquidations are
     sales that do NOT express an opinion about value. They are mechanical and they run out when
     the collateral does. The hole they leave gets filled.

And the failure mode it shares, also declared in advance: in a sustained trend, this whole family
sells strength and buys weakness at exactly the moment when that is the opposite of what to do. A
reversion statistic "pinned" at the extreme throughout a directional leg is not a signal, it is a
slow loss. Several of the hypotheses below exist precisely in order to measure that failure instead
of hiding it (`bb20_2_range_adx14` and `rsi14_cardwell_sma200` against their versions with no
regime filter).

A note on `min_warmup`. It is a decision of numerical hygiene, not a parameter of the signal: it is
chosen to discard the indicators' start-up (and the unstable stretch of the long EMAs, ~2x the
period), not to improve any result. It is not touched after seeing data.
"""

from __future__ import annotations

import numpy as np
import talib

from wavelab.hypotheses.base import Hypothesis, Series, register

# --------------------------------------------------------------------------------------------
# Utilities. None of them looks ahead.
# --------------------------------------------------------------------------------------------


def _f(x: np.ndarray) -> np.ndarray:
    """talib demands contiguous float64; otherwise it raises 'input array type is not double'."""
    return np.ascontiguousarray(x, dtype=np.float64)


def _prev(a: np.ndarray) -> np.ndarray:
    """Value of the PREVIOUS bar, aligned to i (that is, a[i-1] placed at position i).

    The shift runs from the past towards the present, never the other way round. `np.roll(a, 1)` is
    banned here: it puts a[-1] —the last data point of the whole series— at position 0, which is
    exactly the information leak this project is hunting.
    """
    out = np.empty_like(a)
    out[0] = np.nan
    out[1:] = a[:-1]
    return out


def _finite(*arrays: np.ndarray) -> np.ndarray:
    """Mask of the positions where ALL the indicators are defined.

    talib returns NaN during warm-up. A comparison against NaN yields False, so it does not invent a
    long; but it can invent a short if the condition is written negated. That is why the check is
    explicit, with np.isnan, and is ANDed into every condition, instead of relying on the implicit
    behaviour or on nan_to_num (which WOULD invent signals, by replacing the NaN with a number that
    compares true).
    """
    ok = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        ok &= ~np.isnan(a)
    return ok


def _sig(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int8)


# --------------------------------------------------------------------------------------------
# 1. Bollinger bands: the naked fade.
# --------------------------------------------------------------------------------------------


def _bb_fade(s: Series) -> np.ndarray:
    c = _f(s.close)
    out = _sig(c.size)
    upper, _mid, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
    ok = _finite(upper, lower)
    out[ok & (c < lower)] = 1
    out[ok & (c > upper)] = -1
    return out


register(Hypothesis(
    name="mean_reversion.bb20_2_fade",
    family="mean_reversion",
    rationale="Closing outside the 2-sigma band over 20 candles means the price has moved further "
              "than the asset's own recent dispersion considers normal. The behaviour that would "
              "produce the effect is that of the market maker and the range trader: they see a "
              "temporary imbalance of flow —an impatient order sweeping the book— and charge a "
              "premium for absorbing it, handing the price back towards the SMA20 once that order "
              "runs out. It is the barest possible formulation of the family and it serves as the "
              "baseline against which the other eleven are measured.",
    prior="We expect a positive but small edge in a ranging regime and a NEGATIVE one in a trending "
          "regime, because in a trend the price 'walks the band' for dozens of candles and this "
          "rule holds the wrong side that entire time. Net over the whole sample we expect "
          "something close to zero, or negative after costs. It is refuted if the edge is positive "
          "and stable in both regimes: that would mean we are not measuring liquidity provision but "
          "something else.",
    fn=_bb_fade,
    params={"period": 20, "deviations": 2.0},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 2. Bollinger with re-entry. SAME parameters, DIFFERENT decision rule (this is not a parameter
#    variant: #1 enters during the excursion, this one waits for the excursion to end).
# --------------------------------------------------------------------------------------------


def _bb_reentry(s: Series) -> np.ndarray:
    c = _f(s.close)
    n = c.size
    out = _sig(n)
    upper, mid, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)

    pos = 0
    for i in range(1, n):
        # Only indices i and i-1 are read. Never i+1.
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(mid[i]) \
                or np.isnan(upper[i - 1]) or np.isnan(lower[i - 1]):
            pos = 0
            out[i] = 0
            continue
        # Exit first: the target is the moving average, which is the hypothesis's anchor.
        if pos == 1 and c[i] >= mid[i] or pos == -1 and c[i] <= mid[i]:
            pos = 0
        # Entry only if we are flat and the previous candle closed OUTSIDE and this one closes IN.
        if pos == 0:
            if c[i - 1] < lower[i - 1] and c[i] >= lower[i]:
                pos = 1
            elif c[i - 1] > upper[i - 1] and c[i] <= upper[i]:
                pos = -1
        out[i] = pos
    return out


register(Hypothesis(
    name="mean_reversion.bb20_2_reentry",
    family="mean_reversion",
    rationale="The same band as hypothesis 1 and the same 20/2, but the decision rule is the "
              "opposite one in time: you do not buy the fall, you buy the end of the fall. "
              "Requiring the previous candle to have closed outside and this one inside is asking "
              "for proof that the impatient flow has already run out, instead of assuming it. The "
              "concrete participant is the forced seller: while they are liquidating, the price "
              "stays outside the band; when they finish, the first candle that closes back inside "
              "marks that the imbalance has been consumed. It is registered alongside #1 because "
              "the comparison between the two is what isolates the cost of anticipating against the "
              "cost of waiting.",
    prior="We expect FEWER trades and a BETTER edge per trade than hypothesis 1, and above all a "
          "much shorter left tail, because we do not enter against a liquidation in progress. We "
          "expect it to lose in markets with very fast V-shaped reversals, where the re-entry "
          "candle arrives once the move back has already happened. It is refuted if its edge per "
          "trade does not beat hypothesis 1's: in that case the 'proof of exhaustion' carries no "
          "information and merely arrives late.",
    fn=_bb_reentry,
    params={"period": 20, "deviations": 2.0, "exit": "SMA20"},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 3. Bollinger filtered by regime. It exists to MEASURE the family's declared failure mode.
# --------------------------------------------------------------------------------------------


def _bb_range_adx(s: Series) -> np.ndarray:
    c, h, low = _f(s.close), _f(s.high), _f(s.low)
    out = _sig(c.size)
    upper, _mid, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
    adx = talib.ADX(h, low, c, timeperiod=14)
    ok = _finite(upper, lower, adx)
    quiet = ok & (adx < 20.0)
    out[quiet & (c < lower)] = 1
    out[quiet & (c > upper)] = -1
    return out


register(Hypothesis(
    name="mean_reversion.bb20_2_range_adx14",
    family="mean_reversion",
    rationale="Hypothesis 1 with a single gate: trade only when the ADX14 is below 20, Wilder's "
              "threshold for 'no trend'. The claim being tested is structural, not cosmetic: the "
              "reversion premium exists because somebody absorbs impatient flow, and absorbing is "
              "only profitable when that flow is NOT informed. When there is a trend, the order "
              "sweeping the book usually carries information —somebody is genuinely "
              "repositioning— and whoever absorbs it is adversely selected. A low ADX is the "
              "classic approximation to 'nobody here is repositioning'.",
    prior="We expect a CLEARLY larger edge than hypothesis 1 over the same candles, and we expect "
          "the difference to come from removing losses, not from adding gains. We also expect fewer "
          "than half the trades. The hypothesis is refuted if the edge does not improve on #1's: "
          "that would mean the ADX finds out about the regime too late to work as a filter, and "
          "then the family's explanation by adverse selection is left without empirical support.",
    fn=_bb_range_adx,
    params={"period": 20, "deviations": 2.0, "adx_period": 14, "adx_max": 20.0},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 4. Keltner: the same idea with the width measured in true range and not in dispersion of closes.
# --------------------------------------------------------------------------------------------


def _keltner_fade(s: Series) -> np.ndarray:
    c, h, low = _f(s.close), _f(s.high), _f(s.low)
    out = _sig(c.size)
    ema = talib.EMA(c, timeperiod=20)
    atr = talib.ATR(h, low, c, timeperiod=14)
    ok = _finite(ema, atr)
    upper = ema + 2.0 * atr
    lower = ema - 2.0 * atr
    out[ok & (c < lower)] = 1
    out[ok & (c > upper)] = -1
    return out


register(Hypothesis(
    name="mean_reversion.keltner20_atr14_fade",
    family="mean_reversion",
    rationale="The Keltner channel in its modern form: EMA20 as the anchor and 2xATR14 as the "
              "width. Against Bollinger, what changes is WHAT counts as a normal move. Bollinger "
              "measures the dispersion of the closes; the ATR measures the true range, wicks and "
              "gaps included. The difference matters for a concrete microstructure reason: a "
              "liquidation cascade leaves an enormous wick and a recovered close, which WIDENS the "
              "ATR immediately but barely moves the sigma of the closes. The Keltner channel "
              "therefore stops giving signals right after a liquidation event, while Bollinger goes "
              "on giving them.",
    prior="We expect FEWER signals than Bollinger in the weeks following a crash, and for that "
          "reason a worse hit rate but a shorter left tail: it gives up the most profitable bounces "
          "in exchange for not entering at the worst moment. We expect it to be inferior to "
          "Bollinger in a quiet market, where the ATR narrows and the channel turns "
          "hypersensitive. It is refuted if its signals are indistinguishable from hypothesis 1's: "
          "that would indicate that in BTC the wick and the close carry the same information.",
    fn=_keltner_fade,
    params={"ema_period": 20, "atr_period": 14, "multiplier": 2.0},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 5. RSI 14 with fixed 30/70 thresholds. The baseline Cardwell says is broken.
# --------------------------------------------------------------------------------------------


def _rsi_fixed(s: Series) -> np.ndarray:
    c = _f(s.close)
    out = _sig(c.size)
    rsi = talib.RSI(c, timeperiod=14)
    ok = _finite(rsi)
    out[ok & (rsi < 30.0)] = 1
    out[ok & (rsi > 70.0)] = -1
    return out


register(Hypothesis(
    name="mean_reversion.rsi14_fixed_3070",
    family="mean_reversion",
    rationale="RSI 14 with Wilder's original thresholds, 30 and 70. It is registered even though "
              "hardly anybody expects it to work, and for a methodological reason: it is the "
              "control for hypothesis 6. Cardwell's claim —that the RSI 'pins' at the extreme "
              "during trends and that this is why fixed thresholds fail— is only testable if the "
              "fixed control is registered in advance. If we only registered the adaptive version "
              "and it came out well, we would not know whether the credit belongs to Cardwell's "
              "bands or simply to the RSI.",
    prior="We expect a NULL or NEGATIVE net edge, and in particular we expect the short side "
          "(RSI>70) to be the one that loses most, because BTC spends most of its history in "
          "bullish legs where the RSI goes above 70 and stays there. We expect most of the loss to "
          "be concentrated in a few long, directional legs. It is refuted if the fixed 30/70 "
          "matches or beats hypothesis 6's Cardwell bands: that would invalidate this family's "
          "central premise about the RSI's bias in a trend.",
    fn=_rsi_fixed,
    params={"period": 14, "oversold": 30.0, "overbought": 70.0},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 6. RSI 14 with Cardwell bands conditioned on the SMA200. The family's headline act.
#    Same period 14 as #5 on purpose: the only thing that changes is the threshold scheme, and that
#    is exactly the claim to be tested.
# --------------------------------------------------------------------------------------------


def _rsi_cardwell(s: Series) -> np.ndarray:
    c = _f(s.close)
    out = _sig(c.size)
    rsi = talib.RSI(c, timeperiod=14)
    sma200 = talib.SMA(c, timeperiod=200)
    ok = _finite(rsi, sma200)

    bullish = ok & (c > sma200)   # Cardwell range 40-80
    bearish = ok & (c <= sma200)  # Cardwell range 20-60

    out[bullish & (rsi < 40.0)] = 1
    out[bullish & (rsi > 80.0)] = -1
    out[bearish & (rsi < 20.0)] = 1
    out[bearish & (rsi > 60.0)] = -1
    return out


register(Hypothesis(
    name="mean_reversion.rsi14_cardwell_sma200",
    family="mean_reversion",
    rationale="Andrew Cardwell observed that the RSI does not oscillate in the same range under "
              "every regime: in an uptrend it moves between 40 and 80, and in a downtrend between "
              "20 and 60. The mechanism behind it is one of participant composition. In a bull "
              "market the falls are caused by profit taking from short-term traders and they run "
              "into structural buyers waiting; the selling pressure runs out sooner, and that is "
              "why the oscillator's floor is at 40 and not at 30. Buying at 30 in that regime means "
              "waiting for a capitulation that almost never arrives, and staying out. The SMA200 on "
              "the close decides which regime applies using the information available at the bar, "
              "without looking ahead.",
    prior="We expect a positive edge and, above all, MORE TRADES on the side that favours the "
          "regime (longs in an uptrend, shorts in a downtrend) than hypothesis 5, and a large "
          "reduction in the losses on the opposite side. We expect it to fail at regime changes, "
          "when the price crosses the SMA200 several times in a row: there the bands alternate and "
          "we buy at 40 just when the relevant regime had already turned bearish. It is refuted if "
          "it does not improve on hypothesis 5, or if all of its edge comes from the SMA200's "
          "directional filter and not from the asymmetric thresholds.",
    fn=_rsi_cardwell,
    params={"period": 14, "regime_sma": 200,
            "bullish_band": (40.0, 80.0), "bearish_band": (20.0, 60.0)},
    timeframes=("1h", "4h", "1d"),
    min_warmup=260,
))


# --------------------------------------------------------------------------------------------
# 7. Williams %R: position within the high-low RANGE, not momentum of closes.
# --------------------------------------------------------------------------------------------


def _willr_fade(s: Series) -> np.ndarray:
    c, h, low = _f(s.close), _f(s.high), _f(s.low)
    out = _sig(c.size)
    wr = talib.WILLR(h, low, c, timeperiod=14)
    ok = _finite(wr)
    out[ok & (wr < -80.0)] = 1
    out[ok & (wr > -20.0)] = -1
    return out


register(Hypothesis(
    name="mean_reversion.willr14_fade",
    family="mean_reversion",
    rationale="%R does not measure momentum: it measures where the price closes within the "
              "high-low range of the last 14 candles. That makes it sensitive to a different "
              "behaviour from the one the RSI picks up. Closing at the absolute floor of a two-week "
              "range is the signature of a stop run: a push looking for the liquidity accumulated "
              "below the previous lows, executed by somebody who needs counterparty for a large "
              "position. That flow is mechanical and uninformed, and once executed it disappears, "
              "which is exactly the condition for the absorption to be profitable.",
    prior="We expect a positive edge concentrated at SHORT horizons (a few candles): the sweep "
          "effect runs out quickly and, if the position is held, the signal turns into a "
          "directional bet, which is not what the hypothesis claims. We expect it to fail on "
          "genuine range breaks, where the low of the range gives way because there is new "
          "information and the %R stays glued to -100 for the whole leg. It is refuted if the edge "
          "does not decay with the holding horizon.",
    fn=_willr_fade,
    params={"period": 14, "oversold": -80.0, "overbought": -20.0},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 8. CCI 20: deviation normalised by the MEAN ABSOLUTE deviation (robust to the tail).
# --------------------------------------------------------------------------------------------


def _cci_fade(s: Series) -> np.ndarray:
    c, h, low = _f(s.close), _f(s.high), _f(s.low)
    out = _sig(c.size)
    cci = talib.CCI(h, low, c, timeperiod=20)
    ok = _finite(cci)
    out[ok & (cci < -100.0)] = 1
    out[ok & (cci > 100.0)] = -1
    return out


register(Hypothesis(
    name="mean_reversion.cci20_fade",
    family="mean_reversion",
    rationale="The CCI uses the typical price (H+L+C)/3 and normalises it by the mean ABSOLUTE "
              "deviation, not by the standard deviation. The difference is not cosmetic in BTC: the "
              "standard deviation squares, so a single liquidation candle inflates the denominator "
              "for 20 candles and anaesthetises the Bollinger signal right after the event. The "
              "mean absolute deviation barely flinches. The hypothesis, therefore, is that the "
              "robust statistic goes on detecting dislocations in the period after a shock, which "
              "is when the premium for providing liquidity ought to be highest because the usual "
              "providers' capital is exhausted.",
    prior="We expect the CCI's edge to beat Bollinger's (hypothesis 1) precisely and only in the "
          "windows that follow the largest-range candles, and to be similar or slightly worse over "
          "the rest of the sample, by giving more signals of poorer quality. We expect the classic "
          "+-100 failure in a trend, just like the rest of the family. It is refuted if its edge "
          "conditioned on 'after a shock' is not larger than Bollinger's in those same windows: "
          "that was the whole argument.",
    fn=_cci_fade,
    params={"period": 20, "threshold": 100.0},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 9. Stretch against the LONG anchor. A different horizon, a different participant.
# --------------------------------------------------------------------------------------------


def _stretch_ema200(s: Series) -> np.ndarray:
    c, h, low = _f(s.close), _f(s.high), _f(s.low)
    out = _sig(c.size)
    ema200 = talib.EMA(c, timeperiod=200)
    atr = talib.ATR(h, low, c, timeperiod=14)
    ok = _finite(ema200, atr)
    out[ok & (c < ema200 - 3.0 * atr)] = 1
    out[ok & (c > ema200 + 3.0 * atr)] = -1
    return out


register(Hypothesis(
    name="mean_reversion.stretch_ema200_atr14",
    family="mean_reversion",
    rationale="The eight previous hypotheses measure dislocations against an anchor of 14-20 "
              "candles, the range trader's horizon. This one changes scale: it measures the "
              "distance to the EMA200 in units of ATR14, with the multiplier 3 from the convention "
              "for ATR-based exits. The participant is a different one. The 200-period average is "
              "the reference that capital allocators and structural holders use to decide whether "
              "the asset is 'expensive' or 'cheap'; when the price moves three daily ranges away "
              "from it, the demand that shows up is not from market makers but from rebalancing, "
              "and it operates on a timescale of weeks.",
    prior="We expect VERY FEW signals, a positive edge per trade but with an enormous variance and "
          "a long recovery horizon, of dozens of candles. We expect the short side (3 ATR above the "
          "EMA200) to work worse than the long one, because in BTC's bubbles the price has stayed "
          "stretched above it for months. It is refuted if the number of signals is high —that "
          "would mean 3 ATR is not an extreme in this asset and the 'structural dislocation' "
          "premise is false— or if the edge runs out within a few candles, which would make it "
          "indistinguishable from short-term reversion.",
    fn=_stretch_ema200,
    params={"ema_period": 200, "atr_period": 14, "multiplier": 3.0},
    timeframes=("4h", "1d"),
    min_warmup=420,
))


# --------------------------------------------------------------------------------------------
# 10. Extreme WITH volume confirmation. This is where "forced closing" becomes explicit.
# --------------------------------------------------------------------------------------------


def _bb_volume_climax(s: Series) -> np.ndarray:
    c, v = _f(s.close), _f(s.volume)
    out = _sig(c.size)
    upper, _mid, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
    v_mean = talib.SMA(v, timeperiod=20)
    ok = _finite(upper, lower, v_mean)
    climax = ok & (v > 2.0 * v_mean)
    out[climax & (c < lower)] = 1
    out[climax & (c > upper)] = -1
    return out


register(Hypothesis(
    name="mean_reversion.bb20_2_volume_climax",
    family="mean_reversion",
    rationale="The same Bollinger extreme as hypothesis 1, with one added condition: the candle's "
              "volume must double its 20-period average, the conventional threshold for climactic "
              "volume. Volume is what separates the two reasons the price can leave the band. If it "
              "leaves on low volume, the book is thin and there is no premium to charge: nobody is "
              "paying to get out. If it leaves on volume that doubles the average, somebody is "
              "closing under duress —cascading perpetual liquidations, margin calls— and that "
              "seller has neither an opinion about value nor the ability to wait. It is the only "
              "case in which the hypothesis asserts that a structurally willing counterparty "
              "exists.",
    prior="We expect a LARGER edge per trade than hypothesis 1 and far fewer trades. We expect the "
          "effect to be markedly asymmetric: much stronger on the long side (buying capitulation) "
          "than on the short one, because bearish liquidations in BTC are more violent and more "
          "concentrated in time than bullish ones. We expect it to fail on the first day of a "
          "genuine macro shock, where high volume does carry information. It is refuted if the "
          "volume filter does not improve hypothesis 1's edge: that would say that volume does not "
          "distinguish forced flow from informed flow in this market.",
    fn=_bb_volume_climax,
    params={"period": 20, "deviations": 2.0, "vol_period": 20, "vol_multiple": 2.0},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 11. Sweep and rejection. The only hypothesis in the family that looks at candle shape.
# --------------------------------------------------------------------------------------------


def _sweep_rejection(s: Series) -> np.ndarray:
    c, h, low = _f(s.close), _f(s.high), _f(s.low)
    out = _sig(c.size)

    # Low/high of the 20 PREVIOUS candles: talib.MIN(...)[i] covers the window ending at i
    # INCLUSIVE, so it has to be shifted by one bar to exclude the current candle. The shift runs
    # towards the future (_prev), never backwards.
    min_prev = _prev(talib.MIN(low, timeperiod=20))
    max_prev = _prev(talib.MAX(h, timeperiod=20))

    rng = h - low
    ok = _finite(min_prev, max_prev, rng) & (rng > 0.0)

    # Rejection: the candle pierces the previous extreme but closes in the opposite third of its
    # range.
    sweep_low = ok & (low < min_prev) & ((c - low) / np.where(ok, rng, 1.0) > 2.0 / 3.0)
    sweep_high = ok & (h > max_prev) & ((h - c) / np.where(ok, rng, 1.0) > 2.0 / 3.0)

    out[sweep_low] = 1
    out[sweep_high] = -1
    return out


register(Hypothesis(
    name="mean_reversion.sweep_rejection_20",
    family="mean_reversion",
    rationale="The candle pierces the low of the previous 20 but closes in the upper third of its "
              "own range. That shape is the observable footprint of a specific sequence: beneath a "
              "visible low, buyers' stops and breakout sell orders accumulate; a participant who "
              "needs to buy size pushes the price down there, those orders execute and give them "
              "the counterparty they needed, and the price comes back because the selling was "
              "mechanical and exhausted itself within the candle. The key is that the rejection is "
              "confirmed INSIDE the same bar, by the close: there is no need to see the next "
              "candle, and that is why the signal is causal.",
    prior="We expect a positive and short-lived edge, with the bulk of the move in the first "
          "candles after the signal. We expect the edge to be larger on 15m and 1h than on 4h, "
          "because the accumulation of stops at visible levels is a microstructure phenomenon and "
          "it blurs under aggregation. We expect it to fail when the piercing of the low is the "
          "real start of a bear leg: there the close in the upper third is only a technical bounce "
          "inside the fall. It is refuted if there is no difference in edge between the candles "
          "that pierce the previous low and the ones that simply close strong.",
    fn=_sweep_rejection,
    params={"lookback": 20, "rejection_fraction": 2 / 3},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 12. A streak of three closes. No indicator, no normalisation: overreaction in the raw.
# --------------------------------------------------------------------------------------------


def _streak3(s: Series) -> np.ndarray:
    c = _f(s.close)
    out = _sig(c.size)
    c1, c2, c3 = _prev(c), _prev(_prev(c)), _prev(_prev(_prev(c)))
    ok = _finite(c1, c2, c3)
    out[ok & (c < c1) & (c1 < c2) & (c2 < c3)] = 1
    out[ok & (c > c1) & (c1 > c2) & (c2 > c3)] = -1
    return out


register(Hypothesis(
    name="mean_reversion.streak3_fade",
    family="mean_reversion",
    rationale="Three consecutive closes to the downside are bought; three to the upside are sold. "
              "There is no indicator, no normalisation by volatility and no threshold to tune, and "
              "that bareness is the point: it is the cleanest test of overreaction by "
              "extrapolation. The concrete behaviour is that of the trader who mistakes a short "
              "streak for a trend and positions in its direction, together with the risk manager "
              "who mechanically cuts exposure after several closes against them. If mean reversion "
              "in BTC really exists as a behavioural phenomenon, it has to show up here; if it only "
              "shows up in more elaborate formulations, we should suspect that what we are "
              "measuring is an artefact of those formulations.",
    prior="We expect a VERY small edge, barely distinguishable from noise, and probably negative "
          "after costs given the high frequency of signals; what we expect to be informative is the "
          "SIGN and its consistency across timeframes, not the magnitude. We expect it to be "
          "negative in strongly trending legs, where streaks of three chain together. It is "
          "refuted, and with it the overreaction premise of the whole family, if the sign is "
          "systematically negative on every timeframe: that would indicate that in BTC short "
          "streaks continue instead of reverting, and then whichever of the eleven preceding "
          "hypotheses work would be capturing something else.",
    fn=_streak3,
    params={"consecutive_bars": 3},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))
