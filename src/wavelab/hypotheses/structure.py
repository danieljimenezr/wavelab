"""`structure` family: market structure as a source of edge.

PRE-REGISTRATION. Nothing here has been run against data. The parameters are the conventional ones
from the literature — Donchian 20 and 55 (Turtle systems 1 and 2), Williams' 5-candle fractal (k=2
on each side), ATR 14, 20-period volume average — and the horizon of the event hypotheses is 10
candles for ALL of them, chosen once and applied without exception so that there is no hidden grid
dressed up as "variants".

THE FAMILY'S SHARED MECHANISM. Previous highs and lows are the few levels every participant sees
the same way, without ambiguity and without parameters. That is where the long puts the stop, where
whoever waits for confirmation puts the breakout order, and where the market maker knows there is
executable volume. In other words: structure does not predict, but it DOES concentrate orders, and
a concentration of orders is the only thing that can move a price. Each hypothesis below bets on
what happens when price reaches that liquidity: either the break drags along whoever was on the
other side (continuation) or the liquidity is consumed in one go and price comes back inside (a
sweep).

The two families of bets CANNOT both be true in the same regime. The conflict is deliberate:
`structure.donchian_break_20` and `structure.sweep_reversal_20` trade the same level in opposite
directions, and so do `structure.bos_swing` and `structure.range_fade_swing`. If both "work" at
once without the regime separating them, the likeliest explanation is not that there are two edges
but that the estimator is measuring noise, and that is informative.

CAUSALITY. Every reference level comes from candles STRICTLY earlier than i (`_shift1` over talib's
windows, which include the current candle) and every pivot is placed on the candle where it is
CONFIRMED, not on the one where the extreme happened. Neither `find_peaks` nor any statistic
defined against the whole array is used.
"""

from __future__ import annotations

import numpy as np
import talib

from wavelab.hypotheses.base import Hypothesis, Series, register

# Single horizon for the event hypotheses (sweep, retest, inside bar, range fade).
# One number for all of them: if each carried its own, this would be a grid.
_HOLD = 10

# Half-width of the Williams fractal: 5 candles, 2 on each side. Confirmation arrives 2 candles
# after the extreme, and that delay is honoured explicitly.
_K = 2

# Series shorter than this return all zeros. It matches `min_warmup` on purpose: that way the
# guard's threshold never falls INSIDE the region the backtest evaluates, and the signal at i is
# identical whether it is computed over the prefix [0..i] or over the whole series.
_MIN_BARS = 200


# --------------------------------------------------------------------------------------------
# Utilities. All causal: position i only looks at positions <= i.
# --------------------------------------------------------------------------------------------

def _f(x: np.ndarray) -> np.ndarray:
    """talib requires contiguous float64."""
    return np.ascontiguousarray(x, dtype=np.float64)


def _shift1(x: np.ndarray) -> np.ndarray:
    """out[i] = x[i-1], out[0] = NaN. Shifts towards the FUTURE (never a negative np.roll)."""
    out = np.empty_like(x)
    out[0] = np.nan
    out[1:] = x[:-1]
    return out


# `_shiftk` used to live here. It goes with the duplicate hypothesis that was its only user (see
# the note at point 11): leaving dead utilities lying around invites the next hypothesis to reuse
# them without rechecking their causality.


def _prev_max(high: np.ndarray, n: int) -> np.ndarray:
    """Maximum of the n candles BEFORE i. talib.MAX includes candle i, so it is shifted."""
    return _shift1(talib.MAX(_f(high), n))


def _prev_min(low: np.ndarray, n: int) -> np.ndarray:
    """Minimum of the n candles BEFORE i."""
    return _shift1(talib.MIN(_f(low), n))


def _gt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a > b, False wherever either is NaN. The check is explicit: nan_to_num with a sentinel
    would invent breakouts during warm-up."""
    out = np.zeros(a.shape, dtype=bool)
    ok = ~(np.isnan(a) | np.isnan(b))
    np.greater(a, b, out=out, where=ok)
    return out


def _lt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a < b, False wherever either is NaN."""
    out = np.zeros(a.shape, dtype=bool)
    ok = ~(np.isnan(a) | np.isnan(b))
    np.less(a, b, out=out, where=ok)
    return out


def _fresh(cond: np.ndarray) -> np.ndarray:
    """Only the candle on which `cond` flips from false to true. Needed for the event hypotheses
    whose condition is a persistent STATE ("price is above the pivot"): without this the event
    re-arms on every candle and a one-off rejection hypothesis turns, unintentionally, into a
    permanent counter-trend position. It looks at i and i-1, nothing else."""
    out = cond.copy()
    out[1:] &= ~cond[:-1]
    return out


def _ffill(x: np.ndarray) -> np.ndarray:
    """Carries the last non-NaN value forward. `maximum.accumulate` is a prefix: the value at i
    can only come from <= i."""
    idx = np.where(~np.isnan(x), np.arange(x.size), 0)
    np.maximum.accumulate(idx, out=idx)
    return x[idx]


def _events(up: np.ndarray, dn: np.ndarray) -> np.ndarray:
    """Encodes events into {-1,0,+1}. If both fire on the same candle the situation is ambiguous
    and 0 is emitted (there is no information), never an arbitrary tie-break."""
    both = up & dn
    ev = np.zeros(up.size, dtype=np.int8)
    ev[up & ~both] = 1
    ev[dn & ~both] = -1
    return ev


def _hold(up: np.ndarray, dn: np.ndarray) -> np.ndarray:
    """Holds the last event until the opposite one appears (always in the market after the
    first)."""
    ev = _events(up, dn)
    idx = np.where(ev != 0, np.arange(ev.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = ev[idx]
    out[idx == 0] = ev[0]          # before the first event there is no position
    return out


def _hold_n(up: np.ndarray, dn: np.ndarray, bars: int = _HOLD) -> np.ndarray:
    """Holds the last event for `bars` candles and then exits. For event hypotheses, where the
    effect —if it exists— is transient by construction."""
    ev = _events(up, dn)
    n = ev.size
    idx = np.where(ev != 0, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    out = ev[idx].copy()
    age = np.arange(n) - idx
    out[age >= bars] = 0
    out[ev[idx] == 0] = 0
    return out


def _pivots(s: Series, k: int = _K) -> tuple[np.ndarray, np.ndarray]:
    """Williams fractals PLACED ON THE CANDLE THAT CONFIRMS THEM, not on the candle of the extreme.

    A fractal high at candle j is only known at j+k, once the k following candles have closed.
    Returning the pivot at j would be looking into the future: it is exactly the mistake that makes
    "structure" strategies look profitable on paper. Here `ph[i]` holds the price of the high
    confirmed AT i (the extreme having occurred at i-k) and NaN if nothing is confirmed at i.
    """
    n = len(s)
    w = 2 * k + 1
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    if n <= w:
        return ph, pl

    hi, lo = _f(s.high), _f(s.low)
    rmax = talib.MAX(hi, w)      # maximum over [i-w+1, i]: every datum <= i
    rmin = talib.MIN(lo, w)

    i = np.arange(w - 1, n)      # from the first complete window onwards
    c = i - k                    # central candle: the candidate extreme

    ok_h = ~(np.isnan(rmax[i]) | np.isnan(hi[c]))
    is_h = np.zeros(i.size, dtype=bool)
    np.equal(hi[c], rmax[i], out=is_h, where=ok_h)
    ph[i[is_h]] = hi[c[is_h]]

    ok_l = ~(np.isnan(rmin[i]) | np.isnan(lo[c]))
    is_l = np.zeros(i.size, dtype=bool)
    np.equal(lo[c], rmin[i], out=is_l, where=ok_l)
    pl[i[is_l]] = lo[c[is_l]]

    return ph, pl


def _pivot_levels(s: Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(last confirmed high, the one before it, last confirmed low, the one before it) at each
    candle."""
    ph, pl = _pivots(s)
    ph1, pl1 = _ffill(ph), _ffill(pl)
    # On a confirmation candle, the "previous" one is the value carried forward to the candle
    # before.
    ph2 = _ffill(np.where(~np.isnan(ph), _shift1(ph1), np.nan))
    pl2 = _ffill(np.where(~np.isnan(pl), _shift1(pl1), np.nan))
    return ph1, ph2, pl1, pl2


def _structure_state(s: Series) -> np.ndarray:
    """+1 if the last confirmed structure is higher highs and higher lows, -1 if lower, 0 if it is
    mixed. Shared base for several hypotheses."""
    ph1, ph2, pl1, pl2 = _pivot_levels(s)
    up = _gt(ph1, ph2) & _gt(pl1, pl2)
    dn = _lt(ph1, ph2) & _lt(pl1, pl2)
    st = np.zeros(len(s), dtype=np.int8)
    st[up] = 1
    st[dn] = -1
    return st


# --------------------------------------------------------------------------------------------
# 1. Donchian 20 breakout, always in the market
# --------------------------------------------------------------------------------------------

def _donchian_break_20(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c = _f(s.close)
    return _hold(_gt(c, _prev_max(s.high, 20)), _lt(c, _prev_min(s.low, 20)))


register(Hypothesis(
    name="structure.donchian_break_20",
    family="structure",
    title="Flip sides on every Donchian 20 break",
    rationale="The high of the previous 20 candles is the level where the largest quantity of "
              "resting orders converges: the shorts' stops, stop entries from those waiting for "
              "confirmation, and hedges from option sellers. When price takes it out, those orders "
              "execute as market orders and consume the book in the same direction, which pushes "
              "price further and forces more closing. If that cascade exists in BTC, a plain "
              "stop-and-reverse on the channel should capture it with no further machinery.",
    prior="We expect a positive edge concentrated in a few right-tail episodes (a handful of "
          "trades supply almost the entire result), with a LOW hit rate, below 45%. We expect it "
          "to fail —a negative edge, not a nil one— in prolonged ranges, where every breakout "
          "reverses and the system pays the spread in both directions. If it comes out with a "
          "positive edge AND a high hit rate, the hypothesis is wrong: that would point to a data "
          "or execution fault, not to a structural effect.",
    fn=_donchian_break_20,
    params={"channel": 20, "mode": "stop-and-reverse"},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 2. Donchian 55 breakout with a Donchian 20 exit (with a flat state)
# --------------------------------------------------------------------------------------------

def _donchian_55_exit_20(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c = _f(s.close)
    hi55, lo55 = _prev_max(s.high, 55), _prev_min(s.low, 55)
    hi20, lo20 = _prev_max(s.high, 20), _prev_min(s.low, 20)
    out = np.zeros(n, dtype=np.int8)
    pos = 0
    for i in range(n):
        # Explicit loop: the state at i depends on the state at i-1 and on data from i. Nothing
        # else.
        if pos == 0:
            if not np.isnan(hi55[i]) and c[i] > hi55[i]:
                pos = 1
            elif not np.isnan(lo55[i]) and c[i] < lo55[i]:
                pos = -1
        elif pos == 1:
            if not np.isnan(lo20[i]) and c[i] < lo20[i]:
                pos = 0          # exit and stay flat: re-entering on the same close is ambiguous
        else:
            if not np.isnan(hi20[i]) and c[i] > hi20[i]:
                pos = 0
        out[i] = pos
    return out


register(Hypothesis(
    name="structure.donchian_55_exit_20",
    family="structure",
    title="Donchian 55 entry, exit flat at 20",
    rationale="This is not the same hypothesis as the 20 channel with a different number, and that "
              "is why it is registered separately: the claim here is not about the entry level but "
              "about the ASYMMETRY between entering and exiting. Entry happens only on the "
              "55-candle break, the one that requires an imbalance that has already absorbed the "
              "whole quarter's liquidity, and the exit uses a shorter 20 channel and goes FLAT, "
              "with no reversal. The bet is that the participant who triggers the cascade "
              "(leveraged liquidations, treasury rebalancing) takes weeks to exhaust itself, while "
              "its disappearance shows up within days; exiting flat instead of flipping avoids "
              "paying the wrong side of a consolidation.",
    prior="We expect a positive edge and FEWER trades than `donchian_break_20`, with a smaller "
          "maximum cumulative loss thanks to the flat state. We expect it to fail on low "
          "timeframes (15m, 1h), where 55 candles are a few hours and the break identifies no "
          "structural flow at all: there the result should be indistinguishable from zero, or "
          "negative after costs. If the edge were equal or greater on 15m than on 1d, the "
          "mechanical explanation above is false.",
    fn=_donchian_55_exit_20,
    params={"entry": 55, "exit": 20, "mode": "long/short with a flat state"},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 3. BOS: break of the last CONFIRMED pivot
# --------------------------------------------------------------------------------------------

def _bos_swing(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c = _f(s.close)
    ph1, _, pl1, _ = _pivot_levels(s)
    return _hold(_gt(c, ph1), _lt(c, pl1))


register(Hypothesis(
    name="structure.bos_swing",
    family="structure",
    title="Break of the last confirmed swing",
    rationale="The structural version of the breakout: instead of a fixed-length channel, the "
              "level is the last swing high or low CONFIRMED by a 5-candle fractal. The mechanical "
              "difference matters: a 20 channel moves through the mere passage of time even if "
              "nothing has happened, whereas a pivot only changes when the market has genuinely "
              "turned and left somebody trapped at the extreme. That is where the real stops are, "
              "because it is the only point that invalidates the thesis of whoever bought the "
              "previous impulse.",
    prior="We expect a positive edge similar to or slightly below the 20 channel's but with FEWER "
          "trades, because the levels are more stable. We expect it to fail when volatility "
          "compresses: with small candles the fractal confirms trivial pivots, the level ends up a "
          "tick away from price and the signal flips sign constantly. The hypothesis is refuted if "
          "the edge per trade does not beat the 20 channel's, because then the 'structural' part "
          "adds nothing over a rolling maximum.",
    fn=_bos_swing,
    params={"fractal_k": _K, "fractal_window": 2 * _K + 1, "confirmation_lag": _K},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 4. CHoCH: change of character
# --------------------------------------------------------------------------------------------

def _choch(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c = _f(s.close)
    ph1, ph2, pl1, pl2 = _pivot_levels(s)
    uptrend = _gt(ph1, ph2) & _gt(pl1, pl2)      # rising sequence in force
    downtrend = _lt(ph1, ph2) & _lt(pl1, pl2)    # falling sequence in force
    # Change of character: first low lost after a rising sequence (and its mirror image).
    choch_dn = uptrend & _lt(c, pl1)
    choch_up = downtrend & _gt(c, ph1)
    return _hold(choch_up, choch_dn)


register(Hypothesis(
    name="structure.choch",
    family="structure",
    title="First break against the structure in force",
    rationale="The change of character is the first time the sequence breaks: we had been making "
              "higher highs and higher lows and suddenly the last low is lost. The mechanism is "
              "not the break itself but WHO is on the other side: in a rising sequence, each low "
              "is where the latest batch of trend buyers got in, with the stop just underneath. "
              "Losing that low means the demand that was holding the impulse up no longer shows "
              "up, and it turns the recent buyers into forced sellers. The difference from "
              "`bos_swing` is that here only the break AGAINST the structure in force is traded, "
              "not the one with it.",
    prior="We expect a positive edge but SMALLER than the continuation one, and heavily dependent "
          "on the timeframe: credible on 4h and 1d, doubtful on 15m. We expect it to fail clearly "
          "in strong trends, where most CHoCHs are shakeouts inside the impulse and price resumes: "
          "there it should give a NEGATIVE edge, and if it does not, the signal is not capturing "
          "what we think it is. It is also refuted if its result is indistinguishable from "
          "`bos_swing`'s with the sign flipped: that would mean we are only measuring price "
          "autocorrelation, not structure.",
    fn=_choch,
    params={"fractal_k": _K, "context": "2 pivots per side"},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 5. Structural regime HH/HL vs LH/LL
# --------------------------------------------------------------------------------------------

def _swing_trend_state(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    return _structure_state(s)


register(Hypothesis(
    name="structure.swing_trend_state",
    family="structure",
    title="Long on higher highs, flat when mixed",
    rationale="The family's minimal hypothesis, and deliberately the dullest: no event and no "
              "breakout, just the classic Dow classification. Long while the last two confirmed "
              "pivots are a higher high AND a higher low, short in the mirror case, and flat when "
              "the structure is mixed. It serves as the BENCHMARK: if the event hypotheses do not "
              "beat this, what they are capturing is the underlying trend and not the liquidity "
              "mechanism they claim to exploit. Being flat on mixed structure is the part that "
              "contributes, because that is where trend following loses money.",
    prior="We expect a small positive edge, dominated by BTC's historical bullish bias, with a "
          "substantial amount of time out of the market (more than 25% of candles). We expect the "
          "short side to have a negative or nil edge on its own: if the shorts contribute as much "
          "as the longs, the data period should be treated with suspicion rather than celebrated. "
          "It is refuted as a useful benchmark if its result is indistinguishable from buying and "
          "sitting still.",
    fn=_swing_trend_state,
    params={"fractal_k": _K},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 6. Liquidity sweep: false break of the 20 channel
# --------------------------------------------------------------------------------------------

def _sweep_reversal_20(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c, hi, lo = _f(s.close), _f(s.high), _f(s.low)
    hh, ll = _prev_max(s.high, 20), _prev_min(s.low, 20)
    short_ev = _gt(hi, hh) & _lt(c, hh)    # pierces the previous high and closes below it
    long_ev = _lt(lo, ll) & _gt(c, ll)     # pierces the previous low and closes above it
    return _hold_n(long_ev, short_ev)


register(Hypothesis(
    name="structure.sweep_reversal_20",
    family="structure",
    title="Fade the break that closes back inside",
    rationale="The opposite face of `donchian_break_20`, on the SAME level and on purpose. If "
              "liquidity is piled up just behind the 20-candle extreme, there is a participant "
              "with a direct incentive to go and get it: whoever needs to execute size and can "
              "only do so against everybody else's stops. The observable signature is a candle "
              "that pierces the level on the high but closes INSIDE the range: the stops were "
              "filled, there was no continuation behind them and price comes back. Closing inside "
              "is the key condition, because it separates absorption from a genuine break.",
    prior="We expect a short-lived positive edge —concentrated in the first candles of the "
          "10-candle horizon— with a HIGH hit rate and a small average gain, the inverse profile "
          "to the breakout's. We expect it to fail in strong trends, where the false break is only "
          "a pause before continuing, and to fail on 1d, where a daily candle bundles too many "
          "events for the close to mean 'absorption'. If both this hypothesis and "
          "`donchian_break_20` come out positive over the same period and regime, the estimator "
          "should be suspected before either of them is believed.",
    fn=_sweep_reversal_20,
    params={"channel": 20, "horizon": _HOLD},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 7. Retest of the broken level
# --------------------------------------------------------------------------------------------

def _retest_hold(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c, hi, lo = _f(s.close), _f(s.high), _f(s.low)
    hh, ll = _prev_max(s.high, 20), _prev_min(s.low, 20)
    ev_up = np.zeros(n, dtype=bool)
    ev_dn = np.zeros(n, dtype=bool)

    lvl_u = np.nan
    ttl_u = 0
    touched_u = False
    lvl_d = np.nan
    ttl_d = 0
    touched_d = False

    for i in range(n):
        # Everything read here is from candle i or from state accumulated over earlier candles.
        brk_u = (not np.isnan(hh[i])) and c[i] > hh[i]
        brk_d = (not np.isnan(ll[i])) and c[i] < ll[i]

        if brk_u:
            lvl_u, ttl_u, touched_u = hh[i], _HOLD, False
        elif ttl_u > 0:
            ttl_u -= 1
            if lo[i] <= lvl_u:
                touched_u = True
            if touched_u and c[i] > lvl_u:
                ev_up[i] = True
                ttl_u = 0

        if brk_d:
            lvl_d, ttl_d, touched_d = ll[i], _HOLD, False
        elif ttl_d > 0:
            ttl_d -= 1
            if hi[i] >= lvl_d:
                touched_d = True
            if touched_d and c[i] < lvl_d:
                ev_dn[i] = True
                ttl_d = 0

    return _hold_n(ev_up, ev_dn)


register(Hypothesis(
    name="structure.retest_hold",
    family="structure",
    title="Trade the break only after its retest",
    rationale="The broken level changes role, and there is a concrete flow reason for it: those "
              "who sold at resistance and got trapped try to get out at their entry point when "
              "price comes back, and those who missed the break have their reference there to buy "
              "without chasing. Both act in the same place and in the same direction. The trading "
              "condition is demanding on purpose: it has to break, COME BACK and touch the level "
              "within 10 candles, and close on the right side again. If only the break mattered, "
              "this would add nothing over hypothesis 1.",
    prior="We expect FEWER trades than the plain break and a better result per trade, because the "
          "retest filters out the breaks with no buyer behind them. We expect it to fail on the "
          "most violent breaks —precisely the ones that contribute most to hypothesis 1— because "
          "those never come back to touch the level and the filter discards them: if the breakout "
          "effect is pure right tail, this filter should DESTROY the edge rather than improve it. "
          "That is the test that matters, and both conclusions are publishable.",
    fn=_retest_hold,
    params={"channel": 20, "retest_window": _HOLD, "horizon": _HOLD},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 8. Breakout after channel compression
# --------------------------------------------------------------------------------------------

def _compression_break(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c = _f(s.close)
    hh, ll = _prev_max(s.high, 20), _prev_min(s.low, 20)
    width = hh - ll
    min_width = talib.MIN(_f(np.nan_to_num(width, nan=np.inf)), 60)
    # `width` at i already depends only on candles < i; talib's 60-window ends at i.
    compressed = np.zeros(n, dtype=bool)
    ok = ~(np.isnan(width) | np.isnan(min_width) | np.isinf(min_width))
    np.less_equal(width, min_width, out=compressed, where=ok)
    return _hold(compressed & _gt(c, hh), compressed & _lt(c, ll))


register(Hypothesis(
    name="structure.compression_break",
    family="structure",
    title="Break with the channel at its narrowest",
    rationale="A break only matters if there was contained disagreement beforehand. When the width "
              "of the 20 channel falls to its lowest of the last 60 candles, buyers and sellers "
              "have reached a narrow equilibrium and both camps have been stacking stops very "
              "close by, a few ticks from price and from each other. Leaving that zone executes "
              "both stop books in a chain, which is the only moment when a break drags along a "
              "volume disproportionate to the size of the preceding move. Compression does not "
              "predict direction: it only says that the move which resolves it will be "
              "disproportionate.",
    prior="We expect an edge per trade HIGHER than `donchian_break_20`'s with far fewer trades, "
          "and that comparison is the hypothesis's real test: if the compression filter improves "
          "nothing, the twin-stop-book idea is false and all we were doing was trading less. We "
          "expect it to fail after violent falls, where the compression appears through exhaustion "
          "and not through equilibrium, and on 1d, where 60 candles are two months and the width "
          "minimum arrives late.",
    fn=_compression_break,
    params={"channel": 20, "compression_window": 60},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 9. Volume-confirmed breakout
# --------------------------------------------------------------------------------------------

def _volume_confirmed_break(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c, v = _f(s.close), _f(s.volume)
    hh, ll = _prev_max(s.high, 20), _prev_min(s.low, 20)
    v_avg = talib.SMA(v, 20)                   # includes candle i: a datum available at i
    strong = _gt(v, v_avg)
    return _hold(_gt(c, hh) & strong, _lt(c, ll) & strong)


register(Hypothesis(
    name="structure.volume_confirmed_break",
    family="structure",
    title="Only the break that volume confirms",
    rationale="If the breakout mechanism is the cascading execution of accumulated orders, then it "
              "has a compulsory fingerprint: volume. A break on volume below its 20-period average "
              "means there was nobody waiting at that level, that price got there by drift and not "
              "by execution, and that there is therefore no forced flow to continue the move. "
              "Volume is not used here as an indicator but as a VERIFICATION of the mechanism that "
              "hypotheses 1 and 8 take for granted.",
    prior="We expect a positive edge and, above all, an edge per trade GREATER than unfiltered "
          "`donchian_break_20`'s. If the volume filter improves nothing, the order-cascade "
          "explanation is seriously weakened for every breakout hypothesis in this family, and "
          "that is a more valuable result than the hypothesis itself. We expect it to fail on 1d, "
          "where BTC's daily volume is dominated by the weekly cycle and by the split across "
          "exchanges, not by the breakout event.",
    fn=_volume_confirmed_break,
    params={"channel": 20, "volume_average": 20},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 10. Breakout with an ATR buffer
# --------------------------------------------------------------------------------------------

def _atr_buffered_break(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c = _f(s.close)
    hh, ll = _prev_max(s.high, 20), _prev_min(s.low, 20)
    atr = talib.ATR(_f(s.high), _f(s.low), c, 14)
    return _hold(_gt(c, hh + 0.5 * atr), _lt(c, ll - 0.5 * atr))


register(Hypothesis(
    name="structure.atr_buffered_break",
    family="structure",
    title="Break half an ATR past the level",
    rationale="A direct complement to `sweep_reversal_20` and an explicit test of hypothesis 1. If "
              "it is true that the stops pile up just behind the level and that somebody goes "
              "looking for them, then the MARGINAL break —the one that pokes a few ticks above— is "
              "systematically the worst, because it is exactly the one produced by whoever wants "
              "to sell. Requiring the close to clear the level by half a 14-candle ATR discards "
              "that zone without changing anything else about the rule, and it makes the buffer "
              "adapt to volatility on its own instead of fixing an arbitrary percentage.",
    prior="We expect a positive edge and a larger one per trade than the break without a buffer, "
          "and this is the comparison that matters. Since `sweep_reversal_20` claims the discarded "
          "zone has a NEGATIVE edge, the two hypotheses have to be consistent with each other: if "
          "the sweep turns out profitable but the buffer does not improve the break, or the other "
          "way round, one of the two is measuring something else. We expect the buffer to hurt on "
          "1d, where half a daily ATR is an enormous move and entering that late gives away most "
          "of the displacement.",
    fn=_atr_buffered_break,
    params={"channel": 20, "atr": 14, "atr_buffer": 0.5},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 11. GAP LEFT BY A HYPOTHESIS WITHDRAWN IN THE AUDIT (2026-09-08)
#
# `structure.inside_bar_break` used to be here. It has been DELETED for being an exact DUPLICATE of
# `candles.inside_bar_break`: same containment condition (candle i-1 inside candle i-2), same
# breakout level (the extremes of the mother candle, i-2) and same confirmation (the close of i).
# The only things separating them were `<` versus `<=` in the containment and the holding horizon
# (10 candles here, a 1-candle pulse there). Measured against data, this hypothesis's event set was
# a STRICT SUBSET of the other's: 244 out of 244 events matched candle for candle.
#
# Why it mattered, and why this is not cosmetic. A pre-registration is only worth anything if the
# number of trials entering the multiple-comparisons correction is the number of DISTINCT BETS.
# Registering the same bet in two families counts it twice, and since the two also share almost all
# their signals, their results are correlated: if the pattern works by chance it "works" twice and
# looks like cross-family confirmation when it is the same observation repeated. That is exactly
# the bias this registry exists to prevent, and the difference in horizon does not fix it: holding
# 10 candles instead of 1 multiplies each event into ten overlapping observations over almost the
# same future window, which inflates `n_signals` without adding information (that is what
# `effective_n` is for).
#
# The `candles` version is kept because it is the superset (non-strict containment) and its
# one-candle pulse is the clean encoding of the bet. The `volatility.inside_bar_breakout` variant
# is NOT removed: it breaks the extremes of the INSIDE candle, not the mother's, which is a
# different level and a different mechanical claim; it is declared as a dependent trial in
# `volatility`'s own file.
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# 12. Fading the extremes when the structure is NOT trending
# --------------------------------------------------------------------------------------------

def _range_fade_swing(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c = _f(s.close)
    ph1, _, pl1, _ = _pivot_levels(s)
    mixed = _structure_state(s) == 0             # neither HH+HL nor LH+LL
    # `_fresh`: the event is the TOUCH of the extreme, not the state of being beyond it.
    short_ev = mixed & _fresh(_gt(c, ph1))
    long_ev = mixed & _fresh(_lt(c, pl1))
    return _hold_n(long_ev, short_ev)


register(Hypothesis(
    name="structure.range_fade_swing",
    family="structure",
    title="Fade the swing extremes in mixed structure",
    rationale="Deliberately contradicts `bos_swing` on the same level, and only activates where "
              "that one should be weakest: when the sequence of pivots is mixed and there is no "
              "directional structure. The reasoning is about inventory. With no trend to absorb, "
              "whoever provides liquidity at the swing extremes runs no risk of ending up on the "
              "wrong side of a sustained move, so they can defend those levels with size; the "
              "break finds no continuation and price reverts towards the middle of the range. The "
              "strong claim is not 'ranges revert', but that the SAME signal changes sign "
              "according to the structural state.",
    prior="We expect a positive edge ONLY under the mixed-structure filter, and we explicitly "
          "expect the same rule without that filter to give a negative edge. AUDIT NOTE "
          "(2026-09-08): that unfiltered rule was not registered, so this hypothesis's central "
          "falsification criterion could not be run; it has been registered as "
          "`structure.swing_fade_unfiltered` and that is what it must be measured against. A "
          "5-candle fractal leaves the level very close to price, so we expect MANY trades and for "
          "costs to eat a small gross edge: if the net edge does not survive realistic commissions "
          "and spread, the hypothesis is refuted even if the gross one is positive, and it is not "
          "legitimate to rescue it by raising k. We expect it to "
          "fail at regime turns, where the structure looks mixed exactly while the new trend is "
          "being built: there the fades execute against the initial impulse and should be the "
          "worst trades in the whole family. If the result is positive both here and in "
          "`bos_swing` without the regime separating them, the filter is doing nothing and both "
          "conclusions must be discarded.",
    fn=_range_fade_swing,
    params={"fractal_k": _K, "filter": "mixed structure", "horizon": _HOLD},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 12b. CONTROL ADDED IN THE AUDIT (2026-09-08) so that `range_fade_swing` is falsifiable.
#
# What was wrong. The prior of `structure.range_fade_swing` says, literally, that it expects a
# positive edge "ONLY under the mixed-structure filter" and that "the same rule without that filter
# gives a negative edge". That is its falsification criterion and it is the right one: the strong
# claim is not that fading extremes wins, but that the STRUCTURAL STATE changes the sign. The
# problem was that the unfiltered rule was registered nowhere, so the test could not be run: a
# prior whose failure criterion names a control that does not exist cannot fail, and a prior that
# cannot fail is not a prior. The control is registered here, with its own prior and counting as
# one more trial in the multiple-comparisons correction.
# --------------------------------------------------------------------------------------------

def _swing_fade_unfiltered(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c = _f(s.close)
    ph1, _, pl1, _ = _pivot_levels(s)
    # Identical to `range_fade_swing` except for the absence of the `mixed` filter.
    return _hold_n(_fresh(_lt(c, pl1)), _fresh(_gt(c, ph1)))


register(Hypothesis(
    name="structure.swing_fade_unfiltered",
    family="structure",
    title="Fade the swing extremes in any structure",
    rationale="Unconditional control for `range_fade_swing`: fading the touch of the last "
              "confirmed pivot in ALL structural states, not only when the sequence is mixed. It "
              "claims no mechanism of its own —on the contrary, the inventory mechanism that "
              "justifies the fade only holds when there is no trend to absorb— and it is here "
              "because a conditional claim is only testable if the unconditional case is measured "
              "too. Without this control, a positive result for `range_fade_swing` would not "
              "distinguish 'the structural filter contributes' from 'fading swing extremes always "
              "wins and the filter is decorative'.",
    prior="We expect a NEGATIVE edge, and we are committed to that sign in advance: without the "
          "filter, the same rule goes short against every breakout of a live trend, which is the "
          "declared failure mode of all fading. The joint reading is what matters and it is fixed "
          "now: if this one comes out negative and `range_fade_swing` positive, the structural "
          "state does separate regimes and the conditional claim holds; if both come out similar, "
          "the filter does nothing and both must be discarded; if this one comes out POSITIVE, "
          "what we have is generic mean reversion at the swing extremes and `range_fade_swing`'s "
          "inventory argument is false even if its number is good.",
    fn=_swing_fade_unfiltered,
    params={"fractal_k": _K, "filter": "none", "horizon": _HOLD},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))
