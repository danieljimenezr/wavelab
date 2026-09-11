"""SEASONALITY family: the clock and the calendar as an explanatory variable.

The thesis common to the whole family is a single one: **liquidity and participation are not
uniform in time**. There are hours in which the book is held up by American desks and hours in
which it is held up by leveraged retail flow; there are days on which the fiat rails are open and
days on which they are not; there are dates on which open interest expires and dates on which it is
rebuilt. If BTC's return depended on information alone, the clock would be irrelevant. If it also
depends on WHO can trade at each moment, it is not.

It is also the family most prone to spurious findings, and it is worth saying why before looking at
anything: the calendar offers an enormous number of partitions (24 hours, 7 days, 31 days of the
month, 4 quarters, cycle phases) and almost all of them are free to test. With enough cuts, one
always comes out "significant". The defences adopted here, all of them BEFORE running anything:

1. **No window is chosen for its performance.** Each one is anchored to an exogenous, verifiable
   institutional fact (the US cash open, the Tokyo session, the CME weekly close, the perpetual
   funding stamp, Deribit expiry, the halving). If the anchor exists off the chart, the window is
   not a degree of freedom.
2. **Every hypothesis declares how it expects to fail**, and in several cases the falsification
   criterion is stricter than "makes money": if the same effect shows up in equivalent control
   windows, the *seasonal* claim is false even if the P&L is positive.
3. **These hypotheses are NOT independent trials.** Several of them share candles (turn of month
   with monthly expiry; the weekend with Friday evening; US hours with Asian hours, which are
   nearly complementary). A Bonferroni correction over 11 trials is conservative here, and a
   correction that assumes independence is simply wrong: whoever runs the test should use FDR under
   dependence, or correlate the signal series before deciding the effective number of trials. It is
   written down so that it is not decided afterwards.
4. **Unconditional drift.** BTC rises over the sample. Any predominantly long hypothesis will
   inherit that drift. None of the ones in this family can be considered confirmed by its absolute
   expectation: only the delta against the null arm (random entries with the same exposure) counts,
   which is exactly what the backtest harness already computes.

--------------------------------------------------------------------------------------------------
ASSUMED CONVENTIONS (declared here so that the result is interpretable, and so that if the harness
uses a different convention it is known exactly what has to be reinterpreted)

*   ``s.ts`` is the OPEN timestamp of the candle in ms epoch UTC, aligned to the timeframe grid. It
    is the project convention: ``resample_from_1m`` indexes by ``open_time_ms`` and
    ``Bar.__post_init__`` requires ``open_time_ms % tf.ms == 0``.

*   The signal on candle ``i`` is decided with information up to the close of ``i`` and the
    position is held through candle ``i+1``. That is why the hourly windows are evaluated over
    ``ts + timeframe_duration``: the hour that matters is that of the candle that will be HELD, not
    that of the one that has just closed.

    This is **not** looking into the future. Not one datum from candle ``i+1`` is read: all that is
    computed is what the clock will read, which is information known since 1970 and available to
    any participant at any earlier instant. The same argument holds for the calendar (when the last
    Friday of March 2019 falls was known in 2018) and for the halving dates, of which only those
    already past relative to the candle being evaluated are used.

    If the harness instead applied signal ``i`` to the return of candle ``i`` itself, ALL the
    hypotheses in this file would be displaced one candle backwards and would have to be read as
    windows brought forward by one period. It is declared now, not after seeing the numbers.

*   The time-of-day hypotheses are registered only on ``15m`` and ``1h``, where the open and the
    close of the candle fall in the same UTC hour and the assignment is unambiguous. On ``4h`` a
    single candle spans four different hours and "the candle's hour" means nothing; registering
    them there would be manufacturing an ambiguity and then blaming it on the market.

*   Daylight saving time (DST) shifts the American anchors (the cash open, the CME weekly close) by
    one hour for part of the year. That ±1h is accepted instead of introducing a DST calendar: the
    windows are chosen wide enough to contain the anchor under both regimes. It is a declared
    imprecision, not a tunable parameter.

CAUSALITY. Every function in this module uses exclusively data up to and including ``i``. The
forward fill of past references (the day's open, Friday's close) is done with
``np.maximum.accumulate`` over indices, which is a prefix operation: the value at ``i`` can only
come from an index ``<= i``. There are no negative ``shift``s, no ``np.roll``, and no extreme
detection defined against the whole array.
"""

from __future__ import annotations

import numpy as np
import talib

from wavelab.hypotheses.base import Hypothesis, Series, register

_MS_HOUR = 3_600_000
_MS_DAY = 86_400_000

#: Duration of each timeframe. Static metadata (the clock grid), not market data.
_TF_MS: dict[str, int] = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000,
    "1w": 604_800_000,
}

#: Halvings that have already happened, in ms epoch UTC. Only the LAST ONE BEFORE each candle is
#: ever used: a date later than the candle being evaluated is never consulted, so the table
#: introduces no lookahead.
_HALVINGS_MS: np.ndarray = np.array(
    [
        np.datetime64("2012-11-28T15:24", "ms"),  # block 210,000
        np.datetime64("2016-07-09T16:46", "ms"),  # block 420,000
        np.datetime64("2020-05-11T19:23", "ms"),  # block 630,000
        np.datetime64("2024-04-20T00:09", "ms"),  # block 840,000
    ],
    dtype="datetime64[ms]",
).astype(np.int64)


# --------------------------------------------------------------------------------------
# Clock and calendar utilities. All of them return arrays aligned with the series.
# --------------------------------------------------------------------------------------

def _hold_ts(s: Series) -> np.ndarray:
    """Timestamp of the candle DURING which the position decided at ``i`` will be open.

    It is ``ts[i] + duration``: clock arithmetic, not a read of future data (see the module
    docstring). The fallback that differences timestamps only kicks in if a timeframe turns up
    that is not in the table, and it is still calendar information.
    """
    ts = np.asarray(s.ts, dtype=np.int64)
    step = _TF_MS.get(s.tf)
    if step is None:
        step = int(ts[1] - ts[0]) if ts.size > 1 else 0
    return ts + step


def _hour_utc(t: np.ndarray) -> np.ndarray:
    """UTC hour (0-23)."""
    return ((t // _MS_HOUR) % 24).astype(np.int64)


def _weekday(t: np.ndarray) -> np.ndarray:
    """Day of the week using Python's convention: 0 = Monday … 6 = Sunday.

    1970-01-01 (epoch day 0) was a Thursday, which is 3 in this convention; hence the offset.
    """
    return (((t // _MS_DAY) + 3) % 7).astype(np.int64)


def _date(t: np.ndarray) -> np.ndarray:
    """UTC date of each timestamp, as ``datetime64[D]``."""
    return t.astype("datetime64[ms]").astype("datetime64[D]")


def _day_of_month(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (day of month 1..31, number of days that month has)."""
    months = dates.astype("datetime64[M]")
    dom = (dates - months.astype("datetime64[D]")).astype(np.int64) + 1
    length = ((months + np.timedelta64(1, "M")).astype("datetime64[D]")
              - months.astype("datetime64[D]")).astype(np.int64)
    return dom, length


def _last_friday(months: np.ndarray) -> np.ndarray:
    """Date of the last Friday of each ``datetime64[M]`` month.

    It is the Deribit expiry (08:00 UTC) and the CME one. It is computed from the calendar, which
    is exogenous and known years in advance.
    """
    last = (months + np.timedelta64(1, "M")).astype("datetime64[D]") - np.timedelta64(1, "D")
    wd = (last.astype(np.int64) + 3) % 7            # 0 = Monday, 4 = Friday
    return last - ((wd - 4) % 7).astype("timedelta64[D]")


def _ffill_idx(flag: np.ndarray) -> np.ndarray:
    """Index of the last flagged position at ``<= i``, or -1 if there has not been one yet.

    ``np.maximum.accumulate`` is a PREFIX operation: the result at ``i`` cannot depend on any later
    position. It is the piece that keeps every forward fill causal.
    """
    n = flag.size
    idx = np.where(flag, np.arange(n, dtype=np.int64), np.int64(-1))
    return np.maximum.accumulate(idx)


def _f64(x: np.ndarray) -> np.ndarray:
    """talib requires contiguous float64; a view of another dtype either raises or computes
    garbage."""
    return np.ascontiguousarray(x, dtype=np.float64)


# --------------------------------------------------------------------------------------
# 1. US cash-session hours
# --------------------------------------------------------------------------------------

def _us_cash_hours(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    h = _hour_utc(_hold_ts(s))
    out[(h >= 13) & (h <= 20)] = 1
    return out


register(Hypothesis(
    name="seasonality.us_cash_hours_long",
    family="seasonality",
    title="Long through the New York session",
    rationale="Since the spot ETFs were approved, BTC's marginal buyer trades on New York hours: "
              "ETF creations and redemptions are struck against the 16:00 ET close, CME liquidity "
              "concentrates in the US session, and the desks that arbitrage the spot-futures basis "
              "only have staff in front of a screen during those hours. A recurring, "
              "institutional, price-insensitive buying flow concentrated in a fixed hourly window "
              "has to leave a mark on the average return of those hours. For the rest of the day "
              "the book is held up by market makers who by definition have no direction.",
    prior="We expect POSITIVE average drift on the candles held between 13:00 and 21:00 UTC, and "
          "higher than that of the null arm at the same exposure. Falsifiable in two ways, and "
          "both count as failure: (a) if the edge does not survive comparison with the null arm, "
          "all we are measuring is BTC's unconditional drift spread across the hours; (b) the "
          "proposed mechanism (ETF/CME flow) is recent, so we expect a weak or nil effect before "
          "2021 and a sharper one from January 2024 — if the effect turns out to be uniform across "
          "the whole sample, the mechanism we propose is NOT the one producing it and the "
          "hypothesis is wrongly explained even if it gets the sign right.",
    fn=_us_cash_hours,
    params={"start_hour_utc": 13, "end_hour_utc": 20, "anchor": "US cash session"},
    timeframes=("15m", "1h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------
# 2. Asian hours
# --------------------------------------------------------------------------------------

def _asia_hours(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    h = _hour_utc(_hold_ts(s))
    out[(h >= 0) & (h <= 7)] = -1
    return out


register(Hypothesis(
    name="seasonality.asia_hours_short",
    family="seasonality",
    title="Short the thin Asian hours",
    rationale="Between 00:00 and 08:00 UTC (09:00-17:00 in Tokyo) the book is at its thinnest: the "
              "European and American desks are not there, and what remains is mostly heavily "
              "leveraged retail perpetual flow. With a thin book an order of the same size moves "
              "price further, and that displacement is mechanical, not informative, so it tends to "
              "be given back once London's depth arrives. East Asian participation also contracted "
              "after the Chinese ban of 2021, so the marginal flow in those hours looks more like "
              "noise than like informed accumulation.",
    prior="We expect the drift of the 00:00-08:00 UTC hours to come out BELOW the unconditional "
          "drift and, specifically, for a short in that window to have positive expectancy net of "
          "costs. It fails if: (a) the short's expectancy is negative, which is the default "
          "outcome when shorting an asset with positive secular drift; (b) the result is an exact "
          "mirror of `us_cash_hours_long`'s — the two windows are nearly complementary, so if "
          "their signal series are correlated near -1 they are not two findings but one, and have "
          "to be counted as ONE trial. We also expect the effect to be weaker before 2021, when "
          "Asian participation was greater and its flow genuinely was informative.",
    fn=_asia_hours,
    params={"start_hour_utc": 0, "end_hour_utc": 7, "anchor": "Tokyo session"},
    timeframes=("15m", "1h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------
# 3. London-New York overlap: continuation of the day's direction
# --------------------------------------------------------------------------------------

def _overlap_continuation(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n == 0:
        return out

    ts = np.asarray(s.ts, dtype=np.int64)
    close = np.asarray(s.close, dtype=np.float64)
    open_ = np.asarray(s.open, dtype=np.float64)

    # Open of the UTC day in progress: that of the day's first candle, forward-filled.
    day = ts // _MS_DAY
    day_start = np.empty(n, dtype=bool)
    day_start[0] = True
    day_start[1:] = day[1:] != day[:-1]
    idx = _ffill_idx(day_start)                  # day_start[0] is True ⇒ never left at -1
    anchor = open_[idx]

    h = _hour_utc(_hold_ts(s))
    window = (h >= 13) & (h <= 16)

    move = close - anchor
    valid = window & ~np.isnan(move)
    out[valid & (move > 0)] = 1
    out[valid & (move < 0)] = -1
    return out


register(Hypothesis(
    name="seasonality.overlap_day_trend",
    family="seasonality",
    title="Follow the day's direction into the overlap",
    rationale="The overlap of the London afternoon with the New York morning (13:00-16:30 UTC) is "
              "the only stretch in which European and American desks are present at the same time, "
              "and therefore the one with the greatest aggregate depth of the day. A directional "
              "intent formed during the thin hours cannot be executed in size until that depth "
              "arrives: institutional slicing algorithms (VWAP/TWAP) do most of their work there. "
              "The mechanism predicts continuation, not reversal: the direction accumulated since "
              "the 00:00 UTC open should EXTEND itself in that window.",
    prior="We expect positive expectancy from following the sign of the day's move so far during "
          "13:00-17:00 UTC. It fails if the expectancy is indistinguishable from zero or from the "
          "null arm, and we specifically expect it to fail in a low-volatility range, where price "
          "orbits the day's open and the sign is a coin flip. Additional and decisive criterion: "
          "if the same day-sign following works just as well in any arbitrary four-hour control "
          "window, then what we have is generic intraday momentum and the SEASONAL claim is false, "
          "whatever it earns.",
    fn=_overlap_continuation,
    params={"start_hour_utc": 13, "end_hour_utc": 16, "anchor": "UTC day open (00:00)"},
    timeframes=("15m", "1h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------
# 4. Weekend
# --------------------------------------------------------------------------------------

def _weekend(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    wd = _weekday(_hold_ts(s))
    out[wd >= 5] = -1                     # Saturday (5) and Sunday (6)
    return out


register(Hypothesis(
    name="seasonality.weekend_short",
    family="seasonality",
    title="Short the weekend, fiat rails closed",
    rationale="At the weekend the fiat rails are closed: no bank transfers, no creation or "
              "redemption of ETF shares, and the CME does not trade. What is left is the 24/7 "
              "perpetual with market makers who trim inventory because they cannot hedge on the "
              "regulated market, so the book thins out and spreads widen. The buying flow that has "
              "a calendar (institutional, payroll, ETF) disappears for 48 hours, while the forced "
              "selling flow — leverage liquidations — never rests. The asymmetry between a buyer "
              "who keeps office hours and an automatic seller is the mechanism.",
    prior="We expect Saturday and Sunday drift to be less than or equal to zero and clearly below "
          "that of weekdays. It fails if the weekend shows positive drift comparable to the "
          "weekday one. We also expect the effect to have WEAKENED since 2021, when perpetuals and "
          "automated market making came to dominate volume: if it only shows up in the 2017-2020 "
          "subsample, it is a fossil of a market that no longer exists and we will declare it as "
          "such instead of selling it as a tradeable regularity.",
    fn=_weekend,
    params={"days": "Saturday and Sunday UTC"},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------
# 5. Friday-evening de-risking
# --------------------------------------------------------------------------------------

def _friday_evening(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    t = _hold_ts(s)
    out[(_weekday(t) == 4) & (_hour_utc(t) >= 18)] = -1
    return out


register(Hypothesis(
    name="seasonality.friday_derisk_short",
    family="seasonality",
    title="Short into the CME weekly close",
    rationale="The CME weekly close (21:00 UTC in summer time, 22:00 in winter) forces basis desks "
              "to flatten or roll whatever they cannot hold against a closed market, and the "
              "leveraged longs paying funding trim before two days without fiat rails or regulated "
              "hedging. Both flows point the same way — sell — and concentrate in the last hours "
              "of Friday. It is a different hypothesis from the weekend one: here the mechanism is "
              "the flattening BEFORE the venue closes, not the thinness of the book DURING the two "
              "following days.",
    prior="We expect negative expectancy for a long (positive for the short) between 18:00 and "
          "24:00 UTC on Friday, smaller in magnitude than the weekend effect itself. It fails if "
          "the expectancy is not negative, and also — this is the important part — if the same "
          "negative drift shows up in hours 18-24 of the other weekdays: in that case it would be "
          "a time-of-day effect, already covered by other hypotheses in this family, and not a "
          "Friday effect. The window is declared wide (six hours) on purpose so as to contain the "
          "CME close in both summer and winter time without having to choose.",
    fn=_friday_evening,
    params={"day": "Friday", "start_hour_utc": 18, "end_hour_utc": 23,
            "anchor": "CME weekly close"},
    timeframes=("15m", "1h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------
# 6. CME gap: reversion towards Friday's close
# --------------------------------------------------------------------------------------

def _cme_gap(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n == 0:
        return out

    ts = np.asarray(s.ts, dtype=np.int64)
    close = np.asarray(s.close, dtype=np.float64)

    # Reference: the close of the candle that OPENS on Friday at 21:00 UTC (the last full hour
    # before the CME weekly close). It is flagged with the clock of the candle ITSELF,
    # not with that of the next one.
    flag = (_weekday(ts) == 4) & (_hour_utc(ts) == 21)
    idx = _ffill_idx(flag)
    has_ref = idx >= 0
    ref = close[np.clip(idx, 0, None)]

    # Freshness: between Friday 21:00 and Monday 12:00 there are ~63 hours. If the last reference
    # is older than that, there was a data gap and the reference no longer describes this weekend.
    fresh = (np.arange(n, dtype=np.int64) - idx) <= 96

    t = _hold_ts(s)
    wd, h = _weekday(t), _hour_utc(t)
    window = ((wd == 6) & (h >= 22)) | ((wd == 0) & (h <= 11))

    atr = talib.ATR(_f64(s.high), _f64(s.low), _f64(s.close), timeperiod=14)
    atr_ok = ~np.isnan(atr)

    gap = close - ref
    valid = (window & has_ref & fresh & atr_ok & ~np.isnan(gap)
             & (np.abs(gap) > np.where(atr_ok, atr, np.inf)))
    out[valid & (gap > 0)] = -1   # above Friday's close ⇒ the gap points down
    out[valid & (gap < 0)] = 1
    return out


register(Hypothesis(
    name="seasonality.cme_gap_monday",
    family="seasonality",
    title="Fade the CME gap into Monday",
    rationale="CME BTC futures stop trading on Friday afternoon and reopen on Sunday night, while "
              "spot never stops: any weekend move opens a gap on the CME chart. On reopening, the "
              "desks that arbitrage the basis have to rebuild the hedge between the two markets, "
              "and that flow pulls spot towards the last price traded on the regulated venue. The "
              "second ingredient is reflexive but real: the gap is a level the whole market is "
              "watching, and orders pile up where people are looking. We require the gap to exceed "
              "one ATR(14) because below that there is no level, there is noise.",
    prior="We expect slightly positive expectancy from fading the weekend's move (short if price "
          "is above Friday's close, long if it is below) between the Sunday reopen and Monday "
          "midday. It fails if the weekend move turns out to be informative rather than noise, "
          "that is, if it continues; we expect exactly that failure in strong trends and on "
          "weekends with macro news or counterparty failures. The direction of the failure is as "
          "informative as success: if it loses systematically, what we have learned is that the "
          "weekend DOES move information.",
    fn=_cme_gap,
    params={"atr": 14, "threshold_in_atr": 1.0, "anchor": "Friday 21:00 UTC close",
            "window": "Sunday 22:00 UTC - Monday 12:00 UTC", "max_staleness_bars": 96},
    timeframes=("1h",),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------
# 7. The hour before the funding stamp
# --------------------------------------------------------------------------------------

def _pre_funding(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)

    roc = talib.ROC(_f64(s.close), timeperiod=8)      # return over the funding window (8 h)
    finite = ~np.isnan(roc)

    h = _hour_utc(_hold_ts(s))
    window = (h == 7) | (h == 15) | (h == 23)

    valid = window & finite
    out[valid & (roc > 0)] = -1                       # fade the window's move
    out[valid & (roc < 0)] = 1
    return out


register(Hypothesis(
    name="seasonality.pre_funding_fade",
    family="seasonality",
    title="Fade the move before the funding stamp",
    rationale="Binance and the other perpetual venues settle funding at 00:00, 08:00 and 16:00 "
              "UTC. When the eight-hour window has been very directional, the crowded side is the "
              "one that pays, and part of that side closes in the hour before the stamp precisely "
              "so as not to pay it; symmetrically, the market maker who has spent eight hours "
              "accumulating the opposite inventory reduces it before the cut-off. Both flows push "
              "against the window's move and concentrate in that one specific hour. The 8-hour "
              "period is not a chosen parameter: it is the length of the funding interval.",
    prior="We expect NEGATIVE autocorrelation (fading) in hours 07, 15 and 23 UTC, and not in the "
          "other twenty-one. The falsification criterion is harder than profitability and it is "
          "the whole reason the hypothesis exists: if the same fade of the 8-candle ROC works just "
          "as well in the remaining hours, what we have is generic mean reversion and the seasonal "
          "claim is FALSE even if the P&L is positive. We also expect it to fail in a sustained "
          "trend, where the crowded side is also the right side and whoever is paying funding "
          "keeps on winning.",
    fn=_pre_funding,
    params={"funding_window_h": 8, "pre_stamp_hours_utc": (7, 15, 23)},
    timeframes=("1h",),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------
# 8. Turn of the month
# --------------------------------------------------------------------------------------

def _turn_of_month(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    dates = _date(_hold_ts(s))
    dom, length = _day_of_month(dates)
    out[(dom <= 3) | (dom == length)] = 1
    return out


register(Hypothesis(
    name="seasonality.turn_of_month_long",
    family="seasonality",
    title="Long the last day of the month and the first three",
    rationale="Recurring purchases have a calendar: payroll, automatic DCA programmes on the "
              "exchanges and periodic contributions from funds and treasuries all cluster at month "
              "end and in the first days of the next one. That flow is price-insensitive — it buys "
              "the amount, whatever it costs — and it arrives synchronised into a book whose "
              "sellers do not share that synchrony. It is the same turn-of-month effect documented "
              "in equities since the 1980s, transplanted to an asset whose retail base buys a "
              "fixed amount rather than on a one-off conviction.",
    prior="We expect a small positive edge on the last calendar day of the month and the first "
          "three of the next, above the null arm. It fails if the edge is nil, which is what to "
          "expect if crypto flow is not synchronised with any payroll, being a global base with no "
          "common payday. Second criterion, and a stricter one: if the edge is concentrated in "
          "just ONE of the four days, it is a coincidence and not a flow — the mechanism predicts "
          "a BLOCK of days, not a date. We expect it to weaken in the months whose 1st falls at a "
          "weekend, with the fiat rails closed.",
    fn=_turn_of_month,
    params={"days": "last calendar day of the month and days 1, 2 and 3 of the next"},
    timeframes=("4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------
# 9. The week before quarterly expiry
# --------------------------------------------------------------------------------------

def _pre_quarterly_expiry(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    dates = _date(_hold_ts(s))
    months = dates.astype("datetime64[M]")
    quarterly = np.isin((months.astype(np.int64) % 12), (2, 5, 8, 11))   # Mar, Jun, Sep, Dec
    days_to = (_last_friday(months) - dates).astype(np.int64)
    out[quarterly & (days_to >= 1) & (days_to <= 7)] = -1
    return out


register(Hypothesis(
    name="seasonality.pre_quarterly_expiry_short",
    family="seasonality",
    title="Short the week before quarterly expiry",
    rationale="The largest open interest in options and futures (Deribit and CME) expires on the "
              "last Friday of March, June, September and December. In the week before, that "
              "interest is closed or rolled: the spot-futures basis compresses, the desks that "
              "were funding the carry unwind the spot long that hedged it, and options holders "
              "adjust delta against an ever-nearer expiry. The net of those trades removes LONG "
              "leverage, which is the structurally dominant side in crypto, and removing the "
              "dominant side pushes the price down.",
    prior="We expect slightly negative drift in the seven days before quarterly expiry. It fails "
          "if the drift is positive or indistinguishable from the rest of the quarter. A warning "
          "that is part of the registration: there are four quarterly expiries a year, so the "
          "effective sample is a few dozen EVENTS, not thousands of candles; the "
          "multiple-comparison correction has to be applied over events and the confidence "
          "intervals computed by blocks, not per candle, or the standard error will come out "
          "falsely tiny. If the effect only shows up in the quarters of one particular bull year, "
          "what we have is that year and not the expiry.",
    fn=_pre_quarterly_expiry,
    params={"months": (3, 6, 9, 12), "days_before": 7,
            "anchor": "last Friday of the quarter, 08:00 UTC"},
    timeframes=("4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------
# 10. The days after monthly expiry
# --------------------------------------------------------------------------------------

def _post_monthly_expiry(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    dates = _date(_hold_ts(s))
    months = dates.astype("datetime64[M]")

    # The expiry in force is this month's if it has already happened; if not, the previous month's.
    # Without this, the first days of a month (when the last Friday falls on the 30th or 31st)
    # would be left out.
    exp_current = _last_friday(months)
    exp_previous = _last_friday(months - np.timedelta64(1, "M"))
    in_force = np.where(dates >= exp_current, exp_current, exp_previous)

    days_since = (dates - in_force).astype(np.int64)
    out[(days_since >= 1) & (days_since <= 3)] = 1
    return out


register(Hypothesis(
    name="seasonality.post_monthly_expiry_long",
    family="seasonality",
    title="Long the three days after monthly expiry",
    rationale="Once the monthly expiry has passed (last Friday, 08:00 UTC on Deribit), the delta "
              "hedging that anchors price to the strikes with the most open interest disappears, "
              "and positioning is rebuilt: expired calls are rolled to the next month, the desks "
              "reopen the carry and the leverage that was trimmed comes back. That rebuilding of "
              "long positioning is, mechanically, buying flow spread over the immediately "
              "following days.",
    prior="We expect small positive drift in the three days after each monthly expiry. It fails if "
          "it is nil, and most especially if it turns out to be the symmetric mirror of "
          "`pre_quarterly_expiry_short`: if what we are measuring is a single pivot around the "
          "expiry, they are not two findings but one and it has to be counted as ONE trial. We "
          "expect it to fail in the months where the expiry falls next to the turn of the month, "
          "where its window blurs into `turn_of_month_long`'s and neither of the two is "
          "attributable.",
    fn=_post_monthly_expiry,
    params={"days_after": 3, "anchor": "last Friday of each month, 08:00 UTC"},
    timeframes=("4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------
# 11. Halving cycle phase
# --------------------------------------------------------------------------------------

def _halving_phase(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n == 0:
        return out

    t = _hold_ts(s)
    # `side="right"` ⇒ only halvings STRICTLY at or before the candle. Never the next one.
    k = np.searchsorted(_HALVINGS_MS, t, side="right") - 1
    has_halving = k >= 0
    days = (t - _HALVINGS_MS[np.clip(k, 0, None)]) // _MS_DAY

    out[has_halving & (days >= 180) & (days <= 540)] = 1      # months 6-18: expansion
    out[has_halving & (days > 540) & (days <= 900)] = -1      # months 18-30: contraction
    return out


register(Hypothesis(
    name="seasonality.halving_cycle_phase",
    family="seasonality",
    title="Long the halving expansion, short the contraction",
    rationale="The halving cuts miner issuance in half from one block to the next. Miners are "
              "structural sellers — they pay for energy in fiat currency — so the daily selling "
              "flow the market has to absorb halves on a date known in advance. The folk model of "
              "the cycle holds that the effect is not instantaneous but accumulates over the "
              "following 6-18 months, as the supply deficit shows itself, and that it gives way to "
              "a contraction of about twelve months once price has already discounted it and the "
              "marginal buyer runs out.",
    prior="We expect positive expectancy in months 6-18 after the halving and negative expectancy "
          "in months 18-30. WARNING, and this is as much part of the pre-registration as the sign: "
          "the effective sample is TWO complete cycles in a Binance series that starts in 2017 "
          "—n≈2, not thousands of candles— and no multiple-comparison correction fixes an n of 2. "
          "Worse still, the 6-, 18- and 30-month boundaries come from a description written while "
          "LOOKING at those very cycles: it is in-sample contamination that can no longer be "
          "undone. We declare in advance that we will read the result as descriptive and never as "
          "evidence, and we register it precisely so that its outcome —success or failure— is on "
          "the record instead of turning up later as an idea we had 'always had'. Additional "
          "falsification of the mechanism: if the cycle that began in 2024, in which issuance is "
          "already insignificant next to daily ETF volume, does not reproduce the pattern, the "
          "mechanism is dead even if the historical average survives.",
    fn=_halving_phase,
    params={"expansion_days": (180, 540), "contraction_days": (541, 900),
            "halvings": ("2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20")},
    timeframes=("1d",),
    min_warmup=200,
))
