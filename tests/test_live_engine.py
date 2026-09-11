"""The live engine must produce EXACTLY the same thing as the offline path.

If live resampling and backtest resampling differed even in the last decimal, the whole promise of
«one single function» would be false: the backtest would be measuring one series and the user would
be looking at another.

Hanging off that same path are the two things the promise is FOR: the labels the backtest learns
from (triple barrier) and the numbers it hands back. A label whose geometry is not the geometry of
the plan on screen, or a statistic that quietly drops the fees, breaks the promise just as
thoroughly as a mis-resampled bar — and it breaks it upwards, which is worse, because the result
looks better rather than broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wavelab.backtest.run import BacktestResult, Signal, run_backtest
from wavelab.core.timeframes import (
    BY_NAME,
    MIN_SOURCE_COVERAGE,
    TF_1H,
    TF_1M,
    TF_15M,
    resample_from_1m,
)
from wavelab.core.types import Direction, MaturityLevel, PivotKind, Verdict
from wavelab.engine import live as live_module
from wavelab.engine.live import LiveEngine, Mode
from wavelab.labeling.barriers import Outcome, resolve_triple_barrier

from .conftest import SYMBOL, T0, make_bars

#: 20 days of 1m bars: long enough for real wave structure to form, resolve and be counted, and
#: still under a second to build. Shorter than this and the backtest tests below become vacuous.
TRIGGER_TF = "1h"
TIMEFRAMES = ("1m", "15m", "1h")
MAX_BARS_HOLD = 12
COOLDOWN_BARS = 3


def _engine(tfs=("1m", "15m", "1h")) -> LiveEngine:
    return LiveEngine(SYMBOL, list(tfs), ring_capacity=4096, trigger_tf="15m")


@pytest.fixture(scope="module")
def bars_20d() -> list:
    return make_bars(60 * 24 * 20, tf=TF_1M, seed=3)


@pytest.fixture(scope="module")
def backtest_20d(bars_20d) -> BacktestResult:
    return run_backtest(bars_20d, trigger_tf=TRIGGER_TF, timeframes=TIMEFRAMES,
                        max_bars_hold=MAX_BARS_HOLD, cooldown_bars=COOLDOWN_BARS)


@pytest.fixture(scope="module")
def replay_20d(bars_20d):
    """The same 20 days again, this time keeping what `decide()` published on every trigger bar.

    The backtest throws the cards away and keeps only the numbers it copied off them, so nothing
    downstream can tell whether the copy is faithful. Two of the properties below are exactly that
    question, and they need both halves side by side. One extra pass over the bars costs ~0.09s.
    """
    e = LiveEngine(SYMBOL, list(TIMEFRAMES), 8192, TRIGGER_TF)
    trigger = BY_NAME[TRIGGER_TF]
    cards: dict[int, dict] = {}
    closes: dict[int, float] = {}
    for b in bars_20d:
        for htf in e.on_bar_1m(b):
            if htf.tf is not trigger:
                continue
            closes[htf.open_time_ms] = htf.close
            cards[htf.open_time_ms] = e.decide(TRIGGER_TF, htf.close)
    return e, cards, closes


class TestLiveResampling:
    def test_it_matches_the_offline_path_bar_for_bar(self):
        bars = make_bars(600, tf=TF_1M)
        e = _engine()
        for b in bars:
            e.on_bar_1m(b)

        df = pd.DataFrame(
            {"open": [b.open for b in bars], "high": [b.high for b in bars],
             "low": [b.low for b in bars], "close": [b.close for b in bars],
             "volume": [b.volume for b in bars]},
            index=pd.Index([b.open_time_ms for b in bars], name="open_time_ms"),
        )
        for tf in (TF_15M, TF_1H):
            offline = resample_from_1m(df, tf)
            offline = offline[offline["n_source_bars"] == tf.expected_source_bars]
            ring = e.state.rings[tf.name]
            w = ring.window(len(ring))
            common = offline.index.intersection(w.ts)
            assert len(common) >= 4, f"too few {tf.name} bars to compare"
            for ts in common:
                i = int(np.flatnonzero(w.ts == ts)[0])
                o = offline.loc[ts]
                assert w.open[i] == pytest.approx(o["open"]), f"{tf.name} open at {ts}"
                assert w.high[i] == pytest.approx(o["high"]), f"{tf.name} high at {ts}"
                assert w.low[i] == pytest.approx(o["low"]), f"{tf.name} low at {ts}"
                assert w.close[i] == pytest.approx(o["close"]), f"{tf.name} close at {ts}"
                assert w.volume[i] == pytest.approx(o["volume"]), f"{tf.name} volume at {ts}"

    def test_it_counts_the_source_bars(self):
        e = _engine()
        for b in make_bars(120, tf=TF_1M):
            e.on_bar_1m(b)
        w = e.state.rings["1h"].window(2)
        assert list(w.n_source_bars) == [60, 60]

    def test_it_marks_a_badly_covered_bar_as_a_gap(self):
        """A 1h bar built out of 20 minutes is not a 1h bar."""
        e = _engine()
        for b in make_bars(120, tf=TF_1M, drop=set(range(40))):
            e.on_bar_1m(b)
        w = e.state.rings["1h"].window(1)
        assert int(w.n_source_bars[0]) == 60, "the 2nd hour is complete"
        assert not bool(w.is_gap[0])

    def test_a_thin_hour_travels_labelled_as_a_gap(self):
        """An hour built out of half its minutes is not an hour, and it has to SAY so.

        The neighbouring test only proves that a COMPLETE bar is not flagged — which an engine
        that never flags anything also passes. This is the other half: if `is_gap` stops being
        computed, a bar made of 30 minutes enters the ring indistinguishable from a real one, and
        every ATR, every pivot and every backtest statistic built on top of it describes a market
        that was never there. Nothing downstream can recover the difference afterwards.
        """
        e = _engine()
        closed: list = []
        # The 2nd hour loses 30 of its 60 minutes. Its closing minute survives, so it still closes.
        for b in make_bars(120, tf=TF_1M, drop=set(range(70, 100))):
            closed += e.on_bar_1m(b)

        thin = [b for b in closed if b.tf is TF_1H][-1]
        assert thin.n_source_bars == 30, (
            f"the 2nd hour should have been built from 30 minutes, not {thin.n_source_bars}: "
            "the fixture no longer produces a thin bar and this test proves nothing"
        )
        assert thin.is_gap, (
            "an hour built out of 30 of its 60 minutes was handed on as a complete bar: is_gap is "
            "not being computed, so nothing downstream can tell a real hour from half of one"
        )
        w = e.state.rings["1h"].window(1)
        assert bool(w.is_gap[0]), "the thin hour was stored in the ring without its gap flag"

    @pytest.mark.parametrize(("minutes", "gap"), [(53, True), (54, False), (55, False)])
    def test_the_coverage_line_is_drawn_where_the_constant_says(self, minutes, gap):
        """Both neighbouring tests probe 30/60 and 60/60 — nowhere near where the line actually is.

        A real feed does not lose half an hour; it loses three or four minutes, right against the
        90% line. So the boundary is where every hour a live user ever sees gets classified, and
        both ways of getting it wrong are silent. Comparing inclusively relabels the last good hour
        a gap; dropping the constant so any missing minute counts flags virtually every hour on a
        real feed. Either one poisons `gaps_in_window`, the health badge and every downstream
        filter that skips gapped bars, and the bars themselves still look perfectly ordinary.
        """
        assert int(MIN_SOURCE_COVERAGE * 60) == 54, (
            f"the 1h coverage floor moved to {int(MIN_SOURCE_COVERAGE * 60)} minutes; the cases "
            "below were chosen to straddle 54 and no longer straddle anything"
        )
        e = _engine()
        closed: list = []
        # The 2nd hour loses its minutes from the front; the closing minute always survives.
        for b in make_bars(120, tf=TF_1M, drop=set(range(60, 120 - minutes))):
            closed += e.on_bar_1m(b)

        thin = [b for b in closed if b.tf is TF_1H][-1]
        assert thin.n_source_bars == minutes, (
            f"the fixture built the 2nd hour from {thin.n_source_bars} minutes instead of "
            f"{minutes}: this case no longer sits where it was aimed"
        )
        assert thin.is_gap is gap, (
            f"an hour made of {minutes} of its 60 minutes was labelled is_gap={thin.is_gap}; the "
            f"90% floor is 54 minutes, so it must be {gap}. The gap flag is being decided by a "
            "different rule from the one the constant declares"
        )

    def test_the_in_flight_bar_closes_nothing(self):
        e = _engine()
        bars = make_bars(60, tf=TF_1M)
        for b in bars[:-1]:
            e.on_bar_1m(b)
        before = len(e.state.rings["1h"])
        last = bars[-1]
        unclosed = type(last)(
            symbol=last.symbol, tf=last.tf, open_time_ms=last.open_time_ms,
            open=last.open, high=last.high, low=last.low, close=last.close,
            volume=last.volume, is_closed=False)
        assert e.on_bar_1m(unclosed) == []
        assert len(e.state.rings["1h"]) == before
        assert e.state.provisional is unclosed

    def test_duplicates_from_the_healing_pass_are_ignored(self):
        e = _engine()
        bars = make_bars(30, tf=TF_1M)
        for b in bars:
            e.on_bar_1m(b)
        n = len(e.state.rings["1m"])
        for b in bars[-10:]:
            assert e.on_bar_1m(b) == []
        assert len(e.state.rings["1m"]) == n


class TestCatchUpMode:
    def test_it_starts_warming_up_and_emits_nothing(self):
        e = _engine()
        assert e.state.health.mode is Mode.WARMUP
        assert not e.emitting

    def test_a_large_lag_prevents_emitting(self):
        e = _engine()
        for b in make_bars(120, tf=TF_1M, start_ms=1_600_000_000_000 - (1_600_000_000_000 % TF_1M.ms)):
            e.on_bar_1m(b)
        # The bars are from 2020: the lag is measured in years.
        e.update_health(connected=True, reconnects=0, healed=0, silent_seconds=0.0)
        assert e.state.health.mode is Mode.CATCH_UP
        assert not e.emitting, (
            "emitting while catching up describes a price that has already been and gone; it is "
            "the likeliest way to lose the user's trust in the first week"
        )

    def test_reloading_history_goes_back_to_warming_up(self, monkeypatch):
        """History is replayed through the very same `on_bar` as the live path.

        That is the point — one function, no second implementation — and it is also why loading
        history has to put the badge back to WARMUP first. Re-hydrating an engine that is already
        LIVE (after a long outage, on a symbol change) walks a week of bars through the emitting
        path: anything asking the engine what it thinks meanwhile is answered with a decision card
        drawn from a half-loaded ring, at a price from last Tuesday, labelled live.
        """
        now = {"ms": T0 + 600 * TF_1M.ms}
        monkeypatch.setattr(live_module.time, "time", lambda: now["ms"] / 1000.0)

        e = _engine()
        e.warmup(make_bars(600, tf=TF_1M, start_ms=T0))
        e.update_health(connected=True, reconnects=0, healed=0, silent_seconds=0.0)
        assert e.emitting, "precondition: the engine is up to date and emitting"

        e.warmup(make_bars(600, tf=TF_1M, start_ms=T0))
        assert e.state.health.mode is Mode.WARMUP, (
            f"a second warm-up left the engine in {e.state.health.mode}: history is being replayed "
            "with the badge still claiming the data is live"
        )
        assert not e.emitting, "nothing may be emitted while history is loading"

    def test_an_outage_stops_emission_and_only_catching_up_restarts_it(self, monkeypatch):
        """The whole round trip: live → the laptop sleeps → live again.

        `on_bar` is deliberately the same code live and in replay, so on waking the engine will
        cheerfully compute an entry zone around a price that went past two hours ago and present it
        as current. The three transitions each guard a different way of getting that wrong: not
        noticing the outage at all, leaving CATCH_UP before the data has caught up, and never
        leaving it once the hole is filled (at which point the tool is silent for good and the user
        assumes it is broken).
        """
        now = {"ms": T0 + 600 * TF_1M.ms}
        monkeypatch.setattr(live_module.time, "time", lambda: now["ms"] / 1000.0)

        e = _engine()                                   # trigger timeframe: 15m
        e.warmup(make_bars(600, tf=TF_1M, start_ms=T0))  # the last minute closes exactly at `now`
        e.update_health(connected=True, reconnects=0, healed=0, silent_seconds=0.0)
        assert e.state.health.mode is Mode.LIVE and e.emitting, (
            "up to date and connected, the engine must be LIVE and emitting"
        )
        assert e.check_clock() is Mode.LIVE, (
            "an ordinary tick with no clock jump was mistaken for an outage: the engine would stop "
            "emitting during normal operation"
        )

        # The process is suspended for two hours: eight trigger bars go by with no data at all.
        now["ms"] += 8 * TF_15M.ms
        assert e.check_clock() is Mode.CATCH_UP, (
            "a two-hour wall-clock jump went unnoticed; the engine still believes it is live and "
            "will publish an entry zone at a price that is two hours stale"
        )
        h = e.update_health(connected=True, reconnects=1, healed=0, silent_seconds=7200.0)
        assert h.lag_bars > 1.0, f"the lag should be about 8 trigger bars, it reads {h.lag_bars}"
        assert e.state.health.mode is Mode.CATCH_UP and not e.emitting, (
            f"the engine went back to emitting {h.lag_bars:.1f} trigger bars behind the market"
        )

        # The healing pass delivers the 120 minutes that were missed.
        for b in make_bars(120, tf=TF_1M, start_ms=T0 + 600 * TF_1M.ms):
            e.on_bar_1m(b)
        h = e.update_health(connected=True, reconnects=1, healed=120, silent_seconds=0.0)
        assert h.lag_bars <= 1.0, f"the hole is filled, the lag should be tiny: {h.lag_bars}"
        assert e.state.health.mode is Mode.LIVE and e.emitting, (
            "the hole is filled and the engine is up to date, but it never left CATCH_UP: it would "
            "stay mute for the rest of the session"
        )


    def _live(self, now: dict) -> LiveEngine:
        """An engine warmed up to `now` and emitting. `_engine` is built under the patched clock,
        so `check_clock` starts from `now` with no jump behind it."""
        e = _engine()
        e.warmup(make_bars(600, tf=TF_1M, start_ms=T0))
        e.update_health(connected=True, reconnects=0, healed=0, silent_seconds=0.0)
        assert e.state.health.mode is Mode.LIVE and e.emitting, "precondition: LIVE and emitting"
        return e

    def test_two_trigger_bars_of_silence_is_not_yet_an_outage(self, monkeypatch):
        """Where the outage detector fires decides whether the tool is usable at all.

        The round-trip test above jumps eight trigger bars, so it holds for any tolerance from one
        bar to seven and says nothing about where the line is. Widen the tolerance and the engine
        keeps emitting through an outage it did not notice — an entry zone at a price that has
        already gone, which is the failure CATCH_UP exists to prevent. Tighten it and ordinary
        scheduling jitter reads as an outage and the tool goes mute during normal operation. Both
        sides of the declared threshold — strictly more than two trigger bars — are pinned here.
        """
        now = {"ms": T0 + 600 * TF_1M.ms}
        monkeypatch.setattr(live_module.time, "time", lambda: now["ms"] / 1000.0)
        e = self._live(now)

        now["ms"] += 2 * TF_15M.ms
        assert e.check_clock() is Mode.LIVE, (
            "a gap of exactly two trigger bars was called an outage: the tolerance is 'more than "
            "two', and a tighter one makes ordinary scheduling jitter silence the tool"
        )
        now["ms"] += 3 * TF_15M.ms
        assert e.check_clock() is Mode.CATCH_UP, (
            "three trigger bars of wall clock went by with no data and the engine still calls "
            "itself LIVE: it will publish an entry zone at a price that is 45 minutes stale"
        )
        assert not e.emitting

    def test_catching_up_ends_exactly_when_the_lag_is_back_inside_one_trigger_bar(
        self, monkeypatch
    ):
        """CATCH_UP is left on a lag threshold, and both mistakes are silent and opposite.

        Leave it too early and the engine emits while still behind the market — the thing the mode
        exists to stop. Leave it too late (or never) and the tool stays mute after the hole is
        filled, which the user reads as broken. The round-trip test only ever measures a lag of
        essentially zero against a lag of eight bars, so anything between one and eight bars passes
        it. Here the boundary itself is exercised: at exactly one trigger bar behind, the engine is
        caught up; at two, it is not.
        """
        now = {"ms": T0 + 600 * TF_1M.ms}
        monkeypatch.setattr(live_module.time, "time", lambda: now["ms"] / 1000.0)
        e = self._live(now)
        last = e.state.health.last_closed_ms

        e.state.health.mode = Mode.CATCH_UP
        now["ms"] = last + 2 * TF_15M.ms
        h = e.update_health(connected=True, reconnects=1, healed=0, silent_seconds=0.0)
        assert h.lag_bars == pytest.approx(2.0), f"the fixture is not two bars behind: {h.lag_bars}"
        assert e.state.health.mode is Mode.CATCH_UP and not e.emitting, (
            "the engine declared itself caught up two trigger bars behind the market: every zone "
            "it now publishes is priced off a bar that closed 30 minutes ago"
        )

        now["ms"] = last + TF_15M.ms
        h = e.update_health(connected=True, reconnects=1, healed=0, silent_seconds=0.0)
        assert h.lag_bars == pytest.approx(1.0), f"the fixture is not one bar behind: {h.lag_bars}"
        assert e.state.health.mode is Mode.LIVE and e.emitting, (
            "one trigger bar of lag is the freshest a bar-closing engine can ever be, and it was "
            "still refused: the engine never leaves CATCH_UP and stays mute for the session"
        )


class TestTheDataHealthBadge:
    #: 5½ days of minutes, and two of the 15m periods in them never arrive at all: every one of
    #: their fifteen source minutes is missing, so no trigger bar is built for them and the ring
    #: carries a real hole in its timestamps rather than a flag someone set. Ages, counted back
    #: from the last trigger bar, land at 298 and 119 — both outside the last 50 bars and both
    #: well inside 400, which is what makes the badge's reach observable at all.
    N_MINUTES = 8_000
    OUTAGE_PERIODS = (232, 412)

    def _engine_with_two_old_outages(self) -> LiveEngine:
        drop = {i for p in self.OUTAGE_PERIODS for i in range(15 * p, 15 * p + 15)}
        # 1m + 15m only: the 1h ring plays no part here and resampling it is pure cost.
        e = LiveEngine(SYMBOL, ["1m", "15m"], ring_capacity=8192, trigger_tf="15m")
        e.warmup(make_bars(self.N_MINUTES, tf=TF_1M, start_ms=T0, seed=11, drop=drop))
        return e

    def test_an_outage_older_than_fifty_trigger_bars_is_still_counted_on_the_badge(
        self, monkeypatch
    ):
        """The gap counter has to reach as far back as the ring the analysis actually reads.

        `gaps_in_window` is the only number on the badge that describes the SHAPE of the history,
        as opposed to how fresh its last bar is, and `web/app.js` paints it as "n gaps" next to a
        green dot. The bars an outage left behind do not age out of the analysis: a hole four days
        back is still inside the ring that feeds the ATR, the ZigZag threshold and therefore every
        entry zone drawn today. If the counter's window is shorter than the hole is old, the badge
        goes back to reporting zero gaps while the series behind the chart still has one in it —
        and it is exactly the stale, half-forgotten outage that nobody is watching for any more.

        Both outages here are older than 50 trigger bars and younger than 400, so the count is a
        statement about how deep the health line looks, not about the number it looks with: at 400
        or at 500 the badge reports both, at 50 it reports neither.
        """
        e = self._engine_with_two_old_outages()
        ring = e.state.ring(TF_15M)
        assert len(ring) > 500, (
            f"the ring holds {len(ring)} trigger bars: with 500 or fewer, every window length "
            "collapses onto len(ring) and this test cannot tell them apart"
        )
        assert ring.window(50).n_gaps == 0, (
            "the last 50 trigger bars are supposed to be unbroken — that is the whole premise: "
            "a badge that only looks 50 bars back would see a perfectly clean series here"
        )

        # The clock sits one minute past the last bar the engine has, so the lag is a fraction of a
        # trigger bar and the badge under test is the one a LIVE user is looking at.
        now_ms = (e.state.health.last_closed_ms or 0) + TF_1M.ms
        monkeypatch.setattr(live_module.time, "time", lambda: now_ms / 1000.0)
        h = e.update_health(connected=True, reconnects=2, healed=0, silent_seconds=0.0)

        assert e.state.health.mode is Mode.LIVE, (
            f"precondition: the badge should be LIVE, it reads {e.state.health.mode}"
        )
        assert h.gaps_in_window == len(self.OUTAGE_PERIODS), (
            f"{len(self.OUTAGE_PERIODS)} trigger bars are missing from the ring the engine is "
            f"analysing and the badge reports {h.gaps_in_window} gap(s): the health line is not "
            "looking back as far as the data it vouches for, so an outage disappears from the "
            "interface while its hole is still shaping every ATR and every entry zone"
        )
        assert h.as_dict()["gaps_in_window"] == len(self.OUTAGE_PERIODS), (
            "the count reaches `Health.as_dict`, which is what the badge is painted from"
        )


class TestTheDecisionCard:
    def test_the_verdict_never_goes_above_watch_while_the_maturity_is_prior(self, bars_20d):
        """PRIOR means not one trade of ours has resolved yet.

        The expectancies behind the card are a hand-written expert table; marking anything
        ACTIONABLE on that basis tells the user to risk money on a number nobody has measured.
        `Decision`'s constructor refuses the combination structurally, but that is the last line of
        defence and it only fires if a `Decision` is ever built: `decide()` returns a plain dict
        straight to the interface, so this is the line that actually holds.
        """
        e = LiveEngine(SYMBOL, list(TIMEFRAMES), 8192, TRIGGER_TF)
        in_zone = 0
        for b in bars_20d:
            for htf in e.on_bar_1m(b):
                if htf.tf is not TF_1H:
                    continue
                d = e.decide(TRIGGER_TF, htf.close)
                assert d["maturity"] == int(MaturityLevel.PRIOR), (
                    f"maturity {d['maturity']} at {htf.open_time_ms}: nothing has been validated, "
                    "the card cannot claim a level above PRIOR"
                )
                assert d["verdict"] != Verdict.ACTIONABLE.value, (
                    f"decide() returned ACTIONABLE at {htf.open_time_ms} while the maturity is "
                    "PRIOR: the user is being told to take a trade whose expectancy comes from a "
                    "hand-written table"
                )
                in_zone += sum(1 for h in d["hypotheses"] if h["viable"] and h["in_zone"])

        assert in_zone > 0, (
            "not one hypothesis was ever viable AND inside its entry zone over 20 days, so the "
            "cap above was never actually exercised: this test proves nothing as it stands"
        )

    def test_every_pivot_it_counts_is_an_extreme_of_the_bar_it_is_stamped_on(self, replay_20d):
        """The wire between the resampler and the detector is tested by nothing at either end.

        The resampler is checked against pandas, bar for bar; the detector is checked directly with
        bars the test hands it. In between sits one call — `det.update(ts, high, low, close)` — and
        its four arguments are three floats of the same magnitude and a timestamp. Swap high and
        low there and every swing high is located at the bar's LOW; shift the timestamp by one
        period and every pivot is attributed to its neighbour. Both keep producing a perfectly
        well-formed count, and every leg length, Elliott ratio, entry zone and invalidation the
        user is shown is then measured off the wrong number. Nothing raises, and no comparison
        against the resampler can see it, because the resampled bars are correct.

        A swing high is the HIGH of its bar and a swing low is the LOW of it (`on_close=False`).
        That is the whole claim, and it ties price and timestamp together, so it holds the
        arguments in place on both counts.
        """
        e, _, _ = replay_20d
        ring = e.state.rings[TRIGGER_TF]
        w = ring.window(len(ring))
        at = {int(ts): i for i, ts in enumerate(w.ts)}
        pivots = e.state.detectors[TRIGGER_TF].store.as_of(int(w.ts[-1]))
        assert len(pivots) >= 10, (
            f"only {len(pivots)} confirmed pivots over 20 days: too few for this to mean anything"
        )
        kinds = {p.kind for p in pivots}
        assert kinds == {PivotKind.HIGH, PivotKind.LOW}, (
            f"only {kinds} appear, so one of the two branches is never checked here"
        )

        for p in pivots:
            assert p.ts_ms in at, (
                f"a pivot is stamped at {p.ts_ms}, which is not the open time of any bar in the "
                f"{TRIGGER_TF} ring: the detector is being fed a timestamp the engine never stored"
            )
            i = at[p.ts_ms]
            side = "high" if p.kind is PivotKind.HIGH else "low"
            want = float(w.high[i]) if p.kind is PivotKind.HIGH else float(w.low[i])
            assert p.price == want, (
                f"the {p.kind.name} pivot at {p.ts_ms} is priced at {p.price}, but that bar's "
                f"{side} is {want} (open {w.open[i]}, high {w.high[i]}, low {w.low[i]}, close "
                f"{w.close[i]}). The detector is not being handed this bar's extremes: every leg "
                "measured from this pivot describes a price the market never traded at"
            )


class TestTheTripleBarrier:
    """The labels are what everything downstream learns from, and they fail upward.

    A label that says «won» where the trade lost does not raise: it produces a prettier hit rate.
    Nothing later in the pipeline can detect it, because every expectancy it is compared against
    was computed from the same labels.
    """

    def test_a_bar_that_touches_both_barriers_is_a_loss_and_says_it_was_ambiguous(self):
        """Inside one bar there is no way to know whether the target or the stop came first.

        Resolving the coin-flip as a win is how a backtest gets a hit rate nobody can reproduce
        with real money. It is resolved pessimistically, and — just as important — the ambiguity is
        RECORDED, because the rate of it is the alarm that says the barriers are too close together
        and the labels have stopped describing real trades.
        """
        out = resolve_triple_barrier(
            100.0, 95.0, 110.0,
            np.array([112.0]), np.array([94.0]), np.array([100.0]),
            max_bars=5,
        )
        assert out.barrier == "sl", (
            f"a bar that touched target AND stop was resolved as '{out.barrier}': with no way to "
            "know which came first, the optimistic reading inflates every statistic downstream"
        )
        assert out.r == -1.0, f"an unresolvable bar must cost a full R, it booked {out.r}"
        assert out.ambiguous is True, (
            "the tie was resolved pessimistically but not flagged: the ambiguity rate is what "
            "tells us the labels have stopped describing real trades, and it would read 0%"
        )

    @pytest.mark.parametrize(
        ("fine_high", "fine_low", "barrier", "r"),
        [
            ([111.0, 96.0], [99.0, 94.0], "tp", 2.0),   # the target came first
            ([101.0, 112.0], [94.0, 99.0], "sl", -1.0),  # the stop came first
        ],
    )
    def test_the_1m_series_breaks_the_tie_and_the_label_stops_being_ambiguous(
        self, fine_high, fine_low, barrier, r
    ):
        """When the minute bars inside the hour DO know the order, the coin-flip is not needed.

        This is the reason 1m is the single stored series: dropping down to it turns a pessimistic
        guess into a fact. If the fine data stops being consulted, every one of these becomes a
        loss and the strategy is measured as worse than it is — and the ambiguity rate stops being
        a signal about the barriers and becomes a constant.
        """
        out = resolve_triple_barrier(
            100.0, 95.0, 110.0,
            np.array([112.0]), np.array([94.0]), np.array([100.0]),
            max_bars=5,
            fine_highs=[np.array(fine_high)], fine_lows=[np.array(fine_low)],
        )
        assert out.barrier == barrier, (
            f"the 1m bars say the {barrier} came first; the label reads '{out.barrier}'"
        )
        assert out.r == pytest.approx(r), f"expected {r}R, got {out.r}R"
        assert out.bars == 1, (
            f"the trade resolved inside the first bar, the label says it was held {out.bars}: "
            "holding time is what the vertical barrier and every duration statistic are built on"
        )
        assert out.ambiguous is False, (
            "the minute bars resolved the order, so this label is a fact and must not be counted "
            "towards the ambiguity rate that decides whether the labels are usable at all"
        )

    def test_a_minute_that_is_itself_ambiguous_stays_pessimistic(self):
        """1m is as fine as the data goes. When even a minute touched both, we still do not know.

        Guessing at that point is the same optimism as guessing at the hour, just better hidden.
        """
        out = resolve_triple_barrier(
            100.0, 95.0, 110.0,
            np.array([112.0]), np.array([94.0]), np.array([100.0]),
            max_bars=5,
            fine_highs=[np.array([112.0])], fine_lows=[np.array([94.0])],
        )
        assert (out.barrier, out.r, out.ambiguous) == ("sl", -1.0, True), (
            f"a minute bar that touched both barriers was resolved as {out.barrier} r={out.r} "
            f"ambiguous={out.ambiguous}: the order is still unknown, so the label must stay "
            "pessimistic and stay flagged"
        )

    @pytest.mark.parametrize(
        ("highs", "lows", "closes", "barrier", "held"),
        [
            ([103.0, 111.0], [96.0, 99.0], [102.0, 110.0], "tp", 2),
            ([103.0, 101.0], [96.0, 94.0], [102.0, 95.0], "sl", 2),
            ([103.0, 104.0], [96.0, 97.0], [102.0, 103.0], "vertical", 2),
        ],
    )
    def test_a_short_is_the_exact_mirror_image_of_the_long(
        self, highs, lows, closes, barrier, held
    ):
        """The interface only shows longs; the engine labels BOTH directions from day one.

        So half the evidence the project will ever accumulate is short evidence, and it is checked
        by nothing that looks at the screen. Every comparison in the short branch is a separate
        `if long else`, and each one is an opportunity to compare against the wrong side of the
        bar: a short's target is BELOW the entry and its stop ABOVE it. Reflecting the whole series
        around the entry price has to produce the identical outcome, R for R and excursion for
        excursion — if it does not, the short half of the book is measuring something else.
        """
        entry = 100.0
        mirror = 2 * entry                        # p -> 200 - p reflects price around the entry

        long_out = resolve_triple_barrier(
            entry, 95.0, 110.0,
            np.array(highs), np.array(lows), np.array(closes),
            max_bars=2, long=True,
        )
        short_out = resolve_triple_barrier(
            entry, mirror - 95.0, mirror - 110.0,
            np.array([mirror - x for x in lows]),     # the mirror of a low is a high
            np.array([mirror - x for x in highs]),
            np.array([mirror - x for x in closes]),
            max_bars=2, long=False,
        )
        assert (long_out.barrier, long_out.bars) == (barrier, held), (
            f"the fixture no longer exercises the case it names: expected {barrier} after {held} "
            f"bars, got {long_out.barrier} after {long_out.bars}"
        )
        assert short_out == long_out, (
            f"the mirrored short resolved differently from the long: {short_out} != {long_out}. "
            "One of the direction branches is comparing against the wrong side of the bar"
        )

    def test_the_time_barrier_closes_the_trade_at_market(self):
        """The third barrier is the one people forget, and it decides roughly half the trades.

        A trade that neither wins nor loses within its holding window is CLOSED at market — that is
        what the user would actually do. If the time barrier stops being applied, positions are
        held until the data runs out: the R booked belongs to a trade nobody would have sat
        through, and the tail of the distribution the sizing is built on becomes fiction.
        """
        highs = np.full(50, 101.0)
        lows = np.full(50, 99.0)
        closes = np.full(50, 100.5)
        closes[9] = 102.0                       # the close at the moment the window expires
        out = resolve_triple_barrier(100.0, 95.0, 110.0, highs, lows, closes, max_bars=10)

        assert out.barrier == "vertical", f"nothing was touched, expected 'vertical': {out}"
        assert out.bars == 10, (
            f"the trade was held for {out.bars} bars with a 10-bar window: the time barrier is not "
            "being applied and the label describes a trade that was never taken"
        )
        assert out.r == pytest.approx((102.0 - 100.0) / 5.0), (
            f"the exit must be booked at the close of the last bar held, got {out.r}R"
        )

    def test_a_stop_out_costs_exactly_one_r(self):
        """R is DEFINED as the distance from entry to stop, so a stop-out is -1R by definition.

        Book it as anything else and every number downstream is quietly rescaled: expectancy,
        profit factor and the required hit rate all move, and none of them looks wrong.
        """
        out = resolve_triple_barrier(
            100.0, 95.0, 110.0,
            np.array([101.0, 99.0]), np.array([98.0, 94.0]), np.array([100.0, 95.0]),
            max_bars=5,
        )
        assert out.barrier == "sl", f"the second bar traded through the stop: {out}"
        assert out.r == -1.0, f"a stop-out must cost exactly -1R, it booked {out.r}R"
        assert out.bars == 2, f"the stop was hit on the 2nd bar, the label says {out.bars}"

    def test_the_two_excursions_measure_opposite_directions(self):
        """MAE is what the trade cost you before it worked; MFE is what it offered before it did.

        They are the input to any future rule about tightening stops or taking partials. Swapped,
        such a rule would be fitted to the best case while believing it was the worst — the single
        most dangerous way to be wrong about risk.
        """
        out = resolve_triple_barrier(
            100.0, 95.0, 110.0,
            np.array([101.0, 108.0]), np.array([97.0, 100.0]), np.array([100.0, 107.0]),
            max_bars=2,
        )
        assert out.mae_r == pytest.approx(0.6), (
            f"price went 3 points against a 5-point risk: MAE is 0.6R, the label says {out.mae_r}"
        )
        assert out.mfe_r == pytest.approx(1.6), (
            f"price went 8 points in favour of a 5-point risk: MFE is 1.6R, the label says "
            f"{out.mfe_r}"
        )

    def test_the_excursions_are_not_interchangeable_on_a_resolved_trade(self):
        """The test above only ever reaches the VERTICAL exit, so it pins the excursions on one of
        the three return paths and leaves the other two free.

        `resolve_triple_barrier` builds an `Outcome` at four separate places, each passing `mae`
        and `mfe` positionally into adjacent fields of the same type. Swapping that pair on the
        `tp` or `sl` return is invisible to every other assertion in this file — and a stop-tighten
        rule fitted to the swapped pair would be fitted to the best case while believing it was
        the worst.

        Both cases below are deliberately asymmetric (MAE != MFE), so the swap cannot hide.
        """
        # --- resolved at the TARGET, after first going against the trade -------------------
        tp = resolve_triple_barrier(
            100.0, 95.0, 110.0,                       # risk = 5
            np.array([101.0, 112.0]), np.array([97.0, 100.0]), np.array([100.0, 111.0]),
            max_bars=5,
        )
        assert tp.barrier == "tp", f"bar 2 traded through the target: {tp}"
        assert tp.mae_r == pytest.approx(0.6), (
            f"the low of 97 is 3 points against a 5-point risk: MAE is 0.6R, not {tp.mae_r}"
        )
        assert tp.mfe_r == pytest.approx(2.4), (
            f"the high of 112 is 12 points in favour of a 5-point risk: MFE is 2.4R, not "
            f"{tp.mfe_r}"
        )

        # --- resolved at the STOP, after first going in favour -----------------------------
        sl = resolve_triple_barrier(
            100.0, 95.0, 120.0,                       # risk = 5, target far away
            np.array([104.0, 101.0]), np.array([98.0, 94.0]), np.array([103.0, 95.0]),
            max_bars=5,
        )
        assert sl.barrier == "sl", f"bar 2 traded through the stop: {sl}"
        assert sl.mae_r == pytest.approx(1.2), (
            f"the low of 94 is 6 points against a 5-point risk: MAE is 1.2R, not {sl.mae_r}"
        )
        assert sl.mfe_r == pytest.approx(0.8), (
            f"the high of 104 is 4 points in favour of a 5-point risk: MFE is 0.8R, not "
            f"{sl.mfe_r}"
        )

    @pytest.mark.parametrize("long", [True, False])
    def test_touching_the_barrier_is_enough(self, long):
        """Both barriers are prices an order actually sits at, so bars land on them to the tick.

        A target is a published fib level rounded to the venue's tick and a stop is the Elliott
        invalidation: they are not two independent floats whose equality is a measure-zero event,
        they are the exact prices everyone's resting orders cluster on. Make either comparison
        strict and a trade filled at its target is not booked as a win — it falls through to
        whatever the LATER bars do, so the label is not merely missing, it is a different trade.
        The error is systematic, it lands on the trades that touched their level and turned, and
        both directions carry a separate comparison.
        """
        entry, risk = 100.0, 5.0
        stop = entry - risk if long else entry + risk
        target = entry + 2 * risk if long else entry - 2 * risk
        near = entry + 1.0 if long else entry - 1.0

        tp = resolve_triple_barrier(
            entry, stop, target,
            np.array([target if long else near]), np.array([near if long else target]),
            np.array([entry]), max_bars=5, long=long,
        )
        assert (tp.barrier, tp.bars) == ("tp", 1), (
            f"a bar whose extreme reached the target exactly was booked as {tp.barrier} after "
            f"{tp.bars} bars: a fill at the level is a fill, and this trade is now labelled by "
            "bars that came after it closed"
        )
        assert tp.r == pytest.approx(2.0), f"a 2R target booked {tp.r}R"

        sl = resolve_triple_barrier(
            entry, stop, target,
            np.array([near if long else stop]), np.array([stop if long else near]),
            np.array([entry]), max_bars=5, long=long,
        )
        assert (sl.barrier, sl.bars) == ("sl", 1), (
            f"a bar whose extreme reached the stop exactly was booked as {sl.barrier} after "
            f"{sl.bars} bars: the trade was stopped out and the label says it was still open"
        )
        assert sl.r == -1.0, f"a stop-out costs exactly -1R, it booked {sl.r}R"

    def test_a_trade_that_never_traded_against_the_entry_has_no_adverse_excursion(self):
        """Every MAE assertion in this file uses a bar that DID go against the trade.

        So the sign of the adverse term is never exercised, and taking its absolute value — the
        one-character edit that makes both excursions "how far it moved" — passes all of them. What
        comes out then is a floor under MAE equal to the favourable move, on precisely the trades
        that never suffered one. A stop-tightening rule reading that table sees every winner
        drawing down before it worked, and tightens stops that were never threatened.
        """
        out = resolve_triple_barrier(
            100.0, 95.0, 110.0,                       # risk = 5
            np.array([104.0, 105.0]), np.array([101.0, 102.0]), np.array([103.0, 104.0]),
            max_bars=2,
        )
        assert out.barrier == "vertical", f"nothing was touched, expected 'vertical': {out}"
        assert out.mae_r == 0.0, (
            f"the price never traded below the entry, so the adverse excursion is 0R, not "
            f"{out.mae_r}R: the sign of the adverse term is being thrown away"
        )
        assert out.mfe_r == pytest.approx(1.0), (
            f"the high of 105 is 5 points in favour of a 5-point risk: MFE is 1.0R, not {out.mfe_r}"
        )

    @pytest.mark.parametrize(
        ("fine_high", "fine_low", "barrier", "ambiguous"),
        [
            ([111.0, 96.0], [99.0, 94.0], "tp", False),    # the minutes say the target came first
            ([101.0, 112.0], [94.0, 99.0], "sl", False),   # the minutes say the stop came first
            (None, None, "sl", True),                      # no minutes: pessimistic and flagged
        ],
    )
    def test_the_excursions_survive_the_tie_break_too(
        self, fine_high, fine_low, barrier, ambiguous
    ):
        """The excursion test next door reaches the plain `tp` and `sl` returns and stops there.

        `resolve_triple_barrier` builds an `Outcome` at five places, and the three inside the
        both-barriers-in-one-bar branch — tie broken to the target, tie broken to the stop, and no
        1m data at all — are reached by nothing that looks at `mae_r` or `mfe_r`. They pass `mae`
        and `mfe` positionally into adjacent float fields, so the swap is invisible in review and
        invisible here. Those are also the most violent bars in the sample, the ones where a
        stop-tighten rule would be fitted to the widest excursions it will ever see — while
        believing the best case was the worst.

        The single bar below is identical in all three cases: it went 6 points against a 5-point
        risk and 12 in favour, so 1.2R and 2.4R can never be mistaken for each other.
        """
        kw = {}
        if fine_high is not None:
            kw = {"fine_highs": [np.array(fine_high)], "fine_lows": [np.array(fine_low)]}
        out = resolve_triple_barrier(
            100.0, 95.0, 110.0,                       # risk = 5
            np.array([112.0]), np.array([94.0]), np.array([100.0]),
            max_bars=5, **kw,
        )
        assert (out.barrier, out.ambiguous) == (barrier, ambiguous), (
            f"the fixture no longer reaches the branch it names: expected {barrier} "
            f"(ambiguous={ambiguous}), got {out.barrier} (ambiguous={out.ambiguous})"
        )
        assert out.mae_r == pytest.approx(1.2), (
            f"the low of 94 is 6 points against a 5-point risk: MAE is 1.2R, not {out.mae_r}R"
        )
        assert out.mfe_r == pytest.approx(2.4), (
            f"the high of 112 is 12 points in favour of a 5-point risk: MFE is 2.4R, not "
            f"{out.mfe_r}R"
        )

    def test_it_refuses_a_trade_whose_stop_is_its_entry(self):
        """Every label is a ratio to |entry - stop|. A zero there poisons silently.

        The division would yield inf or nan, those flow into the expectancy, and the backtest comes
        back with a number that is not wrong so much as meaningless. Refusing loudly is the only
        way anyone finds out.
        """
        with pytest.raises(ValueError, match="zero risk"):
            resolve_triple_barrier(
                100.0, 100.0, 110.0,
                np.array([101.0]), np.array([99.0]), np.array([100.0]),
                max_bars=5,
            )


class TestTheBacktest:
    """What the backtest measures has to be the trade the user would have taken. Nothing else."""

    def test_a_trade_is_resolved_only_with_bars_that_come_after_the_decision(
        self, bars_20d, backtest_20d
    ):
        """The decision is causal; the RESOLUTION is allowed to look forward — but not that far.

        Entry is the close of the bar that produced the signal, so that bar's own high and low
        happened BEFORE the entry existed. Resolving against them books targets and stops at prices
        the trade could never have touched, and the error fails upward: on a 4h bar whose range
        spans both barriers, roughly half those phantom fills are wins. Here the outcome of every
        signal is recomputed from the offline resampler — a different code path from the engine's —
        using strictly the bars that came after the decision.
        """
        df = pd.DataFrame(
            {"open": [b.open for b in bars_20d], "high": [b.high for b in bars_20d],
             "low": [b.low for b in bars_20d], "close": [b.close for b in bars_20d],
             "volume": [b.volume for b in bars_20d]},
            index=pd.Index([b.open_time_ms for b in bars_20d], name="open_time_ms"),
        )
        offline = resample_from_1m(df, BY_NAME[TRIGGER_TF])
        high = offline["high"].to_numpy()
        low = offline["low"].to_numpy()
        close = offline["close"].to_numpy()
        at = {ts: i for i, ts in enumerate(offline.index)}

        resolved = [s for s in backtest_20d.signals if s.outcome is not None]
        assert len(resolved) >= 20, (
            f"only {len(resolved)} resolved signals: too few for this comparison to mean anything"
        )
        for s in resolved:
            i = at[s.ts_ms]
            expected = resolve_triple_barrier(
                s.entry, s.stop, s.target, high[i + 1:], low[i + 1:], close[i + 1:],
                max_bars=MAX_BARS_HOLD, long=s.direction is Direction.LONG,
            )
            assert (s.outcome.barrier, s.outcome.bars) == (expected.barrier, expected.bars), (
                f"the signal at {s.ts_ms} was booked as {s.outcome.barrier} after "
                f"{s.outcome.bars} bars; resolved against the bars that came after the decision it "
                f"is {expected.barrier} after {expected.bars}. The entry bar is being resolved "
                "against itself"
            )
            assert s.outcome.r == pytest.approx(expected.r), (
                f"the signal at {s.ts_ms} booked {s.outcome.r}R, the bars after it give "
                f"{expected.r}R"
            )

    def test_every_signal_is_a_trade_you_could_take(self, backtest_20d):
        """Stop on the losing side, target on the winning side, and a risk that is not zero.

        A signal with the stop on the wrong side of the entry is not a bad trade, it is not a trade
        at all: it is stopped out the instant it opens, and the label it produces is noise mixed
        into the statistics of every other signal that shares its archetype.
        """
        assert backtest_20d.signals, "no signals: this test is checking an empty list"
        longs = sum(1 for s in backtest_20d.signals if s.direction is Direction.LONG)
        assert 0 < longs < len(backtest_20d.signals), (
            "both directions have to appear, or the short branch is never checked here"
        )
        for s in backtest_20d.signals:
            if s.direction is Direction.LONG:
                assert s.stop < s.entry < s.target, (
                    f"long signal at {s.ts_ms}: stop {s.stop}, entry {s.entry}, target {s.target} "
                    "— a long's stop goes below the entry and its target above it"
                )
            else:
                assert s.target < s.entry < s.stop, (
                    f"short signal at {s.ts_ms}: stop {s.stop}, entry {s.entry}, target "
                    f"{s.target} — a short's stop goes above the entry and its target below it"
                )

    def test_a_signal_books_the_trade_the_card_actually_offered(self, backtest_20d, replay_20d):
        """★ The geometry of the label has to be the geometry of the plan on screen.

        The barrier module's opening paragraph is about this and calls it subtle and lethal: label
        a trade with a stop the interface never showed — a flat ATR multiple where the card
        published the Elliott invalidation, or T1 where the card quoted the R:R to T2 — and every
        probability the user is given describes a DIFFERENT trade from the one they take. Nothing
        downstream can find it, because every expectancy it is compared against came out of the
        same labels, and the archetype id still matches.

        The neighbouring resolution test cannot see any of it either: it recomputes each outcome
        from the signal's own recorded stop and target, so a signal is checked against itself. This
        is the other side of that — the recorded numbers against the card that was published on
        that bar, replayed here down a separate pass of the engine.
        """
        assert backtest_20d.signals, "no signals: this test is checking an empty list"
        _, cards, closes = replay_20d
        for s in backtest_20d.signals:
            assert s.ts_ms in cards, (
                f"the signal at {s.ts_ms} is stamped at an instant no {TRIGGER_TF} bar closed on: "
                "it is not pointing at the decision that produced it"
            )
            assert s.entry == closes[s.ts_ms], (
                f"the signal at {s.ts_ms} was entered at {s.entry}; the bar that carried that "
                f"decision closed at {closes[s.ts_ms]}. The fill is at a price the decision never "
                "saw"
            )
            arch = s.archetype.rsplit("_", 1)[0]
            want_dir = "LONG" if s.direction is Direction.LONG else "SHORT"
            published = [h for h in cards[s.ts_ms]["hypotheses"]
                         if h["viable"] and h["archetype"] == arch
                         and h["direction"] == want_dir]
            assert published, (
                f"the signal at {s.ts_ms} claims a viable {want_dir} {arch}, and the card for that "
                "bar published no such hypothesis at all"
            )
            recorded = (s.stop, s.target, s.rr, s.score, s.cost_r, s.size_factor, s.invalidation)
            offered = [(h["stop"], h["targets"][1], h["rr_t2"], h["score"], h["cost_r"],
                        h["size_factor"], h["invalidation_price"]) for h in published]
            assert recorded in offered, (
                f"the signal at {s.ts_ms} booked (stop, target, rr, score, cost_r, size, "
                f"invalidation) = {recorded}, and no {want_dir} {arch} on that card offered those "
                f"numbers — the card offered {offered}. The trade being measured is not the trade "
                "that was shown"
            )

    def test_the_booked_R_R_is_the_zone_s_and_not_the_fill_s_and_that_is_recorded_here(
        self, backtest_20d
    ):
        """CHARACTERISATION, not approval. This pins a known inconsistency so it cannot move
        without someone deciding that it should.

        The test above proves the signal faithfully COPIES the card's numbers. It does not — and
        by construction cannot — say those numbers are consistent with each other, and they are
        not. `build_plan` computes `rr_t2` and `cost_r` from `entry = (zone_lo + zone_hi) / 2`, the
        middle of the published entry zone; `run_backtest` fills at `htf.close`. Measured on this
        fixture: 131 of 132 signals carry an R:R describing a different entry price from the one
        their outcome is booked at, the worst by 2.29R.

        So `expectancy_r` is measured off the fill and `rr` is quoted off the zone, and the two
        travel side by side in the same report. That is a source decision to make — book at the
        zone midpoint, or recompute the plan's arithmetic at the fill — and not one a test should
        make quietly. Until it is made, the discrepancy is asserted rather than left as folklore:
        anyone who resolves it will see this test go red and have to say which way they went.

        See docs/TEST_COVERAGE.md, "Known inconsistencies the suite pins rather than fixes".
        """
        resolved = [s for s in backtest_20d.signals if s.outcome]
        assert len(resolved) >= 20, "too few signals for this to characterise anything"

        geometric = [abs(s.target - s.entry) / abs(s.entry - s.stop) for s in resolved]
        drift = [abs(g - s.rr) for g, s in zip(geometric, resolved, strict=True)]
        disagreeing = sum(1 for d in drift if d > 0.01)

        assert disagreeing > len(resolved) // 2, (
            f"only {disagreeing} of {len(resolved)} signals show the zone/fill R:R discrepancy. "
            "If it has been fixed, that is good news and this test has to go: delete it and move "
            "the entry in docs/TEST_COVERAGE.md from «known inconsistencies» to «verified». If it "
            "has merely moved, find out where"
        )
        # Every signal still records the R:R its own card published, whatever entry that was
        # computed at. That half IS a real invariant and it is what the test above rests on.
        assert all(s.rr > 0 for s in resolved), "an R:R must at least be positive"

    def test_the_fees_are_subtracted_and_paying_more_can_only_make_it_worse(self, backtest_20d):
        """A backtest that reports gross R is a brochure. The user pays the taker fee twice.

        `cost_r` is that fee expressed in R, and it is the difference between an edge and a story:
        at a 0.5% stop it eats 0.30R, which is most of what any honest expectancy is worth. The
        direction of the sign is the whole point — turn it around and the tighter the stop, the
        better the strategy looks.
        """
        priced = [s for s in backtest_20d.signals if s.outcome and s.cost_r > 0]
        assert priced, "no signal carried a cost: the fee arithmetic is not being exercised"
        for s in priced:
            assert s.net_r == pytest.approx(s.outcome.r - s.cost_r), (
                f"the signal at {s.ts_ms} gross {s.outcome.r}R with {s.cost_r}R of fees reports a "
                f"net of {s.net_r}R: the fee is not being taken off"
            )

        cheap = backtest_20d.stats(cost_mult=1.0)
        dear = backtest_20d.stats(cost_mult=3.0)
        assert dear["expectancy_r"] < cheap["expectancy_r"], (
            f"tripling the fees left the expectancy at {dear['expectancy_r']} against "
            f"{cheap['expectancy_r']}: the cost multiplier is not reaching the statistics, so a "
            "fee-sensitivity check would always come back reassuring"
        )
        per_trade = cheap["expectancy_r"] - dear["expectancy_r"]
        assert per_trade == pytest.approx(
            2 * float(np.mean([s.cost_r for s in backtest_20d.signals if s.outcome]))
        ), "tripling the fee has to subtract exactly two more fees from the average trade"

    def test_the_null_arm_is_a_different_trade_from_the_signal(self, backtest_20d):
        """The null arm is the control, and a control that copies the treatment is not one.

        Every signal spawns a twin at a random nearby instant with the same geometry in R. That
        pairing is what makes the result interpretable long before the absolute hit rate is: it
        answers «does the count add anything?» instead of «did BTC go up?». If the twin were
        entered at the same instant it would BE the signal, `edge_vs_null` would be exactly zero
        by construction, and the number that decides whether the whole Elliott premise is worth
        anything would be a tautology reading "no edge" for ever.
        """
        twinned = [s for s in backtest_20d.signals if s.outcome and s.null_outcome]
        assert len(twinned) >= 20, f"only {len(twinned)} twinned signals: too few to compare"

        same = sum(1 for s in twinned if s.null_outcome == s.outcome)
        assert same < len(twinned) // 2, (
            f"{same} of {len(twinned)} null twins resolved identically to their signal: the "
            "control is entering at the same instant as the trade it is supposed to control for"
        )
        st = backtest_20d.stats()
        assert st["edge_vs_null"] is not None and st["edge_vs_null"] != 0.0, (
            f"edge_vs_null came back {st['edge_vs_null']}: an exact zero means the null arm is the "
            "signal itself"
        )

    def test_the_control_is_held_for_the_same_window_as_the_trade_it_controls(self, backtest_20d):
        """The twin copies the signal's stop and target in R. The third barrier is time.

        Its window is the one parameter of the control that is not derived from the signal's
        geometry, so it is the one that can silently drift — and holding time is not a detail of a
        triple-barrier trade, it is a third of its definition. Give the control twice as long and
        it gets twice as many chances to reach the same target before its time runs out, so
        `edge_vs_null` stops being a paired difference and becomes a comparison between a 12-bar
        trade and a 24-bar one. The direction of that bias is towards understating the edge, which
        is the direction nobody investigates.
        """
        twinned = [s for s in backtest_20d.signals if s.null_outcome]
        assert len(twinned) >= 20, f"only {len(twinned)} twins: too few to check"
        at_the_limit = sum(1 for s in twinned if s.null_outcome.bars == MAX_BARS_HOLD)
        assert at_the_limit > 0, (
            "not one control ran to the end of its holding window, so a longer window would look "
            "identical here and this test proves nothing"
        )
        for s in twinned:
            # `1 <=` is NOT asserted here: `resolve_triple_barrier` returns `i + 1` on every
            # barrier path and `n >= 1` on the vertical one, so a lower bound of 1 cannot fail for
            # any implementation. It read as coverage and was a comment with an `assert` in front
            # of it. The upper bound is the real content.
            assert s.null_outcome.bars <= MAX_BARS_HOLD, (
                f"the control for the signal at {s.ts_ms} was held {s.null_outcome.bars} bars "
                f"against a {MAX_BARS_HOLD}-bar time barrier: it is not the same trade as the one "
                "it is being subtracted from"
            )

    def test_the_control_is_the_same_trade_taken_at_a_different_moment(
        self, bars_20d, backtest_20d
    ):
        """★ `edge_vs_null` is this project's headline number and the control arm it is measured
        against had almost no test at all.

        The twin's whole claim is «the same trade, at a nearby instant that the count did not
        pick». Everything in that sentence is geometry — same risk, same reward in R, same
        direction, entered LATER, resolved only from bars after its own entry — and none of it was
        observable, so four separate faults in those four lines survived the entire suite. Measured
        on this fixture: a twin whose target lands on the losing side turns `edge_vs_null` from
        +0.065 to −1.540, a twin allowed to enter on the signal's own bar flips its sign, and a
        twin allowed to resolve on its own entry bar nearly doubles the reported edge — the last
        being lookahead the signal arm is explicitly forbidden and the control arm was not, an
        asymmetry that flatters the strategy.

        The oracle is the same shape as the signal arm's: rebuild the outcome from the offline
        resampler, a different code path, using strictly the bars after the twin's own entry. It
        pins all four faults at once, and it pins them per signal rather than through a summary
        statistic that could absorb one bad twin in a hundred.
        """
        df = pd.DataFrame(
            {"open": [b.open for b in bars_20d], "high": [b.high for b in bars_20d],
             "low": [b.low for b in bars_20d], "close": [b.close for b in bars_20d],
             "volume": [b.volume for b in bars_20d]},
            index=pd.Index([b.open_time_ms for b in bars_20d], name="open_time_ms"),
        )
        offline = resample_from_1m(df, BY_NAME[TRIGGER_TF])
        high, low, close = (offline[c].to_numpy() for c in ("high", "low", "close"))
        at = {ts: i for i, ts in enumerate(offline.index)}

        twinned = [s for s in backtest_20d.signals if s.null_outcome]
        assert len(twinned) >= 20, f"only {len(twinned)} twins: too few to check"

        for s in twinned:
            is_long = s.direction is Direction.LONG
            sign = 1 if is_long else -1
            risk = abs(s.entry - s.stop)
            null_risk = abs(s.null_entry - s.null_stop)

            assert s.null_ts_ms > s.ts_ms, (
                f"the control for the signal at {s.ts_ms} entered at {s.null_ts_ms}, on or before "
                "the signal's own bar. A twin that overlaps the signal is partly the signal, so "
                "the difference between them stops being a control and starts being noise "
                "subtracted from itself"
            )
            assert null_risk == pytest.approx(risk, rel=1e-9), (
                f"the control risks {null_risk} against the signal's {risk}. The twin copies the "
                "risk in absolute terms; rescale it and the two arms are denominated in different "
                "Rs and their difference means nothing"
            )
            assert sign * (s.null_stop - s.null_entry) < 0 < sign * (s.null_target - s.null_entry), (
                f"a {s.direction.name} control entered at {s.null_entry} with its stop at "
                f"{s.null_stop} and its target at {s.null_target}: one of them is on the wrong "
                "side of the entry, so the twin books a profit for going the way it was betting "
                "against and the control arm beats the strategy for free"
            )
            assert abs(s.null_target - s.null_entry) / null_risk == pytest.approx(
                abs(s.target - s.entry) / risk, rel=1e-9), (
                "the control's reward-to-risk differs from the signal's, so the paired difference "
                "is comparing two different trades"
            )

            j = at[s.null_ts_ms]
            expected = resolve_triple_barrier(
                s.null_entry, s.null_stop, s.null_target,
                high[j + 1:], low[j + 1:], close[j + 1:],
                max_bars=MAX_BARS_HOLD, long=is_long,
            )
            assert (s.null_outcome.barrier, s.null_outcome.bars) == (
                expected.barrier, expected.bars), (
                f"the control entered at {s.null_ts_ms} was booked as {s.null_outcome.barrier} "
                f"after {s.null_outcome.bars} bars; resolved against the bars that came after its "
                f"own entry it is {expected.barrier} after {expected.bars}. The control arm is "
                "being resolved with its own entry bar — the exact lookahead the signal arm is "
                "forbidden, and it flatters the null, which understates the edge"
            )

    def test_one_count_does_not_become_twenty_trades(self, backtest_20d):
        """The debounce, and it is a statistics question rather than a tidiness one.

        The same wave count stays valid and in its zone for bar after bar. Taken on each of them it
        produces twenty near-identical trades out of ONE independent observation, and every measure
        of significance downstream treats them as twenty: the sample looks four times bigger than
        it is and the confidence interval shrinks around a number that never earned it. That is
        pseudo-replication, and it is how a backtest talks itself into a strategy.

        WHICH KEY the cooldown uses is a separate question, and this test could not see it. It
        grouped by `Signal.archetype`, which carries the direction suffix (`w2_long`), while the
        cooldown is keyed on the bare archetype (`w2`) — so a change from one to the other was
        structurally invisible here. It is not a small change: on this fixture it takes the run
        from 132 signals to 153 and turns `expectancy_r` from +0.043 to −0.009 and `edge_vs_null`
        from +0.065 to −0.066. Both readings are defensible, and that is exactly why the one in
        force has to be written down: a long and a short off the same count are either one
        observation of that count or two observations of the market, and the answer decides
        whether this strategy is profitable.

        Both groupings are asserted below. The bare-archetype one is the strictly stronger claim
        and it is the behaviour that ships.
        """
        tf_ms = BY_NAME[TRIGGER_TF].ms
        for key, label in ((lambda s: s.archetype, "archetype and direction"),
                           (lambda s: s.archetype.rsplit("_", 1)[0], "archetype alone")):
            last: dict[str, int] = {}
            for s in sorted(backtest_20d.signals, key=lambda s: s.ts_ms):
                k = key(s)
                previous = last.get(k)
                if previous is not None:
                    apart = (s.ts_ms - previous) // tf_ms
                    assert apart >= COOLDOWN_BARS, (
                        f"two {k!r} signals {apart} trigger bars apart with a cooldown of "
                        f"{COOLDOWN_BARS}, grouping by {label}: the same count is being counted "
                        "as several independent trades"
                    )
                last[k] = s.ts_ms
            assert len(last) > 1, (
                f"only one key under {label}; the debounce was barely exercised")

    def test_the_score_filter_actually_filters(self, bars_20d, backtest_20d):
        """`min_score` is the operator saying «only the counts that fit well».

        A filter that is read and then ignored is worse than no filter: the report comes back
        looking like the selective run that was asked for, and the trades it was meant to exclude
        are in there raising the sample size and lowering the quality of every archetype's stats.
        """
        assert backtest_20d.signals, "no unfiltered signals to compare against"
        strict = run_backtest(bars_20d, trigger_tf=TRIGGER_TF, timeframes=TIMEFRAMES,
                              max_bars_hold=MAX_BARS_HOLD, cooldown_bars=COOLDOWN_BARS,
                              min_score=1.01)      # the score is a weighted mean: it cannot pass 1
        assert not strict.signals, (
            f"a score floor no hypothesis can clear still let {len(strict.signals)} signals "
            "through: min_score is not being applied"
        )

    def test_the_same_seed_replays_the_same_backtest(self, bars_20d):
        """Two runs of the same code over the same bars have to agree, or nothing can be compared.

        The null arm draws its entry at random. Unseeded, the answer to «did this change help?»
        moves on its own between runs, and every comparison — a config change, a new rule, a
        refactor — is measuring the generator instead of the change.
        """
        kw = {"trigger_tf": TRIGGER_TF, "timeframes": TIMEFRAMES,
              "max_bars_hold": MAX_BARS_HOLD, "cooldown_bars": COOLDOWN_BARS}

        def fingerprint(r: BacktestResult) -> list[tuple]:
            return [(s.ts_ms, s.entry, s.stop, s.target, s.outcome, s.null_outcome)
                    for s in r.signals]

        a = run_backtest(bars_20d, seed=17, **kw)
        b = run_backtest(bars_20d, seed=17, **kw)
        assert fingerprint(a) == fingerprint(b), (
            "two runs with the same seed produced different backtests: the result is not "
            "reproducible and no comparison between two runs means anything"
        )

        c = run_backtest(bars_20d, seed=99, **kw)
        assert [s.outcome for s in a.signals] == [s.outcome for s in c.signals], (
            "changing the seed changed the SIGNALS: the trades themselves must not depend on the "
            "random draw, only their null twins do"
        )
        assert [s.null_outcome for s in a.signals] != [s.null_outcome for s in c.signals], (
            "changing the seed left the null arm untouched: the draw is not seeded from the "
            "argument, so `seed` is not actually controlling anything"
        )


class TestTheBacktestStatistics:
    """The report card. Every one of these numbers ends up in a decision about real money."""

    @staticmethod
    def _result(*rs: float, cost_r: float = 0.0, nulls: tuple[float, ...] = (),
                barriers: tuple[str, ...] = ()) -> BacktestResult:
        res = BacktestResult()
        for i, r in enumerate(rs):
            s = Signal(ts_ms=i, archetype="w2_long", direction=Direction.LONG, entry=100.0,
                       stop=95.0, target=110.0, rr=2.0, score=1.0, cost_r=cost_r,
                       size_factor=1.0, invalidation=94.0)
            s.outcome = Outcome(barriers[i] if barriers else ("tp" if r > 0 else "sl"),
                                r, 0.0, 0.0, 1)
            if nulls:
                s.null_outcome = Outcome("tp" if nulls[i] > 0 else "sl", nulls[i], 0.0, 0.0, 1)
            res.signals.append(s)
        return res

    def test_the_edge_is_what_the_signals_made_over_what_the_control_made(self):
        """`edge_vs_null` is the one number that answers «is the wave count worth anything?».

        It is a paired difference against a twin trade taken at a random nearby instant, and that
        pairing is the whole reason it becomes interpretable long before the absolute hit rate
        does. Which also means the subtraction has a direction, and nothing else on the card can
        contradict it: turn it around and a strategy that beats its control by 3R reports −3R and
        gets abandoned, while one that loses to a random entry reports a healthy edge and gets
        traded. The neighbouring null-arm test only asserts the figure is not exactly zero, which
        both orientations satisfy.
        """
        st = self._result(2.0, 2.0, 2.0, nulls=(-1.0, -1.0, -1.0)).stats()
        assert st["null_expectancy_r"] == pytest.approx(-1.0), (
            f"the control arm lost 1R on every trade, the card says {st['null_expectancy_r']}"
        )
        assert st["edge_vs_null"] == pytest.approx(3.0), (
            f"signals averaging +2R against a control averaging −1R is an edge of +3R, the card "
            f"says {st['edge_vs_null']}: the difference is being taken the wrong way round"
        )

    def test_the_fee_multiplier_charges_the_control_arm_too(self):
        """«And if the spread were three times worse?» — asked of the treatment but not the control.

        Both arms are the same trade size at the same fee, so the fee is common to the pair and has
        to cancel out of the paired difference: turning the knob may move both expectancies, never
        the edge between them. Charge only the signals and the sensitivity check answers a question
        nobody asked — it compares a strategy paying triple fees against a control paying single —
        and every edge shrinks under fee stress for a reason that is pure arithmetic. The existing
        multiplier tests look at the signal arm alone, where the mistake is invisible.
        """
        res = self._result(2.0, -1.0, nulls=(1.0, -1.0), cost_r=0.1)
        cheap, dear = res.stats(cost_mult=1.0), res.stats(cost_mult=3.0)
        assert cheap["edge_vs_null"] == pytest.approx(0.5), (
            f"the fixture no longer has an edge to preserve: {cheap['edge_vs_null']}"
        )
        assert dear["expectancy_r"] < cheap["expectancy_r"], "precondition: the knob does something"
        assert dear["edge_vs_null"] == pytest.approx(cheap["edge_vs_null"]), (
            f"tripling the fee moved the edge from {cheap['edge_vs_null']} to "
            f"{dear['edge_vs_null']}: the same fee is being charged to the signals and not to "
            "their controls, so the fee-sensitivity answer is arithmetic, not evidence"
        )

    def test_the_report_card_describes_the_trades_it_actually_resolved(self):
        """Two figures on the card that no test reads: the barrier tally and the spread of R.

        The tally is what tells an operator «half of these time out» — the single most useful thing
        on the report, because a book that is mostly vertical exits is not the strategy anyone
        thought they were testing. A tally that counts every trade against every barrier sums to
        three times the sample and still reads as a plausible breakdown. `r_std` is the dispersion
        every sizing decision leans on; sourced from the control arm it describes a different book
        entirely, and it looks perfectly reasonable because the control is the same trade size.
        """
        res = self._result(2.0, -1.0, 0.5, 0.0,
                           barriers=("tp", "sl", "vertical", "vertical"),
                           nulls=(0.0, 0.0, 0.0, 0.0))
        st = res.stats()
        assert st["barriers"] == {"tp": 1, "sl": 1, "vertical": 2}, (
            f"one target, one stop and two time exits were reported as {st['barriers']}"
        )
        assert sum(st["barriers"].values()) == st["n"], (
            f"the barrier tally adds up to {sum(st['barriers'].values())} over {st['n']} resolved "
            "trades: every trade is being counted against more than one barrier"
        )
        assert st["r_std"] == pytest.approx(float(np.std([2.0, -1.0, 0.5, 0.0]))), (
            f"the spread of the signals' R is {float(np.std([2.0, -1.0, 0.5, 0.0]))}, the card "
            f"says {st['r_std']}: it is describing a different set of trades from the expectancy "
            "printed beside it"
        )

    def test_a_breakeven_trade_is_not_a_win(self):
        """The hit rate is the number the user reads first and the one they overweight.

        A trade that closes exactly where it opened won nothing. Counted as a win it lifts the
        stated hit rate above the one the expectancy was computed from, and the two numbers on the
        same card then quietly contradict each other.
        """
        st = self._result(2.0, -1.0, 0.0).stats()
        assert st["n"] == 3
        assert st["hit_rate"] == pytest.approx(1 / 3), (
            f"one win in three trades is a hit rate of 0.33, the card says {st['hit_rate']}: a "
            "breakeven trade is being counted as a winner"
        )

    def test_the_profit_factor_is_what_was_won_over_what_was_lost(self):
        """Profit factor below 1 means the strategy loses money. It has to be that way round.

        Inverted, a losing system reports 2.0 and a winning one 0.5 — and because the figure is
        read as a quality score rather than checked, it would be believed.
        """
        st = self._result(3.0, 1.0, -2.0).stats()
        assert st["profit_factor"] == pytest.approx(2.0), (
            f"4R won against 2R lost is a profit factor of 2.0, the card says "
            f"{st['profit_factor']}"
        )
        losing = self._result(1.0, -2.0, -2.0).stats()
        assert losing["profit_factor"] < 1.0, (
            f"a system that lost 4R to win 1R reports a profit factor of "
            f"{losing['profit_factor']}: above 1 it reads as profitable"
        )

    def test_the_cost_multiplier_charges_every_trade_again(self):
        """The fee-sensitivity knob: «and if the spread were three times worse?».

        It is the cheapest test of whether an edge is real or is really a rebate. If the multiplier
        does not reach the arithmetic, the answer is always «it still works», which is precisely
        the answer nobody should trust.
        """
        res = self._result(2.0, -1.0, 0.0, cost_r=0.1)
        assert res.stats(cost_mult=1.0)["expectancy_r"] == pytest.approx(1 / 3 - 0.1)
        assert res.stats(cost_mult=2.0)["expectancy_r"] == pytest.approx(1 / 3 - 0.2), (
            "doubling the multiplier has to charge each trade one more fee"
        )
        assert res.stats(cost_mult=0.0)["expectancy_r"] == pytest.approx(1 / 3), (
            "at zero cost the expectancy is the gross one"
        )
