"""Assay is the product, so its guards are the part it matters most not to break."""

from __future__ import annotations

import numpy as np
import pytest

from wavelab.hypotheses.base import Hypothesis, Series
from wavelab.validation.battery import BatteryResult, run_battery
from wavelab.validation.battery import Test as BatteryTest  # aliased: pytest collects `Test*`
from wavelab.validation.csv_import import ImportError_, align_to_bars, parse_signals_csv
from wavelab.validation.evaluate import evaluate_all, forward_returns
from wavelab.validation.expr import (
    ExprError,
    build_series,
    crosses_above,
    crosses_below,
    evaluate_rule,
)
from wavelab.validation.reality_check import (
    effective_n,
    optimal_block_length,
    reality_check,
    stationary_bootstrap_indices,
)

DAY = 86_400_000
BASE = 1_600_000_000_000 - (1_600_000_000_000 % DAY)


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(4)
    n = 800
    c = 30_000 * np.exp(np.cumsum(rng.standard_normal(n) * 0.02))
    sp = np.abs(rng.standard_normal(n)) * c * 0.01
    o = np.concatenate([[c[0]], c[:-1]])
    return build_series(o, c + sp, c - sp, c, np.abs(rng.standard_normal(n)) * 100 + 10)


class TestEditorSecurity:
    """If this is offered as a service, the rule is written by a stranger."""

    @pytest.mark.parametrize("attack", [
        "__import__('os').system('ls')",
        "().__class__.__bases__[0].__subclasses__()",
        "open('/etc/passwd')",
        "exec('x=1')",
        "eval('1+1')",
        "[x for x in range(10)]",
        "lambda x: x",
        "close[0]",
        "globals()",
        "close.__class__",
    ])
    def test_rejects_code_execution(self, series, attack):
        with pytest.raises(ExprError):
            evaluate_rule(series, attack)

    def test_the_module_has_no_eval(self):
        """Regression: the interpreter is recursive precisely so that it needs no eval()."""
        from pathlib import Path

        import wavelab.validation.expr as m
        src = Path(m.__file__).read_text()
        assert "eval(compile" not in src and "exec(" not in src


class TestYouCannotLookAtTheFuture:
    """The domain's own constraint: the costliest mistake in the trade must not be writable."""

    @pytest.mark.parametrize("n", [-1, -5, -100])
    def test_negative_shift_is_forbidden(self, series, n):
        with pytest.raises(ExprError, match="FUTURE"):
            evaluate_rule(series, f"shift(close, {n}) > close")

    def test_positive_shift_is_allowed(self, series):
        r = evaluate_rule(series, "close > shift(close, 5)")
        assert r.n_long > 0

    def test_shift_returns_the_past(self):
        from wavelab.validation.expr import shift
        x = np.arange(10.0)
        d = shift(x, 3)
        assert np.isnan(d[:3]).all()
        assert (d[3:] == x[:-3]).all()

    def test_zero_is_the_present_and_stays_legal(self):
        """The ban is on the FUTURE, and zero is not the future: `shift(x, 0)` is today's value.

        The boundary has to sit exactly between 0 and -1, and it is worth pinning from both sides.
        Moved one step the safe way it rejects a legal rule — someone sweeping a lookback from 0
        upward gets an error on a rule that was never looking anywhere; moved one step the other
        way it silently admits one bar of lookahead, which is the disaster the module exists to
        prevent. Only a test on both sides holds it in place.
        """
        from wavelab.validation.expr import shift
        x = np.arange(1.0, 11.0)
        assert (shift(x, 0) == x).all(), "shift(x, 0) is the present and must pass it through"
        flat = build_series(x, x, x, x, np.ones_like(x))
        r = evaluate_rule(flat, "shift(close, 0) == close")
        assert r.n_long == x.size, "a rule using shift(x, 0) must evaluate, not raise"


class TestWarmupIsNotASignal:
    """An indicator has no value until it has enough bars. Those bars must read as NO, not YES.

    Every moving average, RSI and shift starts life as NaN. NaN is truthy the moment it reaches
    `astype(bool)`, so the one thing standing between the user and a burst of invented signals
    across the whole warm-up is that NaN is forced to false first. Get it backwards and a rule
    using `ema(close, 200)` fires on the first 200 bars of every backtest — at the START of the
    series, where the battery's out-of-sample folds and base rate all begin. It does not look like
    a bug; it looks like an edge.
    """

    def test_a_rule_that_is_still_warming_up_does_not_fire(self, series):
        r = evaluate_rule(series, "shift(close, 5)")
        assert r.signal[:5].tolist() == [0] * 5, (
            "the first 5 bars have no shift(close, 5) value yet and must be flat, not long")
        assert r.n_long == series["close"].size - 5, (
            "every bar that DOES have a value should fire; only the warm-up is excluded")

    def test_warmup_stays_off_through_and_or_and_not(self, series):
        """`and`/`or`/`not` run through a separate coercion from the one the final result uses,
        so the same NaN question has to be asked of it separately or half the guard is missing."""
        r = evaluate_rule(series, "shift(close, 5) and close > 0")
        assert r.signal[:5].tolist() == [0] * 5, (
            "a conjunction whose left side is still NaN must be false, not true")
        r_not = evaluate_rule(series, "not shift(close, 5)")
        assert r_not.signal[:5].tolist() == [1] * 5, (
            "`not` of an absent value is true, which is the mirror image of the same rule")


class TestCrossings:
    """A cross is a change of side, and touching counts as having been on the level.

    Prices sit exactly on a round number, on yesterday's close, or on a level the user typed in
    far more often than a random-walk intuition suggests. If `crosses_above` demands that the
    previous bar was STRICTLY below, every breakout that first rests on the line and then clears
    it is silently missed — the rule simply never fires, and the user concludes their idea does
    not work rather than that we dropped it.
    """

    LEVEL = np.full(6, 2.0)
    #                       below, equal, equal, above, equal, below
    PATH = np.array([1.0, 2.0, 2.0, 3.0, 2.0, 1.0])

    def test_rising_off_the_level_is_a_cross(self):
        fired = crosses_above(self.PATH, self.LEVEL)
        assert fired.tolist() == [0, 0, 0, 1, 0, 0], (
            "sitting on the level and then clearing it IS a cross above; "
            f"got fires at {np.flatnonzero(fired).tolist()}, expected [3]")

    def test_falling_off_the_level_is_a_cross(self):
        fired = crosses_below(self.PATH, self.LEVEL)
        assert fired.tolist() == [0, 0, 0, 0, 0, 1], (
            "sitting on the level and then dropping through it IS a cross below; "
            f"got fires at {np.flatnonzero(fired).tolist()}, expected [5]")

    def test_a_cross_fires_once_not_for_as_long_as_it_stays_over(self):
        """Otherwise `crosses_above` is just `>`, and the user's trade count is nonsense."""
        a = np.array([1.0, 3.0, 4.0, 5.0])
        fired = crosses_above(a, np.full(4, 2.0))
        assert fired.tolist() == [0, 1, 0, 0], "only the bar of the crossing itself counts"


class TestOperators:
    """`and`/`or` have to work element by element: that is what anyone actually writes."""

    def test_and_is_elementwise(self, series):
        a = evaluate_rule(series, "close > ema(close,50)").n_long
        b = evaluate_rule(series, "rsi(close,14) < 70").n_long
        both = evaluate_rule(series, "close > ema(close,50) and rsi(close,14) < 70").n_long
        assert both <= min(a, b), "a conjunction cannot fire on more bars than its parts"

    def test_or_is_elementwise(self, series):
        r = evaluate_rule(series, "rsi(close,14) < 30 or rsi(close,14) > 70")
        assert r.n_long > 0

    def test_long_and_short_at_once_means_flat(self, series):
        """A rule that says buy and sell at the same time is a contradiction, not a signal.
        Resolving it silently in favour of one side would hide the mistake from the user."""
        r = evaluate_rule(series, "close > 0", "close > 0")
        assert r.n_long == 0 and r.n_short == 0

    def test_chained_comparison_gives_a_useful_message(self, series):
        with pytest.raises(ExprError, match="one at a time"):
            evaluate_rule(series, "1 < close < 2")


class TestCsvImport:
    @pytest.mark.parametrize("csv,n_long", [
        # Spanish and Catalan column names and values are USER DATA, not code: a Spanish
        # spreadsheet says «fecha» and a Catalan one says «data», and both must import.
        ("fecha,señal\n2024-01-01,largo\n2024-02-01,fuera\n", 1),
        ("data,senyal\n2024-01-01,llarg\n2024-02-01,fora\n", 1),
        ("data,senyal\n2024-01-01,compra\n2024-02-01,venda\n", 1),
        ("timestamp,position\n1704067200000,1\n1706745600000,0\n", 1),
        ("time,signal\n1704067200,1\n1706745600,-1\n", 1),
        ("date,side\n2024-01-01,buy\n2024-02-01,sell\n", 1),
        ("date;pos\n2024-01-01;1\n2024-02-01;-1\n", 1),
    ])
    def test_accepts_human_formats(self, csv, n_long):
        assert parse_signals_csv(csv).n_long == n_long

    def test_normalises_position_size_to_its_sign(self):
        r = parse_signals_csv("date,position\n2024-01-01,0.5\n2024-02-01,-2.5\n2024-03-01,0\n")
        assert (r.n_long, r.n_short, r.n_flat) == (1, 1, 1)
        assert any("SIGN" in i for i in r.report), "the change must be REPORTED, not made quietly"

    @pytest.mark.parametrize("csv,frag", [
        ("fecha,precio\n2024-01-01,100\n", "signal column"),
        ("fecha,señal\n2024-01-01,quizá\n", "don't understand"),
        ("fecha,señal\n5,1\n7,-1\n", "neither a timestamp nor a date"),
        ("", "could not read"),
    ])
    def test_explains_its_errors_properly(self, csv, frag):
        with pytest.raises(ImportError_, match=frag):
            parse_signals_csv(csv)

    def test_the_position_persists_until_it_changes(self):
        """It is what people mean by «on the 3rd I was long». Reading it as «only that day»
        would turn a hold-for-weeks strategy into a one-day one."""
        r = parse_signals_csv("fecha,señal\n2020-09-13,largo\n2020-09-16,fuera\n")
        bars = np.array([BASE + i * DAY for i in range(6)])
        assert align_to_bars(r, bars, DAY).tolist() == [1, 1, 1, 0, 0, 0]

    def test_sorts_and_deduplicates(self):
        r = parse_signals_csv("fecha,señal\n2024-03-01,corto\n2024-01-01,largo\n2024-01-01,fuera\n")
        assert r.ts_ms[0] < r.ts_ms[-1]
        assert len(r.ts_ms) == 2, "the duplicate timestamp must collapse"


def _ts(n):
    return np.array([BASE + i * DAY for i in range(n)])


@pytest.fixture(scope="module")
def coin_flip_battery():
    """A signal with no information in it at all, run through the whole battery once."""
    rng = np.random.default_rng(2)
    n = 1200
    c = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.02))
    return run_battery(c, _ts(n), (rng.random(n) < 0.3).astype(float), bar_ms=DAY, n_random=60)


class TestTheVerdictLadder:
    """The verdict is the one line the user reads. Its grades have to mean what they say.

    `passed` is deliberately three-valued: True, False, and None for "not enough data to say".
    Folding None in with False turns "we could not test this" into "this failed", which both
    slanders a strategy we never actually tested and — worse — burns the second failure that
    would otherwise have distinguished a doubtful result from a dead one.
    """

    @staticmethod
    def _result(**passed):
        return BatteryResult("s", 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             [BatteryTest(k, k, v, 0.0, 0.0, "", "") for k, v in passed.items()])

    @pytest.mark.parametrize("tests,expected", [
        ({"delay": True, "base_rate": True, "random_control": True}, "survives"),
        ({"delay": True, "base_rate": False}, "doubtful"),
        ({"delay": True, "base_rate": False, "random_control": False}, "fails"),
        ({"delay": False, "base_rate": True}, "lookahead"),
        # reading the future outranks everything, however much else also fails
        ({"delay": False, "base_rate": False, "random_control": False}, "lookahead"),
    ])
    def test_the_grades(self, tests, expected):
        got = self._result(**tests).verdict
        assert got == expected, f"{tests} should read {expected!r}, read {got!r}"

    @pytest.mark.parametrize("tests,expected", [
        ({"delay": True, "base_rate": None, "random_control": None}, "survives"),
        ({"delay": True, "base_rate": None, "random_control": False}, "doubtful"),
        ({"delay": None, "base_rate": True}, "survives"),
    ])
    def test_inconclusive_is_not_a_failure(self, tests, expected):
        """«We could not measure this» is not «this is broken», and the summary the user is shown
        is chosen by exactly this count."""
        got = self._result(**tests).verdict
        assert got == expected, (
            f"an inconclusive test was counted as a failure: {tests} read {got!r}, "
            f"expected {expected!r}")

    def test_every_grade_has_a_summary(self):
        for tests, expected in (({"delay": True}, "survives"), ({"delay": True, "a": False}, None),
                                ({"delay": False}, "lookahead")):
            r = self._result(**tests)
            assert r.summary, f"verdict {r.verdict!r} has no sentence to show the user"
            if expected:
                assert r.verdict == expected


class TestTheBacktestArithmetic:
    """The three headline numbers the report prints. Each is quietly easy to get backwards."""

    def test_paying_more_to_trade_can_never_pay_better(self):
        """Turnover is a COST, and it is charged on the size of the change, not on its direction.

        A flip from long to short is two trades, not minus-one trade. Lose the magnitude and the
        churniest strategies — the ones costs exist to expose — start being PAID to trade: the
        charge telescopes away to nothing over the run and a high-frequency rule that cannot
        survive its own commissions comes out looking better than one that barely trades.
        Prices are held flat here so that what is left in the equity curve is only the cost.
        """
        flat, ts = np.full(400, 100.0), _ts(400)
        flips = np.concatenate([np.ones(200), -np.ones(200)])   # in, flip, and still on at the end

        def cagr(sig, bps):
            return run_battery(flat, ts, sig, bar_ms=DAY, n_random=3, cost_bps=bps).cagr

        free, cheap, dear = cagr(flips, 0), cagr(flips, 100), cagr(flips, 500)
        assert free > cheap > dear, (
            "raising the cost of trading must lower the result, never raise it: "
            f"0bps={free:.4f} 100bps={cheap:.4f} 500bps={dear:.4f}")
        in_and_out = np.concatenate([np.ones(200), np.zeros(200)])
        assert cagr(flips, 500) < cagr(in_and_out, 500), (
            "reversing a position is two trades and must cost more than simply closing it")

    def test_being_short_is_being_in_the_market(self):
        """Exposure is time at risk, and it feeds the random control: the random filters are
        generated to match it. Sign it, and a short-heavy strategy is measured against controls
        that are almost always FLAT, which is a bar it cannot help but clear."""
        n = 600
        c = 100 * np.exp(np.cumsum(np.random.default_rng(1).standard_normal(n) * 0.02))
        r = run_battery(c, _ts(n), -np.ones(n), bar_ms=DAY, n_random=20)
        assert r.exposure == pytest.approx(1.0), (
            f"a permanently short strategy is in the market all the time, not {r.exposure:.2f}")
        assert r.n_signals == n - 1, "every bar of it is a bar holding a position"

    def test_max_drawdown_is_the_worst_fall_not_the_best_rise(self):
        """A drawdown that always reports 0% is the single most flattering bug a report can have:
        it tells the user the strategy never lost, on a curve that halved."""
        path = np.array([100.0, 100.0, 100.0, 200.0, 100.0, 150.0, 150.0, 150.0])
        r = run_battery(path, _ts(path.size), np.ones(path.size), bar_ms=DAY, n_random=3)
        assert r.max_dd_bh == pytest.approx(-0.5, abs=1e-9), (
            f"the curve went 200 -> 100, a 50% drawdown, but max_dd_bh reads {r.max_dd_bh:.4f}")
        assert r.max_dd <= 0, f"a drawdown is a fall and cannot be positive: {r.max_dd:.4f}"


class TestBattery:
    def test_catches_a_signal_that_looks_at_the_future(self):
        """The delay test against a signal with deliberate lookahead."""
        from wavelab.validation.battery import run_battery
        rng = np.random.default_rng(1)
        n = 1200
        c = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.02))
        ts = np.array([BASE + i * DAY for i in range(n)])
        cheating = np.zeros(n)
        cheating[:-1] = (np.diff(c) > 0).astype(float)   # it knows whether tomorrow goes up
        r = run_battery(c, ts, cheating, bar_ms=DAY, n_random=60)
        assert r.verdict == "lookahead", f"the lookahead got through: {r.verdict}"
        assert not next(t for t in r.tests if t.id == "delay").passed

    def test_a_random_signal_does_not_survive(self):
        from wavelab.validation.battery import run_battery
        rng = np.random.default_rng(2)
        n = 1200
        c = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.02))
        ts = np.array([BASE + i * DAY for i in range(n)])
        sig = (rng.random(n) < 0.3).astype(float)
        r = run_battery(c, ts, sig, bar_ms=DAY, n_random=60)
        assert r.verdict != "survives"

    def test_a_signal_that_is_itself_a_random_filter_lands_mid_pack_and_is_failed(self):
        """Not just «the battery dislikes it somewhere» but the specific test that exists for it.

        Test 3 asks whether the strategy beats filters that enter and exit at its own frequency
        and spend its own time in the market — and only the top 5% counts as an answer. The
        signal here IS one of those filters: random blocks of twenty bars. It duly lands in the
        middle of them, and being in the middle has to read as a FAIL. That is the whole content
        of the test: anything short of clearly above the pack is exposure, not judgement. Relax
        the bar and this test still appears in the report, still says «pass», and means nothing.
        """
        n = 1200
        c = 100 * np.exp(np.cumsum(np.random.default_rng(2).standard_normal(n) * 0.02))
        blocks = (np.random.default_rng(11).random(n // 20) < 0.5).astype(float)
        r = run_battery(c, _ts(n), np.repeat(blocks, 20)[:n], bar_ms=DAY, n_random=60)
        t = next(t for t in r.tests if t.id == "random_control")
        assert 5.0 < t.value < 95.0, (
            f"this signal is meant to be mid-pack for the test to mean anything; it scored "
            f"percentile {t.value:.0f}")
        assert t.passed is False, (
            f"a signal indistinguishable from the random controls reached percentile "
            f"{t.value:.0f} and was still graded {t.status!r}")

    def test_a_strategy_with_no_edge_to_begin_with_gets_no_verdict_out_of_sample(
            self, coin_flip_battery):
        """Test 4 asks how much of the edge survives outside the sample. When there was never an
        edge in sample, «kept 122% of it» is arithmetic about a negative number, not survival.
        Reporting that as a FAILURE would be double-counting: the strategy is already being
        marked down for having no edge, and a second mark for failing to keep it is the extra
        failure that tips «doubtful» into «does not survive» on evidence that does not exist.
        """
        t = next(t for t in coin_flip_battery.tests if t.id == "out_of_sample")
        assert t.passed is None, (
            f"in-sample excess is negative, so out_of_sample cannot be judged, but it was "
            f"graded {t.status!r}")
        assert "negative" in t.detail, "the report has to say WHY it could not be judged"


class TestEffectiveN:
    """Test 5 of the battery: how many of the signals are actually SEPARATE observations.

    Overlapping signals share the same price move, so they are one bet reported many times. It is
    the difference between a p-value of 0.001 and one of 0.3, and it is the mechanism by which an
    honest-looking «n=500» becomes a study of twelve episodes. Everything here is arithmetic on
    timestamps, so it can be pinned exactly rather than statistically.
    """

    WINDOW_BARS, BAR = 5, 1000     # a five-bar horizon on one-second bars

    def n(self, *ms):
        return effective_n(np.array(ms, dtype=np.int64), self.WINDOW_BARS, self.BAR)

    def test_signals_inside_the_horizon_are_one_observation(self):
        assert self.n(0, 1000, 2000) == 1, (
            "three signals inside a single five-second horizon are one bet, not three")

    def test_a_signal_a_full_horizon_later_is_a_new_observation(self):
        """The boundary is inclusive because at exactly one horizon the first bet has closed:
        the two no longer share a single price move. Excluding it would quietly discount a
        strategy that trades on precisely the rhythm it was designed to trade on."""
        assert self.n(0, 5000) == 2, (
            "5,000ms is exactly the five-bar horizon, so the second signal is independent")
        assert self.n(0, 4999) == 1, "one millisecond short of the horizon, they still overlap"

    def test_the_window_is_measured_from_the_last_episode_counted(self):
        """Not from the previous signal. Measured from the previous signal, a steady drip of
        signals each just inside the horizon chains forever, and hours of trading collapse into
        a single «observation» however far apart the ends are."""
        assert self.n(0, 4000, 8000, 12000) == 2, (
            "0 and 8000 are more than a horizon apart and must count separately, however many "
            "signals sit between them")

    def test_the_count_does_not_depend_on_the_order_they_arrive_in(self):
        """Signals reach this from a CSV the user sorted however they liked."""
        assert self.n(12000, 8000, 4000, 0) == self.n(0, 4000, 8000, 12000), (
            "the same signals in reverse order are the same signals")

    def test_no_signals_is_no_observations(self):
        assert effective_n(np.array([], dtype=np.int64), self.WINDOW_BARS, self.BAR) == 0, (
            "a strategy that never fired has zero independent observations, not one")


@pytest.fixture(scope="module")
def rc_pair():
    """One set of pure-noise strategies, and the same set plus a genuine edge."""
    rng = np.random.default_rng(31)
    n = 1500
    noise = {f"noise{i}": rng.standard_normal(n) * 0.01 for i in range(6)}
    with_edge = {**noise, "winner": rng.standard_normal(n) * 0.01 + 0.006}
    return (reality_check(noise, n_boot=400), reality_check(with_edge, n_boot=400))


class TestRealityCheck:
    """Correcting for having tried many strategies — the guard against the study that is really
    a search. Its whole job is to answer one question in two directions, so both are tested.
    """

    def test_the_best_of_a_pile_of_noise_is_not_an_edge(self, rc_pair):
        """This is the failure mode the module exists for: keep the best of K random strategies
        and its own p-value looks wonderful, because you ran K draws and kept the maximum."""
        noise, _ = rc_pair
        assert not noise.significant, (
            "the best of six pure-noise strategies was declared significant at "
            f"p={noise.p_value:.4f}")

    def test_a_real_edge_still_gets_through(self, rc_pair):
        """The other direction, and the one a broken null quietly destroys. A correction that
        rejects everything is not conservative, it is useless: it would make the tool unable to
        say yes to anything, including the strategies it was built to find."""
        noise, edge = rc_pair
        assert edge.significant, (
            "a strategy with a real, consistent edge did not survive the correction: "
            f"p={edge.p_value:.4f}")
        assert edge.p_value < noise.p_value, (
            "a real edge must be more significant than the best of the same noise without it: "
            f"edge p={edge.p_value:.4f} vs noise p={noise.p_value:.4f}")

    def test_it_names_the_winner(self, rc_pair):
        _, edge = rc_pair
        assert edge.best_name == "winner", (
            f"the strategy with the highest mean is «winner», but the report names "
            f"«{edge.best_name}»")
        assert edge.n_strategies == 7, "all seven were tried and all seven must be counted"

    def test_every_strategy_is_given_its_own_p_value_not_just_the_winner(self, rc_pair):
        """Reporting only the best one is how a study turns into a brochure: the reader cannot
        tell a set with one good idea from a set that was dredged."""
        _, edge = rc_pair
        assert set(edge.p_values) == {"winner", *(f"noise{i}" for i in range(6))}, (
            "every strategy tried has to appear, the failures included")
        worst_noise = min(edge.p_values[f"noise{i}"] for i in range(6))
        assert edge.p_values["winner"] < worst_noise, (
            "the individual p-values are ordered backwards: winner scored "
            f"{edge.p_values['winner']:.3f} against the luckiest noise at {worst_noise:.3f}")

    def test_it_renders_its_answer_for_a_human(self, rc_pair):
        _, edge = rc_pair
        text = edge.render()
        assert "winner" in text and "7" in text, (
            "the rendered verdict must name the strategy and how many were tried")


# ---------------------------------------------------------------- out-of-sample discipline

def _rising_series(n=1000, tf="1d", bar_ms=DAY, seed=9):
    """A series with a clear upward drift, which is the condition every one of these guards
    exists to survive: in a rising market, doing nothing clever already looks clever."""
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01 + 0.002))
    ts = np.array([BASE + i * bar_ms for i in range(n)])
    return Series(tf, ts, c, c, c, c, np.ones(n))


def _hyp(name, fn, timeframes=("1d",)):
    return Hypothesis(name, "test", "declared up front", "declared up front", fn,
                      timeframes=timeframes, min_warmup=0)


class TestForwardReturns:
    """What «the future» means, defined once and used by every hypothesis result.

    This is the label every hypothesis is scored against, so an error here does not produce a
    wrong number for one strategy: it produces a consistent, plausible, wrong answer for all of
    them at once, and nothing else in the pipeline is in a position to notice.
    """

    def test_it_looks_exactly_horizon_bars_forward(self):
        prices = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        fwd = forward_returns(prices, 2)
        assert fwd[:3].tolist() == pytest.approx([np.log(4)] * 3), (
            "bar i must be scored on the move from i to i+2, a quadrupling here; "
            f"got {fwd[:3].tolist()}")

    def test_the_bars_with_no_future_left_are_nan_not_zero(self):
        """The last bars of the series have no outcome yet. Zero would be a real observation —
        a flat one — and would quietly dilute every average it entered."""
        fwd = forward_returns(np.array([1.0, 2.0, 4.0, 8.0, 16.0]), 2)
        assert np.isnan(fwd[-2:]).all(), (
            f"the final 2 bars cannot be scored and must be NaN; got {fwd[-2:].tolist()}")
        assert not np.isnan(fwd[:-2]).any(), "every bar that DOES have a future must be scored"

    def test_a_series_barely_longer_than_the_horizon_still_yields_its_one_answer(self):
        """The guard is `size > horizon`, and off by one it throws away real observations at the
        short end while looking perfectly correct on the long series everyone tests with."""
        fwd = forward_returns(np.array([1.0, 3.0]), 1)
        assert fwd[0] == pytest.approx(np.log(3)), (
            f"two bars and a one-bar horizon is exactly one observation; got {fwd.tolist()}")
        assert np.isnan(fwd[1])

    def test_a_rising_series_scores_positive_not_negative(self):
        """Invert the ratio and every long becomes a short. The battery would still run, the
        report would still print, and the sign of every conclusion would be flipped."""
        fwd = forward_returns(np.array([1.0, 2.0, 3.0, 4.0]), 1)
        assert (fwd[:-1] > 0).all(), (
            f"prices only rise here, so every forward return is positive: {fwd.tolist()}")


class TestOutOfSampleDiscipline:
    """The four controls in `evaluate_all`, checked by what they EXCLUDE.

    Each of these is a guard that fails silently: nothing crashes, a number simply appears where
    there should have been none, and it is always a flattering number.
    """

    HORIZON = 10
    FOLDS = 5

    def _run(self, fn, name="h", **kw):
        res, _ = evaluate_all({name: _hyp(name, fn)}, {"1d": _rising_series()},
                              horizon_bars={"1d": self.HORIZON}, folds=self.FOLDS,
                              n_boot=0, **kw)
        assert res, "the synthetic hypothesis produced no result to inspect"
        return res[0]

    def test_the_training_fold_is_never_reported_as_out_of_sample(self):
        """The first 1/folds of history is what the idea was allowed to be found in. If it leaks
        back into the out-of-sample number, «it also works out of sample» is a statement about
        the very data the strategy was fitted to, which is the whole disease this is a cure for.
        """
        r = self._run(lambda s: np.where(np.arange(len(s)) < 200, 1, 0).astype(np.int8),
                      min_signals=20)
        assert r.n_signals == 200, "the hypothesis should have fired on all 200 in-sample bars"
        assert np.isnan(r.oos_edge), (
            "every signal sits inside the held-back training fold, so there is nothing to report "
            f"out of sample, yet oos_edge came back as {r.oos_edge}")

    def test_the_horizon_is_purged_at_every_fold_boundary(self):
        """A trade opened in the last bars of one segment is still open when the next begins, so
        its outcome was decided by data from the segment it is being counted against. Purging is
        the only thing that stops the boundary from leaking, and dropping it leaks a little at
        every single fold — invisibly, and always in the direction of a better result."""
        def only_in_the_purge_zones(s):
            out = np.zeros(len(s), dtype=np.int8)
            for k in range(1, self.FOLDS):
                out[k * 200: k * 200 + self.HORIZON] = 1
            return out

        r = self._run(only_in_the_purge_zones, min_signals=20)
        assert r.n_signals == (self.FOLDS - 1) * self.HORIZON, "40 signals, 10 per boundary"
        assert np.isnan(r.oos_edge), (
            "every signal falls in a purge window at a fold boundary and must be excluded, "
            f"yet oos_edge came back as {r.oos_edge}")

    def test_simply_being_long_in_a_rising_market_scores_nothing(self):
        """★ The correction is fed return IN EXCESS OF THE DRIFT, not raw return.

        Fed raw return, the winner of any comparison on a rising asset is whichever strategy was
        long the longest — which is beta, and beta costs nothing to buy. Scoring against the
        average bar instead asks the only question worth asking: did THIS moment beat any old
        moment? A rule that is long always has, by construction, no answer to that, and must
        score exactly zero rather than the market's own return.
        """
        r = self._run(lambda s: np.ones(len(s), dtype=np.int8))
        assert r.base_ret > 0, "the fixture must actually drift upward for this to mean anything"
        assert r.per_bar.sum() == pytest.approx(0.0, abs=1e-9), (
            "an always-long rule earned "
            f"{r.per_bar.sum():+.4f} of credit on drift alone; it must earn exactly 0")
        assert r.edge == pytest.approx(0.0, abs=1e-9), (
            f"its edge over the base rate is 0 by construction, not {r.edge:+.4f}")

    def test_the_correction_is_not_run_across_timeframes(self):
        """White's Reality Check compares strategies bar by bar, so it needs series aligned in
        time. Stacking a 1d series against a 4h one is not a correction, it is a shape error —
        and it takes down the whole evaluation run rather than one hypothesis."""
        h = _hyp("both", lambda s: np.ones(len(s), dtype=np.int8), timeframes=("1d", "4h"))
        res, rc = evaluate_all(
            {"both": h},
            {"1d": _rising_series(1000), "4h": _rising_series(800, "4h", 14_400_000, seed=10)},
            horizon_bars={"1d": self.HORIZON, "4h": self.HORIZON}, n_boot=100)
        assert len({r.tf for r in res}) == 2, "both timeframes should have produced a result"
        assert rc is None, (
            "series of different lengths cannot be compared bar by bar, so the correction must "
            "be declined rather than attempted")


class TestCsvImportGetsTheTimeRight:
    """Everything downstream is indexed by time, so a misread clock is not a small error.

    A signal placed on the wrong bar is not noise: it is either lookahead or a missed trade, and
    the battery will faithfully measure the strategy that results rather than the one the user
    described.
    """

    def test_an_epoch_in_seconds_becomes_milliseconds(self):
        """Read as milliseconds, a 2024 epoch in seconds lands in January 1970 — before every
        bar we hold. The signal then aligns to nothing at all and the strategy silently becomes
        «flat forever», which passes a good many tests by doing nothing."""
        imp = parse_signals_csv("time,signal\n1704067200,1\n1706745600,-1\n")
        assert imp.ts_ms.tolist() == [1_704_067_200_000, 1_706_745_600_000], (
            f"a seconds epoch must be scaled to ms; got {imp.ts_ms.tolist()}")
        assert any("SECONDS" in line for line in imp.report), (
            "the guess about the unit must be REPORTED, since it is a guess")

    def test_a_signal_timed_inside_a_bar_lands_on_that_bar(self):
        """People export signals with real clock times, not with our bar boundaries. A timestamp
        of 17:30 belongs to the day that contains it; matched exactly instead, it matches no bar
        at all and the whole strategy quietly evaporates."""
        imp = parse_signals_csv("date,signal\n2020-09-13 17:30,largo\n")
        bars = np.array([BASE + i * DAY for i in range(4)])
        aligned = align_to_bars(imp, bars, DAY)
        assert aligned.tolist() == [1, 1, 1, 1], (
            "a signal at 17:30 belongs to that day's bar and persists after it; "
            f"got {aligned.tolist()}")

    def test_an_exactly_named_column_beats_one_that_merely_contains_the_word(self):
        """Detection is exact-first for a reason: a column literally called «date» is the date,
        even when some other column happens to have «time» buried in its name. Guessing the
        wrong column is the failure the whole `report` exists to make visible."""
        csv = ("update_time,date,signal\n"
               "2030-01-01,2024-01-01,largo\n2030-02-01,2024-02-01,fuera\n")
        imp = parse_signals_csv(csv)
        assert imp.ts_ms[0] == 1_704_067_200_000, (
            "«date» is an exact match and must win over «update_time», which only contains "
            f"«time»; the first timestamp read was {imp.ts_ms[0]}")
        assert "«date»" in " ".join(imp.report), "the report must name the column it chose"


class TestCsvImportLastWordWins:
    """When the same moment is described twice, the later row is the correction.

    Spreadsheets are edited by appending. Someone who writes a row, thinks again and writes
    another for the same timestamp means the second one; taking the first silently trades the
    opposite of what they asked for, which is the worst possible way to be wrong.
    """

    def test_a_repeated_timestamp_keeps_the_later_row(self):
        # A third, distinct row matters: with the duplicates as the WHOLE file, keeping the
        # first and keeping the last pick the same row and the bug hides.
        r = parse_signals_csv(
            "fecha,señal\n2024-01-01,largo\n2024-01-01,corto\n2024-02-01,fuera\n")
        assert r.signal.tolist() == [-1, 0], (
            "the second row says «corto», so January is short, not long; "
            f"got {r.signal.tolist()}")
        assert (r.n_long, r.n_short, r.n_flat) == (0, 1, 1), (
            f"the discarded «largo» must not be counted: {(r.n_long, r.n_short, r.n_flat)}")
        assert any("duplicate" in line for line in r.report), (
            "silently dropping one of the user's rows must be reported")

    def test_two_signals_inside_one_bar_leave_the_later_one_standing(self):
        """We only have one bar to put them in, and at that bar's close the position in force is
        the one given last, not the one given first."""
        imp = parse_signals_csv(
            "date,signal\n2020-09-13 09:00,largo\n2020-09-13 17:00,corto\n")
        aligned = align_to_bars(imp, np.array([BASE, BASE + DAY]), DAY)
        assert aligned.tolist() == [-1, -1], (
            "long at 09:00 then short at 17:00 leaves the day short; "
            f"got {aligned.tolist()}")


class TestCsvImportPrices:
    def test_a_column_of_non_positive_prices_is_refused_not_used(self):
        """The user's own prices are turned into log returns. A zero or a negative there is not a
        cheap price, it is a broken column — and it makes the whole equity curve undefined. It
        has to be dropped, and the user told, rather than quietly carried into the report."""
        r = parse_signals_csv("date,signal,close\n2024-01-01,largo,0\n2024-02-01,fuera,0\n")
        assert not r.has_prices, "a price column of zeros cannot be used as a price series"
        assert any("discarded" in line for line in r.report), (
            "dropping the user's own prices is a decision they must be told about")

    def test_real_prices_are_taken_from_the_file(self):
        """The other half: without this the tool only serves people trading exactly what we do."""
        r = parse_signals_csv("date,signal,close\n2024-01-01,largo,100\n2024-02-01,fuera,110\n")
        assert r.has_prices and r.price.tolist() == [100.0, 110.0], (
            f"the user's own prices must be used: got {None if r.price is None else r.price}")


class TestTheBootstrapItself:
    """The resampling underneath the correction, on the short series that take the other path.

    `reality_check` has two implementations inside it — an aggregated one for long series and the
    plain stationary bootstrap for short ones — and a user with a year of daily signals gets the
    second. It is the half that no long-series test ever reaches, so it is checked directly.
    """

    def test_it_still_separates_an_edge_from_noise_on_a_short_history(self):
        rng = np.random.default_rng(31)
        n = 120
        noise = {f"noise{i}": rng.standard_normal(n) * 0.01 for i in range(4)}
        assert not reality_check(noise, n_boot=300).significant, (
            "the best of four noise strategies is still noise on a short history")
        edge = reality_check({**noise, "winner": rng.standard_normal(n) * 0.01 + 0.02},
                             n_boot=300)
        assert edge.significant and edge.best_name == "winner", (
            f"a large edge on a short history must still be found: p={edge.p_value:.4f}, "
            f"best={edge.best_name!r}")

    def test_a_resample_is_the_same_length_and_stays_inside_the_series(self):
        idx = stationary_bootstrap_indices(50, 5.0, np.random.default_rng(0))
        assert idx.size == 50, f"a resample of 50 observations must have 50 of them, not {idx.size}"
        assert idx.min() >= 0 and idx.max() < 50, (
            f"every index must point at a real observation; got {idx.min()}..{idx.max()}")

    def test_it_resamples_BLOCKS_and_not_loose_observations(self):
        """Returns are autocorrelated and signals bunch into episodes. Shuffle single
        observations and that dependence is destroyed, which narrows every interval and makes
        the correction confidently wrong in the one direction that matters."""
        idx = stationary_bootstrap_indices(200, 10.0, np.random.default_rng(1))
        consecutive = int((np.diff(idx) == 1).sum())
        assert consecutive > 100, (
            f"with a mean block of 10 most steps should continue the previous run; only "
            f"{consecutive} of 199 did, which is a shuffle, not a block bootstrap")

    def test_the_block_length_tracks_the_dependence_it_has_to_preserve(self):
        """The block length is the one knob the bootstrap's honesty rests on, and nothing pinned it.

        `stationary_bootstrap_indices` is handed a block length; the tests above always hand it a
        hard-coded one. So `optimal_block_length` could return a constant 1.0 — degenerating the
        stationary bootstrap into the loose-observation shuffle that the test above exists to rule
        out — and every assertion in this file would still pass.

        Its contract: at least 2, never more than a quarter of the sample, and LONGER the more
        persistent the series, because that is the whole reason blocks are used at all.
        """
        rng = np.random.default_rng(5)
        n = 600
        white = rng.standard_normal(n)

        ar = np.empty(n)                      # strongly autocorrelated: rho ~ 0.9
        ar[0] = rng.standard_normal()
        for i in range(1, n):
            ar[i] = 0.9 * ar[i - 1] + rng.standard_normal()

        b_white, b_ar = optimal_block_length(white), optimal_block_length(ar)

        for name, b in (("white noise", b_white), ("AR(1)", b_ar)):
            assert b >= 2.0, (
                f"{name}: a block length of {b} is not a block at all — at 1.0 the stationary "
                "bootstrap degenerates into an iid shuffle and every interval comes out too narrow"
            )
            assert b <= n / 4, f"{name}: a block of {b} over {n} observations resamples nothing"

        assert b_ar > b_white, (
            f"a persistent series needs LONGER blocks to carry its dependence across, but AR(1) "
            f"got {b_ar:.2f} and white noise got {b_white:.2f}"
        )

    def test_a_short_series_still_gets_a_usable_block(self):
        assert optimal_block_length(np.arange(5.0)) >= 2.0, (
            "even below the n<20 shortcut the result has to be a block, not a single observation"
        )
