"""Assay is the product, so its guards are the part it matters most not to break."""

from __future__ import annotations

import numpy as np
import pytest

from wavelab.hypotheses.base import Hypothesis, Series
from wavelab.validation import reality_check as reality_check_mod
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

    def test_a_position_smaller_than_one_unit_is_still_a_position(self):
        """The test above uses 0.5 and −2.5, whose largest magnitude is over 1, so it takes the
        «these are position sizes» branch. A file where EVERY size is under 1 — a book kept in
        fractions of capital, which is how a careful person keeps one — takes the other branch,
        and there truncating to int8 instead of taking the sign turns every row into 0.

        The failure is silent and total: the import succeeds, the report cheerfully lists the
        values it read, and the strategy is «never in the market». It passes several of the
        battery's tests by doing nothing at all.
        """
        r = parse_signals_csv("date,position\n2024-01-01,0.5\n2024-02-01,-0.5\n2024-03-01,0\n")
        assert (r.n_long, r.n_short, r.n_flat) == (1, 1, 1), (
            "half a unit long is long and half a unit short is short; got "
            f"{(r.n_long, r.n_short, r.n_flat)} — all-flat means the sign was truncated away")

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


def _battery_test(result, test_id):
    """The one test in a battery result carrying this id."""
    found = [t for t in result.tests if t.id == test_id]
    assert len(found) == 1, f"expected exactly one {test_id!r} test, found {len(found)}"
    return found[0]


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

    def test_the_headline_figures_are_the_ones_the_published_curve_supports(self):
        """CAGR and Sharpe are checked here against the equity curve the SAME result publishes,
        which is an oracle the arithmetic cannot drift away from.

        Nothing in this file asserts the magnitude of either number — only orderings between them
        — and both are one operator away from nonsense that still prints. Divide by the sample
        length instead of multiplying and every CAGR in the report reads 0.0000%, the lookahead
        cheat's +1,022% included. Annualise the Sharpe by the number of periods rather than by its
        square root and a Sharpe of 0.9 is published as 17, a figure no strategy in history has
        ever had, on a page whose whole claim is that it tells you the truth.
        """
        rng = np.random.default_rng(21)
        n = 900
        close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.02 + 0.001))
        sig = (rng.random(n) < 0.5).astype(float)
        r = run_battery(close, _ts(n), sig, bar_ms=DAY, n_random=3)
        per_year = 365.25 * 24 * 3600 * 1000 / DAY

        for label, eq, cagr, sharpe in (("strategy", r.equity, r.cagr, r.sharpe),
                                        ("buy and hold", r.equity_bh, r.cagr_bh, r.sharpe_bh)):
            ret = np.diff(np.log(eq))
            assert cagr == pytest.approx(eq[-1] ** (per_year / len(eq)) - 1), (
                f"{label}: a curve ending at {eq[-1]:.4f} over {len(eq) / per_year:.2f} years "
                f"compounds at {eq[-1] ** (per_year / len(eq)) - 1:+.4f}, not {cagr:+.4f}")
            assert sharpe == pytest.approx(ret.mean() / ret.std() * np.sqrt(per_year)), (
                f"{label}: the annualised Sharpe of that curve is "
                f"{ret.mean() / ret.std() * np.sqrt(per_year):.4f}, not {sharpe:.4f}")

    def test_the_cost_of_a_trade_is_charged_on_the_bar_the_trade_happens(self):
        """Opening a position is a trade and it is paid for on the way IN. Prices are flat here,
        so the only thing left in the equity curve is the commission, and there is exactly one
        trade in the whole history: the entry on the first bar.

        Charged anywhere else the total over a long run is unchanged, so every ordering test in
        this class still passes — but a strategy that opens a position and holds it for years is
        reported as free until the day it closes, which is precisely the report someone reads
        while deciding whether the position was worth opening.
        """
        n = 300
        r = run_battery(np.full(n, 100.0), _ts(n), np.ones(n), bar_ms=DAY, n_random=3,
                        cost_bps=100)
        assert r.equity[0] == pytest.approx(np.exp(-100 / 10_000)), (
            "one entry at 100bps costs 1% and it is due on the entry bar; the curve opens at "
            f"{r.equity[0]:.6f} instead of {np.exp(-100 / 10_000):.6f}")
        assert r.equity[-1] == pytest.approx(r.equity[0]), (
            "and nothing else is ever traded, so the curve never moves again")


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

    def test_the_random_control_grades_from_the_top_of_the_pack_not_the_bottom(self):
        """Test 3 is a percentile, and a percentile is a comparison that can be run backwards.

        The only test that pins it uses a MID-PACK signal, and mid-pack is the fixed point of the
        inversion: 50 maps to 50 and fails either way. So the test that exists to defend this
        cannot see the fault at all. Inverted, the battery grades every strategy backwards and
        hands its top verdict — «SURVIVES the battery» — to the exposure-only strategies it was
        built to expose, while failing the ones with a real edge. Two anchors at opposite ends of
        the scale is the least that pins a direction; one in the middle pins nothing.
        """
        n = 1200
        c = 100 * np.exp(np.cumsum(np.random.default_rng(3).standard_normal(n) * 0.02))
        up_tomorrow = np.zeros(n)
        up_tomorrow[:-1] = (np.diff(c) > 0).astype(float)

        best = _battery_test(run_battery(c, _ts(n), up_tomorrow, bar_ms=DAY, n_random=60),
                             "random_control")
        assert best.value > 95.0 and best.passed is True, (
            "a signal that knows tomorrow's direction beats every random filter, so it belongs "
            f"at the top of the pack; it landed at percentile {best.value:.0f}, graded "
            f"{best.status!r}")

        worst = _battery_test(run_battery(c, _ts(n), -up_tomorrow, bar_ms=DAY, n_random=60),
                              "random_control")
        assert worst.value < 5.0 and worst.passed is False, (
            "and the same signal taken the wrong way round is beaten by every one of them, so it "
            f"belongs at the bottom; it landed at percentile {worst.value:.0f}, graded "
            f"{worst.status!r}")

    def test_the_bar_the_random_control_has_to_clear_is_the_ninety_fifth_percentile(self):
        """The two anchors above sit at percentile 100 and percentile 0, which pins the DIRECTION
        and leaves the bar free in both directions: moved to 0.90 or to 0.99, both of them still
        pass and nothing anywhere asserts that the number is 95.

        That matters because the bar is the verdict. At 0.90 one strategy in ten clears test 3 by
        chance and the battery starts handing its top grade to noise; at 0.99 almost nothing ever
        clears it and the tool refuses everything, which reads as rigour and is just a broken gate.
        Two assertions close it: the reference the report PRINTS, and the invariant that the grade
        is that same printed number compared against the printed value. Together they leave nowhere
        for a threshold to move to — shift the comparison and it stops agreeing with the reference
        beside it, which is also the shape of the bug a reader of the report could never spot.
        """
        n = 1200
        c = 100 * np.exp(np.cumsum(np.random.default_rng(3).standard_normal(n) * 0.02))
        up_tomorrow = np.zeros(n)
        up_tomorrow[:-1] = (np.diff(c) > 0).astype(float)

        def barely(seed):
            """Mostly a coin flip, with a 6% dose of tomorrow's direction mixed in.

            A clean oracle sits at percentile 100 and a coin flip near the middle, so neither can
            straddle the bar. These land ON it: seeds 5, 7 and 18 come in at 90.0, 91.7 and 93.3,
            seeds 42 and 22 at exactly 95.0 and at 96.7. Move the bar to 0.90 and the first three
            change grade; move it to 0.99 and the last two do. Seed 42 lands on 95.0 itself, which
            is the `>=` boundary and the only place a strict/non-strict slip is visible.
            """
            rng = np.random.default_rng(seed)
            return np.where(rng.random(n) < 0.06, up_tomorrow, (rng.random(n) < 0.5).astype(float))

        seen = []
        for sig in (up_tomorrow, -up_tomorrow, np.ones(n),
                    *(barely(s) for s in (5, 7, 18, 42, 22))):
            t = _battery_test(run_battery(c, _ts(n), sig, bar_ms=DAY, n_random=60),
                              "random_control")
            assert t.reference == 95.0, (
                f"the report tells the user the bar is the {t.reference:.0f}th percentile. Test 3 "
                "exists to say a strategy is clearly above random, and how far above is the whole "
                "content of the claim"
            )
            assert t.passed == (t.value >= t.reference), (
                f"the report prints percentile {t.value:.0f} against a bar of {t.reference:.0f} "
                f"and grades it {t.status!r}: the grade is being decided on a threshold other "
                "than the one shown next to it, so the page contradicts itself"
            )
            seen.append(t.value)

        # The invariant only bites where the fixtures straddle the bar, and closely: a set that all
        # sits at 0 and 100 leaves the threshold free to move anywhere in between.
        assert any(90.0 <= v < 95.0 for v in seen), f"nothing lands just below the bar: {seen}"
        assert any(95.0 <= v < 99.0 for v in seen), f"nothing lands just above the bar: {seen}"
        assert 95.0 in seen, f"nothing lands ON the bar, where `>=` is decided: {seen}"

    def test_the_random_filters_trade_at_the_strategy_s_OWN_frequency(self):
        """★ Test 3 promises controls with «the same frequency AND the same time spent inside the
        market». The exposure half is guarded; the frequency half was not guarded at all.

        The block length is set from the number of times the signal CHANGES, counted as a sum of
        absolute differences. Drop the `abs()` and the differences telescope: a signal that flips
        forty-five times reports one flip, the controls are generated as a single block spanning
        the whole history, and the strategy is then compared against buy-and-hold-shaped filters
        rather than against filters that trade like it does. That is the comparison test 3 exists
        to avoid, inverted, and it moves the percentile by tens of points in whichever direction
        the market happened to go.

        Both signals below hold a position for half the history, so the exposure half of the
        promise is held fixed and only the frequency differs. Under the fault they collapse onto
        the same control population — measured: medians 67.5% and 66.8% against 44.0% and 66.8%
        when the count is right.
        """
        n = 900
        c = 100 * np.exp(np.cumsum(
            np.random.default_rng(3).standard_normal(n) * 0.0002 + 0.002))
        fast = np.where((np.arange(n) // 20) % 2 == 0, 1.0, 0.0)   # in and out every 20 bars
        slow = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])  # in once, and stays in

        def control_median(sig):
            t = _battery_test(run_battery(c, _ts(n), sig, bar_ms=DAY, n_random=200),
                              "random_control")
            # The median is published in the report line: it is what the user is shown the
            # strategy was measured against, so it is the right thing to assert on.
            return float(t.detail.split("median ")[1].split("%")[0])

        f_med, s_med = control_median(fast), control_median(slow)
        assert s_med - f_med > 10.0, (
            f"a signal that enters and exits {n // 20} times was measured against controls with a "
            f"median CAGR of {f_med:.1f}%, and one that enters once against {s_med:.1f}%. These "
            "are within noise of each other, so the control population no longer depends on how "
            "often the strategy trades: it is being compared with slow filters it was never "
            "meant to be compared with"
        )

    def test_one_position_held_from_start_to_finish_is_one_observation(self):
        """Test 5 measures how long the strategy actually stays in, because that is the horizon
        over which its bets overlap — not the `horizon_bars` parameter, which is one bar by
        default everywhere in production.

        Take the parameter instead and a position opened on the first day and never closed is
        reported as 1,199 independent episodes rather than the single bet it is. That is the exact
        input to every p-value the product publishes, and the battery would go on to award its top
        verdict on the strength of one trade.
        """
        n = 1200
        c = 100 * np.exp(np.cumsum(np.random.default_rng(5).standard_normal(n) * 0.02))
        r = run_battery(c, _ts(n), np.ones(n), bar_ms=DAY, n_random=3)
        t = _battery_test(r, "effective_n")
        assert r.n_effective == 1, (
            f"one position, opened once and never closed, is ONE bet; it was counted as "
            f"{r.n_effective} independent episodes out of {r.n_signals} bars")
        assert t.passed is False, "and one observation is not enough to conclude anything"

    @pytest.mark.parametrize("n,hold,gap,episodes,expected", [
        (1200, 10, 100, 24, False),     # comfortably short
        (1200, 10, 40, 60, True),       # comfortably long
        (1500, 8, 107, 29, False),      # ONE episode short of the bar
        (1500, 8, 100, 30, True),       # exactly the bar: `>=`, so it counts
    ])
    def test_thirty_independent_episodes_is_the_bar_and_it_is_thirty(self, n, hold, gap, episodes,
                                                                     expected):
        """The number 30 is printed next to the answer as the `reference` the user is told they
        had to reach, and it is written separately from the number the decision is made against.
        Move one and not the other and the report grades a strategy «pass» on the same line where
        it prints «24 episodes, reference 30» — self-contradictory on its face, and believed,
        because nobody re-does the comparison by eye.

        The first two rows sit at 24 and 60, which pins the bar only to the half-open interval
        (24, 60]: a bar of 29 or of 45 satisfies both, and I verified both survive that pair. The
        last two rows close it. 29 against 30 is the smallest gap the fixture can express, and 30
        itself is where `>=` is decided — the row a strict-inequality slip would fail while every
        comfortable case stayed green.
        """
        c = 100 * np.exp(np.cumsum(np.random.default_rng(5).standard_normal(n) * 0.02))
        sig = np.zeros(n)
        for start in range(0, n, gap):
            sig[start:start + hold] = 1.0                # `hold` bars in, then out until the next
        r = run_battery(c, _ts(n), sig, bar_ms=DAY, n_random=3)
        t = _battery_test(r, "effective_n")
        assert r.n_effective == episodes, (
            f"blocks of {hold} bars every {gap} are {episodes} separate bets; the count came back "
            f"{r.n_effective}")
        assert t.reference == 30.0, f"the bar the user is shown is 30, printed as {t.reference}"
        assert t.passed is expected, (
            f"{episodes} independent episodes against a bar of {t.reference:.0f} has to read "
            f"{'pass' if expected else 'fail'}; it read {t.status!r}")


def _ar_momentum(n=800, seed=7, phi=0.7):
    """A market whose returns are persistent, and the honest rule that trades that persistence.

    It is the only shape of fixture on which the delay test has anything to measure: the rule
    reads YESTERDAY's return, so it beats buy and hold, and it beats it by less when it is
    executed a day late instead — which is the whole quantity test 1 is about. A coin flip and a
    lookahead cheat both sit at an extreme of that scale and can be told apart by the grade alone.
    """
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n) * 0.01
    r = np.empty(n)
    r[0] = eps[0]
    for i in range(1, n):
        r[i] = phi * r[i - 1] + eps[i]
    close = 100 * np.exp(np.cumsum(r))
    sig = np.zeros(n)
    sig[1:] = np.sign(r[:-1])
    return close, sig


class TestTheDelayTestIsADelay:
    """Test 1, whose verdict outranks every other: «IT READS THE FUTURE».

    Three tests already exercise it and all three assert only the final verdict STRING. That is a
    coarse instrument — it moves only when a fault is big enough to carry a strategy across a
    grade boundary on one hard-coded seed — and it leaves the two things the test is actually
    made of unchecked: which signal gets executed, and the retained fraction that decides the
    grade. That fraction (`keeps`) is stored nowhere; it exists only inside an f-string. So it is
    re-derived here from the public fields, because a threshold nothing can read is a threshold
    nothing can defend.
    """

    @staticmethod
    def _keeps(r):
        """The decision variable, reconstructed from what the result actually publishes."""
        return (_battery_test(r, "delay").value - r.cagr_bh) / (r.cagr - r.cagr_bh)

    def test_the_delayed_result_is_the_result_of_the_delayed_signal(self):
        """★ The oracle: running the battery on a signal we shifted ourselves must reproduce,
        exactly, the number the delay test reports for the unshifted one.

        Everything that can go wrong with a one-bar shift is a shift of some other size or
        direction, and each one is a lie in a different direction. Shift the wrong WAY and the
        detector advances the signal instead — clearing a strategy that peeks at tomorrow and
        condemning an honest one, both at once. Shift by two and the test is quietly harsher than
        the one bar it claims. Shift by none and it is not a test. Price the delayed curve without
        the trading costs the raw one paid and the two sides are not comparable at all, because
        the delayed signal has different turnover.

        No assertion about a verdict can see any of that; an equality against an independently
        computed curve sees all four.
        """
        close, sig = _ar_momentum()
        r = run_battery(close, _ts(close.size), sig, bar_ms=DAY, n_random=3)
        by_hand = np.concatenate([[0.0], sig[:-1]])       # the same signal, executed one bar late
        expected = run_battery(close, _ts(close.size), by_hand, bar_ms=DAY, n_random=3).cagr
        assert _battery_test(r, "delay").value == pytest.approx(expected), (
            f"the delay test reports {_battery_test(r, 'delay').value:+.4f} as the CAGR of this "
            f"signal executed one bar later, but running that exact signal gives {expected:+.4f}. "
            f"For reference: undelayed it is {r.cagr:+.4f}")

    def test_a_strategy_that_beats_buy_and_hold_late_as_well_as_early_passes(self):
        """The sign of the retained fraction. `keeps` is a ratio of two excesses over buy and
        hold, and a ratio is the easiest thing in the file to write upside down — nothing about
        the resulting number looks wrong, it is just negative when it should be positive.

        Inverted, the product's most severe and least recoverable message fires on strategies
        that do nothing of the kind: an honest rule that still beats the market a day late is
        told it reads the future, and there is no way for the user to argue with that.
        """
        close, sig = _ar_momentum()
        r = run_battery(close, _ts(close.size), sig, bar_ms=DAY, n_random=3)
        late = _battery_test(r, "delay")
        assert r.cagr > r.cagr_bh and late.value > r.cagr_bh, (
            f"the fixture must beat buy and hold both ways for this to mean anything: "
            f"{r.cagr:+.4f} raw and {late.value:+.4f} delayed against {r.cagr_bh:+.4f}")
        assert self._keeps(r) > 0.5, f"and it must keep most of that excess: {self._keeps(r):.3f}"
        assert late.passed is True, (
            "a rule that is still ahead of buy and hold when executed a day late is not reading "
            f"the future, but it was graded {late.status!r} (keeps {self._keeps(r):.3f})")

    def test_a_strategy_that_loses_to_buy_and_hold_is_still_subject_to_the_test(self):
        """Prices here are flat except for one 40% day, and the rule is long throughout except on
        exactly that day. Its excess over buy and hold is therefore hugely NEGATIVE, and a day
        late it catches the move and matches buy and hold precisely: it keeps none of what it had.

        Wave the negative case through — «there was nothing to lose» — and the detector becomes
        structurally blind to the whole population it most needs to catch: a rule that leaks
        tomorrow's prices and STILL trails a market that rose faster than it did. In a bull run
        that is not an exotic case, it is the common one.
        """
        n = 300
        r_bars = np.zeros(n)
        r_bars[150] = 0.4
        close = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r_bars)]))
        sig = np.ones(n + 1)
        sig[150] = 0.0                                   # sits out the single day that matters
        r = run_battery(close, _ts(close.size), sig, bar_ms=DAY, n_random=3, cost_bps=0)
        assert r.cagr < r.cagr_bh, (
            f"the fixture must UNDERPERFORM buy and hold: {r.cagr:+.4f} vs {r.cagr_bh:+.4f}")
        assert self._keeps(r) < 0.5, (
            f"and must keep less than half its excess when delayed: {self._keeps(r):.3f}")
        assert _battery_test(r, "delay").passed is False, (
            "a strategy whose timing edge evaporates on a one-bar delay has to fail the delay "
            "test whether or not it was beating buy and hold to begin with; it was graded "
            f"{_battery_test(r, 'delay').status!r}")

    def test_the_bar_is_half_the_excess_and_not_a_quarter_of_it(self):
        """Where the line sits. This rule avoids two consecutive crash days; delayed, it avoids
        only the second, so it retains about two fifths of its excess — deliberately between a
        quarter and a half, which is the one band where moving the threshold shows up.

        A relaxed bar does not remove the test from the report. It leaves it there, still printing
        its number, still saying «pass», and meaning nothing — which is worse than not running it,
        because the user reads the pass.
        """
        n = 300
        r_bars = np.zeros(n)
        r_bars[10] = 0.02
        r_bars[150] = r_bars[151] = -0.30                # two crash days in a row
        close = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r_bars)]))
        sig = np.ones(n + 1)
        sig[150] = sig[151] = 0.0                        # flat for both of them
        r = run_battery(close, _ts(close.size), sig, bar_ms=DAY, n_random=3, cost_bps=0)
        keeps = self._keeps(r)
        assert r.cagr > r.cagr_bh, "the fixture must have a positive excess to retain"
        assert 0.25 < keeps < 0.5, (
            f"this fixture is built to land between a quarter and a half so that the threshold "
            f"is what decides it; it retained {keeps:.3f} and the test no longer straddles a bar")
        assert _battery_test(r, "delay").passed is False, (
            f"retaining {keeps:.0%} of its excess is under a half and must FAIL; it was graded "
            f"{_battery_test(r, 'delay').status!r}, so the bar is not where the report says")


#: 1,200 bars of hard drift and almost no noise, so every bar's forward move is essentially the
#: same number and the base-rate arithmetic below is exact rather than statistical.
RISING_N = 1200


@pytest.fixture(scope="module")
def rising():
    rng = np.random.default_rng(3)
    return 100 * np.exp(np.cumsum(rng.standard_normal(RISING_N) * 0.0002 + 0.002))


class TestTheBaseRateIsARelativeQuestion:
    """Test 2, and the reason the whole product exists: «right a lot» is not «right more often».

    In an asset that rose 1,748% any mostly-long rule is right most of the time. The comparison
    against the base rate is the only thing standing between that fact and a report congratulating
    the user on having discovered the market. `evaluate.py`'s equivalent guard is tested and
    sharp; the battery's — the one the server actually calls — had no test at all.

    The fixture drifts hard and is nearly noiseless, so every bar's forward move is essentially the
    same number. That makes the arithmetic below exact rather than statistical: an always-long rule
    IS the base rate, by construction, and any answer other than «no edge» is the bug.
    """

    @staticmethod
    def _base_rate(rising, sig):
        return _battery_test(
            run_battery(rising, _ts(RISING_N), sig, bar_ms=DAY, n_random=3), "base_rate")

    def test_the_forward_move_a_signal_is_graded_on_is_a_known_number(self):
        """★ The one assertion that pins the forward window itself, and nothing else did.

        Everything else in this class compares the strategy's score against the base rate, and on
        an always-long rule those two are the SAME arithmetic mean of the SAME array — an identity
        of the fixture, true for any `fwd` whatsoever. Measured: with no other change to the suite,
        `fwd` can be multiplied by seven, or reversed end to end, or shifted a bar earlier, and all
        544 tests stay green. The block that answers the question this whole product exists to ask
        was pinned only in the sign of one comparison.

        So this fixture makes the answer a constant known outside the code. Returns cycle
        0%, 1%, 2%, 3%, 4% and the rule fires only on the bars where the NEXT return is the 1% one.
        Its score is therefore exactly 1% per period, and nothing else: shift the window a bar and
        it reads 0%, scale it and it reads a multiple, reverse it and it reads the cycle backwards.

        It also writes the convention down, which nothing did. `fwd[i]` is the return from close
        `i+1` to close `i+1+h`, so a signal on bar `i` is graded on the move AFTER the bar it
        trades — a fill at the next open, not the close it was computed from. That is a deliberate
        choice and it is not the one `battery.py`'s module docstring describes; whoever changes
        either must now change this number too.
        """
        n = 200
        moves = 0.01 * (np.arange(n) % 5)                    # 0%, 1%, 2%, 3%, 4%, 0%, ...
        close = np.exp(np.concatenate([[np.log(100.0)], np.log(100.0) + np.cumsum(moves)]))
        # Fire wherever the FOLLOWING bar is a 1% bar. `fwd` has no value for the last bar.
        sig = np.zeros(n)
        fires = [i for i in range(n - 1) if i % 5 == 0]
        sig[fires] = 1.0
        assert len(fires) >= 20, "below the min-signal floor the battery would refuse to answer"

        t = _battery_test(
            run_battery(close, _ts(n + 1), sig, bar_ms=DAY, n_random=3, horizon_bars=1),
            "base_rate")

        assert t.value == pytest.approx(1.0, abs=1e-9), (
            f"the rule fires only on bars whose forward move is exactly +1%, so it scores "
            f"+1.000% per period. It scored {t.value:+.4f}%. At 0% the window is a bar early and "
            f"every signal is graded on the move it was computed FROM; a multiple means the "
            f"returns are scaled; anything else means the window is not the one described above"
        )
        assert t.reference == pytest.approx(100 * float(np.mean(moves[1:n])), abs=1e-9), (
            f"the base rate is the average forward move of this market ({t.reference:+.4f}%), "
            "and it is computed over every bar that has one, not over the bars the rule chose"
        )
        assert t.value < t.reference and t.passed is False, (
            "this rule deliberately picks the market's FLATTEST bars, so it must be graded as "
            f"having no edge; it was graded {t.status!r}"
        )

    def test_being_long_all_the_time_is_graded_as_no_edge(self, rising):
        """The regression the module's own docstring is written about. Compare the strategy
        against zero instead of against the base rate and a rule that does nothing but hold the
        market passes — while the report prints the two numbers side by side, IDENTICAL, above
        the word «pass». The evidence against the verdict is in the same paragraph as the verdict.

        Note what is and is not evidence here. `value == reference` is an ALGEBRAIC IDENTITY of an
        always-long rule — `m` is the whole valid mask, so both sides average the same array — and
        it holds for every implementation, including a badly broken one. It is asserted below as a
        precondition, not as a finding. The load-bearing line is the grade.
        """
        t = self._base_rate(rising, np.ones(RISING_N))
        assert t.reference > 0, "the fixture has to actually drift upward for this to mean anything"
        assert t.value == pytest.approx(t.reference), (
            "precondition, not a property: an always-long rule averages the same bars the base "
            f"rate does. It scored {t.value:+.4f} against a base of {t.reference:+.4f}")
        assert t.passed is False, (
            f"scoring exactly the base rate is no edge at all, but it was graded {t.status!r}")

    def test_a_permanent_short_earns_the_base_rate_with_the_sign_it_took(self, rising):
        """The sign of the position is what turns «what the market did» into «what the strategy
        earned». Drop it and a permanent short in a bull market scores identically to a permanent
        long — the test stops being able to tell the two apart, which is the one distinction it
        has to make before any other."""
        t = self._base_rate(rising, -np.ones(RISING_N))
        assert t.value == pytest.approx(-t.reference), (
            f"being short a market that rose {t.reference:+.4f} per period earns "
            f"{-t.reference:+.4f}, not {t.value:+.4f}")
        assert t.passed is False, "and losing money steadily is not beating the base rate"

    def test_the_base_rate_is_a_fact_about_the_market_not_about_the_strategy(self, rising):
        """It is «being in the market at a random instant», so it is the same number whoever is
        asking. Compute it over the bars the strategy chose instead and it becomes the strategy's
        own score by construction: the two sides of the comparison collapse into one, every
        strategy ties itself, and test 2 turns into a coin toss on floating-point noise while
        still printing two convincing-looking figures.
        """
        every_fourth = np.zeros(RISING_N)
        every_fourth[::4] = 1.0
        whole_market = self._base_rate(rising, np.ones(RISING_N)).reference
        picky = self._base_rate(rising, every_fourth).reference
        assert picky == pytest.approx(whole_market), (
            "the base rate is the average bar of this market and cannot depend on which bars a "
            f"strategy traded: {picky:+.6f} for a rule firing on a quarter of them against "
            f"{whole_market:+.6f} for one firing on all of them")

    def test_a_rule_is_scored_on_the_bars_it_fires_on_not_on_the_whole_history(self, rising):
        """Two rules, one firing twice as often as the other, both right by the same margin when
        they do fire. Their scores must match. Average over every bar instead — counting the bars
        the rule sat out as zeroes — and the score is diluted in proportion to how selective the
        rule is: a good rule that trades rarely is punished for its patience, and the more
        selective it is the worse it looks.
        """
        half, quarter = np.zeros(RISING_N), np.zeros(RISING_N)
        half[::2] = 1.0
        quarter[::4] = 1.0
        wide, narrow = self._base_rate(rising, half), self._base_rate(rising, quarter)
        assert narrow.value == pytest.approx(wide.value, rel=0.05), (
            "these two rules are right by the same margin and differ only in how often they "
            f"fire, so they must score the same: {narrow.value:+.4f} on 300 bars against "
            f"{wide.value:+.4f} on 600 — a ratio of {narrow.value / wide.value:.2f}")

    def test_the_same_market_grades_a_timed_rule_and_a_mistimed_one_differently(self):
        """The other direction, stated as a CONTRAST because one direction alone proves nothing.

        This test used to run one rule on `_ar_momentum` and assert it passed. That fixture falls,
        so its base rate is −0.346% per period and anything averaging near zero clears it: I ran
        five independent coin flips and a permanent short through it, and all six passed. It
        demonstrated that anything passes on a falling market, not that a real edge passes, and
        its only mutant kill was already covered alone and more sharply by the permanent-short
        test above.

        Both rules here trade the same market and the same number of bars. One fires where the
        forward move is the cycle's best (+4%), the other where it is +1%, against a base rate of
        +2%. Same history, same count of signals, opposite grades — so a pass is evidence of
        timing rather than of drift, and the failure is evidence the bar is real.

        The mistimed rule deliberately MAKES MONEY, just less than the market does. That is the
        case the base rate exists for, and the one a comparison against zero waves through: graded
        against 0% it is a winner, graded against the market it is a way of underperforming while
        paying to trade.
        """
        n = 200
        moves = 0.01 * (np.arange(n) % 5)
        close = np.exp(np.concatenate([[np.log(100.0)], np.log(100.0) + np.cumsum(moves)]))

        def graded(phase):
            sig = np.zeros(n)
            sig[[i for i in range(n - 1) if i % 5 == phase]] = 1.0
            return _battery_test(
                run_battery(close, _ts(n + 1), sig, bar_ms=DAY, n_random=3, horizon_bars=1),
                "base_rate")

        timed, mistimed = graded(3), graded(0)     # next move +4% vs next move +1%
        base = 100 * float(np.mean(moves[1:n]))

        assert timed.value == pytest.approx(4.0, abs=1e-9) and timed.reference == pytest.approx(base)
        assert mistimed.value == pytest.approx(1.0, abs=1e-9)
        assert timed.passed is True, (
            f"a rule that fires only on the market's best bars scored {timed.value:+.3f}% against "
            f"a base of {base:+.3f}% and was still graded {timed.status!r}: test 2 can only ever "
            "fail, which is not conservatism but a broken gate that condemns every real edge"
        )
        assert mistimed.passed is False, (
            f"a rule that makes money on every trade but less than the market does scored "
            f"{mistimed.value:+.3f}% against {base:+.3f}% and was graded {mistimed.status!r}"
        )

    def test_too_few_signals_is_inconclusive_rather_than_a_verdict(self, rising):
        """A handful of signals cannot establish anything, and saying so is not the same as
        failing. Folded into a failure it becomes the second black mark that tips «doubtful» into
        «does not survive», on evidence that was never gathered."""
        rare = np.zeros(RISING_N)
        rare[::150] = 1.0                                # eight signals in nine hundred bars
        t = self._base_rate(rising, rare)
        assert t.passed is None, (
            f"eight signals is far too few to judge a base rate, but it was graded {t.status!r}")
        assert "few" in t.detail, "and the report has to say why there is no answer"


OOS_N = 1200


@pytest.fixture(scope="module")
def oos_market():
    rng = np.random.default_rng(3)
    return 100 * np.exp(np.cumsum(rng.standard_normal(OOS_N) * 0.02))


class TestTheBatteryHoldsBackItsOwnOutOfSample:
    """Test 4, and the place where the project's discipline is written down twice.

    `evaluate._walk_forward_mask` is guarded by two sharp tests — drop the purge or include the
    training fold and they go red on the spot. The battery carries its own copy of the same nine
    lines, and neither fault was visible there at all. The battery is the code path the server
    calls; `evaluate_all` is the offline sweep. The tested copy was the one nobody sees.

    Both tests below use a signal that holds NO position anywhere it should be measured, so the
    out-of-sample return has an exact answer — zero — and any leak turns it into something else.
    """

    FOLDS, HORIZON = 5, 10
    #: `run_battery` folds the RETURN series, which is one shorter than the price series.
    STEP = (OOS_N - 1) // FOLDS

    @classmethod
    def _oos(cls, close, sig):
        r = run_battery(close, _ts(OOS_N), sig, bar_ms=DAY, n_random=3,
                        horizon_bars=cls.HORIZON, folds=cls.FOLDS)
        return _battery_test(r, "out_of_sample")

    def test_the_training_fold_is_never_measured_as_out_of_sample(self, oos_market):
        """The first fifth of the history is what the idea was allowed to be found in. Let it back
        into the out-of-sample figure and «it also works out of sample» becomes a statement about
        the very data the strategy was fitted to — the disease this test is the cure for, being
        reported as the cure.
        """
        train_only = np.zeros(OOS_N)
        train_only[:self.STEP] = 1.0                          # fires ONLY in the held-back first fold
        t = self._oos(oos_market, train_only)
        assert t.value == pytest.approx(0.0, abs=1e-12), (
            "this rule holds nothing outside the training fold, so its out-of-sample return is "
            f"exactly zero; it was measured at {t.value:+.4f}, which can only have come from "
            "inside the fold it was fitted in")

    def test_the_horizon_is_purged_at_every_fold_boundary(self, oos_market):
        """A trade opened in the last bars of one segment is still open when the next begins, so
        its outcome was decided by data from the segment it is being counted against. This rule
        fires ONLY inside those purge windows: purged, it contributes nothing and scores zero;
        unpurged, every one of its bars leaks straight into the out-of-sample number. The leak is
        small at each boundary, happens at all of them, and always flatters.
        """
        in_the_purge = np.zeros(OOS_N)
        for k in range(1, self.FOLDS):
            in_the_purge[k * self.STEP: k * self.STEP + self.HORIZON] = 1.0
        t = self._oos(oos_market, in_the_purge)
        assert t.value == pytest.approx(0.0, abs=1e-12), (
            f"all {(self.FOLDS - 1) * self.HORIZON} of this rule's bars sit inside a purge window "
            f"and must be excluded, yet the out-of-sample CAGR came back as {t.value:+.4f}")

    @pytest.mark.parametrize("in_sample,out_of_sample,expected", [(3, 1, False), (1, 3, True)])
    def test_half_the_edge_has_to_survive_and_half_means_half(self, in_sample, out_of_sample,
                                                              expected):
        """What the out-of-sample test decides on is the FRACTION of the in-sample excess that
        holds up outside — a ratio that, like the delay test's, is never stored and can only be
        rebuilt from the published CAGRs. Two things go wrong with a ratio: it gets written the
        other way up, and its threshold drifts.

        Both are invisible without a fixture that straddles the bar, so this one does: a market
        with four crash days that the strategy sits out, dealt three-to-one either side of the
        training fold. Take them the wrong way round and «kept 35% of it» reads as 289% and the
        strategy is congratulated for the very thing that condemns it — an edge that lived almost
        entirely in the segment it was found in.
        """
        n = 1200
        step = (n - 1) // self.FOLDS
        # crash days, dealt between the held-back training fold and the out-of-sample segments
        crashes = ([int(step * 0.5 * (k + 1) / in_sample) for k in range(in_sample)]
                   + [int(step * (1.5 + 3.0 * k / out_of_sample)) for k in range(out_of_sample)])
        moves = np.zeros(n)
        moves[5] = 0.01
        for i in crashes:
            moves[i] = -0.25
        close = 100 * np.exp(np.concatenate([[0.0], np.cumsum(moves)]))
        sig = np.ones(n + 1)
        for i in crashes:
            sig[i] = 0.0                                 # flat on exactly the days that crash

        r = run_battery(close, _ts(n + 1), sig, bar_ms=DAY, n_random=3, cost_bps=0,
                        horizon_bars=1, folds=self.FOLDS)
        t = _battery_test(r, "out_of_sample")
        kept = (t.value - t.reference) / (r.cagr - r.cagr_bh)
        assert r.cagr > r.cagr_bh, "the fixture must have an in-sample edge to hold up or lose"
        assert (kept < 0.5) is not expected, (
            f"this fixture is built to land {'below' if not expected else 'above'} half; it kept "
            f"{kept:.3f}, so it no longer straddles the bar and the test proves nothing")
        assert t.passed is expected, (
            f"keeping {kept:.0%} of the in-sample excess out of sample has to read "
            f"{'pass' if expected else 'fail'} against a bar of half; it read {t.status!r}")


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

    def test_a_strategy_that_did_nothing_is_never_significant(self):
        """The p-value counts the resampled maxima that reach the observed one, and whether that
        count INCLUDES the ties is invisible on continuous data — every test above compares
        floats that never land on each other.

        The tie is reachable, and it is the degenerate case: a set of strategies whose returns are
        identically zero — a rule that never fired, a CSV parsed into «flat forever», a signal
        column read as the wrong unit. Then every bootstrap maximum is 0.0 and so is the observed
        one. Exclude the ties and the p-value is 0.0000: the correction declares a strategy that
        never took a position significant at the strongest possible level, in exactly the case the
        user is least equipped to notice.
        """
        r = reality_check({"never_fired": np.zeros(200), "also_flat": np.zeros(200)}, n_boot=200)
        assert r.p_value == 1.0, (
            "a strategy that returned exactly nothing is exactly as good as every resample of "
            f"itself, so its p-value is 1.0, not {r.p_value:.4f}")
        assert not r.significant, "and it certainly does not beat a multiple-testing correction"


# ---------------------------------------------------------------- out-of-sample discipline

def _rising_series(n=1000, tf="1d", bar_ms=DAY, seed=9):
    """A series with a clear upward drift, which is the condition every one of these guards
    exists to survive: in a rising market, doing nothing clever already looks clever."""
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01 + 0.002))
    ts = np.array([BASE + i * bar_ms for i in range(n)])
    return Series(tf, ts, c, c, c, c, np.ones(n))


def _hyp(name, fn, timeframes=("1d",)):
    return Hypothesis(name, "test", "a test hypothesis", "declared up front", "declared up front",
                      fn, timeframes=timeframes, min_warmup=0)


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

    def _run(self, fn, name="h", series=None, folds=None, **kw):
        res, _ = evaluate_all({name: _hyp(name, fn)},
                              {"1d": _rising_series() if series is None else series},
                              horizon_bars={"1d": self.HORIZON}, folds=folds or self.FOLDS,
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

    @pytest.mark.parametrize("n_oos,answers", [(19, False), (20, True)])
    def test_an_out_of_sample_number_needs_twenty_out_of_sample_signals(self, n_oos, answers):
        """`oos_edge` is the number that decides whether an idea survived where nobody was
        looking, and it refuses to be computed from fewer than twenty observations.

        The floor is separate from `min_signals` and does a different job: a hypothesis can fire
        hundreds of times inside the training fold and land only a handful outside it. Lower the
        floor and the report publishes an out-of-sample edge estimated off five trades — a number
        with a confidence interval wider than any effect it could show, printed in the same column
        and the same font as one built on hundreds, and read as the same kind of evidence. The
        honest output is NaN, which the report renders as «not enough data»; a wrong number is
        strictly worse than no number, and this floor is the only thing between them.

        Both sides are pinned. Nineteen must refuse, twenty must answer.
        """
        oos_at = 300      # inside the first out-of-sample segment [210, 400)

        def mostly_in_sample(s):
            out = np.zeros(len(s), dtype=np.int8)
            out[:200] = 1                              # the whole training fold: not out of sample
            out[oos_at: oos_at + n_oos] = 1            # and a handful of genuinely OOS bars
            return out

        r = self._run(mostly_in_sample, min_signals=20)
        assert r.n_signals == 200 + n_oos, f"the fixture fired {r.n_signals} times"
        # `bool(...)`, not `is`: np.isnan returns np.bool_, which is never `is` a Python bool, so
        # `np.isnan(x) is not answers` is true for every input and asserts nothing at all.
        assert bool(np.isnan(r.oos_edge)) is not answers, (
            f"{n_oos} out-of-sample signals: oos_edge came back {r.oos_edge}. Nineteen "
            "observations cannot support an out-of-sample claim and the field has to stay empty; "
            "twenty is the declared floor and has to produce an answer"
        )

    @pytest.mark.parametrize("n_signals,evaluated", [(49, False), (50, True)])
    def test_a_hypothesis_with_exactly_the_minimum_number_of_signals_is_evaluated(
            self, n_signals, evaluated):
        """`min_signals` is the floor, and a floor includes its own value.

        Off by one here is quiet in the direction that matters least on any single run and most
        across a sweep: a hypothesis sitting exactly on the floor is dropped from the results
        list entirely, so it never reaches the Reality Check either. The multiple-testing
        correction is computed over the strategies that survived to be counted, so silently
        dropping one makes every other p-value slightly better — a correction that gets weaker
        the more borderline ideas you tried, which is backwards.
        """
        def fires_exactly(s):
            out = np.zeros(len(s), dtype=np.int8)
            out[300: 300 + n_signals] = 1
            return out

        res, _ = evaluate_all({"h": _hyp("h", fires_exactly)}, {"1d": _rising_series()},
                              horizon_bars={"1d": self.HORIZON}, folds=self.FOLDS, n_boot=0,
                              min_signals=50)
        assert bool(res) is evaluated, (
            f"{n_signals} signals against a minimum of 50 produced {len(res)} results. The "
            "minimum is a floor to REACH: at exactly 50 the hypothesis is evaluated, at 49 it is "
            "not"
        )
        if evaluated:
            assert res[0].n_signals == n_signals

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

    def test_a_one_bar_leak_at_the_boundary_is_caught_and_not_just_a_total_one(self):
        """The purge test above cannot see a purge that is one bar too SHORT. With five folds it
        leaks four bars, which is under the twenty-signal floor `evaluate_all` needs before it
        reports anything, so `oos_edge` stays NaN whether the boundary leaks or not — the test
        detects the total absence of purging and nothing weaker.

        Twenty-five folds is the same fault with twenty-four leaked bars, which clears the floor.
        A number appearing where a NaN belongs is the whole failure: it is built entirely out of
        trades whose outcomes were decided inside the segment they are being scored against.
        """
        folds, step = 25, 1000 // 25

        def only_in_the_purge_zones(s):
            out = np.zeros(len(s), dtype=np.int8)
            for k in range(1, folds):
                out[k * step: k * step + self.HORIZON] = 1
            return out

        r = self._run(only_in_the_purge_zones, folds=folds, min_signals=20)
        assert r.n_signals == (folds - 1) * self.HORIZON, "10 signals at each of 24 boundaries"
        assert np.isnan(r.oos_edge), (
            "every one of these signals sits inside a purge window, including the very last bar "
            f"of each one, so none may be scored out of sample; oos_edge came back {r.oos_edge}")

    def test_an_always_long_rule_is_in_the_market_all_of_the_time_and_ahead_by_none_of_it(self):
        """Two more consequences of scoring against the drift, both exactly computable and neither
        asserted anywhere. `exposure` is the fraction of the SCORABLE bars a rule is in for, so a
        rule that is always long is at 1.0; denominate it by the whole series instead and it comes
        out just under, which is the sort of number nobody questions. And `oos_edge` is the same
        excess-over-base as `edge`, so it is zero for an always-long rule out of sample too —
        net it against the whole sample's base rate rather than the out-of-sample one and it stops
        being zero, which is the first step back towards paying for beta.
        """
        r = self._run(lambda s: np.ones(len(s), dtype=np.int8))
        assert r.exposure == 1.0, (
            f"a rule that is long on every scorable bar has an exposure of exactly 1.0, not "
            f"{r.exposure:.4f}")
        assert r.oos_edge == pytest.approx(0.0, abs=1e-9), (
            "out of sample as in sample, an always-long rule has no answer to «did THIS moment "
            f"beat any old moment?» and must score 0, not {r.oos_edge:+.6f}")

    @pytest.mark.parametrize("tf,bar_ms", [("1d", DAY), ("4h", 14_400_000)])
    def test_overlapping_signals_are_counted_in_the_bars_own_units(self, tf, bar_ms):
        """`effective_n` is handed a horizon in BARS and a bar width in milliseconds, and it is
        the bar width that `evaluate_all` derives itself from the timestamps. Get that unit wrong
        by a factor of a thousand and the overlap window shrinks to nothing: every signal becomes
        an independent observation, so an always-long rule reports 990 of them instead of 99.

        Test 5's entire purpose evaporates and every p-value downstream is wrong by orders of
        magnitude, silently and always in the flattering direction. Running it at two different
        bar widths is what pins the unit rather than the constant — 1d and 4h are a factor of six
        apart and must give the same count, because the horizon is quoted in bars.
        """
        n, horizon = 1000, self.HORIZON
        res, _ = evaluate_all({"h": _hyp("h", lambda s: np.ones(len(s), dtype=np.int8), (tf,))},
                              {tf: _rising_series(n, tf, bar_ms, seed=10)},
                              horizon_bars={tf: horizon}, folds=self.FOLDS, n_boot=0)
        r = res[0]
        scorable = n - horizon
        assert r.n_signals == scorable, "an always-long rule fires on every bar that has a future"
        assert r.n_effective == -(-scorable // horizon), (
            f"{scorable} consecutive signals over a {horizon}-bar horizon are "
            f"{-(-scorable // horizon)} independent episodes, not {r.n_effective}")

    def test_a_market_that_went_nowhere_produces_no_hits(self):
        """A hit is a trade that made money. On a perfectly flat market every forward return is
        exactly zero, so there are no hits — and counting a flat outcome as a hit is not a
        rounding difference, it reports a strategy that made nothing as never having been wrong.

        Exact zeroes are not the measure-zero curiosity they look like: a halted market, a stale
        feed repeating its last price, or a price column parsed into a constant all produce them,
        and those are precisely the cases where a 100% hit rate would be believed.
        """
        n = 1000
        flat = np.full(n, 100.0)
        s = Series("1d", np.array([BASE + i * DAY for i in range(n)]),
                   flat, flat, flat, flat, np.ones(n))
        r = self._run(lambda x: np.ones(len(x), dtype=np.int8), series=s)
        assert r.hit_rate == 0.0, (
            f"nothing moved, so nothing was won: hit_rate reads {r.hit_rate:.2f}")
        assert r.hit_rate == r.base_rate, (
            "and the strategy's hit rate and the market's must be counted by the same rule, or "
            f"they cannot be compared: {r.hit_rate:.2f} against {r.base_rate:.2f}")

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

    def test_the_bars_before_the_first_signal_are_flat(self):
        """A `ffill` has nothing to carry backwards, so every bar before the user's first row
        comes out of the reindex as NaN — and the very next line casts to int8.

        NaN→int8 is UNDEFINED in NumPy. It does not raise; it emits a RuntimeWarning and produces
        whatever the platform's conversion happens to yield, which on this machine is −128: a
        position 128 times the size of anything the user ever asked for, held over exactly the
        stretch of history they said nothing about. Nothing downstream can question it, because
        int8 is the declared type and −128 is a legal value in it.

        This is the only path into that cast, and no test reached it: every existing fixture
        starts its bar grid at or after the first signal. The suite now treats a NumPy
        RuntimeWarning as a failure (see `filterwarnings` in pyproject.toml), so this test fails
        twice over if the fill is dropped — on the value and on the warning.
        """
        imp = parse_signals_csv("date,signal\n2020-09-15,largo\n2020-09-17,fuera\n")
        bars = np.array([BASE + i * DAY for i in range(6)])   # BASE is 2020-09-13
        aligned = align_to_bars(imp, bars, DAY)

        assert aligned.dtype == np.int8
        assert aligned.tolist() == [0, 0, 1, 1, 0, 0], (
            f"got {aligned.tolist()}. Before the first row the user has expressed no opinion, so "
            "the only honest position is flat; anything else is a trade they never took, sized by "
            "an undefined float-to-int conversion")

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

    #: The same instant, written the four ways an export actually arrives: seconds (unix `date`),
    #: milliseconds (JavaScript, our own store), microseconds (a Postgres or DuckDB dump),
    #: nanoseconds (a pandas `int64` datetime column, which is what `df.to_csv()` writes).
    @pytest.mark.parametrize("epoch,unit", [
        (1_704_067_200, "SECONDS"),
        (1_704_067_200_000, "milliseconds"),
        (1_704_067_200_000_000, "MICRO"),
        (1_704_067_200_000_000_000, "NANO"),
    ])
    def test_every_epoch_unit_lands_on_the_same_instant(self, epoch, unit):
        """Only the seconds branch was covered, and it is the one people think of. The three
        others are each a factor of 1,000 away from the truth, which does not look like a small
        error on a chart: it puts 2024 in 1970, or in the year 55000. The signal then aligns to
        no bar we hold and the strategy silently becomes «flat forever».
        """
        imp = parse_signals_csv(f"time,signal\n{epoch},1\n{epoch + 1},0\n")
        assert imp.ts_ms[0] == 1_704_067_200_000, (
            f"an epoch of {epoch} is 2024-01-01 written in {unit.lower()}, which is "
            f"1704067200000 ms; it was read as {imp.ts_ms[0]}")
        assert any(unit in line for line in imp.report), (
            f"the unit is a GUESS and has to be reported as {unit}: {imp.report}")

    def test_a_millisecond_epoch_from_the_1990s_is_not_mistaken_for_seconds(self):
        """The ms/seconds boundary is a magnitude test, so it has a low side as well as a high
        one. Set it a decade too high and every millisecond timestamp before ~2001 is multiplied
        by a thousand — a backtest starting in 1990 is silently relocated to the year 21,000,
        which is past every bar in existence and therefore, again, «flat forever»."""
        imp = parse_signals_csv("time,signal\n631152000000,1\n631152000001,0\n")
        assert imp.ts_ms[0] == 631_152_000_000, (
            "631152000000 ms is 1990-01-01 and must stay there; it was read as "
            f"{imp.ts_ms[0]}, which is {'far later' if imp.ts_ms[0] > 1e12 else 'far earlier'}")


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

    def test_the_later_row_still_wins_on_a_file_big_enough_to_need_a_stable_sort(self):
        """«Keep the last» is implemented as «keep the last position AFTER sorting», so it is only
        the same thing as «keep the last ROW» while the sort preserves the order of equal
        timestamps. The three-row file above cannot tell: numpy's introsort falls back to
        insertion sort at that size and is stable by accident.

        At a realistic size it is not. Drop the `kind="stable"` and equal timestamps come back in
        an arbitrary order, so on a file of a few hundred rows an arbitrary one of the user's
        duplicates wins — deterministically, invisibly, and never on any test small enough to
        read. Taking the wrong one means trading the opposite of what they asked for.
        """
        rows = []
        for day in range(40):
            for k in range(5):            # five rows per timestamp; only the LAST says «corto»
                rows.append(f"{1_704_067_200 + day * 86_400},{'largo' if k < 4 else 'corto'}")
        r = parse_signals_csv("time,signal\n" + "\n".join(rows) + "\n")
        assert len(r.ts_ms) == 40, (
            f"40 distinct timestamps must collapse to 40 rows, got {len(r.ts_ms)}")
        assert (r.signal == -1).all(), (
            "the last row written for every timestamp says «corto», so every surviving signal is "
            f"short; {int((r.signal != -1).sum())} of 40 kept an earlier row instead")


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

    def test_each_price_stays_with_the_row_it_arrived_on(self):
        """Both tests above use two rows already in date order, so the sort that reorders the
        signals is the identity and cannot be seen. Spreadsheets exported newest-first are
        ordinary, and there the signals get sorted into time order while the prices do not have
        to be — pairing March's price with January's short.

        Nothing raises and nothing looks wrong: the equity curve is computed from a coherent
        price series and a coherent signal series that describe different days. It is a wrong
        backtest with plausible numbers, which is the only kind that gets believed.
        """
        r = parse_signals_csv(
            "date,signal,close\n"
            "2024-03-01,largo,300\n"
            "2024-01-01,corto,100\n"
            "2024-02-01,fuera,200\n")
        assert r.signal.tolist() == [-1, 0, 1], "the signals sort into January, February, March"
        assert r.price.tolist() == [100.0, 200.0, 300.0], (
            "and each price must travel with its own row: January is 100 and March is 300, not "
            f"{r.price.tolist()}")

    @pytest.mark.parametrize("n_rows,n_bad,usable", [
        (9, 3, False),      # 67% readable: well under the bar
        (20, 3, False),     # 85%: under it, and above a bar that had slipped to 0.7
        (20, 1, True),      # 95%: over it
        (20, 2, True),      # 90% exactly — the bar itself, and `>=` means it counts
    ])
    def test_the_price_column_has_to_be_ninety_percent_readable(self, n_rows, n_bad, usable):
        """The bar is 90% parseable, and it exists because a column can be a price column by name
        and a comment column in fact. Half-readable is not a price series: the holes become NaN,
        every return that touches one is NaN, and the report is computed from whatever is left
        without saying which days it silently skipped.

        This used to be a single 67%-readable file, which pins the bar only to somewhere in
        (0.667, 1.0] — I verified a bar of 0.7 survives it, so the documented 90% could slide to
        70% with nothing going red, and the docstring would still say 90%. The rows above bracket
        it from both sides and land one case exactly ON it, which is where `>=` is decided.
        """
        rows = "".join(
            f"2024-01-{i + 1:02d},largo,{'n/a' if i < n_bad else 100 + i}\n" for i in range(n_rows))
        r = parse_signals_csv("date,signal,close\n" + rows)
        readable = (n_rows - n_bad) / n_rows
        assert r.has_prices is usable, (
            f"{n_rows - n_bad} of {n_rows} prices are readable ({readable:.0%}) and the column "
            f"was {'accepted' if r.has_prices else 'refused'}. The declared bar is 90%"
        )
        if not usable:
            assert any("discarded" in line for line in r.report), (
                "and the user has to be told their own prices were dropped")

    def test_a_numeric_time_column_too_small_to_be_an_epoch_is_refused(self):
        """The bottom rung of the unit ladder, which nothing reached.

        The four unit branches are each pinned and so is the ms/seconds boundary from the low
        side, but not the point below which a numeric column stops being a timestamp at all. Below
        it the honest answer is a question — «is this the right column?» — because a column of
        small integers is a row index, a trade id or a bar number far more often than it is a
        date. Read as seconds instead, an index of 1, 2, 3 becomes January 1970 and every signal
        aligns to no bar we hold: the strategy comes out flat and correct-looking.
        """
        with pytest.raises(ImportError_, match="neither a timestamp nor a date"):
            parse_signals_csv("time,signal\n1,1\n2,0\n3,1\n")
        with pytest.raises(ImportError_, match="neither a timestamp nor a date"):
            parse_signals_csv("time,signal\n50000000,1\n50000001,0\n")   # 5e7, just under the rung

        # And immediately above it the column IS an epoch in seconds, so the boundary cuts
        # somewhere between: 2e8 is 1976, an implausible but well-formed unix timestamp.
        imp = parse_signals_csv("time,signal\n200000000,1\n200000001,0\n")
        assert imp.ts_ms[0] == 200_000_000_000 and any("SECONDS" in x for x in imp.report)


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

    def test_every_observation_can_START_a_block_including_the_very_last_one(self):
        """Draw the block starts from `integers(0, n-1)` and the final observation never begins a
        block, so the resample is systematically short of the END of the series — which for a
        returns series is the most recent data, the part a user is actually betting on. The null
        distribution then gets built out of a sample that leans on the older regime, and the
        correction answers a question about last year.

        WHY ONE RESAMPLE CANNOT SHOW THIS, and why this test is not flaky. A single 50-index
        resample draws only ~11 starts and so covers about a fifth of the positions; asserting
        coverage from one draw would fail at random. This draws MANY resamples from one fixed
        seed, which makes the union of their starts a fixed set of integers, not a random one: it
        either reaches index 49 on every run forever, or it never does.

        THE COUNT IS LOAD-BEARING. The unmutated bootstrap first covers all 50 positions after 23
        resamples with this seed; 120 is a 5x margin. Do not trim it to speed the suite up — at
        6,000 loop iterations it is already invisible in the runtime, and the margin is what
        stops a legitimate change to the draw order from turning this into a flaky gate.
        """
        n, draws = 50, 120
        rng = np.random.default_rng(0)
        started_at: set[int] = set()

        for _ in range(draws):
            idx = stationary_bootstrap_indices(n, 5.0, rng)
            # STARTS, not visits. A continuation is always exactly (previous + 1) % n, so the wrap
            # makes index n-1 appear in almost every resample even when nothing ever starts there
            # — assert on `idx` itself and the whole bias is invisible. Any index that is NOT the
            # continuation is certainly a fresh draw; a fresh draw that happens to land on the
            # continuation index is missed, which only shrinks this set. It is a subset of the
            # true starts, so it can under-report coverage but never invent it.
            started_at.add(int(idx[0]))
            started_at.update(
                int(idx[t]) for t in range(1, n) if idx[t] != (idx[t - 1] + 1) % n)

        missing = sorted(set(range(n)) - started_at)
        assert not missing, (
            f"over {draws} resamples of {n} observations, no block ever started at "
            f"{missing} — those observations can only ever be reached by wrapping into them from "
            f"their predecessor, so they are under-sampled"
            + (f", and index {n - 1} is the newest bar in the series" if n - 1 in missing else ""))

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

    def test_a_MEAN_REVERTING_series_is_dependent_too_and_needs_long_blocks(self):
        """The test above uses white noise and a POSITIVELY autocorrelated series, so a rule that
        keyed on the raw autocorrelation rather than on its magnitude would satisfy it.

        Negative autocorrelation is dependence: a series that alternates is every bit as
        predictable from its own past as one that trends, and shuffling it in loose observations
        destroys that structure just as completely. Read the sign instead of the size and every
        mean-reverting series — which is most of what a filtered trading signal looks like — drops
        straight to the floor of 2, the stationary bootstrap degenerates into an iid shuffle, and
        every interval comes out too narrow in the one direction that flatters.
        """
        rng = np.random.default_rng(5)
        n = 600
        white = rng.standard_normal(n)
        alternating = np.empty(n)                        # rho ~ -0.85
        alternating[0] = rng.standard_normal()
        for i in range(1, n):
            alternating[i] = -0.85 * alternating[i - 1] + rng.standard_normal()

        b_white, b_alt = optimal_block_length(white), optimal_block_length(alternating)
        assert b_alt > b_white, (
            "a strongly mean-reverting series carries as much dependence as a trending one and "
            f"needs longer blocks for it, but it got {b_alt:.2f} against {b_white:.2f} for noise")
        assert b_alt > 2.0, (
            f"and a block length of {b_alt:.2f} is the floor, which means no dependence was "
            "detected at all")

    def test_the_dependence_it_measures_is_the_one_bar_kind(self):
        """The block length is documented as adjusting for FIRST-order autocorrelation, and both
        tests above are blind to which lag it actually reads: an AR(1) with rho −0.85 has a lag-2
        autocorrelation of +0.72, so a estimator keyed on lag 2 still reports a long block and both
        of them stay green.

        This fixture separates them by construction. An MA(1) — `x[i] = e[i] + 0.9·e[i-1]` — has
        strong first-order dependence (rho ≈ +0.50) and EXACTLY ZERO at every longer lag, because
        no two observations two steps apart share a shock. Read the wrong lag and it looks like
        white noise: blocks collapse toward the floor, the resample tears apart the one dependence
        the series has, and the interval comes out too narrow in the direction that flatters.
        """
        rng = np.random.default_rng(17)
        n = 800
        e = rng.standard_normal(n + 1)
        ma1 = e[1:] + 0.9 * e[:-1]

        x = ma1 - ma1.mean()
        denom = float((x * x).sum())
        lag1 = float((x[:-1] * x[1:]).sum() / denom)
        lag2 = float((x[:-2] * x[2:]).sum() / denom)
        assert lag1 > 0.4 and abs(lag2) < 0.1, (
            f"fixture check: this MA(1) must be dependent at lag 1 ({lag1:.3f}) and independent "
            f"at lag 2 ({lag2:.3f}), or it cannot tell the two estimators apart")

        b_ma1 = optimal_block_length(ma1)
        b_white = optimal_block_length(rng.standard_normal(n))
        assert b_ma1 > 1.5 * b_white, (
            f"a series with rho={lag1:.2f} at one lag got a block length of {b_ma1:.2f}, against "
            f"{b_white:.2f} for white noise. The adjustment is reading a lag at which this series "
            "is independent, so first-order dependence is being resampled away")

    def test_the_shortcut_for_tiny_samples_starts_where_it_says_it_does(self):
        """`optimal_block_length` gives up and returns the floor below twenty observations. Move
        that boundary up and the giving-up spreads silently over the population it matters most
        for: a year of daily signals is 250 observations and a year of weekly is 52, so most real
        users would get the iid shuffle and a p-value computed as if their returns were
        independent. Both sides of the boundary, or it slides.
        """
        rng = np.random.default_rng(6)
        assert optimal_block_length(rng.standard_normal(150)) > 2.0, (
            "150 observations is plenty to estimate a block length from and must not take the "
            "give-up path")
        assert optimal_block_length(rng.standard_normal(19)) == 2.0, (
            "and 19 is below the declared boundary, where the floor is the honest answer")

    def test_which_of_the_two_bootstraps_runs_is_decided_at_thirty_blocks(self, monkeypatch):
        """★ `reality_check` carries two independent implementations of the same statistic, and
        nothing said which one runs when.

        The switch is `nb >= 30`, where `nb` is how many whole blocks the history divides into.
        Move it and series in the vacated range are silently re-routed onto the other estimator —
        so a user's published p-value can change because their history grew by two bars. The two
        paths are close but not identical (the sibling test below exists precisely because they
        compute `p_values` differently), which is what makes a sliding switch a real change in the
        answer rather than a refactor.

        The two lengths below are adjacent prefixes of the SAME series, chosen so `nb` is 29 and
        then 30. Both halves are asserted: that the boundary is where it is claimed to be, and
        that the two estimators agree across it — the second being the reason the first is only a
        performance decision and not a statistical one.
        """
        calls: list[int] = []
        real = reality_check_mod.stationary_bootstrap_indices
        monkeypatch.setattr(reality_check_mod, "stationary_bootstrap_indices",
                            lambda n, block, rng: (calls.append(n), real(n, block, rng))[1])

        rng = np.random.default_rng(4)
        full = rng.standard_normal(400) * 0.01
        other = np.random.default_rng(5).standard_normal(400) * 0.01

        def run(n):
            calls.clear()
            r = reality_check({"a": full[:n] + 0.004, "b": other[:n]}, n_boot=250, seed=3)
            return r, bool(calls)

        short, short_stationary = run(179)      # nb == 29
        long_, long_stationary = run(180)       # nb == 30

        assert short_stationary, (
            "179 observations divide into 29 blocks, below the declared switch, and must be run "
            "through the stationary bootstrap; the aggregated path took it instead"
        )
        assert not long_stationary, (
            "180 observations divide into 30 blocks, which is the switch, and must be run through "
            "the aggregated block-means path; the stationary bootstrap took it instead"
        )
        assert short.p_value == pytest.approx(long_.p_value, abs=0.15), (
            f"the two estimators disagree across the switch: p={short.p_value:.3f} on 29 blocks "
            f"and p={long_.p_value:.3f} on 30. One bar of extra history is changing the answer, "
            "so the switch is a statistical decision and not the performance shortcut it is "
            "documented as"
        )

    def test_the_short_path_gives_every_strategy_its_own_p_value_too(self):
        """`reality_check` has two implementations, and the one that reports each strategy's
        individual p-value against ITS OWN null is only exercised on long histories. On the short
        path the same numbers can quietly collapse into a single joint figure repeated K times —
        which is the brochure this correction exists to prevent, since a reader then cannot tell a
        set with one good idea in it from a set that was dredged.
        """
        rng = np.random.default_rng(31)
        n = 120
        noise = {f"noise{i}": rng.standard_normal(n) * 0.01 for i in range(4)}
        r = reality_check({**noise, "winner": rng.standard_normal(n) * 0.01 + 0.02}, n_boot=300)
        assert n // max(2, round(r.block_length)) < 30, (
            "this fixture must take the short-series path for the test to reach it")
        assert len(set(r.p_values.values())) > 1, (
            "five different strategies cannot all have the same p-value; they were all reported "
            f"as {sorted(set(r.p_values.values()))}")
        assert r.p_values["winner"] < min(r.p_values[f"noise{i}"] for i in range(4)), (
            f"and the one with the edge must be the most significant of them: {r.p_values}")

    def test_renaming_a_strategy_does_not_change_the_answer(self):
        """The block length is estimated from the strategy under examination — the best one. Take
        it from whichever strategy happens to sort first alphabetically instead and the
        correction's verdict depends on what the user called things: renaming «my_idea» to
        «zz_idea» would change its p-value, which is not a statistic, it is a coincidence.
        """
        rng = np.random.default_rng(2)
        n = 400
        noise = rng.standard_normal(n) * 0.01
        persistent = np.empty(n)                         # the winner, and strongly autocorrelated
        persistent[0] = 0.0
        for i in range(1, n):
            persistent[i] = 0.9 * persistent[i - 1] + rng.standard_normal() * 0.01
        winner = persistent + 0.02

        first = reality_check({"aaa": winner, "zzz": noise}, n_boot=200)
        last = reality_check({"aaa": noise, "zzz": winner}, n_boot=200)
        assert (first.best_name, last.best_name) == ("aaa", "zzz"), (
            "the same series must win under either naming, or the fixture is not testing this")
        assert first.block_length == pytest.approx(last.block_length), (
            "the block length is a property of the winning series, not of its label: it read "
            f"{first.block_length:.2f} with the winner named first and {last.block_length:.2f} "
            "with it named last")
        # The block length is the mechanism; the ANSWER is what the test is named for, and
        # asserting only the mechanism would leave a change that made the block length
        # name-independent while the verdict stayed name-dependent entirely invisible.
        assert first.p_value == pytest.approx(last.p_value), (
            f"the same two series produced p={first.p_value:.4f} when the winner was called "
            f"'aaa' and p={last.p_value:.4f} when it was called 'zzz'. That is not a statistic")
        assert first.best_stat == pytest.approx(last.best_stat), (
            f"and the statistic itself moved, {first.best_stat:+.5f} against {last.best_stat:+.5f}")
        assert (pytest.approx(first.p_values["aaa"]) == last.p_values["zzz"]
                and pytest.approx(first.p_values["zzz"]) == last.p_values["aaa"]), (
            "each strategy's individual p-value has to follow the series, not the slot: "
            f"{first.p_values} against {last.p_values}")
