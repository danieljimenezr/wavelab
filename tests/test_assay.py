"""Assay is the product, so its guards are the part it matters most not to break."""

from __future__ import annotations

import numpy as np
import pytest

from wavelab.validation.csv_import import ImportError_, align_to_bars, parse_signals_csv
from wavelab.validation.expr import ExprError, build_series, evaluate_rule

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
