"""The rule editor's sandbox, tested as a sandbox.

`tests/test_assay.py::TestEditorSecurity` already fires ten attack strings at `evaluate_rule` and
asserts each one raises. That is necessary and it is not sufficient, because `ExprError` has more
than one source. An attack that gets PAST the allowlist and evaluates to a scalar still raises —
from the output-shape check at the bottom of `evaluate_rule` ("the rule has to produce a series the
same size as the prices"). `close.__class__` and `close[0]` are both scalars, so both of those
"rejections" are shape errors wearing a security test's name.

That is not a hypothetical distinction. Add a `v_Attribute` visitor to `_Interpreter` — one line,
the sort of thing someone adds to support `close.mean()` — and arbitrary `getattr` becomes
reachable on every live object in the interpreter, `abs.__globals__` included, while every
assertion in that class stays green.

So the tests here assert on the GROUND of the rejection, not merely that something was raised.

Six of the ten attack strings are additionally chosen to produce a correctly-shaped array if they
were ever allowed to run — both `Attribute` cases, `Subscript`, `IfExp`, `NamedExpr` and `Tuple` —
so that for those the shape check demonstrably cannot launder the rejection into looking like a
security decision. The remaining four (`ListComp`, `Lambda`, `JoinedStr`, `Dict`) do not have that
property; they are sound tests because the `match=` on the message does the work regardless, but
they are not evidence about the shape check and should not be read as such. Say which kind a new
parameter is when adding one: an attack that only the shape check rejects proves nothing here.
"""

from __future__ import annotations

import ast

import numpy as np
import pytest

from wavelab.validation.expr import (
    FUNC_DOCS,
    SAFE_FUNCS,
    ExprError,
    _Interpreter,
    build_series,
    evaluate_rule,
)

#: The node types the interpreter is allowed to execute. Adding a name here is a SECURITY
#: decision, not a convenience: every one of them is a capability handed to a string a stranger
#: typed into a web form.
REVIEWED_NODES = frozenset({
    "Expression", "Constant", "Name", "BinOp", "UnaryOp", "BoolOp", "Compare", "Call",
})

#: The functions the interpreter will call. Adding a name here is the SAME security decision as
#: adding a node type above, and a sharper one: `v_Name` resolves bare names out of
#: `{**series, **SAFE_FUNCS}`, so every entry is also a first-class VALUE a rule can pass around.
#: One higher-order function in this dict — anything that calls its own argument — hands arbitrary
#: calls back to a string typed into a web form.
REVIEWED_FUNCS = frozenset({
    "sma", "ema", "rsi", "std", "highest", "lowest", "change", "shift",
    "crosses_above", "crosses_below", "abs",
})


@pytest.fixture(scope="module")
def prices():
    """A short, boring series. Nothing here depends on its values, only on its shape."""
    c = np.arange(100.0, 150.0)
    return build_series(c - 1.0, c + 2.0, c - 2.0, c, np.ones_like(c))


class TestTheAllowlistIsWhatRejects:
    """Every attack must be turned away by the allowlist, on the allowlist's own grounds.

    If the rejection comes from anywhere else, the allowlist can be widened without a single test
    going red — and widening it is exactly the change a well-meaning person makes when they want
    `close[0]` or a ternary in the editor. The message is the evidence: "expression not allowed"
    can only come from `_Interpreter.visit` finding no visitor for the node.
    """

    #: Each of these evaluates to an array the same shape as `close` if the allowlist lets it
    #: through, so `evaluate_rule`'s shape check cannot raise on its behalf and take the credit.
    @pytest.mark.parametrize("attack,node", [
        ("close.real > 0", "Attribute"),                        # arbitrary getattr
        ("-close.real > 0", "Attribute"),                       # ... reached through a unary op
        ("close + close[0] * 0 > 0", "Subscript"),              # indexing
        ("(close if close > 0 else open) > 0", "IfExp"),         # scalar truthiness on an array
        ("(c := close) > 0", "NamedExpr"),                      # the walrus, named in the docstring
        ("close > (0, 1)", "Tuple"),
        ("[x for x in range(10)]", "ListComp"),
        ("lambda x: x", "Lambda"),
        ('f"{close}"', "JoinedStr"),
        ("{1: 2}", "Dict"),
    ])
    def test_it_is_refused_for_being_a_forbidden_node(self, prices, attack, node):
        with pytest.raises(ExprError, match="expression not allowed") as e:
            evaluate_rule(prices, attack)
        assert node in str(e.value), (
            f"{attack!r} must be refused as a {node} node the allowlist does not carry; "
            f"the message instead reads: {str(e.value)[:120]!r}")

    def test_getattr_is_not_reachable_even_when_the_result_has_the_right_shape(self, prices):
        """`close.real` is `close`. It is the cheapest possible attribute expression, it returns a
        perfectly well-shaped array, and it is one visitor away from `abs.__globals__`.

        This is the case the shape check cannot cover for, which makes it the one that says
        whether the sandbox is a sandbox or a coincidence.
        """
        with pytest.raises(ExprError, match="expression not allowed"):
            evaluate_rule(prices, "close.real > 0")

    def test_a_string_never_becomes_a_value(self, prices):
        """Strings are what the payload in every one of these attacks is made of. A constant that
        is not a number has no meaning in a trading rule, so admitting one buys nothing and hands
        the interpreter the one type that can carry a filename, a module name or a command."""
        with pytest.raises(ExprError, match="only numbers are accepted"):
            evaluate_rule(prices, "'/etc/passwd'")

    def test_the_reviewed_set_of_node_types_is_the_one_the_interpreter_runs(self):
        """The allowlist is the set of `v_*` methods, so it is editable by adding a method and
        there is nothing else in the file that says which set was agreed to.

        This is the guard that catches the visitor nobody in this test file thought of. It fails
        loudly on a widening (a capability was granted) and equally loudly on a narrowing (a rule
        people write stopped working), and in both cases the fix is to change `REVIEWED_NODES` in
        the same commit, deliberately, where a reviewer will see it.
        """
        live = frozenset(name[2:] for name in dir(_Interpreter) if name.startswith("v_"))
        assert live == REVIEWED_NODES, (
            "the AST nodes the rule interpreter will execute have changed. Granted since the "
            f"last review: {sorted(live - REVIEWED_NODES)}; withdrawn: "
            f"{sorted(REVIEWED_NODES - live)}")

    def test_the_reviewed_set_of_CALLABLES_is_the_one_the_interpreter_offers(self):
        """★ `_Interpreter` has two capability lists and freezing one of them is not enough.

        The test above pins the AST nodes. Nothing pinned `SAFE_FUNCS` — and every entry in it is
        a first-class value, because `v_Name` resolves bare names out of `{**series, **SAFE_FUNCS}`
        rather than only resolving them at a call site. So one higher-order entry is a call
        primitive: I added `"apply": lambda f, x: f(x)` to that dict and the whole suite stayed
        green while `apply(abs, body) > 0` evaluated and fired on every bar. That is the same
        class of hole the node allowlist was frozen to close, one door over, now wearing the
        reassurance of a file called `test_expr_sandbox.py`.

        The second assertion is the paperwork half. A function shipped without a doc entry is
        invisible in the editor's help panel, which is the shape an undocumented back door takes:
        callable by anyone who knows the name, listed to nobody.
        """
        assert frozenset(SAFE_FUNCS) == REVIEWED_FUNCS, (
            "the functions the rule interpreter will call have changed. Granted since the last "
            f"review: {sorted(frozenset(SAFE_FUNCS) - REVIEWED_FUNCS)}; withdrawn: "
            f"{sorted(REVIEWED_FUNCS - frozenset(SAFE_FUNCS))}. Every name here is reachable as a "
            "bare value, so a single higher-order entry restores arbitrary calls")
        assert frozenset(SAFE_FUNCS) == frozenset(k.split("(")[0] for k in FUNC_DOCS), (
            "SAFE_FUNCS and FUNC_DOCS disagree: "
            f"callable but undocumented {sorted(frozenset(SAFE_FUNCS) - frozenset(k.split('(')[0] for k in FUNC_DOCS))}, "
            f"documented but not callable {sorted(frozenset(k.split('(')[0] for k in FUNC_DOCS) - frozenset(SAFE_FUNCS))}")

    def test_a_function_can_only_be_reached_by_a_bare_name(self, prices):
        """The other half of the same boundary: `f(x)` is checked against `SAFE_FUNCS` by name, so
        anything that arrives as a call on an EXPRESSION — `().__class__(...)`, `close.mean()` —
        has to be refused before the name lookup, or the lookup is looking at the wrong thing."""
        for attack in ("close.mean()", "().__class__.__bases__[0].__subclasses__()"):
            with pytest.raises(ExprError, match="only be called by name"):
                evaluate_rule(prices, attack)


class TestTheRuleArithmeticMeansWhatItSays:
    """What an expression computes, checked by value rather than by how many bars it fired on.

    Every existing test of the interpreter counts signals (`n_long > 0`, `both <= min(a, b)`), and
    a count survives almost any arithmetic error: swap the operands of every subtraction and a
    rule still fires on some bars, just not the right ones. A user whose rule is `close - open`
    would be trading its exact inverse and the only symptom is that their idea "does not work".
    """

    #: open, high, low, close chosen so every quantity below is exact in binary.
    O, H, L, C = 10.0, 12.0, 9.0, 11.0

    @pytest.fixture
    def bars(self):
        two = np.array([1.0, 1.0])
        return build_series(two * self.O, two * self.H, two * self.L,
                            np.array([11.0, 9.0]), two)

    def evaluate(self, bars, src):
        env = {**bars, **SAFE_FUNCS}
        return np.asarray(_Interpreter(env).visit(ast.parse(src, mode="eval")), dtype=float)

    def test_a_subtraction_is_left_minus_right(self, bars):
        """Operand order. Evaluated the other way round, every `close - open`, every
        `high - low` and every difference a user writes comes back negated: a momentum rule
        becomes a mean-reversion rule and reports itself as a losing idea rather than as a bug."""
        assert self.evaluate(bars, "close - open").tolist() == [1.0, -1.0], (
            "close - open on bars closing at 11 and 9 against an open of 10 is [+1, -1]")
        assert self.evaluate(bars, "open - close").tolist() == [-1.0, 1.0], (
            "and the reverse subtraction is its negation, not the same number")

    def test_a_division_is_left_over_right(self, bars):
        assert self.evaluate(bars, "close / open").tolist() == pytest.approx([1.1, 0.9]), (
            "close / open is the ratio of the close TO the open")

    def test_abs_takes_an_absolute_value(self, bars):
        """`abs(body) > x` is how anyone asks for a large candle in either direction. Without the
        absolute value it asks for a large UP candle only, and the rule silently stops seeing
        every sell-off — the direction-blindness that has already cost this project once."""
        assert self.evaluate(bars, "abs(body)").tolist() == [1.0, 1.0], (
            "a body of -1 is one unit of body; abs() is what makes the down bar count")

    def test_body_is_close_minus_open_and_range_is_high_minus_low(self, bars):
        """These two series are named in the editor's own help text. `body` inverted turns every
        bullish candle bearish for every rule that names it, at once."""
        assert bars["body"].tolist() == [1.0, -1.0], "body is close − open, signed"
        assert bars["range"].tolist() == [3.0, 3.0], "range is high − low, always positive"

    def test_atr_is_the_true_range_of_the_bar_not_of_the_bar_upside_down(self):
        """Every bar here is exactly 20 wide with its close in the middle and no gaps, so the
        true range is 20 on every bar and the ATR is 20 whatever the period.

        Hand `talib.ATR` its high and low the wrong way round and the true range becomes the
        distance from the previous close instead — 10 here, half the real thing. ATR is the
        volatility unit the whole editor's thresholds are quoted in, so halving it does not break
        a rule, it silently doubles how often every ATR-scaled rule fires.
        """
        n = 40
        c = np.full(n, 100.0)
        s = build_series(c, c + 10.0, c - 10.0, c, np.ones(n))
        assert s["atr"][-1] == pytest.approx(20.0), (
            f"a bar 20 wide with no gap has a true range of 20; atr reads {s['atr'][-1]:.4f} "
            "(10.0 is the signature of high and low being passed the wrong way round)")

    def test_change_is_a_percentage_because_that_is_what_the_help_text_promises(self):
        """`FUNC_DOCS` tells the user "percent change against n bars ago" and that text ships to
        the editor. Return a fraction instead and it is 100x smaller: `change(close, 1) > 2`
        stops firing entirely and the user concludes their idea is worthless."""
        from wavelab.validation.expr import change
        assert change(np.array([100.0, 110.0]), 1)[1] == pytest.approx(10.0), (
            "100 -> 110 is a change of 10 PERCENT, not 0.1")

    def test_a_non_strict_comparison_includes_the_boundary(self, bars):
        """Prices sit exactly on round numbers, on yesterday's close and on levels the user typed
        in. `<=` quietly compiled as `<` drops precisely those bars — the same silent-miss that
        `TestCrossings` exists to prevent, one layer lower down."""
        flat = np.full(5, 1.0)
        s = build_series(flat, flat, flat, flat, flat)
        assert evaluate_rule(s, "close <= close").n_long == 5, (
            "every value is equal to itself, so `close <= close` is true on every bar")
        assert evaluate_rule(s, "close >= close").n_long == 5, (
            "and so is `close >= close`")
        assert evaluate_rule(s, "close < close").n_long == 0, (
            "the strict comparison is the one that must exclude the boundary")

    def test_a_bare_series_is_true_where_it_is_NON_ZERO_not_where_it_is_positive(self):
        """`body and volume > 0` is a rule someone writes. Every series the editor offers except
        `body` happens to be positive, so a truthiness rule of "greater than zero" behaves
        identically on all of them — and then drops every DOWN bar from any rule that names
        `body`, which is half the market and the half a short strategy lives in.
        """
        close = np.array([-1.0, 0.0, 1.0])
        s = build_series(np.zeros(3), close, close, close, np.ones(3))
        assert s["body"].tolist() == [-1.0, 0.0, 1.0], "the fixture must actually be signed"
        assert evaluate_rule(s, "body").signal.tolist() == [1, 0, 1], (
            "a body of -1 is a real body: only the zero bar is false")
        assert evaluate_rule(s, "not body").signal.tolist() == [0, 1, 0], (
            "and its negation is true only where the series is actually zero")
