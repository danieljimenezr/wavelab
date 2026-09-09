"""Safe interpreter for trading rules written by the user.

There is NO eval() anywhere. It walks Python's own AST against an allowlist: any node that is not
explicitly permitted is rejected. That is not paranoia: if Assay is offered as a service, calling
eval() on a string a stranger sent you is remote code execution.

Forbidden: imports, attribute access (`().__class__`), subscription (`x[i]`), lambdas,
comprehensions, assignments, f-strings, the walrus operator, and any function not on the list.

★ Plus one constraint that comes from the domain rather than from security: `shift(x, n)` accepts
positive n ONLY. Shifting forward is looking at the future, and the most expensive mistake in this
trade should not even be expressible.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import talib

__all__ = ["FUNC_DOCS", "SAFE_FUNCS", "SERIES_DOCS", "ExprError", "build_series", "evaluate_rule"]


class ExprError(ValueError):
    """The rule is not valid. The message is shown to the user exactly as it is."""


# ------------------------------------------------------------------ allowed functions

def _n(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def sma(x, n): return talib.SMA(_n(x), int(n))
def ema(x, n): return talib.EMA(_n(x), int(n))
def rsi(x, n=14): return talib.RSI(_n(x), int(n))
def std(x, n): return talib.STDDEV(_n(x), int(n))
def highest(x, n): return talib.MAX(_n(x), int(n))
def lowest(x, n): return talib.MIN(_n(x), int(n))
def change(x, n=1): return talib.ROC(_n(x), int(n))


def shift(x, n=1):
    """Shifts the series n bars BACKWARD. Negative n is FORBIDDEN: it would be looking ahead."""
    n = int(n)
    if n < 0:
        raise ExprError(
            "shift() does not accept negative values. A negative shift brings in data from the "
            "FUTURE, and that is the bug that turns a beautiful backtest into money lost live. "
            "If you really do want to look forward, this tool is not for you."
        )
    a = _n(x)
    if n == 0:
        return a
    out = np.full_like(a, np.nan)
    out[n:] = a[:-n]
    return out


def crosses_above(a, b):
    a, b = _n(a), _n(b)
    prev_a, prev_b = shift(a), shift(b)
    return ((a > b) & (prev_a <= prev_b)).astype(float)


def crosses_below(a, b):
    a, b = _n(a), _n(b)
    prev_a, prev_b = shift(a), shift(b)
    return ((a < b) & (prev_a >= prev_b)).astype(float)


SAFE_FUNCS = {
    "sma": sma, "ema": ema, "rsi": rsi, "std": std,
    "highest": highest, "lowest": lowest, "change": change,
    "shift": shift, "crosses_above": crosses_above, "crosses_below": crosses_below,
    "abs": lambda x: np.abs(_n(x)),
}

FUNC_DOCS = {
    "sma(x, n)": "simple moving average over n bars",
    "ema(x, n)": "exponential moving average over n bars",
    "rsi(x, n)": "RSI over n bars (14 by default)",
    "std(x, n)": "standard deviation over n bars",
    "highest(x, n)": "highest value of the last n bars",
    "lowest(x, n)": "lowest value of the last n bars",
    "change(x, n)": "percent change against n bars ago",
    "shift(x, n)": "the value n bars ago (negative n FORBIDDEN)",
    "crosses_above(a, b)": "1 on the bar where a crosses above b",
    "crosses_below(a, b)": "1 on the bar where a crosses below b",
    "abs(x)": "absolute value",
}

SERIES_DOCS = {
    "close": "closing price", "open": "opening price",
    "high": "high of the bar", "low": "low of the bar",
    "volume": "volume", "atr": "14-bar ATR",
    "range": "high − low of the bar", "body": "close − open",
}

_BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
        ast.Div: np.divide, ast.Mod: np.mod, ast.Pow: np.power,
        ast.BitAnd: np.logical_and, ast.BitOr: np.logical_or}
_CMP = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater,
        ast.GtE: np.greater_equal, ast.Eq: np.equal, ast.NotEq: np.not_equal}


class _Interpreter:
    """Recursive AST interpreter. There is NO eval() anywhere in this module.

    Walking the tree by hand instead of compiling it buys two things that pay for the extra code:
    total control over what is allowed to run, and the ability to treat `and`/`or` ELEMENT BY
    ELEMENT. Python evaluates `a and b` on the array's global truth value and raises "the truth
    value of an array is ambiguous" — an incomprehensible error for someone who only wanted to
    write a rule.
    """

    def __init__(self, env: dict) -> None:
        self.env = env

    def visit(self, n: ast.AST):
        m = getattr(self, "v_" + type(n).__name__, None)
        if m is None:
            raise ExprError(
                f"expression not allowed: {type(n).__name__}. Only comparisons, arithmetic, "
                "`and`/`or`/`not` and the functions on the list are accepted. "
                "No imports, no attributes, no indexing, no lambdas.")
        return m(n)

    def v_Expression(self, n): return self.visit(n.body)

    def v_Constant(self, n):
        if not isinstance(n.value, (int, float, bool)):
            raise ExprError(f"only numbers are accepted, not {type(n.value).__name__}")
        return float(n.value)

    def v_Name(self, n):
        if n.id not in self.env:
            raise ExprError(
                f"unknown name: {n.id}. "
                f"Series: {', '.join(sorted(k for k in self.env if not callable(self.env[k])))}. "
                f"Functions: {', '.join(sorted(SAFE_FUNCS))}")
        return self.env[n.id]

    def v_BinOp(self, n):
        op = _BIN.get(type(n.op))
        if op is None:
            raise ExprError(f"operator not allowed: {type(n.op).__name__}")
        with np.errstate(all="ignore"):
            return op(self.visit(n.left), self.visit(n.right))

    def v_UnaryOp(self, n):
        v = self.visit(n.operand)
        if isinstance(n.op, ast.USub): return np.negative(v)
        if isinstance(n.op, ast.UAdd): return v
        if isinstance(n.op, (ast.Not, ast.Invert)): return np.logical_not(_bool(v))
        raise ExprError(f"unary operator not allowed: {type(n.op).__name__}")

    def v_BoolOp(self, n):
        # ELEMENT BY ELEMENT. This is the main reason for writing the interpreter at all.
        vals = [_bool(self.visit(v)) for v in n.values]
        f = np.logical_and if isinstance(n.op, ast.And) else np.logical_or
        out = vals[0]
        for v in vals[1:]:
            out = f(out, v)
        return out

    def v_Compare(self, n):
        if len(n.ops) != 1:
            raise ExprError("write comparisons one at a time: `a > b and b > c`, "
                            "not `a > b > c`")
        op = _CMP.get(type(n.ops[0]))
        if op is None:
            raise ExprError(f"comparison not allowed: {type(n.ops[0]).__name__}")
        with np.errstate(all="ignore"):
            return op(self.visit(n.left), self.visit(n.comparators[0]))

    def v_Call(self, n):
        if not isinstance(n.func, ast.Name):
            raise ExprError("functions can only be called by name")
        f = SAFE_FUNCS.get(n.func.id)
        if f is None:
            raise ExprError(f"unknown function: {n.func.id}(). "
                            f"Available: {', '.join(sorted(SAFE_FUNCS))}")
        if n.keywords:
            raise ExprError("functions do not take keyword arguments")
        return f(*[self.visit(a) for a in n.args])


def _bool(v) -> np.ndarray:
    a = np.asarray(v)
    return a if a.dtype == bool else np.nan_to_num(a, nan=0.0) != 0


def build_series(o, h, l, c, v) -> dict[str, np.ndarray]:
    """The series a rule is allowed to name. All of them causal by construction."""
    o, h, l, c, v = (_n(x) for x in (o, h, l, c, v))
    return {
        "close": c, "open": o, "high": h, "low": l, "volume": v,
        "atr": talib.ATR(h, l, c, 14),
        "range": h - l, "body": c - o,
    }


# NOTE: an earlier `_check(node)` helper lived here that walked the tree against a `_NODES`
# allowlist tuple. The tuple never existed, nothing ever called the helper, and calling it would
# have raised NameError rather than rejecting anything. It has been removed: the allowlist is
# `_Interpreter`'s `v_*` dispatch, which rejects any node type it has no visitor for, and a second
# half-written gate beside it is a trap, not defence in depth.


@dataclass(frozen=True, slots=True)
class RuleResult:
    signal: np.ndarray
    n_long: int
    n_short: int


def evaluate_rule(series: dict[str, np.ndarray], long_rule: str,
                  short_rule: str = "") -> RuleResult:
    """Evaluates the entry rules and returns the signal in {-1, 0, +1}.

    If both fire on the same bar, 0 (flat) wins: a rule that says buy and sell at once is not a
    signal, it is a contradiction, and settling it silently in favour of one side would hide the
    mistake from the user.
    """
    env = {**series, **SAFE_FUNCS}

    def _eval(src: str) -> np.ndarray:
        src = src.strip()
        if not src:
            return np.zeros_like(series["close"], dtype=bool)
        try:
            tree = ast.parse(src, mode="eval")
        except SyntaxError as e:
            raise ExprError(f"syntax error: {e.msg}") from None
        try:
            v = _Interpreter(env).visit(tree)
        except ExprError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ExprError(f"error while evaluating: {type(e).__name__}: {e}") from None
        a = np.asarray(v)
        if a.shape != series["close"].shape:
            raise ExprError("the rule has to produce a series the same size as the prices "
                            "(did you write a constant instead of a comparison?)")
        return np.nan_to_num(a, nan=0.0).astype(bool)

    long_hit = _eval(long_rule)
    short_hit = _eval(short_rule)
    sig = np.zeros(series["close"].size, dtype=np.int8)
    sig[long_hit & ~short_hit] = 1
    sig[short_hit & ~long_hit] = -1
    return RuleResult(sig, int((sig == 1).sum()), int((sig == -1).sum()))
