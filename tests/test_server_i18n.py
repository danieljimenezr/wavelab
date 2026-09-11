"""Every English sentence the API sends has to have a Spanish and a Catalan version.

The API contract is English-only on purpose, so `web/i18n.js` translates the server's prose on
ARRIVAL: `tx()` looks the text up in SERVER_ES / SERVER_CA, and then walks an ordered list of
anchored PATTERNS for the lines that carry numbers the battery computed. A miss is SILENT — no
error, no blank, just an English sentence sitting on a Spanish page. Fourteen strings had already
fallen through that hole once, the shift() look-ahead refusal among them: the one message whose
entire job is to stop somebody shipping a look-ahead bug.

Nothing but this module keeps those tables complete. Reword a sentence in battery.py,
csv_import.py, expr.py, projection.py, limits.py, live.py or app.py and its translation stops
matching; that line reverts to English and no other test notices.

HOW IT WORKS, and why it is built this way:

* **No JavaScript runtime.** deno is on the author's Mac and node is on the GitHub runner. A test
  that needs either one is green in one place and skipped in the other, which is worse than no
  test at all because the suite still reads as passing. So i18n.js is PARSED here and `tx()` is
  reimplemented in Python, in the same lookup order: exact table hit first, then the first
  PATTERN that both matches and has the language.

* **One source of truth.** The tables are not retyped: `_i18n()` reads the real objects and the
  real regexes out of web/i18n.js. A test carrying its own copy of the data only ever proves the
  copy agrees with itself.

* **Two harvests of the English side, and they check each other.** `_emitted()` RUNS the code — a
  real battery, real malformed CSVs, every expr.py refusal, every plan rejection, the rate
  limiter — and keeps whatever came out; it is the proof the sentences are real and reachable.
  `_specimens()` walks the AST of the same modules and rebuilds one sentence per emit site, with
  `⟨…⟩` standing in wherever an f-string interpolates; it reaches the sites a test cannot run (the
  25-second concurrency timeout) and it is what makes a REWORDING fail, because the rebuilt
  sentence changes the moment the source does.

* **The reverse direction too.** `test_no_table_entry_is_orphaned` fails when an entry or a
  pattern matches nothing the server can emit any more. Without it a rewording could be "fixed" by
  adding a second entry beside the stale one, and the tables would grow a layer of dead prose that
  hides the next hole.

`test_no_pattern_shadows_a_later_one` covers the other failure mode this list has, and it is not
hypothetical: PATTERNS is ordered and `tx()` takes the FIRST match, so a broad pattern added near
the top would quietly capture a message meant for a later, narrower one and render it with the
wrong words — visibly wrong to a reader, invisible to everybody else.

WHAT THIS MODULE DOES **NOT** COVER. Read this before treating a green run as "the product is
translated", because that is not what went green:

* **The interface's own words are only checked for KEY PARITY.** Nearly everything a reader sees —
  buttons, headings, table columns, the verdict labels — comes from the EN/ES/CA dicts through
  `t(key)`, not from the server through `tx()`. `test_the_ui_dictionaries_cover_the_same_keys`
  proves only that the three have the SAME KEYS. A key whose Spanish value was left in English is
  green here. Nothing in this suite reads those values.
* **SPEAKING_MODULES is a hand-maintained list.** A module that starts sending prose to a browser
  is invisible to both harvests until somebody adds it. As of this run the list is complete: every
  prose literal in those seven files is either harvested or is a `[server] …` line printed to
  stdout for an operator, which is English on purpose.
* **`_texts` only understands literals.** A sentence assembled by a helper call, by `%`, or by
  `"".join(...)` yields NOTHING at its emit site, so `_specimens()` cannot see it and a rewording
  of it is caught only if `_emitted()` happens to execute that branch. Every emit site in the
  speaking modules is a literal or an f-string today; that is a property of the code as written,
  not something this file enforces.
* **matcher.py is watched by RUNNING it, not by its AST** — see `_label_strings`. Its chart label
  is the only prose it has, and it reached the page in English in every language until this test
  was extended to cover it.
* **The hypotheses' pre-registration is elsewhere.** `rationale` and `prior` — the longest prose in
  the product, ~110 KB per language in web/hyp.{es,ca}.json — belong to tests/test_hypothesis_i18n.py,
  which pins them against a fingerprint of the English. Nothing here looks at them.
* **What a pattern CAPTURES is deliberately not translated.** `$1`…`$9` carry the server's figures
  and identifiers through verbatim, so an English noun captured by a group stays English inside an
  otherwise Spanish sentence. That was a real bug for the chart label and is now a table entry;
  for `close`, `BitXor` and the function lists it is intended, because those name things the rule
  editor will only accept in English.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
I18N_JS = ROOT / "web" / "i18n.js"
PKG = ROOT / "src" / "wavelab"
LANGS = ("es", "ca")

#: Stands in for an f-string's interpolated value. Non-empty and space-free, so it satisfies both
#: `(.+?)` and `(\S+?)`, and unmistakable in a failure message.
FILL = "⟨…⟩"

#: The modules whose prose reaches a browser. Every one of them has at least one string that is
#: passed through `tx()` in web/assay.js or web/app.js: d.error, d.report[], d.summary, d.name,
#: test.title/.detail/.explanation, and the decision card's reasons.
SPEAKING_MODULES = (
    "validation/battery.py",
    "validation/csv_import.py",
    "validation/expr.py",
    "waves/projection.py",
    "server/app.py",
    "server/limits.py",
    "engine/live.py",
)


# --------------------------------------------------------------------------- reading i18n.js
#
# A hand-written reader for the small slice of JavaScript these three declarations use: string
# literals, `+` concatenation, computed keys (`['a' + 'b']:`), regex literals and `new RegExp(...)`.
# It is deliberately strict — anything it does not recognise raises instead of being skipped,
# because a parser that silently drops an entry would report full coverage of half a table.

_SKIPPABLE = re.compile(r"(?:\s+|//[^\n]*|/\*.*?\*/)*", re.DOTALL)
_IDENT = re.compile(r"[A-Za-z_$][\w$]*")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0"}


def _skip(s: str, i: int) -> int:
    return _SKIPPABLE.match(s, i).end()


def _read_string(s: str, i: int) -> tuple[str, int]:
    quote = s[i]
    assert quote in "'\"", f"expected a string literal at offset {i}, found {s[i:i + 40]!r}"
    i += 1
    out: list[str] = []
    while s[i] != quote:
        if s[i] != "\\":
            out.append(s[i])
            i += 1
            continue
        nxt = s[i + 1]
        if nxt == "u":
            out.append(chr(int(s[i + 2:i + 6], 16)))
            i += 6
        elif nxt == "x":
            out.append(chr(int(s[i + 2:i + 4], 16)))
            i += 4
        elif nxt == "\n":
            i += 2                       # a line continuation contributes nothing
        else:
            out.append(_ESCAPES.get(nxt, nxt))
            i += 2
    return "".join(out), i + 1


def _read_concat(s: str, i: int) -> tuple[str, int]:
    """One string expression: `'a'`, or `'a' + 'b' + 'c'` spread over several lines."""
    i = _skip(s, i)
    parts: list[str] = []
    while True:
        piece, i = _read_string(s, i)
        parts.append(piece)
        j = _skip(s, i)
        if j < len(s) and s[j] == "+":
            i = _skip(s, j + 1)
            continue
        return "".join(parts), i


def _read_object(s: str, i: int) -> tuple[dict[str, str], int]:
    assert s[i] == "{"
    i = _skip(s, i + 1)
    out: dict[str, str] = {}
    while s[i] != "}":
        if s[i] == "[":                                  # computed key: ['a' + 'b']
            key, i = _read_concat(s, i + 1)
            i = _skip(s, i)
            assert s[i] == "]"
            i = _skip(s, i + 1)
        elif s[i] in "'\"":
            key, i = _read_string(s, i)
            i = _skip(s, i)
        else:                                            # bare identifier, e.g. `volume:`
            m = _IDENT.match(s, i)
            assert m, f"unreadable object key at offset {i}: {s[i:i + 40]!r}"
            key, i = m.group(0), _skip(s, m.end())
        assert s[i] == ":", f"expected ':' after key {key!r}"
        value, i = _read_concat(s, i + 1)
        out[key] = value
        i = _skip(s, i)
        if s[i] == ",":
            i = _skip(s, i + 1)
    return out, i + 1


def _read_regex_literal(s: str, i: int) -> tuple[str, int]:
    assert s[i] == "/"
    i += 1
    out: list[str] = []
    in_class = False
    while True:
        c = s[i]
        if c == "\\":
            out.append(s[i:i + 2])
            i += 2
            continue
        if c == "[":
            in_class = True
        elif c == "]":
            in_class = False
        elif c == "/" and not in_class:
            break
        out.append(c)
        i += 1
    return "".join(out), re.compile(r"[a-z]*").match(s, i + 1).end()


def _read_patterns(s: str, i: int) -> list[tuple[str, dict[str, str]]]:
    assert s[i] == "["
    i = _skip(s, i + 1)
    out: list[tuple[str, dict[str, str]]] = []
    while s[i] != "]":
        assert s[i] == "[", f"expected a [regex, translations] pair at offset {i}"
        i = _skip(s, i + 1)
        if s[i] == "/":
            source, i = _read_regex_literal(s, i)
        else:
            assert s.startswith("new RegExp", i), f"unreadable pattern at offset {i}"
            j = _skip(s, i + len("new RegExp"))
            assert s[j] == "("
            source, i = _read_concat(s, j + 1)
            i = _skip(s, i)
            assert s[i] == ")"
            i += 1
        i = _skip(s, i)
        assert s[i] == ","
        i = _skip(s, i + 1)
        translations, i = _read_object(s, i)
        i = _skip(s, i)
        assert s[i] == "]"
        i = _skip(s, i + 1)
        if s[i] == ",":
            i = _skip(s, i + 1)
        out.append((source, translations))
    return out


def _i18n() -> tuple[dict[str, dict[str, str]], list[tuple[str, dict[str, str]]]]:
    src = I18N_JS.read_text(encoding="utf-8")
    tables: dict[str, dict[str, str]] = {}
    for lang, name in (("es", "SERVER_ES"), ("ca", "SERVER_CA")):
        decl = f"const {name} = "
        assert decl in src, f"{name} has gone from i18n.js: nothing translates the API any more"
        tables[lang], _ = _read_object(src, _skip(src, src.index(decl) + len(decl)))
    decl = "const PATTERNS = "
    assert decl in src, "PATTERNS has gone from i18n.js: every computed line reverts to English"
    patterns = _read_patterns(src, _skip(src, src.index(decl) + len(decl)))
    return tables, patterns


SERVER, PATTERN_SOURCES = _i18n()
PATTERNS = [(re.compile(source), tr) for source, tr in PATTERN_SOURCES]


def tx(text: str, lang: str) -> str | None:
    """`tx()` from i18n.js, to the letter. Returns None where the page would show English."""
    table = SERVER[lang]
    if text in table:
        return table[text]
    for rx, tr in PATTERNS:
        m = rx.search(text)
        if m and tr.get(lang):
            # `$1`…`$9` carry over the figures the battery computed, exactly as the JS does —
            # including its table lookup on each group, without which a capture that IS a
            # translatable sentence (the chart label) would come out English inside a translated
            # one, and this reimplementation would not notice.
            groups = m.groups()

            def _sub(g, gs=groups, table=table):
                v = gs[int(g.group(1)) - 1] or ""
                return table.get(v, v)

            return re.sub(r"\$(\d)", _sub, tr[lang])
    return None


def _hit(text: str) -> str | None:
    """Which table entry or pattern claims this text, for the orphan check."""
    for lang in LANGS:
        if text in SERVER[lang]:
            return f"table:{text}"
    for rx, _ in PATTERNS:
        if rx.search(text):
            return f"pattern:{rx.pattern}"
    return None


# ------------------------------------------------------------- harvest 1: run the code, keep the
#                                                                          sentences it produced

def _battery_strings() -> set[str]:
    from wavelab.validation.battery import BatteryResult, Test, run_battery

    day = 86_400_000
    out: set[str] = set()

    def collect(close, ts, sig, **kw):
        r = run_battery(close, ts, np.asarray(sig, dtype=float), bar_ms=day, n_random=12, **kw)
        out.add(r.summary)
        for t in r.tests:
            out.update(x for x in (t.title, t.explanation, t.detail) if x)

    rng = np.random.default_rng(3)
    n = 800
    ret = rng.normal(0.0015, 0.02, n)
    close = 30_000 * np.exp(np.cumsum(ret))
    ts = np.arange(n, dtype=np.int64) * day + 1_500_000_000_000

    # Long only where the NEXT bar rises: a deliberate look-ahead, so the delay test has a large
    # positive in-sample excess and test 4 takes its `in-sample excess … → out of sample …` branch.
    leak = np.concatenate([(ret[1:] > 0).astype(float), [0.0]])
    collect(close, ts, leak)

    # Short on an uptrend: the in-sample excess is negative, which is test 4's OTHER branch — the
    # one that refuses to report "keeps 122%" about a strategy that loses consistently.
    collect(close, ts, -np.ones(n))

    # Ten signals in eight hundred bars: test 2 cannot conclude and says so.
    sparse = np.zeros(n)
    sparse[100:110] = 1.0
    collect(close, ts, sparse)

    # Short enough that the walk-forward has under 100 out-of-sample bars to work with.
    collect(close[:120], ts[:120], leak[:120])

    # The four verdicts. Built from Test objects rather than from prices because reaching all four
    # through run_battery means hand-tuning a series until it fails exactly two tests, and that
    # series would silently stop meaning anything the first time a threshold moved.
    def verdict_of(*failures: str) -> str:
        tests = [Test(tid, "t", tid not in failures, 0.0, 0.0, "u", "e", "d")
                 for tid in ("delay", "base_rate", "random_control", "effective_n")]
        return BatteryResult("n", 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, tests).summary

    summaries = {verdict_of("delay"), verdict_of("base_rate", "effective_n"),
                 verdict_of("base_rate"), verdict_of()}
    assert len(summaries) == 4, "the four verdicts no longer produce four distinct summaries"
    return out | summaries


_CSV_CASES = (
    # epochs, at each of the four scales the importer distinguishes
    "time,signal\n1700000000000000000,long\n1700086400000000000,flat\n1700172800000000000,short",
    "time,signal\n1700000000000000,1\n1700086400000000,0\n1700172800000000,-1",
    "time,signal\n1700000000000,1\n1700086400000,0\n1700172800000,-1",
    "time,signal\n1700000000,1\n1700086400,0\n1700172800,-1",
    # text dates, text signals, and a duplicate timestamp
    "date,signal\n2024-01-15,long\n2024-01-15,flat\n2024-01-16,short",
    # a numeric signal with more than twelve distinct values: only its SIGN is used
    "date,signal\n" + "\n".join(f"2024-02-{d:02d},{d / 100:.2f}" for d in range(1, 20)),
    # the user's own prices, accepted…
    "date,price,signal\n2024-01-15,185.5,long\n2024-01-16,187.2,long\n2024-01-17,183.1,flat",
    # …and refused, because a price of zero is not a price
    "date,close,signal\n2024-01-15,0,long\n2024-01-16,-2,long\n2024-01-17,-3,flat",
)

_CSV_REJECTED = (
    "a,b\n1,2\n3,4",                                          # no time column
    "date,x\n2024-01-15,7\n2024-01-16,8",                     # no signal column
    "date,signal\n2024-01-15,maybe\n2024-01-16,perhaps",      # unreadable signal words
    "date,signal\n5,long\n6,flat\n7,short",                   # numbers too small to be a time
    "date,signal\nnot-a-date,long\nalso-not,flat",            # unparseable dates
    "date,signal\n",                                          # a header and nothing under it
    "",                                                       # not even a header
)


def _csv_strings() -> set[str]:
    from wavelab.validation.csv_import import ImportError_, parse_signals_csv

    out: set[str] = set()
    for text in _CSV_CASES:
        out.update(parse_signals_csv(text).report)
    for text in _CSV_REJECTED:
        with pytest.raises(ImportError_) as e:
            parse_signals_csv(text)
        out.add(str(e.value))
    return out


_RULES = (
    "[1, 2, 3] > close",                 # a node type with no visitor
    "close > 'x'",                       # a constant that is not a number
    "closs > open",                      # unknown series name
    "close // 2 > open",                 # operator off the allowlist
    "close is close",                    # comparison off the allowlist
    "close > open > low",                # a chained comparison
    "close.mean() > open",               # a call whose callee is not a plain name
    "rsi(close, n=14) > 70",             # keyword arguments
    "foo(close) > open",                 # unknown function
    "close >",                           # a syntax error
    "sma(close) > open",                 # blows up inside a permitted function
    "shift(close, -1) > close",          # the look-ahead refusal
    "1 > 0",                             # a constant, not a series
)


def _expr_strings() -> set[str]:
    from wavelab.validation.expr import (
        FUNC_DOCS,
        SERIES_DOCS,
        ExprError,
        build_series,
        evaluate_rule,
    )

    rng = np.random.default_rng(9)
    n = 300
    c = 30_000 * np.exp(np.cumsum(rng.standard_normal(n) * 0.02))
    sp = np.abs(rng.standard_normal(n)) * c * 0.01
    o = np.concatenate([[c[0]], c[:-1]])
    series = build_series(o, c + sp, c - sp, c, np.abs(rng.standard_normal(n)) * 100 + 10)

    out = set(SERIES_DOCS.values()) | set(FUNC_DOCS.values())
    for rule in _RULES:
        with pytest.raises(ExprError) as e:
            evaluate_rule(series, rule)
        out.add(str(e.value))
    return out


def _plan_strings() -> set[str]:
    from wavelab.core.types import Direction, Pivot, PivotKind
    from wavelab.waves.matcher import Hypothesis
    from wavelab.waves.projection import PlanConfig, build_plan
    from wavelab.waves.rules import ImpulseState

    cfg = PlanConfig()
    piv = (Pivot(0, 1_500_000_000_000, 100.0, PivotKind.LOW, 1.0),)

    def hyp(state, archetype, points, invalidation):
        return Hypothesis(state=state, direction=Direction.LONG, pivots=piv, points=points,
                          score=1.0, invalidation_price=invalidation,
                          invalidation_rule="close below", archetype=archetype)

    out: set[str] = set()

    def collect(h, price, atr):
        out.update(build_plan(h, price, atr, cfg).reasons)

    # Inside wave 3 there is no entry to offer, only management.
    collect(hyp(ImpulseState.AT_3, None, (100.0, 110.0, 105.0, 108.0), 99.0), 106.0, 1.0)

    # A wave 4 zone truncated until nothing is left of it: P1 sits above the whole retracement.
    collect(hyp(ImpulseState.AT_4, "w4", (100.0, 500.0, 200.0, 300.0, 250.0), 99.0), 260.0, 1.0)

    # The healthy wave 2 geometry every remaining rejection is a variation on.
    w2 = (100.0, 110.0, 105.0)
    lo, hi = 110.0 - 10.0 * 0.786, 110.0 - 10.0 * 0.500
    entry = (lo + hi) / 2.0

    def buffer_at(price: float, atr: float) -> float:
        # The same three-way max build_plan uses, read off the config object rather than retyped,
        # so a changed cushion moves these scenarios instead of quietly invalidating them.
        return max(cfg.stop_buffer_atr * atr, 2 * cfg.tick_size, cfg.stop_buffer_pct * price)

    # Stop exactly on the entry: zero risk, which is the far-side-of-the-stop refusal.
    collect(hyp(ImpulseState.AT_2, "w2", w2, entry + buffer_at(106.0, 1.0)), 106.0, 1.0)

    # A tiny ATR makes the stop many ATRs away: the size is cut, the stop is never tightened.
    collect(hyp(ImpulseState.AT_2, "w2", w2, 90.0), 106.0, 0.05)

    # A huge ATR buries the stop in the noise floor.
    collect(hyp(ImpulseState.AT_2, "w2", w2, entry - 0.01), 106.0, 40.0)

    # A stop close enough that the round trip in fees eats more than a fifth of R.
    atr = 0.4
    risk = 0.6
    collect(hyp(ImpulseState.AT_2, "w2", w2, entry + buffer_at(106.0, atr) - risk), 106.0, atr)

    # A viable plan the price has walked away from.
    collect(hyp(ImpulseState.AT_2, "w2", w2, 100.0), 130.0, 2.0)
    return out


def _limit_strings() -> set[str]:
    from wavelab.server.limits import RateLimiter, TooMany

    out: set[str] = set()
    for kwargs in ({"per_minute": 2, "per_hour": 999}, {"per_minute": 999, "per_hour": 2}):
        lim = RateLimiter(**kwargs)
        with pytest.raises(TooMany) as e:
            for _ in range(10):
                lim._check_ip("203.0.113.7")
        out.add(str(e.value))
    assert len(out) == 2, "the per-minute and per-hour refusals now read the same"
    return out


def _label_strings() -> set[str]:
    """`Hypothesis.terminal_label`: the heading of every hypothesis card on the decision page.

    It lives in waves/matcher.py, which is NOT a speaking module — matcher.py holds no other
    user-facing prose, and a whole AST visitor for one property returning a dict subscript would
    be more machinery than the thing it watches. So it is harvested by RUNNING the property over
    every ImpulseState instead: a reworded label, or a new state, changes what comes back here.

    It reaches a reader twice, and was English in both places until this was added: `live.py` puts
    it in the payload as `label`, and projection.py interpolates it into the "state '…' offers no
    entry" refusal, where the surrounding sentence WAS translated and the label inside it was not.
    """
    from wavelab.core.types import Direction, Pivot, PivotKind
    from wavelab.waves.matcher import Hypothesis
    from wavelab.waves.rules import ImpulseState

    piv = (Pivot(0, 1_500_000_000_000, 100.0, PivotKind.LOW, 1.0),)
    out = {
        Hypothesis(state=state, direction=Direction.LONG, pivots=piv,
                   points=(100.0, 110.0, 105.0), score=1.0, invalidation_price=99.0,
                   invalidation_rule="close below", archetype=None).terminal_label
        for state in ImpulseState
    }
    assert len(out) == len(ImpulseState), "two states now share a chart label"
    return out


@pytest.fixture(scope="module")
def emitted() -> set[str]:
    """Sentences the running code actually produced. Proof the specimens below are not fiction."""
    return (_battery_strings() | _csv_strings() | _expr_strings()
            | _plan_strings() | _limit_strings() | _label_strings())


# --------------------------------------------------- harvest 2: one specimen per emit site, from
#                                                                the AST of the speaking modules

def _texts(node: ast.AST | None) -> list[str]:
    """The user-visible sentence(s) an expression evaluates to, with FILL for interpolations."""
    if node is None:
        return []
    if isinstance(node, ast.Constant):
        return [node.value] if isinstance(node.value, str) and node.value.strip() else []
    if isinstance(node, ast.JoinedStr):
        s = "".join(v.value if isinstance(v, ast.Constant) else FILL for v in node.values)
        return [s] if s.strip() else []
    if isinstance(node, ast.IfExp):                       # `a if cond else b`: both are emitted
        return _texts(node.body) + _texts(node.orelse)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return [t for e in node.elts for t in _texts(e)]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _texts(node.left) + _texts(node.right)     # `imp.report + [...]`, a list of lines
    return []                                             # str(e), a name, a call: not our prose


#: Exceptions whose message is defined to go straight to the user. `raise ValueError(...)` is not
#: here on purpose: only these four reach a browser.
_USER_ERRORS = {"ImportError_", "ExprError", "TooBusy", "TooMany"}
#: Dict keys whose value the JS puts through `tx()` (see web/assay.js and web/app.js).
_SPEAKING_KEYS = {"error", "report", "reasons"}
#: `Test(id, title, passed, value, reference, unit, explanation, detail)` — the three that are read.
_TEST_PROSE_ARGS = (1, 6, 7)


class _EmitSites(ast.NodeVisitor):
    def __init__(self, module: str) -> None:
        self.module = module
        self.found: list[tuple[str, int, str]] = []

    def _add(self, node: ast.AST | None, line: int) -> None:
        for text in _texts(node):
            self.found.append((self.module, line, text))

    def visit_Raise(self, node: ast.Raise) -> None:
        exc = node.exc
        if (isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name)
                and exc.func.id in _USER_ERRORS and exc.args):
            self._add(exc.args[0], node.lineno)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # `report = imp.report + [ … ]` in app.py: the endpoint's own lines, added to the
        # importer's, are the ones the reader sees under "what we understood".
        if any(isinstance(t, ast.Name) and t.id in ("report", "reasons") for t in node.targets):
            self._add(node.value, node.lineno)
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for k, v in zip(node.keys, node.values, strict=True):
            if isinstance(k, ast.Constant) and k.value in _SPEAKING_KEYS:
                self._add(v, node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr == "append"
                and isinstance(f.value, ast.Name) and f.value.id in ("report", "reasons")):
            self._add(node.args[0] if node.args else None, node.lineno)
        elif isinstance(f, ast.Name):
            if f.id == "Test":
                for k in _TEST_PROSE_ARGS:
                    if len(node.args) > k:
                        self._add(node.args[k], node.lineno)
            elif f.id == "PlanResult" and node.args:
                self._add(node.args[-1], node.lineno)          # the tuple of refusals
            elif f.id == "_series_and_battery" and len(node.args) > 2:
                self._add(node.args[2], node.lineno)           # the heading of the report
            elif f.id == "_battery_own_series" and len(node.args) > 3:
                self._add(node.args[3], node.lineno)
        self.generic_visit(node)


def _specimens() -> list[tuple[str, int, str]]:
    out: list[tuple[str, int, str]] = []
    for rel in SPEAKING_MODULES:
        path = PKG / rel
        assert path.exists(), f"{rel} has moved: this test is no longer watching it"
        v = _EmitSites(rel)
        v.visit(ast.parse(path.read_text(encoding="utf-8")))
        assert v.found, f"no user-facing string found in {rel}: has an emit site changed shape?"
        out.extend(v.found)
    return out


SPECIMENS = _specimens()


# ------------------------------------------------------------------------------------- the pin

@pytest.mark.parametrize("lang", LANGS)
def test_every_emitted_string_is_translated(emitted, lang):
    """The whole point. A miss here is an English line on a page that is not in English."""
    missing = sorted(s for s in emitted if tx(s, lang) is None)
    assert not missing, (
        f"{len(missing)} sentence(s) the server actually emitted have no {lang} version and "
        f"will be read in English:\n" + "\n".join(f"  {s!r}" for s in missing))


@pytest.mark.parametrize("lang", LANGS)
def test_every_emit_site_is_translated(lang):
    """Catches the sites a test cannot run, and turns a REWORDING red instead of silent."""
    missing = [(m, ln, s) for m, ln, s in SPECIMENS if tx(s, lang) is None]
    assert not missing, (
        f"{len(missing)} emit site(s) produce a sentence with no {lang} version. Either add it to "
        f"SERVER_{lang.upper()} / PATTERNS in web/i18n.js, or the text was reworded and the "
        f"existing entry no longer matches:\n"
        + "\n".join(f"  {m}:{ln}  {s!r}" for m, ln, s in missing))


def test_no_table_entry_is_orphaned(emitted):
    """The other direction: prose that no longer matches anything the server can say.

    A reworded message can be "fixed" by adding a new entry beside the old one, and nothing would
    complain. Do that a few times and the tables become a layer of dead prose thick enough to hide
    the next real hole in.
    """
    reachable = {_hit(s) for s in emitted} | {_hit(s) for _, _, s in SPECIMENS}
    # Only the es keys are walked because test_both_languages_cover_the_same_strings already
    # pins the two key sets equal; walking both would report every orphan twice.
    orphans = [k for k in SERVER["es"] if f"table:{k}" not in reachable]
    orphans += [f"/{rx.pattern}/" for rx, _ in PATTERNS if f"pattern:{rx.pattern}" not in reachable]
    assert not orphans, (
        f"{len(orphans)} entr(y/ies) in i18n.js match nothing the server emits. Either the Python "
        f"was reworded and this is the stale half, or the message was deleted:\n"
        + "\n".join(f"  {o!r}" for o in orphans))


# ---------------------------------------------------------------- the shape of the tables itself

def test_the_ui_dictionaries_cover_the_same_keys():
    """The interface's own words: `t(key)` reads these, and a missing key renders the raw key.

    This is a KEY check and nothing more — see "WHAT THIS MODULE DOES NOT COVER" at the top. It
    cannot tell a translated value from an English one left in place. It is here because `t()`
    falls back to printing the key itself, so a dropped key does not show up as English prose, it
    shows up as `hyp.invalidation` in the middle of the page.
    """
    dicts = {}
    src = I18N_JS.read_text(encoding="utf-8")
    for name in ("EN", "ES", "CA"):
        decl = f"const {name} = "
        assert decl in src, f"the {name} interface dictionary has gone from i18n.js"
        dicts[name], _ = _read_object(src, _skip(src, src.index(decl) + len(decl)))
    problems = []
    for name in ("ES", "CA"):
        for k in sorted(set(dicts["EN"]) - set(dicts[name])):
            problems.append(f"{name} is missing {k!r}: the page will print the key itself")
        for k in sorted(set(dicts[name]) - set(dicts["EN"])):
            problems.append(f"{name} has {k!r}, which EN does not: dead or misspelt")
    assert not problems, "\n".join(problems)


def test_both_languages_cover_the_same_strings():
    only_es = sorted(set(SERVER["es"]) - set(SERVER["ca"]))
    only_ca = sorted(set(SERVER["ca"]) - set(SERVER["es"]))
    assert not only_es and not only_ca, (
        "SERVER_ES and SERVER_CA have drifted apart; a Catalan reader gets English for:\n"
        + "\n".join(f"  ca missing: {s!r}" for s in only_es)
        + "\n".join(f"  es missing: {s!r}" for s in only_ca))


def test_every_pattern_is_anchored_and_complete():
    """Anchoring is what keeps the list analysable, and both languages or neither."""
    problems = []
    for rx, tr in PATTERNS:
        p = rx.pattern
        if not (p.startswith("^") and p.endswith("$")):
            problems.append(f"/{p}/ is not anchored: it can match the middle of another message")
        for lang in LANGS:
            if not tr.get(lang):
                problems.append(f"/{p}/ has no {lang} translation")
                continue
            used = {int(d) for d in re.findall(r"\$(\d)", tr[lang])}
            wanted = set(range(1, rx.groups + 1))
            if used != wanted:
                problems.append(
                    f"/{p}/ captures {sorted(wanted)} but its {lang} text uses {sorted(used)}: "
                    "a number the server computed would be dropped or duplicated")
    assert not problems, "\n".join(problems)


# ------------------------------------------------------------------------------- pattern order

#: The capture shapes the specimen builder knows how to fill. Anything else raises rather than
#: being guessed at, so a new kind of group forces this helper to be extended instead of quietly
#: turning the shadowing check into a no-op for that pattern.
_KNOWN_GROUPS = {
    "(.+?)": FILL,
    "(.+)": FILL,
    "(.*)": FILL,
    r"(\S+?)": FILL,
    r"(\$.+?)": "$" + FILL,
}
_GROUP = re.compile(r"\((?:[^()\\]|\\.)*\)")


def _specimen_of(pattern: str) -> str:
    """A message that this pattern, and only a pattern at least as broad, would match."""
    out: list[str] = []
    pos = 0
    for m in _GROUP.finditer(pattern):
        assert m.group(0) in _KNOWN_GROUPS, (
            f"/{pattern}/ uses the capture {m.group(0)}, which _KNOWN_GROUPS does not know how to "
            "fill. Add it there — leaving it out would silently exempt this pattern from the "
            "shadowing check.")
        out.append(pattern[pos:m.start()])
        out.append(_KNOWN_GROUPS[m.group(0)])
        pos = m.end()
    out.append(pattern[pos:])
    literal = "".join(out)
    literal = literal.removeprefix("^").removesuffix("$")
    return re.sub(r"\\(.)", lambda esc: esc.group(1), literal)


def test_no_pattern_shadows_a_later_one():
    """PATTERNS is ordered and `tx()` takes the first hit.

    A broad pattern placed above a narrow one swallows the narrow one's messages and renders them
    with the wrong words — wrong to the reader, invisible to everyone else, and impossible to spot
    by looking at either pattern on its own. The list has to be checked as a list.
    """
    problems = []
    for j, (rx, _) in enumerate(PATTERNS):
        sample = _specimen_of(rx.pattern)
        assert rx.search(sample), (
            f"the specimen built for /{rx.pattern}/ does not match it: _specimen_of is wrong, "
            "and every shadowing check below it is meaningless")
        for i in range(j):
            if PATTERNS[i][0].search(sample):
                problems.append(
                    f"pattern #{i} /{PATTERNS[i][0].pattern}/ sits above #{j} /{rx.pattern}/ and "
                    f"also matches {sample!r}. #{j} is unreachable: its messages come out in "
                    f"#{i}'s words. Move the narrower pattern above the broader one.")
    assert not problems, "\n".join(problems)


def test_the_bold_is_on_the_ratio_the_reader_sizes_on_in_every_language():
    """★ Which of the two R:R figures is bold, in all three dictionaries.

    A viable card out of the entry zone prints two ratios: `{now}`, quoted at a fill you can always
    get by buying at market, and `{zone}`, which needs a resting limit order to fill and may never
    happen. The whole point of showing both is that the reader sizes on the first one, and on a
    card of eleven numbers the thing that decides which one they read is the weight of the type.

    Nothing guarded this. The template can be edited to bold `{zone}` instead — the change is two
    pairs of tags in one string, it reads perfectly well in review, it renders without an error,
    the key check above still passes because the key is still there, and the reader is then sizing
    on a number that depends on an order nobody promised them. Applied to `hyp.rr_now_zone` in EN,
    the whole suite stayed green.

    `{zone}` is the larger figure most of the time but not always — below the entry zone a long is
    buying cheaper than the midpoint the zone figure is quoted at, and the headline is then the
    bigger number. So this cannot be written as "the smaller one is bold". It is positional: the
    bold belongs to `{now}`, wherever it lands.
    """
    src = I18N_JS.read_text(encoding="utf-8")
    dicts = {}
    for name in ("EN", "ES", "CA"):
        decl = f"const {name} = "
        assert decl in src, f"the {name} interface dictionary has gone from i18n.js"
        dicts[name], _ = _read_object(src, _skip(src, src.index(decl) + len(decl)))

    problems = []
    for name, d in dicts.items():
        tpl = d.get("hyp.rr_now_zone")
        if tpl is None:
            problems.append(f"{name} has no 'hyp.rr_now_zone': the second ratio is gone from "
                            f"{name} cards and the reader is back to one number")
            continue
        for slot in ("{now}", "{zone}"):
            if slot not in tpl:
                problems.append(f"{name} 'hyp.rr_now_zone' has no {slot} slot: {tpl!r}")
        if "{now}" not in tpl or "{zone}" not in tpl:
            continue
        if "<b>{now}</b>" not in tpl:
            problems.append(
                f"{name}: the headline ratio is not bold in {tpl!r}. `{{now}}` is the number the "
                "reader sizes on — it is the one they can get by buying at market — and on this "
                "card the type weight is what tells them which of the two that is")
        if "<b>{zone}</b>" in tpl:
            problems.append(
                f"{name}: the IN-ZONE ratio is bold in {tpl!r}. That figure needs a limit order to "
                "fill and may never be reached; bolding it points the reader at the trade they "
                "may not get, which is the exact bias this card was rebuilt to remove")
    assert not problems, "\n".join(problems)


def test_the_arithmetic_line_leaves_the_ratio_markup_to_the_branch_that_fills_it():
    """The other half of the same guarantee, one level up.

    `hyp.arith` used to wrap its first slot in `<b>…</b>` itself. It cannot any more: `app.js`
    fills that slot with EITHER a single bold figure OR the whole two-ratio sentence, and a `<b>`
    on the outside would bold both ratios together and undo the test above without touching the
    string it checks.
    """
    src = I18N_JS.read_text(encoding="utf-8")
    problems = []
    for name in ("EN", "ES", "CA"):
        decl = f"const {name} = "
        d, _ = _read_object(src, _skip(src, src.index(decl) + len(decl)))
        tpl = d.get("hyp.arith", "")
        assert "{rr}" in tpl, f"{name} 'hyp.arith' no longer has the {{rr}} slot: {tpl!r}"
        if "<b>{rr}</b>" in tpl:
            problems.append(
                f"{name}: 'hyp.arith' bolds {{rr}} itself ({tpl!r}). That slot receives the whole "
                "'X now — Y if your limit fills in the zone' sentence, so this bolds the in-zone "
                "figure along with the headline and the reader can no longer tell them apart")
    assert not problems, "\n".join(problems)
