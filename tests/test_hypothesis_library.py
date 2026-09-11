"""The hundred hypotheses, run rather than read.

`test_hypothesis_i18n.py` checks that their prose has not drifted. Nothing checked that the code
underneath it computes anything, and that code is ~5,400 lines producing the signal series every
verdict in the product is derived from. A hypothesis whose `signals()` returns a NaN, a 0.5, or a
value that peeks at the next candle is caught today only by `evaluate_all`'s bare `except`, which
prints a line and moves on.

A hundred hand-written tests, one per hypothesis, would be a hundred restatements of each
function's own arithmetic — they would agree with the code because they were copied from it. What
is asserted here instead are the properties that must hold for ALL of them, over the whole
registry, and that are therefore not copied from any one of them:

  1. **Causality.** The signal at bar `i` may not depend on bar `i+1`. The sharp form is
     `acausal_probe`: truncate the series at i+1, so bar i is the last candle in existence, and
     require the signal there to be what the full series said — twelve bars per hypothesis per
     timeframe, two thirds of them spent on bars where the hypothesis is in the market. Two
     coarser arms run under it over the whole registry at once: one truncating, one rewriting the
     future in place while holding the length constant. All of them carry a regime change — a
     tripling of volatility, a reversal of drift, a sevenfold jump in volume — on the far side of
     the cut, so a lookahead has something worth leaking. This is the project's entire reason to
     exist and it had no test at the hundred places where it matters most.
  2. **The alphabet.** {-1, 0, +1}, finite, one value per candle, checked on the RAW output of
     `fn` before `signals()` launders it: the int8 cast turns 0.5 into 0 and a NaN into an
     arbitrary position, and both then produce a verdict with no error anywhere.
  3. **The declared warm-up.** No signal before the candles it said it needed — and, as a side
     effect of where the causality probes start, no hypothesis declaring a warm-up shallower than
     the number of candles its own function refuses to compute below.
  4. **Scale.** Multiplying every price by a constant must not move a signal. Measured answer for
     this registry: all 100 are scale-free, on a power-of-two factor (where the arithmetic is
     exact) and on 3.7 (where it is not). The test exists to catch the first hypothesis that
     compares a price against an absolute number.
  5. **Purity.** It does not write to the candles it was handed, and it answers the same thing
     twice. `evaluate_all` runs all hundred over the same arrays, in registry order.
  6. **Degeneracy.** An empty series, a two-candle series and a market that never moves must not
     raise and must not produce a RuntimeWarning (`filterwarnings` in pyproject makes one a
     failure, because that warning is how a NaN becomes a position).
  7. **The shared helpers.** The six properties above are blind to arithmetic. That is the right
     trade for a hundred hand-written hypothesis functions and the wrong one for the two dozen
     UTILITIES underneath them: `_pivots`, `_clv`, `_rolling_rank`, `_prior_extreme`, the three
     `_hold` variants and the calendar helpers are each shared by between two and eight
     hypotheses, so one fault there is not one hypothesis wrong, it is a family. And each of them
     computes something whose definition exists outside this repository — a Williams fractal, a
     percentile, a weekday — so a test for one is an oracle and not a restatement of its own code.
     Measured example: inverting the sign of `_clv` leaves every `flow` hypothesis causal, finite,
     scale-free, pure and two-sided, and silently reads selling pressure as buying pressure.

Everything runs over synthetic series with known structure AND over real BTC candles out of the
store, resampled through the same `resample_from_1m` the server uses. The real arm is not
decoration: a planted one-bar lookahead was caught there before the synthetic probe existed.

WHAT IS NOT HERE. No property in this file looks at what any INDIVIDUAL hypothesis computes. A
Bollinger band built at one sigma instead of two, an RSI threshold of 40 instead of 30, a sign
flipped end to end — all of those are causal, finite, scale-free and pure, and all of them pass.
Two such faults were planted to confirm it and both survived. That boundary is deliberate: the
alternative is a hundred restatements of a hundred formulas. The line is drawn UNDER the shared
helpers rather than under the whole library, because a helper's answer is defined outside this
repository and is reached by a whole family, while one hypothesis's arithmetic is defined by that
hypothesis and reached by nothing else.

The harness's teeth are not claimed, they are demonstrated. `TestTheHarnessHasTeeth` fabricates
hypotheses that read one candle ahead, that threshold against a statistic of the whole array, that
`np.roll` the series backwards, and that read the last candle of whatever array they are given, and
requires the same checkers used on the registry to catch each one — plus one causal function of the
same shape that must come back clean, because a checker that is always red is decoration too.

Three things turned out NOT to hold and are recorded rather than smoothed over: `min_warmup` is a
mask on the head of the series and not a bound on how far back a signal reads; four price-driven
hypotheses read a market with zero movement as maximally oversold; and `candles.range_gap` fires so
rarely that nine years of BTC hold 102 events of it and the fixture window holds none.
"""

from __future__ import annotations

import calendar
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import talib

from wavelab.core.timeframes import BY_NAME
from wavelab.hypotheses import candles, flow, load_all, momentum, seasonality, structure, volatility
from wavelab.hypotheses.base import Hypothesis, Series, register

REGISTRY = load_all()
NAMES = sorted(REGISTRY)

#: Long enough that the deepest warm-up in the registry (600) leaves 900 bars to compare over.
N_SYNTH = 1500
#: Where the synthetic series is cut in two. The regime change lives immediately after it.
CUT_SYNTH = 1000

#: Length of the quiet market whose only event is its last candle. Above the deepest declared
#: warm-up (600) by enough that `acausal_probe` has a hundred traded bars to spend its probes on.
N_QUIET = 750

#: Real candles per timeframe. 1,600 covers the deepest warm-up (600) with a thousand left to
#: probe, and keeps the fixture inside a fraction of a second on all four timeframes.
#: Allowed to be silent over any single window without that meaning it is dead.
#:
#: A full-range gap needs the order book to empty. Nine years of BTC contain 102 of them — 99 on
#: 15m, 3 on 1h, none at all on 4h — so a 1,600-candle window containing none is the normal case,
#: not a broken hypothesis. Adding a name here is a claim that it is rare, not that it is allowed
#: to be broken; the burden is on whoever adds it to show the event happens at all.
RARE_ENOUGH_TO_BE_SILENT = {"candles.range_gap"}

N_REAL = 1600
#: Where the real series is truncated for the broad net. Well above MIN_PREFIX.
CUT_REAL = 1200

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "bars"
SYMBOL = "BTCUSDT"
#: Enough months that 1d — the timeframe with the deepest warm-up — gets its full N_REAL bars.
REAL_MONTHS = 56
#: The window ENDS here, at a fixed instant, and not at the newest bar in the store.
#:
#: It used to end at the newest bar, which made these properties a statement about whatever the
#: market had done that morning. `test_every_hypothesis_takes_a_position_at_some_point` pinned the
#: exact set of hypotheses that stayed quiet, and two days after it was written a full-range gap
#: arrived, `candles.range_gap` fired, the set became empty and the deploy gate went red on a
#: repository nobody had touched. A test that gates deploys and answers differently depending on
#: the hour is worse than no test: it teaches people that red means "run it again".
#:
#: 2026-08-31 23:59 UTC — the last minute of the last complete month at the time of writing.
#: Everything after it is live data that keeps arriving.
REAL_END_MS = 1788220740000


# ------------------------------------------------------------------------------------------
# Synthetic candles
# ------------------------------------------------------------------------------------------
def synthetic(n: int, tf: str, *, seed: int = 11, shock_at: int | None = None) -> Series:
    """A seeded geometric random walk, optionally with a regime change bolted onto its tail.

    The shock is what gives the causality test something to detect. A hypothesis that reads one bar
    ahead on a series whose statistics never change can agree with its own prefix by luck; one that
    reads ahead across a tripling of volatility, a reversal of drift and a sevenfold jump in volume
    cannot. It is sized as a bear market — 3x the volatility and a drift that takes roughly 95% off
    over five hundred candles — and not as a bigger number, because a shock steep enough to carry
    the price to 1e10 puts every threshold in the library permanently on one side of itself and
    quietly stops testing them.

    Each timeframe draws its own path. With one shared seed the four series differ only in their
    timestamps, so a property that holds on one price path was being asserted four times instead of
    being asserted over four price paths — which is how a hypothesis gated on an absolute price of
    $30,000 survived the scale test in this file's first draft.
    """
    rng = np.random.default_rng(seed * 1_000 + BY_NAME[tf].minutes)
    r = rng.standard_normal(n) * 0.01
    vol = np.abs(rng.standard_normal(n)) * 100.0 + 10.0
    if shock_at is not None:
        r[shock_at:] = r[shock_at:] * 3.0 - 0.006
        vol[shock_at:] *= 7.0
    close = 30_000.0 * np.exp(np.cumsum(r))
    spread = np.abs(rng.standard_normal(n)) * close * 0.003
    open_ = np.concatenate([[30_000.0], close[:-1]])
    step = BY_NAME[tf].ms
    t0 = 1_600_000_000_000 - (1_600_000_000_000 % step)
    return Series(
        tf=tf,
        ts=(t0 + np.arange(n, dtype=np.int64) * step).astype(np.int64),
        open=open_,
        high=np.maximum(open_, close) + spread,
        low=np.minimum(open_, close) - spread,
        close=close,
        volume=vol,
    )


def flat(n: int, tf: str, price: float = 30_000.0, volume: float = 100.0) -> Series:
    """A market that never moves: every OHLC value identical. Every ratio in the library that
    normalises by a range divides zero by zero here."""
    a = np.full(n, price)
    step = BY_NAME[tf].ms
    t0 = 1_600_000_000_000 - (1_600_000_000_000 % step)
    return Series(tf, (t0 + np.arange(n, dtype=np.int64) * step).astype(np.int64),
                  a.copy(), a.copy(), a.copy(), a.copy(), np.full(n, volume))


def quiet_then_break(n: int, tf: str, *, price: float = 30_000.0, up: bool = True) -> Series:
    """A dead market whose ONLY event is the very last candle in the series.

    Every other fixture in this file hands the event-driven hypotheses their first breakout well
    inside their own warm-up, so `signals()`'s mask zeroes exactly the bars on which "what do I
    hold before anything has happened to me?" is answered. Here nothing happens at all for `n-1`
    candles — no new high, no new low, no engulfing, no gap, no event of any kind — and the single
    break is the last candle, several hundred bars past the deepest declared warm-up. That makes
    the pre-event region a TRADED region instead of a masked one, and it is the only region in
    which a fill that reads the last element of an array instead of the first is observable at all.
    """
    step = BY_NAME[tf].ms
    t0 = 1_600_000_000_000 - (1_600_000_000_000 % step)
    a = np.full(n, price)
    o, h, l, c = a.copy(), a.copy(), a.copy(), a.copy()
    broken = price * (1.02 if up else 0.98)
    c[-1] = broken
    (h if up else l)[-1] = broken
    return Series(tf, (t0 + np.arange(n, dtype=np.int64) * step).astype(np.int64),
                  o, h, l, c, np.full(n, 100.0))


def head(s: Series, k: int) -> Series:
    """The first k candles, as if the rest had not happened yet."""
    return Series(s.tf, s.ts[:k], s.open[:k], s.high[:k], s.low[:k], s.close[:k], s.volume[:k])


def tail(s: Series, k: int) -> Series:
    """Everything from k onwards, as if the series had started there."""
    return Series(s.tf, s.ts[k:], s.open[k:], s.high[k:], s.low[k:], s.close[k:], s.volume[k:])


def rewritten_future(s: Series, k: int, *, seed: int) -> Series:
    """The same candles up to k, a completely different market afterwards, same total length.

    The companion to the prefix cut. Truncating changes the array length, and a function that
    branches on `len(s)` — most of this library does — takes a different path for that reason
    alone. Here the length, the timestamps and the first k candles are untouched, so the only thing
    that can move a signal at i < k is a read of data at j >= k.
    """
    other = synthetic(len(s), s.tf, seed=seed)
    scale = s.close[k - 1] / other.close[k - 1]
    def splice(a: np.ndarray, b: np.ndarray, mul: float) -> np.ndarray:
        out = a.copy()
        out[k:] = b[k:] * mul
        return out
    return Series(
        s.tf, s.ts,
        splice(s.open, other.open, scale), splice(s.high, other.high, scale),
        splice(s.low, other.low, scale), splice(s.close, other.close, scale),
        splice(s.volume, other.volume, 3.0),
    )


def scaled(s: Series, k: float) -> Series:
    """Every price multiplied by k. Volume and the clock are left alone."""
    return Series(s.tf, s.ts, s.open * k, s.high * k, s.low * k, s.close * k, s.volume)


# ------------------------------------------------------------------------------------------
# The checkers. The registry tests and the teeth tests call exactly these.
# ------------------------------------------------------------------------------------------
#: Below this many candles the `candles` family declines to compute at all (`_MIN_BARS = 320`,
#: the largest such floor in the library), so a shorter prefix is legitimately all zeros while the
#: full series is not. That is a refusal, not a lookahead, and the prefix arm has to stay above it.
#: The rewrite arm holds the length constant and is therefore valid at any cut.
MIN_PREFIX = 320


class Cuts:
    """The two altered copies of a series, built once and shared by all hundred hypotheses.

    `head` and `rewritten_future` produce the same arrays for every hypothesis at a given
    (series, cut). Rebuilding them 293 times per cut was most of this file's runtime, on a suite
    that gates deploys.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int, str], Series] = {}

    def prefix(self, s: Series, cut: int) -> Series:
        return self._cache.setdefault((id(s), cut, "p"), head(s, cut))

    def rewritten(self, s: Series, cut: int, seed: int = 4242) -> Series:
        return self._cache.setdefault((id(s), cut, f"r{seed}"),
                                      rewritten_future(s, cut, seed=seed))


class Signals:
    """`h.signals(s)` on the UNALTERED series, memoised.

    Nine properties times a hundred hypotheses times the timeframes each declares is several
    thousand recomputations of one answer, on a suite that gates deploys. The altered copies are
    deliberately not memoised — each cut is a different question — and `TestPurity` calls
    `h.signals` directly, because caching the thing it is testing for determinism would be the
    definition of a test that cannot fail.

    The key is `(name, series)`, so it is only safe while names are unique. Four fabricated
    lookaheads in `TestTheHarnessHasTeeth` once shared the name `teeth.probe` and were handed each
    other's answers, which turned a deliberate lookahead green.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    def of(self, h: Hypothesis, s: Series) -> np.ndarray:
        key = (h.name, id(s))
        if key not in self._cache:
            self._cache[key] = h.signals(s)
        return self._cache[key]


#: Module-level rather than a fixture: `Series` objects come from session fixtures and live for the
#: whole session, so the ids used as keys stay valid and keep pointing at the same candles.
CUTS = Cuts()
SIGNALS = Signals()


def prefix_disagreement(h: Hypothesis, s: Series, cut: int) -> np.ndarray:
    """Bars before `cut` whose signal moved when the candles from `cut` on were never supplied."""
    assert cut >= MIN_PREFIX, f"a prefix of {cut} is below the library's own {MIN_PREFIX} floor"
    return np.flatnonzero(SIGNALS.of(h, s)[:cut] != h.signals(CUTS.prefix(s, cut)))


def rewrite_disagreement(h: Hypothesis, s: Series, cut: int, *, seed: int = 4242) -> np.ndarray:
    """Bars before `cut` whose signal moved when the candles from `cut` on were replaced.

    The complement of the truncating arm rather than a duplicate of it. Truncating changes
    `len(s)`, and most of this library branches on it, so a prefix can take a different path for a
    reason that has nothing to do with causality — and cannot be pushed below MIN_PREFIX at all.
    Here the length, the timestamps and every candle before `cut` are byte-identical, so the only
    thing that can move a signal at i < cut is a read of data at j >= cut, and the cut can go
    anywhere, including bar 60 and the last bar in the series.
    """
    full = SIGNALS.of(h, s)
    rewritten = h.signals(CUTS.rewritten(s, cut, seed))
    return np.flatnonzero(full[:cut] != rewritten[:cut])


#: How many bars each hypothesis is probed at, per timeframe. Two thirds of them are spent on bars
#: where it holds a position, because that is where a stolen candle shows up: a lookahead that
#: turns a +1 into a 0 is invisible on a bar that was 0 either way.
N_PROBES = 12


def probe_bars(full: np.ndarray, lo: int, hi: int, k: int = N_PROBES) -> list[int]:
    """`k` bars in [lo, hi), spread evenly, biased towards bars carrying a position."""
    idx = np.arange(lo, hi)
    active, quiet = idx[full[idx] != 0], idx[full[idx] == 0]
    n_active = (2 * k) // 3
    out: list[int] = []
    for pool, want in ((active, n_active), (quiet, k - n_active)):
        if pool.size and want:
            take = min(want, pool.size)
            out += pool[np.linspace(0, pool.size - 1, take).astype(int)].tolist()
    return sorted({int(i) for i in out})


def acausal_probe(h: Hypothesis, s: Series) -> int | None:
    """The first bar whose signal changes when the candles after it are taken away, or None.

    This is `signals()`'s own docstring turned into an executable claim: "position i may only
    depend on data up to and including i". For a sample of bars i, the series is truncated at i+1
    — so bar i is the last candle that exists — and the signal there must be what the full series
    said. Nothing else in this file states the property this tightly.

    It also pins a second thing, and callers should know which failure they are looking at: the
    probes start at `max(MIN_PREFIX, min_warmup)`, so a hypothesis whose declared warm-up is
    SHALLOWER than the number of candles its own function refuses to compute below fails here too
    — it answers 0-for-want-of-data at bars it declared itself ready for. Measured: no hypothesis
    in the registry does that today, and dropping `tsmom_12_1`'s warm-up from 420 to 8 (its
    function needs 391) turns this red.
    """
    full = SIGNALS.of(h, s)
    lo = max(MIN_PREFIX, h.min_warmup)
    if lo >= len(s):
        return None
    for i in probe_bars(full, lo, len(s)):
        if h.signals(head(s, i + 1))[i] != full[i]:
            return i
    return None


def acausal_indices(h: Hypothesis, s: Series, cut: int, *, seed: int = 4242) -> np.ndarray:
    """Both arms, unioned. They fail on different mistakes: the prefix arm catches anything that
    reads past the end of the data it was given, the rewrite arm anything that reads the future
    while the array length is held constant."""
    return np.union1d(prefix_disagreement(h, s, cut),
                      rewrite_disagreement(h, s, cut, seed=seed))


def sweep(check, series: dict[str, Series]) -> list[str]:
    """Run `check(hypothesis, series)` over all 100, on every timeframe each of them declares.

    The properties that are not parametrised per hypothesis are run this way instead: 100 pytest
    ids apiece for eight properties is 800 lines of collection on a suite that gates deploys, and
    the failure is named just as precisely by the list this returns. The two causality properties
    ARE parametrised, because that is the one place where a red test id naming the hypothesis is
    worth the collection cost.
    """
    out: list[str] = []
    for name in NAMES:
        h = REGISTRY[name]
        for tf in h.timeframes:
            if tf not in series:
                continue
            for problem in check(h, series[tf]):
                out.append(f"  {name} @ {tf}: {problem}")
    return out


def out_of_alphabet(h: Hypothesis, s: Series) -> list[str]:
    """Everything wrong with the raw output of `h.fn`, before `signals()` launders it.

    `signals()` casts to int8, which turns 0.5 into 0 and a NaN into whatever the cast feels like;
    the fault has to be caught upstream of that or it is caught nowhere.
    """
    raw = np.asarray(h.fn(s), dtype=np.float64)
    problems: list[str] = []
    if raw.ndim != 1 or raw.size != len(s):
        problems.append(f"shape {raw.shape} for {len(s)} candles")
        return problems
    if not np.isfinite(raw).all():
        n_bad = int((~np.isfinite(raw)).sum())
        problems.append(f"{n_bad} non-finite values (a NaN cast to int8 is an arbitrary position)")
    illegal = sorted(set(np.unique(raw[np.isfinite(raw)]).tolist()) - {-1.0, 0.0, 1.0})
    if illegal:
        problems.append(f"values outside {{-1,0,+1}}: {illegal[:8]}")
    return problems


# ------------------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def synth_series() -> dict[str, Series]:
    """One shocked series per timeframe the registry declares. Built once for the whole session."""
    return {tf: synthetic(N_SYNTH, tf, shock_at=CUT_SYNTH) for tf in BY_NAME}


@pytest.fixture(scope="session")
def quiet_series() -> dict[tuple[str, bool], Series]:
    """The quiet market, in both directions, for every timeframe the registry declares.

    Session-scoped for the same reason `synth_series` is: `SIGNALS` and `CUTS` key on `id(series)`,
    so a `Series` built inside a test function and then collected could hand its cached answers to
    whatever object lands on its address next.
    """
    return {(tf, up): quiet_then_break(N_QUIET, tf, up=up)
            for tf in ("15m", "1h", "4h", "1d") for up in (True, False)}


@pytest.fixture(scope="session")
def real_series() -> dict[str, Series]:
    """Real BTC candles out of the store, resampled the way the server resamples them.

    Read-only, and only the last `REAL_MONTHS` files: the whole history is 4.76M 1m rows and
    holding it would cost more memory than the service is allowed. Skipped, loudly, when the store
    is not present — the properties below still run over synthetic candles, but nothing here has
    then been checked against a real market.
    """
    import pandas as pd

    from wavelab.core.timeframes import resample_from_1m
    from wavelab.store.bars import BarStore

    store = BarStore(DATA_ROOT)
    months = store.months(SYMBOL)
    if len(months) < REAL_MONTHS:
        pytest.skip(f"{DATA_ROOT} holds {len(months)} months; {REAL_MONTHS} are needed to give 1d "
                    f"its {N_REAL} candles. The real-candle arm of these properties is NOT running.")
    first = months[-REAL_MONTHS]
    frame = pd.concat([d for key, d in store.iter_months(SYMBOL) if key >= first])
    if int(frame.index[-1]) < REAL_END_MS:
        pytest.skip(f"the store ends before {REAL_END_MS} and these properties are anchored there; "
                    "the real-candle arm is NOT running.")
    last = REAL_END_MS
    out: dict[str, Series] = {}
    for tf in ("15m", "1h", "4h", "1d"):
        # Resample only the minutes each timeframe needs. 1d wants all 56 months; 15m wants
        # seventeen days of them, and resampling 2.4M rows to throw away 99% is a second of the
        # deploy gate spent on nothing.
        window = frame.loc[frame.index >= last - 2 * N_REAL * BY_NAME[tf].ms]
        r = resample_from_1m(window, BY_NAME[tf]).tail(N_REAL)
        assert len(r) == N_REAL, f"only {len(r)} real {tf} candles; the window is too short"
        out[tf] = Series(
            tf, r.index.to_numpy(np.int64),
            r["open"].to_numpy(float), r["high"].to_numpy(float),
            r["low"].to_numpy(float), r["close"].to_numpy(float), r["volume"].to_numpy(float),
        )
    return out


# ------------------------------------------------------------------------------------------
class TestTheRegistryItself:
    def test_the_catalogue_is_the_hundred_that_were_pre_registered(self):
        assert len(REGISTRY) == 100, (
            f"{len(REGISTRY)} hypotheses are registered, not 100. Every one of them counts towards "
            "the multiple-comparisons correction, so the count is part of the record."
        )
        assert len(NAMES) == len(set(NAMES))

    @pytest.mark.parametrize("name", NAMES)
    def test_a_hypothesis_declares_timeframes_the_project_can_actually_resample(self, name):
        """A timeframe outside `BY_NAME` cannot be built from the 1m store, so the hypothesis would
        never run — and `seasonality` would silently fall back to differencing timestamps for its
        hold window instead of using the declared bar length."""
        h = REGISTRY[name]
        assert h.timeframes, f"{name} declares no timeframe and can never be evaluated"
        unknown = sorted(set(h.timeframes) - set(BY_NAME))
        assert not unknown, f"{name} declares {unknown}, which resample_from_1m cannot produce"
        assert h.name.split(".")[0] == h.family, f"{name} is filed under family {h.family!r}"
        assert h.min_warmup > 0, f"{name} declares no warm-up at all"

    def test_a_hypothesis_cannot_be_registered_without_a_usable_title(self):
        """The two guards `base.py` added when titles arrived, neither of which had a test.

        They protect the same failure from two sides and it is a QUIET one: the picker does not
        break on a missing title, it degrades. `hypLabel` in web/assay.js falls back to the bare
        identifier, so hypothesis 101 appears in the catalogue as `family.some_key` while the other
        hundred read as sentences, every test passes, and the only person who finds out is the
        reader who has to decode it. That is the exact screen this whole change replaced, restored
        for one row at a time.

        The field carries no default, so the omission is a `TypeError` at import rather than an
        empty string that reads as a working entry; and `register()` refuses whitespace, which the
        dataclass cannot see. Deleting either one is silent today — the hundred already have
        titles, so nothing else in this suite would notice.
        """
        kwargs = {"name": "teeth.untitled", "family": "teeth", "rationale": "declared up front",
                  "prior": "declared up front",
                  "fn": lambda s: np.zeros(len(s), dtype=np.int8),
                  "timeframes": ("1h",), "min_warmup": 200}

        with pytest.raises(TypeError, match="title"):
            Hypothesis(**kwargs)          # type: ignore[call-arg]

        for blank in ("", "   ", "\n\t"):
            with pytest.raises(ValueError, match="title is empty"):
                register(Hypothesis(title=blank, **kwargs))

        assert "teeth.untitled" not in REGISTRY, (
            "a hypothesis with a blank title was admitted to the registry before the check ran; "
            "the guard has to refuse it, not record it and complain")

    def test_a_hypothesis_cannot_be_registered_without_a_rationale_and_a_prior(self):
        """The older sibling of the title guard, three lines above it in `register()` and, until
        now, the only one of the two with no test at all.

        This is the guard the whole pre-registration argument rests on. A hypothesis admitted with
        a blank `prior` is one nobody committed to a direction for before looking, and it still
        counts towards the multiple-comparisons correction — so it makes every other result look
        worse while contributing nothing that can be falsified. `or` softened to `and` admits it
        whenever EITHER field is filled in, which is the shape the mistake actually takes: the
        rationale gets written, the prior gets left for later, and later never comes.

        Both fields are tested on their own, and with whitespace rather than "", because the
        dataclass cannot tell a space from a sentence.
        """
        kwargs = {"name": "teeth.unpriced", "family": "teeth", "title": "a title it does have",
                  "rationale": "declared up front", "prior": "declared up front",
                  "fn": lambda s: np.zeros(len(s), dtype=np.int8),
                  "timeframes": ("1h",), "min_warmup": 200}

        for field in ("rationale", "prior"):
            for blank in ("", "   "):
                with pytest.raises(ValueError, match="rationale and a prior"):
                    register(Hypothesis(**{**kwargs, field: blank}))
                assert "teeth.unpriced" not in REGISTRY, (
                    f"a hypothesis with a blank {field} reached the registry. Pre-registration is "
                    "the only thing separating this catalogue from a search over a hundred draws, "
                    "and it is enforced here or nowhere.")


# ------------------------------------------------------------------------------------------
class TestEverySignalIsCausal:
    """The one property this project exists to enforce, applied where it was never applied."""

    @pytest.mark.parametrize("name", NAMES)
    def test_a_signal_is_what_it_would_have_been_with_nothing_after_it(self, name, synth_series):
        """The definition, applied at the bar itself rather than in bulk.

        A single cut is a weak instrument against a shallow lookahead: reading `close[i+1]` moves
        exactly ONE bar of the overlap — the one adjacent to the cut — and if that bar happens to
        be flat in both runs, the cut comes back clean. Measured: a deliberate one-bar lookahead
        planted in `mean_reversion.streak3_fade` survived the bulk comparison at four of five cut
        points. So this probes bars instead of ranges, and it spends most of its probes on bars
        where the hypothesis is actually IN the market, which is where a stolen candle changes the
        answer.
        """
        h = REGISTRY[name]
        for tf in h.timeframes:
            bad = acausal_probe(h, synth_series[tf])
            assert bad is None, (
                f"{name} on {tf}: the signal at bar {bad} is "
                f"{int(SIGNALS.of(h, synth_series[tf])[bad])} when the whole series is available "
                f"and something else when the series stops at that bar. Either it reads a candle "
                f"that has not closed, or its declared warm-up of {h.min_warmup} is shallower than "
                f"the number of candles its own function refuses to compute below."
            )

    @pytest.mark.parametrize("name", NAMES)
    def test_the_same_holds_on_real_candles(self, name, real_series):
        """Synthetic candles are smooth by construction. Real BTC has liquidation wicks, thin
        holiday sessions and a stretch of 2021 that a random walk will never produce, and a
        lookahead can hide in any of them. This arm also covers the fourteen hypotheses that never
        fire on the synthetic series, where the probe above has only flat bars to work with."""
        h = REGISTRY[name]
        for tf in h.timeframes:
            bad = acausal_probe(h, real_series[tf])
            assert bad is None, (
                f"{name} on real {tf} candles: the signal at bar {bad} changes once the candles "
                f"after it are removed (declared warm-up {h.min_warmup})."
            )

    @pytest.mark.parametrize("up", [True, False])
    def test_it_holds_before_the_first_event_too(self, up, quiet_series):
        """The bars a hypothesis trades BEFORE anything has happened to it.

        `structure._hold` decides what is held before the first event with `out[idx == 0] = ev[0]`.
        One index away, `ev[-1]`, is the last candle of the entire series — and every other fixture
        in this file is blind to the difference, because on the shocked walk and on real BTC every
        event-driven hypothesis gets its first breakout inside its own warm-up, where `signals()`
        masks the answer to zero regardless. Measured on the fixture below: with `ev[-1]` in place
        of `ev[0]`, `structure.donchian_break_20` holds a position on 199 of the 200 traded bars of
        a market in which, so far, nothing whatsoever has happened.

        What that looks like to a user: a flat, silent stretch of chart carrying a confident
        long — sized, stopped and shown as actionable — whose direction was decided by a candle
        that has not printed yet. Both break directions are run because a fault that leaks only
        one sign is still a lookahead.
        """
        for tf in ("15m", "1h", "4h", "1d"):
            s = quiet_series[(tf, up)]
            for name in NAMES:
                h = REGISTRY[name]
                if tf not in h.timeframes:
                    continue
                bad = acausal_probe(h, s)
                assert bad is None, (
                    f"{name} on {tf}: in a market where the only event is the LAST candle, the "
                    f"signal at bar {bad} is {int(SIGNALS.of(h, s)[bad])} with the whole series "
                    f"available and something else when the series stops at that bar. Nothing has "
                    f"happened yet at bar {bad}; whatever this is holding, it read it from the "
                    f"far end of the array."
                )

    @pytest.mark.parametrize("cut", [199, 1001, N_SYNTH - 1])
    def test_the_future_can_be_rewritten_from_anywhere(self, cut, synth_series):
        """A broad net under the probe above, which checks twelve bars each and no others.
        This compares EVERY bar before the cut at once, so a lookahead the probe happened to miss
        still has three chances: on the warm-up boundary, after the regime change, and at the last
        candle, where a one-bar lookahead has exactly one bar left to read."""
        offenders = []
        for name in NAMES:
            h = REGISTRY[name]
            for tf in h.timeframes:
                if rewrite_disagreement(h, synth_series[tf], cut).size:
                    offenders.append(f"{name}@{tf}")
        assert not offenders, f"acausal when the future is rewritten at bar {cut}: {offenders}"

    def test_the_broad_net_over_real_candles_too(self, real_series):
        """One truncation of the real series, comparing every bar before it rather than twelve.
        The probe above is sharper and this is wider; between them, a lookahead has to be both
        shallow and land away from all twelve probes to get through."""
        offenders = []
        for name in NAMES:
            h = REGISTRY[name]
            for tf in h.timeframes:
                if prefix_disagreement(h, real_series[tf], CUT_REAL).size:
                    offenders.append(f"{name}@{tf}")
        assert not offenders, f"acausal on real candles truncated at bar {CUT_REAL}: {offenders}"

    @pytest.mark.parametrize("cut", [MIN_PREFIX, 1300])
    def test_the_prefix_can_be_cut_anywhere_the_library_will_still_compute(self, cut,
                                                                          synth_series):
        """The same broad net for the truncating arm, which cannot go below MIN_PREFIX
        without measuring the library's own refusal to compute instead of its causality."""
        offenders = []
        for name in NAMES:
            h = REGISTRY[name]
            for tf in h.timeframes:
                if prefix_disagreement(h, synth_series[tf], cut).size:
                    offenders.append(f"{name}@{tf}")
        assert not offenders, f"acausal when truncated at bar {cut}: {offenders}"


class TestTheHarnessHasTeeth:
    """`acausal_probe` is the only thing standing between this suite and a lookahead in any of the
    hundred. If it cannot catch a deliberate one it is decoration, so here are three — and one
    causal function of the same shape, because a checker that is always red is decoration too.

    Every one of these is also run through the bulk arm (`acausal_indices`), so the two are never
    both quietly broken at once.
    """

    @staticmethod
    def _fabricate(fn) -> Hypothesis:
        # Named after the function, not "probe": `Signals` memoises on (name, series), so four
        # fabrications sharing one name would be handed each other's answers — which is how the
        # first draft of this class got a green causality test out of a deliberate lookahead.
        return Hypothesis(
            name=f"teeth.{fn.__name__}", family="teeth", title="probe",
            rationale="a deliberate lookahead", prior="caught", fn=fn,
            timeframes=("1h",), min_warmup=200,
        )

    def test_reading_one_bar_ahead_is_caught(self, synth_series):
        """The canonical mistake, and the cheapest to make: `close[i+1] > close[i]`."""
        def peek(s: Series) -> np.ndarray:
            out = np.zeros(len(s), dtype=np.int8)
            nxt = np.concatenate([s.close[1:], s.close[-1:]])
            out[nxt > s.close] = 1
            out[nxt < s.close] = -1
            return out

        h = self._fabricate(peek)
        assert acausal_probe(h, synth_series["1h"]) is not None, (
            "a signal built from the NEXT candle's close was not caught")

    def test_a_statistic_over_the_whole_array_is_caught(self, synth_series):
        """The subtler one: nothing is shifted, but the level compared against is a statistic of
        the ENTIRE series, so the answer on bar 200 is computed partly out of 2026.

        The mean, not the maximum, and the choice is the point. An extreme of the whole array is
        only detectable at bars where the running extreme differs from the final one, and this
        fixture's regime change is a crash, so the high of the series sits early and the fault is
        invisible after it — a version of this test built on `close.max()` passed while the leak
        was still there. The running mean differs from the final mean at essentially every bar.
        """
        def whole_array(s: Series) -> np.ndarray:
            out = np.zeros(len(s), dtype=np.int8)
            out[s.close > s.close.mean()] = 1
            out[s.close < s.close.mean()] = -1
            return out

        h = self._fabricate(whole_array)
        assert acausal_probe(h, synth_series["1h"]) is not None, (
            "a threshold taken over the whole series was not caught")
        assert prefix_disagreement(h, synth_series["1h"], CUT_SYNTH).size, (
            "the truncating arm missed a whole-array statistic, which is what it is best at")

    def test_a_threshold_read_off_the_last_candle_is_caught_by_both_arms(self, synth_series):
        """`s.close[-1]` — "compare against the latest price" — written once over the whole array
        instead of bar by bar. It moves every bar in the series at once, which is what the bulk
        arms are for, so this is where they are pinned."""
        def latest(s: Series) -> np.ndarray:
            out = np.zeros(len(s), dtype=np.int8)
            out[s.close > s.close[-1]] = 1
            out[s.close < s.close[-1]] = -1
            return out

        h = self._fabricate(latest)
        s = synth_series["1h"]
        assert acausal_probe(h, s) is not None, "a threshold read off the last candle survived"
        assert prefix_disagreement(h, s, CUT_SYNTH).size, "the truncating arm missed it"
        assert rewrite_disagreement(h, s, CUT_SYNTH).size, "the rewriting arm missed it"

    def test_a_negative_roll_is_caught(self, synth_series):
        """`np.roll(x, -1)` is banned by name in three docstrings in the library. This is the test
        that would notice if one arrived."""
        def rolled(s: Series) -> np.ndarray:
            out = np.zeros(len(s), dtype=np.int8)
            out[np.roll(s.close, -1) > s.open] = 1
            return out

        h = self._fabricate(rolled)
        assert acausal_probe(h, synth_series["1h"]) is not None, "np.roll(close, -1) was not caught"

    def test_a_lookahead_planted_in_a_REAL_hypothesis_is_caught_and_named(self, synth_series):
        """The four probes above are hand-built functions with a 200-candle warm-up, one timeframe
        and a signal on nearly every bar. A registry hypothesis is none of those things: it can
        declare a warm-up of 600, it runs on four timeframes with four different price paths, and
        several of them hold a position on a small minority of bars — which is the case where
        twelve probes have the least to work with.

        So the same fault is planted where it would actually be made. `streak3_fade` is handed a
        series whose closes are shifted one bar towards the present, which is `close[i+1] > close[i]`
        written the way it usually arrives: not as an index slip inside the formula, but as a
        helper that shifted the wrong way round. The requirement is not merely that something goes
        red — the bulk arms would do that — but that `acausal_probe`, the sharp instrument, names a
        bar on EVERY timeframe the hypothesis declares. A probe that only works on the timeframe
        the fixture happens to fire most on is not a guard on the other three.

        Only the probe is asserted, and that is the finding rather than an omission: measured,
        `acausal_indices` at CUT_SYNTH — both bulk arms, every bar before bar 1000 compared — comes
        back EMPTY on this exact fault. A one-bar lookahead moves one bar of the overlap and that
        bar was flat in both runs. The bulk arms are a net for faults that move the whole series;
        against the commonest mistake in the industry the twelve probes are the only thing
        standing there, which is what this test is for. Do not `assert not ...size` here to record
        the miss: that would pin the weakness and go red the day somebody fixes it.
        """
        victim = REGISTRY["mean_reversion.streak3_fade"]

        def reads_one_bar_ahead(s: Series) -> np.ndarray:
            return victim.fn(replace(s, close=np.concatenate([s.close[1:], s.close[-1:]])))

        # Renamed as well as rewired: `Signals` memoises on (name, series), so a fabrication
        # carrying the victim's own name would be handed the victim's clean answer.
        h = replace(victim, name="teeth.streak3_reading_one_bar_ahead", fn=reads_one_bar_ahead)
        missed = [tf for tf in victim.timeframes if acausal_probe(h, synth_series[tf]) is None]
        assert not missed, (
            f"a one-bar lookahead planted in {victim.name} was not caught on {missed}. Twelve "
            f"probes per timeframe is the budget; if a real hypothesis can carry a stolen candle "
            f"past them, N_PROBES is too low or probe_bars is sampling the wrong bars."
        )

    def test_a_causal_hypothesis_of_the_same_shape_is_not_flagged(self, synth_series):
        """The other half of the claim: the checker is not simply always red. Same function with
        the shift running the legal way round has to come back clean."""
        def past(s: Series) -> np.ndarray:
            out = np.zeros(len(s), dtype=np.int8)
            prev = np.concatenate([s.close[:1], s.close[:-1]])
            out[s.close > prev] = 1
            out[s.close < prev] = -1
            return out

        h = self._fabricate(past)
        assert acausal_probe(h, synth_series["1h"]) is None
        assert acausal_indices(h, synth_series["1h"], CUT_SYNTH).size == 0


# ------------------------------------------------------------------------------------------
class TestTheAlphabet:
    """A position series is int8 in {-1, 0, +1}. Anything else reaches the verdict silently."""

    def test_a_hypothesis_emits_only_minus_one_zero_or_plus_one(self, synth_series):
        offenders = sweep(out_of_alphabet, synth_series)
        assert not offenders, "\n".join(offenders)

    def test_and_on_real_candles_too(self, real_series):
        offenders = sweep(out_of_alphabet, real_series)
        assert not offenders, "\n".join(offenders)

    def test_signals_returns_one_int8_per_candle(self, synth_series):
        def check(h: Hypothesis, s: Series) -> list[str]:
            out = h.signals(s)
            bad = []
            if out.dtype != np.int8:
                bad.append(f"dtype is {out.dtype}, not int8")
            if out.size != len(s):
                bad.append(f"{out.size} signals for {len(s)} candles")
            return bad

        offenders = sweep(check, synth_series)
        assert not offenders, "\n".join(offenders)

    def test_signals_is_the_layer_that_refuses_a_malformed_answer(self, synth_series):
        """`signals()` carries two guards that no registered hypothesis reaches today, which is
        exactly why both can be deleted without a single test going red — and why hypothesis 101
        is the one that will find out.

        The cast is not decoration. `np.asarray(x, dtype=np.int8)` on a float array silently
        truncates, so a function that returns 0.5 — a half-size, a probability, an unrounded
        z-score — becomes flat everywhere and reports nothing; on a NaN the same cast produces an
        arbitrary int8, and -128 is a position 128 times the size of any this library can express.
        The length check is the difference between a `ValueError` naming the hypothesis and a
        signal array that is silently misaligned with its own candles for every bar after the
        first missing one.

        Fabricated rather than registered: the property belongs to `signals()`, and no registered
        hypothesis can demonstrate it precisely because all hundred are well behaved.
        """
        s = synth_series["1h"]

        def fabricate(name, fn):
            return Hypothesis(name=f"teeth.{name}", family="teeth", title="a malformed answer",
                              rationale="fabricated", prior="refused", fn=fn,
                              timeframes=("1h",), min_warmup=200)

        for name, n_out in (("too_short", len(s) - 1), ("too_long", len(s) + 1)):
            h = fabricate(name, lambda s, n=n_out: np.zeros(n, dtype=np.int8))
            with pytest.raises(ValueError, match="signals for"):
                h.signals(s)

        half = fabricate("halves", lambda s: np.full(len(s), 0.5, dtype=np.float64))
        out = half.signals(s)
        assert out.dtype == np.int8, (
            f"`signals()` handed back a {out.dtype} array. Everything downstream multiplies this "
            "by a return; a float position is a size, and nothing in the product is sized here.")
        assert not out.any(), "a half position has to land on 0, not on an invented direction"


# ------------------------------------------------------------------------------------------
class TestTheDeclaredWarmUp:
    def test_no_signal_before_the_candles_it_said_it_needed(self, synth_series, real_series):
        def check(h: Hypothesis, s: Series) -> list[str]:
            early = np.flatnonzero(h.signals(s)[: h.min_warmup])
            if early.size:
                return [(f"a position at bar {int(early[0])}, inside its declared warm-up of "
                         f"{h.min_warmup} candles")]
            return []

        offenders = sweep(check, synth_series) + sweep(check, real_series)
        assert not offenders, "\n".join(offenders)

    def test_the_mask_is_what_enforces_it_and_not_the_hypotheses(self):
        """Where the guarantee actually lives: `Hypothesis.signals` zeroes the head, and nothing
        else does. Measured: 93 of the hundred emit inside their own declared warm-up without that
        line, so it is the only thing holding the property above — and if it is ever deleted as
        redundant, this is the test that says what it was for."""
        emitting_early = {
            name for name in NAMES
            for tf in REGISTRY[name].timeframes
            if np.asarray(REGISTRY[name].fn(synthetic(N_SYNTH, tf)),
                          dtype=np.int8)[: REGISTRY[name].min_warmup].any()
        }
        assert len(emitting_early) > 50, (
            f"only {len(emitting_early)} of the hundred emit inside their own declared warm-up. "
            "Measured when this was written: 93. If it has collapsed, the guarantee has moved out "
            "of `signals()` into the hypotheses themselves and this file says the wrong thing "
            "about where it lives."
        )
        # And the mask is what turns those into silence, on the same series.
        for name in sorted(emitting_early)[:5]:
            h = REGISTRY[name]
            tf = h.timeframes[0]
            assert not h.signals(synthetic(N_SYNTH, tf))[: h.min_warmup].any()

    def test_the_warm_up_is_a_mask_on_the_head_and_not_a_bound_on_memory(self):
        """Recorded because it is the natural thing to assume and it is false.

        `min_warmup` says "do not trade the first N candles". It does NOT say "a signal at bar i
        depends only on the last N candles": `structure._hold` carries the last event forward until
        the opposite one arrives, with no limit at all.

        The witness is built rather than sampled, so it does not depend on a seed. Six hundred
        candles climb in a straight line — `donchian_break_20` breaks out early and, being a
        hold-until-reversed rule, is long from then on — and then the market stops dead: no new
        high, no new low, no event of any kind for five hundred candles. Given the whole series the
        hypothesis is long throughout that silence. Given only the silence, plus a re-warm of its
        full declared 200 candles, it is flat, and stays flat: the position it holds is information
        from a candle 900 bars back.

        (Three hypotheses disagree with themselves by one to seven bars after a re-warm for the
        much milder reason that an EMA's memory is infinite: `candles.engulfing_trend`,
        `mean_reversion.stretch_ema200_atr14` and the `trend` EMA stack. That is rounding. This is
        not.)

        A convention, not a defect — but it matters to anyone sizing a ring buffer from
        `min_warmup`, and if someone bounds `_hold`, this goes red and they get to write down what
        the new convention is.
        """
        j, after = 600, 500
        n = j + after
        step = BY_NAME["1h"].ms
        ts = (1_600_000_000_000 - 1_600_000_000_000 % step) + np.arange(n, dtype=np.int64) * step
        c = np.empty(n)
        c[:j] = 30_000.0 * np.exp(np.linspace(0.0, 0.5, j))   # a clean climb, breaking out early
        c[j:] = c[j - 1]                                      # and then nothing happens at all
        s = Series("1h", ts, c.copy(), c.copy(), c.copy(), c.copy(), np.full(n, 100.0))

        h = REGISTRY["structure.donchian_break_20"]
        w = h.min_warmup
        full, cut = h.signals(s)[j + w:], h.signals(tail(s, j))[w:]
        assert (full == 1).all(), "the fixture no longer leaves donchian_break_20 long; rebuild it"
        assert (cut == 0).all() and full.size == after - w, (
            f"donchian_break_20 now agrees with itself after a re-warm of its {w} declared candles "
            f"({int((full != cut).sum())} of {full.size} bars still differ). If _hold was given a "
            "bound, min_warmup has become a memory bound too and this file's account of it is out "
            "of date — say what the new one is."
        )


# ------------------------------------------------------------------------------------------
class TestScale:
    """BTC went from $4,000 to $120,000 over the stored history. A hypothesis whose behaviour
    depends on the absolute number is measuring the year, not the market."""

    #: 4.0 is exact in binary floating point, so every sum, difference and ratio downstream scales
    #: by exactly 4 and the comparison against a scale-free threshold is bit-identical. Any
    #: disagreement under this factor is a real dependence on the price level, never rounding.
    EXACT = 4.0
    #: 3.7 is not, so this arm additionally tolerates nothing but is allowed to be the one that
    #: goes red first if a threshold is sitting on a knife edge. Measured: all 100 pass both.
    INEXACT = 3.7

    def test_multiplying_every_price_by_a_constant_changes_nothing(self, synth_series,
                                                                   real_series):
        def check(h: Hypothesis, s: Series) -> list[str]:
            base = h.signals(s)
            bad = []
            for k in (self.EXACT, self.INEXACT):
                moved = np.flatnonzero(base != h.signals(scaled(s, k)))
                if moved.size:
                    bad.append(f"{moved.size} signals moved when every price was multiplied by "
                               f"{k} — first at bar {int(moved[0])}. It compares against an "
                               "absolute price, so it depends on what BTC happened to cost")
            return bad

        # Real candles as well as synthetic: BTC spans 20k to 120k inside the fixture's window, so
        # any absolute price a hypothesis might compare against falls INSIDE the series and the
        # multiplied copy answers differently. A synthetic walk that never crosses the constant
        # gives the same answer twice and the fault survives.
        offenders = sweep(check, synth_series) + sweep(check, real_series)
        assert not offenders, "\n".join(offenders)


# ------------------------------------------------------------------------------------------
class TestPurity:
    def test_a_hypothesis_does_not_write_to_the_candles_it_was_handed(self):
        """`evaluate_all` runs all hundred over the same arrays, in registry order. One in-place
        write and every hypothesis after it is graded on data the first one edited — and the result
        would then depend on alphabetical order, which nobody would think to check.

        Built fresh here rather than taken from the session fixture, and that is not tidiness. An
        in-place edit that is idempotent — a clip, a nan_to_num, a sort — changes the shared arrays
        the FIRST time anything runs and is a no-op every time after, so if a causality test has
        already touched the fixture this test compares the damage against itself and passes. It did
        exactly that, on a planted fault, until the fixture was made private to it.
        """
        own = {tf: synthetic(N_SYNTH, tf, shock_at=CUT_SYNTH) for tf in BY_NAME}

        def check(h: Hypothesis, s: Series) -> list[str]:
            fields = ("ts", "open", "high", "low", "close", "volume")
            before = {f: getattr(s, f).copy() for f in fields}
            first = h.signals(s)
            bad = [f"wrote to s.{f}; every hypothesis evaluated after it sees the edit"
                   for f in fields if not np.array_equal(getattr(s, f), before[f])]
            if not np.array_equal(first, h.signals(s)):
                bad.append("answered differently the second time it was asked the same question")
            return bad

        offenders = sweep(check, own)
        assert not offenders, "\n".join(offenders)


# ------------------------------------------------------------------------------------------
class TestDegenerateSeries:
    """The inputs a live feed produces at the worst moment: nothing yet, almost nothing, or a
    stretch where the price did not move at all."""

    def test_an_empty_series_gets_an_empty_answer(self):
        """This is the one that found something. `mean_reversion._prev` did `out[0] = np.nan` on a
        zero-length array, which is an IndexError, and it took down `sweep_rejection_20` and
        `streak3_fade` — the only two hypotheses in the file that reach it with no length guard in
        front. `evaluate_all` catches that in a bare `except` and prints a line, so the visible
        symptom was two hypotheses quietly absent from the report."""
        empty = {tf: flat(0, tf) for tf in BY_NAME}

        def check(h: Hypothesis, s: Series) -> list[str]:
            out = h.signals(s)
            return [] if out.size == 0 else [f"invented {out.size} signals out of no candles"]

        offenders = sweep(check, empty)
        assert not offenders, "\n".join(offenders)

    def test_fewer_candles_than_any_indicator_needs_is_answered_with_silence(self):
        for n in (1, 2, 5, 50):
            short = {tf: synthetic(n, tf) for tf in BY_NAME}

            def check(h: Hypothesis, s: Series, n: int = n) -> list[str]:
                out = h.signals(s)
                if out.size != n:
                    return [f"returned {out.size} signals for {n} candles"]
                return [] if not out.any() else [
                    f"took a position on {n} candles, with a declared warm-up of {h.min_warmup}"]

            offenders = sweep(check, short)
            assert not offenders, "\n".join(offenders)

    def test_a_market_that_never_moves_does_not_crash_or_warn(self):
        """Every range in the library is zero here, so every normalisation is 0/0. A RuntimeWarning
        is a suite-wide failure (pyproject) precisely because that is the road from a NaN to an
        int8 position of 128."""
        still = {tf: flat(800, tf) for tf in BY_NAME}
        offenders = sweep(out_of_alphabet, still)
        assert not offenders, "\n".join(offenders)

    def test_four_of_them_read_a_market_with_zero_movement_as_maximally_oversold(self):
        """Recorded, not fixed, because it is a product decision.

        TA-Lib returns 0 rather than NaN for RSI, MFI and Williams %R when there is neither a gain
        nor a loss to divide, and 0 is below every oversold threshold in the library. So a market
        that has not moved a tick reads as the most oversold market possible and these four go
        long. It cannot happen on BTC 1m, which is why it is a note and not a bug report; it can
        happen the day this catalogue is pointed at a stablecoin, a halted equity or a padded
        series, and then it fires continuously.

        The eight `seasonality` hypotheses also signal on a flat market, and for them it is
        correct: they read the clock, and the clock does not care what the price did.
        """
        offenders = {
            name for name in NAMES
            for tf in REGISTRY[name].timeframes
            if REGISTRY[name].family != "seasonality" and REGISTRY[name].signals(flat(800, tf)).any()
        }
        assert offenders == {
            "flow.mfi14_extremes",
            "mean_reversion.rsi14_cardwell_sma200",
            "mean_reversion.rsi14_fixed_3070",
            "mean_reversion.willr14_fade",
        }, f"the set that mistakes a still market for an extreme has changed: {sorted(offenders)}"

    def test_the_clock_hypotheses_do_signal_on_a_flat_market_and_that_is_correct(self):
        """The positive half of the same measurement: `seasonality` is a claim about WHEN, not
        about what the price did, so a flat series must not silence it."""
        silent = [
            name for name in NAMES
            if REGISTRY[name].family == "seasonality"
            and not any(REGISTRY[name].signals(flat(800, tf)).any()
                        for tf in REGISTRY[name].timeframes)
        ]
        assert silent == [
            "seasonality.cme_gap_monday",
            "seasonality.overlap_day_trend",
            "seasonality.pre_funding_fade",
        ], (
            f"the seasonality hypotheses silent on a flat market are {silent}. The three expected "
            "ones are exactly those that gate their calendar window on a price condition too — a "
            "gap, a day's direction, a stretch of the move — and a flat market denies all three. "
            "The other eight read the clock alone, and the clock does not care what the price did."
        )


# ------------------------------------------------------------------------------------------
class TestTheSharedHelpers:
    """Property 7: the utilities the hundred are built out of, where one fault is a family wrong.

    The six properties above are deliberately blind to arithmetic, and that blindness is paid for
    per hypothesis: a sign flipped inside `flow._clv` is causal, finite, scale-free, pure and
    two-sided, so it passes all six — while every hypothesis in the family reads selling pressure
    as buying pressure. What makes these helpers testable where the hundred are not is that each
    computes something DEFINED OUTSIDE this repository: a Williams fractal, a percentile, the
    weekday of an instant, "hold this for ten candles". The assertions below are written against
    those definitions, so they are oracles rather than restatements of the code they cover, and
    each one covers between two and eight hypotheses at once.
    """

    def test_a_pivot_is_reported_on_the_candle_that_confirms_it(self):
        """A fractal high at candle j is not KNOWN until j+k, once the k candles after it have
        closed. Reporting it at j is the single mistake that makes structure strategies look
        profitable on paper, and `_pivots` is the only place in the library that decides where it
        goes — eight `structure` hypotheses take their levels from it.

        The witness is a strict tent peaking at j, so there is exactly one fractal high in the
        whole series and its confirmation bar can be named: j+k and no other. Its mirror is that
        the centre of a tent is never the minimum of a window, so there is no fractal LOW at all —
        a fractal window that is not centred on its candidate (one candle left, two right) would
        invent one, and a candidate placed off-centre would report the high on the wrong bar.
        """
        n, j, k = 40, 20, structure._K
        ramp = 1_000.0 - (np.arange(n, dtype=float) - j) ** 2 * 0.1
        s = Series("1h", np.arange(n, dtype=np.int64) * BY_NAME["1h"].ms,
                   ramp, ramp + 1.0, ramp - 1.0, ramp, np.full(n, 100.0))

        ph, pl = structure._pivots(s, k)
        confirmed = np.flatnonzero(~np.isnan(ph))
        assert confirmed.tolist() == [j + k], (
            f"the only fractal high in a tent peaking at candle {j} is confirmed at candles "
            f"{confirmed.tolist()}; it is knowable at {j + k} — k={k} candles after the extreme — "
            "and nowhere else. Earlier is a lookahead, later is a level nobody can trade."
        )
        assert ph[j + k] == s.high[j], (
            f"the pivot confirmed at {j + k} carries the price {ph[j + k]}, not the high "
            f"{s.high[j]} of the candle it marks"
        )
        assert not (~np.isnan(pl)).any(), (
            "the centre of a tent is the maximum of every window it is centred in and the minimum "
            f"of none, yet a fractal LOW was confirmed at {np.flatnonzero(~np.isnan(pl)).tolist()}"
        )

    def test_every_pivot_is_the_extreme_of_the_window_centred_on_it(self, synth_series):
        """The definition, over a real price path rather than one shape: candle j is a fractal
        high iff its high is the maximum of [j-k, j+k], and it is reported at j+k. Stated as a
        loop over the definition, which is not how `_pivots` computes it (talib's rolling MAX,
        shifted), so agreement is evidence and not an echo.

        A window that is off-centre still produces plausible-looking pivots — they are still
        causal, still price-shaped, still sparse — and every level, break, retest and sweep in the
        `structure` family is then measured against a line drawn at the wrong candle.
        """
        for tf in ("15m", "1h"):
            s = synth_series[tf]
            k = structure._K
            want_h = np.full(len(s), np.nan)
            want_l = np.full(len(s), np.nan)
            for j in range(k, len(s) - k):
                window = slice(j - k, j + k + 1)
                if s.high[j] == s.high[window].max():
                    want_h[j + k] = s.high[j]
                if s.low[j] == s.low[window].min():
                    want_l[j + k] = s.low[j]

            got_h, got_l = structure._pivots(s, k)
            assert (~np.isnan(want_h)).sum() > 20, "the fixture holds almost no pivots to compare"
            for got, want, side in ((got_h, want_h, "high"), (got_l, want_l, "low")):
                bad = np.flatnonzero(~((np.isnan(got) & np.isnan(want)) | (got == want)))
                assert not bad.size, (
                    f"on {tf}, `_pivots` disagrees with the fractal definition at candles "
                    f"{bad[:5].tolist()} on the {side} side: it reports {got[bad[:5]]} where a "
                    f"{k}-each-side window centred on candle i-{k} gives {want[bad[:5]]}"
                )

    def test_the_close_location_value_is_plus_one_at_the_high_and_minus_one_at_the_low(self):
        """`_clv` is the `flow` family's proxy for WHO took the candle: +1 means the buyer closed
        it on its high, -1 means the seller closed it on its low. Its sign is the only directional
        content in `flow.volume_thrust` and `flow.absorption_narrow_range`; transposed, both take
        the opposite side of every trade they signal and nothing anywhere raises, because a
        backwards aggression reading is still a number in [-1, +1].

        The three candles below are the definition itself, and the fourth pins the documented
        answer for a candle with no range at all — 0, not a NaN and not an infinity, because a
        NaN here becomes an arbitrary int8 position.
        """
        prices = [(10.0, 12.0, 8.0, 12.0),    # closes on its high: the buyer took it
                  (10.0, 12.0, 8.0, 8.0),     # closes on its low: the seller took it
                  (10.0, 12.0, 8.0, 10.0),    # closes dead centre: nobody took it
                  (10.0, 10.0, 10.0, 10.0)]   # no range at all
        cols = np.array(prices, dtype=float)
        s = Series("1h", np.arange(4, dtype=np.int64) * BY_NAME["1h"].ms,
                   cols[:, 0], cols[:, 1], cols[:, 2], cols[:, 3], np.full(4, 100.0))
        assert flow._clv(s).tolist() == [1.0, -1.0, 0.0, 0.0], (
            f"_clv reads {flow._clv(s).tolist()} for candles that close at their high, at their "
            "low, at their midpoint and with no range; the contract is [+1, -1, 0, 0] and the "
            "first two are the ones that carry the direction"
        )

    def test_a_percentile_is_measured_against_the_window_that_ends_at_the_bar(self):
        """`_rolling_rank` answers "how extreme is today, against the last n days?" and seven
        `volatility` hypotheses gate on it. Two faults leave it perfectly causal and quietly wrong:
        ranking the OLDEST bar of the window instead of the current one — which describes a bar up
        to n-1 candles stale, so the regime filter lags by a window — and dividing by n instead of
        the n-1 other observations, which shaves every percentile just under its true value and
        drags each gated hypothesis across its own threshold.

        On a strictly increasing series the current bar is the largest thing in any window that
        ends on it, so its percentile is exactly 1, and 0 on the mirror. The third case is the
        interior: 2.5 beats three of the four observations in [0, 1, 2, 3, 2.5], which is 0.75.
        """
        rising = np.arange(500.0)
        assert np.unique(volatility._rolling_rank(rising, 100)[99:]).tolist() == [1.0], (
            "on a strictly increasing series every bar is the highest of the 100 that end on it, "
            "so its percentile is 1.0 — this is what a rank of the wrong bar of the window, or a "
            "divisor of n rather than the n-1 observations it is compared against, cannot produce"
        )
        assert np.unique(volatility._rolling_rank(rising[::-1].copy(), 100)[99:]).tolist() == [0.0]
        interior = volatility._rolling_rank(np.array([0.0, 1.0, 2.0, 3.0, 2.5]), 5)
        assert interior[4] == pytest.approx(0.75), (
            f"2.5 is above three of the other four observations in [0, 1, 2, 3], so its percentile "
            f"is 0.75; `_rolling_rank` says {interior[4]}"
        )
        assert np.isnan(interior[:4]).all(), "a percentile was reported before a full window exists"

    def test_the_prior_extreme_is_the_extreme_of_the_window_it_names(self):
        """`_prior_extreme` finds the swing the current bar is diverging FROM, and the two
        `momentum` divergence hypotheses compare today's high against it. Transposing max and min
        leaves it causal and in range while comparing today's high to the lowest low of the last
        hundred candles, so "divergence" becomes an unrelated inequality that is true most of the
        time — and a hypothesis that fires almost always is worse than one that never fires,
        because it looks like an edge.

        Asserted against the window the docstring names, over a random walk with no ties.
        """
        win, sep = momentum._DIV_WINDOW, momentum._DIV_SEPARATION
        x = np.random.default_rng(7).standard_normal(400).cumsum()
        for find_max, pick, word in ((True, np.max, "highest"), (False, np.min, "lowest")):
            j = momentum._prior_extreme(x, find_max)
            assert (j[:win] == -1).all(), "an extreme was claimed before a full window exists"
            for i in range(win, x.size):
                lo, hi = i - win, i - sep
                assert lo <= j[i] <= hi, (
                    f"bar {i} was handed the extreme at {j[i]}, outside the window "
                    f"[{lo}, {hi}] it is documented to search — {i - j[i]} candles back"
                )
                assert x[j[i]] == pick(x[lo:hi + 1]), (
                    f"bar {i} was handed {x[j[i]]} as the {word} value of x[{lo}:{hi + 1}], "
                    f"whose {word} is {pick(x[lo:hi + 1])}"
                )

    def test_an_isolated_event_is_held_for_exactly_the_candles_the_family_declared(self):
        """The event hypotheses bet that an effect is TRANSIENT, and the horizon is the bet.
        `structure`, `flow` and `candles` each carry their own copy of "hold this for n candles",
        and each is one character from holding for n+1 — which quietly lengthens the horizon of
        eleven hypotheses past the `params` they were pre-registered with. A pre-registered
        parameter that the code does not obey is not a hypothesis any more, it is a search.

        One pulse, far from either end, so the answer is a run length and not an edge effect.
        The three are asserted against the same expected indices as well as against each other:
        they implement one convention, and a convention that three modules disagree about is not
        one.
        """
        n, at, bars = 30, 5, structure._HOLD
        pulse = np.zeros(n, dtype=np.int8)
        pulse[at] = 1
        expected = list(range(at, at + bars))

        held = {
            "structure._hold_n": structure._hold_n(pulse == 1, pulse == -1, bars),
            "flow._hold": flow._hold(pulse, bars),
            "candles._hold": candles._hold(pulse, bars),
        }
        for who, out in held.items():
            assert np.flatnonzero(out).tolist() == expected, (
                f"{who} holds a single event fired at candle {at} over "
                f"{np.flatnonzero(out).tolist()} — {int((out != 0).sum())} candles. The declared "
                f"horizon is {bars}, counting the signal candle itself, so it is candles "
                f"{at}..{at + bars - 1}. One candle either way is a different hypothesis."
            )
            assert (out[expected] == 1).all(), f"{who} held the event with the wrong sign"

    def test_the_clock_helpers_agree_with_the_calendar(self):
        """The eleven `seasonality` hypotheses are claims about WHEN and nothing else, so the
        calendar helpers are their entire content. An epoch offset one day out moves every one of
        them to the wrong day of the week; an expiry that is not a Friday is not an expiry; a
        day-of-month that starts at 0 shifts the turn-of-month window at both ends. None of that
        raises, none of it is visible in a signal series, and all of it is checkable against a
        calendar this project did not write.

        Six years of daily instants, a leap year included, plus a stretch of hours.
        """
        step = BY_NAME["1d"].ms
        days = (1_500_000_000_000 - 1_500_000_000_000 % step) + np.arange(2200, dtype=np.int64) * step
        hours = 1_500_000_000_000 + np.arange(500, dtype=np.int64) * BY_NAME["1h"].ms
        ts = np.concatenate([days.astype(np.int64), hours])
        truth = [datetime.fromtimestamp(int(t) / 1000, tz=UTC) for t in ts]

        assert seasonality._weekday(ts).tolist() == [d.weekday() for d in truth], (
            "the weekday of an instant does not agree with the calendar; 1970-01-01 was a "
            "Thursday and every seasonality window is placed relative to that"
        )
        assert seasonality._hour_utc(ts).tolist() == [d.hour for d in truth]

        dom, length = seasonality._day_of_month(seasonality._date(ts))
        assert dom.tolist() == [d.day for d in truth], (
            "the day of the month does not agree with the calendar — the turn-of-month window is "
            "counted from both ends of this number"
        )
        assert length.tolist() == [calendar.monthrange(d.year, d.month)[1] for d in truth]

        expiry = seasonality._last_friday(seasonality._date(ts).astype("datetime64[M]"))
        as_dates = [datetime.fromisoformat(str(v)) for v in expiry.astype("datetime64[D]")]
        assert {d.weekday() for d in as_dates} == {4}, (
            "the monthly expiry fell on "
            f"{sorted({calendar.day_name[d.weekday()] for d in as_dates})}; it is the LAST FRIDAY "
            "of the month, and a Deribit or CME expiry on any other day is not an expiry"
        )
        assert all((d + timedelta(days=7)).month != d.month for d in as_dates), (
            "the expiry date is a Friday, but another Friday follows it in the same month, so it "
            "is not the LAST one"
        )


# ------------------------------------------------------------------------------------------
class TestAmbiguityIsNotATieBreak:
    """The two places in the library where a candle can satisfy the long rule and the short rule
    at once, and the convention both of them declare: on such a candle there is NO information, so
    nothing new is entered and whatever was held is held.

    This is not a style rule, it is the difference between a measurement and a bias. `_events`
    states it in a docstring; `_chandelier_trail` states it in a twenty-line comment written after
    an audit found the opposite in the code — the two assignments trampled each other, the short
    was written second and therefore always won, and the ambiguous zone appears precisely in wide
    ranges, which is the regime that hypothesis's own `prior` predicts losses in. A tie-break that
    systematically goes short exactly where the answer is being measured does not add noise, it
    adds sign, and the result stops being interpretable.

    Reintroducing that fault today leaves all six properties above green. So it is pinned here.
    """

    def test_an_event_that_is_both_directions_at_once_is_no_event(self):
        """`structure._events` is the encoder every event hypothesis in the family goes through.
        On a candle that breaks up and breaks down there is no direction to be had, and picking
        one — whichever of the two assignments happens to be written second — silently gives the
        whole family a house bias on exactly the candles that are hardest to read."""
        up = np.array([True, False, True, False])
        dn = np.array([True, True, False, False])
        assert structure._events(up, dn).tolist() == [0, -1, 1, 0], (
            f"_events reads {structure._events(up, dn).tolist()} for candles that fire both ways, "
            "down only, up only, and neither. The first must be 0: an arbitrary tie-break there "
            "is a direction nobody chose, taken on the most ambiguous candles in the sample."
        )

    @pytest.mark.parametrize("name", ["volatility.chandelier_atr_trail",
                                      "volatility.chandelier_fixed_pct"])
    def test_the_chandelier_holds_its_position_through_the_zone_where_both_stops_apply(
            self, name, synth_series):
        """The audited fault, made red. When the 22-candle range is wider than twice the buffer the
        long stop sits ABOVE the short stop, and a close between them is above one and below the
        other. The Chandelier rule — and the hypothesis's own rationale — is that the trail is only
        ever left by crossing it, so such a candle carries the previous position forward.

        Both arms are run, and that is the point rather than thoroughness: `chandelier_fixed_pct`
        is the CONTROL that makes this family's claim testable, and a control that tie-breaks
        differently from the hypothesis it controls is measuring the tie-break as well as the
        scaling. The two must share the rule or the comparison between them means nothing.

        The stops are rebuilt from the PRE-REGISTERED `params` — the record each hypothesis was
        declared with — rather than from the function under test. The zone being non-empty and
        being entered at least once while LONG are asserted too: a rule that always went short on
        an ambiguous candle agrees with a carried short, so without those two lines this would
        pass on a fixture that happened never to enter the zone long.
        """
        h = REGISTRY[name]
        window = h.params["window"]

        for tf in h.timeframes:
            s = synth_series[tf]
            hi = np.ascontiguousarray(s.high, dtype=np.float64)
            lo = np.ascontiguousarray(s.low, dtype=np.float64)
            close = np.ascontiguousarray(s.close, dtype=np.float64)
            if "mult" in h.params:
                buffer_ = h.params["mult"] * talib.ATR(hi, lo, close, h.params["atr_period"])
                long_stop = talib.MAX(hi, window) - buffer_
                short_stop = talib.MIN(lo, window) + buffer_
                width = f"{h.params['mult']}·ATR-{h.params['atr_period']}"
            else:
                pct = h.params["buffer_pct"]
                long_stop = talib.MAX(hi, window) * (1.0 - pct)
                short_stop = talib.MIN(lo, window) * (1.0 + pct)
                width = f"{pct:.0%} of the level"
            both = (close > short_stop) & (close < long_stop)     # NaN compares False either way

            sig = h.signals(s)
            zone = np.flatnonzero(both)
            zone = zone[zone > h.min_warmup]
            assert zone.size, (
                f"{name} on {tf}: the fixture never puts the close between the two stops, so this "
                f"test asserts nothing. It needs a {window}-candle range wider than twice "
                f"{width} to exist at all.")
            assert (sig[zone - 1] == 1).any(), (
                f"{name} on {tf}: every ambiguous candle is entered from a short, so a rule that "
                "always went short on them would pass this unchanged. The fixture has to enter "
                "the zone long at least once for the assertion below to mean anything.")
            moved = zone[sig[zone] != sig[zone - 1]]
            assert not moved.size, (
                f"{name} on {tf}: the position changed on candles {moved[:5].tolist()}, where the "
                f"close is both above the {window}-candle low plus {width} and below the "
                f"{window}-candle high minus it. Neither stop was crossed, so there is nothing "
                f"there to act on: {int(moved.size)} candles were handed a direction by whichever "
                "of the two assignments happens to run second."
            )


# ------------------------------------------------------------------------------------------
class TestARetestIsAReturnToTheLevel:
    def test_the_level_has_to_be_touched_again_before_the_retest_fires(self):
        """`structure.retest_hold` is named after the thing it waits for: price breaks a level,
        comes BACK to it, and holds. The coming back is read off the candle's low, which is where
        a pullback lives; read off its high instead, the condition becomes "the whole candle fell
        back below the level", which is a different and much rarer configuration — so the
        hypothesis keeps firing, keeps clearing every property in this file, and is no longer
        measuring a retest. A hypothesis mis-named in the catalogue is worse than one that does
        not fire, because the reader validates it believing it is something else.

        Two hand-built markets, identical except for one candle's low: in the first, price wicks
        back to the broken level and closes above it; in the second it never comes back at all.
        The second is the control — without it, a rule that fired on every breakout would pass.
        """
        def channel(*, comes_back: bool) -> Series:
            n, brk = 320, 240
            step = BY_NAME["1h"].ms
            t0 = 1_600_000_000_000 - 1_600_000_000_000 % step
            ts = t0 + np.arange(n, dtype=np.int64) * step
            o, h, l, c = (np.full(n, 105.0), np.full(n, 110.0),
                          np.full(n, 100.0), np.full(n, 105.0))
            o[brk], h[brk], l[brk], c[brk] = 110.0, 116.0, 104.0, 115.0   # the break, closing above
            for i in range(brk + 1, n):                                   # and then it drifts
                o[i], h[i], l[i], c[i] = 113.0, 114.0, 112.0, 113.0
            if comes_back:
                l[brk + 2] = 109.0                      # a wick down to the broken level, and only
            return Series("1h", ts.astype(np.int64), o, h, l, c, np.full(n, 100.0))  # the wick

        h = REGISTRY["structure.retest_hold"]
        fired = np.flatnonzero(h.signals(channel(comes_back=True)))
        assert fired.tolist() == list(range(242, 242 + structure._HOLD)), (
            f"price broke the 20-candle high at candle 240 and wicked back to it at 242, closing "
            f"above it again. The retest is at 242 and is held for {structure._HOLD} candles; the "
            f"hypothesis fired on {fired.tolist()}. Nothing fires if the return is read off the "
            "candle's high, because the high never came back below the level."
        )
        never = np.flatnonzero(h.signals(channel(comes_back=False)))
        assert not never.size, (
            f"price broke the level at candle 240 and never traded back to it, yet the retest "
            f"fired on {never.tolist()}. Then it is not waiting for a retest, it is a breakout."
        )


# ------------------------------------------------------------------------------------------
class TestEveryDeclaredTimeframeRuns:
    def test_it_runs_on_every_timeframe_it_declared(self, real_series):
        """`timeframes` is what the picker offers the user, and `/api/validate` runs whatever they
        pick. A hypothesis registered on 4h that throws on 4h is a 500 on the served path.

        Counted rather than merely swept, so that a `timeframes` tuple silently emptied — which
        would make every sweep in this file pass by iterating over nothing — is caught here.
        """
        ran = 0
        for name in NAMES:
            h = REGISTRY[name]
            for tf in h.timeframes:
                assert h.signals(real_series[tf]).size == len(real_series[tf]), f"{name} on {tf}"
                ran += 1
        assert ran == 293, f"{ran} (hypothesis, timeframe) pairs ran, not the 293 registered"

    def test_the_label_on_the_series_is_load_bearing(self, real_series):
        """The same candles, told they are 1h and told they are 15m, must not produce the same
        seasonality signal: `_hold_ts` adds the bar's own duration to decide which hour the
        position will be held through. If the label were ignored the whole family would be
        silently evaluating the wrong window."""
        h = REGISTRY["seasonality.us_cash_hours_long"]
        candles = real_series["1h"]
        as_1h = h.signals(candles)
        as_15m = h.signals(replace(candles, tf="15m"))
        assert not np.array_equal(as_1h, as_15m), (
            "seasonality.us_cash_hours_long ignores Series.tf, so its window is being read off "
            "timestamps alone and every timeframe gets the same answer"
        )


#: The hypotheses that take one side and say so in their own name: a hammer is a bullish reversal,
#: a shooting star a bearish one, "weekend short" is a short. For these, holding one direction over
#: nine years of BTC is the claim itself, not a symptom of a lost direction.
ONE_SIDED_BY_CONSTRUCTION = {
    "candles.hammer_oversold",
    "candles.range_gap",
    "candles.shooting_star_overbought",
    "candles.three_directional_bars",
    "flow.nvi_fosback",
    "seasonality.asia_hours_short",
    "seasonality.friday_derisk_short",
    "seasonality.post_monthly_expiry_long",
    "seasonality.pre_quarterly_expiry_short",
    "seasonality.turn_of_month_long",
    "seasonality.us_cash_hours_long",
    "seasonality.weekend_short",
}


# ------------------------------------------------------------------------------------------
class TestTheyActuallyFire:
    def test_every_hypothesis_takes_a_position_at_some_point_on_real_btc(self, real_series):
        """A hypothesis that never fires is not a failed hypothesis, it is dead code that still
        consumes a slot in the multiple-comparisons correction — it makes every other result look
        worse for nothing. This is also the only test here that would notice a hypothesis whose
        condition can never be true.
        """
        silent = [
            name for name in NAMES
            if not any(REGISTRY[name].signals(real_series[tf]).any()
                       for tf in REGISTRY[name].timeframes)
        ]
        # A SUBSET, not an equality. The property is "no hypothesis is dead code"; whether a
        # genuinely rare one happened to fire inside this particular window is a fact about the
        # window. Pinning the exact set made the test fail the day `candles.range_gap` finally
        # fired — a strictly better outcome reported as a regression.
        unexpected = sorted(set(silent) - RARE_ENOUGH_TO_BE_SILENT)
        assert not unexpected, (
            f"hypotheses that never fire on {N_REAL} real candles of any timeframe they declare: "
            f"{unexpected}. A hypothesis that cannot fire is dead code that still consumes a slot "
            "in the multiple-comparisons correction, making every other result look worse for "
            "nothing. Either its condition is unsatisfiable or its warm-up never clears."
        )

    def test_a_hypothesis_that_is_not_one_sided_by_name_takes_both_sides(self, real_series):
        """`.any()` is satisfied by a rule that can only ever go one way, so the test above cannot
        tell a working two-sided hypothesis from one that has quietly lost a direction.

        That is not hypothetical arithmetic. `trend._supertrend` releases its band on the PREVIOUS
        close; move that read to the current close and the release condition becomes unsatisfiable
        — the hypothesis is short on 100% of the candles of every timeframe it declares, over nine
        years of BTC, and every property in this file stays green. A trend follower that cannot go
        long in a market that spent most of the sample rising does not fail loudly; it reports a
        plausible negative edge, which is exactly the kind of result a catalogue like this one is
        supposed to be able to believe.

        Asserted as a subset so that a rare hypothesis finally showing its second side is an
        improvement rather than a red gate — the same lesson the exact-set assertion above was
        rewritten for. The window is the fixed one, so the measurement does not move with the
        market.
        """
        one_sided = {
            name for name in NAMES
            if not (any((SIGNALS.of(REGISTRY[name], real_series[tf]) == 1).any()
                        for tf in REGISTRY[name].timeframes)
                    and any((SIGNALS.of(REGISTRY[name], real_series[tf]) == -1).any()
                            for tf in REGISTRY[name].timeframes))
        }
        lost_a_side = sorted(one_sided - ONE_SIDED_BY_CONSTRUCTION)
        assert not lost_a_side, (
            f"these hypotheses take only one side over {N_REAL} real candles of every timeframe "
            f"they declare, and their names do not say they should: {lost_a_side}. Either the "
            "condition for the other direction has become unsatisfiable, or the hypothesis is "
            "directional by construction and belongs in ONE_SIDED_BY_CONSTRUCTION with a line "
            "saying why."
        )
