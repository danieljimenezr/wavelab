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

Everything runs over synthetic series with known structure AND over real BTC candles out of the
store, resampled through the same `resample_from_1m` the server uses. The real arm is not
decoration: a planted one-bar lookahead was caught there before the synthetic probe existed.

WHAT IS NOT HERE. No property in this file looks at what any hypothesis actually computes. A
Bollinger band built at one sigma instead of two, an RSI threshold of 40 instead of 30, a sign
flipped end to end — all of those are causal, finite, scale-free and pure, and all of them pass.
Two such faults were planted to confirm it and both survived. That boundary is deliberate: the
alternative is a hundred restatements of a hundred formulas.

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

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from wavelab.core.timeframes import BY_NAME
from wavelab.hypotheses import load_all
from wavelab.hypotheses.base import Hypothesis, Series, register

REGISTRY = load_all()
NAMES = sorted(REGISTRY)

#: Long enough that the deepest warm-up in the registry (600) leaves 900 bars to compare over.
N_SYNTH = 1500
#: Where the synthetic series is cut in two. The regime change lives immediately after it.
CUT_SYNTH = 1000

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
