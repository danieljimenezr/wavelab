"""The validation battery: five tests that almost no published strategy gets through.

THIS IS THE PRODUCT. It does not tell you when to buy: it tells you whether the thing you believe
works is real.

The five, in order of brutality:

1. **Delay test.** Execute the signal one bar later. A real effect gives up a little; an
   information leak COLLAPSES. It is the cheapest test in existence and it caught a bug of our own
   that was inflating CAGR from 26% to 99%.

2. **Base rate.** Is it right more often than being in the market at any old moment? In an asset
   that rose 1,748% over the 2017-2026 sample this was written against, any mostly-long
   rule is right a lot, and that is not skill. The figure the user is shown is computed
   from the series in hand, not from this comment.

3. **Random control at the same exposure.** Random filters that enter and exit at the same
   frequency and spend the same time in the market. If the strategy does not beat their median,
   what it has is beta, not judgement.

4. **Out of sample, with purging.** Walk-forward, purging the horizon at every boundary. How much
   of the edge survives outside the sample it was found in.

5. **Effective n.** Signals that overlap inside the horizon are ONE observation. An "n=500" that
   is really 12 episodes produces p-values that are wrong by orders of magnitude.

And on top of all that, if several strategies are evaluated, White's Reality Check corrects for
having tried them all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wavelab.validation.reality_check import effective_n

__all__ = ["BatteryResult", "Test", "run_battery"]


@dataclass(frozen=True, slots=True)
class Test:
    id: str
    title: str
    passed: bool | None          # None = inconclusive, not enough data
    value: float
    reference: float
    unit: str
    explanation: str
    detail: str = ""

    @property
    def status(self) -> str:
        return "inconclusive" if self.passed is None else ("pass" if self.passed else "fail")


@dataclass(slots=True)
class BatteryResult:
    name: str
    n_signals: int
    n_effective: int
    exposure: float
    cagr: float
    sharpe: float
    max_dd: float
    cagr_bh: float
    sharpe_bh: float
    max_dd_bh: float
    tests: list[Test] = field(default_factory=list)
    equity: np.ndarray = field(repr=False, default=None)
    equity_bh: np.ndarray = field(repr=False, default=None)

    @property
    def verdict(self) -> str:
        failures = [t for t in self.tests if t.passed is False]
        if any(t.id == "delay" for t in failures):
            return "lookahead"     # the worst one of all: it is reading the future
        if len(failures) >= 2:
            return "fails"
        if failures:
            return "doubtful"
        return "survives"

    @property
    def summary(self) -> str:
        return {
            "lookahead": "IT READS THE FUTURE. The result is not reachable in real time.",
            "fails": "DOES NOT SURVIVE. It fails several independent tests.",
            "doubtful": "DOUBTFUL. Passes most of them but fails one that matters.",
            "survives": "SURVIVES the battery. That is not proof it makes money: it is proof "
                        "it is not one of the known mistakes.",
        }[self.verdict]


def _curve(ret: np.ndarray, weight: np.ndarray | None = None, cost_bps: float = 15) -> np.ndarray:
    r = ret if weight is None else ret * weight
    if weight is not None and cost_bps:
        turnover = np.abs(np.diff(np.concatenate([[0.0], weight])))
        r = r - turnover * cost_bps / 10_000
    return np.exp(np.cumsum(r))


def _stats(eq: np.ndarray, per_year: float) -> tuple[float, float, float]:
    ret = np.diff(np.log(np.maximum(eq, 1e-12)))
    years = len(eq) / per_year
    cagr = eq[-1] ** (1 / years) - 1 if years > 0 and eq[-1] > 0 else -1.0
    sh = ret.mean() / ret.std() * np.sqrt(per_year) if ret.std() > 0 else 0.0
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return float(cagr), float(sh), dd


def run_battery(
    close: np.ndarray,
    ts_ms: np.ndarray,
    signal: np.ndarray,
    *,
    name: str = "strategy",
    bar_ms: int,
    horizon_bars: int = 1,
    cost_bps: float = 15,
    n_random: int = 300,
    seed: int = 11,
    folds: int = 5,
) -> BatteryResult:
    """Puts a signal through the five tests.

    ``signal[i]`` is the position held FROM the close of bar i until the close of bar i+1. It is
    the only convention possible without looking at the future, and it is where nearly everybody
    gets it wrong.
    """
    c = np.asarray(close, dtype=float)
    sig = np.asarray(signal, dtype=float)[: len(c) - 1]
    r = np.diff(np.log(c))                       # r[i]: from close i to close i+1
    n = len(r)
    rng = np.random.default_rng(seed)
    per_year = 365.25 * 24 * 3600 * 1000 / bar_ms

    eq = _curve(r, sig, cost_bps)
    eq_bh = _curve(r)
    cagr, sh, dd = _stats(eq, per_year)
    cagr_bh, sh_bh, dd_bh = _stats(eq_bh, per_year)
    active = sig != 0
    exposure = float(np.abs(sig).mean())

    tests: list[Test] = []

    # ---- 1. one-bar delay ---------------------------------------------------------------------
    sig_lag = np.concatenate([[0.0], sig[:-1]])
    cagr_lag, _, _ = _stats(_curve(r, sig_lag, cost_bps), per_year)
    # A real effect gives up something when it is executed late; a leak collapses. The threshold
    # is that it keeps at least half of its excess over buy and hold.
    exc, exc_lag = cagr - cagr_bh, cagr_lag - cagr_bh
    keeps = (exc_lag / exc) if abs(exc) > 1e-9 else 1.0
    tests.append(Test(
        "delay", "Does it survive being executed one bar later?",
        bool(keeps >= 0.5) if abs(exc) > 1e-9 else True,
        float(cagr_lag), float(cagr), "CAGR",
        "If the result collapses when the signal is delayed by a single bar, the strategy is "
        "using information you would not have had in real time. It is the cheapest test there is "
        "and the one that knocks over the most strategies.",
        f"CAGR {cagr*100:.1f}% → {cagr_lag*100:.1f}% when delayed (keeps {keeps*100:.0f}% "
        f"of the excess over buy and hold)"))

    # ---- 2. base rate -------------------------------------------------------------------------
    fwd = np.full(n, np.nan)
    if n > horizon_bars:
        fwd[:-horizon_bars] = np.log(c[horizon_bars + 1: n + 1] / c[1: n - horizon_bars + 1])
    val = ~np.isnan(fwd)
    base = float(fwd[val].mean()) if val.any() else 0.0
    m = val & active
    prop = float((sig[m] * fwd[m]).mean()) if m.sum() >= 20 else float("nan")
    has_prop = not np.isnan(prop)      # clearer than the idiomatic `prop == prop`
    tests.append(Test(
        "base_rate", "Is it right more often than any old moment?",
        bool(prop > base) if has_prop else None,
        prop * 100 if has_prop else 0.0, base * 100, "% per period",
        f"In an asset that rose {(c[-1] / c[0] - 1) * 100:,.0f}%, any mostly-long rule looks "
        "right. What counts is whether it is right MORE often than being in the market at a "
        "random instant.",
        f"strategy {prop*100:+.3f}% vs base {base*100:+.3f}% per period"
        if has_prop else "far too few signals"))

    # ---- 3. random control at the same exposure -----------------------------------------------
    real_flips = int(np.abs(np.diff(np.concatenate([[0.0], sig]))).sum())
    dur = max(2, int(n / max(real_flips, 1)))
    randoms = np.empty(n_random)
    for k in range(n_random):
        w = np.zeros(n); i = 0
        while i < n:
            L = int(rng.integers(max(2, dur // 2), max(3, dur * 2)))
            w[i:i + L] = float(rng.random() < exposure)
            i += L
        randoms[k] = _stats(_curve(r, w, cost_bps), per_year)[0]
    pct = float((randoms < cagr).mean())
    tests.append(Test(
        "random_control", "Does it beat RANDOM filters with the same exposure?",
        bool(pct >= 0.95), pct * 100, 95.0, "percentile",
        "Hundreds of filters are generated that enter and exit at random with the same frequency "
        "and the same time spent inside the market. If your strategy is not clearly above them, "
        "what you have is exposure, not judgement.",
        f"percentile {pct*100:.0f} against {n_random} random filters "
        f"(median {np.median(randoms)*100:.1f}% CAGR)"))

    # ---- 4. out of sample, with purging --------------------------------------------------------
    step = n // folds
    oos = np.zeros(n, dtype=bool)
    for k in range(1, folds):
        oos[k * step + horizon_bars: min((k + 1) * step, n)] = True
    if oos.sum() > 100:
        cagr_oos, _, _ = _stats(_curve(r[oos], sig[oos], cost_bps), per_year)
        cagr_bh_oos, _, _ = _stats(_curve(r[oos]), per_year)
        exc_oos = cagr_oos - cagr_bh_oos
        keeps_oos = (exc_oos / exc) if abs(exc) > 1e-9 else 1.0
        # If the IN-SAMPLE excess is already negative, "keeps 122%" means it loses consistently,
        # not that it holds up. The test only means anything about a positive edge: with a
        # negative excess there is nothing to survive, so it is reported as inconclusive.
        if exc <= 0:
            oos_passed = None
        else:
            oos_passed = bool(keeps_oos >= 0.5)
        tests.append(Test(
            "out_of_sample", "Does it hold up outside the sample it was found in?",
            oos_passed, float(cagr_oos), float(cagr_bh_oos), "CAGR",
            "The history is cut into segments and the horizon is purged at every boundary, so "
            "that no open trade crosses from one segment into the next. The edge has to survive "
            "where nobody was looking.",
            (f"in-sample excess {exc*100:+.1f}% → out of sample {exc_oos*100:+.1f}% "
             f"(keeps {keeps_oos*100:.0f}%)") if exc > 0 else
            (f"not applicable: the excess is already negative in sample ({exc*100:+.1f}%). "
             "There is no edge that could survive outside it.")))
    else:
        tests.append(Test("out_of_sample", "Does it hold up out of sample?", None, 0, 0, "CAGR",
                          "Series too short to cut up.", ""))

    # ---- 5. effective n -----------------------------------------------------------------------
    # The horizon that matters is NOT the `horizon_bars` parameter but how long each position
    # actually lasts: if the strategy stays in for 40 bars in a row, those 40 bars are ONE bet,
    # not forty. So we measure the mean run length of the signal itself.
    sig_flips = np.abs(np.diff(np.concatenate([[0.0], np.sign(sig)]))) > 0
    n_runs = max(1, int(sig_flips.sum()))
    mean_dur = max(horizon_bars, round(active.sum() / n_runs) if n_runs else horizon_bars)
    n_ef = effective_n(np.asarray(ts_ms)[: n][active], mean_dur, bar_ms)
    tests.append(Test(
        "effective_n", "Are there enough INDEPENDENT observations?",
        bool(n_ef >= 30), float(n_ef), 30.0, "episodes",
        "Signals that overlap inside the holding horizon are the SAME observation. An n=500 that "
        "is really 12 episodes produces p-values that are wrong by orders of magnitude. It is "
        "López de Prado's label overlap problem.",
        f"{int(active.sum())} bars holding a position across {n_runs} runs "
        f"(mean duration {mean_dur} bars) → {n_ef} independent episodes"))

    return BatteryResult(
        name=name, n_signals=int(active.sum()), n_effective=n_ef, exposure=exposure,
        cagr=cagr, sharpe=sh, max_dd=dd, cagr_bh=cagr_bh, sharpe_bh=sh_bh, max_dd_bh=dd_bh,
        tests=tests, equity=eq, equity_bh=eq_bh)
