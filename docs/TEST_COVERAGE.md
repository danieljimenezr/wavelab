# What the test suite protects, and what it does not

**962 tests, ~6.0 s, `pytest -m "not net"`.** One test is deselected by that flag (`net`, it hits
the real Binance archive). The suite gates deploys and it is no longer under five seconds.

Two traps in that sentence, both measured. The flag is not a default: `pyproject.toml` sets
`addopts = "-q --strict-markers"` and nothing else, so a bare `uv run pytest` — which is what
`make test` runs — SELECTS the `net` test, goes to the internet, and takes ten seconds instead of
five. Only CI (`ci-cd.yml`) passes `-m "not net"`. And the second: every figure in this file is
`-m "not net"`, so a bare run reports one test more than any count written down here.

`test_hypothesis_library.py` is 1.5 s of it — measured as a delta, running the suite with and
without that file back to back on the same machine, because absolute seconds here are a fact about
the laptop. It runs all 100 hypotheses over four synthetic timeframes and four real ones, with
~6,200 truncated recomputations for the causality probe among them — 3,120 per arm, counted, not
estimated: the probe asks for up to twelve bars per (hypothesis, timeframe) pair and `probe_bars`
dedupes, so 293 pairs yield 3,120 and not 3,516. It is the only file allowed to cost that much, and
the reason is that it covers the largest surface in the repo. If it grows
again, cut the number of cut points before cutting the probes: the probes are what caught the
planted one-bar lookahead, and the cut points are the broad net under them.

This document exists because a green suite is a claim, and this project is sold on refusing to
flatter its user. The same standard applies to its own tooling: a test that cannot fail reads as
coverage and is worse than no test, and a coverage table that lists a hole as closed is worse than
one that admits the hole. Everything below is measured, not assumed — every "killed" claim was
established by applying the fault to the source, watching a named test go red, reverting, and
verifying the file is byte-identical by sha256.

**The short version of where this suite is weak:**

1. **The hypothesis library is now run, as properties rather than as a hundred hand-written
   tests** (`tests/test_hypothesis_library.py`). What is asserted about all 100 is causality, the
   {-1,0,+1} alphabet, the declared warm-up, scale-freedom, purity and behaviour on degenerate
   input — over synthetic candles and over real ones from the store. What is NOT asserted is that
   any hypothesis computes the indicator it says it computes: a `_bb_fade` that quietly used a
   1-sigma band would pass every property here.

   Six properties over a hundred hypotheses is not six hundred behaviours covered. Read the list
   as what it is: a floor every hypothesis must clear, plus five recorded facts about the registry
   that would move if somebody changed one. The BEHAVIOUR of any individual hypothesis — the level
   it compares against, the period it uses, the direction it takes — is protected by nothing.
2. **The served HTTP path is only tested where the product is served.** `/api/decide` and the
   PUBLIC gate are now measured (`tests/test_decide_route.py`, 30 tests), and `/api/hypotheses` is
   measured for the catalogue prose alone — that it carries a `title` for all 100 under the key
   `web/assay.js` actually reads. `/api/validate`, `/api/validate_csv` and `/api/validate_rule`
   are still not exercised end to end; what they call is heavily tested, what they put on the wire
   is not.
3. **Five modules have no tests at all**: `core/bus.py`, `feeds/binance_derivs.py`,
   `store/trials.py`, `collect.py`, `registry.py`.
4. **One known internal inconsistency is pinned, not fixed** — the R:R quoted on a signal is
   computed at a different entry price from the one the trade is booked at.

---

## How the numbers below were produced

Mutation testing, not line coverage. Line coverage says a line ran; it says nothing about whether
anything would notice if the line were wrong. Each fault below is a real edit to the source — an
`abs()` removed, a `>=` loosened, a threshold nudged, an index shifted by one — applied one at a
time, with the full suite run against it.

A fault is **closed** when a named test fails on it. A fault is **equivalent** when no reachable
input can distinguish the mutated code from the original, in which case a test for it would be
theatre. A fault is a **survivor** when it is genuinely observable and nothing notices.

The distinction that matters most is between a test that fails and a test that fails *for the right
reason*. Several entries below were "covered" by a test that went red for an unrelated reason —
a fixture crashing, a precondition tripping — which is coverage that evaporates the moment someone
tidies the unrelated thing.

---

## Kill rate by module

| Module | Faults closed | Left open | Equivalent — no test possible |
|---|---|---|---|
| `core/` (causality, ring, types, timeframes, clock) | 18 | 0 | 4 |
| `waves/pivots.py`, `waves/store.py` | 10 | 0 | 3 |
| `waves/rules.py`, `matcher.py`, `projection.py` | 41 | 0 | 2 |
| `validation/battery.py`, `evaluate.py` | 33 | 0 | 2 |
| `validation/expr.py`, `reality_check.py`, `csv_import.py` | 32 | 1 | 1 |
| `feeds/`, `store/bars.py`, `store/hydrate.py` | 28 | 0 | 1 |
| `engine/live.py`, `labeling/barriers.py`, `backtest/run.py` | 30 | 0 | 13 |
| the served decision path (`/api/decide`, the PUBLIC gate, `LiveEngine.decide`'s wire keys) | 10 | 0 | — |
| `web/app.js`, decision panel only (the keys it reads off the card) | 4 | — | — |
| `hypotheses/` (100 hypotheses, ~5,400 lines) — six properties, not their arithmetic | 11 | 2 | — |
| the catalogue's titles (`Hypothesis.title`, `/api/hypotheses`, `web/assay.js`, hyp.{es,ca}.json) | 8 | 0 | — |
| **Total, where it was measured** | **225** | **3** | **26** |
| `server/app.py`, the three validation routes | **not measured** | — | — |

So the honest headline is **225 of 228 observable faults closed in the modules that were
measured.** Seven of those were closed in one pass, all of them boundary or reach faults this
document had itself declared untestable; the reasons it gave are dissected under "Faults
deliberately left open" below, because four of the five were wrong. The observable total went from
227 to 228 in the same pass: wiring `RETRACEMENTS` into `build_plan` turned a mutant that used to be
equivalent — moving a number nothing read — into one that changes the entry zone on the card. The hypothesis library is no longer a blank row, but read its eleven with the caveat
below: they are eleven faults against six PROPERTIES held over all 100 hypotheses, not eleven
faults sampled across 5,400 lines of arithmetic. Most of that arithmetic is still unmeasured.

The `hypotheses/` row is narrower than its name suggests, and so are the two rows above it.

What the hypothesis row covers is every property `tests/test_hypothesis_library.py` asserts, each
one killed by an edit to a real hypothesis: a one-bar lookahead planted in
`mean_reversion._streak3` and a five-bar one in `volatility._squeeze_release`; the classic
`structure._pivots` fault of stamping a Williams fractal on the bar of the extreme instead of the
bar that confirms it, two candles before it is knowable; a 0.5 and a NaN emitted into a position
series; a hypothesis gated on an absolute price of $30,000; `Hypothesis.signals`'s warm-up mask
deleted; a hypothesis writing to the caller's `close` array in place; one whose condition can never
be true; one declaring a timeframe the store cannot build; and one whose declared `min_warmup` is
dropped below the number of candles its own function refuses to compute below. Each was applied,
watched go red on a NAMED test, reverted, and the file confirmed byte-identical by sha256.

Two faults in the same file are left open, both inside `Hypothesis.signals`, and both are open for
the same reason: the registry as it stands cannot tell the difference.

- **The `dtype=np.int8` in `np.asarray(self.fn(s), dtype=np.int8)` can be deleted and the suite
  stays green.** Every one of the hundred already builds its own `int8` array, so the cast has
  nothing to convert and `test_signals_returns_one_int8_per_candle`'s dtype branch is unreachable.
  The cast is a defence against a hypothesis that returns floats, and the day one arrives the
  alphabet property catches it on the RAW output anyway — so the cast being untested is a
  statement about which layer holds the guarantee, not a missing test. Do not delete it: it is
  what stops a float reaching an int8 position silently.
- **The `raise ValueError` for a wrong-length return can be deleted too.** No hypothesis returns
  the wrong number of signals, so nothing reaches the line. `test_signals_returns_one_int8_per_candle`
  would catch the resulting mismatch — but only in a run where some hypothesis is already broken.

The titles row is the one added with this document's own machinery pointed at the newest work,
and it is worth reading because seven of its eight faults were live survivors before the tests
that kill them were written: `title` dropped from the `/api/hypotheses` payload; the wire key
renamed on the server side only; the same key renamed in `web/assay.js` only; `register()`'s
blank-title refusal deleted; `title` given a default so it can be omitted at registration; an
English title replaced by its own identifier; an English title de-slugged from its identifier
("Turn of month, long"); an es/ca title replaced by the identifier. Every one of those leaves the
product looking like it works. The picker does not break on a missing title — `hypLabel` degrades
to the bare identifier, which is the exact screen this change existed to replace, restored one row
at a time with nothing anywhere saying so.

What it does not cover is any hypothesis's arithmetic. A hypothesis that is causal, finite,
scale-free and pure while measuring entirely the wrong thing passes every property in the file. Two
faults were planted to confirm that and both survived, as expected: a Bollinger band rebuilt at one
sigma instead of two, and `streak3_fade`'s sign flipped end to end. The suite stays green for both.
They are not counted above, because a survivor that no property claims to catch is a boundary
rather than a hole — but it is a real boundary, and it is where the next reader should look.
Closing it means testing 100 formulas against 100 restatements of themselves, which is exactly the
work these properties were written to avoid, so it should probably not be closed at all.

The two server rows are narrower than their names suggest too, and the narrowness is the point. The
decision-path row covers the route the chart is served through and nothing else: the gate on all
three private routes, the accepted spellings of `WAVELAB_PUBLIC`, the WATCH ceiling at the route,
and each key the panel reads. The `web/app.js` row is not a front-end test — no JavaScript is
executed anywhere in this suite. It is measured because `test_decide_route.py` parses the required
key names OUT of `web/app.js` instead of retyping them, so renaming `h.entry_lo`,
`h.invalidation_price`, `d.reasons` or a `VERDICT` key **in the JavaScript** turns the suite red
too. What is
checked is that the two sides agree on the names; whether the panel then draws the right thing is
still checked by nobody.

"Left open" counts only faults that are observable and deliberately not chased; the reason for each
is given below. Nothing is left open because it was hard.

One source change was made while closing the hypothesis hole, and it is recorded here rather than
counted as a kill. `mean_reversion._prev` did `out[0] = np.nan` on a zero-length array, which is an
`IndexError`, and it took down `sweep_rejection_20` and `streak3_fade` — the only two hypotheses in
that file reaching `_prev` with no length guard in front of it — on a series with no candles.
`evaluate_all` swallows that in a bare `except`, so the visible symptom was two hypotheses quietly
absent from the report, which is the exact failure mode this document named as untested. The guard
is two lines; the empty-series test that found it is counted once, against the behaviour, not twice.

A second source change was made while measuring the titles, and it is recorded the same way.
`seasonality.turn_of_month_long` was titled "Long the turn of the month", which is its own
identifier with the underscores taken out. The picker prints `title · name`, so that row said the
same thing twice and the reader was back to decoding a key — the bug titles were added to fix,
surviving in one of the hundred. It now reads "Long the last day of the month and the first three",
which is what `_turn_of_month` does, and es/ca were retranslated to match. The `en` fingerprints in
hyp.{es,ca}.json were NOT touched: they hash rationale and prior, not title, and re-stamping them
for a title change would be the silencing this document warns about. Measured before the change:
one of the hundred failed the new check, and it is the only one.

One accounting note, because it is the kind of thing that quietly inflates a table: closing
`provisional_window`'s boundary required a source change (`n < 1` now raises instead of dying inside
NumPy on `int(ts[-1])`), and that change made the original fault — `window(max(0, n-1))` becoming
`window(max(1, n-1))` — equivalent rather than killed, since `n >= 2` makes the two expressions
identical. It is counted once, as closed, on the strength of the two tests that pin the new
boundary; it is not also counted as an equivalent.

A third source change was made while closing the `--quick` cutoff, and it is a refactor rather than
a fix: the cutoff expression moved out of the `--quick` branch into a module-level `_quick_cutoff()`
in `store/hydrate.py`. The expression is unchanged character for character and the long comment moved
with it into the docstring, so public behaviour is identical — verified by the end-to-end test, which
drives the real `hydrate(quick=True)` and not the helper. It is recorded because it is REACHABILITY,
not tidying: the cutoff sat three lines above the first `httpx` call, so the only way to reach it was
to run a hydration. The extraction is what lets the rule be asserted at 0.00 s across four timezones;
the end-to-end test is what stops the extracted helper from being silently bypassed, and both are
present. A refactor that exists to make something testable is still a change to the file under test,
and it belongs in this list.

---

## Faults deliberately left open, and why

One entry stands here. Six more used to, closed in a single pass, and the reasons this document gave
for leaving them open are reproduced against each one below — because every one of those reasons was
wrong, and wrong in the same way each time: **it described the hardest test it could think of, found
that test bad, and stopped.** Four reasons covered the six faults; all four fell to an easier test
that nobody had looked for. That is the failure mode to watch for in this section. If an entry here
says "a test for this would have to X", the first question is whether it would.

### `validation/` — one

**Blocks are cut from the tail rather than the head.** `Mc[:, : nb * L]` keeps the first `nb` whole
blocks, so the final `n % L` observations — the most recent ones — are dropped. Taking
`Mc[:, n - nb * L:]` instead drops the oldest ones. Re-measured this pass: the suite is **green**
under that edit, so this is still a live survivor. Which end is a defensible design choice rather
than a defect, and it now carries a comment in the source saying so — including the bound that makes
it arbitrary: the `nb >= 30` guard forces `L <= n/30`, so under 3.3% of the sample is dropped either
way, and `Mc` was centred over all `n` observations so neither end carries the mean. The comment also
names the condition under which it would stop being arbitrary: at `nb = 4` this would be a quarter of
the series, and then the end you keep is a real decision about which regime the null is drawn from.

Do not confuse this with `nb = n // L` itself, which is a different edit on the same line and **is**
killed: `nb` also decides the `nb >= 30` switch between the two estimators, so
`max(1, n // L - 1)` re-routes borderline series onto the other one and
`test_which_of_the_two_bootstraps_runs_is_decided_at_thirty_blocks` goes red. That conflation was
made twice while closing this section. The block COUNT is guarded; the choice of END is not.

### Closed this pass, and what the reason for leaving them open got wrong

**The cost ceiling and the noise floor** (`cost_r > max_cost_r` loosened to `>=`,
`stop_atr < min_stop_atr` to `<=`). *The reason given:* a test would have to solve for an input
whose arithmetic lands exactly on `0.20` or `0.75` to the last bit, and would then be pinned to that
solution rather than to the rule. *Why it was wrong:* both thresholds are CONFIGURATION, and
`build_plan` takes the config as an argument. Run the plan once, read the `cost_r` or `stop_atr` the
function computed, and hand the same scenario back under `replace(PlanConfig(), max_cost_r=that)`.
Neither threshold feeds the computation of the number compared against it, so the second run
reproduces the identical float and the comparison sits exactly on equality — no float is solved for,
nothing is written down, and the boundary follows the fee arithmetic if it ever changes. Each test
is a pincer: accepted at the threshold, refused one ULP past it via `math.nextafter`. Without the
second half both would also pass against the gate deleted outright — verified, deleting either gate
turns the matching test red. `TestTheGatesDecideAtTheirOwnBoundary`, `tests/test_wave_rules.py`.

**The empty-zone check** (`lo >= hi` tightened to `lo > hi`). *The reason given:* the same. *Why it
was wrong:* this one needs no float at all. `lo == hi` is unreachable from `match_impulses` — every
route needs a zero-length leg, which breaks R1 — but `build_plan` is exported and takes any
`Hypothesis`, so the flat count is built by hand, and the test asserts R1 *does* reject it so its
own premise fails loudly if that ever changes. Under the mutant the single surviving price is priced
as a range and the user is refused by the FEE gate instead, told the round trip costs 150% of R for
a zone that does not exist.

**The stationary bootstrap never starts a block at the last observation**
(`rng.integers(0, n)` → `(0, n - 1)`, both draw sites). *The reason given:* pinning it needs a
distributional test over many seeds, and a single 50-index resample covers only ~31 positions, so
any coverage assertion is flaky by construction. *Why it was wrong:* the argument is about ONE
resample. Draw 120 of them from a single `default_rng(0)` and the union of their block starts is a
fixed set of integers — deterministic, not probabilistic. It either reaches index 49 every run
forever or never. Re-measured: unmutated, this seed first covers all 50 positions after 23
resamples, so 120 is a 5× margin, and the test's docstring says so to stop anyone trimming it.
The assertion is on block STARTS, not on visited indices: `i = (i + 1) % n` means index `n-1` is
visited in nearly every resample even under the mutant, which is why `idx.max() < n` passes mutated.
A continuation is always exactly `(prev + 1) % n`, so any other index is certainly a fresh draw;
a fresh draw landing on the continuation index is missed, which makes the observed set a strict
subset of the true starts — it can under-report coverage, never invent it.
`test_every_observation_can_START_a_block_including_the_very_last_one`, `tests/test_assay.py`.

**`--quick` computes its month cutoff from the local civil date rather than UTC**
(`datetime.now(UTC)` → `datetime.now()`). *The reason given:* a test for it would pin the clock
rather than a property. *Why it was wrong:* the property is "two machines at the same instant ask
for the same months", and that needs the clock frozen as a CONTROLLED variable, not asserted. One
instant (23:30 UTC on the last day of a month), four boxes from UTC-12 to UTC+14, one cutoff. The
anti-tautology half is separate and necessary — agreement across timezones is also satisfied by a
constant — so `test_the_cutoff_still_follows_the_utc_month` asserts that one hour crossing into a
new UTC month moves the cutoff by exactly the length of the month it left, which is the
`.replace(day=1)` snap. Verified against two further mutants: a hard-coded `date(2022, 3, 2)` and
`.replace(day=1)` dropped both turn it red while leaving the timezone test green. A third test runs
two full `hydrate(quick=True)` passes over the stub archive at one instant in two zones — it is the
only test in the suite that enters the `--quick` branch at all, and it also pins `timedelta(days=730)`,
which nothing did before. `TestQuickAsksForTheSameTwoYearsOnEveryMachine`,
`tests/test_archive_and_store.py`.

**The gap counter's window length is unobservable here.** *The reason given:* `gaps_in_window` is
measured over 50 trigger bars, changing it to 500 makes no difference on any fixture, and a
synthetic test whose only content is the number 500 would assert the constant against itself.
*Two things wrong with that.* First it had the source backwards: the shipped value is
`min(500, len(ring))` and 50 is the mutant, so the paragraph described the fault as the code.
Second, the fixture objection assumed `make_bars` cannot produce a hole, and it can — drop all
fifteen source minutes of a 15m period and no trigger bar is built, leaving a real timestamp gap
that `Ring.window` recomputes from `np.diff(ts)`. Two such outages, at ages 298 and 119 in a ring of
531, are outside the last 50 bars and inside 400. The test is not about the number: it asserts
`ring.window(50).n_gaps == 0` as a stated precondition and then that the badge reports 2, and those
two numbers cannot be the same number by construction. Verified to be about reach and not about the
literal — `500 → 400` leaves it green; the kill band is any window below 299.
`TestTheDataHealthBadge`, `tests/test_live_engine.py`.

---

## Known inconsistencies the suite pins rather than fixes

### The R:R on a signal describes a different entry from the one it is booked at

`build_plan` computes `rr_t2` and `cost_r` from `entry = (zone_lo + zone_hi) / 2` — the middle of the
published entry zone. `run_backtest` fills at `htf.close`. Measured on the 20-day fixture:
**131 of 132 signals carry an R:R describing a different entry price from the one their outcome is
resolved at**, the worst by 2.29R (recorded 4.03 against a geometric 6.32).

So `expectancy_r` is measured off the fill and `rr` is quoted off the zone, and the two travel side
by side in the same report. `test_a_signal_books_the_trade_the_card_actually_offered` proves the
signal faithfully *copies* the card's numbers; it cannot say those numbers are consistent with each
other, and they are not.

Resolving this is a product decision — book at the zone midpoint, or recompute the plan's arithmetic
at the fill — and not one a test should make quietly.
`test_the_booked_R_R_is_the_zone_s_and_not_the_fill_s_and_that_is_recorded_here` asserts the
discrepancy so that whoever fixes it sees a red test and has to say which way they went.

### A stop on the wrong side of the entry is planned normally

`build_plan` computes `risk = abs(entry - stop)` and refuses only when that is zero. A stop the same
distance on the *wrong* side — a long whose stop sits above where it buys — measures as the same
positive risk and produces a plan with a size, a cost and a required hit rate. Only the exactly-zero
case is refused, and that case is now tested
(`test_a_stop_that_lands_on_the_entry_is_refused_and_does_not_take_the_engine_down`), which also
pins the wrong-side behaviour as it stands so it cannot change silently.

### Four hypotheses read a market with zero movement as maximally oversold

TA-Lib returns 0, not NaN, for RSI, MFI and Williams %R when there is neither a gain nor a loss to
divide by, and 0 is below every oversold threshold in the library. So a series in which the price
never moves a tick reads as the most oversold market possible, and `flow.mfi14_extremes`,
`mean_reversion.rsi14_fixed_3070`, `mean_reversion.rsi14_cardwell_sma200` and
`mean_reversion.willr14_fade` go long on every candle of it.

It cannot happen on BTC 1m, which is why this is a note and not a bug report. It can happen the day
the catalogue is pointed at a stablecoin, a halted equity or a padded series, and then it fires
continuously and confidently. The test asserts the set is exactly those four, so a fifth arrival
is caught, and a fix turns the test red rather than passing silently.

The eight `seasonality` hypotheses also signal on a flat market, and for them it is correct: they
read the clock, and the clock does not care what the price did. The three that do not
(`cme_gap_monday`, `overlap_day_trend`, `pre_funding_fade`) are the three that gate their calendar
window on a price condition as well. Both halves are asserted, because "the price-blind ones fire
and the price-gated ones do not" is the statement that has content; either half alone is a number.

### `min_warmup` is a mask on the head of the series, not a bound on memory

The natural reading of `min_warmup=200` is "a signal at bar i depends only on the last 200
candles", and it is false. `Hypothesis.signals` uses it for one thing: zeroing the first N
positions. `structure._hold` carries the last event forward until the opposite one arrives, with no
limit at all, so a hypothesis built on it can hold a position that is information from a candle
hundreds of bars outside any warm-up window.

Measured with a built fixture rather than a seed: six hundred candles climb in a straight line, so
`donchian_break_20` breaks out early and is long from then on; then the market stops dead for five
hundred candles, with no new high, no new low and no event of any kind. Given the whole series the
hypothesis is long through all of that silence. Given only the silence plus a re-warm of its full
declared 200 candles, it is flat, and stays flat, for every one of the 300 bars compared.

This matters to anyone sizing a ring buffer from `min_warmup`, and it is a design choice rather
than a defect, so it is pinned rather than fixed. Three other hypotheses disagree with themselves
by one to seven bars after a re-warm for the much milder reason that an EMA's memory is infinite
(`candles.engulfing_trend`, `mean_reversion.stretch_ema200_atr14`, the `trend` EMA stack); that is
rounding, and it is not pinned.

### Dead configuration — both entries resolved, and they went opposite ways

Recorded rather than deleted, because "wire it up" and "delete it" were the right answers to two
things that looked identical in a diff.

- **`RETRACEMENTS` in `rules.py` is now wired up.** It had exactly one reference in the repo — its
  own definition — while `projection.py` open-coded `0.500, 0.786` and `0.382, 0.500`. That is two
  definitions of one published level: the zone drawn on the card followed the literals and the table
  could have gone on claiming something else with nothing to notice. `build_plan` reads both bounds
  from the table now. The values were checked bit-for-bit before the change (`struct.pack(">d", …)`
  on both sides) and are unchanged, so **no ratio moved and no card changed** — this is not a
  product change. Measured both directions afterwards: moving `w2`'s upper bound to 0.886 now turns
  three tests red where before it turned none, and re-typing the literals back into `projection.py`
  — behaviour-identical, so invisible to every other test in the suite — is caught by
  `TestTheZoneIsReadFromTheRetracementTable`. That class is the only guard on the wiring itself, and
  it names no Fibonacci number: it sets the row to `(0.0, 1.0)` and requires the zone to be the whole
  of wave 1, then to `(0.0, 0.5)` and requires it to be the top half. `_fib_zone` sorts its two
  outputs, so reading the two bounds in the wrong ORDER is an equivalent mutant here, not a hole.
  `(core_lo, core_hi)` is still descriptive and still computed from by nothing: it is deliberately
  not wired into `matcher.score_guidelines`, whose `0.618`/`0.382` are hand-picked SCORING constants
  that go to trials.sqlite when they move, and coupling them would mean redrawing the entry zone
  silently retunes the scorer that ranks the counts. That reasoning is now a comment above the table.
- **`max_starts` in `matcher.py` is deleted.** It guarded a branch unreachable under any shipped
  config, and the claim was verified exhaustively before acting — over `n` in 0..399 and `k` in 3..6
  at the shipped `max_from_edge = 3`, the branch is reached 0 times; it first binds at
  `max_starts <= 2`, below `max_from_edge`. The cause is that every candidate is right-anchored with
  a fixed length, so each `end` yields exactly one `start = end - k` and the number of starting
  points is identically `max_from_edge + 1`. It was a second dial on an axis another dial already
  owned, and the tighter one always won. Deleted rather than "made to do what its name says",
  because what its name says is what `max_from_edge` already does. Nothing outside `matcher.py`
  referenced it — no config file, no route, no test. The module docstring's cost line was corrected
  with it: it read "~13 starts ≈ 100 checks per update", which was this dead knob's arithmetic and
  overstated the real cost by 3× (it is 4 × 4 × 2 = 32). `if start < 0: continue` is also provably
  unreachable and was KEPT and labelled — it is a slice guard, not a knob, and a negative start
  would wrap `pivots[start:end]` silently instead of raising.

---

## Properties nothing verifies at all

These are not mutants that survived. They are behaviours with no test of any kind.

### 1. The served path — the decision route is now covered; the validation routes are not

`server/app.py` is 582 lines. `test_server_status.py` (4 tests) covers warm-up status transitions,
`test_limits.py` (7 tests) covers rate limiting, `test_server_i18n.py` (9 tests) checks that every
emitted string is translated, and `test_decide_route.py` (30 tests) covers `/api/decide` and the
PUBLIC gate. `/api/hypotheses` is driven only for the catalogue prose — one test in
`test_hypothesis_i18n.py` asserts the payload carries a non-empty `title` for all 100 matching the
registry, under the key `web/assay.js` reads. Its other fields, and `/api/validate`,
`/api/validate_csv` and `/api/validate_rule` in full, are still driven by no test: their bodies are
thin, but "thin" is what was said about `/api/decide`.

`test_decide_route.py` builds its own engine — 6,000 synthetic 1m bars, ~20 ms — rather than
warming from the store, and reaches the routes with `TestClient` NOT used as a context manager, so
the lifespan never runs and the feed never opens a socket to Binance.

What it closes, and what it deliberately only pins:

- **The PUBLIC gate is no longer protected by code review alone.** With `WAVELAB_PUBLIC` set,
  `/api/decide` and `/api/history` answer 404 and `/ws` closes with 1008 before `accept()`. The
  flag is exercised through a second, isolated execution of `server/app.py` with the variable set,
  not by monkeypatching `PUBLIC` — because `PUBLIC` is an import-time expression and the
  accepted-spellings list is half the guarantee. Dropping `"yes"` from it is now a red test named
  after the value.
- **`emitting` still has no production caller, and that is now stated by a test instead of by this
  paragraph.** `/api/decide` calls `APP.engine.decide(tf, price)` in any mode and returns its plain
  dict, so `Decision`'s constructor — the documented last line of defence — never runs on the
  served path. `test_catching_up_changes_nothing_on_the_card` measures the consequence: the card
  served while CATCH_UP is byte-identical to the one served while LIVE, entry zones and all. That
  is harmless only because the PRIOR ceiling caps every verdict at WATCH, so a stale zone is shown
  as something to watch and never as something to take. The test asserts the ceiling is still
  PRIOR, so whoever raises maturity lands on a red test and has to wire `emitting` in.
- **The invariants are exercised through the route, not around it.** The card the route just served
  is lifted into a `Decision` exactly as a caller would have to read it, and the constructor is
  asked to accept it. Each of the three refusals — no plan, catching up, PRIOR maturity — is
  provoked from a real response: removing any one of them from `__post_init__` turns exactly one
  named test in that file red. This is not a second copy of
  `TestAnActionableDecisionHasToBeEarned`; it is the claim that the route's output is something
  those properties would accept.
- **`LIVE` is never demoted to `CATCH_UP` by drift.** Untouched, still open, and it is the reason
  the point above is only pinned and not fixed: even a route that consulted `emitting` would be
  told LIVE. `update_health` promotes WARMUP → LIVE and demotes CATCH_UP → LIVE; there is no
  LIVE → CATCH_UP edge. The only way in is `check_clock`, which
  compares *consecutive calls*. Measured: warm to LIVE, then advance the clock one second at a time
  for an hour, and the engine ends four trigger bars stale with the badge still reading LIVE and
  `emitting` still True. That is a stalled websocket that never disconnects, which is more common
  than a suspended laptop. The two CATCH_UP tests in the suite pin the thresholds (two bars of
  silence, and the exact lag at which catching-up ends) and neither reaches this: one moves the
  clock in a single three-bar step, which `check_clock` does see, and the other enters CATCH_UP by
  assigning the mode directly.

### 2. Modules with no tests

| Module | Lines | What it does |
|---|---|---|
| `core/bus.py` | 108 | the internal event bus |
| `feeds/binance_derivs.py` | 123 | funding and open interest |
| `store/trials.py` | 78 | the trials ledger — where parameter changes are meant to be recorded |
| `collect.py` | 66 | the collection entry point |
| `registry.py` | 59 | provider registration |

`store/trials.py` is worth singling out. The README's argument against parameter search rests on
recording every experiment so the effective N cannot lie. The ledger that records them is untested.

### 3. Cross-cutting

- **Nothing RUNS the web front end.** No JavaScript is executed anywhere in this suite, so nothing
  checks that the panel draws the right thing. Three files now read `web/` as data rather than
  ignoring it: `test_server_i18n.py` parses the translation tables out of `i18n.js`,
  `test_decide_route.py` parses the decision panel's key names out of `app.js`, and
  `test_hypothesis_i18n.py` parses the picker's key names out of `assay.js` — so a rename on
  either side of any of the three fails. That is agreement on names, not behaviour, and the rest of
  `web/` — the chart, the health badge, the timeframe selector, `paintCatalogue`'s grouping and
  its keep-the-selection branch, all of `assay.html` — is covered by nothing. Concretely, and this
  was checked in a browser rather than assumed: switching en/es/ca relabels all 100 options, keeps
  the current selection and leaves the console clean. No test in this repo would notice if any of
  that stopped being true.
- **Nothing tests behaviour under a real feed.** Every websocket, REST and archive test drives a
  stub transport. The one test against real data is marked `net` and excluded from the gate.
- **`provisional_window` has no caller in `src/`.** Its boundary is now tested because a chart
  endpoint asking for "the current candle" would hit it first, but nothing exercises it in anger.

---

## Things that look like coverage and are not

Recorded so nobody re-derives them.

**One synthetic price path wearing four timeframes.** The hypothesis fixture built its 15m, 1h, 4h
and 1d series from one seed, so all four had identical prices and differed only in their
timestamps: a property asserted on four timeframes was being asserted four times on one price path
rather than over four of them. It was found by a planted fault — a hypothesis gated on an absolute
price of $30,000 — surviving the scale test, because that one path never crossed $30,000 after its
warm-up. Each timeframe now draws its own path, and the scale test also runs on real candles, where
BTC spans 20k to 120k inside the window and any absolute constant a hypothesis might compare
against falls inside the series.

**A shock so large it stopped testing anything.** The same fixture's regime change was originally
`r*3 + 0.03` per bar, which over five hundred candles carries the price to 1e10. Every threshold in
the library is then permanently on one side of itself and none of them is being exercised. It is
now sized as a bear market: three times the volatility and a drift that takes about 95% off. A
violent fixture and a useful one are not the same thing.

**A purity test shielded by the tests that ran before it.** The hypothesis library's "does not
write to the candles it was handed" test compares the input arrays before and after the call, and
it took its series from the shared session fixture. An in-place edit that is idempotent — a clip, a
`nan_to_num`, a sort — damages those arrays the first time anything runs and is a no-op every time
after, so by the time the purity test ran it was comparing the damage against itself. A planted
in-place `np.clip` survived it. The test now builds its own series. Any test that asserts "the
input was not modified" has to own its input.

**A single cut point against a shallow lookahead.** The first version of the causality property cut
the series in two and compared the overlap. That is a weak instrument against `close[i+1]`: reading
one bar ahead moves exactly ONE bar of the overlap — the one adjacent to the cut — and if that bar
is flat in both runs the comparison comes back clean. Measured: a deliberate one-bar lookahead
planted in `mean_reversion.streak3_fade` survived the bulk comparison at four of five cut points
and was caught only by luck on the fifth. The property now probes twelve individual bars per
hypothesis per timeframe, truncating at each one and requiring the signal there to be unchanged,
and spends two thirds of its probes on bars where the hypothesis is actually in the market — which
is where a stolen candle changes the answer. The bulk comparison is kept as a cheap broad net under
it, not as the guard.

**A test whose headline is guaranteed by a line somewhere else.**
`test_fewer_candles_than_any_indicator_needs_is_answered_with_silence` runs the registry on 1, 2, 5
and 50 candles and asserts none of them takes a position. It cannot fail on that claim: the
shallowest `min_warmup` in the registry is 200, and `Hypothesis.signals` zeroes the first
`min_warmup` positions, so on a 50-candle series the ENTIRE output is zeroed before the assertion
sees it, whatever the hypothesis computed. What the test really holds down is that none of the
hundred raises or returns the wrong length on a series far shorter than any indicator it uses —
which is worth having, and is how the `_prev` `IndexError` above was found. Read the name as
"does not blow up", not as "declines to trade". If a hypothesis is ever registered with a warm-up
below 50, the headline claim starts being tested for the first time.

**Fabricated lookaheads are not a substitute for a real one.** `TestTheHarnessHasTeeth` proves the
causality checker bites, but it proves it on hand-built functions with one timeframe, a 200-candle
warm-up and a signal on nearly every bar — the friendliest case the probe will ever see.
`test_a_lookahead_planted_in_a_REAL_hypothesis_is_caught_and_named` plants the same fault in
`mean_reversion.streak3_fade` and requires `acausal_probe` to name a bar on all four timeframes it
declares. Measured while writing it, and this is the part worth carrying forward: `acausal_indices`
at bar 1000 — BOTH bulk arms, every bar before the cut compared — comes back EMPTY on that fault.
The broad net does not catch a one-bar lookahead in a real hypothesis at all. Against the commonest
mistake in the industry the twelve probes are the only thing standing there, so the note at the top
of this file about cutting cut points before cutting probes is not a preference, it is the
measurement.

**Four fabricated lookaheads sharing one name.** The same file memoises `h.signals(s)` on
`(hypothesis name, series)` to stay inside the deploy gate's time budget. The tests that prove the
causality checker has teeth fabricate throwaway `Hypothesis` objects, and all four were called
`teeth.probe`, so they were handed each other's cached answers and one deliberate lookahead came
back green. They are named after their functions now. A memo keyed on a name is a trap wherever
names are not unique.

**A `>=` boundary tested only from far away.** Several thresholds in this codebase were "tested" by
fixtures sitting nowhere near them — the 90% price-parseability bar checked with a 67%-readable
file, the 30-episode bar checked at 24 and 60, the 95th-percentile bar checked at 0 and 100. Each of
those pins the constant only to an interval. All three now have fixtures landing on the bar itself
and immediately below it; when adding a threshold test, put a case *on* the boundary or it will
slide.

**`np.isnan(x) is False`.** NumPy returns `np.bool_`, which is never `is` a Python `bool`, so an
assertion of that shape is true for every input. I wrote one myself while closing the out-of-sample
signal floor and caught it only because the mutant it was aimed at survived. Use `bool(...)` or `==`.

**A test defined inside another function.** Python accepts it, pytest never collects it, and it
looks completely normal in a diff. One of mine landed nested inside a `net`-marked function and
"passed" without running. The suite is now checked for this by an AST scan; if you are moving tests
around, re-run it.

**A key list derived from a file that moved.** `test_decide_route.py` parses the keys it requires
out of `web/app.js` by brace matching, which is the right call — a retyped list only proves the
list agrees with itself — but an extractor that finds nothing returns an empty set, and every
`required - set(payload)` assertion built on it then passes by asserting nothing. Rename a function
in `app.js` and four shape tests go green while checking zero keys.
`test_the_extractor_read_something_real_out_of_app_js` is the whole defence: it is the one test in
that file that hard-codes names, and it exists so the others do not have to. Same shape as the
`at_the_limit > 0` problem below, arriving from the opposite direction — there a precondition was
doing a test's work, here a test would be doing no work at all with nothing to say so.

**A test whose only kill comes from a precondition.** `test_the_control_is_held_for_the_same_window`
used to hold down two real faults through its `at_the_limit > 0` fixture-health check — an assertion
whose own message says "this test proves nothing". Any change to the fixture, the seed or
`MAX_BARS_HOLD` would have freed both faults silently. Both are now held by
`test_the_control_is_the_same_trade_taken_at_a_different_moment`, which asserts the twin's geometry
directly.

**A survivor-only property test.** `test_every_pivot_sits_on_its_own_bar_s_high_or_low` inspects the
pivots that survived to the end of a run, and a wrongly-seeded extreme is almost always overwritten
before it gets there. Measured across 50 seeds it caught its target fault on 5 of them, and on those
it caught one bad pivot out of ~57. It read as the strongest test in its class and was the weakest
by an order of magnitude. It is kept as a cheap second net;
`test_every_flip_seeds_the_new_extreme_by_the_rule_the_knob_selected` is the real one, checking at
the moment the extreme is written and under both `on_close` settings.

**A parametrised test whose kill is single-point.** `test_a_bar_that_misses_the_grid_is_refused_
however_narrowly` runs four offsets and only `off=1` kills a grid check weakened from `!= 0` to
`> 1`; the other three satisfy `> 1` and sail through. The parametrisation correctly states the
general claim, and trimming the "redundant-looking" first case would reopen the hole in silence.
There is now a comment on the list saying so.

**The replay determinism gate does not cover the feed site.**
`test_the_real_engine_replays_identically_and_reads_no_bar_it_was_not_given` is the most valuable
test in the suite and its prefix arm cannot be broken by any edit confined to `live.py` — `streaming()`
throws the bar array away, so the engine can only reach what `on_bar` handed it. It is a standing
guard against a future warm-up that precomputes over the whole array, which is what it is for. It is
*not* a second net under the detector's feed call: when the high/low arguments there are swapped, the
gate's own two assertions do not fire and the test goes red on its fixture-health precondition
instead. `test_every_pivot_it_counts_is_an_extreme_of_the_bar_it_is_stamped_on` is the sole guard on
that site.

**Two capability lists where only one is frozen.** `test_expr_sandbox.py` pins the AST node types
the rule interpreter will execute. It did not pin `SAFE_FUNCS`, and because `v_Name` resolves bare
names out of `{**series, **SAFE_FUNCS}`, every entry there is a first-class *value* — so one
higher-order function in that dict (`"apply": lambda f, x: f(x)`) restores arbitrary calls to a
string typed into a web form, with the whole suite green. Both lists are frozen now. If you add a
third capability surface, freeze it in the same file.

---

## Conventions this suite pins that are not written down elsewhere

These are decisions the code makes silently. A test is now the only place they are stated, so
changing the code means changing a number in a test and noticing.

- **A bar owns the last millisecond of its period.** `close_time_ms == open + tf.ms - 1`, and two
  adjacent bars abut with no shared instant and none between them.
- **The base rate grades a signal on the move AFTER the bar it fires on.** `fwd[i]` is the return
  from close `i+1` to close `i+1+h` — a fill at the next open, not at the close the signal was
  computed from. This is *not* what `battery.py`'s module docstring describes.
- **The gap-heal window overlaps by exactly one bar** on every websocket reconnect.
- **`parse_klines_zip` and `BarStore.ingest` both keep the FIRST of a duplicated minute.** They see
  the same minute through different paths and the anti-join makes the winner permanent.
- **The cooldown debounces on the bare archetype** (`w2`), not on archetype-plus-direction, so a
  long and a short off the same count are one observation. Reading it the other way takes this
  fixture from +0.043 to −0.009 expectancy and flips `edge_vs_null`. Both readings are defensible;
  this is the one in force.
- **A hypothesis's `signals()` is checked against the six properties in
  `test_hypothesis_library.py` and against nothing else.** Causality, alphabet, warm-up, scale,
  purity, degeneracy. Its formula is checked by no test in this repo, deliberately.
- **Every hypothesis in the registry is scale-free.** All 100, measured on a power-of-two factor
  where the float arithmetic is exact and on 3.7 where it is not. There is no exception list, and
  the first hypothesis to compare a price against an absolute number will need one — or a reason.
- **The declared `min_warmup` is at least as deep as the hypothesis's own refusal floor.** Several
  families decline to compute below a module constant (`candles._MIN_BARS = 320`, momentum's 391
  for the 12-1 window); every hypothesis declares a warm-up that covers its own, so none of them
  ever answers "flat, for want of data" at a bar it declared itself ready for. Nothing said this
  before; the causality probe now enforces it as a side effect and says so in its failure message.
- **A hypothesis never writes to the `Series` it is given.** `evaluate_all` runs all hundred over
  the same arrays in registry order, so one in-place edit would make the result depend on
  alphabetical order.
- **`candles.range_gap` fires 102 times in nine years** — 99 on 15m, 3 on 1h, none on 4h. It is the
  only hypothesis silent over the 1,600-candle real-data fixture, and it is rare rather than dead.
  If a second name joins it there, that one probably is dead.
- **Every hypothesis has an English title that adds at least one word its own identifier does not
  contain**, filler words aside, and a Spanish and Catalan one that is neither empty, nor the
  English, nor the identifier. The picker prints `title · name` and shows both, so a title made of
  the identifier's own words is a row that says the same thing twice. Measured: one of the hundred
  was exactly that when the check was written. Shared domain words are fine and expected —
  "Hammer after an oversold fall", "Flip sides on every Donchian 20 break" — what is refused is a
  title whose remainder is empty.
- **`title` is not in the es/ca staleness fingerprint, deliberately.** `en` hashes rationale and
  prior only, so rewording a title in Python leaves the translations stale with nothing to say so.
  That hole is open on purpose: covering it would move all 200 stamps at once, and a re-stamp of
  200 entries nobody retranslated hides a whole stale rationale to catch one stale line. If a title
  fingerprint is ever wanted it belongs in a SECOND field, stamped on its own.
- **Bars before a user's first imported signal are flat**, and a NumPy `RuntimeWarning` is a test
  failure suite-wide (`filterwarnings` in `pyproject.toml`) because NaN→int8 is undefined and
  silently produces a position 128× anything anyone asked for.

---

## Reproducing any claim in this document

```sh
export PATH="/opt/homebrew/bin:$PATH"
uv run pytest -m "not net"      # 962 passed, 1 deselected, ~6.0 s
uv run pytest                   # 963 passed, ~10 s — RUNS the net test, and needs the internet
uv run ruff check .             # clean
```

`test_hypothesis_library.py` reads real candles from `data/bars/`. If that store holds fewer than
56 months it SKIPS its real-candle arm, loudly, and the synthetic arm still runs. A skip there is
not a pass: half of every property in that file is then unmeasured, including the arm that caught
the planted one-bar lookahead when the synthetic arm did not.

To check a single fault: edit the source, run the suite, confirm the named test fails, revert, and
verify with `shasum -a 256` that the file is byte-identical. Purge `__pycache__` between runs, or set
`PYTHONDONTWRITEBYTECODE=1` — several of these faults are same-byte-length edits and a stale `.pyc`
will keep executing the mutant after the revert, which produces a confidently wrong verdict.

### When this document was last audited against itself

A coverage document that is wrong about its own residual list is worse than none, and this one has
been wrong. The last pass re-applied all seven faults from the previous round independently — six
killed by the tests that claim them, one (`nb = n // L`) killed by a test nobody had connected to it
— and then spot-checked nine claims it was NOT changing, drawn from nine different modules, by
applying the mutant and watching the named test go red:

| Claim spot-checked | Mutant applied | Held? |
|---|---|---|
| a bar owns the last ms of its period (`core/timeframes.py`) | `+ tf.ms - 1` → `+ tf.ms` | yes |
| the base rate grades the move AFTER the signal bar (`validation/battery.py`) | `fwd` shifted back one bar | yes |
| ingest keeps the FIRST of a duplicated minute (`store/bars.py`) | `keep="first"` → `"last"` | yes |
| `SAFE_FUNCS` is frozen (`validation/expr.py`) | added `"apply": lambda f, x: f(x)` | yes |
| the warm-up mask is what enforces `min_warmup` (`hypotheses/base.py`) | mask deleted | yes |
| the cooldown debounces on the bare archetype (`backtest/run.py`) | key → archetype + direction | yes |
| `provisional_window` refuses `n < 1` (`core/ring.py`) | `n < 1` → `n < 0` | yes |
| a pivot is priced at its bar's wick (`waves/pivots.py`) | high/low swapped | yes |
| blocks are cut from the tail (`validation/reality_check.py`) | `[:, :nb*L]` → `[:, n-nb*L:]` | **no — still a survivor, as this document says** |

Nine of nine behaved as written. The one thing the audit found stale was in the paragraph describing
the gap counter, which had the shipped constant and the mutant the wrong way round (it said the
window was 50 and the mutant 500; the source ships 500). When adding an entry here, re-run at least
half a dozen of the claims you are NOT touching — the entry that goes stale is never the one you are
looking at.
