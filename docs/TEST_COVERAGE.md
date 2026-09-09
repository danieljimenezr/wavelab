# What the test suite protects, and what it does not

**582 tests, ~4.1 s, `pytest -m "not net"`.** One test is deselected by default (`net`, it hits the
real Binance archive). The suite gates deploys, so it is kept under five seconds.

This document exists because a green suite is a claim, and this project is sold on refusing to
flatter its user. The same standard applies to its own tooling: a test that cannot fail reads as
coverage and is worse than no test, and a coverage table that lists a hole as closed is worse than
one that admits the hole. Everything below is measured, not assumed — every "killed" claim was
established by applying the fault to the source, watching a named test go red, reverting, and
verifying the file is byte-identical by sha256.

**The short version of where this suite is weak:**

1. **~5,400 lines of hypothesis library have no test that runs them.** All 100 hypotheses. Only
   their prose is checked.
2. **The served HTTP path is barely tested.** `server/app.py` is 582 lines; four tests cover
   warm-up status and seven cover rate limiting. `/decide` itself is not exercised.
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
| `waves/rules.py`, `matcher.py`, `projection.py` | 37 | 3 | 2 |
| `validation/battery.py`, `evaluate.py` | 33 | 0 | 2 |
| `validation/expr.py`, `reality_check.py`, `csv_import.py` | 31 | 2 | 1 |
| `feeds/`, `store/bars.py`, `store/hydrate.py` | 27 | 1 | 1 |
| `engine/live.py`, `labeling/barriers.py`, `backtest/run.py` | 29 | 1 | 13 |
| **Total, where it was measured** | **185** | **7** | **26** |
| `hypotheses/` (100 hypotheses, ~5,400 lines) | **0** | — | — |
| `server/app.py` (582 lines) | **not measured** | — | — |

So the honest headline is **185 of 192 observable faults closed in the modules that were measured,
and two of the largest surfaces in the repo not measured at all.** A kill rate of 96% over 40% of
the codebase is not a 96% kill rate.

"Left open" counts only faults that are observable and deliberately not chased; the reason for each
is given below. Nothing is left open because it was hard.

One accounting note, because it is the kind of thing that quietly inflates a table: closing
`provisional_window`'s boundary required a source change (`n < 1` now raises instead of dying inside
NumPy on `int(ts[-1])`), and that change made the original fault — `window(max(0, n-1))` becoming
`window(max(1, n-1))` — equivalent rather than killed, since `n >= 2` makes the two expressions
identical. It is counted once, as closed, on the strength of the two tests that pin the new
boundary; it is not also counted as an equivalent.

---

## Faults deliberately left open, and why

### `waves/` — three, all the same shape

**The cost ceiling, the noise floor and the empty-zone check are pinned only away from their exact
boundary.** Loosening `cost_r > max_cost_r` to `>=`, `stop_atr < min_stop_atr` to `<=`, or
`lo >= hi` to `lo > hi` survives. Each needs a derived float to land *exactly* on a config constant
— an arithmetic result equal to `0.20` or `0.75` to the last bit. A test would have to solve for the
input that produces exact float equality and would then be pinned to that solution rather than to
the rule. This is not the same as the `in_zone` edge, which **is** tested: there the comparison is
against a published Fibonacci level a limit order can literally rest on, so the boundary is
reachable in practice.

### `validation/` — two

**The stationary bootstrap never starts a block at the last observation.** `rng.integers(0, n-1)`
excludes the final index, so it is slightly under-sampled. The bias is real and pinning it needs a
distributional test over many seeds; a single 50-index resample covers only about 31 distinct
positions even unmutated, so any coverage assertion would be flaky by construction. A flaky test in
a deploy gate costs more than this bias does.

**Blocks are cut from the tail rather than the head.** `n % L` observations are dropped either way.
Which end is a defensible design choice, not a defect. It wants a comment in the source saying the
choice is deliberate, not a test.

### `feeds/` — one

**`--quick` computes its month cutoff from the local civil date rather than UTC.** For a two-hour
window in CEST that also crosses a month boundary, `--quick` fetches 25 months instead of 24. Ingest
is idempotent and these are whole-month keys, so no stored byte differs. The month list is
observably different, so it is not an equivalent mutant in the strict sense — but everything
downstream of it is identical, and a test for it would pin the clock rather than a property.

### `engine/` — one

**The gap counter's window length is unobservable here.** `gaps_in_window` is measured over 50
trigger bars; changing it to 500 produces no difference at all on any fixture in this repo, because
`make_bars` produces unbroken series and there is no gap anywhere in the ring to count. It is only
reachable on a feed with real gaps *older* than the last 50 trigger bars. Genuinely unguarded,
genuinely not mutation-testable here; a synthetic test whose only content is the number 500 would
assert the constant against itself.

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

### Dead configuration

- `RETRACEMENTS` in `rules.py` has exactly one reference in the whole repo: its own definition.
  `projection.py` open-codes `0.500, 0.786` instead. The *live* ratios are pinned by
  `test_a_wave_two_entry_is_measured_from_the_end_of_wave_two`; the orphan copy is not, and should
  be deleted or wired up rather than tested.
- `max_starts` in `matcher.py` guards a branch that is unreachable under any shipped config: with
  `max_from_edge = 3`, `start >= n - 3 - k`, which never falls below `n - max_starts - k` for
  `max_starts >= 3`. It is an operator-facing knob that does nothing. A test here would test the
  knob's inertness.

---

## Properties nothing verifies at all

These are not mutants that survived. They are behaviours with no test of any kind.

### 1. The hypothesis library — the single largest hole

`src/wavelab/hypotheses/` is ~5,400 lines across eight files defining about 100 hypotheses. **No
test calls a single one of their `signals()` methods.** `test_hypothesis_i18n.py` checks that their
Spanish and Catalan prose has not drifted from the English, which is worth having and is a test
about translations, not about signals. `test_assay.py` builds its own synthetic `Hypothesis` objects
to exercise the validation machinery.

So: the machinery that grades hypotheses is heavily tested, and the hypotheses themselves are not
tested at all. A hypothesis whose `signals()` returns the wrong sign, reads a feature it should not
have, or throws on an empty series would be caught only by `evaluate_all`'s bare `except`, which
prints a line and moves on.

### 2. The served path

`server/app.py` is 582 lines. `test_server_status.py` (4 tests) covers warm-up status transitions
and `test_limits.py` (7 tests) covers rate limiting. `test_server_i18n.py` (9 tests) checks that
every emitted string is translated. **`/decide` itself has no test.**

Two consequences are known and open:

- **`emitting` has no production caller.** `/decide` calls `APP.engine.decide(tf, price)`
  unconditionally in any mode, and `decide()` returns a plain dict, so `Decision`'s constructor —
  the documented last line of defence, which refuses an actionable decision while catching up, while
  stale, at PRIOR maturity or without a plan — never runs on the served path. Those refusals are
  well tested as unit properties (`TestAnActionableDecisionHasToBeEarned`) and are not reached by
  the code that answers HTTP requests.
- **`LIVE` is never demoted to `CATCH_UP` by drift.** `update_health` promotes WARMUP → LIVE and
  demotes CATCH_UP → LIVE; there is no LIVE → CATCH_UP edge. The only way in is `check_clock`, which
  compares *consecutive calls*. Measured: warm to LIVE, then advance the clock one second at a time
  for an hour, and the engine ends four trigger bars stale with the badge still reading LIVE and
  `emitting` still True. That is a stalled websocket that never disconnects, which is more common
  than a suspended laptop. The two CATCH_UP tests in the suite pin the thresholds (two bars of
  silence, and the exact lag at which catching-up ends) and neither reaches this: one moves the
  clock in a single three-bar step, which `check_clock` does see, and the other enters CATCH_UP by
  assigning the mode directly.

### 3. Modules with no tests

| Module | Lines | What it does |
|---|---|---|
| `core/bus.py` | 108 | the internal event bus |
| `feeds/binance_derivs.py` | 123 | funding and open interest |
| `store/trials.py` | 78 | the trials ledger — where parameter changes are meant to be recorded |
| `collect.py` | 66 | the collection entry point |
| `registry.py` | 59 | provider registration |

`store/trials.py` is worth singling out. The README's argument against parameter search rests on
recording every experiment so the effective N cannot lie. The ledger that records them is untested.

### 4. Cross-cutting

- **Nothing tests the web front end.** `web/` is not covered by this suite at all beyond the i18n
  key checks.
- **Nothing tests behaviour under a real feed.** Every websocket, REST and archive test drives a
  stub transport. The one test against real data is marked `net` and excluded from the gate.
- **`provisional_window` has no caller in `src/`.** Its boundary is now tested because a chart
  endpoint asking for "the current candle" would hit it first, but nothing exercises it in anger.

---

## Things that look like coverage and are not

Recorded so nobody re-derives them.

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
- **Bars before a user's first imported signal are flat**, and a NumPy `RuntimeWarning` is a test
  failure suite-wide (`filterwarnings` in `pyproject.toml`) because NaN→int8 is undefined and
  silently produces a position 128× anything anyone asked for.

---

## Reproducing any claim in this document

```sh
export PATH="/opt/homebrew/bin:$PATH"
uv run pytest -m "not net"      # 582 passed, ~4.1 s
uv run ruff check .             # clean
```

To check a single fault: edit the source, run the suite, confirm the named test fails, revert, and
verify with `shasum -a 256` that the file is byte-identical. Purge `__pycache__` between runs, or set
`PYTHONDONTWRITEBYTECODE=1` — several of these faults are same-byte-length edits and a stale `.pyc`
will keep executing the mutant after the revert, which produces a confidently wrong verdict.
