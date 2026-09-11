# What the test suite protects, and what it does not

**996 tests, ~6.5 s, `pytest -m "not net"`.** One test is deselected by that flag (`net`, it hits
the real Binance archive). The suite gates deploys and it is no longer under five seconds.
Measured 2026-09-11 against commit `01ab1fd`, on the machine, with `PYTHONDONTWRITEBYTECODE=1`.

**The trap this paragraph used to describe is gone, and saying so is the point of the date above.**
It read: the flag is not a default, `pyproject.toml` sets `addopts = "-q --strict-markers"` and
nothing else, so a bare `uv run pytest` — which is what `make test` runs — selects the `net` test
and goes to the internet. It no longer does. `addopts` is now `-q --strict-markers -m 'not net'`,
with a comment in `pyproject.toml` giving the same reasoning this document gave, so a bare run,
`make test` and CI all execute the same 996 tests and none of them opens a socket. `-m net` (or
`make test-net`) runs that one test on its own, in about 5 s, and needs the internet. Verified by
running all three. The stale version of this paragraph had survived at least one commit.

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
4. **The R:R quoted on a signal is no longer computed at a different entry price from the one the
   trade is booked at.** That was this document's headline inconsistency for two rounds; it has
   been resolved at the fill and the record of what was wrong is under "Inconsistencies that have
   been resolved". **The sentence that used to follow this one was wrong and is corrected below:**
   it said the second ratio, `rr_in_zone`, "still divides by an absolute distance". It does not —
   `risk_zone = s * (entry_zone - stop)` is signed, and a midpoint on the dead side of its stop is
   quoted 0.0R rather than a flattering number. What is actually left is narrower again: the sign
   test is applied at the zone's MIDPOINT, so a zone that straddles its own stop still quotes a
   ratio for the half of it that is already a loss. See "Known inconsistencies" below, re-measured
   2026-09-11.

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

**Every row carries the date it was last measured on, and two of them are from today.** The table
has now been stale once — it read "Left open: 0" for two rows that between them held twenty-two
live survivors — and the fix for that is a date, not a promise. A row with no date is a row whose
number has not been reproduced since it was written, and it should be read as history rather than
as a claim about the code as it stands. **Do not add the columns together.** A mixed total that
looks uniform is exactly how this document went wrong before; the two halves are totalled
separately below.

### Re-measured 2026-09-11 against commit `01ab1fd`

Two independent campaigns, one per module group, each running the full suite against one fault at
a time and verifying the source byte-identical by sha256 after every revert.

| Module | Faults injected | Closed | Left open | Equivalent — no test possible |
|---|---|---|---|---|
| `waves/projection.py`, `waves/rules.py` | 78 | 73 | 0 | 4 (+1, see below) |
| `engine/live.py`, `backtest/run.py` | 52 | 49 | 0 | 3 |
| **Total, re-measured** | **130** | **122** | **0** | **7 (+1)** |

Both rows read "Left open: 0" only because twenty-two holes were closed in this pass. As the
campaigns found them the numbers were **67 of 78** and **33 of 52** — six genuine holes in
`projection.py` and sixteen in `decide()` and `run_backtest`. Each of the twenty-two is listed
under "Closed 2026-09-11" below with the fault, the test that now kills it, and what it changed
while nothing noticed. Every one was re-applied after its test was written, watched go red on that
named test, reverted, and the file confirmed byte-identical.

The `(+1)` is one mutant that is neither closed nor equivalent and is deliberately not chased: the
noise-floor refusal hardcodes `1.0` as its size factor, and replacing that with `size` is
observable only when `cfg.max_stop_atr < stop_atr < cfg.min_stop_atr`, which needs a `PlanConfig`
whose floor is above its own ceiling. A test for it would be a test of a self-contradictory
configuration. The real finding is a missing invariant — `PlanConfig` does not validate the two
against each other — and it is recorded as a config-validation gap, not as a test hole.

`waves/rules.py` came through the re-measurement untouched: 24 of 24 realistic faults died,
including every hard-rule boundary (`>` loosened to `>=` on R1, R2b and R3 each dies to a named,
direction-parametrised test), the truncation boundary, the invalidation vertex in both directions,
the evaluable/non-evaluable filter, and both Fibonacci helpers. Every hole was in `projection.py`
and in what `decide()` publishes.

### NOT re-measured this round — carried from the previous pass

These numbers were produced the same way, but the last time they were reproduced was a previous
round whose date this document did not record. That omission is the reason for the column above.

| Module | Faults closed | Left open | Equivalent | Last reproduced |
|---|---|---|---|---|
| `core/` (causality, ring, types, timeframes, clock) | 18 | 0 | 4 | previous round, undated |
| `waves/pivots.py`, `waves/store.py` | 10 | 0 | 3 | previous round, undated |
| `validation/battery.py`, `evaluate.py` | 33 | 0 | 2 | previous round, undated |
| `validation/expr.py`, `reality_check.py`, `csv_import.py` | 32 | 1 | 1 | previous round, undated |
| `feeds/`, `store/bars.py`, `store/hydrate.py` | 28 | 0 | 1 | previous round, undated |
| the served decision path (`/api/decide`, the PUBLIC gate, `LiveEngine.decide`'s wire keys) | 10 | 0 | — | previous round, undated |
| `web/app.js`, decision panel only (the keys it reads off the card) | 4 | — | — | previous round, undated |
| `hypotheses/` (100 hypotheses, ~5,400 lines) — six properties, not their arithmetic | 11 | 2 | — | previous round, undated |
| the catalogue's titles (`Hypothesis.title`, `/api/hypotheses`, `web/assay.js`, hyp.{es,ca}.json) | 8 | 0 | — | previous round, undated |
| **Total, carried forward** | **154** | **3** | **11** | |
| `waves/matcher.py` | **share not recorded** | ? | ? | previous round, undated |
| `labeling/barriers.py` | **share not recorded** | ? | ? | previous round, undated |
| `server/app.py`, the three validation routes | **never measured** | — | — | — |

The last two "share not recorded" rows are a debt this round created and cannot pay. The previous
table bundled `waves/rules.py, matcher.py, projection.py` into one row of 41 and
`engine/live.py, labeling/barriers.py, backtest/run.py` into one row of 30, with no per-file
breakdown written down anywhere. This round re-measured four of those six files and produced fresh
numbers for them; there is no way to say how many of the old 41 belonged to `matcher.py` or how
many of the old 30 to `labeling/barriers.py`, so their counts are not carried forward as though
they were current and are not silently folded into a total either. Whoever measures those two files
next should give them their own rows. The lesson generalises: **bundle files into a row and you
lose the ability to re-measure any of them separately.**

So the honest headline is **122 of 122 observable faults closed in the two module groups measured
today**, and **154 of 157 in the nine rows whose last measurement predates today**. Read the
second number as history. The `hypotheses/` eleven in particular are eleven faults against six
PROPERTIES held over all 100 hypotheses, not eleven faults sampled across 5,400 lines of
arithmetic; most of that arithmetic is still unmeasured.

One denominator caveat that applies to today's two rows and would apply to any future one. Both
campaigns ran in worktrees where `data/bars` was empty, so **107 of their tests skipped** — the
real-candle arm of `tests/test_hypothesis_library.py` — and their kill rates are against 876
executing tests rather than 983. Every one of the twenty-two holes was then re-confirmed in the
real repository, where that store is present and all 983 tests execute: each fault was re-applied
here and survived the complete suite before its test was written. So the survivors are not an
artefact of the skip. A skipped arm is still a smaller net, and a campaign run that way should say
which denominator it is quoting.

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

## Closed 2026-09-11 — the twenty-two the table had counted as closed already

Every one of these was a live survivor against the full 983-test suite at commit `01ab1fd` while
the table above said "Left open: 0" for the rows they sit in. They are grouped by what they have
in common, because the grouping is the finding: **the holes were not where the last change was, they
were in the code the change made load-bearing, and in everything the change's one consumer does not
read.**

### `waves/projection.py` — the half of an if/else that had no fixture

The wave-4 entry zone is truncated so it can never invade wave 1's territory. The LONG branch is
`lo = max(lo, P[1] * 1.0005)` and the SHORT branch is `hi = min(hi, P[1] * 0.9995)`. Every mutation
of the long line died; **every mutation of the short line survived the entire suite** — the nudge
flipped, `min` turned into `max`, the truncation deleted outright.

The cause was one missing row. Three parametrised suites carried a `w4_long_tight_zone` case and no
short counterpart, and the generic `w4_short` case has a zone the truncation never binds on, so it
ran the `else:` branch without constraining it. The fix is the mirror fixture —
`("w4_short_tight_zone", (300.0, 190.0, 250.0, 140.0, 185.0), Direction.SHORT, 186.0, 3.0)`, the
long one reflected about 200 — plus a short twin of the truncation test. On that fixture:

| | `entry_hi` | `rr_in_zone` |
|---|---|---|
| shipped | 189.905 | 20.85 |
| nudge flipped | 190.095 | 21.30 |
| truncation removed | 195.0 | 45.71 |

The count dies at 190 and the stop sits at 190.75, so the middle row publishes a short being invited
to sell below the price at which its own count is already dead — the exact defect the long-side test
asserts against. The bottom row publishes a zone five points past the invalidation and more than
doubles the ratio quoted for it. Three tests now go red on it:
`test_the_entry_is_never_on_the_wrong_side_of_its_own_invalidation[w4_short_tight_zone]`,
`test_inside_the_zone_the_two_numbers_agree_to_the_width_of_the_zone[w4_short_tight_zone]` and
`test_a_wave_four_zone_is_truncated_against_the_end_of_wave_one_on_a_SHORT_too`.

**The general rule, and it is cheap to apply:** when a guard is written as an if/else on direction,
a fixture for one branch is not a fixture for the other. Grep the parametrisation for the mirror.

### `waves/projection.py` — three more, one to each of the three shapes this document keeps meeting

- **A gate untested at its own boundary.** `stop_atr > cfg.max_stop_atr` loosened to `>=` survived.
  `TestTheGatesDecideAtTheirOwnBoundary` exists for exactly this and covered the other two gates;
  the size-cut ceiling was missing from it, and `test_a_far_stop_costs_size_and_only_size` samples
  100 / 900 / 2000 ATR, none of which lands on the limit. What the loosened gate emits is
  self-contradicting on its face — "stop at 3.0 ATR (>3.0): size cut to 100%" — and nothing read
  the string. Closed by `test_a_stop_landing_exactly_on_the_size_cut_ceiling_is_not_cut`, which
  derives the threshold from the code at runtime the way its two siblings do: run the plan, read
  the `stop_atr` it computed, hand the scenario back under a config whose ceiling is that float.
- **A refusal that quotes an unasserted constant.** The size-cut reason interpolates
  `cfg.max_stop_atr`; substituting `cfg.min_stop_atr` survived, because the only assertion anywhere
  on that string was `any("size cut" in x for x in r.reasons)`. The card would then tell the user
  the limit it tripped is 0.75 ATR when it is 3.0. This module's whole argument is that a refusal
  with numbers is an argument the user can push back on, so a wrong number in one is worse than no
  number. Closed by extending the existing assertion to `any(f"(>{cfg.max_stop_atr})" in x ...)`.
- **A reachable input nobody had tried: `atr == 0.0`.** `stop_atr = risk / atr if atr else
  float("inf")` — the guard is truthiness, not `is None`. Changing `inf` to `0.0` flips the answer
  completely and the suite stayed green either way. The shipped answer is pinned rather than fixed,
  and it is an inconsistency in its own right; see "A market with no range" under "Known
  inconsistencies" below.

### `waves/projection.py` — one that was "killed" and should have comforted nobody

`rr_in_zone = ... if risk_zone > 0 else 0.0` loosened to `>= 0` IS killed by the suite as it
stands — by three ERRORs in `tests/test_server_i18n.py`, whose fixture happens to build a plan
whose zone midpoint sits exactly on its stop and takes a `ZeroDivisionError` through an unrelated
assertion. The class written for this guard, `TestTheZoneFigureIsGuardedLikeTheHeadline`, does not
kill it: it kills the two sign faults and not the boundary. Red for the wrong reason, in the
document's own category, and fragile in exactly the way it warns about — tidy that i18n fixture and
the boundary has no coverage at all, while the defect it guards takes the served route down with a
division by zero. Closed by `test_a_zone_midpoint_sitting_exactly_on_the_stop_quotes_no_reward`,
whose geometry is exact in binary rather than solved for.

### `engine/live.py` and `backtest/run.py` — sixteen, and one structural reason for all of them

**Every test that guarded the published card guarded it through the backtest, and the backtest
reads seven of the card's keys.** It decides on `viable`, `in_zone`, `score`, `archetype` and
`direction`, and it books `stop`, `targets[1]` and `rr_t2`. Everything else — the entry zone, the
third target, `rr_in_zone`, `cost_r`'s fourth decimal, `size_factor`, `invalidation_price`, the
card's own `price`, the verdict's reason — it either copies without checking or never touches.
`test_a_signal_books_the_trade_the_card_actually_offered` compares the signal against the card, so
anything the two get wrong TOGETHER is invisible to it by construction, and
`test_the_booked_R_R_is_the_R_R_the_card_quoted` was the only test in the suite checking a
published figure against arithmetic computed outside the engine — for exactly one figure.

The shape of the result says the same thing: all five faults aimed at what `run_backtest` books
died, usually to two or three named tests at once; ten of sixteen aimed at what `decide()`
publishes survived untouched.

- **Ten payload faults, one test.** `entry_lo`/`entry_hi` swapped on the wire (all 808 viable rows
  on the fixture); `targets` sliced to `[:2]`, so `drawPlan` stops drawing T3 (all 808); `targets`
  reversed — which survives *because* the backtest reads `targets[1]`, the fixed point of reversing
  a 3-tuple (all 808); `rr_in_zone` filled with `rr_t2`, which restores the old single-figure card
  and makes `app.js`'s dual-number branch dead code (all 808); and six roundings: `stop` and
  `invalidation_price` to 1dp (737 and 769 rows), `cost_r` to 2dp so a value of 0.004 becomes 0.0
  (799), `size_factor` to 1dp (408), `score` to 2dp, which also shifts what `min_score` filters
  (687), and the card's own `price` to 1dp (442 cards). Closed by
  `test_the_card_publishes_the_plan_that_was_actually_built`, which rebuilds every row through
  `match_impulses` + `build_plan` with the engine's own configs on the same pivots at the same
  instant, and asserts the published block IS that plan at the precision the card prints — equality
  against `round(...)`, not a tolerance, because the precision is the claim.
- **The verdict line.** `verdict = Verdict.WATCH if out else Verdict.NO_TRADE` re-gated on
  `best_in_zone` flips **311 of 480 cards** from WATCH to NO_TRADE, and the suite stayed green.
  That is precisely the edit the rewritten line exists to forbid — its own comment says
  `best_in_zone` "does NOT move the verdict". The two tests that look as if they would see it do
  not: one checks only the ceiling, the other only the empty card. The band between them — structure
  present, price outside every zone, 65% of all cards, and the whole point of a WATCH — was
  unguarded in both directions. Closed by
  `test_the_verdict_does_not_depend_on_anything_being_in_a_zone`.
- **The reason line, which is what `best_in_zone` actually decides.** Inverting `elif not
  best_in_zone:` puts "there is structure, but the price is not inside any entry zone" on **446 of
  480 cards** where it is false; computing `best_in_zone` from `r.viable` instead of `r.in_zone`
  deletes it from **302** that should carry it. Both survived. Closed by
  `test_the_zone_reason_is_on_exactly_the_cards_with_nothing_in_a_zone`, asserted as an if-and-only-if
  because each direction alone is satisfied by a constant.
- **The cooldown, guarded in one direction only.** `test_one_count_does_not_become_twenty_trades`
  asserts `apart >= COOLDOWN_BARS`, which a TIGHTER cooldown satisfies a fortiori. So it holds the
  direction that causes pseudo-replication — and it holds it well; a loosened cooldown and a
  re-keyed one both die to it and to nothing else — while being blind to the direction that
  quietly discards half the sample. `< cooldown_bars` tightened to `<=` deletes roughly half the
  signals, because **29 of the fixture's 59 same-archetype gaps sit at exactly the cooldown**, and
  a backtest that silently drops 48% of its signals reports a smaller `n` around a number nobody
  asked for. Closed by adding `min(gaps) == COOLDOWN_BARS` to that test: the boundary is inclusive.
- **`min_score` is an inclusive floor and nothing said so.** `h["score"] < min_score` tightened to
  `<=` survived: the only test of the filter runs at 1.01, above anything a weighted mean of
  guideline fits can reach, where the two comparisons agree, and no hypothesis on the fixture
  scores exactly 0.0. Closed by `test_the_score_floor_is_a_floor_and_not_the_first_score_refused`,
  which reads the lowest score off a completed run and re-runs with `min_score` set to exactly it.
- **The end-of-history guard, which no fixture had ever reached.** `if i is None or i + 1 >= len(H)`
  shifted to `>` makes the condition unsatisfiable — `i <= len(H) - 1` always — so the guard is
  gone, a signal on the final trigger bar is resolved against empty arrays, and
  `resolve_triple_barrier` returns a zero-bar flat trade that counts in `n`, in `hit_rate` and in
  `expectancy_r`. It survived only because no signal lands on the last bar of this fixture, which
  is a fact about the seed; in the use that matters most — a backtest run to the present minute —
  a signal on the last bar is the common case. Closed by
  `test_a_signal_on_the_last_bar_of_history_is_left_unresolved`, which cuts the bar stream off at
  the instant the trigger bar carrying the last signal closes.

### The three survivors from this round that are EQUIVALENT, and are not chased

Proven rather than asserted, and recorded so nobody re-derives them.

- **`build_plan(h, price, atr, …)` → `build_plan(h, price, det.atr, …)` in `decide()`.** `atr =
  det.atr` is bound three lines above the loop, `det.atr` is a property over state mutated only in
  `ZigZag.update()`, and nothing in the loop body reaches `det`. Constant across the loop, equal at
  every evaluation; 0 differences over 1,588 rows.
- **`w.ts < htf_open_ms + tf.ms` → `<=` in the resampler.** It looks like a one-bar lookahead and
  would be one if any source bar could carry `ts == htf_open_ms + tf.ms`. It cannot, structurally:
  `_close_htf` has one call site, invoked as `_close_htf(tf, end - tf.ms)`, and `Ring.window()`
  returns only CLOSED bars — the in-flight bar lives in a separate channel. Instrumented over all
  2,400 `_close_htf` calls on the 20-day fixture, `max(w.ts) - (htf_open_ms + tf.ms)` is exactly
  `-60000` ms every time: one full 1m period below the boundary, the maximum it can be.
- **`s * (entry - stop) <= 0` → `< 0` at the wrong-side gate.** The two differ only at
  `entry == stop`, and eleven lines earlier `risk = abs(entry - stop)` with `if risk <= 0: return`
  has already refused that. The argument covers NaN too: `abs(nan) <= 0` is False, so NaN passes
  the first gate, and `nan <= 0` and `nan < 0` are both False, so shipped and mutant still agree.
  Measured as well as argued: 56,430 legal counts swept, each priced at ten points including
  exactly the computed stop and both `math.nextafter` neighbours of it — the gate was reached
  276,099 times and the value was never 0, smallest `|s*(entry-stop)|` observed 2.12e-22.

And one that is equivalent only by coupling, which is worth its own line: `in_zone = lo <= price <=
hi` rewritten as `lo <= entry <= hi` is equivalent because `entry` has exactly one binding in
`build_plan` and it is `entry = price`. AST-verified. It is equivalent *because of* the contract
that moved in the last change, so the day "quote at the fill" moves again, `in_zone` follows it
silently with nothing red.

One more, recorded because it is not equivalent in the strict sense and a test for it would still
be noise: `build_plan(h, round(price, 2), …)`. The fixture's closes are full-precision floats, so
the rounding really does move the entry by up to half a cent — and it changes nothing anyone can
read: 0 of 1,588 published rows differ, over the whole row dict, so no gate inside `build_plan`
flipped either. The residual is real (the gates decide on the unrounded value, so a plan within
5e-3 of a gate boundary would flip) but a test that could see it would assert below the precision
the product publishes, which this suite's own standard refuses.

---

## Faults deliberately left open, and why

One entry stands here. Six more used to, closed in a pass before this one, and the reasons this
document gave for leaving them open are reproduced against each one below — because every one of those reasons was
wrong, and wrong in the same way each time: **it described the hardest test it could think of, found
that test bad, and stopped.** Four reasons covered the six faults; all four fell to an easier test
that nobody had looked for. That is the failure mode to watch for in this section. If an entry here
says "a test for this would have to X", the first question is whether it would.

### `validation/` — one

**Blocks are cut from the tail rather than the head.** `Mc[:, : nb * L]` keeps the first `nb` whole
blocks, so the final `n % L` observations — the most recent ones — are dropped. Taking
`Mc[:, n - nb * L:]` instead drops the oldest ones. Re-measured again 2026-09-11, as one of this
round's ten spot-checks: the suite is **green** under that edit, so this is still a live survivor
and this entry is still true. Which end is a defensible design choice rather
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

### Closed in the PREVIOUS pass, and what the reason for leaving them open got wrong

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

### A stop on the wrong side of the entry — refused at the fill, refused at the zone's middle, and still quoted for a zone that straddles it

This entry has been re-measured twice and has shrunk both times. It used to read "a stop on the
wrong side of the entry is planned normally"; then "the same defect survives at the published
ZONE"; it is now a straddle, and only a straddle. Both re-measurements were done by applying the
scenarios rather than reasoning about them, because the entry price moved underneath this entry
once (see "Inconsistencies that have been resolved" below) and the zone arithmetic moved underneath
it again. An entry that assumes it knows what a change did to it is how this document goes stale.

**The case it described is now refused.** `build_plan` still computes `risk = abs(entry - stop)`,
but a sign test now stands immediately under it: a long whose market is below its stop, or a short
whose market is above it, is turned down with "price $X is already past the stop $Y" instead of
being sized off a distance whose sign was thrown away. Applied: the old entry's own scenario — a
stop five dollars above a long's entry, built by hand-setting the invalidation — comes back
`viable=False`, `stop_atr=0`, `rr_t2=0`, refused by name. Both directions are pinned by
`TestAnEntryPastItsOwnStopIsRefusedRatherThanSized`, and the sibling assertion in
`test_a_stop_that_lands_on_the_entry_is_refused_and_does_not_take_the_engine_down` holds that the
two gates do not overlap: five dollars on the wrong side is a live, positive risk and still a no.

**The same defect was said to survive at the published ZONE, and two thirds of that is no longer
true.** Re-measured 2026-09-11 by applying the entry's own scenarios rather than re-reading it.
The sign test is NOT confined to the headline: `risk_zone = s * (entry_zone - stop)` is signed as
well, and `rr_in_zone` is `0.0` — "no reward", which is the honest answer for a geometry that makes
no sense — whenever the zone midpoint is on the dead side. The same three hand-built plans, a long
on (100, 110, 105) with ATR 1, run again today against the numbers this entry used to carry:

| price | stop | published zone | verdict | headline `rr_t2` | `rr_in_zone` then | now |
|---|---|---|---|---|---|---|
| 108.00 | 105.50 | 102.14 – 105.00 | viable, size 100% | 5.27 | 9.12 | **0.00** |
| 112.00 | 105.50 | 102.14 – 105.00 | viable, size 46% | 1.41 | 9.12 | **0.00** |
| 108.00 | 103.00 | 102.14 – 105.00 | viable, size 60% | 2.64 | 30.90 | **30.89** |

**What is left is the third row, and it is a different defect from the one described.** That zone
STRADDLES its own stop: 102.14–103.00 is dead and 103.00–105.00 is live. The sign test is applied
at the MIDPOINT, 103.57, which is on the live side, so the plan passes and the card quotes 30.89R
for a zone half of which can only fill into an instant loss. Narrower than the old entry, and the
same kind of thing: a guard that holds at one point of an interval and is quoted for the interval.

**And it is pinned, which the paragraph that used to close this entry denied.**
`TestTheZoneFigureIsGuardedLikeTheHeadline` holds the dead-side case in both directions, and
`test_a_zone_midpoint_sitting_exactly_on_the_stop_quotes_no_reward` — added 2026-09-11 — holds the
boundary, where the mutant divides by zero and takes the served route down. Before that second
test, the `risk_zone > 0` boundary was killed only by three ERRORs in an unrelated i18n fixture:
coverage that would have evaporated the moment someone tidied it. What is NOT pinned is the
straddle, and nothing in the suite reaches it.

**How reachable it is, measured rather than assumed.** Not through `match_impulses`. Over six months
of 4h bars, 1,532 viable plans: the edge of the zone nearest the stop is never closer than 0.058R
above it (1st percentile 0.096R, median 0.314R) and not one plan reaches or crosses. That is
structural — the w2 zone is a retracement of a leg that ends above P0, and the w4 zone is truncated
against P1 — which is the same argument that makes the fill-side gate above a NEW refusal rather
than a newly noticed one. It is also the argument that the w4 truncation has to be right on BOTH
sides for it to hold at all, and until 2026-09-11 the SHORT half of that truncation was guarded by
nothing whatsoever (see "Closed 2026-09-11" above). `build_plan` is exported and takes any
`Hypothesis`, so the reachability is the same one the original entry rested on: a hand-set
invalidation.

Closing what remains means deciding what the card does with a zone that straddles its stop: truncate
the zone at the stop, drop the second ratio, or refuse the plan outright. That is the same kind of
product decision as the one that moved the arithmetic to the fill, so it is recorded here rather
than settled by a test. **The lesson this entry has now taught twice is the one to carry forward:
it was stale in the direction that flatters the document, both times, and both times the way it was
found was by executing the scenario it describes rather than re-reading the prose.**

### A market with no range produces a viable plan sized at 0% — RESOLVED, refused by name

Found 2026-09-11, pinned the same day, resolved the same day. Kept here in full, because the
value of this entry is the record of what shipped and for how long, not the headline that it
is now fixed. What follows describes the behaviour as it WAS.

`build_plan` computes `stop_atr = risk / atr if atr else float("inf")`. The guard is truthiness,
not `is None`, so a genuine `0.0` takes the `inf` branch — and every gate downstream then lets it
through. An infinite `stop_atr` is above `max_stop_atr`, so the size is cut by `3.0 / inf = 0.0`;
it is not below `min_stop_atr`, so the noise floor does not refuse it; `cost_r` does not involve
the ATR, so the fee gate does not either. What comes back is a complete card — entry zone, stop,
three targets, a required hit rate — marked `viable=True`, instructing the reader to take 0% of a
position. A "yes" that cannot be acted on, from a tool whose argument is that its refusals come
with arithmetic.

| | `viable` | `stop_atr` | `size_factor` | entry zone |
|---|---|---|---|---|
| shipped (`float("inf")`) | True | inf | 0.0 | 72,140 – 75,000 |
| the one-word edit (`0.0`) | False | 0.0 | 1.0 | none — refused by the noise floor |

Both are defensible and neither was asserted, so the suite was green either way. The pin existed so
that whoever decided would land on a red test and have to say which way they went.

**The decision: refused by name.** Every number in a plan is measured in ATRs, so with no ATR
there is nothing to measure a stop against and no plan to make — and a card that says yes at 0%
of a position is a yes nobody can act on, which is the one thing this tool is not allowed to
produce. `build_plan` now returns a refusal whose reason names the flat market, rather than a
plan whose reason blames the size cut. `TestAMarketWithNoRangeIsRefusedByName` asserts it and
carries the history; deleting the guard turns it red.

`LiveEngine.decide` still guards `det.atr is None` and not `== 0.0`, which is now harmless —
the zero reaches `build_plan` and is refused there, with a reason the reader sees — but it is
worth knowing that the served route is not where this is caught.

**How reachable, stated precisely rather than generously.** `WilderATR` returns a real `0.0` over a
series with no range — measured: 40 flat bars and `ready` is True with `value` exactly `0.0`, not
`None` — and `LiveEngine.decide` guards `det.atr is None` and not `det.atr == 0.0`, so the served
route has no gate here. What is NOT demonstrated is a card carrying both: hypotheses need pivots and
pivots need movement, so a wholly flat series produces no structure to plan, and Wilder smoothing
decays toward zero rather than reaching it. The honest statement is that `build_plan` is exported,
takes the ATR it is handed, and is the layer where this decides — which is the layer it is pinned
at. A halted symbol, a padded series or a stub feed is the shape of input to watch for.

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

---

## Inconsistencies that have been resolved, and which way they went

Nothing is deleted from this document when it gets fixed. An entry that quietly becomes "fixed",
or disappears, teaches nothing and leaves the next reader with no way to tell a hole that was
closed from a hole nobody ever found. So a resolved entry keeps the measurement of what was wrong,
gains the decision that was taken, and moves here.

### The R:R on a signal described a different entry from the one it was booked at — resolved at the fill

**What was wrong, kept as the record.** `build_plan` computed `rr_t2`, `cost_r`, `p_required`,
`stop_atr` and `size_factor` from `entry = (zone_lo + zone_hi) / 2` — the middle of the published
entry zone — while `run_backtest` fills at `htf.close`. Measured on the 20-day fixture: **131 of 132
signals carried an R:R describing a different entry price from the one their outcome was resolved
at**, the worst by 2.29R (recorded 4.03 against a geometric 6.32).

Measured again over six months of 4h bars — BTCUSDT 2026-04-01 to 2026-09-11 out of `data/bars`,
replayed through `LiveEngine.on_bar_1m` and re-planned with `match_impulses` + `build_plan` on each
4h close, which is the only harness in this document that has been reproduced twice:

| counting | plans | overstated by the zone figure | median | p90 | worst |
|---|---|---|---|---|---|
| every w2/w4 hypothesis per trigger bar | 3,253 (1,532 viable) | **1,357** | +2.54R | +3.61R | +9.61R |
| top-ranked hypothesis only | 930 | **513** | +2.40R | +3.33R | +4.09R |

**The often-quoted "649 of 925 plans, median +1.65R" is not one of them, and nobody has reproduced
it.** It is the figure carried by `projection.py`'s comment, `app.js`'s comment and several test
docstrings, so it is left standing there rather than replaced by a third set; but the denominator
closest to it is 930, its overstated count is 513 rather than 649, and every attempt to re-measure
it lands nearer +2.4R than +1.65R. The shape of the finding is not in doubt — four-fifths of plans
flattered, by more than 2R at the median. The exact triple is unsourced and should be treated that
way until someone produces the harness that made it.

The error had a direction, and that is the part that made it a product defect rather than an
accounting one. The zone is a retracement you are *waiting* for, so the zone price is the better
price; quoting it as the headline meant the card was systematically flattering its reader, in a
tool whose entire argument is that it does not. `expectancy_r` was measured off the fill and `rr`
was quoted off the zone, and the two travelled side by side in the same report.

**Which way it went: to the fill.** `rr_t2`, `cost_r`, `p_required`, `stop_atr` and `size_factor`
are now computed at `price`, the close of the bar that produced the plan and the same price the
backtest books at. The reason that side won is reachability, not conservatism for its own sake: a
limit order resting in the zone MAY NEVER FILL, so quoting it assumes a price you might not get,
while the close assumes only that you can buy at market, which is always true. The three gates
(`max_cost_r`, `min_stop_atr`, `max_stop_atr`) decide on that conservative number too — a gate that
accepted on the optimistic figure and reported the conservative one would be the same defect with
an extra step.

**The zone figure was not thrown away.** `rr_in_zone` is new on `PlanResult` and travels beside the
headline as the better case, on the wire out of `decide()` and on the card as a condition rather
than as a second bare figure ("5.27 now — 9.12 if your limit fills in the zone").
`TestTheHeadlineCanNeverBeTheFlatteringNumber` is the property that holds the whole change down:
from the FAR side of the zone the headline may never reach the zone figure. Far side, not
"outside" — see the two corrections below, both of which this entry originally got wrong.

**Correction 1: the two numbers do NOT agree inside the zone, and the card no longer pretends they
do.** This entry, `projection.py`'s comment and `app.js` all said that inside the zone the two
describe the same trade and agree to within rounding, and `app.js` dropped the second figure there
on the strength of it. Measured over six months of 4h bars: of 273 viable in-zone plans, **99.6%
disagree by more than the two decimals the card prints**, by a median **0.78R** — 19.9% of the zone
figure — p90 1.58R and up to **2.95R**. They are the same ratio read at two entry prices, so the gap
is `(rr_in_zone + 1) * |price − mid| / risk`: zero only at the midpoint and widening the whole way
to either rim, which is exactly what
`test_inside_the_zone_the_two_numbers_agree_to_the_width_of_the_zone` asserts as a closed form —
this document was contradicting the test it cites. The card now drops the second figure only when
the two round to the same printed number.

**Correction 2: the headline is not always the smaller of the two.** Below its entry zone a long is
buying CHEAPER than the midpoint `rr_in_zone` is quoted at, so the fill is the better trade and the
headline is legitimately the larger figure — **179 of 1,532 viable plans** over the same window, and
50 of those with the price fully outside the zone rather than merely below its middle. That is not
the card flattering anyone; it is reachable at market, which is the whole test. The invariant is
"the headline is the ratio at a price the reader can actually get", not "the headline is the lower
number", and the property test is named for the far side only because that is the side where the
claim is an inequality.

**What it cost, measured.** The two ratios diverge in proportion to how far the price is from the
middle of the zone, inside it as well as outside — see Correction 1 above; on the far side, where
the old arithmetic flattered, the divergence is exactly the honest answer "buying this now is a
worse trade than the card is describing", so there it is the point rather than the price. What it
did cost is signals. On the 20-day fixture the same seed now produces **61 signals
where it produced 132**: the gates decide on the conservative number, and 106 of that fixture's 964
plans are refused outright by the new gate below.

**One justification in the source did not survive being checked, and it is recorded rather than
repeated.** `projection.py`'s comment argues the change is cheap because "the verdict only reaches
its ceiling when the price is already INSIDE the zone". It does not. `best_in_zone` changes the
reasons line and nothing else. Measured over the 20-day fixture: 446 WATCH verdicts, **311 of them
with no hypothesis in its entry zone at all**, and 34 NO_TRADE, every one of those for having no
hypothesis rather than for being out of the zone. Those three numbers reproduce exactly on
re-measurement (2026-09-11: 480 cards, 34 empty, 311 with structure and nothing in a zone, 135 with
structure in a zone). The change is still cheap for the reason above — the numbers agree where the
price is in the zone — but it is cheap on the CARD, not on the verdict.

**Two corrections to that paragraph, both dated 2026-09-11.** It quoted `decide()` as computing
`WATCH if (out and best_in_zone) else (WATCH if out else NO_TRADE)` and called the dead branch "a
separate thing somebody should look at". Both have moved. The source now reads
`verdict = Verdict.WATCH if out else Verdict.NO_TRADE`, with a comment saying `best_in_zone` does
not move the verdict and that the line should be rewritten when maturity rises rather than left
looking as though it already had; the ternary this document quotes has not existed for a commit,
which is the fourth time this file has been caught describing code it no longer describes. And the
closing clause — "nothing pins what `best_in_zone` is supposed to do underneath it" — was correct
and is no longer: re-gating the verdict on `best_in_zone` flipped 311 of 480 cards with the suite
green, and inverting or re-keying the reason line moved 446 and 302. All three are now held by
`test_the_verdict_does_not_depend_on_anything_being_in_a_zone` and
`test_the_zone_reason_is_on_exactly_the_cards_with_nothing_in_a_zone`.
`test_catching_up_changes_nothing_on_the_card` still pins the PRIOR ceiling at WATCH.

**A refusal that came with it, and could not have existed before.** An entry already PAST its own
stop — a long whose market is below where it would give up — was reported by `abs()` as ordinary
positive risk, so the plan came out looking normal while describing a purchase that loses on the
first tick. **71 of 925 plans land there.** It was unreachable while the card quoted the zone,
because the zone midpoint is on the correct side of the stop by construction; the fill has no such
guarantee, because it is wherever the market is. What was said to be left of that same `abs()` —
at the zone, in the divisor of `rr_in_zone` — was itself wrong: that divisor is signed too. What is
actually left is a zone that straddles its stop, and it is the first entry under "Known
inconsistencies" above, re-measured.

**Verified afterwards, on the fixture the original measurement was taken on.** 0 of 61 resolved
signals now disagree; the largest residual anywhere is 0.0049R, which is the two-decimal rounding
the card applies to the numbers the signal copies off it.

**The characterisation test did exactly what it was written to do.**
`test_the_booked_R_R_is_the_zone_s_and_not_the_fill_s_and_that_is_recorded_here` asserted the
discrepancy so that whoever resolved it would see a red test and have to say which way they went.
It went red on this change, and its own failure message said what to do next: delete it and move
the entry here. Both halves happened. It is succeeded by
`test_the_booked_R_R_is_the_R_R_the_card_quoted`, which is the same claim with the sign turned
round and on the same yardstick — the characterisation called a drift above 0.01R a disagreement
and demanded that most signals show one; the successor demands that none do. That is the only
evidence in this repo that the "pin it rather than fix it" idiom pays for itself, so it is written
down: a pinned inconsistency is not a resigned one, it is a tripwire with a note attached.

**Re-measured 2026-09-11, and the debt this paragraph recorded is paid.** It used to read: the
`waves/rules.py, matcher.py, projection.py` row has not been re-run against the two new gates, so
the observable total in that table is a count from before this change — do not read the total as
covering them. That mutation pass has now been run, on `projection.py` and `rules.py` (78 faults)
and on `engine/live.py` and `backtest/run.py` (52). What it found is worth reading beside the
paragraph that predicted where to look, because the prediction was exactly wrong in an instructive
way. It named "the entry the arithmetic is computed at, and the two gates that now decide on it" as
the first places to re-run. Both turned out to be **well** covered: `entry = price` and everything
downstream of it scored 10 of 10, and the two new refusal gates 7 of 8 with the one survivor proven
equivalent. Every hole was somewhere else — in the wave-4 truncation the new gate made
load-bearing, in the size-cut arithmetic, and in the sixteen published figures nothing downstream
reads. **A change's own author is the worst-placed person to say where its risk landed**, and the
only reliable answer was to inject faults everywhere and look at which ones lived.

### Dead configuration — both entries resolved, and they went opposite ways

Recorded rather than deleted, because "wire it up" and "delete it" were the right answers to two
things that looked identical in a diff.

- **`RETRACEMENTS` in `rules.py` is now wired up.** It had exactly one reference in the repo — its
  own definition — while `projection.py` open-coded `0.500, 0.786` and `0.382, 0.500`. That is two
  definitions of one published level: the zone drawn on the card followed the literals and the table
  could have gone on claiming something else with nothing to notice. `build_plan` reads both bounds
  from the table now. The values were checked bit-for-bit before the change (`struct.pack(">d", …)`
  on both sides) and are unchanged, so **no ratio moved and no card changed** — this is not a
  product change. Measured both directions afterwards, and **one of the two numbers below was
  wrong and is corrected here (2026-09-11).** Moving `w2`'s upper bound from 0.786 to 0.886 turns
  exactly **one** test red, not three:
  `test_a_wave_two_entry_is_measured_from_the_end_of_wave_two`. Moving `w4`'s from 0.500 to 0.618
  likewise turns exactly one red. Both re-run against the full suite. The reason the count is lower
  than claimed is a real property of the design rather than a miscount, and it is the thing to
  carry forward: `TestTheZoneIsReadFromTheRetracementTable` MONKEYPATCHES the table, so it is
  insensitive to the shipped ratios by construction — it guards the wiring and never the values.
  The original sentence conflated the two guards, and a document that miscounts its own tests
  upwards is the same failure as a table that miscounts its own holes downwards.

  The other half reproduces exactly: re-typing the literals back into `projection.py` —
  behaviour-identical, so invisible to every other test in the suite — is caught by
  `TestTheZoneIsReadFromTheRetracementTable`, `w2` by `test_the_wave_two_zone_follows_the_w2_row`
  and `w4` by `test_the_wave_four_zone_follows_the_w4_row`, one test each and nothing else. That
  class is the only guard on the wiring itself, and it names no Fibonacci number: it sets the row
  to `(0.0, 1.0)` and requires the zone to be the whole of wave 1, then to `(0.0, 0.5)` and
  requires it to be the top half. `_fib_zone` sorts its two outputs, so reading the two bounds in
  the wrong ORDER is an equivalent mutant here, not a hole — **confirmed for the `w4` row as well
  as the `w2` one**, which this entry did not say, and confirmed exhaustively rather than argued:
  over all 20,736 combinations of `(a, b, lo_r, hi_r)` drawn from a twelve-value set spanning zero,
  denormals, the shipped ratios, negatives and 1e300, `_fib_zone(a,b,x,y) == _fib_zone(a,b,y,x)` in
  every case. The sorting itself is NOT equivalent and is well guarded: making `_fib_zone` return
  `(max, min)` turns 25 tests red.
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

**A consistency test comparing two copies of the same mistake.**
`test_a_signal_books_the_trade_the_card_actually_offered` checks the booked signal against the card
that produced it, which is the right claim and a strong one — but it means anything the card and
the signal get wrong TOGETHER is invisible to it by construction. Ten faults in `decide()`'s
payload survived behind it, each changing between 408 and 1,588 published rows, because the
backtest copies those numbers rather than checking them. A consistency test needs a third party:
`test_the_card_publishes_the_plan_that_was_actually_built` rebuilds the plan through
`match_impulses` + `build_plan` and compares the wire against that. Added 2026-09-11.

**A guard written as an if/else with a fixture for one branch.** The wave-4 zone truncation is
`if LONG: lo = max(...) else: hi = min(...)`. Three parametrised suites carried a
`w4_long_tight_zone` case and no short counterpart, so every mutation of the long line died and
every mutation of the short line survived the whole suite — including deleting it outright. The
generic `w4_short` fixture existed and made it look covered: its zone never reaches the vertex the
truncation binds against, so it ran the `else:` without constraining it. **Executing a branch is
not constraining it**, which is the entire argument of this document restated at the level of a
single fixture list. When you see a direction-parametrised guard, grep the parametrisation for the
mirror case.

**An assertion the mutant satisfies a fortiori.** `test_one_count_does_not_become_twenty_trades`
asserted `apart >= COOLDOWN_BARS` on every pair of same-archetype signals. A cooldown loosened by
one bar breaks it; a cooldown TIGHTENED by one bar satisfies it more comfortably than the shipped
code does, and quietly deletes 48% of the sample. A one-sided inequality only ever tests one side,
and for a debounce both sides are defects: too loose is pseudo-replication, too tight is a smaller
`n` around a number nobody asked for. The fix was one line — `min(gaps) == COOLDOWN_BARS` — and it
is the same shape as the "test a threshold ON the boundary" note further down.

**A mutant killed only by a crash somewhere else.** Loosening `risk_zone > 0` to `>= 0` in
`build_plan` is killed by three tests — all of them ERRORs in `tests/test_server_i18n.py`, whose
fixture happens to build a plan whose zone midpoint sits exactly on its stop and takes a
`ZeroDivisionError` through an assertion about translations. It counted as covered. It was not: the
class written for that guard does not kill it, and tidying an unrelated i18n fixture would have
freed it in silence. This has a methodological corollary worth more than the instance. **Score
mutants with `pytest -rfE`, not `-rf`.** `-rf` lists failures and not errors, so a mutant that
makes the code CRASH — which is a large share of the interesting ones — reads as a survivor. One of
this round's two campaigns scored a false survivor exactly that way on its first pass.

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
- **The cooldown's boundary is INCLUSIVE**: two signals exactly `cooldown_bars` apart are both
  taken, and only a closer pair is dropped. On the 20-day fixture 29 of the 59 same-archetype gaps
  sit exactly on it, so reading the boundary the other way discards about half the sample.
- **`min_score` is an inclusive FLOOR**: a hypothesis scoring exactly the number the operator asked
  for is kept. "Only the counts that fit at least this well", not "better than this".
- **A signal fired on the last trigger bar of the record is left unresolved**, with
  `outcome is None`, and is not counted in `n`, `hit_rate` or `expectancy_r`. There are no bars
  after it, and a trade resolved against an empty array comes back as a zero-bar flat one.
- ~~A market with no range (`atr == 0.0`) is planned, not refused~~ — **resolved the same day**:
  refused by name. A `PlanConfig` whose `min_stop_atr` exceeds its own `max_stop_atr` is now
  refused at construction too, which was the other way to make every plan unanswerable while
  blaming the market for it.
  See "Known inconsistencies": this one is pinned rather than settled, and deliberately so.
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
uv run pytest -m "not net"      # 996 passed, 1 deselected, ~6.5 s
uv run pytest                   # the same 996: `-m 'not net'` is in addopts now, no socket opened
uv run pytest -m net            # the 1 excluded test, ~5 s — needs the internet
uv run ruff check .             # clean
```

`test_hypothesis_library.py` reads real candles from `data/bars/`. If that store holds fewer than
56 months it SKIPS its real-candle arm, loudly, and the synthetic arm still runs. A skip there is
not a pass: half of every property in that file is then unmeasured, including the arm that caught
the planted one-bar lookahead when the synthetic arm did not.

To check a single fault: edit the source, run the suite, confirm the named test fails, revert, and
verify with `shasum -a 256` that the file is byte-identical. Three things that have each produced a
confidently wrong verdict in this repository, all of them measured rather than feared:

- **Purge `__pycache__` between runs, or set `PYTHONDONTWRITEBYTECODE=1`.** Several of these faults
  are same-byte-length edits and CPython's source-mtime check has one-second granularity, so a
  stale `.pyc` keeps executing the mutant after the revert. Confirmed again this round: a fast
  in-process probe loop produced three results that were each the PREVIOUS mutant's outcome. The
  ~6 s suite spaces its writes far enough apart that pytest runs were never affected; probe loops
  are where this bites.
- **Score with `-rfE`, not `-rf`.** `-rf` lists failures and omits errors, so a mutant that makes
  the code crash reads as a survivor.
- **When a mutant's killer is not the test you expected, suspect the mutant.** One fault written to
  test the cooldown key also changed the string the signal is LABELLED with, so it went red on a
  label mismatch rather than on the debounce — the right colour for the wrong reason, and it would
  have been recorded as a kill for a test that never saw the fault. Rewritten to touch only the
  dictionary key, it dies to the test that claims it.

### When this document was last audited against itself

A coverage document that is wrong about its own residual list is worse than none, and this one has
been wrong. Each pass that adds to this file re-applies a sample of the claims it is NOT touching
and watches the named test go red. The passes are listed newest first.

#### 2026-09-11, commit `01ab1fd` — 36 claims checked, 34 held

The largest gap between this document and the code that has been found yet, and the second time in
a row the stale entry was one nobody was looking at.

**Twenty-two claims did not hold, and all twenty-two were the same claim: "Left open: 0".** Two
rows of the kill-rate table said no observable fault survived in them. Six did in
`waves/projection.py` and sixteen in `engine/live.py` and `backtest/run.py`. Every one was
re-applied here against the full 983-test suite and survived it before a line of test code was
written; every one is now closed, and each closing test was watched go red on the exact fault,
reverted, and the source confirmed byte-identical by sha256. They are listed under "Closed
2026-09-11" above. The table now carries a measurement date per row, which is the only mechanism
that would have caught this earlier.

**Four more prose claims did not hold and are corrected in place**, each marked with its date where
it sits:

| Claim checked | How it was checked | Held? |
|---|---|---|
| a bare `uv run pytest` selects the `net` test and goes to the internet | ran it, and `make test`, and `-m net` | **no — `addopts` now carries `-m 'not net'`; nothing opens a socket** |
| `decide()` computes `WATCH if (out and best_in_zone) else (WATCH if out else NO_TRADE)` | read the source | **no — that ternary has not existed for a commit** |
| moving `w2`'s upper bound to 0.886 turns three tests red | applied it, full suite, `-rfE` | **no — exactly one** |
| `rr_in_zone` "still divides by an absolute distance" | applied the entry's own three scenarios | **no — `risk_zone` is signed; two of the three now quote 0.00R** |
| nothing pins what `best_in_zone` is supposed to do | re-gated the verdict on it, 480 cards | **no longer — it was true, and is now held by two named tests** |
| re-typing the Fibonacci literals back into `projection.py` is caught | applied it to both rows | yes — one named test each, and nothing else |
| `_fib_zone` sorts, so swapping the two bounds is equivalent | exhausted 20,736 combinations | yes — and it holds for the `w4` row too, which this file did not say |
| the 20-day fixture's 446 / 311 / 34 verdict counts | re-measured on the fixture | yes — reproduced exactly |
| the `.pyc` warning in "Reproducing any claim" is load-bearing | a probe loop without the purge | yes — three results were the previous mutant's |

**And ten claims from ten modules this pass did not touch, spot-checked the usual way:**

| Claim spot-checked | Mutant applied | Held? |
|---|---|---|
| a bar owns the last ms of its period (`core/timeframes.py`) | `+ tf.ms - 1` → `+ tf.ms` | yes |
| the base rate grades the move AFTER the signal bar (`validation/battery.py`) | `fwd` shifted back one bar | yes |
| ingest keeps the FIRST of a duplicated minute (`store/bars.py`) | `keep="first"` → `"last"` | yes |
| `SAFE_FUNCS` is frozen (`validation/expr.py`) | added `"apply": lambda f, x: f(x)` | yes |
| the warm-up mask is what enforces `min_warmup` (`hypotheses/base.py`) | mask deleted | yes |
| `provisional_window` refuses `n < 1` (`core/ring.py`) | `n < 1` → `n < 0` | yes |
| a pivot is priced at its bar's wick (`waves/pivots.py`) | high/low swapped | yes |
| the gap-heal window overlaps by exactly one bar (`feeds/binance_klines.py`) | the `- TF_1M.ms` dropped | yes |
| the cooldown debounces on the bare archetype (`backtest/run.py`) | key → archetype + direction, and ONLY the key | yes |
| blocks are cut from the tail (`validation/reality_check.py`) | `[:, :nb*L]` → `[:, n-nb*L:]` | **no — still a survivor, as this document says** |

Ten of ten behaved as written, the last one by surviving, which is what its entry claims. Counting
that row as "held" and the twenty-two table rows as "did not hold": **36 claims checked, 34 held.**

The lesson this pass adds to the one below it: the entry that goes stale is not only the one you
are not looking at, it is preferentially **the summary**. Every prose measurement in this file that
was re-run reproduced — the corpus numbers, the verdict counts, the wrong-side scenarios, the
`.pyc` warning. What had rotted was the table summarising them, and it had rotted in the direction
that flatters. Numbers produced by executing the code age well; numbers that summarise other
numbers do not, and they need a date.

#### The previous pass

The last pass re-applied all seven faults from the previous round independently — six
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

#### The pass that moved the R:R to the fill — a debt, now paid

That pass did NOT run an audit, and recorded the fact. What it ran was the behaviour: every figure
in the two entries it touched was produced by executing the shipped code — six months of 4h bars
for the corpus numbers, the 20-day fixture for the before-and-after on the signals, and hand-built
plans through the exported `build_plan` for the wrong-side table. What it did not do was re-apply
the faults behind the kill-rate table. It named the two places to re-run first — "the entry the
arithmetic is computed at, and the two gates that now decide on it" — and both turned out to be
well covered; the twenty-two holes the 2026-09-11 pass found were everywhere else. The debt is
paid, and the paragraph is kept because the prediction being wrong is the useful part.
