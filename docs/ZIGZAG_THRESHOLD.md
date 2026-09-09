# The ZigZag threshold: what `k` actually controls

Measured on BTCUSDT, 2025-01 → 2026-09, with `min_pct=0` to isolate the ATR effect.

| tf | k | pivots | bars/pivot | median \|leg\| | median lag |
|---|---|---|---|---|---|
| 15m | **1.5** | 10,441 | 5.7 | 0.75% | **2** |
| 15m | 3.0 | 2,730 | 21.7 | 1.50% | 5 |
| 15m | 5.0 | 856 | 69.1 | 2.79% | 15 |
| 15m | 8.0 | 281 | 210.4 | 4.80% | 47 |
| 1h | **1.5** | 2,496 | 5.9 | 1.65% | **1** |
| 1h | 5.0 | 221 | 66.9 | 5.39% | 16 |
| 4h | **1.5** | 601 | 6.1 | 3.65% | **2** |
| 4h | 3.0 | 169 | 21.9 | 6.49% | 5 |
| 4h | 5.0 | 54 | 68.4 | 12.43% | 19 |
| 1d | **1.5** | 100 | 6.2 | 9.46% | **2** |
| 1d | 5.0 | 6 | 102.5 | 32.01% | 24 |

## What the numbers say

**1. `k` really is dimensionless.** Pivot density is practically the same across every timeframe for
a given `k`: ~6 bars per pivot at k=1.5, ~22 at k=3, ~68 at k=5, ~200 at k=8. That is no accident:
the threshold is `k·ATR`, and ATR scales with the timeframe. This is what makes the multi-asset
requirement real — the same k=1.5 means the same thing on BTC, on EURUSD and on AAPL.

**2. `k` sets the DEGREE, the timeframe sets the SCALE.** At k=1.5 the median leg is 0.75% on 15m,
1.65% on 1h, 3.65% on 4h and 9.46% on 1d. That is the fractal property Elliott needs: the same
structure at different scales.

**3. The cost of raising `k` is NOT visual, it is latency.** On 15m, going from k=1.5 to k=8 drops
from 10,441 pivots to 281 — far cleaner structure — but the median confirmation lag goes from **2
bars to 47**, about twelve hours. You find out there was a top half a day after it happened.

That is the real trade-off, and it has no solution: it is a first-passage time to a barrier. The
further out you put the barrier, the longer you take to cross it. A "clean" ZigZag is a ZigZag that
tells you late, and half the value of a pivot is in finding out in time.

## Current decision

`k = 1.5` and `min_pct = 0.005`, which are the plan's values. **They have not been tuned by looking
at charts**: this table exists so that a future choice is an informed one, not to justify a change
now.

If it is ever changed, that is a TRIAL and it goes to `trials.sqlite` with its configuration hash.
The "we don't search for parameters" policy does not eliminate the search: it moves it to unrecorded
eyeball fitting, and then the Deflated Sharpe's effective N counts ~1 while the real
multiple-comparison burden is in the hundreds.

## A note on degrees

A single `k` produces **one degree** of pivots. Higher degrees do NOT come from raising `k`: they
come from the recursion in the grammar (`Motive` and `Corr` through themselves), where a wave 3 is
itself parsed as a complete impulse. Raising `k` to "see bigger waves" would be substituting
smoothing for structure, and paying for it in latency.
