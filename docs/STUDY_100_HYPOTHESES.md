# Exhaustive study: 100 hypotheses, 3 timeframes, 9 years. None survives.

BTCUSDT · 2017-08 → 2026-09 · 4,757,335 1m bars · **238 combinations evaluated**

## How it was done, and why the order matters

1. **Pre-registration.** Eight agents wrote 100 hypotheses in 8 families, each declaring a
   `rationale` (why it ought to work, with the concrete market mechanism) and a `prior` (what effect
   is expected and **under what conditions it would be falsified**) **before seeing a single
   result**.
2. **Independent audit.** A prefix-invariance test over all 100 × all their timeframes: compute the
   signals over the full series and over a prefix, and require bar-by-bar equality, with a regime
   shock injected after the cut to expose any statistic defined against the whole array. **Zero
   leaks.** Also: 1 duplicate removed, 1 sign bug fixed (it affected 2.7% of bars) and 4 control
   arms added, because six hypotheses cited a control that did not exist.
3. **Evaluation on excess over drift.** The statistic is NOT the raw return but
   `signal × (return − mean drift)`. On raw returns, the winner is always whichever spends the most
   time long, because BTC rose 1,748%: that is beta, and buying it costs nothing.
4. **White's Reality Check** with a block bootstrap, over ALL hypotheses including the failures,
   plus a Šidák correction for having tried 3 timeframes.

## The result

| timeframe | Reality Check winner | p corrected for 100 hypotheses |
|---|---|---|
| 1h | `trend.ema200_filter` | 0.1235 — does not pass |
| **4h** | `trend.ema200_filter` | **0.0200 — passes** |
| 1d | `seasonality.halving_cycle_phase` | 0.0688 — does not pass |

Šidák correction for the 3 timeframes: **1 − (1 − 0.0200)³ = 0.0588**.

**That does not clear 5%. Nothing survives.**

## And the best part: the winner falsifies itself

The `prior` of `trend.ema200_filter`, written before seeing any number:

> "if the mechanism is reflexive, the effect must be LARGER on 1d, which is where the level is
> published and watched, and close to nil on 15m and 1h. An edge equal or greater on 15m would
> falsify the explanation even if the number came out profitable."

What was measured:

| timeframe | excess | out of sample |
|---|---|---|
| 1h | +0.026% | **−0.023%** |
| 4h | **+0.139%** | +0.122% |
| 1d | **−0.063%** | **−0.119%** |

It is **negative on 1d**, exactly where its own theory predicted the strongest effect. A real effect
does not change sign between adjacent timeframes: that is the signature of noise.

**The winner of the multiple-comparison test is falsified by the criterion it wrote for itself.**
Without the pre-registration, the headline would have been "the EMA200 filter works on 4h, p=0.02".

## Out-of-sample degradation

| timeframe | positive excess IN sample | OUT of sample |
|---|---|---|
| 1h | 19% | 12% |
| 4h | 45% | **25%** |
| 1d | 42% | 38% |

If the effects were real, they would persist. On 4h almost half of them vanish on stepping out of
sample.

A revealing detail: on 1h **only 19%** of the strategies have positive excess. Less than half of
what coin flips would give — because a wrong signal is not neutral: it pays the spread of being on
the wrong side.

## The only thing worth following up (which is NOT the same as "it works")

Two hypotheses keep their edge almost intact out of sample on 4h, which is the hardest test to pass
by chance:

| | in sample | out | retains | effective n |
|---|---|---|---|---|
| `candles.body_flow_20` | +1.150% | +1.123% | **98%** | 149 |
| `momentum.ao_confirms_extreme` | +0.337% | +0.379% | **112%** | 387 |

They do not clear the multiple-comparison test, so they are **not a finding**. They are candidates
for FORWARD verification, on data nobody has seen yet. Carrying on searching backwards over the same
nine years cannot settle it: every extra look spends more degrees of freedom without contributing
any new information.

`candles.body_flow_20` measures what fraction of the range traded over 20 bars turned into net
open-to-close progress: high when the range is travelled rather than retraced.
`momentum.ao_confirms_extreme` requires a new price high to come with an Awesome Oscillator higher
than at the previous high — the signature of a new cohort buying rather than the same one rotating.

## Trials recorded

238 combinations + 3 Reality Checks. All of it to `trials.sqlite`. Any future result over these same
data drags this number along with it.
