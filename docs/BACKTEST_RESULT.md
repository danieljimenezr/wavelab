# The first honest result: the Elliott archetypes have NO edge

Date: 2026-09-08 · BTCUSDT 4h · 2017-08 → 2026-09 · 4,757,255 1m bars · 19,820 decisions
evaluated · **1,449 resolved trades**

## The number

| costs | n | expectancy | hit rate | profit factor | null arm | **edge** |
|---|---|---|---|---|---|---|
| ×1.0 | 1,449 | +0.007R | 33.3% | 1.01 | +0.015R | **−0.008R** |
| ×1.5 | 1,449 | −0.023R | 33.3% | 0.97 | −0.015R | −0.008R |
| ×2.0 | 1,449 | −0.053R | 33.1% | 0.93 | −0.045R | −0.008R |

Bootstrap confidence interval on the expectancy: **[−0.076R, +0.090R]**. It contains zero with room
on both sides. The minimum required to admit a trade was +0.15R, and the lower bound is not remotely
close.

## The decisive test: the null arm

Every signal generated a twin trade entered at a nearby random instant, with the **same geometry in
R and the same time barrier**. It isolates "does the wave count contribute anything?" from "did BTC
go up?", which is what a backtest without a control actually measures.

| | n | signal | null | edge | 95% CI of the edge |
|---|---|---|---|---|---|
| longs | 658 | +0.144R | **+0.165R** | −0.021R | [−0.146, +0.105] |
| shorts | 791 | −0.107R | **−0.109R** | +0.003R | [−0.116, +0.122] |
| all | 1,449 | +0.007R | +0.015R | −0.008R | [−0.095, +0.081] |

**This is the whole story.** At first glance there appeared to be signal: longs gave +0.144R and
shorts −0.107R. But **the null arm reproduces exactly the same asymmetry** (+0.165R and −0.109R).
That is: over a period in which BTC rose 1,748%, buying worked and selling did not, and it made no
difference whether the entry was picked by Elliott or by a die. The wave count contributed nothing.

Without the null arm, this result would have read as "wave 2 longs work", and it would have been
false.

## The Fibonacci score does not predict anything either

Rank correlation between guideline fit and outcome: **rho = +0.0038, p = 0.886**.

That is: a count fitting the Fibonacci proportions beautifully **says nothing** about whether that
trade is going to win. The entire scoring function — wave 2 retracement, wave 3 extension,
alternation — has no detectable predictive power over 1,449 cases.

## Context

- Buy and hold: **+1,748%** ($4,261 → $78,734)
- The strategy: **+10.2R** over 1,449 trades, which is to say, noise

## What this does NOT mean

- **It does not mean the code is wrong.** The causal engine, the determinism harness and the null
  arm did exactly what they were supposed to: they detected that there is no edge instead of
  manufacturing one.
- **It does not mean Elliott is useless as a framework.** It still produces an objective
  invalidation price, and that is worth something as a risk-definition instrument even if the
  direction is noise.
- **It is not a surprising result.** It agrees with van Ginneken (Tilburg), the only rigorous, open
  study of algorithmic Elliott, which likewise found no significant profitability. And this
  project's plan anticipated it in so many words: "the validation gate will probably reject the
  strategy on its first run".

## What it does mean

**That the honesty machinery earned its cost the first day it was used.** Without the paired null
arm, without the causality harness and without the cost filter, this system would have been showing
professional-looking signals with zero expectancy, and it would have taken months of live trading to
find out.

## Trials recorded

The five analyses from this session are in `trials.sqlite`. Effective N = 2 configuration hashes.
Any future Sharpe will have to be deflated by that number: I have looked at these data five times,
and that is on the record instead of being conveniently forgotten.
