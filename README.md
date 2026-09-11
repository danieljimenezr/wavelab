# WaveLab

Real-time BTC analyser: **causal** Elliott waves, market regime, and entry/exit points with an
explicit invalidation price.

A **visual, advisory** tool. It places no orders, stores no exchange keys, and connects to no
account. You decide and you trade.

---

## Read this before using it

### 1. The product is the invalidation price, not the wave label

At every bar close on the trigger timeframe, WaveLab answers three questions in a falsifiable way:

1. **What structure is the market in?** An *ordered* set of hypotheses, never a single one.
2. **At what exact price is that structure false?** A price that comes out of the rule engine
   carrying the name of the rule that produced it (R1, R2b, R3, the 2-4 line).
3. **Does the resulting R:R justify the trade?** A filter that shows the arithmetic, whether it
   says yes or no — and quotes it at the price you can actually get.

Wave labels are the **means** of computing the invalidation price. The invalidation price is the
**product**.

The third question has a trap in it worth naming, because the tool fell into it. The entry zone is
a retracement you are *waiting* for, so the R:R measured at the middle of the zone is the better
number — and it is the one a limit order gets IF it fills. A limit order may never fill. So the
headline R:R is quoted at the close of the bar that produced the plan: the price you get by buying
at market, which you can always do, and the same price the backtest books at. The zone figure
travels beside it as a condition ("… if your limit fills in the zone"), never as the headline, and
the filters that reject decide on the headline too.

Quoting the zone figure as the headline overstated the reachable R:R on roughly four plans in five
over six months of 4h bars — 1,357 of 1,659, by a median of 2.5R — and on 71 of them the market had
already gone past the stop, which `abs()` had been reporting as ordinary risk. A tool that refuses
to flatter its user does not get to make an exception for its own best number.

Both numbers are on the card, because they are not the same number even when the price is in the
zone: they are one ratio read at two entry prices, and inside the zone they still differ by a
median 0.78R. The headline is not always the smaller of the two either — below its zone a long is
buying cheaper than the zone midpoint — so what the card promises is that the bold figure is the
one you can get by buying now, not that it is the lower one.

### 2. Elliott may well have no directional edge

The only rigorous, open study of algorithmic Elliott (van Ginneken, Tilburg University: three
objective, replicable algorithms over AUD/USD, EUR/USD, USD/CAD and USD/JPY, 2005-2013, with a
random-walk bootstrap and adjustment for costs, risk and interest-rate differentials) **found no
conclusive evidence of significant profitability**. It is a single unreplicated study, on a
different asset class and a different era — not the last word, but the best public evidence there is.

The defensible value of this tool is not that Elliott predicts. It is that Elliott produces an
**objective** invalidation price, and that is a risk-definition instrument even if the direction is
noise.

### 3. It will say NO_TRADE most of the time

That is the correct behaviour and it will feel like the thing is broken. When it refuses, it shows
the arithmetic of the refusal.

### 4. The ceiling on autonomy is theoretical, not an engineering problem

BTC since 2017 is ~9 years. A swing system gives 40-80 trades a year, of which only ~1/7 land out
of sample under walk-forward, spread across 4 regimes → **13-35 observations per regime**. At that
n, a 70% hit rate and a 40% hit rate **are not distinguishable**.

Consequence: "the system decides which indicators to use or discard" is satisfiable **at bar level**
(tens of thousands of observations) and **is not** at archetype-trade level. That is why the design
separates the two:

- the **information coefficient tilts** the weights using abundant bar-level data,
- **Hedge selects** using scarce trade-level data,

and the interface shows the two terms separately so you can see which one is doing the work. Any
tool that promises autonomous indicator selection at this data rate is lying.

### 5. Every backtest number is contaminated, including the ones that pass the gate

Whoever designed this had already seen BTC's history. Every threshold, every rule and every
Fibonacci level chosen is an implicit trial fitted to that same history. That is why
`trials.sqlite` records every configuration hash ever evaluated, and the Deflated Sharpe is
computed with that real effective N.

### 6. The test suite is held to the same standard, and it is written down

A tool that refuses to flatter its user should not flatter itself about its own tooling. **[What the
test suite protects, and what it does not](docs/TEST_COVERAGE.md)** is the honest statement: the
kill rate per module measured by mutation rather than by line coverage, every fault deliberately
left open with the reason, every one resolved with the way it went and the measurement of what was
wrong kept beside it, and — the part that matters — the properties nothing verifies at all. The
largest of those is still the catalogue: all 100 hypotheses are held to six properties (causality,
alphabet, warm-up, scale, purity, degeneracy) and not one test checks that any of them computes the
indicator it says it computes.

---

## Design principles

**Causality is a type, not a convention.** `PivotStore.__getitem__` raises `CausalityError`, so
`as_of(t)` is the only reachable path. `AsOf[T]` carries `available_at`. The `@causal` decorator
checks every argument. A lint rule can be silenced; a `raise` cannot. Repainting is the one class of
bug that **fails upwards** (the backtest comes out *better*, not worse), so it has to be impossible
to write, not merely tested for.

**One single function, live and in backtest.** The engine exposes
`on_bar(state, bar) -> (state, signals, decision)` and the backtester runs exactly that same
function. **There is no vectorised path.** If backtest and live were two implementations, their
divergence would silently reintroduce lookahead.

**Anti-astronaut rule.** There are five seams (`FeedAdapter`, `FeatureProvider`, `Production`,
`SignalSource`, `Veto`), each with a single method that matters. **If a seam needs a second
meaningful method to be useful, it has been drawn in the wrong place.** No dependency injection
container, no entry-point discovery, no plugin framework.

**No statistic as a bare number.** Always the triple *(value, the n it comes from, the state of the
precondition)*. If the precondition fails, it shows "not computable — missing N", never a number. A
statistic that returns a reassuring value from insufficient data is worse than not having it: it
manufactures confidence.

---

## What is deliberately NOT built

It is written down here so it does not erode three months from now:

- Grid search, Optuna or genetic search over indicator parameters. The parameters are fixed
  (RSI-14, ATR-14, EMA 21/55/200, MACD 12/26/9) and are not touched. Parameter search **is** the
  backtest-overfitting generator.
- Reinforcement learning. Neural price prediction. Bandits over strategies.
- Continuous re-optimisation and automatic indicator discovery.
- Order execution, exchange keys, or any multi-user deployment.

---

## Status

Under construction. See `docs/` and the M0 → M9 milestone order.

## Licence

MIT. See `LICENSE` and `NOTICE` for third-party attributions.
