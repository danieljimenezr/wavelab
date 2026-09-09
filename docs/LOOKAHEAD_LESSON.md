# The day I walked straight past my own architecture

2026-09-08. This project has: an `AsOf` type that raises if you read too early, a `PivotStore` whose
`__getitem__` raises, a `@causal` decorator that checks every argument, a `ProvisionalWindow` type
structurally incapable of feeding the causal path, and a replay harness that detects lookahead by
comparing prefixes.

And I still wrote this in a throwaway analysis script:

```python
r = np.diff(np.log(c))        # r[i] = return from day i to i+1
w = c[1:] > ema[1:]           # w[i] uses the close of day i+1
performance = r * w           # decides using data that does not exist yet
```

Result: CAGR of **99.3%**, Sharpe 1.62, max drawdown −33%. Spectacular. And false.

The causal version — swapping `c[1:]` for `c[:-1]` — gives CAGR **26.1%**, Sharpe 0.55, drawdown
−56%. That is: **worse than buy and hold** on return.

## How it was caught

Not by reading the code. By a routine robustness check: *what happens if I delay the signal by a
day?*

| delay | CAGR |
|---|---|
| 0 days | 99.3% |
| **1 day** | **26.1%** |
| 2 days | 23.5% |

A real effect does not evaporate because you executed a day later. **If a strategy dies when you
delay it by one bar, it is not a strategy: it is an information leak.** It is the cheapest and most
brutal test there is, and it should be run every time.

## The lesson that matters

The causality architecture protects the code that goes THROUGH it. A quick analysis script, written
to "just take a look", goes through no guard at all — and it is exactly where the decisions about
what to build get made.

And it happened TWICE on the same day:
1. The Reality Check fed with raw return instead of excess over drift, which rewarded whichever
   strategy spent the most time long in an asset that rose 1,748%.
2. This off-by-one index misalignment.

If someone who has built all this machinery makes the mistake twice in one afternoon, the reasonable
conclusion is that **most published and sold strategies have this bug**. And not out of bad faith:
because the bug produces beautiful numbers and raises no error at all.

## Rule added to the project

Every analysis that produces a number that influences a decision must pass, without exception:

1. **Delay test.** Delay the signal by one bar. If the result collapses, there is a leak.
2. **Random control with the same exposure.** If a random filter with the same average exposure
   performs similarly, there is no skill: there is beta.
3. **Excess over drift, never raw return.**
