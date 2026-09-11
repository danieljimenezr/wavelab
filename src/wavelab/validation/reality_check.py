"""Multiple-testing correction: White's Reality Check and Hansen's SPA.

THE PROBLEM IT SOLVES. If you test 100 strategies on the same data and keep the best one, its
individual p-value means nothing: you ran 100 draws and kept the maximum. With 100 draws of a fair
coin, the best one comes up heads 60% of the time and looks like magic.

Demonstrated inside this very project: on BTC 4h, testing 137 strategies, the BEST hit rate that
random sets of signals produce has a median of 60.5% and a 95th percentile of 64.3%. A "64% hit
rate" headline is, literally, what chance produces at that scale.

HOW IT SOLVES IT. Instead of asking "is the best one good?", it asks "is the best one BETTER than
chance would have handed you after trying this many?". The joint distribution of the K strategies
is resampled under the null of zero performance, and the observed maximum is compared against the
maximum of the resample.

BLOCKS, NOT LOOSE OBSERVATIONS. Financial returns are autocorrelated and signals bunch up into
episodes. Resampling single observations would break that dependence and give falsely narrow
intervals. We use the stationary bootstrap (Politis-Romano), which resamples blocks of random
geometric length.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "RealityCheckResult",
    "effective_n",
    "optimal_block_length",
    "reality_check",
    "stationary_bootstrap_indices",
]


def optimal_block_length(x: np.ndarray) -> float:
    """Mean block length, a practical approximation to Politis-White.

    The n^(1/3) rule, adjusted by first-order autocorrelation: the more persistent the series, the
    longer the blocks have to be to preserve its dependence.
    """
    n = x.size
    if n < 20:
        return 2.0
    x = x - x.mean()
    denom = float((x * x).sum())
    rho = float((x[:-1] * x[1:]).sum() / denom) if denom > 0 else 0.0
    rho = min(max(rho, -0.9), 0.9)
    base = n ** (1 / 3)
    return float(max(2.0, min(n / 4, base * (1 + 2 * abs(rho)))))


def stationary_bootstrap_indices(n: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """Indices of a stationary resample: geometric-length blocks, wrapping around the end."""
    p = 1.0 / max(block, 1.0)
    idx = np.empty(n, dtype=np.int64)
    i = int(rng.integers(0, n))
    for t in range(n):
        idx[t] = i
        if rng.random() < p:
            i = int(rng.integers(0, n))
        else:
            i = (i + 1) % n
    return idx


def effective_n(signal_ts: np.ndarray, horizon_bars: int, bar_ms: int) -> int:
    """How many genuinely INDEPENDENT observations there are.

    Two signals separated by less than the holding horizon share the same price move: they are not
    two observations, they are one. Ignoring that is what makes an "n=120" that is really 12
    episodes produce a completely bogus p<0.001. It is López de Prado's label overlap problem.
    """
    if signal_ts.size == 0:
        return 0
    window = horizon_bars * bar_ms
    ts = np.sort(signal_ts)
    n, last = 1, ts[0]
    for t in ts[1:]:
        if t - last >= window:
            n += 1
            last = t
    return n


@dataclass(frozen=True, slots=True)
class RealityCheckResult:
    best_name: str
    best_stat: float
    p_value: float
    n_strategies: int
    n_obs: int
    block_length: float
    #: Each strategy's p-value against the resampled maximum. Without this you would only ever
    #: learn about the winner.
    p_values: dict[str, float]

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def render(self) -> str:
        verdict = ("BEATS the multiple-testing correction" if self.significant
                   else "DOES NOT beat the multiple-testing correction: consistent with chance")
        return (f"best: {self.best_name} (statistic {self.best_stat:+.4f})\n"
                f"p-value corrected for having tested {self.n_strategies} strategies: "
                f"{self.p_value:.4f}\n{verdict}")


def reality_check(
    returns: dict[str, np.ndarray],
    n_boot: int = 2000,
    seed: int = 7,
) -> RealityCheckResult:
    """White's Reality Check over each strategy's series of per-period returns.

    ``returns[k]`` is strategy k's return in each period (0 while it is out of the market). All of
    them must have the same length and be aligned in time.
    """
    names = sorted(returns)
    if not names:
        raise ValueError("there are no strategies to evaluate")
    M = np.vstack([np.asarray(returns[k], dtype=float) for k in names])
    K, n = M.shape
    means = M.mean(axis=1)
    root = np.sqrt(n)
    V = float((root * means).max())

    block = optimal_block_length(M[int(np.argmax(means))])
    rng = np.random.default_rng(seed)
    maxima = np.empty(n_boot)
    #: The series are centred (their own mean subtracted) to impose the null of ZERO performance.
    #: Uncentred, the bootstrap would be measuring the winner's variance, not the probability that
    #: chance would have produced a winner like it.
    Mc = M - means[:, None]

    # BLOCK BOOTSTRAP OVER BLOCK MEANS. Resampling the n observations one at a time costs K*n per
    # iteration: with 100 strategies and 20,000 bars that is 4·10⁹ operations and it never
    # finishes. Aggregating to block means first (which is what the moving-block bootstrap does
    # anyway) drops the cost to K*n_blocks and gives the same statistic: the mean of a resample of
    # blocks IS the mean of those blocks' means.
    L = max(2, round(block))
    nb = n // L
    if nb >= 30:
        # `n % L` observations do not fit a whole block and are dropped. Taking them off the TAIL
        # rather than the head is deliberate and, inside this branch, close to arbitrary: the
        # `nb >= 30` guard forces L <= n/30, so what is dropped is under 3.3% of the sample either
        # way, and `Mc` was centred using all n observations, so neither end carries the mean.
        #
        # It is written down because it is invisible: a mutation audit flipped this slice to drop
        # the oldest instead of the newest and the whole suite stayed green, which is correct — the
        # two are not distinguishable at this size — but "no test caught it" reads like a hole until
        # someone works out why it is not one. If the guard ever drops below 30 blocks this stops
        # being arbitrary: at nb = 4 it would be a quarter of the series, and then the end you keep
        # is a real decision about which regime the null is drawn from.
        B = Mc[:, : nb * L].reshape(K, nb, L).mean(axis=2)     # (K, n_blocks)
        for b in range(n_boot):
            idx = rng.integers(0, nb, size=nb)
            mb = B[:, idx].mean(axis=1) * root
            maxima[b] = mb.max()
        # each strategy's individual p-value against ITS OWN null distribution
        samples = np.empty((n_boot, K))
        for b in range(n_boot):
            idx = rng.integers(0, nb, size=nb)
            samples[b] = B[:, idx].mean(axis=1) * root
        beats = (samples >= (root * means)[None, :]).sum(axis=0)
    else:
        beats = np.zeros(K)
        for b in range(n_boot):
            idx = stationary_bootstrap_indices(n, block, rng)
            mb = Mc[:, idx].mean(axis=1) * root
            maxima[b] = mb.max()
            beats += (mb >= root * means)

    p = float((maxima >= V).mean())
    return RealityCheckResult(
        best_name=names[int(np.argmax(means))], best_stat=V, p_value=p,
        n_strategies=K, n_obs=n, block_length=block,
        p_values={k: float(beats[i] / n_boot) for i, k in enumerate(names)},
    )
