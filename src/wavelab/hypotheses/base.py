"""The hypothesis contract. Every one of them is REGISTERED BEFORE any result is looked at.

Why pre-registration. What separates science from self-deception is not testing many things
—that is fine— but deciding WHAT COUNTS as success before looking. If you test first and choose
afterwards you always find something, and you have no way of telling signal from the maximum of N
draws.

Every hypothesis declares:
  - `rationale`: WHY it should work. Written before seeing a single number.
  - `prior`:     what effect is expected, and in which direction.
  - `params`:    FIXED. Not searched over. Changing one is a NEW hypothesis and another trial.

And all of them count towards the multiple-comparisons correction, the ones that fail included.
Hiding the failures is what turns a study into a brochure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["REGISTRY", "Hypothesis", "Series", "register"]


@dataclass(frozen=True, slots=True)
class Series:
    """The candles of one timeframe. Everything a hypothesis is allowed to see."""
    tf: str
    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self) -> int:
        return int(self.close.size)


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """A testable idea, declared before it is tested."""

    name: str
    family: str
    rationale: str
    prior: str
    fn: object                    # (Series) -> np.ndarray of {-1, 0, +1}
    params: dict = field(default_factory=dict)
    timeframes: tuple[str, ...] = ("15m", "1h", "4h", "1d")
    min_warmup: int = 200

    def signals(self, s: Series) -> np.ndarray:
        """+1 = long, -1 = short, 0 = flat. Strictly causal: position i may only depend on data up
        to and including i. A botched shift here invents an edge that does not exist, and it is the
        single most common mistake in the whole industry."""
        out = np.asarray(self.fn(s), dtype=np.int8)
        if out.size != len(s):
            raise ValueError(f"{self.name}: returned {out.size} signals for {len(s)} candles")
        out[: self.min_warmup] = 0
        return out


REGISTRY: dict[str, Hypothesis] = {}


def register(h: Hypothesis) -> Hypothesis:
    if h.name in REGISTRY:
        raise ValueError(f"duplicate hypothesis: {h.name}")
    if not h.rationale.strip() or not h.prior.strip():
        raise ValueError(f"{h.name}: every hypothesis must declare a rationale and a prior BEFORE "
                         "it is run. Without those it is not a hypothesis, it is a search.")
    REGISTRY[h.name] = h
    return h
