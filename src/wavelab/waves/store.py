"""Pivot store whose only reachable accessor is ``as_of``.

``__getitem__`` **raises**. It does not return anything, it does not warn: it raises. A lint rule
can be silenced with a comment; a ``raise`` cannot be silenced without deleting it, and deleting it
shows up in the diff.

The property that holds up everything else: **confirmed pivots never change, so the history can only
GROW**. Three things come out of that for free: the count is genuinely append-only, the "as it stood
at bar t" snapshot costs nothing, and the replay harness is O(n) instead of O(n²).

That property only holds if the confirmation threshold is FROZEN at the extreme bar
(``Pivot.thr_at_extreme``). If it were recomputed with today's ATR, a pivot confirmed under low
volatility could stop satisfying the inequality tomorrow under expanded volatility, and a count the
user had already seen would disappear with no invalidation event.
"""

from __future__ import annotations

from bisect import bisect_right

from wavelab.core.causality import CausalityError
from wavelab.core.types import Pivot

__all__ = ["PivotStore"]

_FORBIDDEN = (
    "PivotStore is deliberately neither indexable nor iterable. Use `as_of(now_ms)`, the only "
    "accessor that respects causality. If you need 'all the pivots', the right question is "
    "'all the pivots knowable at which instant'."
)


class PivotStore:
    """Confirmed pivots (append-only) plus, at most, one provisional pivot."""

    __slots__ = ("_conf_ts", "_confirmed", "_provisional")

    def __init__(self) -> None:
        self._confirmed: list[Pivot] = []
        self._conf_ts: list[int] = []          # parallel list, for bisect
        self._provisional: Pivot | None = None

    # ------------------------------------------------------------------ forbidden

    def __getitem__(self, _i):
        raise CausalityError(_FORBIDDEN)

    def __iter__(self):
        raise CausalityError(_FORBIDDEN)

    def __len__(self):
        # Raises too: the total number of pivots includes the ones confirmed AFTER `now_ms`, so it
        # is information from the future however innocent it looks.
        raise CausalityError(_FORBIDDEN + " To count, use `n_as_of(now_ms)`.")

    # ------------------------------------------------------------------ writes

    def append_confirmed(self, pivot: Pivot) -> None:
        if not pivot.is_confirmed:
            raise ValueError(
                f"append_confirmed got an unconfirmed pivot at idx={pivot.idx}. "
                "Provisional pivots go through set_provisional()."
            )
        if self._conf_ts and pivot.confirmed_ts_ms < self._conf_ts[-1]:
            raise ValueError(
                f"out-of-order confirmation: {pivot.confirmed_ts_ms} < {self._conf_ts[-1]}. "
                "Confirmation order must be monotone or `as_of` would stop returning a prefix."
            )
        if self._confirmed and pivot.idx <= self._confirmed[-1].idx:
            raise ValueError(
                f"confirmed pivot out of positional order: idx={pivot.idx} <= "
                f"{self._confirmed[-1].idx}"
            )
        self._confirmed.append(pivot)
        self._conf_ts.append(int(pivot.confirmed_ts_ms))

    def set_provisional(self, pivot: Pivot | None) -> None:
        if pivot is not None and pivot.is_confirmed:
            raise ValueError("set_provisional got an already-confirmed pivot")
        self._provisional = pivot

    # ------------------------------------------------------------------ causal reads

    def as_of(self, now_ms: int) -> tuple[Pivot, ...]:
        """The pivots that were CONFIRMED at ``now_ms``.

        By construction this is a prefix of the full history, and the prefix returned at ``t`` is a
        prefix of the one returned at ``t+1``. That is the property ``test_pivot_monotonicity``
        checks.
        """
        k = bisect_right(self._conf_ts, int(now_ms))
        return tuple(self._confirmed[:k])

    def n_as_of(self, now_ms: int) -> int:
        return bisect_right(self._conf_ts, int(now_ms))

    def last_as_of(self, now_ms: int) -> Pivot | None:
        k = self.n_as_of(now_ms)
        return self._confirmed[k - 1] if k else None

    def provisional_as_of(self, now_ms: int) -> Pivot | None:
        """The provisional pivot, if its extreme had already happened at ``now_ms``.

        What comes out of here can only produce a TENTATIVE annotation: dashed stroke, hollow "?"
        label. Never a signal, never a journal row, never a hit rate.
        """
        p = self._provisional
        if p is None or p.ts_ms > now_ms:
            return None
        return p
