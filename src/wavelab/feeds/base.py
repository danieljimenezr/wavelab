"""The feed seam: a Protocol with one method that matters.

Anti-astronaut rule: if this seam needed a second meaningful method to be useful, it would be drawn
in the wrong place. ``fetch_aux`` does not count as a second method because it returns empty by
default and exists so that v2's news layer is one line of configuration instead of a rewrite of the
transport.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from wavelab.core.timeframes import Timeframe
from wavelab.core.types import AuxEvent, Bar

__all__ = ["FeedAdapter", "FeedCaps", "Market"]


@dataclass(frozen=True, slots=True)
class FeedCaps:
    """What a feed can do. The engine asks this instead of assuming.

    ``max_klines_per_request`` and ``weight_budget_per_min`` are not decoration: the weight
    governor uses them to brake before the 418, which is an IP ban of up to 3 days and, for a
    local application, total downtime.
    """

    timeframes: frozenset[str] = frozenset()
    has_websocket: bool = False
    max_klines_per_request: int = 1000
    weight_budget_per_min: int = 6000
    deep_history_from_ms: int | None = None
    supports_aux: frozenset[str] = field(default_factory=frozenset)


@runtime_checkable
class FeedAdapter(Protocol):
    """A source of normalised bars."""

    name: str

    def caps(self) -> FeedCaps: ...

    async def fetch_klines(
        self, symbol: str, tf: Timeframe, start_ms: int, end_ms: int
    ) -> Sequence[Bar]: ...

    def stream(self, symbol: str, tf: Timeframe) -> AsyncIterator[Bar]:  # pragma: no cover
        """Live bars. A feed without a WebSocket may not implement this; `caps()` declares it."""
        raise NotImplementedError

    async def stream_events(self, symbol: str) -> AsyncIterator[AuxEvent]:  # pragma: no cover
        """Everything that is not a bar. Empty by default: this is the hook that keeps v2's news
        layer from forcing a redraw of the seam."""
        return
        yield  # type: ignore[unreachable]


class Market:
    SPOT = "spot"
    FUTURES_UM = "um"      # USD-M perpetuals
    FUTURES_CM = "cm"
