"""In-process event bus. No Redis, no Kafka, no Celery.

An external broker would break "lightweight" for a single local user, and would add a failure mode
(the broker being down) more likely than the ones it is supposed to prevent.

The bus carries ``Event = Bar | AuxEvent`` **from day 1**, even though the news layer is v2. If it
only carried ``Bar``, a news item — which is not a bar and does not arrive on a bar close — would
force a second method onto the adapter, and by the anti-astronaut rule that would mean the seam was
drawn in the wrong place. Twenty lines now save a refactor later.

Overflow policy: **drop the oldest and count it**. A slow subscriber (the UI, a notifier on a bad
network) can never be allowed to block the WebSocket consumer: losing bars is recoverable with a
REST backfill; having Binance disconnect us for not answering a ping escalates into an IP ban of up
to 3 days.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from wavelab.core.types import Event

__all__ = ["Bus", "Subscription"]

_CLOSED = object()


class Subscription:
    """A subscriber's own queue. Consumed as an async iterator."""

    __slots__ = ("_bus", "_dropped", "_q", "name")

    def __init__(self, bus: Bus, name: str, maxsize: int) -> None:
        self._bus = bus
        self.name = name
        self._q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._dropped = 0

    @property
    def dropped(self) -> int:
        """Events dropped because the subscriber was too slow. Exposed on the data-health badge:
        if this is not zero, the interface is lying about how fresh it is."""
        return self._dropped

    @property
    def qsize(self) -> int:
        return self._q.qsize()

    def _offer(self, event: Event | object) -> None:
        try:
            self._q.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self._q.get_nowait()      # drop the oldest
                self._dropped += 1
                self._q.put_nowait(event)
            except (asyncio.QueueEmpty, asyncio.QueueFull):  # pragma: no cover
                self._dropped += 1

    async def __aiter__(self) -> AsyncIterator[Event]:
        while True:
            item = await self._q.get()
            if item is _CLOSED:
                return
            yield item

    def close(self) -> None:
        self._bus.unsubscribe(self)


class Bus:
    """Fan-out publication to independent subscribers."""

    __slots__ = ("_maxsize", "_published", "_subs")

    def __init__(self, maxsize: int = 1024) -> None:
        self._subs: list[Subscription] = []
        self._maxsize = maxsize
        self._published = 0

    def subscribe(self, name: str, maxsize: int | None = None) -> Subscription:
        sub = Subscription(self, name, maxsize or self._maxsize)
        self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: Subscription) -> None:
        if sub in self._subs:
            self._subs.remove(sub)
            sub._offer(_CLOSED)

    def publish(self, event: Event) -> None:
        """Never blocks, by design. See the note about the IP ban above."""
        self._published += 1
        for sub in self._subs:
            sub._offer(event)

    @property
    def published(self) -> int:
        return self._published

    @property
    def total_dropped(self) -> int:
        return sum(s.dropped for s in self._subs)

    def close(self) -> None:
        for sub in list(self._subs):
            self.unsubscribe(sub)
