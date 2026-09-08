"""Bus de eventos en proceso. Sin Redis, sin Kafka, sin Celery.

Un broker externo rompería «ligera» para un único usuario local, y añadiría un modo de fallo (el
broker caído) más probable que los que se supone que evita.

El bus transporta ``Event = Bar | AuxEvent`` **desde el día 1**, aunque la capa de noticias sea v2.
Si transportase solo ``Bar``, una noticia — que no es una vela y no llega en un cierre de vela —
obligaría a añadir un segundo método al adaptador, y por la regla anti-astronauta eso significaría que
la costura estaba mal dibujada. Veinte líneas ahora evitan un refactor después.

Política de desbordamiento: **descartar lo más viejo y contar**. Un suscriptor lento (la UI, un
notificador con la red mal) jamás puede bloquear al consumidor del WebSocket: perder velas es
recuperable con un relleno REST; que Binance nos desconecte por no responder al ping escala hacia un
baneo de IP de hasta 3 días.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from wavelab.core.types import Event

__all__ = ["Bus", "Subscription"]

_CLOSED = object()


class Subscription:
    """Cola propia de un suscriptor. Se consume como iterador asíncrono."""

    __slots__ = ("name", "_q", "_dropped", "_bus")

    def __init__(self, bus: "Bus", name: str, maxsize: int) -> None:
        self._bus = bus
        self.name = name
        self._q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._dropped = 0

    @property
    def dropped(self) -> int:
        """Eventos descartados por lentitud. Se expone en la insignia de salud de datos:
        si esto no es cero, la interfaz está mintiendo sobre su frescura."""
        return self._dropped

    @property
    def qsize(self) -> int:
        return self._q.qsize()

    def _offer(self, event: Event | object) -> None:
        try:
            self._q.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self._q.get_nowait()      # descarta el más viejo
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
    """Publicación en abanico a suscriptores independientes."""

    __slots__ = ("_subs", "_maxsize", "_published")

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
        """No bloquea nunca, por diseño. Ver la nota sobre el baneo de IP arriba."""
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
