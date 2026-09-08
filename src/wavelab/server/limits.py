"""Límites de uso. Obligatorios antes de exponer esto a internet.

Cada validación cuesta entre 5 y 10 segundos de CPU: los 250 filtros aleatorios de control no son
gratis, y son justamente lo que hace que el veredicto valga algo. Sin límites, cualquiera con un
bucle de tres líneas deja el servicio inservible para los demás — y el nodo comparte máquina con
aplicaciones que facturan.

Tres capas, cada una para un problema distinto:

1. **Concurrencia global.** Como mucho N validaciones a la vez. Protege la CPU del nodo.
2. **Cubo por IP.** Un usuario no puede acaparar el servicio aunque el nodo esté ocioso.
3. **Tamaño de cuerpo.** Un CSV de 500 MB no es un usuario, es un ataque.

Deliberadamente NO se reduce el número de controles aleatorios bajo carga: sería degradar la
honestidad del resultado en silencio, y un veredicto peor calculado es peor que un veredicto que
tarda o que se rechaza con un mensaje claro.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

__all__ = ["RateLimiter", "TooBusy", "TooMany"]


class TooBusy(RuntimeError):
    """Demasiadas validaciones simultáneas."""


class TooMany(RuntimeError):
    """Esta IP ha gastado su cuota."""


@dataclass
class RateLimiter:
    max_concurrent: int = 2
    per_hour: int = 30
    per_minute: int = 6
    _sem: asyncio.Semaphore | None = None
    _hist: dict[str, deque] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._sem = asyncio.Semaphore(self.max_concurrent)

    def _check_ip(self, ip: str) -> None:
        ahora = time.monotonic()
        q = self._hist.setdefault(ip, deque())
        while q and ahora - q[0] > 3600:
            q.popleft()
        if len(q) >= self.per_hour:
            espera = int(3600 - (ahora - q[0]))
            raise TooMany(
                f"has hecho {self.per_hour} validaciones en la última hora, que es el límite. "
                f"Vuelve en {espera // 60} min. Cada validación ejecuta 250 controles aleatorios "
                "y eso cuesta CPU de verdad.")
        recientes = sum(1 for t in q if ahora - t < 60)
        if recientes >= self.per_minute:
            raise TooMany(
                f"máximo {self.per_minute} validaciones por minuto. Espera unos segundos.")
        q.append(ahora)

        # Poda de IPs inactivas: sin esto el diccionario crece indefinidamente y es una fuga de
        # memoria lenta, del tipo que solo se nota tras semanas en producción.
        if len(self._hist) > 5000:
            for k in [k for k, v in self._hist.items() if not v or ahora - v[-1] > 7200]:
                self._hist.pop(k, None)

    class _Ctx:
        def __init__(self, lim: RateLimiter, ip: str) -> None:
            self.lim, self.ip = lim, ip

        async def __aenter__(self):
            self.lim._check_ip(self.ip)
            try:
                await asyncio.wait_for(self.lim._sem.acquire(), timeout=25)
            except TimeoutError:
                raise TooBusy(
                    "hay demasiadas validaciones en marcha ahora mismo. Prueba en un minuto: "
                    "cada una tarda unos segundos y solo se ejecutan dos a la vez para no "
                    "degradar el resultado de nadie.") from None
            return self

        async def __aexit__(self, *exc):
            self.lim._sem.release()

    def slot(self, ip: str) -> _Ctx:
        return RateLimiter._Ctx(self, ip)

    @property
    def stats(self) -> dict:
        return {"ips_activas": len(self._hist),
                "libres": self._sem._value if self._sem else 0,
                "max_concurrent": self.max_concurrent}
