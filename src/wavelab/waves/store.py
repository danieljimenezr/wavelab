"""Almacén de pivotes cuyo único accesor alcanzable es ``as_of``.

``__getitem__`` **lanza**. No devuelve nada, no avisa: lanza. Una regla de linter se puede silenciar
con un comentario; un ``raise`` no se puede silenciar sin borrarlo, y borrarlo sale en el diff.

La propiedad que sostiene todo lo demás: **los pivotes confirmados nunca cambian, así que el
histórico solo puede CRECER**. De ahí salen tres cosas gratis: el conteo es genuinamente append-only,
la instantánea «como estaba en la vela t» no cuesta nada, y el arnés de replay es O(n) en vez de
O(n²).

Esa propiedad solo se sostiene si el umbral de confirmación está CONGELADO en la vela del extremo
(``Pivot.thr_at_extreme``). Si se recalculase con el ATR de hoy, un pivote confirmado con volatilidad
baja podría dejar de cumplir la desigualdad mañana con volatilidad expandida, y un conteo que el
usuario ya vio desaparecería sin evento de invalidación.
"""

from __future__ import annotations

from bisect import bisect_right

from wavelab.core.causality import CausalityError
from wavelab.core.types import Pivot

__all__ = ["PivotStore"]

_FORBIDDEN = (
    "PivotStore no es indexable ni iterable a propósito. Usa `as_of(now_ms)`, que es el único "
    "accesor que respeta la causalidad. Si necesitas «todos los pivotes», la pregunta correcta es "
    "«todos los pivotes conocibles en qué instante»."
)


class PivotStore:
    """Pivotes confirmados (append-only) más, como mucho, un pivote provisional."""

    __slots__ = ("_conf_ts", "_confirmed", "_provisional")

    def __init__(self) -> None:
        self._confirmed: list[Pivot] = []
        self._conf_ts: list[int] = []          # paralelo, para bisect
        self._provisional: Pivot | None = None

    # ------------------------------------------------------------------ prohibido

    def __getitem__(self, _i):
        raise CausalityError(_FORBIDDEN)

    def __iter__(self):
        raise CausalityError(_FORBIDDEN)

    def __len__(self):
        # También lanza: el número total de pivotes incluye los confirmados DESPUÉS de `now_ms`,
        # así que es información del futuro por mucho que parezca inocente.
        raise CausalityError(_FORBIDDEN + " Para contar, usa `n_as_of(now_ms)`.")

    # ------------------------------------------------------------------ escritura

    def append_confirmed(self, pivot: Pivot) -> None:
        if not pivot.is_confirmed:
            raise ValueError(
                f"append_confirmed recibió un pivote sin confirmar en idx={pivot.idx}. "
                "Los pivotes provisionales van por set_provisional()."
            )
        if self._conf_ts and pivot.confirmed_ts_ms < self._conf_ts[-1]:
            raise ValueError(
                f"confirmación fuera de orden: {pivot.confirmed_ts_ms} < {self._conf_ts[-1]}. "
                "El orden de confirmación debe ser monótono o `as_of` dejaría de devolver un prefijo."
            )
        if self._confirmed and pivot.idx <= self._confirmed[-1].idx:
            raise ValueError(
                f"pivote confirmado fuera de orden posicional: idx={pivot.idx} <= "
                f"{self._confirmed[-1].idx}"
            )
        self._confirmed.append(pivot)
        self._conf_ts.append(int(pivot.confirmed_ts_ms))

    def set_provisional(self, pivot: Pivot | None) -> None:
        if pivot is not None and pivot.is_confirmed:
            raise ValueError("set_provisional recibió un pivote ya confirmado")
        self._provisional = pivot

    # ------------------------------------------------------------------ lectura causal

    def as_of(self, now_ms: int) -> tuple[Pivot, ...]:
        """Los pivotes que estaban CONFIRMADOS en ``now_ms``.

        Por construcción esto es un prefijo del histórico completo, y el prefijo devuelto en
        ``t`` es prefijo del devuelto en ``t+1``. Es la propiedad que verifica
        ``test_pivot_monotonicity``.
        """
        k = bisect_right(self._conf_ts, int(now_ms))
        return tuple(self._confirmed[:k])

    def n_as_of(self, now_ms: int) -> int:
        return bisect_right(self._conf_ts, int(now_ms))

    def last_as_of(self, now_ms: int) -> Pivot | None:
        k = self.n_as_of(now_ms)
        return self._confirmed[k - 1] if k else None

    def provisional_as_of(self, now_ms: int) -> Pivot | None:
        """El pivote provisional, si su extremo ya había ocurrido en ``now_ms``.

        Lo que sale de aquí solo puede producir anotación TENTATIVA: trazo discontinuo, etiqueta
        hueca «?». Nunca una señal, nunca una fila de journal, nunca una tasa de acierto.
        """
        p = self._provisional
        if p is None or p.ts_ms > now_ms:
            return None
        return p
