"""La causalidad es un tipo, no una convención.

El repintado es la única clase de bug de este proyecto que **falla hacia arriba**: si el motor mira
al futuro, el backtest sale *mejor*, no peor. Por eso no basta con testearlo — tiene que ser
imposible de escribir.

Tres mecanismos, en orden de fuerza:

1. ``AsOf[T]`` transporta ``available_at_ms``. Leerlo antes de tiempo lanza.
2. ``@causal`` inspecciona cada argumento y lanza si alguno no estaba disponible en ``now_ms``.
3. ``ProvisionalWindow`` es un tipo DISTINTO de ``Window``: una función ``@causal`` lo rechaza
   siempre, así que el canal provisional no puede alimentar la ruta de señales ni por accidente.

Este módulo no importa nada de ``wavelab.core.types`` a propósito: comprueba por atributos (duck
typing) para no crear un ciclo y para que cualquier tipo futuro que exponga el mismo contrato quede
protegido sin tocar este fichero.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["CausalityError", "AsOf", "causal", "is_causal"]


class CausalityError(RuntimeError):
    """Se ha intentado leer un dato antes del instante en que estuvo disponible.

    Nunca se captura para continuar: es un fallo de programación, no una condición de ejecución.
    """


@dataclass(frozen=True, slots=True)
class AsOf[T]:
    """Un valor junto al instante en que pasó a ser conocible.

    ``available_at_ms`` NO es cuándo ocurrió el hecho, sino el primer milisegundo en que el sistema
    tenía derecho a saberlo. Para un pivote de ZigZag son cosas muy distintas: el extremo ocurre en
    la vela ``t``, pero solo se confirma cientos de velas después. Confundirlas es toda la clase de
    bug que este módulo existe para impedir.
    """

    value: T
    available_at_ms: int

    def get(self, now_ms: int) -> T:
        if now_ms < self.available_at_ms:
            raise CausalityError(
                f"lectura acausal: el valor estuvo disponible en {self.available_at_ms} "
                f"y se ha pedido en {now_ms} "
                f"({self.available_at_ms - now_ms} ms en el futuro)"
            )
        return self.value

    def known_at(self, now_ms: int) -> bool:
        """Igual que ``get`` pero sin lanzar. Para ramificar, nunca para leer."""
        return now_ms >= self.available_at_ms


def _reject(what: str, detail: str, fn_name: str) -> None:
    raise CausalityError(f"{fn_name}: {what} — {detail}")


def _check(name: str, v: Any, now_ms: int, fn_name: str) -> None:
    """Rechaza cualquier argumento que no estuviese disponible en ``now_ms``.

    Recursivo sobre tuplas y listas porque las secuencias de pivotes se pasan así.
    """
    # 1. Canal provisional: prohibido en cualquier función causal, sin excepción y sin mirar fechas.
    if getattr(type(v), "__wavelab_provisional__", False):
        _reject(
            f"argumento `{name}`",
            "es un canal PROVISIONAL y no puede alimentar la ruta causal. "
            "Los datos provisionales solo pueden producir anotaciones tentativas, "
            "nunca señales, journal ni estadísticas.",
            fn_name,
        )

    # 2. AsOf: el caso explícito.
    if isinstance(v, AsOf):
        if now_ms < v.available_at_ms:
            _reject(
                f"argumento `{name}`",
                f"AsOf disponible en {v.available_at_ms}, pedido en {now_ms} "
                f"({v.available_at_ms - now_ms} ms en el futuro)",
                fn_name,
            )
        return

    # 3. Bar: ni sin cerrar, ni cerrada después de `now_ms`.
    close = getattr(v, "close_time_ms", None)
    if close is not None and getattr(v, "open_time_ms", None) is not None:
        if getattr(v, "is_closed", True) is False:
            _reject(
                f"argumento `{name}`",
                "es una vela SIN CERRAR. Las features causales solo consumen velas cerradas; "
                "usa el canal provisional si de verdad quieres el precio en curso.",
                fn_name,
            )
        if close > now_ms:
            _reject(
                f"argumento `{name}`",
                f"la vela cierra en {close}, después de now_ms={now_ms}",
                fn_name,
            )
        return

    # 4. Window: su último índice cerrado no puede caer en el futuro.
    end = getattr(v, "end_closed_ts_ms", None)
    if end is not None:
        if end > now_ms:
            _reject(
                f"argumento `{name}`",
                f"la ventana termina en {end}, después de now_ms={now_ms}",
                fn_name,
            )
        return

    # 5. Secuencias: pivotes, señales, hipótesis.
    if isinstance(v, (tuple, list)):
        for i, item in enumerate(v):
            _check(f"{name}[{i}]", item, now_ms, fn_name)
        return

    # 6. Objetos con marca temporal propia (Pivot confirmado, Signal, ...).
    for attr in ("confirmed_ts_ms", "available_at_ms", "ts_event_ms"):
        ts = getattr(v, attr, None)
        if ts is not None and isinstance(ts, int) and ts > now_ms:
            _reject(
                f"argumento `{name}`",
                f"su `{attr}`={ts} es posterior a now_ms={now_ms}",
                fn_name,
            )
            return


def causal[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Marca una función como causal y verifica sus argumentos en cada llamada.

    La función DEBE aceptar un parámetro ``now_ms``: sin un instante de referencia explícito no hay
    forma de decidir qué era conocible, y una comprobación que adivina el instante no es una
    comprobación.

    Coste: ~2-4 µs por llamada (no usa ``Signature.bind``, que sería ~20× más caro). El análisis
    corre una vez por vela cerrada, así que es irrelevante — y esta comprobación NO se desactiva en
    producción, porque el bug que evita no se manifiesta como un fallo sino como un backtest bonito.
    """
    params = list(inspect.signature(fn).parameters)
    if "now_ms" not in params:
        raise TypeError(
            f"@causal exige un parámetro `now_ms` en {fn.__qualname__}: "
            "sin instante de referencia no se puede verificar la causalidad."
        )
    now_pos = params.index("now_ms")
    name = fn.__qualname__

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if "now_ms" in kwargs:
            now_ms = kwargs["now_ms"]
        elif len(args) > now_pos:
            now_ms = args[now_pos]
        else:
            raise TypeError(f"{name}: falta `now_ms` (obligatorio en una función @causal)")
        if not isinstance(now_ms, int):
            raise TypeError(f"{name}: `now_ms` debe ser un int en ms, no {type(now_ms).__name__}")

        for i, v in enumerate(args):
            if i != now_pos:
                _check(params[i] if i < len(params) else f"arg{i}", v, now_ms, name)
        for k, v in kwargs.items():
            if k != "now_ms":
                _check(k, v, now_ms, name)
        return fn(*args, **kwargs)

    wrapper.__wavelab_causal__ = True  # type: ignore[attr-defined]
    return wrapper


def is_causal(fn: Any) -> bool:
    """¿Está esta función marcada como causal? Lo usa el registro de features para rechazar
    proveedores sin marcar en la ruta viva."""
    return bool(getattr(fn, "__wavelab_causal__", False))
