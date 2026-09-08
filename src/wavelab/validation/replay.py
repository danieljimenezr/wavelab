"""El arnés de determinismo. Se escribe ANTES del primer indicador y va como puerta de CI.

Comprueba dos propiedades distintas, y la segunda es la que de verdad importa:

**1. Determinismo.** Dos replays completos de las mismas velas producen salidas idénticas. Atrapa
estado global, dependencia del reloj de pared, iteración sobre conjuntos sin ordenar y aleatoriedad
sin semilla.

**2. Prefijo — el detector de lookahead.** Reproducir solo ``bars[:k]`` debe producir exactamente las
mismas ``k`` primeras salidas que el replay completo. Si el motor mira aunque sea una vela hacia
adelante, su salida en la vela ``k`` cambia según lo que venga DESPUÉS, y las dos series divergen.

La segunda propiedad es irremplazable porque el repintado **falla hacia arriba**: un motor que mira al
futuro produce un backtest más bonito, no un error. Ninguna suite de tests de valores esperados lo
detecta, porque los valores esperados también se calcularon con el mismo lookahead.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

from wavelab.core.clock import SimClock
from wavelab.core.types import Bar

__all__ = ["replay", "assert_replay_deterministic", "ReplayDivergence", "diff_path",
           "streaming", "EngineFactory"]


class ReplayDivergence(AssertionError):
    """Dos replays que deberían coincidir no coinciden. Siempre nombra el campo culpable."""


# --------------------------------------------------------------------------- diff

def diff_path(a: Any, b: Any, path: str = "") -> str | None:
    """Primera diferencia entre dos estructuras, como ruta legible.

    Sin esto, un fallo del arnés dice «las salidas difieren» y te deja media hora buscando cuál de
    cuarenta campos. Con esto dice ``salida[137].plan.stop: 106880.0 != 106884.5``.
    """
    if type(a) is not type(b):
        return f"{path or '<raíz>'}: tipos distintos {type(a).__name__} != {type(b).__name__}"

    if is_dataclass(a) and not isinstance(a, type):
        for f in fields(a):
            sub = diff_path(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}" if path else f.name)
            if sub:
                return sub
        return None

    if isinstance(a, (tuple, list)):
        if len(a) != len(b):
            return f"{path or '<raíz>'}: longitudes {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            sub = diff_path(x, y, f"{path}[{i}]")
            if sub:
                return sub
        return None

    if isinstance(a, dict):
        if a.keys() != b.keys():
            return f"{path or '<raíz>'}: claves {sorted(a.keys())} != {sorted(b.keys())}"
        for k in a:
            sub = diff_path(a[k], b[k], f"{path}[{k!r}]")
            if sub:
                return sub
        return None

    if a != b:
        return f"{path or '<raíz>'}: {a!r} != {b!r}"
    return None


# --------------------------------------------------------------------------- replay

def streaming(on_bar: Callable[[Any, Bar], tuple[Any, Any]],
              initial_state: Callable[[], Any]) -> "EngineFactory":
    """Fábrica para un motor correcto: ignora las velas que se le ofrecen.

    Un motor causal solo consume lo que ``on_bar`` le va entregando. Que esta fábrica descarte su
    argumento no es un detalle de conveniencia: es la definición operativa de «causal».
    """
    def factory(_bars: Sequence[Bar]) -> tuple[Any, Any]:
        return on_bar, initial_state
    return factory


#: Recibe las velas que ESTA corrida va a reproducir y devuelve ``(on_bar, initial_state)``.
#:
#: El argumento representa «todo lo que el motor puede ver en su almacén». Un motor correcto lo
#: ignora. Uno que precalcula sobre la serie entera y luego la trocea —el bug clásico de llamar al
#: detector de pivotes una vez sobre todo el array— lo usa, y por eso el test de prefijo lo caza:
#: en la corrida de prefijo solo recibe ``bars[:k]``, así que su salida cambia.
EngineFactory = Callable[[Sequence[Bar]], tuple[Callable[[Any, Bar], tuple[Any, Any]], Callable[[], Any]]]


def replay(
    factory: "EngineFactory",
    bars: Sequence[Bar],
    clock: SimClock | None = None,
) -> list[Any]:
    """Ejecuta el motor vela a vela y devuelve la salida de cada una.

    Es la MISMA función que ejecuta el motor en vivo. No hay ruta vectorizada, y no la habrá: si
    backtest y live fuesen dos implementaciones, su divergencia reintroduciría lookahead en silencio,
    y en un proyecto que etiqueta ondas a partir de pivotes que repintan esa es la forma más probable
    de que todo falle sin que nadie se entere.
    """
    on_bar, initial_state = factory(bars)
    clock = clock or SimClock(bars[0].open_time_ms if bars else 0)
    state = initial_state()
    out: list[Any] = []
    for bar in bars:
        if not bar.is_closed:
            raise ValueError(
                f"replay recibió una vela sin cerrar en {bar.open_time_ms}. "
                "El replay solo consume velas cerradas, igual que la ruta viva."
            )
        clock.set(bar.close_time_ms)
        state, result = on_bar(state, bar)
        out.append(result)
    return out


def assert_replay_deterministic(
    factory: "EngineFactory",
    bars: Sequence[Bar],
    checkpoints: Sequence[int] | None = None,
) -> None:
    """Verifica determinismo y ausencia de lookahead. Lanza ``ReplayDivergence`` si falla."""
    if len(bars) < 4:
        raise ValueError("hacen falta al menos 4 velas para que el test tenga sentido")

    full_a = replay(factory, bars)
    full_b = replay(factory, bars)

    d = diff_path(full_a, full_b, "salida")
    if d:
        raise ReplayDivergence(
            "NO DETERMINISTA: dos replays idénticos difieren.\n"
            f"  {d}\n"
            "  Causas típicas: estado global entre instancias, lectura del reloj de pared, "
            "iteración sobre un set/dict sin ordenar, o aleatoriedad sin semilla."
        )

    if checkpoints is None:
        n = len(bars)
        checkpoints = sorted({max(2, n // 4), max(3, n // 2), max(4, (3 * n) // 4), n - 1})

    for k in checkpoints:
        if not (2 <= k <= len(bars)):
            continue
        prefix = replay(factory, bars[:k])
        d = diff_path(prefix, full_a[:k], f"prefijo(k={k})")
        if d:
            raise ReplayDivergence(
                f"LOOKAHEAD DETECTADO en el prefijo k={k} de {len(bars)} velas.\n"
                f"  {d}\n"
                "  La salida de una vela cambia según qué velas EXISTAN después de ella, así que el "
                "motor está leyendo el futuro.\n"
                "  Sospechosos habituales: llamar al detector de pivotes sobre el array completo y "
                "luego cortarlo; usar `prominence` de scipy.find_peaks (se define contra el array "
                "ENTERO); indexar con [-1] una serie que incluye la vela en curso; o consumir un "
                "pivote por su `idx` en vez de por su `confirmed_idx`."
            )
