"""Reglas duras de Elliott. Booleanas, y cada una emite SU PROPIO precio de invalidación.

Esta es la mejor propiedad del diseño: el número más grande de la tarjeta de señal lo produce el
motor de reglas, no se ensambla aguas abajo. Así no puede desviarse de la regla que lo justifica —
si el stop y su razón se calculasen en módulos distintos, tarde o temprano dirían cosas distintas y
nadie se enteraría.

★ NO-TERMINALES PARCIALES. Un motor que solo evalúa impulsos COMPLETOS es un juguete de anotación:
el 100% de las entradas viven en impulsos incompletos (final de onda 2, final de onda 4). Por eso
cada estado parcial tiene su propio SUBCONJUNTO de reglas evaluables y su propia invalidación:

    Impulse@2  (w1,w2 hechas, dentro de w3)   R1          invalidación P0   -> arquetipo w2_long
    Impulse@3  (dentro de w4)                 R1, R2b     invalidación P1   -> vigilancia
    Impulse@4  (dentro de w5)                 R1,R2b,R3   invalidación P1   -> arquetipo w4_long
    Impulse    (completo)                     + R2        invalidación P1

R2 ("la 3 nunca es la más corta") NO es evaluable antes de que exista w5: se degrada a GUÍA mientras
el impulso está incompleto. Marcarla como regla dura y evaluarla igual sería inventarse un veredicto.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from wavelab.core.types import Direction, RuleVerdict

__all__ = ["ImpulseState", "check_impulse", "invalidation_for", "ARCHETYPES",
           "fib_retracement", "fib_projection", "RuleSet"]


class ImpulseState(StrEnum):
    """Hasta dónde está construido el impulso. Determina qué se puede afirmar."""
    AT_2 = "impulse@2"     # P0,P1,P2 conocidos; dentro de la onda 3
    AT_3 = "impulse@3"     # + P3; dentro de la onda 4
    AT_4 = "impulse@4"     # + P4; dentro de la onda 5
    COMPLETE = "impulse"   # + P5

    @property
    def n_points(self) -> int:
        return {"impulse@2": 3, "impulse@3": 4, "impulse@4": 5, "impulse": 6}[self.value]


#: Qué arquetipo de operación ofrece cada estado. `None` = no ofrece entrada, solo vigilancia.
ARCHETYPES: dict[ImpulseState, str | None] = {
    ImpulseState.AT_2: "w2",   # la entrada de mayor calidad de Elliott
    ImpulseState.AT_3: None,   # dentro de la 3: ya no se entra, se gestiona
    ImpulseState.AT_4: "w4",   # menor confianza; vigilar truncación y diagonal terminal
    ImpulseState.COMPLETE: "abc",  # tras la 5 se espera correctiva
}


@dataclass(frozen=True, slots=True)
class RuleSet:
    """Veredictos de un candidato, con su invalidación y su arquetipo."""

    state: ImpulseState
    direction: Direction
    verdicts: tuple[RuleVerdict, ...]
    invalidation_price: float
    invalidation_rule: str
    archetype: str | None
    truncated: bool = False

    @property
    def valid(self) -> bool:
        """Solo cuentan las reglas EVALUABLES. Una regla no evaluable no invalida nada."""
        return all(v.ok for v in self.verdicts if v.evaluable)

    @property
    def broken(self) -> tuple[str, ...]:
        return tuple(v.rule for v in self.verdicts if v.evaluable and not v.ok)


def _s(direction: Direction) -> int:
    return 1 if direction is Direction.LONG else -1


def check_impulse(
    points: list[float],
    direction: Direction = Direction.LONG,
    *,
    allow_overlap: bool = False,
) -> RuleSet:
    """Evalúa un impulso a partir de sus vértices ``[P0, P1, P2, ...]``.

    3 puntos = Impulse@2, 4 = @3, 5 = @4, 6 = completo. Con menos de 3 no hay nada que afirmar.

    ``allow_overlap`` es por patrón Y por activo: siempre cierto en diagonales, y cierto para
    impulsos solo en futuros y materias primas (la excepción de Prechter). Una línea de
    configuración, que es la implementación más barata posible del gancho multi-activo.
    """
    n = len(points)
    if n < 3:
        raise ValueError(f"hacen falta al menos 3 vértices (P0,P1,P2); hay {n}")
    if n > 6:
        raise ValueError(f"un impulso tiene como mucho 6 vértices; hay {n}")

    state = {3: ImpulseState.AT_2, 4: ImpulseState.AT_3,
             5: ImpulseState.AT_4, 6: ImpulseState.COMPLETE}[n]
    s = _s(direction)
    P = points
    v: list[RuleVerdict] = []

    # --- R1: la onda 2 nunca retrocede el 100% de la onda 1. Invalidación en P0. -------------
    r1_ok = s * P[2] > s * P[0]
    retr2 = abs(P[2] - P[1]) / abs(P[1] - P[0]) if P[1] != P[0] else float("inf")
    v.append(RuleVerdict(
        "R1", r1_ok, P[0],
        f"w2 retrocede {retr2:.1%} de w1" + ("" if r1_ok else " — supera el 100%")))

    # --- R2b: la onda 3 supera el final de la onda 1. Invalidación en P1. --------------------
    if n >= 4:
        ok = s * P[3] > s * P[1]
        v.append(RuleVerdict("R2b", ok, P[1],
                             "w3 supera el final de w1" if ok else "w3 NO supera el final de w1"))
    else:
        v.append(RuleVerdict("R2b", True, P[1], "aún dentro de w3: no evaluable", evaluable=False))

    # --- R3: la onda 4 no entra en territorio de la onda 1. Invalidación en P1. --------------
    if n >= 5:
        ok = (s * P[4] > s * P[1]) or allow_overlap
        nota = "w4 respeta el territorio de w1"
        if not ok:
            nota = "w4 ENTRA en territorio de w1"
        elif allow_overlap and s * P[4] <= s * P[1]:
            nota = "w4 solapa, permitido en este activo/patrón (excepción de Prechter)"
        v.append(RuleVerdict("R3", ok, P[1], nota))
    else:
        v.append(RuleVerdict("R3", True, P[1], "aún dentro de w4: no evaluable", evaluable=False))

    # --- R2: la onda 3 nunca es la más corta de 1/3/5. -------------------------------------
    # NO evaluable sin w5. Degradarla a guía en vez de fingir un veredicto es la diferencia
    # entre un motor honesto y uno que se inventa certezas sobre estructura que aún no existe.
    if n == 6:
        l1, l3, l5 = abs(P[1] - P[0]), abs(P[3] - P[2]), abs(P[5] - P[4])
        ok = not (l3 < l1 and l3 < l5)
        v.append(RuleVerdict("R2", ok, P[1],
                             f"longitudes w1={l1:.0f} w3={l3:.0f} w5={l5:.0f}"
                             + ("" if ok else " — w3 es la MÁS CORTA")))
    else:
        v.append(RuleVerdict("R2", True, P[1],
                             "w5 no existe todavía: R2 es guía, no regla", evaluable=False))

    # --- Truncación: LEGAL. No existe ninguna regla que exija que w5 supere a w3. ------------
    # Rechazarla tiraría justo los conteos que más quieres cerca de un techo.
    truncated = n == 6 and (s * P[5] <= s * P[3])

    price, rule = invalidation_for(state, points, direction)
    return RuleSet(state, direction, tuple(v), price, rule, ARCHETYPES[state], truncated)


def invalidation_for(state: ImpulseState, points: list[float],
                     direction: Direction = Direction.LONG) -> tuple[float, str]:
    """El precio que falsifica esta estructura, y el NOMBRE de la regla que lo produce.

    Dentro de la onda 3 la invalidación es P0 por R1. A partir de la onda 4 pasa a P1, porque tanto
    R2b como R3 se apoyan ahí: es un stop más ceñido, y es el motivo de que la entrada en onda 4
    tenga peor R:R aunque parezca más segura.
    """
    if state is ImpulseState.AT_2:
        return points[0], "R1 (w2 no puede retroceder el 100% de w1)"
    return points[1], "R3 (w4 no puede entrar en territorio de w1)"


# ---------------------------------------------------------------------------- Fibonacci

#: Retrocesos típicos por onda. El "bolsillo dorado" 0,618-0,65 es el núcleo de la zona de w2.
RETRACEMENTS = {
    "w2": (0.500, 0.786, 0.618, 0.650),   # (mín, máx, núcleo_lo, núcleo_hi)
    "w4": (0.382, 0.500, 0.382, 0.450),
}

#: Extensiones para proyectar objetivos.
EXTENSIONS = (1.000, 1.272, 1.618, 2.618)


def fib_retracement(a: float, b: float, ratios: tuple[float, ...]) -> list[float]:
    """Retrocesos del tramo a→b. En espacio LINEAL porque un retroceso es una fracción del
    tramo, no una razón de precios: aquí el log no aporta y confunde."""
    return [b - (b - a) * r for r in ratios]


def fib_projection(origin: float, leg_a: float, leg_b: float,
                   ratios: tuple[float, ...] = EXTENSIONS) -> list[float]:
    """Proyecta objetivos desde ``origin`` usando la longitud del tramo ``leg_a→leg_b``."""
    length = leg_b - leg_a
    return [origin + length * r for r in ratios]
