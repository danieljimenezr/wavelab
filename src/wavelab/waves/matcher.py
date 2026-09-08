"""Emparejador de plantillas: encuentra estructuras candidatas en la secuencia de pivotes.

TRES DECISIONES QUE LO HACEN VIABLE:

1. **Solo pivotes CONSECUTIVOS** a un grado dado. Permitir saltos convierte el problema en O(k⁵)
   tuplas y es lo que hunde a las implementaciones conocidas. Los grados superiores no salen de
   saltar pivotes: salen de la recursión de la gramática.

2. **Anclado a la DERECHA**: se descarta cualquier candidato cuyo último vértice esté a más de
   ``max_from_edge`` pivotes del borde. La historia que no puedes operar no merece CPU, y un conteo
   sobre estructura terminada hace meses no produce ninguna decisión.

3. **Estados PARCIALES incluidos**. Un emparejador que solo busca cincos completos no encontraría
   jamás una entrada, porque las entradas están en impulsos a medias.

Coste: ~13 inicios × 4 estados × 2 direcciones ≈ 100 comprobaciones por actualización, y solo se
recalcula cuando se CONFIRMA un pivote nuevo (una vez cada 6-20 velas). Es más barato que la batería
de indicadores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wavelab.core.types import Direction, Pivot, PivotKind
from wavelab.waves.rules import ARCHETYPES, ImpulseState, check_impulse

__all__ = ["Hypothesis", "MatcherConfig", "match_impulses", "score_guidelines"]


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """Una lectura posible de la estructura actual. NUNCA se produce una sola.

    El producto es un conjunto ORDENADO de hipótesis: un conteo real no es único, y presentar uno
    solo es afirmar una certeza que no existe.
    """

    state: ImpulseState
    direction: Direction
    pivots: tuple[Pivot, ...]
    points: tuple[float, ...]
    score: float
    fit: dict[str, float] = field(default_factory=dict)
    invalidation_price: float = 0.0
    invalidation_rule: str = ""
    archetype: str | None = None
    truncated: bool = False

    @property
    def id(self) -> str:
        return f"{self.state.value}:{self.direction.name}:{self.pivots[0].ts_ms}"

    @property
    def terminal_label(self) -> str:
        """En qué onda estamos AHORA. Es lo que se escribe en la etiqueta del gráfico."""
        return {ImpulseState.AT_2: "w2 completa → dentro de w3",
                ImpulseState.AT_3: "w3 completa → dentro de w4",
                ImpulseState.AT_4: "w4 completa → dentro de w5",
                ImpulseState.COMPLETE: "cinco completo → se espera ABC"}[self.state]


@dataclass(frozen=True, slots=True)
class MatcherConfig:
    #: Cuántos pivotes desde el borde puede estar el último vértice. Más allá, la estructura ya
    #: terminó y no produce decisión.
    max_from_edge: int = 3
    #: Cuántos inicios se prueban hacia atrás.
    max_starts: int = 13
    #: Pesos de las guías. Son CONSTANTES ELEGIDAS A MANO: cambiarlas es un ensayo y va a
    #: trials.sqlite, porque ajustar a ojo sin registrar es lo que hace que el N efectivo mienta.
    w_fib2: float = 0.30
    w_fib3: float = 0.30
    w_fib4: float = 0.20
    w_alt: float = 0.20
    top_n: int = 4


def _gauss(x: float, mu: float, sigma: float) -> float:
    """Cercanía a una razón de Fibonacci, en [0,1]. Sin normalizar: solo se compara consigo misma."""
    return float(pow(2.718281828459045, -0.5 * ((x - mu) / sigma) ** 2))


def score_guidelines(points: tuple[float, ...], cfg: MatcherConfig) -> tuple[float, dict[str, float]]:
    """Puntúa las GUÍAS (que no invalidan) frente a las reglas duras (que sí).

    Separadas a propósito: una guía incumplida baja la confianza, una regla rota mata el conteo.
    Mezclarlas convertiría "poco típico" en "imposible", y las estructuras atípicas son
    justamente las que más información llevan.
    """
    fit: dict[str, float] = {}
    n = len(points)

    # w2: retroceso típico 0,5-0,786, con el bolsillo dorado en 0,618.
    if n >= 3 and points[1] != points[0]:
        r2 = abs(points[2] - points[1]) / abs(points[1] - points[0])
        fit["retr_w2"] = r2
        s2 = _gauss(r2, 0.618, 0.18)
    else:
        s2 = 0.0

    # w3: extensión típica 1,618 de w1. La onda 3 extendida es la firma del impulso sano.
    if n >= 4 and points[1] != points[0]:
        e3 = abs(points[3] - points[2]) / abs(points[1] - points[0])
        fit["ext_w3"] = e3
        s3 = max(_gauss(e3, 1.618, 0.45), _gauss(e3, 2.618, 0.55))
    else:
        s3 = 0.0

    # w4: retroceso típico 0,382 de w3, mucho menos profundo que w2.
    if n >= 5 and points[3] != points[2]:
        r4 = abs(points[4] - points[3]) / abs(points[3] - points[2])
        fit["retr_w4"] = r4
        s4 = _gauss(r4, 0.382, 0.15)
    else:
        s4 = 0.0

    # Alternancia: si w2 es profunda, w4 es plana, y al revés. Es la guía más fiable de Elliott.
    if "retr_w2" in fit and "retr_w4" in fit:
        alt = abs(fit["retr_w2"] - fit["retr_w4"])
        fit["alternancia"] = alt
        sa = min(1.0, alt / 0.30)
    else:
        sa = 0.0

    pesos = [(cfg.w_fib2, s2, n >= 3), (cfg.w_fib3, s3, n >= 4),
             (cfg.w_fib4, s4, n >= 5), (cfg.w_alt, sa, n >= 5)]
    activos = [(w, s) for w, s, ok in pesos if ok]
    total_w = sum(w for w, _ in activos) or 1.0
    score = sum(w * s for w, s in activos) / total_w
    return score, fit


def match_impulses(
    pivots: tuple[Pivot, ...],
    cfg: MatcherConfig | None = None,
    *,
    allow_overlap: bool = False,
    directions: tuple[Direction, ...] = (Direction.LONG, Direction.SHORT),
) -> list[Hypothesis]:
    """Devuelve las hipótesis viables, ORDENADAS de mejor a peor.

    Solo se consideran pivotes confirmados: lo tentativo puede anotar el gráfico, nunca producir
    una hipótesis con la que se opere.
    """
    cfg = cfg or MatcherConfig()
    if len(pivots) < 3:
        return []

    n = len(pivots)
    out: list[Hypothesis] = []
    estados = ((3, ImpulseState.AT_2), (4, ImpulseState.AT_3),
               (5, ImpulseState.AT_4), (6, ImpulseState.COMPLETE))

    for k, state in estados:
        # Anclado a la derecha: el final del candidato debe estar cerca del borde.
        for fin in range(n, max(n - cfg.max_from_edge - 1, k - 1), -1):
            ini = fin - k
            if ini < 0 or ini < n - cfg.max_starts - k:
                continue
            seq = pivots[ini:fin]
            pts = tuple(p.price for p in seq)

            for d in directions:
                # La dirección la fija el primer tramo: un impulso alcista arranca desde un mínimo.
                esperado = PivotKind.LOW if d is Direction.LONG else PivotKind.HIGH
                if seq[0].kind is not esperado:
                    continue
                r = check_impulse(list(pts), d, allow_overlap=allow_overlap)
                if not r.valid:
                    continue
                score, fit = score_guidelines(pts, cfg)
                out.append(Hypothesis(
                    state=state, direction=d, pivots=seq, points=pts,
                    score=score, fit=fit,
                    invalidation_price=r.invalidation_price,
                    invalidation_rule=r.invalidation_rule,
                    archetype=ARCHETYPES[state], truncated=r.truncated,
                ))

    # Desempate: primero la puntuación de guías; a igualdad, la estructura MÁS COMPLETA, que
    # afirma más y por tanto es más falsable.
    out.sort(key=lambda h: (h.score, h.state.n_points), reverse=True)
    return out[: cfg.top_n]
