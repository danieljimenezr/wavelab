"""De hipótesis a plan: zona de entrada, stop, objetivos y la aritmética del rechazo.

EL STOP SALE DE LA INVALIDACIÓN, no de un múltiplo de ATR elegido a gusto. Esa es la ventaja
práctica real de Elliott, funcione o no su capacidad predictiva: el conteo produce un precio
OBJETIVO en el que la estructura es falsa. El ATR solo aporta un colchón para que una caza de
mechas no mate un conteo todavía válido.

REGLA QUE PARECE MENOR Y NO LO ES: si la distancia al stop supera ``max_stop_atr``, se REDUCE EL
TAMAÑO, nunca se ciñe el stop. Ceñirlo rompe el vínculo lógico entre el stop y la invalidación, que
es todo el sentido del marco: pasarías a tener un stop que no significa nada.
"""

from __future__ import annotations

from dataclasses import dataclass

from wavelab.core.types import Direction, TradePlan
from wavelab.waves.matcher import Hypothesis
from wavelab.waves.rules import ImpulseState

__all__ = ["PlanConfig", "PlanResult", "build_plan", "required_hit_rate"]


@dataclass(frozen=True, slots=True)
class PlanConfig:
    stop_buffer_atr: float = 0.25
    stop_buffer_pct: float = 0.001
    tick_size: float = 0.01
    max_stop_atr: float = 3.0
    min_stop_atr: float = 0.75
    fee_bps_taker: float = 7.5
    ev_min_r: float = 0.15
    expected_loss_r: float = 1.10
    exit_template_id: str = "std_2r_48b"


@dataclass(frozen=True, slots=True)
class PlanResult:
    """El plan más la aritmética con la que se acepta o se rechaza, siempre visible."""

    plan: TradePlan | None
    in_zone: bool
    rr_t2: float
    cost_r: float
    p_required: float
    stop_atr: float
    size_factor: float           # 1.0 normal; <1 si el stop está lejos
    reasons: tuple[str, ...]     # por qué NO, cuando no hay plan

    @property
    def viable(self) -> bool:
        return self.plan is not None


def required_hit_rate(rr: float, cost_r: float, cfg: PlanConfig) -> float:
    """``p = (L + c + EV_min) / (RR + L)``.

    Se muestra SIEMPRE, incluso cuando la herramienta rechaza. Es el encuadre intuitivo — «este
    R:R exige acertar el 36% de las veces» — mientras la máquina decide sobre la distribución de R.
    """
    L = cfg.expected_loss_r
    return (L + cost_r + cfg.ev_min_r) / (rr + L)


def _fib_zone(a: float, b: float, lo_r: float, hi_r: float) -> tuple[float, float]:
    """Zona de retroceso del tramo a→b entre dos razones. Devuelta ordenada."""
    x, y = b - (b - a) * lo_r, b - (b - a) * hi_r
    return (min(x, y), max(x, y))


def build_plan(h: Hypothesis, price: float, atr: float, cfg: PlanConfig | None = None) -> PlanResult:
    """Construye el plan de una hipótesis, o explica por qué no hay ninguno."""
    cfg = cfg or PlanConfig()
    s = 1 if h.direction is Direction.LONG else -1
    P = h.points
    razones: list[str] = []

    if h.archetype not in ("w2", "w4"):
        return PlanResult(None, False, 0, 0, 0, 0, 1.0,
                          (f"el estado «{h.terminal_label}» no ofrece entrada; "
                           "dentro de la onda 3 se gestiona, no se entra.",))

    if h.archetype == "w2":
        # Onda 2 completa, empieza la 3: la entrada de mayor calidad de Elliott.
        # Zona = retroceso 0,5-0,786 de la onda 1. Núcleo: bolsillo dorado 0,618-0,65.
        lo, hi = _fib_zone(P[0], P[1], 0.500, 0.786)
        base = P[2]
        w1 = P[1] - P[0]
        objetivos = (base + w1 * 1.000, base + w1 * 1.618, base + w1 * 2.618)
    else:
        # Onda 4 completa, empieza la 5: menor confianza, y el stop es MÁS CEÑIDO (P1 en vez de P0),
        # así que el R:R suele ser peor aunque intuitivamente parezca la entrada más segura.
        lo, hi = _fib_zone(P[2], P[3], 0.382, 0.500)
        # Truncar la zona para que nunca invada el territorio de la onda 1: una entrada ahí sería
        # una entrada por debajo de su propia invalidación.
        if h.direction is Direction.LONG:
            lo = max(lo, P[1] * 1.0005)
        else:
            hi = min(hi, P[1] * 0.9995)
        base = P[4]
        w1 = P[1] - P[0]
        objetivos = (base + w1 * 1.000, base + (P[3] - P[0]) * 0.618, base + w1 * 1.618)

    if lo >= hi:
        return PlanResult(None, False, 0, 0, 0, 0, 1.0,
                          ("la zona de entrada queda vacía tras truncarla contra la "
                           "invalidación: no hay sitio donde entrar por encima del stop.",))

    entrada = (lo + hi) / 2.0

    # --- stop: invalidación más colchón -------------------------------------------------------
    colchon = max(cfg.stop_buffer_atr * atr, 2 * cfg.tick_size, cfg.stop_buffer_pct * price)
    stop = h.invalidation_price - s * colchon
    riesgo = abs(entrada - stop)
    if riesgo <= 0:
        return PlanResult(None, False, 0, 0, 0, 0, 1.0,
                          ("la zona de entrada está al otro lado del stop.",))

    stop_atr = riesgo / atr if atr else float("inf")
    size = 1.0
    if stop_atr > cfg.max_stop_atr:
        # REDUCIR TAMAÑO, jamás ceñir el stop: ceñirlo rompe el vínculo con la invalidación.
        size = cfg.max_stop_atr / stop_atr
        razones.append(f"stop a {stop_atr:.1f} ATR (>{cfg.max_stop_atr}): tamaño reducido a "
                       f"{size:.0%}, el stop NO se ciñe")
    if stop_atr < cfg.min_stop_atr:
        return PlanResult(None, False, 0, 0, 0, stop_atr, 1.0,
                          (f"stop a {stop_atr:.2f} ATR: está dentro del suelo de ruido "
                           f"(<{cfg.min_stop_atr} ATR) y lo barrería cualquier mecha.",))

    # --- costes y umbral de acierto -----------------------------------------------------------
    # En spot no hay funding: el funding es del perpetuo, y aquí sería un coste inventado.
    cost_r = (2 * cfg.fee_bps_taker / 10_000) * entrada / riesgo
    rr_t2 = abs(objetivos[1] - entrada) / riesgo
    p_req = required_hit_rate(rr_t2, cost_r, cfg)

    en_zona = lo <= price <= hi
    if not en_zona:
        razones.append(f"precio ${price:,.0f} fuera de la zona ${lo:,.0f}-${hi:,.0f}")

    plan = TradePlan(
        archetype=f"{h.archetype}_{'long' if s > 0 else 'short'}",
        direction=h.direction, entry_lo=lo, entry_hi=hi, stop=stop,
        invalidation_price=h.invalidation_price, invalidation_rule=h.invalidation_rule,
        targets=objetivos, exit_template_id=cfg.exit_template_id, count_id=h.id,
    )
    return PlanResult(plan, en_zona, rr_t2, cost_r, p_req, stop_atr, size, tuple(razones))
