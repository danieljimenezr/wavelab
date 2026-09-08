"""Los tipos que definen las costuras del sistema.

Nada de lógica aquí: solo los contratos. Si un tipo de este módulo necesita cambiar para añadir una
funcionalidad prevista (otro activo, otra fuente de señal, noticias), la costura estaba mal dibujada.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

from wavelab.core.timeframes import Timeframe, close_time_for

__all__ = [
    "AuxEvent",
    "Bar",
    "Decision",
    "Direction",
    "Event",
    "ExitTemplate",
    "MaturityLevel",
    "Pivot",
    "PivotKind",
    "RuleVerdict",
    "Signal",
    "SourceKind",
    "Stat",
    "TradePlan",
    "Verdict",
]


# --------------------------------------------------------------------------------------
# Transporte
# --------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Bar:
    """Una vela. Inmutable, y con el contrato de `close_time` aseverado en construcción.

    ``n_source_bars`` e ``is_gap`` viajan CON la vela, no en una estructura paralela: una vela de 1h
    construida con 43 minutos tiene que ser reconocible como tal en cualquier punto del sistema, y
    una máscara aparte se pierde en el primer slice.
    """

    symbol: str
    tf: Timeframe
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    trades: int = 0
    taker_buy_base: float = 0.0
    taker_buy_quote: float = 0.0
    is_closed: bool = True
    n_source_bars: int = 0
    is_gap: bool = False

    def __post_init__(self) -> None:
        # Barato (una comparación de enteros) y atrapa el error más caro del proyecto:
        # un adaptador que invente su propio convenio de close_time.
        if self.open_time_ms % self.tf.ms != 0:
            raise ValueError(
                f"Bar {self.symbol} {self.tf}: open_time_ms={self.open_time_ms} no cae en la "
                f"rejilla UTC de {self.tf} (resto {self.open_time_ms % self.tf.ms}). "
                "El adaptador debe alinear a la rejilla, no redondear."
            )

    @property
    def close_time_ms(self) -> int:
        return close_time_for(self.open_time_ms, self.tf)

    @property
    def coverage(self) -> float:
        """Fracción de velas de 1m realmente presentes. 1.0 = completa."""
        exp = self.tf.expected_source_bars
        return 1.0 if exp <= 1 else self.n_source_bars / exp

    @property
    def range(self) -> float:
        return self.high - self.low


@dataclass(frozen=True, slots=True)
class AuxEvent:
    """Cualquier cosa que no sea una vela: funding, liquidación, noticia, evento macro.

    Existe desde el día 1 aunque la capa de noticias sea v2. Las dos marcas de tiempo separadas son
    el motivo: ``ts_event_ms`` (cuándo ocurrió) y ``ts_ingest_ms`` (cuándo nos enteramos). Esa
    distinción NO se puede reconstruir a posteriori, así que o se registra desde el principio o el
    histórico de noticias nace inservible para calibrar latencia.
    """

    ts_event_ms: int
    ts_ingest_ms: int
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> int:
        return self.ts_ingest_ms - self.ts_event_ms


Event = Bar | AuxEvent


# --------------------------------------------------------------------------------------
# Enumeraciones
# --------------------------------------------------------------------------------------

class SourceKind(StrEnum):
    """De dónde viene una señal. El tope por tipo se configura y se aplica al fusionar,
    de modo que activar noticias en v2 es una línea de config y no un refactor."""
    STRUCTURE = "structure"      # Elliott, ruptura de estructura, niveles
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    FLOW = "flow"                # volumen, CVD, VWAP
    DERIVATIVES = "derivatives"  # funding, OI, ratios (contexto cross-instrumento en spot)
    ONCHAIN = "onchain"
    SENTIMENT = "sentiment"
    NEWS = "news"                # tope 0.0 en v1; se sube a 0.25 en v2 cambiando config


class Direction(IntEnum):
    SHORT = -1
    FLAT = 0
    LONG = 1

    @property
    def sign(self) -> int:
        return int(self.value)


class Verdict(StrEnum):
    """Tres estados, nunca un booleano.

    NO_TRADE es el caso COMÚN y muestra siempre la aritmética del rechazo.
    WATCH cubre dos situaciones distintas que el usuario debe poder diferenciar: hay estructura pero
    el precio no está en zona, o el conteo todavía es tentativo.
    """
    NO_TRADE = "no_trade"
    WATCH = "watch"
    ACTIONABLE = "actionable"


class MaturityLevel(IntEnum):
    """Cuánta evidencia respalda lo que se está mostrando. Se calcula por celda, no global."""
    PRIOR = 0             # tabla escrita a mano; ninguna probabilidad en pantalla
    HISTORICAL = 1        # walk-forward purgado sobre 9 años; contaminado por selección
    HISTORICAL_RIGOR = 2  # CPCV, SPA, Deflated Sharpe, PBO
    FORWARD = 3           # evidencia no contaminada acumulada en vivo


# --------------------------------------------------------------------------------------
# Estructura
# --------------------------------------------------------------------------------------

class PivotKind(IntEnum):
    LOW = -1
    HIGH = 1


@dataclass(frozen=True, slots=True)
class Pivot:
    """Un extremo del ZigZag, con sus DOS marcas de tiempo separadas.

    ``idx``/``ts_ms`` es dónde se DIBUJA (la vela del extremo).
    ``confirmed_idx``/``confirmed_ts_ms`` es la primera vela en que estaba PERMITIDO saberlo.

    Confundirlas es la clase de bug entera. El retardo entre ambas es un tiempo de primer paso a una
    barrera: mediana de pocas velas, cola derecha muy pesada, cientos de velas en tendencia fuerte.
    Nunca se puede asumir un retardo fijo.

    ``thr_at_extreme`` se CONGELA en la vela del extremo. Es lo que hace la confirmación monótona: si
    el umbral se recalculase con el ATR de hoy, un pivote confirmado ayer podría dejar de estarlo, el
    histórico de conteos dejaría de ser append-only, y un conteo que el usuario ya vio desaparecería
    sin evento de invalidación — justo la deshonestidad que este diseño existe para eliminar.
    """

    idx: int
    ts_ms: int
    price: float
    kind: PivotKind
    thr_at_extreme: float
    confirmed_idx: int | None = None
    confirmed_ts_ms: int | None = None

    @property
    def is_confirmed(self) -> bool:
        return self.confirmed_idx is not None

    @property
    def confirm_price(self) -> float:
        """El precio al que este pivote quedaría confirmado.

        Se dibuja como línea gris discontinua: «el conteo confirma por debajo de 108.240». Convierte
        la debilidad de repintado de Elliott en la línea más accionable del gráfico, porque el usuario
        deja de ver «esto podría ser el techo» y pasa a ver el precio exacto en que deja de ser un
        quizá.
        """
        return (self.price - self.thr_at_extreme if self.kind is PivotKind.HIGH
                else self.price + self.thr_at_extreme)

    def confirmed_at(self, idx: int, ts_ms: int) -> Pivot:
        """Devuelve la versión confirmada. Escritura ÚNICA: reconfirmar es un error de programa."""
        if self.is_confirmed:
            raise ValueError(
                f"Pivot en idx={self.idx} ya estaba confirmado en {self.confirmed_idx}; "
                "la confirmación es de escritura única para que el beam solo pueda CRECER."
            )
        return Pivot(self.idx, self.ts_ms, self.price, self.kind, self.thr_at_extreme, idx, ts_ms)


@dataclass(frozen=True, slots=True)
class RuleVerdict:
    """El resultado de una regla dura, CON su propio precio de invalidación.

    Que cada regla emita su invalidación es la mejor propiedad del diseño: el número más grande de la
    tarjeta de señal lo produce el motor de reglas y no se ensambla aguas abajo, así que no puede
    desviarse de la regla que lo justifica.
    """

    rule: str                       # "R1", "R2b", "R3", "diag_2_4"
    ok: bool
    invalidation_price: float | None
    detail: str = ""
    evaluable: bool = True          # R2 no es evaluable mientras el impulso esté incompleto


# --------------------------------------------------------------------------------------
# Plan y señales
# --------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ExitTemplate:
    """Plantilla de salida compartida por el etiquetador y la interfaz.

    UN SOLO objeto congelado lo consumen a la vez ``labeling/barriers.py`` y las tarjetas de la UI.
    Si el modelo entrenase con barreras de 2R/1R/48 velas mientras la pantalla muestra un objetivo de
    3R con stop dinámico, cada probabilidad mostrada describiría una operación que el usuario no está
    haciendo. Se asevera en la construcción de cada decisión.
    """

    id: str
    tp_r: float                 # objetivo en múltiplos de R
    max_bars: int               # barrera vertical
    trail: str = "none"         # "none" | "chandelier" | "structure"
    breakeven_after_r: float | None = None


@dataclass(frozen=True, slots=True)
class TradePlan:
    """Qué hacer, dónde deja de tener sentido, y por qué."""

    archetype: str              # "w2_long", "w4_long", "wc_long", "diag_exit"
    direction: Direction
    entry_lo: float
    entry_hi: float
    stop: float
    invalidation_price: float
    invalidation_rule: str      # el nombre de la regla que produjo la invalidación
    targets: tuple[float, ...]
    exit_template_id: str
    count_id: str | None = None

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry_mid - self.stop)

    @property
    def entry_mid(self) -> float:
        return (self.entry_lo + self.entry_hi) / 2.0

    def rr_at(self, target: float) -> float:
        risk = self.risk_per_unit
        return abs(target - self.entry_mid) / risk if risk > 0 else 0.0


@dataclass(frozen=True, slots=True)
class Signal:
    """La unidad que fusiona el motor. `source` y las dos marcas de tiempo existen desde v1
    precisamente para que la capa de noticias no obligue a redibujar nada."""

    ts_event_ms: int
    ts_ingest_ms: int
    source: str                 # nombre punteado del proveedor en el REGISTRY
    kind: SourceKind
    direction: Direction
    strength: float             # normalizado a [-1, 1]
    detail: str = ""
    tentative: bool = False     # tocado por el pivote provisional: nunca entra en estadística


@dataclass(frozen=True, slots=True)
class Stat:
    """Un estadístico NUNCA se muestra como número pelado.

    Siempre la terna (valor, n del que sale, estado de la precondición). Un estadístico que devuelve
    un valor tranquilizador a partir de datos insuficientes es peor que no tenerlo: fabrica confianza.
    """

    name: str
    value: float | None
    n: int
    n_required: int
    detail: str = ""

    @property
    def computable(self) -> bool:
        return self.value is not None and self.n >= self.n_required

    @property
    def missing(self) -> int:
        return max(0, self.n_required - self.n)

    def render(self) -> str:
        if self.computable:
            return f"{self.name}: {self.value:.4g} (n={self.n})"
        return f"{self.name}: no computable — faltan {self.missing} (n={self.n}/{self.n_required})"


@dataclass(frozen=True, slots=True)
class Decision:
    """Lo que el motor concluye en una vela cerrada. Es la unidad que se journaliza.

    ``stale`` y ``catching_up`` viajan aquí y no en una variable global porque son propiedades de
    ESTA decisión: una decisión emitida durante una puesta al día tras un corte describe un precio
    que ya pasó, y el usuario tiene derecho a saberlo mirando la propia tarjeta.
    """

    ts_ms: int
    symbol: str
    tf: Timeframe
    verdict: Verdict
    maturity: MaturityLevel
    plan: TradePlan | None = None
    signals: tuple[Signal, ...] = ()
    stats: tuple[Stat, ...] = ()
    reasons: tuple[str, ...] = ()
    stale: bool = False
    catching_up: bool = False

    def __post_init__(self) -> None:
        if self.verdict is Verdict.ACTIONABLE:
            if self.plan is None:
                raise ValueError("Decision ACCIONABLE sin plan: no hay nada que operar")
            if self.catching_up:
                raise ValueError(
                    "Decision ACCIONABLE durante CATCH_UP: la zona de entrada describe un precio "
                    "que ya pasó. Suprime la emisión mientras se reproduce el hueco."
                )
            if self.maturity is MaturityLevel.PRIOR:
                raise ValueError(
                    "Decision ACCIONABLE en nivel PRIOR: sin evidencia, el verdict se topa en WATCH. "
                    "Ver la escalera de madurez."
                )

    @property
    def actionable(self) -> bool:
        return self.verdict is Verdict.ACTIONABLE
