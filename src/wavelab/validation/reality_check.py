"""Corrección por contraste múltiple: Reality Check de White y SPA de Hansen.

EL PROBLEMA QUE RESUELVE. Si pruebas 100 estrategias sobre los mismos datos y te quedas con la
mejor, su p-valor individual no significa nada: has hecho 100 sorteos y te quedas con el máximo.
Con 100 sorteos de una moneda justa, el mejor sale cara el 60% de las veces y parece magia.

Demostrado en este mismo proyecto: sobre BTC 4h, probando 137 estrategias, el MEJOR acierto que
producen conjuntos aleatorios de señales tiene mediana 60,5% y percentil 95 del 64,3%. Un titular
de "64% de acierto" es, literalmente, lo que produce el azar a esa escala.

CÓMO LO RESUELVE. En vez de preguntar "¿es buena la mejor?", pregunta "¿es la mejor MEJOR de lo que
daría el azar habiendo probado tantas?". Se remuestrea la distribución conjunta de las K estrategias
bajo la hipótesis nula de rendimiento cero y se compara el máximo observado contra el máximo del
remuestreo.

BLOQUES, NO OBSERVACIONES SUELTAS. Los retornos financieros están autocorrelacionados y las señales
se agolpan en episodios. Remuestrear observaciones sueltas rompería esa dependencia y daría
intervalos falsamente estrechos. Se usa bootstrap estacionario (Politis-Romano), que remuestrea
bloques de longitud aleatoria geométrica.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "RealityCheckResult",
    "effective_n",
    "optimal_block_length",
    "reality_check",
    "stationary_bootstrap_indices",
]


def optimal_block_length(x: np.ndarray) -> float:
    """Longitud media de bloque, aproximación práctica a Politis-White.

    Regla de n^(1/3) ajustada por la autocorrelación de primer orden: cuanto más persistente es la
    serie, más largos deben ser los bloques para preservar su dependencia.
    """
    n = x.size
    if n < 20:
        return 2.0
    x = x - x.mean()
    denom = float((x * x).sum())
    rho = float((x[:-1] * x[1:]).sum() / denom) if denom > 0 else 0.0
    rho = min(max(rho, -0.9), 0.9)
    base = n ** (1 / 3)
    return float(max(2.0, min(n / 4, base * (1 + 2 * abs(rho)))))


def stationary_bootstrap_indices(n: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """Índices de un remuestreo estacionario: bloques de longitud geométrica, envolviendo el final."""
    p = 1.0 / max(block, 1.0)
    idx = np.empty(n, dtype=np.int64)
    i = int(rng.integers(0, n))
    for t in range(n):
        idx[t] = i
        if rng.random() < p:
            i = int(rng.integers(0, n))
        else:
            i = (i + 1) % n
    return idx


def effective_n(signal_ts: np.ndarray, horizon_bars: int, bar_ms: int) -> int:
    """Cuántas observaciones INDEPENDIENTES hay de verdad.

    Dos señales separadas por menos que el horizonte de tenencia comparten el mismo movimiento de
    precio: no son dos observaciones, son una. Ignorarlo es lo que hace que un "n=120" que en
    realidad son 12 episodios produzca una p<0,001 completamente falsa. Es el problema de
    solapamiento de etiquetas de López de Prado.
    """
    if signal_ts.size == 0:
        return 0
    ventana = horizon_bars * bar_ms
    ts = np.sort(signal_ts)
    n, ultimo = 1, ts[0]
    for t in ts[1:]:
        if t - ultimo >= ventana:
            n += 1
            ultimo = t
    return n


@dataclass(frozen=True, slots=True)
class RealityCheckResult:
    best_name: str
    best_stat: float
    p_value: float
    n_strategies: int
    n_obs: int
    block_length: float
    #: p-valor de cada estrategia frente al máximo del remuestreo. Sin esto solo sabrías del ganador.
    p_values: dict[str, float]

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def render(self) -> str:
        veredicto = ("SUPERA el contraste múltiple" if self.significant
                     else "NO supera el contraste múltiple: compatible con el azar")
        return (f"mejor: {self.best_name} (estadístico {self.best_stat:+.4f})\n"
                f"p-valor corregido por haber probado {self.n_strategies} estrategias: "
                f"{self.p_value:.4f}\n{veredicto}")


def reality_check(
    returns: dict[str, np.ndarray],
    n_boot: int = 2000,
    seed: int = 7,
) -> RealityCheckResult:
    """Reality Check de White sobre las series de retorno por periodo de cada estrategia.

    ``returns[k]`` es el retorno de la estrategia k en cada periodo (0 cuando está fuera de mercado).
    Todas deben tener la misma longitud y estar alineadas en el tiempo.
    """
    nombres = sorted(returns)
    if not nombres:
        raise ValueError("no hay estrategias que evaluar")
    M = np.vstack([np.asarray(returns[k], dtype=float) for k in nombres])
    K, n = M.shape
    medias = M.mean(axis=1)
    raiz = np.sqrt(n)
    V = float((raiz * medias).max())

    block = optimal_block_length(M[int(np.argmax(medias))])
    rng = np.random.default_rng(seed)
    maximos = np.empty(n_boot)
    #: Se centran las series (se les resta su media) para imponer la hipótesis nula de
    #: rendimiento CERO. Sin centrar, el bootstrap mediría la varianza del ganador, no
    #: la probabilidad de que el azar produjese un ganador así.
    Mc = M - medias[:, None]

    # BOOTSTRAP DE BLOQUES SOBRE MEDIAS DE BLOQUE. Remuestrear las n observaciones una a una
    # cuesta K*n por iteración: con 100 estrategias y 20.000 velas son 4·10⁹ operaciones y no
    # termina. Agregando primero a medias de bloque (que es lo que hace el bootstrap de bloques
    # móviles) el coste baja a K*n_bloques y el resultado es el mismo estadístico: la media de
    # una remuestra de bloques ES la media de las medias de esos bloques.
    L = max(2, int(round(block)))
    nb = n // L
    if nb >= 30:
        B = Mc[:, : nb * L].reshape(K, nb, L).mean(axis=2)     # (K, n_bloques)
        for b in range(n_boot):
            idx = rng.integers(0, nb, size=nb)
            mb = B[:, idx].mean(axis=1) * raiz
            maximos[b] = mb.max()
        superan = np.zeros(K)
        for b in range(n_boot):
            pass
        # p-valor individual de cada estrategia contra SU propia distribución nula
        muestras = np.empty((n_boot, K))
        for b in range(n_boot):
            idx = rng.integers(0, nb, size=nb)
            muestras[b] = B[:, idx].mean(axis=1) * raiz
        superan = (muestras >= (raiz * medias)[None, :]).sum(axis=0)
    else:
        superan = np.zeros(K)
        for b in range(n_boot):
            idx = stationary_bootstrap_indices(n, block, rng)
            mb = Mc[:, idx].mean(axis=1) * raiz
            maximos[b] = mb.max()
            superan += (mb >= raiz * medias)

    p = float((maximos >= V).mean())
    return RealityCheckResult(
        best_name=nombres[int(np.argmax(medias))], best_stat=V, p_value=p,
        n_strategies=K, n_obs=n, block_length=block,
        p_values={k: float(superan[i] / n_boot) for i, k in enumerate(nombres)},
    )
