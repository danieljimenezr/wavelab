"""La batería de validación: cinco pruebas que casi ninguna estrategia publicada supera.

ESTO ES EL PRODUCTO. No dice cuándo comprar: dice si lo que crees que funciona es real.

Las cinco, en orden de brutalidad:

1. **Prueba de retraso.** Ejecutar la señal una barra más tarde. Un efecto real pierde algo; una
   fuga de información se DESPLOMA. Es la prueba más barata que existe y detectó un bug propio que
   inflaba el CAGR del 26% al 99%.

2. **Tasa base.** ¿Acierta más que estar en el mercado en un momento cualquiera? En un activo que
   subió un 1.748%, cualquier regla mayoritariamente larga acierta mucho, y eso no es habilidad.

3. **Control aleatorio de misma exposición.** Filtros aleatorios que entran y salen con la misma
   frecuencia y el mismo tiempo dentro. Si la estrategia no bate a la mediana de ellos, lo que
   tiene es beta, no criterio.

4. **Fuera de muestra con purga.** Walk-forward, purgando el horizonte en cada frontera. Cuánto de
   la ventaja sobrevive fuera de la muestra donde se miró.

5. **n efectivo.** Señales que se solapan dentro del horizonte son UNA observación. Un "n=500" que
   en realidad son 12 episodios produce p-valores falsos por órdenes de magnitud.

Y por encima, si se evalúan varias estrategias, el Reality Check de White corrige por haberlas
probado todas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wavelab.validation.reality_check import effective_n

__all__ = ["BatteryResult", "Test", "run_battery"]


@dataclass(frozen=True, slots=True)
class Test:
    id: str
    titulo: str
    passed: bool | None          # None = no concluyente por falta de datos
    valor: float
    referencia: float
    unidad: str
    explicacion: str
    detalle: str = ""

    @property
    def estado(self) -> str:
        return "no_concluyente" if self.passed is None else ("pasa" if self.passed else "falla")


@dataclass(slots=True)
class BatteryResult:
    nombre: str
    n_signals: int
    n_effective: int
    exposure: float
    cagr: float
    sharpe: float
    max_dd: float
    cagr_bh: float
    sharpe_bh: float
    max_dd_bh: float
    tests: list[Test] = field(default_factory=list)
    equity: np.ndarray = field(repr=False, default=None)
    equity_bh: np.ndarray = field(repr=False, default=None)

    @property
    def veredicto(self) -> str:
        fallos = [t for t in self.tests if t.passed is False]
        if any(t.id == "retraso" for t in fallos):
            return "fuga"          # lo más grave: mira al futuro
        if len(fallos) >= 2:
            return "no_sobrevive"
        if fallos:
            return "dudosa"
        return "sobrevive"

    @property
    def resumen(self) -> str:
        return {
            "fuga": "MIRA AL FUTURO. El resultado no es alcanzable en tiempo real.",
            "no_sobrevive": "NO SOBREVIVE. Falla varias pruebas independientes.",
            "dudosa": "DUDOSA. Pasa la mayoría pero falla una prueba relevante.",
            "sobrevive": "SOBREVIVE la batería. No prueba que gane dinero: prueba que "
                         "no es de los errores conocidos.",
        }[self.veredicto]


def _curva(ret: np.ndarray, peso: np.ndarray | None = None, coste_bps: float = 15) -> np.ndarray:
    r = ret if peso is None else ret * peso
    if peso is not None and coste_bps:
        cambios = np.abs(np.diff(np.concatenate([[0.0], peso])))
        r = r - cambios * coste_bps / 10_000
    return np.exp(np.cumsum(r))


def _stats(eq: np.ndarray, n_por_año: float) -> tuple[float, float, float]:
    ret = np.diff(np.log(np.maximum(eq, 1e-12)))
    años = len(eq) / n_por_año
    cagr = eq[-1] ** (1 / años) - 1 if años > 0 and eq[-1] > 0 else -1.0
    sh = ret.mean() / ret.std() * np.sqrt(n_por_año) if ret.std() > 0 else 0.0
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return float(cagr), float(sh), dd


def run_battery(
    close: np.ndarray,
    ts_ms: np.ndarray,
    signal: np.ndarray,
    *,
    nombre: str = "estrategia",
    bar_ms: int,
    horizon_bars: int = 1,
    coste_bps: float = 15,
    n_random: int = 300,
    seed: int = 11,
    folds: int = 5,
) -> BatteryResult:
    """Somete una señal a las cinco pruebas.

    ``signal[i]`` es la posición que se mantiene DESDE el cierre de la vela i hasta el de la i+1.
    Es el único convenio posible sin mirar al futuro, y es donde casi todo el mundo se equivoca.
    """
    c = np.asarray(close, dtype=float)
    sig = np.asarray(signal, dtype=float)[: len(c) - 1]
    r = np.diff(np.log(c))                       # r[i]: del cierre i al i+1
    n = len(r)
    rng = np.random.default_rng(seed)
    por_año = 365.25 * 24 * 3600 * 1000 / bar_ms

    eq = _curva(r, sig, coste_bps)
    eq_bh = _curva(r)
    cagr, sh, dd = _stats(eq, por_año)
    cagr_bh, sh_bh, dd_bh = _stats(eq_bh, por_año)
    activo = sig != 0
    exposure = float(np.abs(sig).mean())

    tests: list[Test] = []

    # ---- 1. retraso de una barra --------------------------------------------------------------
    sig_lag = np.concatenate([[0.0], sig[:-1]])
    cagr_lag, _, _ = _stats(_curva(r, sig_lag, coste_bps), por_año)
    # Un efecto real pierde algo al ejecutarse tarde; una fuga se desploma. El umbral es que
    # conserve al menos la mitad de su exceso sobre comprar y mantener.
    exc, exc_lag = cagr - cagr_bh, cagr_lag - cagr_bh
    retiene = (exc_lag / exc) if abs(exc) > 1e-9 else 1.0
    tests.append(Test(
        "retraso", "¿Sobrevive a ejecutarse una barra más tarde?",
        bool(retiene >= 0.5) if abs(exc) > 1e-9 else True,
        float(cagr_lag), float(cagr), "CAGR",
        "Si el resultado se desploma al retrasar la señal una sola barra, la estrategia está "
        "usando información que en tiempo real no tendrías. Es la prueba más barata y la que más "
        "estrategias tumba.",
        f"CAGR {cagr*100:.1f}% → {cagr_lag*100:.1f}% al retrasar (retiene {retiene*100:.0f}% "
        f"del exceso sobre comprar y mantener)"))

    # ---- 2. tasa base -------------------------------------------------------------------------
    fwd = np.full(n, np.nan)
    if n > horizon_bars:
        fwd[:-horizon_bars] = np.log(c[horizon_bars + 1: n + 1] / c[1: n - horizon_bars + 1])
    val = ~np.isnan(fwd)
    base = float(fwd[val].mean()) if val.any() else 0.0
    m = val & activo
    prop = float((sig[m] * fwd[m]).mean()) if m.sum() >= 20 else float("nan")
    hay_prop = not np.isnan(prop)      # más claro que el idiomático `prop == prop`
    tests.append(Test(
        "tasa_base", "¿Acierta más que un momento cualquiera?",
        bool(prop > base) if hay_prop else None,
        prop * 100 if hay_prop else 0.0, base * 100, "% por periodo",
        "En un activo que subió un 1.748%, cualquier regla mayoritariamente larga parece acertar. "
        "Lo que cuenta es si acierta MÁS que estar dentro en un instante al azar.",
        f"estrategia {prop*100:+.3f}% vs base {base*100:+.3f}% por periodo"
        if hay_prop else "muy pocas señales"))

    # ---- 3. control aleatorio de misma exposición ---------------------------------------------
    cambios_reales = int(np.abs(np.diff(np.concatenate([[0.0], sig]))).sum())
    dur = max(2, int(n / max(cambios_reales, 1)))
    aleatorios = np.empty(n_random)
    for k in range(n_random):
        w = np.zeros(n); i = 0
        while i < n:
            L = int(rng.integers(max(2, dur // 2), max(3, dur * 2)))
            w[i:i + L] = float(rng.random() < exposure)
            i += L
        aleatorios[k] = _stats(_curva(r, w, coste_bps), por_año)[0]
    pct = float((aleatorios < cagr).mean())
    tests.append(Test(
        "control_aleatorio", "¿Bate a filtros ALEATORIOS de la misma exposición?",
        bool(pct >= 0.95), pct * 100, 95.0, "percentil",
        "Se generan cientos de filtros que entran y salen al azar con la misma frecuencia y el "
        "mismo tiempo dentro del mercado. Si tu estrategia no está claramente por encima de ellos, "
        "lo que tienes es exposición, no criterio.",
        f"percentil {pct*100:.0f} frente a {n_random} filtros aleatorios "
        f"(mediana {np.median(aleatorios)*100:.1f}% CAGR)"))

    # ---- 4. fuera de muestra con purga --------------------------------------------------------
    paso = n // folds
    oos = np.zeros(n, dtype=bool)
    for k in range(1, folds):
        oos[k * paso + horizon_bars: min((k + 1) * paso, n)] = True
    if oos.sum() > 100:
        cagr_oos, _, _ = _stats(_curva(r[oos], sig[oos], coste_bps), por_año)
        cagr_bh_oos, _, _ = _stats(_curva(r[oos]), por_año)
        exc_oos = cagr_oos - cagr_bh_oos
        ret_oos = (exc_oos / exc) if abs(exc) > 1e-9 else 1.0
        # Si el exceso EN MUESTRA ya es negativo, "retiene el 122%" significa que pierde de forma
        # consistente, no que aguante. La prueba solo tiene sentido sobre una ventaja positiva:
        # con exceso negativo no hay nada que sobrevivir y se marca como no concluyente.
        if exc <= 0:
            paso_oos = None
        else:
            paso_oos = bool(ret_oos >= 0.5)
        tests.append(Test(
            "fuera_de_muestra", "¿Aguanta fuera de la muestra donde se miró?",
            paso_oos, float(cagr_oos), float(cagr_bh_oos), "CAGR",
            "Se parte el histórico en tramos y se purga el horizonte en cada frontera, para que "
            "ninguna operación abierta cruce de un tramo a otro. La ventaja tiene que sobrevivir "
            "donde no se miró.",
            (f"exceso en muestra {exc*100:+.1f}% → fuera {exc_oos*100:+.1f}% "
             f"(retiene {ret_oos*100:.0f}%)") if exc > 0 else
            (f"no aplica: el exceso ya es negativo en muestra ({exc*100:+.1f}%). "
             "No hay ventaja que pueda sobrevivir fuera de ella.")))
    else:
        tests.append(Test("fuera_de_muestra", "¿Aguanta fuera de la muestra?", None, 0, 0, "CAGR",
                          "Serie demasiado corta para partirla.", ""))

    # ---- 5. n efectivo ------------------------------------------------------------------------
    # El horizonte relevante NO es el parámetro `horizon_bars` sino cuánto dura de hecho cada
    # posición: si la estrategia se mantiene dentro 40 barras seguidas, esas 40 barras son UNA
    # apuesta, no cuarenta. Se mide la longitud media de racha de la propia señal.
    cambios_sig = np.abs(np.diff(np.concatenate([[0.0], np.sign(sig)]))) > 0
    n_rachas = max(1, int(cambios_sig.sum()))
    dur_media = max(horizon_bars, int(round(activo.sum() / n_rachas)) if n_rachas else horizon_bars)
    n_ef = effective_n(np.asarray(ts_ms)[: n][activo], dur_media, bar_ms)
    tests.append(Test(
        "n_efectivo", "¿Hay observaciones INDEPENDIENTES suficientes?",
        bool(n_ef >= 30), float(n_ef), 30.0, "episodios",
        "Señales que se solapan dentro del horizonte de tenencia son la MISMA observación. Un "
        "n=500 que en realidad son 12 episodios produce p-valores falsos por órdenes de magnitud. "
        "Es el error de solapamiento de etiquetas de López de Prado.",
        f"{int(activo.sum())} barras con posición en {n_rachas} rachas "
        f"(duración media {dur_media} barras) → {n_ef} episodios independientes"))

    return BatteryResult(
        nombre=nombre, n_signals=int(activo.sum()), n_effective=n_ef, exposure=exposure,
        cagr=cagr, sharpe=sh, max_dd=dd, cagr_bh=cagr_bh, sharpe_bh=sh_bh, max_dd_bh=dd_bh,
        tests=tests, equity=eq, equity_bh=eq_bh)
