"""Familia `structure`: la estructura de mercado como fuente de ventaja.

REGISTRO PREVIO. Nada de lo que hay aquí se ha ejecutado contra datos. Los parámetros son los
convencionales de la literatura —Donchian 20 y 55 (sistemas 1 y 2 de las Tortugas), fractal de
Williams de 5 velas (k=2 a cada lado), ATR 14, media de volumen 20— y el horizonte de las
hipótesis de evento es 10 velas para TODAS, elegido una sola vez y aplicado sin excepción para que
no haya rejilla encubierta disfrazada de "variantes".

MECANISMO COMÚN DE LA FAMILIA. Los máximos y mínimos previos son los pocos niveles que todos los
participantes ven igual, sin ambigüedad y sin parámetros. Ahí es donde el que va largo pone el
stop, donde el que espera confirmación pone la orden de ruptura y donde el creador de mercado sabe
que hay volumen ejecutable. Es decir: la estructura no predice, pero SÍ concentra órdenes, y una
concentración de órdenes es lo único que puede mover un precio. Cada hipótesis de abajo apuesta
sobre qué pasa cuando el precio llega a esa liquidez: o la ruptura arrastra a los que estaban al
otro lado (continuación) o la liquidez se consume de golpe y el precio vuelve dentro (barrida).

Las dos familias de apuestas NO pueden ser ciertas a la vez en el mismo régimen. El conflicto es
deliberado: `structure.donchian_break_20` y `structure.sweep_reversal_20` operan sobre el mismo
nivel en direcciones opuestas, igual que `structure.bos_swing` y `structure.range_fade_swing`. Si
las dos "funcionan" a la vez sin que el régimen las separe, la explicación más probable no es que
haya dos ventajas sino que el estimador está midiendo ruido, y eso es informativo.

CAUSALIDAD. Todo nivel de referencia proviene de velas ESTRICTAMENTE anteriores a i (`_shift1`
sobre las ventanas de talib, que incluyen la vela actual) y todo pivote se coloca en la vela en la
que queda CONFIRMADO, no en la que ocurrió el extremo. No se usa `find_peaks` ni ningún estadístico
definido contra el array entero.
"""

from __future__ import annotations

import numpy as np
import talib

from wavelab.hypotheses.base import Hypothesis, Series, register

# Horizonte único para las hipótesis de evento (barrida, retest, inside bar, fade de rango).
# Un solo número para todas: si cada una llevara el suyo, esto sería una rejilla.
_HOLD = 10

# Semiamplitud del fractal de Williams: 5 velas, 2 a cada lado. La confirmación llega 2 velas
# después del extremo, y ese retardo se respeta de forma explícita.
_K = 2

# Series más cortas que esto devuelven todo ceros. Coincide con `min_warmup` a propósito: así el
# umbral del guarda nunca cae DENTRO de la región que el backtest evalúa, y la señal en i es
# idéntica se calcule sobre el prefijo [0..i] o sobre la serie entera.
_MIN_BARS = 200


# --------------------------------------------------------------------------------------------
# Utilidades. Todas causales: la posición i solo mira posiciones <= i.
# --------------------------------------------------------------------------------------------

def _f(x: np.ndarray) -> np.ndarray:
    """talib exige float64 contiguo."""
    return np.ascontiguousarray(x, dtype=np.float64)


def _shift1(x: np.ndarray) -> np.ndarray:
    """out[i] = x[i-1], out[0] = NaN. Desplaza hacia el FUTURO (nunca np.roll negativo)."""
    out = np.empty_like(x)
    out[0] = np.nan
    out[1:] = x[:-1]
    return out


# `_shiftk` vivía aquí. Se retira con la hipótesis duplicada que era su único uso (véase la nota
# del punto 11): dejar utilidades muertas invita a que la siguiente hipótesis las reutilice sin
# volver a comprobar su causalidad.


def _prev_max(high: np.ndarray, n: int) -> np.ndarray:
    """Máximo de las n velas ANTERIORES a i. talib.MAX incluye la vela i, así que se desplaza."""
    return _shift1(talib.MAX(_f(high), n))


def _prev_min(low: np.ndarray, n: int) -> np.ndarray:
    """Mínimo de las n velas ANTERIORES a i."""
    return _shift1(talib.MIN(_f(low), n))


def _gt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a > b, False donde cualquiera sea NaN. Comprobación explícita: nan_to_num con un centinela
    inventaría rupturas durante el calentamiento."""
    out = np.zeros(a.shape, dtype=bool)
    ok = ~(np.isnan(a) | np.isnan(b))
    np.greater(a, b, out=out, where=ok)
    return out


def _lt(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a < b, False donde cualquiera sea NaN."""
    out = np.zeros(a.shape, dtype=bool)
    ok = ~(np.isnan(a) | np.isnan(b))
    np.less(a, b, out=out, where=ok)
    return out


def _fresh(cond: np.ndarray) -> np.ndarray:
    """Solo la vela en que `cond` pasa de falsa a verdadera. Necesario para las hipótesis de
    evento cuya condición es un ESTADO persistente ("el precio está por encima del pivote"): sin
    esto el evento se rearma en cada vela y una hipótesis de rechazo puntual se convierte, sin
    quererlo, en una posición contratendencia permanente. Mira i e i-1, nada más."""
    out = cond.copy()
    out[1:] &= ~cond[:-1]
    return out


def _ffill(x: np.ndarray) -> np.ndarray:
    """Arrastra hacia adelante el último valor no-NaN. `maximum.accumulate` es un prefijo: el
    valor en i solo puede venir de <= i."""
    idx = np.where(~np.isnan(x), np.arange(x.size), 0)
    np.maximum.accumulate(idx, out=idx)
    return x[idx]


def _events(up: np.ndarray, dn: np.ndarray) -> np.ndarray:
    """Codifica eventos en {-1,0,+1}. Si los dos disparan en la misma vela la situación es
    ambigua y se emite 0 (no hay información), nunca un desempate arbitrario."""
    both = up & dn
    ev = np.zeros(up.size, dtype=np.int8)
    ev[up & ~both] = 1
    ev[dn & ~both] = -1
    return ev


def _hold(up: np.ndarray, dn: np.ndarray) -> np.ndarray:
    """Mantiene el último evento hasta que aparezca el contrario (siempre en mercado tras el
    primero)."""
    ev = _events(up, dn)
    idx = np.where(ev != 0, np.arange(ev.size), 0)
    np.maximum.accumulate(idx, out=idx)
    out = ev[idx]
    out[idx == 0] = ev[0]          # antes del primer evento no hay posición
    return out


def _hold_n(up: np.ndarray, dn: np.ndarray, bars: int = _HOLD) -> np.ndarray:
    """Mantiene el último evento `bars` velas y luego se sale. Para hipótesis de evento, donde el
    efecto —si existe— es transitorio por construcción."""
    ev = _events(up, dn)
    n = ev.size
    idx = np.where(ev != 0, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    out = ev[idx].copy()
    age = np.arange(n) - idx
    out[age >= bars] = 0
    out[ev[idx] == 0] = 0
    return out


def _pivots(s: Series, k: int = _K) -> tuple[np.ndarray, np.ndarray]:
    """Fractales de Williams COLOCADOS EN LA VELA QUE LOS CONFIRMA, no en la del extremo.

    Un máximo fractal en la vela j solo se sabe en j+k, cuando ya han cerrado las k velas
    posteriores. Devolver el pivote en j sería mirar al futuro: es exactamente el error que hace
    que las estrategias de "estructura" parezcan rentables en papel. Aquí `ph[i]` guarda el precio
    del máximo confirmado EN i (extremo ocurrido en i-k) y NaN si en i no se confirma nada.
    """
    n = len(s)
    w = 2 * k + 1
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    if n <= w:
        return ph, pl

    hi, lo = _f(s.high), _f(s.low)
    rmax = talib.MAX(hi, w)      # máximo de [i-w+1, i]: todo dato <= i
    rmin = talib.MIN(lo, w)

    i = np.arange(w - 1, n)      # primera ventana completa en adelante
    c = i - k                    # vela central: el extremo candidato

    ok_h = ~(np.isnan(rmax[i]) | np.isnan(hi[c]))
    is_h = np.zeros(i.size, dtype=bool)
    np.equal(hi[c], rmax[i], out=is_h, where=ok_h)
    ph[i[is_h]] = hi[c[is_h]]

    ok_l = ~(np.isnan(rmin[i]) | np.isnan(lo[c]))
    is_l = np.zeros(i.size, dtype=bool)
    np.equal(lo[c], rmin[i], out=is_l, where=ok_l)
    pl[i[is_l]] = lo[c[is_l]]

    return ph, pl


def _pivot_levels(s: Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(último máximo confirmado, penúltimo, último mínimo confirmado, penúltimo) en cada vela."""
    ph, pl = _pivots(s)
    ph1, pl1 = _ffill(ph), _ffill(pl)
    # En una vela de confirmación, el "anterior" es el arrastre hasta la vela previa.
    ph2 = _ffill(np.where(~np.isnan(ph), _shift1(ph1), np.nan))
    pl2 = _ffill(np.where(~np.isnan(pl), _shift1(pl1), np.nan))
    return ph1, ph2, pl1, pl2


def _structure_state(s: Series) -> np.ndarray:
    """+1 si la última estructura confirmada es máximos y mínimos crecientes, -1 si decrecientes,
    0 si está mezclada. Base compartida por varias hipótesis."""
    ph1, ph2, pl1, pl2 = _pivot_levels(s)
    up = _gt(ph1, ph2) & _gt(pl1, pl2)
    dn = _lt(ph1, ph2) & _lt(pl1, pl2)
    st = np.zeros(len(s), dtype=np.int8)
    st[up] = 1
    st[dn] = -1
    return st


# --------------------------------------------------------------------------------------------
# 1. Ruptura de Donchian 20, siempre en mercado
# --------------------------------------------------------------------------------------------

def _donchian_break_20(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c = _f(s.close)
    return _hold(_gt(c, _prev_max(s.high, 20)), _lt(c, _prev_min(s.low, 20)))


register(Hypothesis(
    name="structure.donchian_break_20",
    family="structure",
    rationale="El máximo de las 20 velas previas es el nivel donde converge la mayor cantidad de "
              "órdenes en reposo: stops de los cortos, entradas en stop de los que esperan "
              "confirmación y coberturas de los vendedores de opciones. Cuando el precio lo "
              "supera, esas órdenes se ejecutan como mercado y consumen el libro en la misma "
              "dirección, lo que empuja el precio más allá y obliga a más cierres. Si esa cascada "
              "existe en BTC, un simple stop-and-reverse sobre el canal debería capturarla sin "
              "más maquinaria.",
    prior="Esperamos ventaja positiva y concentrada en pocos episodios de cola derecha (pocas "
          "operaciones aportan casi todo el resultado), con tasa de acierto BAJA, por debajo del "
          "45%. Esperamos que falle —ventaja negativa, no nula— en mercados laterales prolongados, "
          "donde cada ruptura se revierte y el sistema paga el diferencial en ambos sentidos. Si "
          "sale ventaja positiva CON tasa de acierto alta, la hipótesis está mal: eso indicaría "
          "un fallo de datos o de ejecución, no un efecto de estructura.",
    fn=_donchian_break_20,
    params={"canal": 20, "modo": "stop-and-reverse"},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 2. Ruptura de Donchian 55 con salida por Donchian 20 (con estado plano)
# --------------------------------------------------------------------------------------------

def _donchian_55_exit_20(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c = _f(s.close)
    hi55, lo55 = _prev_max(s.high, 55), _prev_min(s.low, 55)
    hi20, lo20 = _prev_max(s.high, 20), _prev_min(s.low, 20)
    out = np.zeros(n, dtype=np.int8)
    pos = 0
    for i in range(n):
        # Bucle explícito: el estado en i depende del estado en i-1 y de datos de i. Nada más.
        if pos == 0:
            if not np.isnan(hi55[i]) and c[i] > hi55[i]:
                pos = 1
            elif not np.isnan(lo55[i]) and c[i] < lo55[i]:
                pos = -1
        elif pos == 1:
            if not np.isnan(lo20[i]) and c[i] < lo20[i]:
                pos = 0          # se sale y se queda plano: reentrar el mismo cierre es ambiguo
        else:
            if not np.isnan(hi20[i]) and c[i] > hi20[i]:
                pos = 0
        out[i] = pos
    return out


register(Hypothesis(
    name="structure.donchian_55_exit_20",
    family="structure",
    rationale="No es la misma hipótesis que el canal de 20 con otro número, y por eso se registra "
              "aparte: la afirmación aquí no es sobre el nivel de entrada sino sobre la ASIMETRÍA "
              "entre entrar y salir. Se entra solo en la ruptura de 55 velas, la que exige un "
              "desequilibrio que ya ha absorbido toda la liquidez del trimestre, y se sale con un "
              "canal más corto de 20 y con estado PLANO, sin reversión. La apuesta es que el "
              "participante que provoca la cascada (liquidaciones apalancadas, reequilibrio de "
              "tesorerías) tarda semanas en agotarse, mientras que su desaparición se nota en "
              "días; salir plano en lugar de darse la vuelta evita pagar el lado equivocado de "
              "una consolidación.",
    prior="Esperamos ventaja positiva y MENOR número de operaciones que en `donchian_break_20`, "
          "con menor pérdida máxima acumulada gracias al estado plano. Esperamos que falle en "
          "timeframes bajos (15m, 1h), donde 55 velas son unas pocas horas y la ruptura no "
          "identifica ningún flujo estructural: ahí el resultado debería ser indistinguible de "
          "cero o negativo por costes. Si la ventaja fuera igual o mayor en 15m que en 1d, la "
          "explicación mecánica de arriba es falsa.",
    fn=_donchian_55_exit_20,
    params={"entrada": 55, "salida": 20, "modo": "largo/corto con estado plano"},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 3. BOS: ruptura del último pivote CONFIRMADO
# --------------------------------------------------------------------------------------------

def _bos_swing(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c = _f(s.close)
    ph1, _, pl1, _ = _pivot_levels(s)
    return _hold(_gt(c, ph1), _lt(c, pl1))


register(Hypothesis(
    name="structure.bos_swing",
    family="structure",
    rationale="La versión estructural de la ruptura: en vez de un canal de longitud fija, el "
              "nivel es el último máximo o mínimo de oscilación CONFIRMADO por un fractal de 5 "
              "velas. La diferencia mecánica importa: un canal de 20 se mueve por el mero paso "
              "del tiempo aunque no haya pasado nada, mientras que un pivote solo cambia cuando "
              "el mercado ha girado de verdad y ha dejado a alguien atrapado en el extremo. Ahí "
              "es donde están los stops reales, porque es el único punto que invalida la tesis "
              "del que compró en el impulso anterior.",
    prior="Esperamos ventaja positiva similar o algo inferior a la del canal de 20 pero con "
          "MENOS operaciones, porque los niveles son más estables. Esperamos que falle cuando la "
          "volatilidad se comprime: con velas pequeñas el fractal confirma pivotes triviales, el "
          "nivel queda a un tick del precio y la señal cambia de signo constantemente. La "
          "hipótesis queda refutada si la ventaja por operación no supera a la del canal de 20, "
          "porque entonces la parte 'estructural' no aporta nada sobre un máximo móvil.",
    fn=_bos_swing,
    params={"fractal_k": _K, "ventana_fractal": 2 * _K + 1, "retardo_confirmacion": _K},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 4. CHoCH: cambio de carácter
# --------------------------------------------------------------------------------------------

def _choch(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c = _f(s.close)
    ph1, ph2, pl1, pl2 = _pivot_levels(s)
    alcista = _gt(ph1, ph2) & _gt(pl1, pl2)      # secuencia creciente vigente
    bajista = _lt(ph1, ph2) & _lt(pl1, pl2)      # secuencia decreciente vigente
    # Cambio de carácter: primer mínimo perdido tras una secuencia creciente (y su simétrico).
    choch_dn = alcista & _lt(c, pl1)
    choch_up = bajista & _gt(c, ph1)
    return _hold(choch_up, choch_dn)


register(Hypothesis(
    name="structure.choch",
    family="structure",
    rationale="El cambio de carácter es la primera vez que la secuencia se rompe: veníamos "
              "haciendo máximos y mínimos crecientes y de pronto se pierde el último mínimo. El "
              "mecanismo no es la ruptura en sí, sino QUIÉN está al otro lado: en una secuencia "
              "creciente, cada mínimo es donde compró la última tanda de tendencia, con el stop "
              "justo debajo. Perder ese mínimo significa que la demanda que sostenía el impulso "
              "ya no aparece y convierte a los compradores recientes en vendedores forzados. La "
              "diferencia con `bos_swing` es que aquí solo se opera la ruptura CONTRA la "
              "estructura vigente, no a favor.",
    prior="Esperamos ventaja positiva pero MENOR que la de continuación, y muy dependiente del "
          "timeframe: creíble en 4h y 1d, dudosa en 15m. Esperamos que falle claramente en "
          "tendencias fuertes, donde la mayoría de los CHoCH son sacudidas dentro del impulso y "
          "el precio reanuda: ahí debe dar ventaja NEGATIVA, y si no la da es que la señal no "
          "está capturando lo que creemos. También queda refutada si su resultado no se distingue "
          "del de `bos_swing` con signo cambiado: eso significaría que solo estamos midiendo "
          "autocorrelación del precio, no estructura.",
    fn=_choch,
    params={"fractal_k": _K, "contexto": "2 pivotes por lado"},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 5. Régimen estructural HH/HL vs LH/LL
# --------------------------------------------------------------------------------------------

def _swing_trend_state(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    return _structure_state(s)


register(Hypothesis(
    name="structure.swing_trend_state",
    family="structure",
    rationale="Hipótesis mínima de la familia, y a propósito la más aburrida: no hay evento ni "
              "ruptura, solo la clasificación clásica de Dow. Se está largo mientras los dos "
              "últimos pivotes confirmados sean máximo creciente Y mínimo creciente, corto en el "
              "caso simétrico, y fuera cuando la estructura está mezclada. Sirve de PATRÓN DE "
              "REFERENCIA: si las hipótesis de evento no baten a esto, lo que están capturando es "
              "la tendencia de fondo y no el mecanismo de liquidez que dicen explotar. El estar "
              "fuera en estructura mezclada es la parte que aporta, porque ahí es donde el "
              "seguimiento de tendencia pierde dinero.",
    prior="Esperamos ventaja positiva pequeña, dominada por el sesgo alcista histórico de BTC, "
          "con un tiempo fuera de mercado sustancial (más del 25% de las velas). Esperamos que la "
          "parte corta tenga ventaja negativa o nula por sí sola: si los cortos aportan tanto "
          "como los largos, hay que sospechar del período de datos, no celebrarlo. Queda refutada "
          "como referencia útil si su resultado es indistinguible de estar comprado y quieto.",
    fn=_swing_trend_state,
    params={"fractal_k": _K},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 6. Barrida de liquidez: ruptura falsa del canal de 20
# --------------------------------------------------------------------------------------------

def _sweep_reversal_20(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c, hi, lo = _f(s.close), _f(s.high), _f(s.low)
    hh, ll = _prev_max(s.high, 20), _prev_min(s.low, 20)
    corto = _gt(hi, hh) & _lt(c, hh)     # perfora el máximo previo y cierra por debajo
    largo = _lt(lo, ll) & _gt(c, ll)     # perfora el mínimo previo y cierra por encima
    return _hold_n(largo, corto)


register(Hypothesis(
    name="structure.sweep_reversal_20",
    family="structure",
    rationale="La cara opuesta de `donchian_break_20`, sobre el MISMO nivel y a propósito. Si la "
              "liquidez está amontonada justo detrás del extremo de 20 velas, hay un participante "
              "con incentivo directo a ir a buscarla: el que necesita ejecutar tamaño y solo "
              "puede hacerlo contra los stops de los demás. La firma observable es una vela que "
              "perfora el nivel en máximos pero cierra DENTRO del rango: se ejecutaron los stops, "
              "no había continuación detrás y el precio vuelve. Cierre dentro es la condición "
              "clave, porque distingue absorción de ruptura genuina.",
    prior="Esperamos ventaja positiva de vida corta —concentrada en las primeras velas del "
          "horizonte de 10— con tasa de acierto ALTA y ganancia media pequeña, el perfil inverso "
          "al de la ruptura. Esperamos que falle en tendencias fuertes, donde la falsa ruptura "
          "solo es una pausa antes de continuar, y que falle en 1d, donde una vela diaria agrupa "
          "demasiados eventos como para que el cierre signifique 'absorción'. Si tanto esta "
          "hipótesis como `donchian_break_20` salen positivas sobre el mismo período y régimen, "
          "hay que sospechar del estimador antes que creerse las dos.",
    fn=_sweep_reversal_20,
    params={"canal": 20, "horizonte": _HOLD},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 7. Retest del nivel roto
# --------------------------------------------------------------------------------------------

def _retest_hold(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c, hi, lo = _f(s.close), _f(s.high), _f(s.low)
    hh, ll = _prev_max(s.high, 20), _prev_min(s.low, 20)
    ev_up = np.zeros(n, dtype=bool)
    ev_dn = np.zeros(n, dtype=bool)

    lvl_u = np.nan
    ttl_u = 0
    toc_u = False
    lvl_d = np.nan
    ttl_d = 0
    toc_d = False

    for i in range(n):
        # Todo lo que se lee aquí es de la vela i o de estado acumulado de velas anteriores.
        rot_u = (not np.isnan(hh[i])) and c[i] > hh[i]
        rot_d = (not np.isnan(ll[i])) and c[i] < ll[i]

        if rot_u:
            lvl_u, ttl_u, toc_u = hh[i], _HOLD, False
        elif ttl_u > 0:
            ttl_u -= 1
            if lo[i] <= lvl_u:
                toc_u = True
            if toc_u and c[i] > lvl_u:
                ev_up[i] = True
                ttl_u = 0

        if rot_d:
            lvl_d, ttl_d, toc_d = ll[i], _HOLD, False
        elif ttl_d > 0:
            ttl_d -= 1
            if hi[i] >= lvl_d:
                toc_d = True
            if toc_d and c[i] < lvl_d:
                ev_dn[i] = True
                ttl_d = 0

    return _hold_n(ev_up, ev_dn)


register(Hypothesis(
    name="structure.retest_hold",
    family="structure",
    rationale="El nivel roto cambia de papel, y hay una razón de flujo concreta para ello: los "
              "que vendieron en la resistencia y quedaron atrapados intentan salir en el punto de "
              "entrada cuando el precio vuelve, y los que se perdieron la ruptura tienen ahí su "
              "referencia para comprar sin perseguir. Ambos actúan en el mismo sitio y en la "
              "misma dirección. La condición operativa es exigente a propósito: hay que romper, "
              "VOLVER a tocar el nivel dentro de 10 velas y cerrar otra vez del lado bueno. Si "
              "solo importara la ruptura, esto no aportaría nada sobre la hipótesis 1.",
    prior="Esperamos MENOS operaciones que la ruptura simple y mejor resultado por operación, "
          "porque el retest filtra las rupturas sin comprador detrás. Esperamos que falle en las "
          "rupturas más violentas —justo las que más aportan a la hipótesis 1— porque esas nunca "
          "vuelven a tocar el nivel y el filtro las descarta: si el efecto de ruptura es de cola "
          "derecha pura, este filtro debería DESTRUIR la ventaja en lugar de mejorarla. Ese es el "
          "contraste que interesa, y las dos conclusiones son publicables.",
    fn=_retest_hold,
    params={"canal": 20, "ventana_retest": _HOLD, "horizonte": _HOLD},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 8. Ruptura tras compresión del canal
# --------------------------------------------------------------------------------------------

def _compression_break(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c = _f(s.close)
    hh, ll = _prev_max(s.high, 20), _prev_min(s.low, 20)
    ancho = hh - ll
    min_ancho = talib.MIN(_f(np.nan_to_num(ancho, nan=np.inf)), 60)
    # `ancho` en i ya solo depende de velas < i; la ventana de 60 de talib termina en i.
    comprimido = np.zeros(n, dtype=bool)
    ok = ~(np.isnan(ancho) | np.isnan(min_ancho) | np.isinf(min_ancho))
    np.less_equal(ancho, min_ancho, out=comprimido, where=ok)
    return _hold(comprimido & _gt(c, hh), comprimido & _lt(c, ll))


register(Hypothesis(
    name="structure.compression_break",
    family="structure",
    rationale="Una ruptura solo importa si antes había desacuerdo contenido. Cuando el ancho del "
              "canal de 20 cae a su mínimo de las últimas 60 velas, compradores y vendedores han "
              "llegado a un equilibrio estrecho y ambos bandos han ido acumulando stops muy "
              "cerca, a pocos ticks del precio y unos de otros. Salir de esa zona ejecuta las dos "
              "carteras de stops en cadena, que es el único momento en que una ruptura arrastra "
              "un volumen desproporcionado respecto al tamaño del movimiento previo. La "
              "compresión no predice la dirección: solo dice que el movimiento que la resuelva "
              "será desproporcionado.",
    prior="Esperamos ventaja por operación SUPERIOR a la de `donchian_break_20` con muchas menos "
          "operaciones, y esa comparación es el contraste real de la hipótesis: si el filtro de "
          "compresión no mejora nada, la idea de la doble cartera de stops es falsa y solo "
          "estábamos operando menos. Esperamos que falle tras caídas violentas, donde la "
          "compresión aparece por agotamiento y no por equilibrio, y en 1d, donde 60 velas son "
          "dos meses y el mínimo de ancho llega tarde.",
    fn=_compression_break,
    params={"canal": 20, "ventana_compresion": 60},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 9. Ruptura confirmada por volumen
# --------------------------------------------------------------------------------------------

def _volume_confirmed_break(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c, v = _f(s.close), _f(s.volume)
    hh, ll = _prev_max(s.high, 20), _prev_min(s.low, 20)
    vmed = talib.SMA(v, 20)                    # incluye la vela i: dato disponible en i
    fuerte = _gt(v, vmed)
    return _hold(_gt(c, hh) & fuerte, _lt(c, ll) & fuerte)


register(Hypothesis(
    name="structure.volume_confirmed_break",
    family="structure",
    rationale="Si el mecanismo de la ruptura es la ejecución en cascada de órdenes acumuladas, "
              "entonces tiene una huella obligatoria: volumen. Una ruptura con volumen por debajo "
              "de su media de 20 significa que no había nadie esperando en ese nivel, que el "
              "precio llegó ahí por deriva y no por ejecución, y que por tanto no hay ningún "
              "flujo forzado que continúe el movimiento. El volumen no se usa aquí como "
              "indicador, sino como VERIFICACIÓN del mecanismo que las hipótesis 1 y 8 dan por "
              "supuesto.",
    prior="Esperamos ventaja positiva y, sobre todo, ventaja por operación MAYOR que la de "
          "`donchian_break_20` sin filtro. Si el filtro de volumen no mejora nada, la explicación "
          "por cascada de órdenes queda seriamente debilitada para todas las hipótesis de "
          "ruptura de esta familia, y eso es un resultado más valioso que la hipótesis en sí. "
          "Esperamos que falle en 1d, donde el volumen diario de BTC está dominado por el ciclo "
          "semanal y por el reparto entre exchanges, no por el evento de ruptura.",
    fn=_volume_confirmed_break,
    params={"canal": 20, "media_volumen": 20},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 10. Ruptura con colchón de ATR
# --------------------------------------------------------------------------------------------

def _atr_buffered_break(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c = _f(s.close)
    hh, ll = _prev_max(s.high, 20), _prev_min(s.low, 20)
    atr = talib.ATR(_f(s.high), _f(s.low), c, 14)
    return _hold(_gt(c, hh + 0.5 * atr), _lt(c, ll - 0.5 * atr))


register(Hypothesis(
    name="structure.atr_buffered_break",
    family="structure",
    rationale="Complemento directo de `sweep_reversal_20` y contraste explícito de la hipótesis 1. "
              "Si es cierto que los stops se amontonan justo detrás del nivel y que alguien los "
              "va a buscar, entonces la ruptura MARGINAL —la que asoma unos ticks por encima— es "
              "sistemáticamente la peor, porque es exactamente la que produce quien quiere "
              "vender. Exigir que el cierre supere el nivel por medio ATR de 14 velas descarta "
              "esa zona sin cambiar nada más de la regla, y hace que el colchón se adapte solo a "
              "la volatilidad en lugar de fijar un porcentaje arbitrario.",
    prior="Esperamos ventaja positiva y mayor por operación que la ruptura sin colchón, y este es "
          "el contraste que importa. Como `sweep_reversal_20` afirma que la zona descartada tiene "
          "ventaja NEGATIVA, las dos hipótesis deben ser coherentes entre sí: si la barrida "
          "resulta rentable pero el colchón no mejora la ruptura, o al revés, alguna de las dos "
          "está midiendo otra cosa. Esperamos que el colchón perjudique en 1d, donde media ATR "
          "diaria es un movimiento enorme y entrar tan tarde regala la mayor parte del "
          "desplazamiento.",
    fn=_atr_buffered_break,
    params={"canal": 20, "atr": 14, "colchon_atr": 0.5},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 11. HUECO DEJADO POR UNA HIPÓTESIS RETIRADA EN AUDITORÍA (2026-09-08)
#
# Aquí estaba `structure.inside_bar_break`. Se ha ELIMINADO por ser un DUPLICADO exacto de
# `candles.vela_interior_ruptura`: misma condición de contención (la vela i-1 dentro de la i-2),
# mismo nivel de ruptura (los extremos de la vela madre, i-2) y misma confirmación (el cierre de
# i). Lo único que las separaba era `<` frente a `<=` en la contención y el horizonte de tenencia
# (10 velas aquí, pulso de 1 vela allí). Medido sobre datos, el conjunto de eventos de esta
# hipótesis era un SUBCONJUNTO ESTRICTO del de la otra: 244 de 244 eventos coincidían vela a vela.
#
# Por qué importaba y no es cosmético. Un registro previo solo sirve si el número de ensayos que
# entra en la corrección por contraste múltiple es el número de APUESTAS DISTINTAS. Registrar la
# misma apuesta en dos familias la cuenta dos veces, y como además las dos comparten casi todas
# sus señales, sus resultados están correlacionados: si el patrón funciona por azar, "funciona"
# dos veces y parece confirmación cruzada entre familias cuando es la misma observación repetida.
# Ese es exactamente el sesgo que este registro existe para evitar, y la diferencia de horizonte
# no lo arregla: mantener 10 velas en vez de 1 multiplica cada evento por diez observaciones
# solapadas sobre casi la misma ventana futura, lo que infla `n_signals` sin aportar información
# (para eso está `effective_n`).
#
# Se conserva la versión de `candles` porque es el superconjunto (contención no estricta) y su
# pulso de una vela es la codificación limpia de la apuesta. La variante de `volatility.
# inside_bar_breakout` NO se elimina: rompe los extremos de la vela INTERIOR, no los de la madre,
# que es un nivel distinto y una afirmación mecánica distinta; queda declarada como ensayo
# dependiente en el propio fichero de `volatility`.
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# 12. Fade de los extremos cuando la estructura NO es tendencial
# --------------------------------------------------------------------------------------------

def _range_fade_swing(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c = _f(s.close)
    ph1, _, pl1, _ = _pivot_levels(s)
    mezclada = _structure_state(s) == 0          # ni HH+HL ni LH+LL
    # `_fresh`: el evento es el TOQUE del extremo, no el estado de estar por encima de él.
    corto = mezclada & _fresh(_gt(c, ph1))
    largo = mezclada & _fresh(_lt(c, pl1))
    return _hold_n(largo, corto)


register(Hypothesis(
    name="structure.range_fade_swing",
    family="structure",
    rationale="Contradice deliberadamente a `bos_swing` sobre el mismo nivel, y solo se activa "
              "donde aquella debería ser más débil: cuando la secuencia de pivotes está mezclada "
              "y no hay estructura direccional. El razonamiento es de inventario. Sin tendencia "
              "que absorba, quien provee liquidez en los extremos de la oscilación no tiene "
              "riesgo de quedarse en el lado equivocado de un movimiento sostenido, así que "
              "puede defender esos niveles con tamaño; la ruptura se queda sin continuación y el "
              "precio revierte hacia el centro del rango. La afirmación fuerte no es 'los rangos "
              "revierten', sino que la MISMA señal cambia de signo según el estado estructural.",
    prior="Esperamos ventaja positiva SOLO bajo el filtro de estructura mezclada, y esperamos "
          "explícitamente que la misma regla sin ese filtro dé ventaja negativa. NOTA DE AUDITORÍA "
          "(2026-09-08): esa regla sin filtro no estaba registrada, así que el criterio de "
          "falsación central de esta hipótesis no se podía ejecutar; se ha registrado como "
          "`structure.swing_fade_unfiltered` y es contra él contra quien debe medirse. Un fractal de 5 "
          "velas deja el nivel muy cerca del precio, así que esperamos MUCHAS operaciones y que "
          "los costes se coman una ventaja bruta pequeña: si la ventaja neta no sobrevive a "
          "comisiones y diferencial realistas, la hipótesis está refutada aunque la bruta sea "
          "positiva, y no vale rescatarla subiendo k. Esperamos que "
          "falle en los giros de régimen, donde la estructura aparece mezclada justo mientras se "
          "está construyendo la tendencia nueva: ahí los fades se ejecutan contra el impulso "
          "inicial y deberían ser las peores operaciones de toda la familia. Si el resultado es "
          "positivo tanto aquí como en `bos_swing` sin que el régimen los separe, el filtro no "
          "está haciendo nada y ambas conclusiones deben descartarse.",
    fn=_range_fade_swing,
    params={"fractal_k": _K, "filtro": "estructura mezclada", "horizonte": _HOLD},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 12b. CONTROL AÑADIDO EN AUDITORÍA (2026-09-08) para que `range_fade_swing` sea falsable.
#
# Qué estaba mal. El prior de `structure.range_fade_swing` dice, literalmente, que espera ventaja
# positiva "SOLO bajo el filtro de estructura mezclada" y que "la misma regla sin ese filtro dé
# ventaja negativa". Ese es su criterio de falsación y es el bueno: la afirmación fuerte no es que
# desvanecer extremos gane, sino que el ESTADO ESTRUCTURAL cambie el signo. El problema es que la
# regla sin filtro no estaba registrada en ninguna parte, así que el contraste no se podía
# ejecutar: un prior cuyo criterio de fracaso nombra un control inexistente no puede fallar, y un
# prior que no puede fallar no es un prior. Se registra aquí el control, con su propio prior y
# contando como un ensayo más en la corrección por contraste múltiple.
# --------------------------------------------------------------------------------------------

def _swing_fade_unfiltered(s: Series) -> np.ndarray:
    n = len(s)
    if n < _MIN_BARS:
        return np.zeros(n, dtype=np.int8)
    c = _f(s.close)
    ph1, _, pl1, _ = _pivot_levels(s)
    # Idéntica a `range_fade_swing` salvo por la ausencia del filtro `mezclada`.
    return _hold_n(_fresh(_lt(c, pl1)), _fresh(_gt(c, ph1)))


register(Hypothesis(
    name="structure.swing_fade_unfiltered",
    family="structure",
    rationale="Control incondicional de `range_fade_swing`: desvanecer el toque del último pivote "
              "confirmado en TODOS los estados estructurales, no solo cuando la secuencia está "
              "mezclada. No afirma un mecanismo propio —al contrario, el mecanismo de inventario "
              "que justifica el fade solo se sostiene cuando no hay tendencia que absorba— y está "
              "aquí porque una afirmación condicional únicamente es comprobable si el caso "
              "incondicional también se mide. Sin este control, un resultado positivo de "
              "`range_fade_swing` no distinguiría 'el filtro estructural aporta' de 'desvanecer "
              "extremos de oscilación gana siempre y el filtro es decorativo'.",
    prior="Esperamos ventaja NEGATIVA, y estamos comprometidos con ese signo por adelantado: sin "
          "filtro, la misma regla se pone corta contra cada ruptura de una tendencia viva, que es "
          "el modo de fallo declarado de todo desvanecimiento. La lectura conjunta es la que "
          "importa y se fija ahora: si esta sale negativa y `range_fade_swing` positiva, el estado "
          "estructural separa régimen y la afirmación condicional se sostiene; si ambas salen "
          "parecidas, el filtro no hace nada y las dos deben descartarse; si esta sale POSITIVA, "
          "lo que hay es reversión a la media genérica en los extremos de oscilación y el "
          "argumento de inventario de `range_fade_swing` es falso aunque su número sea bueno.",
    fn=_swing_fade_unfiltered,
    params={"fractal_k": _K, "filtro": "ninguno", "horizonte": _HOLD},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))
