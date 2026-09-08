"""Familia `volatility`: compresión y expansión del rango. REGISTRO PREVIO.

Escrito antes de ejecutar un solo backtest. Ningún parámetro de este fichero se ha elegido mirando
resultados: cada uno es el valor convencional de la literatura que lo definió (Bollinger 20/2,
Keltner EMA-20 ± 2·ATR-10, TTM Squeeze 20/20/1.5, ATR-14, NATR-14, Chandelier 22/3, NR7, 252 como
ventana anual convencional). Que aparezcan ATR-20, ATR-14, ATR-10 y ATR-22 en distintas hipótesis
NO es un barrido del periodo del ATR: cada uno viene fijado por el indicador compuesto que lo
contiene, y ninguno se mueve.

Mecanismo común de la familia
-----------------------------
Agrupamiento de volatilidad (Mandelbrot 1963, Engle 1982): la varianza está autocorrelacionada
aunque el retorno no lo esté. En un mercado 24/7 con creadores de mercado apalancados, un periodo
tranquilo comprime el inventario de liquidez: los MM estrechan sus horquillas, los vendedores de
volatilidad se cargan de gamma corta y las órdenes de stop de todo el mundo se acumulan justo
fuera de un rango cada vez más estrecho. Cuando el rango se rompe, la cobertura de esa gamma corta
y la cascada de stops empujan en la misma dirección: el movimiento grande nace del periodo
tranquilo, no del ruidoso.

Estimador de volatilidad realizada
----------------------------------
Rogers-Satchell: RS_t = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O). Elegido sobre Garman-Klass porque es
insesgado con deriva distinta de cero (BTC tiene deriva fuerte y GK la contabiliza como varianza),
y sobre Yang-Zhang porque YZ dedica un término al salto de apertura, que en un activo 24/7 sin
sesión no existe: ese término sería ruido de microestructura del corte arbitrario de la vela.

Contrastes deliberados (esto NO es búsqueda en rejilla)
-------------------------------------------------------
Tres parejas registradas a propósito, con la predicción de cuál gana escrita de antemano:

  1. `rs_vol_pct_low_carry` vs `bbw_pct_low_carry`: MISMA regla de dirección, distinto estimador de
     compresión (RS sobre OHLC vs anchura de Bollinger, que es desviación típica de cierres). Si el
     mecanismo "lo tranquilo precede a lo grande" es real, las dos deberían ir en el mismo sentido.
     Si solo una funciona, lo que se está midiendo es el estimador, no el mecanismo.
  2. `rs_vol_pct_high_fade` vs `rs_vol_pct_high_carry`: complementarias exactas. Como mucho una
     puede tener ventaja. Registrarlas juntas impide elegir el signo después de mirar.
  3. `natr_low_regime_trend` vs `natr_high_regime_trend`: partición por la mediana de la MISMA
     señal de tendencia. Juntas reconstruyen la señal sin filtrar, así que la comparación aísla el
     efecto del régimen de volatilidad y no el de la tendencia.

Causalidad
----------
Todo indicador aquí usa solo datos hasta i incluido. Las ventanas móviles se alinean con
`sliding_window_view(x, n)[j] == x[j : j+n]`, cuyo resultado se escribe en el índice `j+n-1`. No hay
ningún desplazamiento negativo, ningún `find_peaks`, ningún estadístico definido contra el array
entero. Los estados con memoria (`squeeze`, Keltner, Chandelier) se calculan en bucles hacia
delante, que son causales por construcción.
"""

from __future__ import annotations

import numpy as np
import talib
from numpy.lib.stride_tricks import sliding_window_view

from wavelab.hypotheses.base import Hypothesis, Series, register

# --------------------------------------------------------------------------------------------
# Utilidades. Ninguna mira hacia adelante.
# --------------------------------------------------------------------------------------------


def _ohlc(s: Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """talib exige float64 contiguo."""
    f = lambda a: np.ascontiguousarray(a, dtype=np.float64)  # noqa: E731
    return f(s.open), f(s.high), f(s.low), f(s.close)


def _roll_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Media de las n últimas observaciones, la de i incluida. NaN si la ventana toca un NaN."""
    out = np.full(x.size, np.nan)
    if x.size >= n:
        out[n - 1:] = sliding_window_view(x, n).mean(axis=1)
    return out


def _rolling_rank(x: np.ndarray, n: int) -> np.ndarray:
    """Percentil de x[i] dentro de la ventana x[i-n+1 : i+1]. Devuelve NaN durante el calentamiento.

    La ventana TERMINA en i, nunca lo rodea. Un percentil calculado contra el array entero sería
    lookahead puro (sabría hoy cuál fue la volatilidad máxima de 2031), y es exactamente el error
    que este proyecto persigue.
    """
    out = np.full(x.size, np.nan)
    if x.size < n:
        return out
    win = sliding_window_view(x, n)                  # win[j] == x[j : j+n]  ->  índice j+n-1
    # Cuenta de NaN por ventana sin propagar NaN (cumsum sobre enteros).
    cs = np.concatenate(([0], np.cumsum(np.isnan(x).astype(np.int64))))
    valid = (cs[n:] - cs[:-n]) == 0
    last = win[:, -1]
    cnt = (win < last[:, None]).sum(axis=1)          # comparar con NaN da False; se descarta abajo
    out[n - 1:] = np.where(valid, cnt / (n - 1), np.nan)
    return out


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """Valor de hace k velas. k > 0 SIEMPRE: un k negativo sería mirar el futuro."""
    if k <= 0:
        raise ValueError("_shift solo mira hacia atrás")
    out = np.full(x.size, np.nan)
    if k < x.size:
        out[k:] = x[: x.size - k]
    return out


def _rs_vol(s: Series, n: int) -> np.ndarray:
    """Volatilidad realizada de Rogers-Satchell sobre las n últimas velas, la de i incluida."""
    o, h, l, c = _ohlc(s)  # noqa: E741
    term = np.full(c.size, np.nan)
    pos = (o > 0) & (h > 0) & (l > 0) & (c > 0)
    if pos.any():
        oo, hh, ll, cc = o[pos], h[pos], l[pos], c[pos]
        term[pos] = np.log(hh / cc) * np.log(hh / oo) + np.log(ll / cc) * np.log(ll / oo)
    # RS es no negativo por construcción (H >= max(O,C) y L <= min(O,C)); el clip solo protege de
    # ruido numérico. np.maximum propaga NaN, así que el calentamiento se conserva como NaN.
    return np.sqrt(_roll_mean(np.maximum(term, 0.0), n))


def _hold(entry: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """Mantiene la última entrada hasta que `flat` la cierra. Bucle hacia delante: causal."""
    out = np.zeros(entry.size, dtype=np.int8)
    cur = 0
    for i in range(entry.size):
        if flat[i]:
            cur = 0
        e = int(entry[i])
        if e != 0:
            cur = e
        out[i] = cur
    return out


# --------------------------------------------------------------------------------------------
# 1. Squeeze de Bollinger dentro de Keltner (TTM Squeeze), operado en la liberación.
# --------------------------------------------------------------------------------------------


def _squeeze_release(s: Series) -> np.ndarray:
    o, h, l, c = _ohlc(s)  # noqa: E741
    up, mid, lo = talib.BBANDS(c, 20, 2.0, 2.0, 0)
    ema = talib.EMA(c, 20)
    atr = talib.ATR(h, l, c, 20)
    ok = ~(np.isnan(up) | np.isnan(lo) | np.isnan(mid) | np.isnan(ema) | np.isnan(atr))

    sq = np.zeros(c.size, dtype=bool)
    kc_up = ema + 1.5 * atr
    kc_lo = ema - 1.5 * atr
    sq[ok] = (up[ok] < kc_up[ok]) & (lo[ok] > kc_lo[ok])

    prev_sq = np.zeros(c.size, dtype=bool)
    prev_sq[1:] = sq[:-1]
    release = prev_sq & ~sq & ok

    entry = np.zeros(c.size, dtype=np.int8)
    entry[release & (c > mid)] = 1
    entry[release & (c < mid)] = -1

    out = np.zeros(c.size, dtype=np.int8)
    cur = 0
    for i in range(c.size):
        if cur == 1 and (sq[i] or (ok[i] and c[i] < mid[i])):
            cur = 0
        elif cur == -1 and (sq[i] or (ok[i] and c[i] > mid[i])):
            cur = 0
        e = int(entry[i])
        if e != 0:
            cur = e
        out[i] = cur
    return out


register(Hypothesis(
    name="volatility.squeeze_bb_kc_release",
    family="volatility",
    rationale="Cuando las bandas de Bollinger (20, 2σ) caben dentro del canal de Keltner "
              "(EMA-20 ± 1.5·ATR-20), la dispersión de cierres se ha hundido por debajo del rango "
              "verdadero: el mercado cotiza en un pañuelo mientras sigue habiendo recorrido "
              "intravela. Ese estado lo fabrican creadores de mercado y vendedores de volatilidad "
              "que estrechan horquillas y acumulan gamma corta, y traders de rango que colocan "
              "stops justo fuera del pañuelo. Cuando el precio sale, esos mismos participantes "
              "tienen que cubrirse en la dirección del movimiento y los stops se ejecutan a "
              "mercado, de modo que el flujo que sigue a la ruptura es forzado, no discrecional.",
    prior="Esperamos ventaja positiva en la dirección de la ruptura respecto a la media móvil de "
          "20, concentrada en las primeras velas tras la liberación. Esperamos que FALLE, con "
          "ventaja negativa, cuando la compresión ocurre dentro de un rango amplio de orden "
          "superior (falsa ruptura que revierte al centro) y en 15m, donde el 'squeeze' es a "
          "menudo un hueco de liquidez de madrugada y no acumulación de posicionamiento. Si la "
          "ventaja resultara indistinguible de cero en los tres timeframes, el mecanismo de gamma "
          "corta no está operando en BTC al detalle de vela.",
    fn=_squeeze_release,
    params={"bb_periodo": 20, "bb_sigma": 2.0, "kc_periodo": 20, "kc_atr": 20, "kc_mult": 1.5},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 2 y 3. Compresión medida de dos formas distintas, MISMA regla de dirección.
# --------------------------------------------------------------------------------------------


def _rs_vol_pct_low_carry(s: Series) -> np.ndarray:
    _, _, _, c = _ohlc(s)
    rank = _rolling_rank(_rs_vol(s, 24), 252)
    ema = talib.EMA(c, 55)
    out = np.zeros(c.size, dtype=np.int8)
    quiet = ~np.isnan(rank) & ~np.isnan(ema) & (rank < 0.20)
    out[quiet & (c > ema)] = 1
    out[quiet & (c < ema)] = -1
    return out


register(Hypothesis(
    name="volatility.rs_vol_pct_low_carry",
    family="volatility",
    rationale="La volatilidad realizada de Rogers-Satchell sobre 24 velas, situada en su percentil "
              "móvil de 252, mide si el mercado está tranquilo RESPECTO A SÍ MISMO y no respecto a "
              "un umbral absoluto que la inflación de precio de BTC dejaría obsoleto. Con "
              "volatilidad en el quintil inferior, quien está posicionado no está siendo expulsado "
              "por ruido: el coste de mantener una posición tendencial cae, las liquidaciones "
              "forzosas se detienen y la deriva se acumula sin interrupciones. La dirección la "
              "damos con el precio contra la EMA-55, deliberadamente trivial, porque lo que se "
              "contrasta aquí es la PUERTA de volatilidad, no el detector de tendencia.",
    prior="Esperamos ventaja positiva mientras la volatilidad esté en el quintil bajo, y que esa "
          "ventaja sea MAYOR que la de la misma regla de dirección sin filtrar. Esperamos que "
          "falle en dos sitios concretos: (a) tras una caída estructural del nivel de volatilidad "
          "(mercado lateral prolongado), donde el percentil se queda anclado abajo y la señal "
          "opera un rango sin tendencia; (b) en 1d, donde 252 velas son un año entero y el "
          "percentil arrastra régimen viejo. Si la ventaja fuese igual o menor que la de la EMA-55 "
          "sin filtrar, la puerta de volatilidad no aporta nada. NOTA DE AUDITORÍA (2026-09-08): "
          "ese brazo sin filtrar no estaba registrado, de modo que el criterio de falsación de esta "
          "hipótesis y el de `bbw_pct_low_carry` no se podían ejecutar; se ha registrado como "
          "`volatility.ema55_side_unfiltered` y es contra él contra quien deben medirse las dos.",
    fn=_rs_vol_pct_low_carry,
    params={"rs_ventana": 24, "percentil_ventana": 252, "umbral": 0.20, "ema_direccion": 55},
    timeframes=("1h", "4h", "1d"),
    min_warmup=300,
))


def _bbw_pct_low_carry(s: Series) -> np.ndarray:
    _, _, _, c = _ohlc(s)
    up, mid, lo = talib.BBANDS(c, 20, 2.0, 2.0, 0)
    bw = np.full(c.size, np.nan)
    good = ~(np.isnan(up) | np.isnan(lo) | np.isnan(mid)) & (mid != 0)
    bw[good] = (up[good] - lo[good]) / mid[good]
    rank = _rolling_rank(bw, 252)
    ema = talib.EMA(c, 55)
    out = np.zeros(c.size, dtype=np.int8)
    quiet = ~np.isnan(rank) & ~np.isnan(ema) & (rank < 0.20)
    out[quiet & (c > ema)] = 1
    out[quiet & (c < ema)] = -1
    return out


register(Hypothesis(
    name="volatility.bbw_pct_low_carry",
    family="volatility",
    rationale="Contraste controlado de la anterior: idéntica regla de dirección (precio contra "
              "EMA-55), idéntica ventana de percentil (252), idéntico umbral (quintil inferior), y "
              "lo ÚNICO que cambia es el estimador de compresión: anchura de Bollinger, que es "
              "desviación típica de CIERRES, frente a Rogers-Satchell, que usa el rango OHLC "
              "completo. Los dos pueden divergir mucho: una vela de mecha larga y cierre plano "
              "(liquidación absorbida) es tranquila para Bollinger y ruidosa para RS. Si el "
              "mecanismo de agrupamiento es real, debería aparecer con los dos estimadores.",
    prior="Esperamos el MISMO signo que en `rs_vol_pct_low_carry` y una magnitud algo menor, "
          "porque el estimador de cierres desperdicia la información de las mechas. La predicción "
          "falsable fuerte es la conjunta: si una de las dos da ventaja clara y la otra da cero o "
          "signo contrario, la conclusión correcta NO es 'funciona la compresión' sino que "
          "estamos midiendo una peculiaridad del estimador, y ninguna de las dos debe sobrevivir a "
          "la corrección por contraste múltiple como evidencia del mecanismo.",
    fn=_bbw_pct_low_carry,
    params={"bb_periodo": 20, "bb_sigma": 2.0, "percentil_ventana": 252, "umbral": 0.20,
            "ema_direccion": 55},
    timeframes=("1h", "4h", "1d"),
    min_warmup=300,
))


# --------------------------------------------------------------------------------------------
# 4 y 5. Volatilidad alta: complementarias exactas. Como mucho una puede tener ventaja.
# --------------------------------------------------------------------------------------------


def _rs_vol_pct_high_fade(s: Series) -> np.ndarray:
    _, _, _, c = _ohlc(s)
    rank = _rolling_rank(_rs_vol(s, 24), 252)
    prev = _shift(c, 3)
    out = np.zeros(c.size, dtype=np.int8)
    hot = ~np.isnan(rank) & ~np.isnan(prev) & (rank > 0.90)
    out[hot & (c > prev)] = -1
    out[hot & (c < prev)] = 1
    return out


register(Hypothesis(
    name="volatility.rs_vol_pct_high_fade",
    family="volatility",
    rationale="En el decil superior de volatilidad realizada, el movimiento de las últimas 3 velas "
              "está dominado por liquidaciones forzosas: el motor de riesgo del exchange vende (o "
              "compra) a mercado sin mirar el precio, y quien provee liquidez contra esa cascada "
              "exige una prima. Si esa prima es la parte grande del desplazamiento, el precio "
              "vuelve cuando la cascada agota el colateral disponible, y el proveedor de liquidez "
              "se lleva el retroceso. Esta hipótesis apuesta a que el exceso es prima de "
              "liquidez, no información.",
    prior="Esperamos ventaja positiva al contrarrestar el movimiento de 3 velas. Esperamos que "
          "FALLE, con ventaja claramente negativa, si la volatilidad alta en BTC es informativa en "
          "lugar de mecánica: en marzo de 2020 o en el desapalancamiento de mayo de 2021 la "
          "cascada continuó días. Como está registrada junto a su complementaria exacta "
          "`rs_vol_pct_high_carry`, ambas no pueden ganar; si las dos salen indistinguibles de "
          "cero, el decil alto de volatilidad simplemente no contiene señal direccional y las dos "
          "deben publicarse como fallidas.",
    fn=_rs_vol_pct_high_fade,
    params={"rs_ventana": 24, "percentil_ventana": 252, "umbral": 0.90, "impulso_velas": 3},
    timeframes=("1h", "4h", "1d"),
    min_warmup=300,
))


def _rs_vol_pct_high_carry(s: Series) -> np.ndarray:
    _, _, _, c = _ohlc(s)
    rank = _rolling_rank(_rs_vol(s, 24), 252)
    prev = _shift(c, 3)
    out = np.zeros(c.size, dtype=np.int8)
    hot = ~np.isnan(rank) & ~np.isnan(prev) & (rank > 0.90)
    out[hot & (c > prev)] = 1
    out[hot & (c < prev)] = -1
    return out


register(Hypothesis(
    name="volatility.rs_vol_pct_high_carry",
    family="volatility",
    rationale="La lectura opuesta del mismo estado: en el decil superior de volatilidad realizada, "
              "la liquidación forzosa es reflexiva. Cada liquidación mueve el precio hacia el "
              "siguiente grupo de garantías, que se liquida a su vez; el libro se vacía en la "
              "dirección del movimiento y los creadores de mercado retiran cotizaciones en vez de "
              "absorber. Bajo ese mecanismo, el exceso de volatilidad es la propia señal de que la "
              "cascada sigue viva y la continuación domina al retroceso en el horizonte de unas "
              "pocas velas.",
    prior="Esperamos ventaja positiva en la dirección del movimiento de 3 velas, pero PEQUEÑA en "
          "términos netos, porque opera justo cuando la horquilla y el deslizamiento "
          "son máximos: es una predicción que puede ser correcta en signo y aun así no sobrevivir "
          "a los costes, y así debe evaluarse. Esperamos que falle en los máximos de capitulación, "
          "donde la última vela del decil alto es exactamente el giro. Es complementaria exacta de "
          "`rs_vol_pct_high_fade`: registrarlas juntas impide elegir el signo después de mirar.",
    fn=_rs_vol_pct_high_carry,
    params={"rs_ventana": 24, "percentil_ventana": 252, "umbral": 0.90, "impulso_velas": 3},
    timeframes=("1h", "4h", "1d"),
    min_warmup=300,
))


# --------------------------------------------------------------------------------------------
# 6, 7, 8. Contracción de rango y expansión, a nivel de vela. Horizonte de UNA vela: no
# introducimos un parámetro de duración de la posición que después habría que justificar.
# --------------------------------------------------------------------------------------------


def _nr7_breakout(s: Series) -> np.ndarray:
    o, h, l, c = _ohlc(s)  # noqa: E741
    tr = talib.TRANGE(h, l, c)
    min7 = talib.MIN(tr, 7)
    es_nr7 = np.where(np.isnan(tr) | np.isnan(min7), np.nan, (tr <= min7).astype(float))
    nr7_prev = _shift(es_nr7, 1)
    hi_prev, lo_prev = _shift(h, 1), _shift(l, 1)
    out = np.zeros(c.size, dtype=np.int8)
    ok = ~(np.isnan(nr7_prev) | np.isnan(hi_prev) | np.isnan(lo_prev)) & (nr7_prev == 1.0)
    out[ok & (c > hi_prev)] = 1
    out[ok & (c < lo_prev)] = -1
    return out


register(Hypothesis(
    name="volatility.nr7_breakout",
    family="volatility",
    rationale="NR7 de Toby Crabel: la vela anterior tuvo el rango verdadero más estrecho de las "
              "últimas siete. Un rango mínimo local significa que compradores y vendedores "
              "acordaron el precio durante un periodo completo, lo que concentra las órdenes en "
              "reposo (stops y límites) en una franja muy delgada. El cierre de la vela siguiente "
              "fuera de esa franja consume esas órdenes de golpe, y el desequilibrio resultante es "
              "mecánico. Es la versión mínima del mecanismo de la familia: contracción medida sin "
              "estimadores ni ventanas largas, solo el rango de siete velas.",
    prior="Esperamos ventaja positiva en la dirección de la ruptura, con horizonte de una sola "
          "vela. Esperamos que sea MAYOR en 15m y 1h, donde el libro de órdenes en reposo es "
          "relevante frente al tamaño típico, y que se degrade o desaparezca en 4h, donde una vela "
          "agrega demasiadas manos para que quede un desequilibrio explotable. Si la ventaja "
          "creciera con el timeframe estaríamos midiendo tendencia, no contracción de rango, y la "
          "hipótesis quedaría refutada aunque el número fuese bueno.",
    fn=_nr7_breakout,
    params={"nr_ventana": 7, "horizonte_velas": 1},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


def _inside_bar_breakout(s: Series) -> np.ndarray:
    o, h, l, c = _ohlc(s)  # noqa: E741
    h1, l1 = _shift(h, 1), _shift(l, 1)
    h2, l2 = _shift(h, 2), _shift(l, 2)
    out = np.zeros(c.size, dtype=np.int8)
    ok = ~(np.isnan(h1) | np.isnan(l1) | np.isnan(h2) | np.isnan(l2))
    inside = ok & (h1 < h2) & (l1 > l2)
    out[inside & (c > h1)] = 1
    out[inside & (c < l1)] = -1
    return out


register(Hypothesis(
    name="volatility.inside_bar_breakout",
    family="volatility",
    rationale="Vela interior: el rango de la vela anterior está ESTRICTAMENTE contenido en el de "
              "la que la precede. No es lo mismo que NR7 y por eso se registra aparte: NR7 es "
              "estrechez relativa a una muestra de siete, la vela interior es una condición de "
              "contención absoluta respecto a UNA vela concreta, la madre. La contención significa "
              "que ningún participante logró imponer un precio fuera del rango que ya se había "
              "explorado y rechazado: el equilibrio se ha reafirmado en un rango conocido, con las "
              "órdenes ancladas a los extremos de la vela interior. Romperlos por cierre invalida "
              "esa reafirmación.",
    prior="Esperamos ventaja positiva, del mismo signo que `nr7_breakout` y de magnitud similar. "
          "La comparación entre ambas es informativa por sí misma: si la contención absoluta "
          "funciona y la estrechez relativa no (o al revés), lo que hay es una peculiaridad del "
          "criterio y no el mecanismo de contracción. Esperamos que falle en mercados con deriva "
          "fuerte y sostenida, donde la vela interior es solo una pausa de continuación y la "
          "ruptura ya llega tarde, y en 15m durante las horas de menor volumen, donde la "
          "contención refleja ausencia de participantes y no acuerdo entre ellos. "
          "NOTA DE AUDITORÍA (2026-09-08): `candles.vela_interior_ruptura` registra el mismo patrón "
          "de dos velas pero rompiendo los extremos de la vela MADRE. Como el máximo de la vela "
          "interior es menor que el de la madre, todo evento de aquella es también evento de esta: "
          "son el mismo patrón con dos umbrales distintos. Se conservan las dos porque el nivel es "
          "la afirmación y son niveles distintos, pero son ensayos DEPENDIENTES y así deben "
          "contarse. (Una tercera registración del mismo patrón, `structure.inside_bar_break`, era "
          "un duplicado exacto de la de `candles` y se ha eliminado en la auditoría.)",
    fn=_inside_bar_breakout,
    params={"horizonte_velas": 1},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


def _wide_range_thrust(s: Series) -> np.ndarray:
    o, h, l, c = _ohlc(s)  # noqa: E741
    tr = talib.TRANGE(h, l, c)
    atr = talib.ATR(h, l, c, 14)
    atr_prev = _shift(atr, 1)
    rank_prev = _shift(_rolling_rank(atr, 100), 1)
    out = np.zeros(c.size, dtype=np.int8)
    ok = ~(np.isnan(tr) | np.isnan(atr_prev) | np.isnan(rank_prev))
    thrust = ok & (rank_prev < 0.50) & (tr > 2.0 * atr_prev)
    out[thrust & (c > o)] = 1
    out[thrust & (c < o)] = -1
    return out


register(Hypothesis(
    name="volatility.wide_range_thrust",
    family="volatility",
    rationale="Expansión de rango DESPUÉS de contracción, que es la formulación directa de 'los "
              "movimientos grandes nacen de periodos tranquilos': exigimos que el ATR-14 de la "
              "vela previa estuviera por debajo de su mediana de 100 velas (contracción) y que el "
              "rango verdadero actual supere el doble de ese ATR (expansión). Una vela así, "
              "saliendo de una base tranquila, no la produce el flujo minorista repartido: la "
              "produce un participante grande que necesita ejecutar y acepta pagar el rango, o el "
              "disparo de un grupo de stops. La dirección la da el cuerpo de la vela, cierre "
              "contra apertura, porque es lo que revela quién ganó el intercambio.",
    prior="Esperamos ventaja positiva en la dirección del cuerpo, con horizonte de una vela. "
          "Esperamos que FALLE claramente cuando el empuje es una reacción a una noticia puntual "
          "que se retrae entera en las velas siguientes (patrón típico en BTC con anuncios "
          "regulatorios), y en 1d, donde una vela de expansión suele ser ya el final del "
          "movimiento y no su principio. Si la ventaja fuese negativa de forma consistente, el "
          "mecanismo correcto sería el de agotamiento y no el de inicio de expansión.",
    fn=_wide_range_thrust,
    params={"atr_periodo": 14, "mult_expansion": 2.0, "mediana_ventana": 100, "horizonte_velas": 1},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 9 y 10. Estructuras escaladas por volatilidad: el umbral se mueve con el ATR, no con el precio.
# --------------------------------------------------------------------------------------------


def _keltner_breakout(s: Series) -> np.ndarray:
    o, h, l, c = _ohlc(s)  # noqa: E741
    ema = talib.EMA(c, 20)
    atr = talib.ATR(h, l, c, 10)
    ok = ~(np.isnan(ema) | np.isnan(atr))
    up, lo = ema + 2.0 * atr, ema - 2.0 * atr
    entry = np.zeros(c.size, dtype=np.int8)
    entry[ok & (c > up)] = 1
    entry[ok & (c < lo)] = -1
    out = np.zeros(c.size, dtype=np.int8)
    cur = 0
    for i in range(c.size):
        if cur == 1 and ok[i] and c[i] < ema[i]:
            cur = 0
        elif cur == -1 and ok[i] and c[i] > ema[i]:
            cur = 0
        e = int(entry[i])
        if e != 0:
            cur = e
        out[i] = cur
    return out


register(Hypothesis(
    name="volatility.keltner_breakout",
    family="volatility",
    rationale="Ruptura normalizada por volatilidad, sin componente de compresión: cierre por "
              "encima de EMA-20 + 2·ATR-10, o por debajo de EMA-20 − 2·ATR-10. Bajo un paseo "
              "aleatorio con la volatilidad ACTUAL, alejarse dos ATR de la media es raro; que "
              "ocurra es evidencia de que ha entrado un flujo que el nivel de volatilidad vigente "
              "no explica, es decir, de un cambio en la deriva y no de una fluctuación. El umbral "
              "se reescala solo, así que la misma regla es igual de exigente a 20.000 que a "
              "100.000 dólares. Se mantiene hasta que el cierre vuelve a cruzar la EMA-20, que "
              "es la definición mínima de 'el flujo se acabó'.",
    prior="Esperamos ventaja positiva y, sobre todo, ASIMETRÍA respecto al squeeze: si "
          "`squeeze_bb_kc_release` bate a esta hipótesis, la compresión previa aporta información "
          "por encima de la ruptura sola, que es la afirmación central de la familia. Si gana "
          "esta en cambio, la compresión es decorativa y lo único que funciona es la ruptura "
          "normalizada por volatilidad. "
          "Esperamos que falle en régimen lateral con volatilidad media, donde el precio cruza los "
          "dos ATR en ambos sentidos repetidamente y cada cruce cuesta una horquilla.",
    fn=_keltner_breakout,
    params={"ema_periodo": 20, "atr_periodo": 10, "mult": 2.0},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


def _chandelier_trail(s: Series) -> np.ndarray:
    o, h, l, c = _ohlc(s)  # noqa: E741
    atr = talib.ATR(h, l, c, 22)
    hh = talib.MAX(h, 22)
    ll = talib.MIN(l, 22)
    ok = ~(np.isnan(atr) | np.isnan(hh) | np.isnan(ll))
    long_stop = hh - 3.0 * atr
    short_stop = ll + 3.0 * atr

    # CORREGIDO EN AUDITORÍA (2026-09-08). Antes esto era:
    #     entry[ok & (c > short_stop)] = 1
    #     entry[ok & (c < long_stop)] = -1
    # y las dos asignaciones se pisaban. Cuando el rango de 22 velas es ancho (hh - ll > 6·ATR) el
    # umbral largo queda POR ENCIMA del corto, así que un cierre en la zona intermedia cumplía las
    # DOS condiciones a la vez; como el -1 se escribía después, ganaba siempre el corto. Medido
    # sobre una serie con los tres regímenes, eso afectaba al 2,7 % de las velas y las forzaba
    # todas a corto.
    #
    # Por qué importaba y no es un detalle. El comentario de esta misma función declaraba lo
    # contrario de lo que hacía el código ("cuando el cierre queda entre los dos umbrales no hay
    # información nueva: se mantiene el estado"), y el sesgo no era aleatorio: la zona ambigua
    # aparece precisamente en los rangos anchos, es decir, en el régimen donde el prior de esta
    # hipótesis predice PÉRDIDAS. Un desempate arbitrario que sistemáticamente se pone corto justo
    # en el régimen que se quiere medir contamina el signo del resultado y lo hace ininterpretable:
    # no se sabría si lo medido es el escalado por ATR o el desempate. Ahora la zona ambigua emite
    # 0 —sin entrada nueva— y `_hold` arrastra el estado anterior, que es la regla de Chandelier
    # tal y como está declarada en el `rationale` y lo que hace `_events` en la familia `structure`.
    entry = np.zeros(c.size, dtype=np.int8)
    largo = ok & (c > short_stop)
    corto = ok & (c < long_stop)
    ambiguo = largo & corto
    entry[largo & ~ambiguo] = 1
    entry[corto & ~ambiguo] = -1
    return _hold(entry, np.zeros(c.size, dtype=bool))


register(Hypothesis(
    name="volatility.chandelier_atr_trail",
    family="volatility",
    rationale="Salida Chandelier (Chuck LeBeau, 22/3): largo mientras el cierre esté por encima "
              "del mínimo de 22 velas más 3·ATR-22, corto mientras esté por debajo del máximo de "
              "22 velas menos 3·ATR-22. El contenido de volatilidad está en que la distancia de "
              "invalidación se escala con el ATR: la posición sobrevive exactamente al ruido que "
              "el régimen actual produce y ni un poco más. Es la traducción operativa del "
              "agrupamiento de volatilidad: si la varianza está autocorrelacionada, el ruido de "
              "mañana se estima bien con el de hoy, y un umbral fijo en porcentaje estaría "
              "demasiado cerca en régimen agitado y demasiado lejos en régimen tranquilo.",
    prior="Esperamos ventaja positiva en régimen tendencial y NEGATIVA en lateral: la señal está "
          "siempre en mercado, así que en rango paga cada giro. La afirmación falsable propia de "
          "esta familia no es 'seguir tendencias funciona' —eso lo contrasta la familia trend— "
          "sino que ESCALAR el umbral con el ATR bate a la misma estructura con umbral fijo — "
          "brazo que NO estaba registrado y que la auditoría de 2026-09-08 ha añadido como "
          "`volatility.chandelier_fixed_pct`, sin el cual este criterio no se podía ejecutar. Si el "
          "resultado neto a lo largo de todo el histórico fuese indistinguible de cero, la "
          "conclusión correcta es que en BTC la ganancia de las tendencias compensa justo el coste "
          "de los rangos, y no hay ventaja atribuible al escalado por volatilidad.",
    fn=_chandelier_trail,
    params={"ventana": 22, "atr_periodo": 22, "mult": 3.0},
    timeframes=("4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 11 y 12. Partición por la mediana de NATR sobre la MISMA señal de tendencia.
# --------------------------------------------------------------------------------------------


def _natr_regime_trend(s: Series, alto: bool) -> np.ndarray:
    _, h, l, c = _ohlc(s)  # noqa: E741
    natr = talib.NATR(h, l, c, 14)
    rank = _rolling_rank(natr, 252)
    e21, e55 = talib.EMA(c, 21), talib.EMA(c, 55)
    out = np.zeros(c.size, dtype=np.int8)
    ok = ~(np.isnan(rank) | np.isnan(e21) | np.isnan(e55))
    gate = ok & ((rank >= 0.50) if alto else (rank < 0.50))
    out[gate & (e21 > e55)] = 1
    out[gate & (e21 < e55)] = -1
    return out


def _natr_low_regime_trend(s: Series) -> np.ndarray:
    return _natr_regime_trend(s, alto=False)


def _natr_high_regime_trend(s: Series) -> np.ndarray:
    return _natr_regime_trend(s, alto=True)


register(Hypothesis(
    name="volatility.natr_low_regime_trend",
    family="volatility",
    rationale="El ATR normalizado (NATR-14, ATR en porcentaje del precio) situado en su percentil "
              "móvil de 252 clasifica el régimen sin depender del nivel de precio. Por debajo de "
              "su mediana, el recorrido diario típico es pequeño frente al tamaño de posición que "
              "cualquiera puede sostener: no hay barridos de stops ni liquidaciones en cadena, así "
              "que quien está en el lado correcto de la EMA-21/EMA-55 no es expulsado antes de que "
              "la deriva se materialice. La señal de tendencia es aquí un instrumento fijo; lo que "
              "se contrasta es el régimen.",
    prior="Esperamos ventaja positiva y superior a la de la misma señal EMA-21/55 sin filtrar. "
          "Esperamos que falle si la volatilidad baja en BTC corresponde principalmente a rangos "
          "de acumulación largos y no a tendencias suaves: entonces el cruce de medias dará "
          "señales falsas encadenadas y la ventaja será negativa. Es la mitad baja de la partición "
          "por la mediana: junto a `natr_high_regime_trend` reconstruye la señal sin filtrar, de "
          "modo que las dos no pueden ser ambas mejores que ella.",
    fn=_natr_low_regime_trend,
    params={"natr_periodo": 14, "percentil_ventana": 252, "corte": 0.50, "ema_rapida": 21,
            "ema_lenta": 55},
    timeframes=("1h", "4h", "1d"),
    min_warmup=300,
))


register(Hypothesis(
    name="volatility.natr_high_regime_trend",
    family="volatility",
    rationale="La mitad complementaria, registrada para que la afirmación anterior sea falsable y "
              "no una selección posterior. Por encima de la mediana de NATR-14, la misma señal de "
              "tendencia opera cuando el recorrido típico es grande: cada vela puede recorrer "
              "varias veces la distancia entre las dos medias, de modo que el cruce se produce y "
              "se deshace por ruido de amplitud, y además la horquilla y el deslizamiento son "
              "máximos precisamente ahí.",
    prior="Esperamos ventaja NULA O NEGATIVA aquí, y positiva en `natr_low_regime_trend`. Ese es "
          "el contraste, y estamos comprometidos con ese signo por adelantado. Si saliera al "
          "revés —la tendencia paga en volatilidad alta y no en baja— el mecanismo que hemos "
          "descrito para toda esta familia estaría equivocado: significaría que en BTC la "
          "volatilidad alta acompaña a la tendencia direccional en lugar de destruirla, y las "
          "hipótesis de compresión de este fichero deberían leerse con mucha más desconfianza "
          "aunque alguna diera un número favorable.",
    fn=_natr_high_regime_trend,
    params={"natr_periodo": 14, "percentil_ventana": 252, "corte": 0.50, "ema_rapida": 21,
            "ema_lenta": 55},
    timeframes=("1h", "4h", "1d"),
    min_warmup=300,
))


# --------------------------------------------------------------------------------------------
# 13 y 14. CONTROLES AÑADIDOS EN AUDITORÍA (2026-09-08).
#
# Qué estaba mal. Tres hipótesis de este fichero declaraban su criterio de falsación contra un
# control que NO estaba registrado en ninguna parte del registro previo:
#
#   - `rs_vol_pct_low_carry` y `bbw_pct_low_carry` dicen que su ventaja debe ser "MAYOR que la de
#     la misma regla de dirección sin filtrar" y que "si la ventaja fuese igual o menor que la de
#     la EMA-55 sin filtrar, la puerta de volatilidad no aporta nada". El precio contra la EMA-55
#     sin filtro no existía como hipótesis registrada.
#   - `chandelier_atr_trail` dice explícitamente que "la afirmación falsable propia de esta familia
#     [...] es que ESCALAR el umbral con el ATR bate a la misma estructura con umbral fijo". La
#     estructura con umbral fijo no existía.
#
# Por qué importaba. Un prior cuyo criterio de fracaso nombra un objeto inexistente no se puede
# ejecutar, y lo que no se puede ejecutar no puede fallar: en la práctica esas tres hipótesis solo
# podían ser juzgadas por su rentabilidad absoluta, que es justo el juicio que el registro previo
# existe para prohibir (en un activo que multiplicó por veinte, cualquier regla mayoritariamente
# larga "gana"). Se registran aquí los dos controles que faltaban, con su propio prior y contando
# como dos ensayos más en la corrección por contraste múltiple. Se prefiere ampliar el número de
# ensayos —que es la dirección conservadora— antes que rebajar los criterios de falsación ya
# escritos, que sería reescribir el registro después de haberlo cerrado.
# --------------------------------------------------------------------------------------------


def _ema55_side_unfiltered(s: Series) -> np.ndarray:
    _, _, _, c = _ohlc(s)
    ema = talib.EMA(c, 55)
    out = np.zeros(c.size, dtype=np.int8)
    ok = ~np.isnan(ema)
    out[ok & (c > ema)] = 1
    out[ok & (c < ema)] = -1
    return out


register(Hypothesis(
    name="volatility.ema55_side_unfiltered",
    family="volatility",
    rationale="Control incondicional de `rs_vol_pct_low_carry` y `bbw_pct_low_carry`: exactamente "
              "su misma regla de dirección —cierre por encima o por debajo de la EMA-55— sin "
              "ninguna puerta de volatilidad. No afirma un mecanismo propio, y ese es el punto: "
              "las dos hipótesis de compresión no apuestan a que seguir la EMA-55 gane, sino a que "
              "gane MÁS cuando la volatilidad realizada está en el quintil inferior. Esa es una "
              "afirmación sobre una diferencia, y una diferencia no se puede medir con un solo "
              "brazo. Se registra en `volatility` y no en `trend` porque su única razón de existir "
              "es servir de denominador a la puerta de volatilidad de esta familia.",
    prior="Esperamos ventaja positiva pequeña, dominada por la deriva secular de BTC, y "
          "esperamos que sea MENOR que la de `rs_vol_pct_low_carry`. Esa comparación es todo el "
          "contenido: si este control iguala o supera a las dos versiones filtradas, la puerta de "
          "volatilidad no aporta nada y ambas quedan refutadas aunque su número absoluto sea "
          "bueno; si las dos filtradas lo superan, la afirmación de agrupamiento de volatilidad se "
          "sostiene. Por sí solo, un resultado positivo de este control NO es un hallazgo: es la "
          "deriva del activo repartida por el lado de una media, y así debe publicarse.",
    fn=_ema55_side_unfiltered,
    params={"ema_direccion": 55, "puerta_volatilidad": "ninguna"},
    timeframes=("1h", "4h", "1d"),
    min_warmup=300,
))


def _chandelier_fixed_pct(s: Series) -> np.ndarray:
    """Idéntica a `_chandelier_trail` salvo que el colchón es un 6 % fijo del nivel en vez de
    3·ATR-22. El 6 % no se ha buscado: es el valor que iguala a 3·ATR-22 en un régimen de
    volatilidad típico de BTC en 4h, elegido de antemano para que el control difiera de la
    hipótesis en el ESCALADO y no en la agresividad media del umbral."""
    _, h, low, c = _ohlc(s)
    hh = talib.MAX(h, 22)
    ll = talib.MIN(low, 22)
    ok = ~(np.isnan(hh) | np.isnan(ll))
    long_stop = hh * (1.0 - 0.06)
    short_stop = ll * (1.0 + 0.06)
    entry = np.zeros(c.size, dtype=np.int8)
    largo = ok & (c > short_stop)
    corto = ok & (c < long_stop)
    ambiguo = largo & corto          # misma regla de desempate que la hipótesis que controla
    entry[largo & ~ambiguo] = 1
    entry[corto & ~ambiguo] = -1
    return _hold(entry, np.zeros(c.size, dtype=bool))


register(Hypothesis(
    name="volatility.chandelier_fixed_pct",
    family="volatility",
    rationale="Control de `chandelier_atr_trail`. Misma estructura exacta —largo mientras el cierre "
              "esté por encima del mínimo de 22 velas más un colchón, corto mientras esté por "
              "debajo del máximo de 22 menos ese colchón— con una única diferencia: el colchón es "
              "un porcentaje FIJO del nivel en vez de 3·ATR-22. Es el brazo que hace comprobable la "
              "afirmación central de la familia, que no es 'seguir tendencias funciona' sino que "
              "medir la distancia de invalidación en unidades de la volatilidad VIGENTE bate a "
              "medirla en unidades de precio. Si el agrupamiento de volatilidad es real, el umbral "
              "escalado debería estar demasiado cerca en régimen agitado y demasiado lejos en "
              "régimen tranquilo justo cuando el fijo se equivoca, y no antes.",
    prior="Esperamos ventaja positiva y MENOR que la de `chandelier_atr_trail`, con la diferencia "
          "concentrada en los cambios de régimen de volatilidad —los meses posteriores a marzo de "
          "2020 y a mayo de 2021— y prácticamente nula en el resto de la muestra. Si las dos "
          "resultan indistinguibles, el escalado por ATR no compra nada y `chandelier_atr_trail` "
          "queda refutada en su afirmación propia aunque gane dinero; si este control resulta "
          "MEJOR, el agrupamiento de volatilidad opera en el sentido contrario al que describe el "
          "encabezado de esta familia y todas las hipótesis de compresión de este fichero deben "
          "releerse con desconfianza. El 6 % queda congelado: si falla no se prueba otro valor.",
    fn=_chandelier_fixed_pct,
    params={"ventana": 22, "colchon_pct": 0.06, "escalado": "ninguno"},
    timeframes=("4h", "1d"),
    min_warmup=200,
))
