"""Familia `flow`: volumen y flujo de órdenes.

REGISTRO PREVIO. Todo lo de este fichero se escribió ANTES de ejecutar un solo backtest y sin mirar
un solo resultado. Los parámetros son los valores convencionales de la literatura, fijados de una
vez; ninguno se eligió comparando rendimientos.

Tesis de la familia: el precio dice qué pasó, el volumen dice cuánta convicción había detrás. Un
avance que mueve el doble de contratos de lo normal deja inventario en manos que tendrán que
defenderlo o cerrarlo; un avance en volumen fino no deja nada y no obliga a nadie a hacer nada
después. Todas las hipótesis de abajo son variaciones de esa idea, y varias se contradicen entre sí
a propósito: si el mercado premiara a la vez la continuación por volumen y el desvanecimiento por
volumen, la familia entera estaría midiendo ruido.

Limitación asumida y declarada: `Series` solo expone o/h/l/c/v, así que NO hay `taker_buy_base` ni
delta de volumen real. Donde hace falta el signo de la agresión se usa el Close Location Value
(dónde cierra la vela dentro de su propio rango), que es el mejor proxy construible con OHLCV y es
exactamente lo que asume la línea de Acumulación/Distribución. Es un proxy sesgado: en una vela con
mecha larga por liquidaciones atribuye compra agresiva donde solo hubo un barrido de stops. Esa
debilidad es parte de lo que se está poniendo a prueba, no un descuido.
"""

from __future__ import annotations

import numpy as np
import talib

from wavelab.hypotheses.base import Hypothesis, Series, register

# Convención del repo (`Bar.open_time_ms`, `Timeframe.ms`): los timestamps son epoch en
# MILISEGUNDOS sobre la rejilla UTC. El anclaje de sesión del VWAP depende de eso.
_DIA_MS = 86_400_000


# --------------------------------------------------------------------------------------------
# Utilidades. Todas estrictamente causales: la posición i solo mira posiciones j <= i.
# --------------------------------------------------------------------------------------------

def _f(a: np.ndarray) -> np.ndarray:
    """talib exige float64 contiguo."""
    return np.ascontiguousarray(a, dtype=np.float64)


def _finito(*arrays: np.ndarray) -> np.ndarray:
    """True donde TODOS los arrays tienen un número real.

    talib devuelve NaN durante el calentamiento y las divisiones por rango cero producen inf.
    Se comprueba explícitamente en vez de confiar en que `NaN > x` dé False, y sobre todo en vez
    de usar nan_to_num, que convertiría el calentamiento en señales inventadas.
    """
    ok = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        ok &= ~np.isnan(a)
        ok &= ~np.isinf(a)
    return ok


def _clv(s: Series) -> np.ndarray:
    """Close Location Value en [-1, +1]: proxy del signo de la agresión.

    +1 = cierra en el máximo (el comprador se llevó la vela), -1 = cierra en el mínimo.
    Las velas de rango cero quedan en 0, no en NaN ni en inf.
    """
    h, l, c = _f(s.high), _f(s.low), _f(s.close)
    rango = h - l
    clv = np.zeros(rango.shape, dtype=np.float64)
    np.divide((c - l) - (h - c), rango, out=clv, where=rango > 0)
    return clv


def _mantener(ev: np.ndarray, k: int) -> np.ndarray:
    """Propaga cada evento no nulo durante k barras.

    Causal por construcción: en la posición i solo puede aparecer un evento cuyo índice j cumple
    j <= i. `np.maximum.accumulate` es un barrido hacia adelante, nunca hacia atrás.
    """
    n = ev.size
    i = np.arange(n)
    idx = np.where(ev != 0, i, -1)
    ultimo = np.maximum.accumulate(idx)
    vivo = (ultimo >= 0) & ((i - ultimo) < k)
    out = np.zeros(n, dtype=np.int8)
    out[vivo] = ev[ultimo[vivo]]
    return out


def _vwap_sesion(s: Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """VWAP anclado al día UTC, su desviación típica ponderada por volumen, y cuántas velas
    llevamos desde el ancla.

    Acumula SOLO desde el inicio del día en curso hasta i incluido. La resta de sumas acumuladas
    usa el índice previo al ancla saturado en 0, de modo que nunca se indexa con -1 (que en numpy
    devolvería el ÚLTIMO elemento del array, es decir, el futuro entero).
    """
    n = len(s)
    tp = talib.TYPPRICE(_f(s.high), _f(s.low), _f(s.close))
    v = _f(s.volume)
    dia = np.asarray(s.ts, dtype=np.int64) // _DIA_MS

    nuevo = np.ones(n, dtype=bool)
    nuevo[1:] = dia[1:] != dia[:-1]
    i = np.arange(n)
    inicio = np.maximum.accumulate(np.where(nuevo, i, 0))
    hay_previo = inicio > 0
    previo = np.maximum(inicio - 1, 0)

    acc_v = np.cumsum(v)
    acc_pv = np.cumsum(tp * v)
    acc_pv2 = np.cumsum(tp * tp * v)
    vol = acc_v - np.where(hay_previo, acc_v[previo], 0.0)
    pv = acc_pv - np.where(hay_previo, acc_pv[previo], 0.0)
    pv2 = acc_pv2 - np.where(hay_previo, acc_pv2[previo], 0.0)

    vwap = np.full(n, np.nan)
    np.divide(pv, vol, out=vwap, where=vol > 0)
    m2 = np.full(n, np.nan)
    np.divide(pv2, vol, out=m2, where=vol > 0)
    var = m2 - vwap * vwap
    var[~np.isnan(var) & (var < 0.0)] = 0.0   # ruido de coma flotante, no varianza negativa
    return vwap, np.sqrt(var), (i - inicio + 1)


# --------------------------------------------------------------------------------------------
# 1. OBV contra su propia media
# --------------------------------------------------------------------------------------------

def _obv_ema21(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 60:
        return out
    obv = talib.OBV(_f(s.close), _f(s.volume))
    ema = talib.EMA(obv, 21)
    ok = _finito(obv, ema)
    out[ok & (obv > ema)] = 1
    out[ok & (obv < ema)] = -1
    return out


register(Hypothesis(
    name="flow.obv_ema21",
    family="flow",
    rationale="OBV suma el volumen entero de cada vela con el signo de su cierre, así que su "
              "pendiente mide si los contratos recientes se han movido más en velas alcistas o "
              "bajistas. El mecanismo concreto: quien compra en la subida queda con inventario "
              "que solo puede deshacer vendiendo, y mientras no lo deshaga sostiene el bid; quien "
              "vendió tiene que recomprar. Cuando la línea acumulada supera su propia media de 21 "
              "hay más inventario nuevo en manos largas del que se ha soltado, y ese desequilibrio "
              "de posicionamiento debería empujar el precio antes de resolverse.",
    prior="Esperamos ventaja positiva pequeña en régimen tendencial (4h/1d) y ventaja nula o "
          "negativa en lateral, donde OBV cruza su media constantemente sin que haya movimiento "
          "real de inventario. Falsación fuerte: si la ventaja es la MISMA en tendencia y en "
          "lateral, el mecanismo declarado es falso y esto no es más que un seguidor de tendencia "
          "lento disfrazado de volumen. También esperamos que falle en periodos dominados por "
          "cascadas de liquidación, donde OBV atribuye acumulación a lo que fue liquidación "
          "forzada.",
    fn=_obv_ema21,
    params={"ema": 21},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 2. Divergencia precio-OBV: el precio hace extremo nuevo y el volumen no acompaña
# --------------------------------------------------------------------------------------------

def _obv_divergencia(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 60:
        return out
    c = _f(s.close)
    obv = talib.OBV(c, _f(s.volume))
    max_c, min_c = talib.MAX(c, 20), talib.MIN(c, 20)
    max_o, min_o = talib.MAX(obv, 20), talib.MIN(obv, 20)
    ok = _finito(c, obv, max_c, min_c, max_o, min_o)
    ev = np.zeros(n, dtype=np.int8)
    # MAX/MIN de talib incluyen la vela actual: `c >= max_c` es exactamente "máximo de 20 nuevo".
    ev[ok & (c >= max_c) & (obv < max_o)] = -1
    ev[ok & (c <= min_c) & (obv > min_o)] = 1
    return _mantener(ev, 5)


register(Hypothesis(
    name="flow.obv_divergencia20",
    family="flow",
    rationale="Un máximo de precio que NO va acompañado de máximo de OBV significa que el último "
              "tramo lo firmaron menos contratos que el anterior: el precio sube porque se ha "
              "retirado oferta, no porque haya entrado demanda nueva. Un libro que sube por "
              "ausencia de vendedores es fino, y basta una orden de tamaño para atravesarlo en "
              "sentido contrario. La lectura simétrica vale en mínimos: precio más bajo con OBV "
              "que ya no lo acompaña indica que la venta se ha quedado sin munición.",
    prior="Esperamos ventaja positiva a horizonte corto (el evento se mantiene 5 velas) y que se "
          "concentre en extremos de rango, no en tendencias establecidas. Esperamos que FALLE, y "
          "que pierda dinero, en tendencias fuertes y persistentes: ahí la divergencia aparece "
          "decenas de veces seguidas mientras el precio sigue subiendo, y este es precisamente el "
          "indicador que arruina a quien intenta poner techos. Si la ventaja resultara igual de "
          "buena en tendencia que en rango, sospecharíamos del proxy antes que del mercado.",
    fn=_obv_divergencia,
    params={"ventana": 20, "barras_mantenidas": 5},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 3. La otra diagonal: OBV hace extremo nuevo y el precio todavía no
# --------------------------------------------------------------------------------------------

def _obv_anticipa(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 60:
        return out
    c = _f(s.close)
    obv = talib.OBV(c, _f(s.volume))
    max_c, min_c = talib.MAX(c, 20), talib.MIN(c, 20)
    max_o, min_o = talib.MAX(obv, 20), talib.MIN(obv, 20)
    ok = _finito(c, obv, max_c, min_c, max_o, min_o)
    ev = np.zeros(n, dtype=np.int8)
    ev[ok & (obv >= max_o) & (c < max_c)] = 1
    ev[ok & (obv <= min_o) & (c > min_c)] = -1
    return _mantener(ev, 5)


register(Hypothesis(
    name="flow.obv_anticipa20",
    family="flow",
    rationale="Es la casilla contraria de la misma tabla 2x2 que `obv_divergencia20`, y se registra "
              "aparte porque afirma un mecanismo distinto y opera en dirección opuesta: aquí el "
              "volumen acumulado marca máximo nuevo mientras el precio aún no. Eso es la firma de "
              "un comprador grande troceando su orden contra la oferta disponible: absorbe todo lo "
              "que sale sin permitir que el precio suba, porque subirlo encarecería su propia "
              "ejecución. Cuando agota esa oferta, el precio salta sin resistencia. Registrar las "
              "dos casillas por separado es lo que permite que una funcione y la otra no; unirlas "
              "en una sola hipótesis escondería ese resultado.",
    prior="Esperamos ventaja positiva pequeña y menos frecuente que la divergencia clásica. "
          "Esperamos que falle en mercados donde el volumen está dominado por market makers que "
          "reciclan inventario en segundos: ahí el OBV se dispara sin que exista ningún "
          "acumulador direccional. Si `obv_anticipa20` y `obv_divergencia20` salieran AMBAS "
          "positivas con magnitud parecida, no lo interpretaríamos como dos hallazgos sino como "
          "prueba de que lo que gana es el simple evento 'extremo de 20 velas' y no el volumen.",
    fn=_obv_anticipa,
    params={"ventana": 20, "barras_mantenidas": 5},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 4. Índice de flujo de dinero en extremos de Wilder
# --------------------------------------------------------------------------------------------

def _mfi14(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 40:
        return out
    mfi = talib.MFI(_f(s.high), _f(s.low), _f(s.close), _f(s.volume), 14)
    ok = _finito(mfi)
    out[ok & (mfi < 20.0)] = 1
    out[ok & (mfi > 80.0)] = -1
    return out


register(Hypothesis(
    name="flow.mfi14_extremos",
    family="flow",
    rationale="El MFI es un RSI en el que cada vela pesa por su volumen en dinero, de modo que "
              "solo llega a 80 cuando la subida se ha hecho con volumen creciente. Ese estado es "
              "el de un mercado en el que casi todo el que quería comprar ya ha comprado y encima "
              "lo ha hecho con tamaño: el flujo entrante se agota por falta de participantes "
              "nuevos, y la posición marginal es apalancada y reciente, o sea, frágil. Bajo 20 "
              "ocurre lo simétrico con la venta forzada.",
    prior="Esperamos ventaja positiva en régimen lateral y NEGATIVA en tendencia fuerte, porque "
          "comprar bajo 20 durante una caída en cascada es comprar delante de un tren. Como la "
          "señal es un estado y no un evento, esperamos también series largas de pérdidas "
          "consecutivas en las capitulaciones de marzo-2020 o mayo-2021. Si la ventaja global "
          "saliera positiva pero solo gracias a un puñado de rebotes enormes, lo contaremos como "
          "no concluyente y no como éxito.",
    fn=_mfi14,
    params={"periodo": 14, "sobreventa": 20, "sobrecompra": 80},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 5. Lado del VWAP de sesión
# --------------------------------------------------------------------------------------------

def _vwap_lado(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 10:
        return out
    vwap, _sd, _k = _vwap_sesion(s)
    c = _f(s.close)
    ok = _finito(vwap, c)
    out[ok & (c > vwap)] = 1
    out[ok & (c < vwap)] = -1
    return out


register(Hypothesis(
    name="flow.vwap_sesion_lado",
    family="flow",
    rationale="El VWAP de sesión no es un indicador más: es el precio contra el que se mide y se "
              "liquida la ejecución institucional, así que hay órdenes reales condicionadas a él. "
              "Un algoritmo de ejecución con orden de compra frena si el precio está por encima "
              "del VWAP y acelera si está por debajo, y las mesas evalúan al trader por esa "
              "diferencia. La consecuencia mecánica es que el lado del VWAP separa dos regímenes "
              "de oferta y demanda distintos dentro del mismo día, no solo dos rangos de precio.",
    prior="Esperamos ventaja positiva pequeña en 15m y 1h, y que se degrade al subir de "
          "timeframe, porque el ancla diaria pierde sentido cuando cada vela es un tercio de la "
          "sesión. Esperamos que falle en días sin dirección, donde el precio cruza el VWAP diez "
          "veces y solo genera coste. Falsación específica de BTC: el cripto cotiza 24/7 y no "
          "tiene una sesión institucional real, así que si el mecanismo es cierto la ventaja "
          "debería ser MENOR que la que este mismo indicador muestra en renta variable; si "
          "saliera enorme, sospecharíamos de que solo estamos midiendo momento intradía.",
    fn=_vwap_lado,
    params={"ancla": "dia_utc"},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 6. Banda de 2 sigma del VWAP: estiramiento contra el precio medio ponderado
# --------------------------------------------------------------------------------------------

def _vwap_banda(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 40:
        return out
    vwap, sd, k = _vwap_sesion(s)
    c = _f(s.close)
    ok = _finito(vwap, sd, c) & (sd > 0.0) & (k >= 12)
    out[ok & (c > vwap + 2.0 * sd)] = -1
    out[ok & (c < vwap - 2.0 * sd)] = 1
    return out


register(Hypothesis(
    name="flow.vwap_banda2sigma",
    family="flow",
    rationale="Usa el mismo indicador que `vwap_sesion_lado` pero afirma el mecanismo CONTRARIO, y "
              "por eso es una hipótesis separada y no una variante: aquí no importa de qué lado "
              "está el precio sino cuánto se ha alejado, medido en desviaciones típicas del propio "
              "reparto de volumen de la sesión. Dos sigmas por encima significa que el precio "
              "actual lo ha pagado una fracción minúscula del volumen del día: casi nadie tiene "
              "posición ahí, y el inventario del creador de mercado que ha absorbido esa subida "
              "está corto y necesita que el precio vuelva a la zona donde se cruzó el grueso del "
              "papel. El umbral de 2 sigmas es el convencional de Bollinger, no un valor buscado.",
    prior="Esperamos ventaja positiva en 15m y 1h dentro de sesiones sin noticia, y pérdidas "
          "claras los días de ruptura, en los que el precio pasa la banda por la mañana y no "
          "vuelve. Como el desvanecimiento gana muchas veces poco y pierde pocas veces mucho, "
          "exigimos que la ventaja sobreviva al examen de la cola: si el resultado depende de no "
          "haber sufrido un 12 de marzo de 2020, es un fracaso, no un éxito. Exigimos al menos 12 "
          "velas de sesión antes de emitir señal porque con menos la sigma es una estimación de "
          "tres puntos; es un mínimo estadístico, no un umbral ajustado.",
    fn=_vwap_banda,
    params={"ancla": "dia_utc", "sigmas": 2.0, "min_barras_sesion": 12},
    timeframes=("15m", "1h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 7. Impulso: esfuerzo alto CON resultado (volumen doble y rango expansivo)
# --------------------------------------------------------------------------------------------

def _impulso_volumen(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 60:
        return out
    h, l, c, v = _f(s.high), _f(s.low), _f(s.close), _f(s.volume)
    media_v = talib.SMA(v, 20)
    atr = talib.ATR(h, l, c, 14)
    tr = talib.TRANGE(h, l, c)
    clv = _clv(s)
    rel = np.full(n, np.nan)
    np.divide(v, media_v, out=rel, where=~np.isnan(media_v) & (media_v > 0.0))
    ok = _finito(rel, atr, tr) & (atr > 0.0)
    fuerte = ok & (rel > 2.0) & (tr > atr)
    ev = np.zeros(n, dtype=np.int8)
    ev[fuerte & (clv > 0.5)] = 1
    ev[fuerte & (clv < -0.5)] = -1
    return _mantener(ev, 5)


register(Hypothesis(
    name="flow.impulso_volumen",
    family="flow",
    rationale="Una vela con el doble del volumen medio, rango mayor que el ATR y cierre en el "
              "cuarto superior de ese rango solo se produce cuando alguien cruza el spread "
              "repetidamente y se lleva por delante varios niveles del libro. Quien hace eso está "
              "pagando por inmediatez, lo que revela que su información o su urgencia valen más "
              "que el coste, y además deja el libro vaciado por ese lado: los siguientes niveles "
              "hay que reponerlos más arriba. Los stops de los cortos atrapados en el recorrido "
              "aportan compra forzada añadida durante las velas siguientes.",
    prior="Esperamos ventaja positiva a 5 velas y que sea MAYOR en 15m que en 4h, porque el "
          "vaciado de libro se repone en minutos u horas, no en días. Esperamos que falle, o se "
          "invierta, cuando la vela de impulso es la última de un tramo largo (agotamiento) y en "
          "los minutos posteriores a una publicación macro, donde el rango expansivo es "
          "reprecio instantáneo sin continuación. Si la ventaja fuera igual en 4h y en 15m, el "
          "mecanismo de reposición de libro que afirmamos sería falso.",
    fn=_impulso_volumen,
    params={"volumen_relativo": 2.0, "media_volumen": 20, "atr": 14, "clv": 0.5,
            "barras_mantenidas": 5},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 8. Clímax: volumen a 3 sigmas en un extremo de 20 velas, a contrapelo
# --------------------------------------------------------------------------------------------

def _climax_volumen(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 140:
        return out
    c, v = _f(s.close), _f(s.volume)
    lv = np.log1p(v)   # log1p y no log: el volumen puede ser exactamente cero
    media = talib.SMA(lv, 100)
    sigma = talib.STDDEV(lv, 100)
    z = np.full(n, np.nan)
    np.divide(lv - media, sigma, out=z, where=~np.isnan(sigma) & (sigma > 0.0))
    max_c, min_c = talib.MAX(c, 20), talib.MIN(c, 20)
    ok = _finito(z, max_c, min_c, c)
    extremo = ok & (z > 3.0)
    ev = np.zeros(n, dtype=np.int8)
    ev[extremo & (c >= max_c)] = -1
    ev[extremo & (c <= min_c)] = 1
    return _mantener(ev, 5)


register(Hypothesis(
    name="flow.climax_volumen",
    family="flow",
    rationale="Un volumen a más de tres sigmas de su media móvil de 100, ocurriendo justo en el "
              "extremo de 20 velas, no es participación: es transferencia. En cripto ese pico "
              "casi siempre es una cascada de liquidaciones, en la que el motor del exchange "
              "envía órdenes a mercado que no representan a nadie que quiera operar a ese precio. "
              "Cuando la cola de liquidaciones se vacía desaparece de golpe toda esa oferta "
              "involuntaria, y el precio vuelve al nivel donde estaba el libro real. La media y la "
              "sigma son móviles a 100 velas, nunca del histórico completo.",
    prior="Esperamos ventaja positiva a 5 velas, concentrada en un número muy pequeño de eventos "
          "(quizá 30-80 en todo el histórico por timeframe), lo que de entrada limita la potencia "
          "estadística: si sale positiva pero con n < 30 la declararemos no concluyente. "
          "Esperamos que falle cuando el pico de volumen es el ARRANQUE de una expansión de "
          "régimen y no su final —una ruptura de rango con noticia detrás—, donde desvanecer es "
          "ponerse delante del movimiento entero. También esperamos que empeore a partir de 2021, "
          "según los exchanges han ido introduciendo motores de liquidación parcial que suavizan "
          "las cascadas.",
    fn=_climax_volumen,
    params={"z_volumen": 3.0, "ventana_z": 100, "ventana_extremo": 20, "barras_mantenidas": 5},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 9. Oscilador de Chaikin: momento del delta de volumen aproximado
# --------------------------------------------------------------------------------------------

def _chaikin(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 40:
        return out
    osc = talib.ADOSC(_f(s.high), _f(s.low), _f(s.close), _f(s.volume),
                      fastperiod=3, slowperiod=10)
    ok = _finito(osc)
    out[ok & (osc > 0.0)] = 1
    out[ok & (osc < 0.0)] = -1
    return out


register(Hypothesis(
    name="flow.chaikin_osc_3_10",
    family="flow",
    rationale="La línea de Acumulación/Distribución es el delta de volumen acumulado que se puede "
              "aproximar sin datos de agresor: reparte el volumen de cada vela entre compra y "
              "venta según dónde cierra dentro del rango, que es justo lo que mediría "
              "`taker_buy_base` si `Series` lo expusiera. El oscilador de Chaikin es su momento "
              "(EMA 3 menos EMA 10), o sea, si el delta acumulado se está acelerando. Un delta "
              "acelerando significa que el desequilibrio entre agresión compradora y vendedora "
              "está creciendo, y ese desequilibrio es lo que mueve el precio a corto plazo.",
    prior="Esperamos ventaja positiva pequeña, y menor que la de `obv_ema21`, porque el reparto "
          "por posición del cierre penaliza precisamente las velas con mecha, que en BTC son las "
          "informativas. Esperamos que falle sistemáticamente en velas de mecha larga por barrido "
          "de stops, donde el cierre vuelve al centro del rango y el indicador registra "
          "'indecisión' en el momento en que más flujo direccional ha habido. Si saliera mejor "
          "que OBV, el proxy de agresión por CLV sería más informativo de lo que creemos y habría "
          "que revisar toda la familia.",
    fn=_chaikin,
    params={"rapida": 3, "lenta": 10},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 10. Tendencia SOLO cuando la participación se expande
# --------------------------------------------------------------------------------------------

def _tendencia_con_participacion(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 140:
        return out
    c, v = _f(s.close), _f(s.volume)
    roc = talib.ROC(c, 20)
    v20, v100 = talib.SMA(v, 20), talib.SMA(v, 100)
    ok = _finito(roc, v20, v100)
    expande = ok & (v20 > v100)
    out[expande & (roc > 0.0)] = 1
    out[expande & (roc < 0.0)] = -1
    return out


register(Hypothesis(
    name="flow.tendencia_con_participacion",
    family="flow",
    rationale="Afirma que el volumen no da dirección pero sí da permiso. Una tendencia sostenida "
              "necesita un flujo continuo de participantes nuevos que compren más caro de lo que "
              "compró el anterior; si el volumen medio de 20 cae por debajo del de 100, los que "
              "quedan operando son los que ya están dentro, rotando entre ellos, y no hay quien "
              "absorba la primera oleada de tomas de beneficio. La señal es el signo del "
              "rendimiento de 20 velas condicionado a que la participación se expanda, y cero "
              "cuando no.",
    prior="Esperamos ventaja positiva, pero la prueba real NO es que sea positiva: es que sea "
          "MAYOR que la del momento simple a 20 velas sin filtro. NOTA DE AUDITORÍA (2026-09-08): "
          "ese control no estaba registrado en ninguna familia —la familia `trend` no registra un "
          "ROC de 20— así que este criterio de falsación no se podía ejecutar; se ha registrado "
          "como `flow.roc20_sin_filtro` y es contra él contra quien debe medirse. "
          "Si el filtro de volumen no aporta nada sobre ese control, esta hipótesis queda "
          "refutada aunque gane dinero, porque lo que estaría ganando es el momento y no el "
          "flujo. Esperamos además que el filtro perjudique en los suelos de mercado bajista, "
          "donde el volumen se seca durante meses y el filtro deja fuera precisamente el inicio "
          "del siguiente ciclo alcista.",
    fn=_tendencia_con_participacion,
    params={"roc": 20, "volumen_corto": 20, "volumen_largo": 100},
    timeframes=("4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 11. Movimiento en volumen fino: desvanecer
# --------------------------------------------------------------------------------------------

def _movimiento_sin_volumen(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 40:
        return out
    c, v = _f(s.close), _f(s.volume)
    v5, v20 = talib.SMA(v, 5), talib.SMA(v, 20)
    roc = talib.ROC(c, 5)
    ok = _finito(v5, v20, roc) & (v20 > 0.0)
    seco = ok & (v5 < v20)
    out[seco & (roc > 0.0)] = -1
    out[seco & (roc < 0.0)] = 1
    return out


register(Hypothesis(
    name="flow.movimiento_sin_volumen",
    family="flow",
    rationale="El reverso exacto de `tendencia_con_participacion`, y por eso se registra: un "
              "movimiento de cinco velas hecho con volumen por debajo de su propia media de 20 no "
              "ha transferido inventario, solo ha desplazado el precio a través de un libro "
              "vacío. Nadie ha tenido que aceptar una posición grande en el lado equivocado, así "
              "que no hay ningún participante obligado a defender el nivel nuevo, y la primera "
              "orden de tamaño que aparezca lo devolverá al sitio. El umbral es 'por debajo de la "
              "media', sin constante libre que ajustar.",
    prior="Esperamos ventaja positiva pequeña en 15m/1h, sobre todo en fines de semana y en las "
          "horas asiáticas, cuando el libro de BTC es más fino. Esperamos que falle en los "
          "arranques lentos de tendencia, en los que el precio sube semanas con volumen "
          "decreciente y desvanecer es perder de forma continuada; ese es el modo de fallo que "
          "más nos preocupa porque es persistente y no ruidoso. Si esta hipótesis y "
          "`tendencia_con_participacion` salieran ambas positivas en el mismo régimen, "
          "concluiríamos que el filtro de volumen no separa nada y que ganan por el signo del "
          "momento, no por el flujo.",
    fn=_movimiento_sin_volumen,
    params={"volumen_corto": 5, "volumen_largo": 20, "roc": 5},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 12. Absorción: esfuerzo alto SIN resultado (volumen doble y rango estrecho)
# --------------------------------------------------------------------------------------------

def _absorcion(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 60:
        return out
    h, l, c, v = _f(s.high), _f(s.low), _f(s.close), _f(s.volume)
    media_v = talib.SMA(v, 20)
    atr = talib.ATR(h, l, c, 14)
    tr = talib.TRANGE(h, l, c)
    clv = _clv(s)
    rel = np.full(n, np.nan)
    np.divide(v, media_v, out=rel, where=~np.isnan(media_v) & (media_v > 0.0))
    ok = _finito(rel, atr, tr) & (atr > 0.0)
    absorbe = ok & (rel > 2.0) & (tr < atr)
    ev = np.zeros(n, dtype=np.int8)
    ev[absorbe & (clv > 0.0)] = 1
    ev[absorbe & (clv < 0.0)] = -1
    return _mantener(ev, 5)


register(Hypothesis(
    name="flow.absorcion_rango_estrecho",
    family="flow",
    rationale="Es la casilla complementaria y disjunta de `impulso_volumen` en la tabla "
              "esfuerzo/resultado de Wyckoff: mismo volumen doble, pero rango MENOR que el ATR. "
              "Que se crucen el doble de contratos de lo normal y el precio no se mueva solo tiene "
              "una explicación mecánica: hay una orden pasiva grande a un lado del libro "
              "reponiéndose tan rápido como se la comen. Quien tiene tamaño para hacer eso conoce "
              "su precio objetivo y no ha terminado; cuando el agresor se cansa, el precio se "
              "desplaza hacia el lado del absorbedor, que es el lado donde ha cerrado la vela. La "
              "dirección la da el signo del cierre dentro del rango, sin umbral libre.",
    prior="Esperamos ventaja positiva pequeña a 5 velas, y menor frecuencia que el impulso. "
          "Esperamos que falle en zonas de valor aceptado —el centro de un rango largo—, donde "
          "volumen alto y rango estrecho es simplemente el equilibrio normal del mercado y no hay "
          "ningún absorbedor. Al ser disjunta de `impulso_volumen` por construcción, la "
          "comparación entre ambas es limpia: si las dos salen positivas, el volumen relativo "
          "alto vale por sí solo y el rango no aporta; si solo una, el eje esfuerzo/resultado es "
          "real.",
    fn=_absorcion,
    params={"volumen_relativo": 2.0, "media_volumen": 20, "atr": 14, "barras_mantenidas": 5},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 13. Índice de Volumen Negativo (Fosback): qué hace el precio los días de volumen fino
# --------------------------------------------------------------------------------------------

def _nvi_fosback(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 400:
        return out
    c, v = _f(s.close), _f(s.volume)
    ret = np.zeros(n, dtype=np.float64)
    anterior = c[:-1]
    np.divide(c[1:] - anterior, anterior, out=ret[1:], where=anterior > 0.0)
    baja_volumen = np.zeros(n, dtype=bool)
    baja_volumen[1:] = v[1:] < v[:-1]
    factor = np.where(baja_volumen, 1.0 + ret, 1.0)
    factor = np.clip(factor, 0.01, None)   # ninguna vela puede llevar el índice a cero o negativo
    nvi = 1000.0 * np.cumprod(factor)
    ema = talib.EMA(nvi, 255)
    ok = _finito(nvi, ema)
    out[ok & (nvi > ema)] = 1
    return out


register(Hypothesis(
    name="flow.nvi_fosback",
    family="flow",
    rationale="El NVI acumula el rendimiento SOLO de las velas cuyo volumen baja respecto a la "
              "anterior, aislando lo que hace el precio cuando el público no está operando. La "
              "premisa de Fosback es que el minorista aparece con el volumen y el dinero informado "
              "no lo necesita: si el precio sube en los días tranquilos, alguien está construyendo "
              "posición sin querer llamar la atención. La señal es asimétrica a propósito, como en "
              "el original: +1 cuando el NVI está por encima de su EMA de 255 (un año de sesiones) "
              "y 0 —no corto— por debajo, porque Fosback nunca afirmó que lo contrario indicase "
              "mercado bajista, solo ausencia de información.",
    prior="Esperamos ventaja positiva modesta en 1d y una tasa de exposición alta (debería estar "
          "dentro la mayor parte del mercado alcista). El modo de fallo esperado es doble: la "
          "distinción minorista/informado se construyó sobre bolsas con horario y ruedas de "
          "prensa, y BTC cotiza 24/7 con un volumen dominado por market makers y arbitraje entre "
          "exchanges, así que puede que la partición por volumen no separe a nadie. Además, con "
          "solo 1d y ~600 velas de calentamiento quedan del orden de 2.500 observaciones y "
          "poquísimos ciclos completos: si la ventaja sale positiva pero depende de haber estado "
          "dentro en 2020-2021, es una observación, no evidencia.",
    fn=_nvi_fosback,
    params={"ema": 255},
    timeframes=("1d",),
    min_warmup=600,
))


# --------------------------------------------------------------------------------------------
# 14. CONTROL AÑADIDO EN AUDITORÍA (2026-09-08).
#
# Qué estaba mal. Dos hipótesis declaraban su criterio de falsación contra un control inexistente:
# `flow.tendencia_con_participacion` dice que "la prueba real NO es que sea positiva: es que sea
# MAYOR que la del momento simple a 20 velas sin filtro que registra la familia `trend`", y
# `momentum.momento_con_volumen` dice que "queda falsada si el filtro de volumen no mejora al ROC
# de 20 sin filtrar". Ni `trend` ni `momentum` registraban un ROC de 20 sin filtrar: `momentum`
# registra `roc10`, que es otro horizonte y por tanto otro ensayo.
#
# Por qué importaba. Las dos hipótesis afirman lo mismo —que el volumen da PERMISO aunque no dé
# dirección— y las dos condicionan el mismo estadístico (el signo del rendimiento de 20 velas) a
# una expansión de participación. Esa es una afirmación sobre un incremento, y sin el brazo sin
# filtrar solo se podía juzgar la rentabilidad absoluta, que en un activo con deriva secular
# positiva confirma casi cualquier regla mayoritariamente larga. Con el control registrado, las dos
# pasan a ser comprobables y las tres cuentan en la corrección por contraste múltiple.
#
# Además queda declarado aquí, para quien haga la corrección: `flow.tendencia_con_participacion`
# (ROC-20 con SMA-20 > SMA-100 de volumen) y `momentum.momento_con_volumen` (ROC-20 con SMA-5 >
# SMA-20 de volumen) son la MISMA construcción con distinta pareja de medias de volumen. Medidas
# sobre datos comparten poco (índice de Jaccard ≈ 0,15 en los eventos), así que no son duplicados y
# no se elimina ninguna, pero tampoco son ensayos independientes: son dos lecturas del mismo
# mecanismo y deben contarse como ensayos dependientes, no como dos confirmaciones cruzadas entre
# familias si ambas salen positivas.
# --------------------------------------------------------------------------------------------

def _roc20_sin_filtro(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < 40:
        return out
    roc = talib.ROC(_f(s.close), 20)
    ok = _finito(roc)
    out[ok & (roc > 0.0)] = 1
    out[ok & (roc < 0.0)] = -1
    return out


register(Hypothesis(
    name="flow.roc20_sin_filtro",
    family="flow",
    rationale="Control incondicional de `tendencia_con_participacion` y de "
              "`momentum.momento_con_volumen`: el signo del rendimiento de 20 velas, sin ninguna "
              "condición sobre el volumen. No afirma un mecanismo de flujo —no puede, porque no "
              "mira el volumen— y ese es exactamente su papel. Las dos hipótesis que controla no "
              "apuestan a que el momento de 20 velas gane, sino a que gane MÁS cuando la "
              "participación se expande; esa es una afirmación sobre una diferencia, y una "
              "diferencia necesita los dos brazos. Se registra en `flow` porque su única razón de "
              "existir es servir de denominador al filtro de volumen de esta familia.",
    prior="Esperamos ventaja positiva pequeña, y esperamos que sea MENOR que la de "
          "`tendencia_con_participacion` en 4h y 1d. Toda la información está en la comparación: si "
          "este control iguala o supera a las versiones filtradas, el volumen no da permiso, las "
          "dos hipótesis filtradas quedan refutadas aunque ganen dinero, y la tesis de esta familia "
          "—que el volumen mide convicción— pierde su apoyo principal. Por sí solo, un resultado "
          "positivo de este control NO es un hallazgo: es la deriva secular de BTC leída por el "
          "signo de un rendimiento pasado, y así debe publicarse.",
    fn=_roc20_sin_filtro,
    params={"roc": 20, "filtro_volumen": "ninguno"},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))
