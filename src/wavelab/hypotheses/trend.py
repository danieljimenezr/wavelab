"""Familia `trend`: seguimiento de tendencia. REGISTRO PREVIO — escrito antes de mirar un número.

Ninguna de estas doce hipótesis se ha ejecutado. Los parámetros son los convencionales de la
literatura (EMA 21/55/200, MACD 12/26/9, ADX/DMI 14, ATR 14, Donchian 55/20 del Sistema 2 de las
Turtles, Ichimoku 9/26/52, Aroon 25, SAR 0.02/0.2). No hay variantes del mismo parámetro: donde
aparecen dos números (55/20 de Donchian, 3xATR de Supertrend) son un par publicado como unidad, y
se justifica en el `rationale` por qué son una hipótesis y no una rejilla.

La pregunta común de la familia es una sola: **¿por qué persistiría una tendencia?** Cada hipótesis
propone un mecanismo DISTINTO (troceo de órdenes, acuerdo entre horizontes, punto focal reflexivo,
aceleración de flujo, resolución de dispersión, cascada de stops, distancia del stop, renovación de
extremos, deriva sobre ruido, horizonte de asignación). Varias predicen resultados que se
CONTRADICEN entre sí (p. ej. `supertrend` vs `psar` sobre si conviene dar aire o apretar el stop).
Eso es deliberado: si todas pudieran ganar a la vez, ninguna sería falsable.

Causalidad. Todo indicador aquí es una función de ventana hacia atrás. El único desplazamiento que
se usa es `_lag(a, k)` con k >= 1, que trae el valor de hace k barras al índice actual; no existe
ningún desplazamiento negativo, ningún `np.roll`, ningún detector de extremos definido contra el
array entero. Dos sitios lo merecen por escrito:

  - Donchian: `talib.MAX(high, 55)[i]` INCLUYE la barra i, así que comparar `close[i]` contra él
    no rompería, pero no rompe nada por el motivo equivocado (`high[i] >= close[i]` siempre). El
    canal de referencia se retrasa una barra para que sea el máximo de las 55 barras ANTERIORES.
  - Ichimoku: la nube va desplazada +26. Eso significa que el nivel de HOY se calculó con datos de
    hace 26 barras. Es información pasada dibujada hacia adelante, jamás futura.

NaN. talib devuelve NaN durante el calentamiento. Aquí no se usa `nan_to_num` en ningún sitio: se
construye una máscara explícita con `np.isfinite` y las señales solo se escriben donde TODOS los
insumos son finitos. Rellenar un NaN con 0 o con 1e18 inventa señales en el arranque de la serie.
"""

from __future__ import annotations

import numpy as np
import talib

from wavelab.hypotheses.base import Hypothesis, Series, register

# --------------------------------------------------------------------------------------------
# utilidades
# --------------------------------------------------------------------------------------------


def _f64(a: np.ndarray) -> np.ndarray:
    """talib exige float64 contiguo."""
    return np.ascontiguousarray(a, dtype=np.float64)


def _lag(a: np.ndarray, k: int) -> np.ndarray:
    """Valor de hace `k` barras, leído en la barra actual. k >= 1, SIEMPRE hacia el pasado.

    La cabeza queda en NaN a propósito: no hay valor pasado que traer, y cualquier relleno
    fabricaría señal donde no hay información.
    """
    if k < 1:
        raise ValueError("_lag solo desplaza hacia el pasado (k >= 1)")
    out = np.full(a.shape, np.nan, dtype=np.float64)
    if k < a.size:
        out[k:] = a[: a.size - k]
    return out


def _ok(*arrays: np.ndarray) -> np.ndarray:
    """Máscara de posiciones donde todos los insumos son finitos."""
    m = np.isfinite(arrays[0])
    for a in arrays[1:]:
        m &= np.isfinite(a)
    return m


# --------------------------------------------------------------------------------------------
# 1. cruce de dos medias: troceo de órdenes
# --------------------------------------------------------------------------------------------


def _ema_cross_21_55(s: Series) -> np.ndarray:
    c = _f64(s.close)
    out = np.zeros(c.size, dtype=np.int8)
    fast = talib.EMA(c, 21)
    slow = talib.EMA(c, 55)
    ok = _ok(fast, slow)
    out[ok & (fast > slow)] = 1
    out[ok & (fast < slow)] = -1
    return out


register(Hypothesis(
    name="trend.ema_cross_21_55",
    family="trend",
    rationale="La información nueva no se incorpora en una vela porque el participante grande no "
              "puede ejecutar en una vela: un mandato de tamaño se trocea durante días para no "
              "mover el precio contra uno mismo, de modo que el comprador de hoy es también el "
              "comprador de mañana. Ese troceo deja autocorrelación positiva de la deriva a escala "
              "de semanas. El cruce EMA21/EMA55 no predice nada: solo declara que el desequilibrio "
              "de flujo sigue activo, y cobra mientras el mismo participante siga ejecutando.",
    prior="Esperamos ventaja positiva en régimen tendencial y NEGATIVA en lateral, con la pérdida "
          "concentrada en pocas semanas de sierra. Si el edge también fuese positivo en lateral, "
          "el mecanismo del troceo sería falso y estaríamos midiendo otra cosa. Predicción cruzada "
          "falsable: el efecto debe degradarse al bajar de timeframe, porque la ejecución "
          "institucional troceada no vive a escala de 15m.",
    fn=_ema_cross_21_55,
    params={"rapida": 21, "lenta": 55},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=250,
))


# --------------------------------------------------------------------------------------------
# 2. alineación de tres horizontes: retirada de la liquidez contraria
# --------------------------------------------------------------------------------------------


def _ema_stack_21_55_200(s: Series) -> np.ndarray:
    c = _f64(s.close)
    out = np.zeros(c.size, dtype=np.int8)
    e21 = talib.EMA(c, 21)
    e55 = talib.EMA(c, 55)
    e200 = talib.EMA(c, 200)
    ok = _ok(e21, e55, e200)
    out[ok & (e21 > e55) & (e55 > e200)] = 1
    out[ok & (e21 < e55) & (e55 < e200)] = -1
    return out


register(Hypothesis(
    name="trend.ema_stack_21_55_200",
    family="trend",
    rationale="Cada horizonte pertenece a un participante distinto: intradía, swing y asignador. "
              "Mientras discrepan, el que está en contra provee la liquidez que absorbe al que "
              "está a favor y el precio revierte. Cuando los tres coinciden no queda nadie "
              "estructuralmente obligado a vender contra la subida, la profundidad del lado "
              "contrario se retira y el MISMO flujo mueve más precio. La hipótesis no es sobre "
              "medias sino sobre ausencia de contraparte natural.",
    prior="Esperamos muchas menos barras en mercado que trend.ema_cross_21_55 y mejor rendimiento "
          "POR BARRA. Falsable de dos maneras: si el tiempo en mercado no cae de forma clara, la "
          "condición de acuerdo no está filtrando nada; y si el rendimiento por barra no supera al "
          "del cruce simple, la 'ausencia de contraparte' no aporta. Esperamos que quede plana y "
          "pierda todo el arranque en los cambios de régimen, porque la EMA200 se alinea tarde.",
    fn=_ema_stack_21_55_200,
    params={"corta": 21, "media": 55, "larga": 200},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=600,
))


# --------------------------------------------------------------------------------------------
# 3. la media de 200: punto focal reflexivo
# --------------------------------------------------------------------------------------------


def _ema200_filter(s: Series) -> np.ndarray:
    c = _f64(s.close)
    out = np.zeros(c.size, dtype=np.int8)
    e200 = talib.EMA(c, 200)
    ok = _ok(c, e200)
    out[ok & (c > e200)] = 1
    out[ok & (c < e200)] = -1
    return out


register(Hypothesis(
    name="trend.ema200_filter",
    family="trend",
    rationale="La media de 200 no mide ninguna propiedad física del mercado: es un punto focal "
              "público que mesas de riesgo, medios y bots de asignación citan a diario. Su poder, "
              "si existe, es enteramente reflexivo — hay mandatos reales que reducen exposición "
              "'por debajo de la 200', y esa reducción es flujo de verdad ejecutado en el nivel. "
              "Se registra el nivel puro y no la pendiente porque para una EMA200 la pendiente es "
              "casi una función del propio cruce, y añadirla sería el mismo ensayo dos veces.",
    prior="La predicción que importa es cruzada, no de rentabilidad: si el mecanismo es reflexivo, "
          "el efecto debe ser MAYOR en 1d, que es donde el nivel se publica y se mira, y casi nulo "
          "en 15m y 1h. Un edge igual o superior en 15m falsaría la explicación aunque el número "
          "saliera rentable, y en ese caso la hipótesis debe darse por fallida pese a ganar "
          "dinero. Esperamos edge nulo o negativo en periodos sin tendencia secular.",
    fn=_ema200_filter,
    params={"periodo": 200},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=600,
))


# --------------------------------------------------------------------------------------------
# 4. MACD: aceleración del desequilibrio
# --------------------------------------------------------------------------------------------


def _macd_12_26_9(s: Series) -> np.ndarray:
    c = _f64(s.close)
    out = np.zeros(c.size, dtype=np.int8)
    macd, signal, _ = talib.MACD(c, 12, 26, 9)
    ok = _ok(macd, signal)
    out[ok & (macd > signal)] = 1
    out[ok & (macd < signal)] = -1
    return out


register(Hypothesis(
    name="trend.macd_12_26_9",
    family="trend",
    rationale="El MACD contra su línea de señal mide la ACELERACIÓN del desequilibrio de flujo, no "
              "su nivel. Los programas sistemáticos de momento escalan tamaño de forma continua en "
              "función de la señal, no binaria: cuando la deriva se acelera añaden posición, y ese "
              "añadido es literalmente la deriva de mañana. La apuesta es que ese bucle de "
              "realimentación se detecta en la segunda derivada antes que en el precio.",
    prior="Esperamos más cambios de posición y menor acierto por operación que "
          "trend.ema_cross_21_55, y una ventaja BRUTA similar que puede desaparecer en neto tras "
          "costes. Falsable de forma directa: si el MACD no supera al cruce de medias en neto, "
          "'anticipar por aceleración' no tiene contenido y es una media cara. Esperamos edge "
          "claramente negativo en lateral de baja volatilidad, donde la aceleración es solo ruido.",
    fn=_macd_12_26_9,
    params={"rapida": 12, "lenta": 26, "senal": 9},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=250,
))


# --------------------------------------------------------------------------------------------
# 5. ADX/DMI: la dispersión de opiniones se ha resuelto
# --------------------------------------------------------------------------------------------


def _adx_dmi_14(s: Series) -> np.ndarray:
    h, l, c = _f64(s.high), _f64(s.low), _f64(s.close)
    out = np.zeros(c.size, dtype=np.int8)
    adx = talib.ADX(h, l, c, 14)
    pdi = talib.PLUS_DI(h, l, c, 14)
    mdi = talib.MINUS_DI(h, l, c, 14)
    ok = _ok(adx, pdi, mdi)
    fuerte = ok & (adx >= 25.0)
    out[fuerte & (pdi > mdi)] = 1
    out[fuerte & (mdi > pdi)] = -1
    return out


register(Hypothesis(
    name="trend.adx_dmi_14",
    family="trend",
    rationale="Separa dos preguntas que las medias mezclan: ¿hay tendencia? (ADX) y ¿en qué "
              "dirección? (DMI). El mecanismo propuesto es que la persistencia solo aparece "
              "cuando la dispersión de opiniones ya se ha resuelto: si la mayor parte del volumen "
              "agresivo llega por el mismo lado, el creador de mercado acumula inventario "
              "direccional y se cubre EN LA MISMA DIRECCIÓN, amplificando el movimiento que le "
              "perjudica. Por debajo de ADX 25 hay dos bandos de tamaño parecido y el flujo se "
              "cancela sin dejar deriva.",
    prior="Esperamos que la ventaja venga de EVITAR el lateral, no de acertar mejor la dirección. "
          "Contraste falsable y explícito: el rendimiento por barra DENTRO del filtro debe superar "
          "al de las barras que el filtro descarta; si no lo hace, el ADX no aporta información y "
          "la hipótesis muere aunque el resultado global sea positivo. Esperamos que entre tarde y "
          "ceda una parte grande del inicio de cada tendencia, porque el ADX necesita movimiento "
          "consumado para subir de 25.",
    fn=_adx_dmi_14,
    params={"periodo": 14, "umbral_adx": 25},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=250,
))


# --------------------------------------------------------------------------------------------
# 6. Donchian 55/20: cascada de stops
# --------------------------------------------------------------------------------------------


def _donchian_turtle_55_20(s: Series) -> np.ndarray:
    h, l, c = _f64(s.high), _f64(s.low), _f64(s.close)
    n = c.size
    out = np.zeros(n, dtype=np.int8)
    # +1 barra de retraso: canal de las 55 (o 20) barras ANTERIORES, sin incluir la actual.
    hh_ent = _lag(talib.MAX(h, 55), 1)
    ll_ent = _lag(talib.MIN(l, 55), 1)
    hh_sal = _lag(talib.MAX(h, 20), 1)
    ll_sal = _lag(talib.MIN(l, 20), 1)
    ok = _ok(hh_ent, ll_ent, hh_sal, ll_sal, c)
    pos = 0
    for i in range(n):
        if not ok[i]:
            pos = 0
            out[i] = 0
            continue
        if pos == 0:
            if c[i] > hh_ent[i]:
                pos = 1
            elif c[i] < ll_ent[i]:
                pos = -1
        elif pos == 1:
            if c[i] < ll_sal[i]:
                pos = -1 if c[i] < ll_ent[i] else 0
        else:
            if c[i] > hh_sal[i]:
                pos = 1 if c[i] > hh_ent[i] else 0
        out[i] = pos
    return out


register(Hypothesis(
    name="trend.donchian_turtle_55_20",
    family="trend",
    rationale="Por encima de un máximo de 55 barras se acumulan stops de cortos y órdenes de "
              "compra por rotura. Su ejecución es un shock de demanda mecánico e insensible al "
              "precio, y el libro está fino justo ahí precisamente porque nadie ha querido vender "
              "a ese nivel todavía; cada stop ejecutado empuja hacia el siguiente. El par 55/20 es "
              "el Sistema 2 de las Turtles publicado como unidad, no una rejilla: la salida corta "
              "existe para que la posición muera antes que el capital, y separar los dos números "
              "sería otra hipótesis distinta.",
    prior="Esperamos una distribución muy asimétrica: mayoría de operaciones perdedoras pequeñas y "
          "una cola derecha que sostiene todo el resultado, es decir MEDIA positiva con MEDIANA "
          "negativa. Si la mediana sale positiva, lo que actúa no es la cascada de stops sino otra "
          "cosa, y el mecanismo declarado queda refutado. Esperamos fracaso claro en mercados de "
          "alta volatilidad sin dirección, donde la rotura falsa es la norma.",
    fn=_donchian_turtle_55_20,
    params={"entrada": 55, "salida": 20},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=250,
))


# --------------------------------------------------------------------------------------------
# 7. Supertrend: el stop donde lo pondría una mesa de riesgo
# --------------------------------------------------------------------------------------------


def _supertrend_atr14_x3(s: Series) -> np.ndarray:
    h, l, c = _f64(s.high), _f64(s.low), _f64(s.close)
    n = c.size
    out = np.zeros(n, dtype=np.int8)
    atr = talib.ATR(h, l, c, 14)
    hl2 = (h + l) / 2.0
    banda_sup = hl2 + 3.0 * atr
    banda_inf = hl2 - 3.0 * atr
    ok = _ok(banda_sup, banda_inf, c)
    sup = np.nan
    inf = np.nan
    pos = 0
    for i in range(1, n):
        if not ok[i]:
            continue
        if not np.isfinite(sup):
            sup, inf = banda_sup[i], banda_inf[i]
            continue
        # trinquete: la banda solo se aprieta, y se libera únicamente cuando el precio la rompe.
        if banda_sup[i] < sup or c[i - 1] > sup:
            sup = banda_sup[i]
        if banda_inf[i] > inf or c[i - 1] < inf:
            inf = banda_inf[i]
        if c[i] > sup:
            pos = 1
        elif c[i] < inf:
            pos = -1
        out[i] = pos
    return out


register(Hypothesis(
    name="trend.supertrend_atr14_x3",
    family="trend",
    rationale="Un umbral de invalidación fijo en porcentaje es incoherente entre regímenes de "
              "volatilidad: el mismo 2% es ruido en un régimen y una señal en otro. Supertrend "
              "coloca el giro a 3 ATR del punto medio, que es aproximadamente donde una mesa de "
              "riesgo pone el stop — lo bastante lejos para que el ruido normal no lo toque. Si "
              "los stops reales se agrupan a esa distancia, el nivel deja de ser una línea "
              "dibujada y pasa a tener flujo detrás. El trinquete de la banda es la hipótesis de "
              "que conviene DAR AIRE a la posición.",
    prior="Esperamos el mismo signo que el cruce de medias en tendencia, pero muchos menos cambios "
          "de posición por la histéresis de la banda, y que TODA la ventaja frente al cruce venga "
          "de esa reducción y no de mejor acierto. Falsable: si el número de cambios de posición "
          "no cae claramente respecto a trend.ema_cross_21_55, la histéresis no está haciendo "
          "nada. Esperamos devolver mucho beneficio en los giros bruscos, justo por lo lejos que "
          "está el nivel; predecimos que pierde frente a trend.psar_002_020 en mercados que giran "
          "en V.",
    fn=_supertrend_atr14_x3,
    params={"atr": 14, "multiplicador": 3.0},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=250,
))


# --------------------------------------------------------------------------------------------
# 8. SAR parabólico: la hipótesis contraria — apretar el stop
# --------------------------------------------------------------------------------------------


def _psar_002_020(s: Series) -> np.ndarray:
    h, l, c = _f64(s.high), _f64(s.low), _f64(s.close)
    out = np.zeros(c.size, dtype=np.int8)
    sar = talib.SAR(h, l, acceleration=0.02, maximum=0.2)
    ok = _ok(sar, c)
    out[ok & (c > sar)] = 1
    out[ok & (c < sar)] = -1
    return out


register(Hypothesis(
    name="trend.psar_002_020",
    family="trend",
    rationale="El SAR aprieta el stop conforme la tendencia madura, acelerando con cada nuevo "
              "extremo. Reproduce un comportamiento observable: el gestor sube el stop tras cada "
              "máximo nuevo para proteger beneficio no realizado, y esa escalera de stops es "
              "oferta latente que, al tocarse, se ejecuta en cascada. Está registrada "
              "explícitamente como la hipótesis OPUESTA a trend.supertrend_atr14_x3 sobre la misma "
              "pregunta: si conviene apretar o dar aire.",
    prior="Predecimos que SAR y Supertrend discrepan de forma sistemática en la duración de la "
          "posición y que UNO de los dos es claramente peor; apostamos por que el SAR pierde en "
          "neto por exceso de operaciones. El resultado que refutaría la pregunta entera es que "
          "ambos rindan igual: eso significaría que la distancia del stop no es una variable "
          "relevante y que las dos racionalizaciones son ruido. El SAR está siempre en mercado, "
          "así que esperamos daño severo en lateral.",
    fn=_psar_002_020,
    params={"aceleracion": 0.02, "maximo": 0.2},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=250,
))


# --------------------------------------------------------------------------------------------
# 9. pendiente de regresión sobre ATR: deriva medida en unidades de ruido
# --------------------------------------------------------------------------------------------


def _linreg_slope_55_atr14(s: Series) -> np.ndarray:
    h, l, c = _f64(s.high), _f64(s.low), _f64(s.close)
    out = np.zeros(c.size, dtype=np.int8)
    pendiente = talib.LINEARREG_SLOPE(c, 55)
    atr = talib.ATR(h, l, c, 14)
    ok = _ok(pendiente, atr) & (atr > 0.0)
    # desplazamiento acumulado de la ventana, en ATR: exigir > 1 es exigir que la deriva
    # domine al ruido en su propia unidad. No es un umbral ajustado: es el cambio de unidad.
    razon = np.full(c.size, np.nan, dtype=np.float64)
    razon[ok] = (pendiente[ok] * 55.0) / atr[ok]
    out[ok & (razon > 1.0)] = 1
    out[ok & (razon < -1.0)] = -1
    return out


register(Hypothesis(
    name="trend.linreg_slope_55_atr14",
    family="trend",
    rationale="La deriva solo es explotable en comparación con el ruido que hay que atravesar para "
              "cobrarla; un mercado que sube despacio en medio de un vendaval no es operable. La "
              "pendiente de regresión de 55 barras estima la deriva y el ATR14 el ruido por barra, "
              "y exigir que el desplazamiento acumulado de la ventana supere 1 ATR es un cambio de "
              "unidad, no un umbral ajustado a datos. Es la formulación más literal de 'tendencia' "
              "de toda la familia: sin cruces, sin puntos focales y sin memoria de extremos.",
    prior="Esperamos que sea el detector con MENOS operaciones y, sobre todo, el más estable entre "
          "timeframes, porque el umbral es adimensional. Ese es el contraste falsable: si su "
          "ventaja no sobrevive al cambio de timeframe mejor que la de trend.ema_cross_21_55, la "
          "normalización por volatilidad no aporta nada. Esperamos que falle justo en los saltos "
          "de volatilidad, porque el ATR reacciona tarde y el umbral se relaja precisamente cuando "
          "debería endurecerse.",
    fn=_linreg_slope_55_atr14,
    params={"ventana": 55, "atr": 14, "umbral_en_atr": 1.0},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=250,
))


# --------------------------------------------------------------------------------------------
# 10. Ichimoku: la única construcción popular con GROSOR de equilibrio
# --------------------------------------------------------------------------------------------


def _ichimoku_kumo(s: Series) -> np.ndarray:
    h, l, c = _f64(s.high), _f64(s.low), _f64(s.close)
    out = np.zeros(c.size, dtype=np.int8)
    tenkan = (talib.MAX(h, 9) + talib.MIN(l, 9)) / 2.0
    kijun = (talib.MAX(h, 26) + talib.MIN(l, 26)) / 2.0
    # +26 de desplazamiento: el nivel de HOY sale de datos de hace 26 barras. Pasado, no futuro.
    span_a = _lag((tenkan + kijun) / 2.0, 26)
    span_b = _lag((talib.MAX(h, 52) + talib.MIN(l, 52)) / 2.0, 26)
    ok = _ok(tenkan, kijun, span_a, span_b, c)
    techo = np.maximum(span_a, span_b)
    suelo = np.minimum(span_a, span_b)
    out[ok & (c > techo) & (tenkan > kijun)] = 1
    out[ok & (c < suelo) & (tenkan < kijun)] = -1
    return out


register(Hypothesis(
    name="trend.ichimoku_kumo",
    family="trend",
    rationale="La nube es la única construcción popular que declara la zona de equilibrio con "
              "GROSOR: entre Senkou A y B no hay señal, y el grosor crece justo cuando los "
              "extremos recientes discrepan entre sí. Codifica la memoria de los rangos de 9, 26 y "
              "52 barras, los horizontes de revisión de una mesa japonesa, y su valor —si lo "
              "tiene— es reflexivo por la enorme adopción del sistema en Asia. El desplazamiento "
              "+26 es lo que la hace interesante y también lo que la hace fácil de implementar "
              "mal: es información de hace 26 barras dibujada adelante, nunca información futura.",
    prior="Esperamos edge parecido al del cruce de medias pero con bastante MÁS tiempo fuera de "
          "mercado, porque la nube declara explícitamente el equilibrio. Falsable: si el tiempo "
          "fuera de mercado no aumenta de forma clara respecto a trend.ema_cross_21_55, la nube no "
          "filtra nada y es una media lenta con dos capas de pintura. Esperamos que falle en las "
          "tendencias que arrancan desde compresión, donde la nube es fina y no detiene nada.",
    fn=_ichimoku_kumo,
    params={"tenkan": 9, "kijun": 26, "senkou_b": 52, "desplazamiento": 26},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=250,
))


# --------------------------------------------------------------------------------------------
# 11. Aroon: la tendencia como proceso de renovación
# --------------------------------------------------------------------------------------------


def _aroon_25(s: Series) -> np.ndarray:
    h, l = _f64(s.high), _f64(s.low)
    out = np.zeros(h.size, dtype=np.int8)
    abajo, arriba = talib.AROON(h, l, 25)
    ok = _ok(arriba, abajo)
    out[ok & (arriba > 70.0) & (abajo < 30.0)] = 1
    out[ok & (abajo > 70.0) & (arriba < 30.0)] = -1
    return out


register(Hypothesis(
    name="trend.aroon_25",
    family="trend",
    rationale="Aroon no mide precio sino TIEMPO desde el último extremo, y esa es una variable "
              "distinta de todo lo demás en esta familia. La tesis es que la tendencia es un "
              "proceso de renovación: mientras sigan apareciendo máximos nuevos con frecuencia, "
              "hay compradores dispuestos a pagar precios que nunca se han pagado, algo que solo "
              "ocurre si la información todavía se está incorporando. Cuando la frecuencia de "
              "extremos nuevos cae, el flujo se ha agotado aunque el precio aún no haya girado.",
    prior="Esperamos que salga ANTES que las medias en los techos —que devuelva menos beneficio en "
          "el giro— y que pague ese seguro con entradas claramente peores. Falsable: si su patrón "
          "de resultados es indistinguible del de trend.ema_cross_21_55 en todos los regímenes, la "
          "información de tiempo es redundante con la de precio y la hipótesis no aporta nada "
          "nuevo a la familia. Esperamos edge nulo o negativo en lateral con extremos alternos, "
          "donde ambos brazos se mantienen altos y la señal se apaga.",
    fn=_aroon_25,
    params={"periodo": 25, "umbral_alto": 70, "umbral_bajo": 30},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=250,
))


# --------------------------------------------------------------------------------------------
# 12. momento de series temporales a 12 meses: horizonte de asignación
# --------------------------------------------------------------------------------------------


def _tsmom_365d(s: Series) -> np.ndarray:
    c = _f64(s.close)
    out = np.zeros(c.size, dtype=np.int8)
    hace_un_ano = _lag(c, 365)
    ok = _ok(c, hace_un_ano) & (hace_un_ano > 0.0)
    out[ok & (c > hace_un_ano)] = 1
    out[ok & (c < hace_un_ano)] = -1
    return out


register(Hypothesis(
    name="trend.tsmom_365d",
    family="trend",
    rationale="Es la hipótesis de tendencia con más respaldo académico transversal (Moskowitz, Ooi "
              "y Pedersen): el signo del rendimiento de los últimos doce meses predice el del "
              "periodo siguiente en casi todas las clases de activo y en más de un siglo de datos. "
              "El mecanismo propuesto es infrarreacción inicial por anclaje y difusión lenta, "
              "seguida de sobrerreacción por flujo de seguidores. En BTC se suma el ciclo de "
              "asignación: un comité aprueba mandatos con trimestres de retraso respecto a la "
              "decisión que los motivó, y ese retraso es exactamente la persistencia. Se registra "
              "solo en 1d porque 'doce meses' es una escala del calendario de asignación, no un "
              "número de barras: a 365 velas de 1h el mecanismo declarado no existe.",
    prior="Esperamos el edge más pequeño POR BARRA de toda la familia y a la vez el más estable, "
          "positivo también en los años fuera de muestra. Contraste falsable frente a un hermano: "
          "si el signo a 365 días no aporta nada sobre lo que ya da trend.linreg_slope_55_atr14, "
          "el argumento del horizonte de asignación es falso y solo estamos midiendo deriva a "
          "cualquier escala. Esperamos un drawdown muy grande en cada giro de ciclo, porque tarda "
          "meses en cambiar de signo.",
    fn=_tsmom_365d,
    params={"retraso_dias": 365},
    timeframes=("1d",),
    min_warmup=420,
))
