"""Familia `candles`: microestructura de vela. REGISTRO PREVIO, escrito sin mirar un solo número.

Postura de partida: escéptica. La mayoría de los patrones de vela del canon no tienen ventaja
demostrada fuera de la muestra, y buena parte de su semántica ("indecisión", "rechazo") importa
supuestos de un mercado con apertura y cierre de sesión que en BTC, que cotiza 24/7, no existen.
Varias hipótesis de este fichero predicen CERO a propósito: un prior de "cero" es falsable por
cualquier efecto significativo en cualquiera de los dos sentidos, y una familia en la que todo
predice éxito no es una familia de hipótesis, es un folleto.

Convenciones, fijadas ANTES de ejecutar nada:

- PULSO frente a ESTADO. Un patrón es un evento: emite ±1 en la vela en la que se completa y
  vuelve a 0. La gestión de salida la impone el motor, no la hipótesis. Solo dos hipótesis
  extienden la señal varias velas, y lo hacen porque su mecanismo declarado dura varias velas
  (una orden institucional troceada, la deriva posterior a un shock); está dicho en su rationale.
  Dos hipótesis son de estado y no de evento (`cierre_en_rango`, `flujo_cuerpos_20`): describen
  una condición sostenida de la vela o de la ventana, no un suceso puntual.

- PARÁMETROS. Un valor por parámetro, elegido por convención de la literatura (RSI 14 con umbrales
  30/70, ADX 14 con umbral 25, ATR 14, EMA 200, ventana 20) o, donde la literatura no ofrece uno,
  un número redondo fijado de antemano (0.8/0.2 de posición del cierre, 2x cuerpo y 0.6 de rango
  para la mecha, 2x ATR para el shock, 0.30 de flujo de cuerpos). Ninguno se ha elegido mirando
  datos, y ninguno se reajustará después. Cambiar uno sería otra hipótesis y otro ensayo.

- CAUSALIDAD. Todo lo que se lee en la posición i es de i o anterior. Los retardos van con
  `_retardo` (rellena NaN al arranque, nunca envuelve), la propagación de pulsos usa
  `np.maximum.accumulate`, que solo mira el prefijo, y no se usa nada que se defina contra el
  array entero (find_peaks con prominence, mínimos globales, normalizaciones por el total).
  Las funciones CDL* de TA-Lib son causales: sus umbrales internos son medias móviles hacia atrás.

- NaN. TA-Lib devuelve NaN durante el calentamiento. Se comprueba explícitamente con np.isnan
  (`_sin_nan`) en vez de confiar en que la comparación con NaN dé False, porque esa protección
  implícita se pierde en cuanto una condición aparece negada.

- COSTES. Los patrones de vela son señales de una vela y disparan mucho. La ventaja bruta no
  significa nada aquí: lo que se registra como éxito es el delta contra el brazo nulo NETO de
  comisiones, y para varias de estas hipótesis el prior es precisamente que el coste se la come.
"""

from __future__ import annotations

import numpy as np
import talib

from wavelab.hypotheses.base import Hypothesis, Series, register

# Ninguna hipótesis de esta familia emite señal con menos historia que esto: cubre el calentamiento
# más largo (EMA 200) con margen y evita pedirle a TA-Lib series más cortas que su lookback.
_MIN_VELAS = 320


def _f(a: np.ndarray) -> np.ndarray:
    """TA-Lib exige float64 contiguo. No copia si ya lo es."""
    return np.ascontiguousarray(a, dtype=np.float64)


def _sin_nan(*arrays: np.ndarray) -> np.ndarray:
    """Máscara True donde NINGUNO de los arrays es NaN.

    Se hace explícito y no se delega en que `NaN > x` sea False: esa protección desaparece en
    cuanto la condición aparece negada (`~(x > y)` sí es True con NaN) y reaparecería como señal
    inventada justo en el calentamiento, que es donde menos se mira.
    """
    m = np.ones(np.shape(arrays[0]), dtype=bool)
    for a in arrays:
        m &= ~np.isnan(np.asarray(a, dtype=np.float64))
    return m


def _retardo(a: np.ndarray, k: int = 1) -> np.ndarray:
    """El valor de la posición i-k, colocado en i. NaN en el arranque. Solo mira hacia atrás."""
    x = np.asarray(a, dtype=np.float64)
    if k <= 0:
        return x.copy()
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if k < x.size:
        out[k:] = x[: x.size - k]
    return out


def _mantener(pulso: np.ndarray, n: int) -> np.ndarray:
    """Extiende cada pulso ±1 durante n velas contando la de la señal.

    Causal: `maximum.accumulate` solo depende del prefijo, así que la posición i solo sabe de la
    última señal ocurrida en i o antes. Un pulso nuevo sustituye al anterior.
    """
    idx = np.arange(pulso.size)
    ultimo = np.maximum.accumulate(np.where(pulso != 0, idx, -1))
    vivo = (ultimo >= 0) & ((idx - ultimo) < n)
    return np.where(vivo, pulso[np.maximum(ultimo, 0)], 0).astype(np.int8)


def _ohlc(s: Series) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _f(s.open), _f(s.high), _f(s.low), _f(s.close)


# ---------------------------------------------------------------------------
# 1. Envolvente sin filtrar: la versión del folclore, registrada para poder refutarla.
# ---------------------------------------------------------------------------
def _envolvente_cruda(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_VELAS:
        return out
    o, h, l, c = _ohlc(s)
    p = np.asarray(talib.CDLENGULFING(o, h, l, c), dtype=np.float64)
    ok = _sin_nan(p)
    out[ok & (p > 0)] = 1
    out[ok & (p < 0)] = -1
    return out


register(Hypothesis(
    name="candles.envolvente_cruda",
    family="candles",
    rationale="Una vela envolvente significa que todo el rango de precios que la vela anterior "
              "consideró aceptable se ha recorrido entero y el cierre ha quedado al otro lado de "
              "su apertura: quien tomó posición durante la vela envuelta está en pérdidas como "
              "bloque, y sus stops quedan justo al otro lado del extremo de la envolvente, lo que "
              "aporta flujo forzado en la misma dirección. Ese es el mecanismo que se le atribuye "
              "y se registra sin ningún filtro, tal como lo enuncia el canon, para que la versión "
              "condicionada (candles.envolvente_tendencia) tenga contra qué contrastarse.",
    prior="Predecimos que NO hay ventaja: esperamos una expectativa indistinguible de cero en "
          "bruto y negativa una vez descontadas comisiones, porque sin contexto una envolvente es "
          "mecánicamente 'una vela grande que cerró fuerte', que en BTC aparece más a menudo al "
          "final de una cascada de liquidaciones que al principio de un movimiento. Queda falsada "
          "si el delta neto contra el brazo nulo es claramente positivo en cualquiera de los tres "
          "timeframes; queda falsada también, en el otro sentido, si es claramente negativo, lo "
          "que indicaría que el patrón funciona invertido.",
    fn=_envolvente_cruda,
    params={"patron": "CDLENGULFING", "mantener": 1},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 2. Envolvente alineada con la tendencia de fondo.
# ---------------------------------------------------------------------------
def _envolvente_tendencia(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_VELAS:
        return out
    o, h, l, c = _ohlc(s)
    p = np.asarray(talib.CDLENGULFING(o, h, l, c), dtype=np.float64)
    ema = talib.EMA(c, 200)
    ok = _sin_nan(p, ema)
    out[ok & (p > 0) & (c > ema)] = 1
    out[ok & (p < 0) & (c < ema)] = -1
    return out


register(Hypothesis(
    name="candles.envolvente_tendencia",
    family="candles",
    rationale="La misma envolvente, pero solo cuando apunta en el sentido de la tendencia de "
              "fondo. El mecanismo de stops solo puede funcionar si hay un lado atrapado y otro "
              "con capacidad de empujar: en un mercado que sube, una envolvente alcista marca a "
              "los vendedores del retroceso como el bloque atrapado, mientras que una envolvente "
              "alcista dentro de una caída marca a compradores que seguirán encontrando oferta "
              "encima. No es una variante de parámetro de la hipótesis anterior: la anterior es la "
              "afirmación incondicional del canon y esta es la afirmación de que el efecto es "
              "CONDICIONAL al régimen; el contraste entre ambas es la cantidad informativa, y la "
              "EMA 200 se fija en su valor convencional y no se busca.",
    prior="Esperamos ventaja positiva y mayor que la de candles.envolvente_cruda, y esperamos que "
          "la diferencia entre ambas sea mayor que la ventaja absoluta de cualquiera de las dos. "
          "Falla si el filtro no mejora nada (el patrón no es sensible al contexto y su mecanismo "
          "declarado es falso) y falla igualmente si mejora pero con expectativa neta negativa, "
          "que sería solo el sesgo alcista estructural de BTC filtrado, no el patrón.",
    fn=_envolvente_tendencia,
    params={"patron": "CDLENGULFING", "ema": 200, "mantener": 1},
    timeframes=("1h", "4h", "1d"),
    min_warmup=300,
))


# ---------------------------------------------------------------------------
# 3. Martillo tras caída (RSI 14 en sobreventa).
# ---------------------------------------------------------------------------
def _martillo_sobreventa(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_VELAS:
        return out
    o, h, l, c = _ohlc(s)
    p = np.asarray(talib.CDLHAMMER(o, h, l, c), dtype=np.float64)
    rsi = talib.RSI(c, 14)
    ok = _sin_nan(p, rsi)
    out[ok & (p > 0) & (rsi < 30.0)] = 1
    return out


register(Hypothesis(
    name="candles.martillo_sobreventa",
    family="candles",
    rationale="Un martillo es una vela cuyo precio bajó mucho y volvió: la mecha inferior es la "
              "huella visible de una demanda pasiva que absorbió toda la venta agresiva y devolvió "
              "el precio. Eso solo dice algo después de una caída, cuando la venta agresiva es "
              "liquidación forzada y no colocación ordenada: el vendedor forzado se agota en "
              "horas, el comprador que absorbió sigue ahí, y el desequilibrio se resuelve al alza. "
              "Sin la caída previa, la misma forma es una vela cualquiera de un rango, por eso la "
              "condición de sobreventa forma parte de la hipótesis y no es un añadido.",
    prior="Esperamos ventaja positiva pequeña y solo en el lado largo. Esperamos que falle "
          "justamente donde la venta no es forzada sino informada: en tendencias bajistas "
          "sostenidas cada absorción se vuelve a probar y el martillo se convierte en cuchillo "
          "cayendo, así que anticipamos una cola izquierda gorda y una tasa de acierto alta con "
          "expectativa mediocre. Falsada si el signo es negativo, o si el delta neto contra el "
          "brazo nulo no se distingue de cero.",
    fn=_martillo_sobreventa,
    params={"patron": "CDLHAMMER", "rsi": 14, "umbral_rsi": 30, "mantener": 1},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 4. Estrella fugaz tras subida (RSI 14 en sobrecompra). El espejo del martillo.
# ---------------------------------------------------------------------------
def _estrella_fugaz_sobrecompra(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_VELAS:
        return out
    o, h, l, c = _ohlc(s)
    p = np.asarray(talib.CDLSHOOTINGSTAR(o, h, l, c), dtype=np.float64)
    rsi = talib.RSI(c, 14)
    ok = _sin_nan(p, rsi)
    out[ok & (p < 0) & (rsi > 70.0)] = -1
    return out


register(Hypothesis(
    name="candles.estrella_fugaz_sobrecompra",
    family="candles",
    rationale="Espejo exacto del martillo: mecha superior larga tras una subida, es decir oferta "
              "pasiva que absorbió toda la compra agresiva y devolvió el precio. Se registra por "
              "separado y no como el lado corto de la misma hipótesis porque el mecanismo NO es "
              "simétrico: la compra agresiva en máximos es discrecional y puede retirarse sin "
              "coste, mientras que la venta agresiva en mínimos suele ser liquidación forzada con "
              "plazo. Si la simetría del canon fuese cierta, ambas hipótesis deberían dar una "
              "ventaja parecida; la comparación entre las dos es el contenido de este par.",
    prior="Esperamos un efecto claramente MENOR que el del martillo y admitimos como resultado "
          "probable una expectativa negativa: ponerse corto contra fuerza en BTC pelea a la vez "
          "contra la deriva positiva del activo y contra el sesgo largo del mercado, y el coste de "
          "financiación del corto no se recupera en una vela. Falsada si su ventaja iguala o "
          "supera a la del martillo, lo que indicaría que la absorción es simétrica y que nuestra "
          "explicación por flujo forzado sobra.",
    fn=_estrella_fugaz_sobrecompra,
    params={"patron": "CDLSHOOTINGSTAR", "rsi": 14, "umbral_rsi": 70, "mantener": 1},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 5. Posición del cierre dentro del rango de la vela (estado, no evento).
# ---------------------------------------------------------------------------
def _cierre_en_rango(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_VELAS:
        return out
    _o, h, l, c = _ohlc(s)
    rango = h - l
    ok = _sin_nan(h, l, c) & (rango > 0.0)
    pos = np.zeros(len(s), dtype=np.float64)
    np.divide(c - l, rango, out=pos, where=ok)
    out[ok & (pos >= 0.80)] = 1
    out[ok & (pos <= 0.20)] = -1
    return out


register(Hypothesis(
    name="candles.cierre_en_rango",
    family="candles",
    rationale="El cierre es el único precio al que ambos lados aceptaron quedarse con la posición "
              "de una vela a la siguiente; el resto del rango son precios que alguien rechazó. Un "
              "cierre clavado en el extremo del rango significa que el lado ganador todavía tenía "
              "demanda sin ejecutar cuando terminó la vela: esa orden no desaparece en el cambio "
              "de vela, se sigue ejecutando. Es la lectura más despojada de la forma de la vela "
              "—no necesita nombre de patrón ni contexto— y por eso sirve de referencia mínima "
              "frente a la que juzgar a los patrones con nombre de esta misma familia.",
    prior="Esperamos continuación: ventaja positiva pequeña en 1h y 4h. Esperamos que se anule o "
          "cambie de signo en 1d, donde el horizonte da tiempo a que domine la reversión, y que "
          "en 15m sea positiva en bruto pero negativa neta, porque la señal dispara en casi una de "
          "cada tres velas y el coste por rotación se la come. Falsada si el signo es negativo en "
          "1h y 4h, que sería evidencia de que el cierre en el extremo marca agotamiento y no "
          "demanda residual.",
    fn=_cierre_en_rango,
    params={"umbral_alto": 0.80, "umbral_bajo": 0.20},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 6. Marubozu de cierre: sesión ganada de punta a punta.
# ---------------------------------------------------------------------------
def _marubozu_continuacion(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_VELAS:
        return out
    o, h, l, c = _ohlc(s)
    p = np.asarray(talib.CDLCLOSINGMARUBOZU(o, h, l, c), dtype=np.float64)
    ok = _sin_nan(p)
    out[ok & (p > 0)] = 1
    out[ok & (p < 0)] = -1
    return out


register(Hypothesis(
    name="candles.marubozu_continuacion",
    family="candles",
    rationale="Un marubozu de cierre es un cuerpo grande sin mecha del lado del cierre: durante "
              "toda la vela no hubo un solo momento en que el lado perdedor consiguiera devolver "
              "el precio, ni siquiera al final. Es la firma de un desequilibrio de flujo que no "
              "encontró oposición pasiva, y un desequilibrio así no se agota exactamente en el "
              "límite arbitrario de la vela. Se usa la variante de cierre y no el marubozu "
              "estricto porque el extremo informativo es el del cierre; exigir además ausencia de "
              "mecha en la apertura solo añade rareza sin añadir mecanismo.",
    prior="Esperamos continuación en el sentido del cuerpo, con ventaja positiva pequeña. "
          "Esperamos que falle en los extremos de rango y al final de recorridos largos, donde la "
          "misma vela es capitulación y no impulso; como no distinguimos ex ante los dos casos, la "
          "media de ambos debería salir débil. Falsada si el signo neto es negativo, lo que "
          "apoyaría la lectura de agotamiento frente a la de continuación.",
    fn=_marubozu_continuacion,
    params={"patron": "CDLCLOSINGMARUBOZU", "mantener": 1},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 7. Doji tras impulso direccional (ADX 14 > 25). Hipótesis que predice CERO.
# ---------------------------------------------------------------------------
def _doji_tras_impulso(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_VELAS:
        return out
    o, h, l, c = _ohlc(s)
    p = np.asarray(talib.CDLDOJI(o, h, l, c), dtype=np.float64)
    adx = talib.ADX(h, l, c, 14)
    di_mas = talib.PLUS_DI(h, l, c, 14)
    di_menos = talib.MINUS_DI(h, l, c, 14)
    ok = _sin_nan(p, adx, di_mas, di_menos)
    impulso = ok & (p != 0.0) & (adx > 25.0)
    out[impulso & (di_mas > di_menos)] = -1
    out[impulso & (di_menos > di_mas)] = 1
    return out


register(Hypothesis(
    name="candles.doji_tras_impulso",
    family="candles",
    rationale="Un doji es la primera vela de un tramo en la que el lado dominante encuentra "
              "contrapartida suficiente para acabar donde empezó; dentro de un movimiento "
              "direccional maduro marcaría el punto en el que se agota el comprador (o vendedor) "
              "marginal, y por eso el canon lo lee como aviso de vuelta. La dirección del impulso "
              "se toma del sistema direccional de Wilder con su parámetro convencional, para no "
              "introducir una ventana propia. La registramos porque es la hipótesis de velas más "
              "citada que somos capaces de enunciar con un mecanismo explícito, no porque "
              "creamos en ella.",
    prior="Predecimos CERO. En un mercado 24/7 no hay apertura ni cierre de sesión, así que un "
          "doji no es un veredicto colectivo sino simplemente una vela de cuerpo pequeño, y "
          "aparece a puñados en las horas de poco volumen; la semántica de 'indecisión' importa "
          "una estructura de sesión que aquí no existe. Un prior de cero es falsable por cualquier "
          "efecto significativo en cualquiera de los dos sentidos: si el desvanecimiento del "
          "impulso da ventaja positiva, nuestra objeción sobre el 24/7 es errónea; si la da "
          "negativa, el doji es continuación y el canon está invertido.",
    fn=_doji_tras_impulso,
    params={"patron": "CDLDOJI", "adx": 14, "umbral_adx": 25, "mantener": 1},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 8. Mecha desproporcionada, sin folclore: pura geometría simétrica.
# ---------------------------------------------------------------------------
def _mecha_desproporcionada(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_VELAS:
        return out
    o, h, l, c = _ohlc(s)
    ok = _sin_nan(o, h, l, c)
    rango = h - l
    ok &= rango > 0.0
    cuerpo = np.abs(c - o)
    mecha_inf = np.minimum(o, c) - l
    mecha_sup = h - np.maximum(o, c)
    largo = ok & (mecha_inf >= 2.0 * cuerpo) & (mecha_inf >= 0.60 * rango)
    corto = ok & (mecha_sup >= 2.0 * cuerpo) & (mecha_sup >= 0.60 * rango)
    out[largo] = 1
    out[corto] = -1
    return out


register(Hypothesis(
    name="candles.mecha_desproporcionada",
    family="candles",
    rationale="Una mecha larga es la huella de un nivel al que el precio llegó y del que volvió: "
              "prueba de que allí había liquidez pasiva de verdad, que absorbió el flujo agresivo "
              "y quedó parcialmente sin ejecutar. Quien tiene una orden grande a ese precio la "
              "vuelve a poner, así que el nivel funciona como suelo o techo blando durante las "
              "velas siguientes. Se define solo con geometría —mecha mayor que el doble del cuerpo "
              "y más del 60% del rango, simétrica arriba y abajo— y sin exigir tendencia previa ni "
              "posición del cuerpo, precisamente para separar el efecto de absorción del "
              "vocabulario de martillo y estrella fugaz, que añaden condiciones de contexto.",
    prior="Esperamos ventaja positiva pequeña en el sentido contrario a la mecha, mayor en 4h que "
          "en 15m. Esperamos que falle en tendencias fuertes, donde una mecha inferior larga "
          "dentro de una caída no es absorción sino un retroceso parcial antes de continuar, y "
          "esperamos que en 15m la ventaja bruta exista pero desaparezca al descontar comisiones. "
          "Falsada si el signo es negativo, o si no supera al brazo nulo en ningún timeframe.",
    fn=_mecha_desproporcionada,
    params={"mecha_vs_cuerpo": 2.0, "mecha_vs_rango": 0.60},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 9. Expansión de rango contra ATR: shock informativo o flujo forzado.
# ---------------------------------------------------------------------------
def _expansion_rango(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_VELAS:
        return out
    o, h, l, c = _ohlc(s)
    tr = talib.TRANGE(h, l, c)
    atr_prev = _retardo(talib.ATR(h, l, c, 14), 1)
    ok = _sin_nan(tr, atr_prev, o, c) & (atr_prev > 0.0)
    shock = ok & (tr >= 2.0 * atr_prev)
    pulso = np.zeros(len(s), dtype=np.int8)
    pulso[shock & (c > o)] = 1
    pulso[shock & (c < o)] = -1
    return _mantener(pulso, 3)


register(Hypothesis(
    name="candles.expansion_rango",
    family="candles",
    rationale="Una vela cuyo rango verdadero dobla al ATR de las 14 anteriores no es ruido: es un "
              "reprecio en el que el libro se atravesó entero. Nuestro mecanismo es informativo: "
              "ha entrado información nueva, la volatilidad se agrupa, y quien tiene que "
              "reposicionar una cartera grande no lo hace en una sola vela, así que la deriva "
              "continúa en el sentido en que cerró el cuerpo. El ATR se compara retardado una vela "
              "para que la referencia sea la norma previa al shock y no una norma ya contaminada "
              "por él. La señal se mantiene tres velas porque el mecanismo declarado —"
              "reposicionamiento troceado— dura más de una, no porque tres funcione mejor.",
    prior="Esperamos continuación: ventaja positiva en el sentido del cuerpo de la vela del "
          "shock. Existe un mecanismo rival explícito y creíble —que el rango grande sea una "
          "cascada de liquidaciones, es decir flujo forzado sin información, que los creadores de "
          "mercado revierten al recotizar— y ese mecanismo predice el signo CONTRARIO. Por eso el "
          "signo del resultado es informativo pase lo que pase: positivo apoya la lectura "
          "informativa, negativo apoya la de flujo forzado, y cero dice que ambas se cancelan "
          "porque no distinguimos ex ante los dos casos, que es lo que consideramos más probable "
          "en 15m. "
          "NOTA DE AUDITORÍA (2026-09-08): `volatility.wide_range_thrust` aplica esta misma regla "
          "(rango verdadero mayor que el doble del ATR previo, dirección dada por el cuerpo) con "
          "una condición añadida —que el ATR de la vela previa estuviera bajo su mediana de 100— y "
          "sin mantener la señal tres velas. Medidos sobre datos, sus eventos son un SUBCONJUNTO "
          "casi exacto de los de esta hipótesis (263 de 267 coinciden vela a vela). No se elimina "
          "ninguna porque el par sin filtro / con filtro es informativo y ese contraste no estaba "
          "declarado en ninguno de los dos ficheros, pero queda dicho ahora: no son dos ensayos "
          "independientes y no deben leerse como confirmación cruzada entre familias si ambas "
          "salen positivas. El contraste que importa es si la puerta de compresión mejora a esta.",
    fn=_expansion_rango,
    params={"atr": 14, "multiplo": 2.0, "mantener": 3},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 10. Vela interior y ruptura de la vela madre.
# ---------------------------------------------------------------------------
def _vela_interior_ruptura(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_VELAS:
        return out
    _o, h, l, c = _ohlc(s)
    h1, l1 = _retardo(h, 1), _retardo(l, 1)
    h2, l2 = _retardo(h, 2), _retardo(l, 2)
    ok = _sin_nan(c, h1, l1, h2, l2)
    dentro = ok & (h1 <= h2) & (l1 >= l2)
    out[dentro & (c > h2)] = 1
    out[dentro & (c < l2)] = -1
    return out


register(Hypothesis(
    name="candles.vela_interior_ruptura",
    family="candles",
    rationale="Una vela contenida dentro del rango de la anterior es una contracción del rango "
              "negociado: los dos lados aceptan el mismo intervalo de precios y la incertidumbre "
              "baja. Ahí se acumulan órdenes de protección a ambos lados de los extremos de la "
              "vela madre. Que el cierre de la vela siguiente quede fuera de ese rango significa "
              "que la liquidez pasiva de un lado se ha consumido, y el camino de menor resistencia "
              "pasa a ser el racimo de stops del otro. La ruptura se evalúa con el CIERRE de la "
              "vela en curso contra extremos de velas ya cerradas, nunca con el máximo intradía, "
              "para que la decisión sea de la vela i y no de datos que solo se conocen después.",
    prior="Esperamos continuación con ventaja positiva pequeña en 4h y 1d. Con poca confianza: hay "
          "un mecanismo contrario igual de plausible en BTC, que el racimo de stops sea el "
          "OBJETIVO del flujo agresivo y no su combustible, en cuyo caso la ruptura es falsa por "
          "construcción y el signo sale negativo. Esperamos además que empeore al bajar de "
          "timeframe, porque la contracción de rango en 1h es en su mayoría horario de poco "
          "volumen y no acuerdo entre participantes. Falsada por un signo negativo estable. "
          "NOTA DE AUDITORÍA (2026-09-08): esta hipótesis estaba registrada DOS VECES. "
          "`structure.inside_bar_break` era la misma regla —misma contención, mismos extremos de "
          "la vela madre como nivel, misma confirmación por cierre— y sus 244 eventos eran un "
          "subconjunto estricto de los 293 de esta; se ha eliminado allí y esta es la registración "
          "superviviente. Queda además declarado que `volatility.inside_bar_breakout` opera sobre "
          "el MISMO patrón de dos velas pero rompiendo los extremos de la vela INTERIOR en vez de "
          "los de la madre: como el máximo de la interior es menor, todo evento de esta hipótesis "
          "es también evento de aquella. Son niveles distintos y afirmaciones mecánicas distintas, "
          "así que aquella se conserva, pero las dos son ensayos DEPENDIENTES y la corrección por "
          "contraste múltiple debe tratarlas como tales.",
    fn=_vela_interior_ruptura,
    params={"velas_madre": 1, "confirmacion": "cierre"},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 11. Hueco de rango entre velas consecutivas en un mercado 24/7.
# ---------------------------------------------------------------------------
def _hueco_rango(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    if len(s) < _MIN_VELAS:
        return out
    _o, h, l, c = _ohlc(s)
    h1, l1 = _retardo(h, 1), _retardo(l, 1)
    ok = _sin_nan(h, l, h1, l1)
    out[ok & (l > h1)] = -1
    out[ok & (h < l1)] = 1
    return out


register(Hypothesis(
    name="candles.hueco_rango",
    family="candles",
    rationale="BTC cotiza sin interrupción, así que un hueco no puede ser el resultado de "
              "información acumulada mientras el mercado estaba cerrado: si el rango entero de una "
              "vela queda por encima del máximo de la anterior, el libro se ha vaciado y el precio "
              "ha viajado sin negociación intermedia. Eso es flujo forzado —liquidaciones en "
              "cadena o un tramo de liquidez muy fina— y no un reprecio consentido: en cuanto los "
              "creadores de mercado vuelven a cotizar, el hueco se rellena. Advertencia registrada "
              "de antemano: un agujero en los datos del proveedor produce exactamente esta misma "
              "forma, así que un resultado positivo obliga a comprobar la continuidad de las "
              "marcas de tiempo antes de creérselo.",
    prior="Esperamos reversión: ventaja positiva al operar CONTRA el hueco, mayor en 15m que en "
          "4h. Esperamos que falle cuando el hueco responde a noticia real (un reprecio legítimo "
          "que no se rellena), caso que no sabemos distinguir ex ante, y esperamos muy pocas "
          "señales en 4h, donde un hueco de rango completo implica un cambio de régimen y debería "
          "continuar en vez de revertir. Falsada si el signo es de continuación en 15m y 1h.",
    fn=_hueco_rango,
    params={"tipo": "hueco_de_rango_completo"},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 12. Tres soldados / tres cuervos: la huella de una orden troceada.
# ---------------------------------------------------------------------------
def _tres_velas_direccionales(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < _MIN_VELAS:
        return out
    o, h, l, c = _ohlc(s)
    sol = np.asarray(talib.CDL3WHITESOLDIERS(o, h, l, c), dtype=np.float64)
    cue = np.asarray(talib.CDL3BLACKCROWS(o, h, l, c), dtype=np.float64)
    ok = _sin_nan(sol, cue)
    pulso = np.zeros(n, dtype=np.int8)
    pulso[ok & (sol > 0)] = 1
    pulso[ok & (cue < 0)] = -1
    return _mantener(pulso, 3)


register(Hypothesis(
    name="candles.tres_velas_direccionales",
    family="candles",
    rationale="Tres velas seguidas con cuerpo amplio, cada una cerrando cerca de su máximo y por "
              "encima de la anterior, no es un impulso: es la huella de una orden grande que se "
              "está ejecutando por tramos a lo largo del tiempo. Un impulso especulativo se agota "
              "en una vela; un programa de ejecución sigue un calendario y por eso deja tres velas "
              "iguales. Si la lectura es correcta, la parte no ejecutada del programa sigue "
              "comprando después de la tercera vela, y por eso la señal se mantiene tres velas: es "
              "el horizonte del mecanismo declarado, no un valor elegido por rendimiento.",
    prior="Esperamos continuación débil. Débil porque la entrada es tardía por construcción: para "
          "cuando el patrón se completa, tres velas del movimiento ya han ocurrido y compramos "
          "contra el propio ejecutor. Registramos de antemano que ambos patrones son raros y que "
          "el número de señales será pequeño, así que el intervalo será ancho y la corrección por "
          "contraste múltiple debe aplicarse igual: un resultado espectacular con n pequeña cuenta "
          "como no concluyente, no como hallazgo. Falsada por signo negativo, que apoyaría la "
          "lectura de agotamiento.",
    fn=_tres_velas_direccionales,
    params={"patrones": "CDL3WHITESOLDIERS/CDL3BLACKCROWS", "mantener": 3},
    timeframes=("4h", "1d"),
    min_warmup=200,
))


# ---------------------------------------------------------------------------
# 13. Flujo de cuerpos sobre 20 velas: cuánto del rango se convierte en avance (estado).
# ---------------------------------------------------------------------------
def _flujo_cuerpos(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n < _MIN_VELAS:
        return out
    o, h, l, c = _ohlc(s)
    suma_cuerpo = talib.SUM(c - o, 20)
    suma_rango = talib.SUM(h - l, 20)
    ok = _sin_nan(suma_cuerpo, suma_rango) & (suma_rango > 0.0)
    flujo = np.zeros(n, dtype=np.float64)
    np.divide(suma_cuerpo, suma_rango, out=flujo, where=ok)
    out[ok & (flujo >= 0.30)] = 1
    out[ok & (flujo <= -0.30)] = -1
    return out


register(Hypothesis(
    name="candles.flujo_cuerpos_20",
    family="candles",
    rationale="Agrega la forma de la vela en lugar de buscar un patrón: mide qué proporción de "
              "todo el rango negociado en 20 velas se ha convertido en avance neto de apertura a "
              "cierre. Cuando esa proporción es alta, las sesiones las está ganando "
              "sistemáticamente el mismo lado y el rango es recorrido, no ida y vuelta: hay "
              "participantes direccionales dominando sobre los creadores de mercado. Cuando es "
              "baja, el precio se mueve pero se devuelve dentro de cada vela, que es la firma del "
              "inventario de los creadores de mercado. No es una media móvil: ignora los saltos "
              "entre velas y pondera por el rango negociado, así que dos series con idéntico "
              "recorrido de precio pueden dar valores opuestos.",
    prior="Esperamos ventaja positiva en régimen tendencial y nula o negativa en lateral, y "
          "esperamos que el conjunto de las dos salga apenas por encima de cero. La afirmación "
          "fuerte y falsable es de valor incremental: si la forma de la vela no añade nada sobre "
          "la simple dirección del precio, esta hipótesis debería ser indistinguible de un filtro "
          "de tendencia corriente sobre las mismas velas; que aporte algo por encima de eso es lo "
          "que la haría interesante, y no lo damos por hecho.",
    fn=_flujo_cuerpos,
    params={"ventana": 20, "umbral": 0.30},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))
