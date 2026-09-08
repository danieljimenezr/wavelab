"""Familia `mean_reversion`: el precio vuelve a un ancla después de alejarse demasiado.

REGISTRO PREVIO. Todo lo que hay en este fichero se escribió ANTES de ejecutar un solo backtest y
antes de mirar un solo número. Los parámetros son los convencionales de la literatura y están
FIJOS: no hay variantes del mismo parámetro "por probar". Las 12 hipótesis se registran juntas y
las 12 deben contarse en la corrección por contraste múltiple, incluidas las que fracasen.

El mecanismo de mercado que comparte la familia —lo que tendría que ser cierto para que esto
funcione— es de tres piezas:

  1. Provisión de liquidez. Cuando una orden grande e impaciente barre el libro, el precio se
     desplaza más de lo que justifica la información que trae esa orden. Quien está al otro lado
     cobra por absorberla. Ese cobro es la prima de reversión, y es el único ingreso "real" que
     esta familia puede capturar.
  2. Sobrerreacción. Los operadores extrapolan el último tramo y sobreponderan lo reciente; el
     precio se aleja del consenso más de lo que el flujo de información justifica y luego corrige.
  3. Cierre forzado. En un mercado con apalancamiento alto (BTC perpetuos), las liquidaciones en
     cascada son ventas que NO expresan una opinión sobre el valor. Son mecánicas y se agotan
     cuando se acaba el colateral. El hueco que dejan se rellena.

Y el modo de fallo que comparte, también declarado de antemano: en una tendencia sostenida, toda
esta familia vende fuerza y compra debilidad justo cuando eso es exactamente lo contrario de lo que
hay que hacer. Un estadístico de reversión "clavado" en el extremo durante un tramo direccional no
es una señal, es una pérdida lenta. Varias de las hipótesis de abajo existen precisamente para
medir ese fallo en lugar de esconderlo (`bb20_2_range_adx14` y `rsi14_cardwell_sma200` frente a sus
versiones sin filtro de régimen).

Nota sobre `min_warmup`. Es una decisión de higiene numérica, no un parámetro de la señal: se elige
para descartar el arranque de los indicadores (y el periodo inestable de las EMA largas, ~2x el
periodo), no para mejorar ningún resultado. No se toca después de ver datos.
"""

from __future__ import annotations

import numpy as np
import talib

from wavelab.hypotheses.base import Hypothesis, Series, register

# --------------------------------------------------------------------------------------------
# Utilidades. Ninguna mira hacia adelante.
# --------------------------------------------------------------------------------------------


def _f(x: np.ndarray) -> np.ndarray:
    """talib exige float64 contiguo; si no, lanza 'input array type is not double'."""
    return np.ascontiguousarray(x, dtype=np.float64)


def _prev(a: np.ndarray) -> np.ndarray:
    """Valor de la barra ANTERIOR, alineado a i (es decir, a[i-1] colocado en la posición i).

    El desplazamiento va del pasado hacia el presente, nunca al revés. `np.roll(a, 1)` está
    prohibido aquí: mete a[-1] —el último dato de toda la serie— en la posición 0, que es
    exactamente la fuga de información que este proyecto persigue.
    """
    out = np.empty_like(a)
    out[0] = np.nan
    out[1:] = a[:-1]
    return out


def _finite(*arrays: np.ndarray) -> np.ndarray:
    """Máscara de posiciones donde TODOS los indicadores están definidos.

    talib devuelve NaN durante el calentamiento. Una comparación con NaN da False, así que no
    inventa un largo; pero sí puede inventar un corto si la condición se escribe negada. Por eso
    la comprobación es explícita con np.isnan y se aplica como AND a cada condición, en vez de
    confiar en el comportamiento implícito o en nan_to_num (que sí inventaría señales al sustituir
    el NaN por un número que compara verdadero).
    """
    ok = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        ok &= ~np.isnan(a)
    return ok


def _sig(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int8)


# --------------------------------------------------------------------------------------------
# 1. Bandas de Bollinger: el desvanecimiento desnudo.
# --------------------------------------------------------------------------------------------


def _bb_fade(s: Series) -> np.ndarray:
    c = _f(s.close)
    out = _sig(c.size)
    upper, _mid, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
    ok = _finite(upper, lower)
    out[ok & (c < lower)] = 1
    out[ok & (c > upper)] = -1
    return out


register(Hypothesis(
    name="mean_reversion.bb20_2_fade",
    family="mean_reversion",
    rationale="Cerrar fuera de la banda de 2 sigma sobre 20 velas significa que el precio se ha "
              "movido más de lo que la dispersión reciente del propio activo considera normal. "
              "El comportamiento que produciría el efecto es el del creador de mercado y el "
              "operador de rango: ven un desequilibrio temporal de flujo —una orden impaciente "
              "que barre el libro— y cobran una prima por absorberla, devolviendo el precio hacia "
              "la SMA20 cuando esa orden se agota. Es la formulación más desnuda posible de la "
              "familia y sirve de línea base contra la que se miden las otras once.",
    prior="Esperamos ventaja positiva pero pequeña en régimen lateral y NEGATIVA en régimen "
          "tendencial, porque en tendencia el precio 'camina la banda' durante decenas de velas y "
          "esta regla mantiene el lado equivocado todo ese tiempo. Neta sobre toda la muestra "
          "esperamos algo cercano a cero, o negativo tras costes. Queda refutada si la ventaja es "
          "positiva y estable en ambos regímenes: eso significaría que no estamos midiendo "
          "provisión de liquidez sino otra cosa.",
    fn=_bb_fade,
    params={"periodo": 20, "desviaciones": 2.0},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 2. Bollinger con reentrada. MISMOS parámetros, REGLA DE DECISIÓN distinta (no es una variante
#    de parámetro: la 1 entra durante la excursión, esta espera a que la excursión termine).
# --------------------------------------------------------------------------------------------


def _bb_reentry(s: Series) -> np.ndarray:
    c = _f(s.close)
    n = c.size
    out = _sig(n)
    upper, mid, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)

    pos = 0
    for i in range(1, n):
        # Solo se leen los índices i e i-1. Nunca i+1.
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(mid[i]) \
                or np.isnan(upper[i - 1]) or np.isnan(lower[i - 1]):
            pos = 0
            out[i] = 0
            continue
        # Salida primero: objetivo en la media móvil, que es el ancla de la hipótesis.
        if pos == 1 and c[i] >= mid[i]:
            pos = 0
        elif pos == -1 and c[i] <= mid[i]:
            pos = 0
        # Entrada solo si estamos planos y la vela anterior cerró FUERA y esta cierra DENTRO.
        if pos == 0:
            if c[i - 1] < lower[i - 1] and c[i] >= lower[i]:
                pos = 1
            elif c[i - 1] > upper[i - 1] and c[i] <= upper[i]:
                pos = -1
        out[i] = pos
    return out


register(Hypothesis(
    name="mean_reversion.bb20_2_reentry",
    family="mean_reversion",
    rationale="Misma banda que la hipótesis 1 y los mismos 20/2, pero la regla de decisión es la "
              "contraria en el tiempo: no se compra la caída, se compra el final de la caída. "
              "Exigir que la vela anterior cerrara fuera y esta dentro es pedir una prueba de que "
              "el flujo impaciente ya se agotó, en lugar de suponerlo. El participante concreto es "
              "el vendedor forzado: mientras liquida, el precio sigue fuera de la banda; cuando "
              "termina, la primera vela que vuelve a cerrar dentro marca que el desequilibrio se "
              "ha consumido. Se registra junto a la 1 porque la comparación entre ambas es la que "
              "aísla el coste de anticipar frente al coste de esperar.",
    prior="Esperamos MENOS operaciones y MEJOR ventaja por operación que la hipótesis 1, y sobre "
          "todo una cola izquierda mucho más corta, porque no se entra contra una liquidación en "
          "curso. Esperamos que pierda en mercados con reversiones en V muy rápidas, donde la "
          "vela de reentrada llega cuando el movimiento de vuelta ya ha ocurrido. Queda refutada "
          "si su ventaja por operación no supera a la de la hipótesis 1: en ese caso la 'prueba de "
          "agotamiento' no aporta información y solo llega tarde.",
    fn=_bb_reentry,
    params={"periodo": 20, "desviaciones": 2.0, "salida": "SMA20"},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 3. Bollinger filtrado por régimen. Existe para MEDIR el modo de fallo declarado de la familia.
# --------------------------------------------------------------------------------------------


def _bb_range_adx(s: Series) -> np.ndarray:
    c, h, low = _f(s.close), _f(s.high), _f(s.low)
    out = _sig(c.size)
    upper, _mid, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
    adx = talib.ADX(h, low, c, timeperiod=14)
    ok = _finite(upper, lower, adx)
    quieto = ok & (adx < 20.0)
    out[quieto & (c < lower)] = 1
    out[quieto & (c > upper)] = -1
    return out


register(Hypothesis(
    name="mean_reversion.bb20_2_range_adx14",
    family="mean_reversion",
    rationale="La hipótesis 1 con una única puerta: operar solo cuando el ADX14 está por debajo "
              "de 20, el umbral de Wilder para 'sin tendencia'. La afirmación que se comprueba es "
              "estructural, no cosmética: la prima de reversión existe porque alguien absorbe "
              "flujo impaciente, y absorber solo es rentable cuando ese flujo NO está informado. "
              "Cuando hay tendencia, la orden que barre el libro suele traer información —alguien "
              "reposiciona de verdad— y el que la absorbe queda seleccionado adversamente. El ADX "
              "bajo es la aproximación clásica a 'aquí no hay nadie reposicionando'.",
    prior="Esperamos ventaja CLARAMENTE mayor que la hipótesis 1 sobre las mismas velas, y que la "
          "diferencia venga de eliminar pérdidas, no de añadir ganancias. Esperamos también menos "
          "de la mitad de operaciones. La hipótesis queda refutada si la ventaja no mejora a la "
          "de la 1: eso significaría que el ADX se entera del régimen demasiado tarde para servir "
          "de filtro, y entonces la explicación de la familia por selección adversa se queda sin "
          "apoyo empírico.",
    fn=_bb_range_adx,
    params={"periodo": 20, "desviaciones": 2.0, "adx_periodo": 14, "adx_max": 20.0},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 4. Keltner: la misma idea con la anchura medida en rango real y no en dispersión de cierres.
# --------------------------------------------------------------------------------------------


def _keltner_fade(s: Series) -> np.ndarray:
    c, h, low = _f(s.close), _f(s.high), _f(s.low)
    out = _sig(c.size)
    ema = talib.EMA(c, timeperiod=20)
    atr = talib.ATR(h, low, c, timeperiod=14)
    ok = _finite(ema, atr)
    upper = ema + 2.0 * atr
    lower = ema - 2.0 * atr
    out[ok & (c < lower)] = 1
    out[ok & (c > upper)] = -1
    return out


register(Hypothesis(
    name="mean_reversion.keltner20_atr14_fade",
    family="mean_reversion",
    rationale="Canal de Keltner en su forma moderna: EMA20 como ancla y 2xATR14 como anchura. "
              "Frente a Bollinger, cambia QUÉ se considera un movimiento normal. Bollinger mide "
              "la dispersión de los cierres; el ATR mide el rango real, mechas y huecos incluidos. "
              "La diferencia importa por un motivo concreto de microestructura: una cascada de "
              "liquidaciones deja una mecha enorme y un cierre recuperado, lo que ENSANCHA el ATR "
              "de inmediato pero apenas mueve la sigma de cierres. El canal de Keltner, por tanto, "
              "deja de dar señales justo después de un evento de liquidación, mientras que "
              "Bollinger sigue dándolas.",
    prior="Esperamos MENOS señales que Bollinger en las semanas posteriores a un desplome, y por "
          "esa razón una peor tasa de acierto pero una cola izquierda más corta: renuncia a los "
          "rebotes más rentables a cambio de no entrar en el peor momento. Esperamos que sea "
          "inferior a Bollinger en mercado tranquilo, donde el ATR se estrecha y el canal se "
          "vuelve hipersensible. Queda refutada si sus señales son indistinguibles de las de la "
          "hipótesis 1: eso indicaría que en BTC la mecha y el cierre llevan la misma información.",
    fn=_keltner_fade,
    params={"ema_periodo": 20, "atr_periodo": 14, "multiplicador": 2.0},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 5. RSI 14 con umbrales fijos 30/70. La línea base que Cardwell dice que está rota.
# --------------------------------------------------------------------------------------------


def _rsi_fixed(s: Series) -> np.ndarray:
    c = _f(s.close)
    out = _sig(c.size)
    rsi = talib.RSI(c, timeperiod=14)
    ok = _finite(rsi)
    out[ok & (rsi < 30.0)] = 1
    out[ok & (rsi > 70.0)] = -1
    return out


register(Hypothesis(
    name="mean_reversion.rsi14_fixed_3070",
    family="mean_reversion",
    rationale="RSI 14 con los umbrales originales de Wilder, 30 y 70. Se registra aunque casi "
              "nadie espere que funcione, y por un motivo metodológico: es el control de la "
              "hipótesis 6. La afirmación de Cardwell —que el RSI se 'clava' en el extremo durante "
              "las tendencias y por eso los umbrales fijos fallan— solo es comprobable si el "
              "control fijo está registrado de antemano. Si solo registrásemos la versión "
              "adaptativa y saliera bien, no sabríamos si el mérito es de las bandas de Cardwell "
              "o simplemente del RSI.",
    prior="Esperamos ventaja NULA o NEGATIVA neta, y en particular esperamos que el lado corto "
          "(RSI>70) sea el que más pierde, porque BTC pasa la mayor parte de su historia en "
          "tramos alcistas donde el RSI supera 70 y se queda ahí. Esperamos que la mayor parte de "
          "la pérdida se concentre en pocos tramos largos y direccionales. Queda refutada si el "
          "30/70 fijo iguala o supera a las bandas de Cardwell de la hipótesis 6: eso invalidaría "
          "la premisa central de esta familia sobre el sesgo del RSI en tendencia.",
    fn=_rsi_fixed,
    params={"periodo": 14, "sobreventa": 30.0, "sobrecompra": 70.0},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 6. RSI 14 con bandas de Cardwell condicionadas por la SMA200. La cabeza de cartel de la familia.
#    Mismo periodo 14 que la 5 a propósito: lo único que cambia es el esquema de umbrales, y esa
#    es exactamente la afirmación que se quiere contrastar.
# --------------------------------------------------------------------------------------------


def _rsi_cardwell(s: Series) -> np.ndarray:
    c = _f(s.close)
    out = _sig(c.size)
    rsi = talib.RSI(c, timeperiod=14)
    sma200 = talib.SMA(c, timeperiod=200)
    ok = _finite(rsi, sma200)

    alcista = ok & (c > sma200)   # rango de Cardwell 40-80
    bajista = ok & (c <= sma200)  # rango de Cardwell 20-60

    out[alcista & (rsi < 40.0)] = 1
    out[alcista & (rsi > 80.0)] = -1
    out[bajista & (rsi < 20.0)] = 1
    out[bajista & (rsi > 60.0)] = -1
    return out


register(Hypothesis(
    name="mean_reversion.rsi14_cardwell_sma200",
    family="mean_reversion",
    rationale="Andrew Cardwell observó que el RSI no oscila en el mismo rango en todos los "
              "regímenes: en tendencia alcista se mueve entre 40 y 80, y en bajista entre 20 y 60. "
              "El mecanismo detrás es de composición de participantes. En un mercado alcista, las "
              "caídas las provocan tomas de beneficio de operadores cortoplacistas y se encuentran "
              "con compradores estructurales esperando; la presión vendedora se agota antes, y por "
              "eso el suelo del oscilador está en 40 y no en 30. Comprar a 30 en ese régimen "
              "significa esperar una capitulación que casi nunca llega, y quedarse fuera. La "
              "SMA200 sobre el cierre decide qué régimen aplica con la información disponible en "
              "la barra, sin mirar adelante.",
    prior="Esperamos ventaja positiva y, sobre todo, MÁS OPERACIONES en el lado favorable al "
          "régimen (largos en alcista, cortos en bajista) que la hipótesis 5, y una reducción "
          "grande de las pérdidas del lado contrario. Esperamos que falle en los cambios de "
          "régimen, cuando el precio cruza la SMA200 varias veces seguidas: ahí las bandas "
          "alternan y se compra a 40 justo cuando el régimen relevante ya era el bajista. Queda "
          "refutada si no mejora a la hipótesis 5, o si toda su ventaja proviene del filtro "
          "direccional de la SMA200 y no de los umbrales asimétricos.",
    fn=_rsi_cardwell,
    params={"periodo": 14, "sma_regimen": 200,
            "banda_alcista": (40.0, 80.0), "banda_bajista": (20.0, 60.0)},
    timeframes=("1h", "4h", "1d"),
    min_warmup=260,
))


# --------------------------------------------------------------------------------------------
# 7. Williams %R: posición dentro del RANGO alto-bajo, no momento de cierres.
# --------------------------------------------------------------------------------------------


def _willr_fade(s: Series) -> np.ndarray:
    c, h, low = _f(s.close), _f(s.high), _f(s.low)
    out = _sig(c.size)
    wr = talib.WILLR(h, low, c, timeperiod=14)
    ok = _finite(wr)
    out[ok & (wr < -80.0)] = 1
    out[ok & (wr > -20.0)] = -1
    return out


register(Hypothesis(
    name="mean_reversion.willr14_fade",
    family="mean_reversion",
    rationale="El %R no mide momento: mide dónde cierra el precio dentro del rango máximo-mínimo "
              "de las últimas 14 velas. Eso lo hace sensible a un comportamiento distinto del que "
              "capta el RSI. Cerrar en el suelo absoluto del rango de dos semanas es la firma de "
              "una barrida de stops: un desplazamiento buscando liquidez acumulada bajo los "
              "mínimos previos, ejecutado por quien necesita contrapartida para una posición "
              "grande. Ese flujo es mecánico y no informado, y una vez ejecutado desaparece, que "
              "es justo la condición para que la absorción sea rentable.",
    prior="Esperamos ventaja positiva concentrada en horizontes CORTOS (pocas velas): el efecto de "
          "barrida se agota rápido y, si se mantiene la posición, la señal se convierte en una "
          "apuesta direccional que no es lo que la hipótesis afirma. Esperamos que falle en "
          "rupturas verdaderas de rango, donde el mínimo del rango se rompe porque hay "
          "información nueva y el %R se queda pegado a -100 durante todo el tramo. Queda refutada "
          "si la ventaja no decae con el horizonte de tenencia.",
    fn=_willr_fade,
    params={"periodo": 14, "sobreventa": -80.0, "sobrecompra": -20.0},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 8. CCI 20: desviación normalizada por desviación media ABSOLUTA (robusta a la cola).
# --------------------------------------------------------------------------------------------


def _cci_fade(s: Series) -> np.ndarray:
    c, h, low = _f(s.close), _f(s.high), _f(s.low)
    out = _sig(c.size)
    cci = talib.CCI(h, low, c, timeperiod=20)
    ok = _finite(cci)
    out[ok & (cci < -100.0)] = 1
    out[ok & (cci > 100.0)] = -1
    return out


register(Hypothesis(
    name="mean_reversion.cci20_fade",
    family="mean_reversion",
    rationale="El CCI usa el precio típico (H+L+C)/3 y lo normaliza por la desviación media "
              "ABSOLUTA, no por la desviación típica. La diferencia no es cosmética en BTC: la "
              "desviación típica eleva al cuadrado, así que una sola vela de liquidación infla el "
              "denominador durante 20 velas y anestesia la señal de Bollinger justo después del "
              "evento. La desviación media absoluta apenas se inmuta. La hipótesis, por tanto, es "
              "que el estadístico robusto sigue detectando dislocaciones en el periodo posterior "
              "a un choque, que es cuando la prima por proveer liquidez debería ser mayor porque "
              "el capital de los proveedores habituales está agotado.",
    prior="Esperamos que la ventaja del CCI supere a la de Bollinger (hipótesis 1) precisamente y "
          "solo en las ventanas que siguen a las velas de mayor rango, y que sea similar o "
          "ligeramente peor en el resto de la muestra por dar más señales de peor calidad. "
          "Esperamos el clásico fallo del +-100 en tendencia, igual que el resto de la familia. "
          "Queda refutada si su ventaja condicionada a 'después de un choque' no es mayor que la "
          "de Bollinger en esas mismas ventanas: ese era todo el argumento.",
    fn=_cci_fade,
    params={"periodo": 20, "umbral": 100.0},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 9. Estiramiento frente al ancla LARGA. Otro horizonte, otro participante.
# --------------------------------------------------------------------------------------------


def _stretch_ema200(s: Series) -> np.ndarray:
    c, h, low = _f(s.close), _f(s.high), _f(s.low)
    out = _sig(c.size)
    ema200 = talib.EMA(c, timeperiod=200)
    atr = talib.ATR(h, low, c, timeperiod=14)
    ok = _finite(ema200, atr)
    out[ok & (c < ema200 - 3.0 * atr)] = 1
    out[ok & (c > ema200 + 3.0 * atr)] = -1
    return out


register(Hypothesis(
    name="mean_reversion.stretch_ema200_atr14",
    family="mean_reversion",
    rationale="Las ocho hipótesis anteriores miden dislocaciones frente a un ancla de 14-20 velas, "
              "el horizonte del operador de rango. Esta cambia de escala: mide la distancia a la "
              "EMA200 en unidades de ATR14, con el multiplicador 3 de la convención de salidas por "
              "ATR. El participante es otro. La media de 200 periodos es la referencia que usan "
              "los asignadores de capital y los tenedores estructurales para decidir si el activo "
              "está 'caro' o 'barato'; cuando el precio se aleja tres rangos diarios de ella, la "
              "demanda que aparece no es de creadores de mercado sino de rebalanceo, y opera en "
              "una escala temporal de semanas.",
    prior="Esperamos MUY POCAS señales, ventaja positiva por operación pero con una varianza "
          "enorme y un horizonte de recuperación largo, de decenas de velas. Esperamos que el "
          "lado corto (3 ATR por encima de la EMA200) funcione peor que el largo, porque en las "
          "burbujas de BTC el precio se ha mantenido estirado por encima durante meses. Queda "
          "refutada si el número de señales es alto —eso significaría que 3 ATR no es un extremo "
          "en este activo y la premisa de 'dislocación estructural' es falsa— o si la ventaja se "
          "agota en pocas velas, lo que la haría indistinguible de la reversión de corto plazo.",
    fn=_stretch_ema200,
    params={"ema_periodo": 200, "atr_periodo": 14, "multiplicador": 3.0},
    timeframes=("4h", "1d"),
    min_warmup=420,
))


# --------------------------------------------------------------------------------------------
# 10. Extremo CON confirmación de volumen. Aquí es donde el "cierre forzado" se hace explícito.
# --------------------------------------------------------------------------------------------


def _bb_volume_climax(s: Series) -> np.ndarray:
    c, v = _f(s.close), _f(s.volume)
    out = _sig(c.size)
    upper, _mid, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
    vmedia = talib.SMA(v, timeperiod=20)
    ok = _finite(upper, lower, vmedia)
    climax = ok & (v > 2.0 * vmedia)
    out[climax & (c < lower)] = 1
    out[climax & (c > upper)] = -1
    return out


register(Hypothesis(
    name="mean_reversion.bb20_2_volume_climax",
    family="mean_reversion",
    rationale="Mismo extremo de Bollinger que la hipótesis 1, con una condición añadida: el "
              "volumen de la vela debe doblar su media de 20, el umbral convencional de volumen "
              "climático. El volumen es lo que separa los dos motivos por los que el precio puede "
              "salirse de la banda. Si sale con volumen bajo, es un libro fino y no hay prima que "
              "cobrar: nadie está pagando por salir. Si sale con volumen que dobla la media, hay "
              "alguien cerrando de forma forzada —liquidaciones en cascada de perpetuos, llamadas "
              "de margen— y ese vendedor no tiene opinión sobre el valor ni capacidad de esperar. "
              "Es el único caso en el que la hipótesis afirma que existe una contrapartida "
              "estructuralmente dispuesta a pagar.",
    prior="Esperamos ventaja MAYOR por operación que la hipótesis 1 y muchas menos operaciones. "
          "Esperamos que el efecto sea marcadamente asimétrico: mucho más fuerte en el lado largo "
          "(comprar capitulación) que en el corto, porque las liquidaciones bajistas en BTC son "
          "más violentas y más concentradas en el tiempo que las alcistas. Esperamos que falle en "
          "el primer día de un choque macro real, donde el volumen alto sí trae información. "
          "Queda refutada si el filtro de volumen no mejora la ventaja de la hipótesis 1: eso "
          "diría que el volumen no distingue flujo forzado de flujo informado en este mercado.",
    fn=_bb_volume_climax,
    params={"periodo": 20, "desviaciones": 2.0, "vol_periodo": 20, "vol_multiplo": 2.0},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 11. Barrida y rechazo. La única hipótesis de la familia que mira la forma de la vela.
# --------------------------------------------------------------------------------------------


def _sweep_rejection(s: Series) -> np.ndarray:
    c, h, low = _f(s.close), _f(s.high), _f(s.low)
    out = _sig(c.size)

    # Mínimo/máximo de las 20 velas ANTERIORES: talib.MIN(...)[i] cubre la ventana que termina en
    # i INCLUIDO, así que hay que desplazarlo una barra para excluir la vela actual. El
    # desplazamiento es hacia el futuro (_prev), nunca hacia atrás.
    min_prev = _prev(talib.MIN(low, timeperiod=20))
    max_prev = _prev(talib.MAX(h, timeperiod=20))

    rango = h - low
    ok = _finite(min_prev, max_prev, rango) & (rango > 0.0)

    # Rechazo: la vela perfora el extremo previo pero cierra en el tercio opuesto de su rango.
    barrida_baja = ok & (low < min_prev) & ((c - low) / np.where(ok, rango, 1.0) > 2.0 / 3.0)
    barrida_alta = ok & (h > max_prev) & ((h - c) / np.where(ok, rango, 1.0) > 2.0 / 3.0)

    out[barrida_baja] = 1
    out[barrida_alta] = -1
    return out


register(Hypothesis(
    name="mean_reversion.sweep_rejection_20",
    family="mean_reversion",
    rationale="La vela perfora el mínimo de las 20 anteriores pero cierra en el tercio superior de "
              "su propio rango. Esa forma es la huella observable de una secuencia concreta: bajo "
              "un mínimo visible se acumulan stops de compradores y órdenes de venta en ruptura; "
              "un participante que necesita comprar tamaño empuja el precio hasta ahí, esas "
              "órdenes se ejecutan y le dan la contrapartida que necesitaba, y el precio vuelve "
              "porque la venta era mecánica y se agotó en la propia vela. La clave es que el "
              "rechazo se confirma DENTRO de la misma barra, con el cierre: no hace falta ver la "
              "vela siguiente, y por eso la señal es causal.",
    prior="Esperamos ventaja positiva y de vida corta, con el grueso del movimiento en las "
          "primeras velas tras la señal. Esperamos que la ventaja sea mayor en 15m y 1h que en 4h, "
          "porque la acumulación de stops en niveles visibles es un fenómeno de microestructura y "
          "se difumina al agregar. Esperamos que falle cuando la perforación del mínimo es el "
          "inicio real de una pierna bajista: ahí el cierre en el tercio superior es solo un "
          "rebote técnico dentro de la caída. Queda refutada si no hay diferencia de ventaja entre "
          "las velas que perforan el mínimo previo y las que simplemente cierran fuertes.",
    fn=_sweep_rejection,
    params={"lookback": 20, "fraccion_rechazo": 2 / 3},
    timeframes=("15m", "1h", "4h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 12. Racha de tres cierres. Sin indicador, sin normalizar: la sobrerreacción en crudo.
# --------------------------------------------------------------------------------------------


def _streak3(s: Series) -> np.ndarray:
    c = _f(s.close)
    out = _sig(c.size)
    c1, c2, c3 = _prev(c), _prev(_prev(c)), _prev(_prev(_prev(c)))
    ok = _finite(c1, c2, c3)
    out[ok & (c < c1) & (c1 < c2) & (c2 < c3)] = 1
    out[ok & (c > c1) & (c1 > c2) & (c2 > c3)] = -1
    return out


register(Hypothesis(
    name="mean_reversion.streak3_fade",
    family="mean_reversion",
    rationale="Tres cierres consecutivos a la baja se compran; tres al alza se venden. No hay "
              "indicador, ni normalización por volatilidad, ni umbral que ajustar, y esa desnudez "
              "es el objetivo: es la prueba más limpia de la sobrerreacción por extrapolación. El "
              "comportamiento concreto es el del operador que confunde una racha corta con una "
              "tendencia y se posiciona en su dirección, junto con el gestor de riesgo que reduce "
              "exposición mecánicamente tras varios cierres en contra. Si la reversión a la media "
              "en BTC existe de verdad como fenómeno de comportamiento, tiene que aparecer aquí; "
              "si solo aparece en formulaciones más elaboradas, hay que sospechar que lo que "
              "estamos midiendo es un artefacto de esas formulaciones.",
    prior="Esperamos una ventaja MUY pequeña, apenas distinguible del ruido, y probablemente "
          "negativa después de costes por la alta frecuencia de señales; lo que esperamos que sea "
          "informativo es el SIGNO y su consistencia entre timeframes, no la magnitud. Esperamos "
          "que sea negativa en los tramos de tendencia fuerte, donde las rachas de tres se "
          "encadenan. Queda refutada, y con ella la premisa de sobrerreacción de toda la familia, "
          "si el signo es sistemáticamente negativo en todos los timeframes: eso indicaría que en "
          "BTC las rachas cortas continúan en lugar de revertir, y entonces las once hipótesis "
          "anteriores que funcionen estarían capturando otra cosa.",
    fn=_streak3,
    params={"velas_consecutivas": 3},
    timeframes=("15m", "1h", "4h", "1d"),
    min_warmup=200,
))
