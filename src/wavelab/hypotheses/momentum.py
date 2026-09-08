"""Familia MOMENTUM: momento de serie temporal sobre BTC.

REGISTRO PREVIO. Escrito el 2026-09-08, antes de ejecutar un solo backtest y sin haber mirado
ningún resultado. Los parámetros son los convencionales de la literatura y quedan CONGELADOS: si
una hipótesis fracasa no se reintenta con otro valor. Las doce se cuentan para la corrección por
contraste múltiple, incluidas —sobre todo— las que esperamos que fracasen.

Mecanismos postulados en toda la familia, en orden de importancia:

  1. Infrarreacción. La información se difunde por capas (mesas OTC y tesorerías → derivados →
     minorista) durante días o meses. El precio se ajusta con retardo y ese ajuste tiene signo
     persistente.
  2. Manada. La entrada de participantes nuevos es autorreforzante mientras haya dinero marginal
     que entrar, y se agota de golpe cuando no lo hay.
  3. Seguimiento sistemático. Los fondos de tendencia y los productos indexados a momento
     rebalancean sobre ventanas conocidas y públicas, con órdenes grandes, lentas y direccionales.

Nota de causalidad: todos los estadísticos de este fichero se calculan hacia atrás. Los índices de
extremos previos (`_extremo_previo`) están acotados por construcción a `i - separacion`, y los
desplazamientos (`_atras`) rechazan k <= 0 con una excepción en vez de con un comentario.
"""

from __future__ import annotations

import numpy as np
import talib
from numpy.lib.stride_tricks import sliding_window_view

from wavelab.hypotheses.base import Hypothesis, Series, register

# --------------------------------------------------------------------------------------------
# Parámetros. UN valor por concepto, fijado a priori. No se buscan.
# --------------------------------------------------------------------------------------------
_TSMOM_LARGO = 360      # ~12 meses en velas diarias
_TSMOM_SALTO = 30       # ~1 mes, el que se salta
_TSMOM_CORTO = 30       # 1 mes  (consenso multi-horizonte)
_TSMOM_MEDIO = 90       # 3 meses
_TSMOM_SEMI = 180       # 6 meses
_ROC_CORTO = 10
_ROC_LARGO = 20
_AO_RAPIDA = 5          # Awesome Oscillator canónico de Bill Williams
_AO_LENTA = 34
_DIV_VENTANA = 100      # ventana en la que se busca el extremo previo
_DIV_SEPARACION = 10    # separación mínima entre el extremo previo y la barra actual
_LINREG = 20
_PERSIST_VENTANA = 20
_PERSIST_ALTO = 14      # 70 % de las velas de la ventana con el mismo signo
_ER_PERIODO = 10        # ventana canónica del ratio de eficiencia de Kaufman
_ER_UMBRAL = 0.30       # umbral de "camino eficiente". Fijado a priori, NO se ajusta.
_KAMA_ER = 10           # TA-Lib fija rápida=2 y lenta=30 internamente
_VOL_CORTA = 5
_VOL_LARGA = 20


# --------------------------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------------------------
def _f(x: np.ndarray) -> np.ndarray:
    """float64 contiguo, que es lo único que acepta TA-Lib."""
    return np.ascontiguousarray(x, dtype=np.float64)


def _base(s: Series, minimo: int) -> tuple[np.ndarray | None, np.ndarray]:
    """(cierres, salida). `cierres` es None si la serie es demasiado corta o inservible."""
    out = np.zeros(len(s), dtype=np.int8)
    c = _f(s.close)
    if c.size < minimo or not np.isfinite(c).any():
        return None, out
    return c, out


def _atras(x: np.ndarray, k: int) -> np.ndarray:
    """x[i-k], con NaN en las primeras k posiciones. k>0 siempre: mirar adelante es un error."""
    if k <= 0:
        raise ValueError(f"_atras exige k>0 (causalidad); recibido {k}")
    out = np.full(x.size, np.nan, dtype=np.float64)
    if k < x.size:
        out[k:] = x[:-k]
    return out


def _ao(s: Series) -> np.ndarray:
    """Awesome Oscillator: SMA(5) - SMA(34) del precio medio (H+L)/2."""
    hl = (_f(s.high) + _f(s.low)) / 2.0
    return talib.SMA(hl, _AO_RAPIDA) - talib.SMA(hl, _AO_LENTA)


def _extremo_previo(x: np.ndarray, buscar_max: bool) -> np.ndarray:
    """Índice del extremo de x dentro de [i-_DIV_VENTANA, i-_DIV_SEPARACION].

    Estrictamente causal: el índice devuelto para la barra i nunca supera i-_DIV_SEPARACION.
    -1 donde no hay ventana completa.
    """
    n = x.size
    j = np.full(n, -1, dtype=np.int64)
    largo = _DIV_VENTANA - _DIV_SEPARACION + 1
    if n < _DIV_VENTANA + 1:
        return j
    v = sliding_window_view(x, largo)            # v[k] = x[k : k+largo]
    rel = np.argmax(v, axis=1) if buscar_max else np.argmin(v, axis=1)
    i = np.arange(_DIV_VENTANA, n)
    k = i - _DIV_VENTANA                         # ventana = x[i-100 : i-9]
    j[i] = k + rel[k]
    return j


# --------------------------------------------------------------------------------------------
# 1. Momento 12-1
# --------------------------------------------------------------------------------------------
def _tsmom_12_1(s: Series) -> np.ndarray:
    c, out = _base(s, _TSMOM_LARGO + _TSMOM_SALTO + 1)
    if c is None:
        return out
    reciente = _atras(c, _TSMOM_SALTO)           # cierre de hace ~1 mes
    antiguo = _atras(c, _TSMOM_LARGO)            # cierre de hace ~12 meses
    ok = np.isfinite(reciente) & np.isfinite(antiguo) & (antiguo > 0)
    ret = np.full(c.size, np.nan)
    ret[ok] = reciente[ok] / antiguo[ok] - 1.0
    out[np.isfinite(ret) & (ret > 0)] = 1
    out[np.isfinite(ret) & (ret < 0)] = -1
    return out


register(Hypothesis(
    name="momentum.tsmom_12_1",
    family="momentum",
    rationale="Los fondos sistemáticos de tendencia y los productos indexados a momento reconstruyen "
              "cartera sobre una ventana de 12 meses; sus órdenes de rebalanceo son grandes, lentas y "
              "direccionales, y empujan el precio en el sentido del retorno pasado. Se excluye el último "
              "mes porque a ese horizonte domina la reversión posterior a las cascadas de liquidación y "
              "el inventario de los creadores de mercado, que tiene el signo contrario y contamina la "
              "medida. En BTC el mismo comportamiento llega por tesorerías corporativas, flujo de ETF de "
              "contado y minorista persiguiendo el titular de 'máximos de doce meses'.",
    prior="Esperamos ventaja positiva pequeña y persistente en diario, concentrada en 2017, 2020-21 y "
          "2023-24, y ventaja NEGATIVA en los 3-6 meses posteriores a un techo de ciclo (2018, 2022), "
          "cuando la señal permanece larga durante toda la caída. Queda falsada si la ventaja media es "
          "≤0 dentro del propio régimen tendencial: eso significaría que ni siquiera cuando el mecanismo "
          "debería actuar, actúa. "
          "NOTA DE AUDITORÍA (2026-09-08): `trend.tsmom_365d` registra el signo del rendimiento a "
          "365 días en 1d, es decir, esta misma apuesta SIN el salto del último mes. Ninguno de los "
          "dos ficheros declaraba al otro. No se elimina ninguna —el salto de un mes es la parte "
          "sustantiva de la hipótesis 12-1 y las dos series de señal difieren lo bastante como para "
          "que la comparación tenga contenido—, pero se deja escrito que son ensayos DEPENDIENTES "
          "sobre el mismo horizonte y el mismo timeframe: si ambas salen positivas eso es UN "
          "resultado, no dos, y el contraste informativo es si saltar el último mes aporta algo.",
    fn=_tsmom_12_1,
    params={"lookback": _TSMOM_LARGO, "salto": _TSMOM_SALTO},
    timeframes=("1d",),
    min_warmup=420,
))


# --------------------------------------------------------------------------------------------
# 2. Consenso multi-horizonte
# --------------------------------------------------------------------------------------------
def _tsmom_consenso(s: Series) -> np.ndarray:
    c, out = _base(s, _TSMOM_SEMI + 1)
    if c is None:
        return out
    r1 = talib.ROC(c, _TSMOM_CORTO)
    r3 = talib.ROC(c, _TSMOM_MEDIO)
    r6 = talib.ROC(c, _TSMOM_SEMI)
    ok = np.isfinite(r1) & np.isfinite(r3) & np.isfinite(r6)
    out[ok & (r1 > 0) & (r3 > 0) & (r6 > 0)] = 1
    out[ok & (r1 < 0) & (r3 < 0) & (r6 < 0)] = -1
    return out


register(Hypothesis(
    name="momentum.tsmom_consenso",
    family="momentum",
    rationale="La información se difunde a velocidades distintas: el minorista reacciona en días, las "
              "tesorerías y los ETF en meses. Exigir que los horizontes de 1, 3 y 6 meses coincidan en "
              "signo selecciona los tramos en que las tres cohortes empujan a la vez, que es cuando el "
              "desequilibrio del libro de órdenes es unidireccional y no hay una cohorte vendiendo a la "
              "otra. No es una rejilla: es una sola configuración fija, la terna canónica del momento de "
              "serie temporal acortada de 1/3/12 a 1/3/6 porque el ciclo de cripto dura la mitad.",
    prior="Esperamos menos tiempo en mercado y MAYOR ventaja por barra que tsmom_12_1, a cambio de "
          "llegar tarde a los suelos de 2019 y 2023 y de quedarse fuera durante los primeros rebotes. "
          "Queda falsada si la ventaja por barra no supera a la de tsmom_12_1: el consenso no aportaría "
          "nada y solo estaría reduciendo la muestra.",
    fn=_tsmom_consenso,
    params={"corto": _TSMOM_CORTO, "medio": _TSMOM_MEDIO, "semi": _TSMOM_SEMI},
    timeframes=("1d",),
    min_warmup=240,
))


# --------------------------------------------------------------------------------------------
# 3. ROC corto
# --------------------------------------------------------------------------------------------
def _roc_corto(s: Series) -> np.ndarray:
    c, out = _base(s, _ROC_CORTO + 1)
    if c is None:
        return out
    roc = talib.ROC(c, _ROC_CORTO)
    ok = np.isfinite(roc)
    out[ok & (roc > 0)] = 1
    out[ok & (roc < 0)] = -1
    return out


register(Hypothesis(
    name="momentum.roc10",
    family="momentum",
    rationale="Infrarreacción pura. Cuando aparece información relevante (un dato macro, un movimiento "
              "de un tenedor grande, una noticia regulatoria) no todos los participantes la procesan a la "
              "vez: el ajuste se reparte en varias velas porque las mesas grandes trocean sus órdenes para "
              "no mover el mercado. El signo del retorno de las últimas diez velas es la medida más "
              "directa de ese ajuste en curso, antes de que el posicionamiento se sature.",
    prior="Esperamos ventaja positiva en 4h y 1d y ventaja NULA O NEGATIVA en 1h, donde el ruido de "
          "microestructura y el coste por rotación se comen el efecto. Queda falsada si el signo del ROC "
          "no separa retornos futuros en ningún timeframe; y si la ventaja resultara MAYOR en 1h que en "
          "1d, el mecanismo que postulamos es falso: estaríamos midiendo autocorrelación de "
          "microestructura, no difusión de información.",
    fn=_roc_corto,
    params={"periodo": _ROC_CORTO},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 4. Estructura de plazos del ROC (aceleración)
# --------------------------------------------------------------------------------------------
def _roc_estructura(s: Series) -> np.ndarray:
    c, out = _base(s, _ROC_LARGO + 1)
    if c is None:
        return out
    # Tasa POR VELA, para que los dos horizontes sean comparables en escala.
    tasa_corta = talib.ROC(c, _ROC_CORTO) / _ROC_CORTO
    tasa_larga = talib.ROC(c, _ROC_LARGO) / _ROC_LARGO
    ok = np.isfinite(tasa_corta) & np.isfinite(tasa_larga)
    out[ok & (tasa_larga > 0) & (tasa_corta > tasa_larga)] = 1
    out[ok & (tasa_larga < 0) & (tasa_corta < tasa_larga)] = -1
    return out


register(Hypothesis(
    name="momentum.roc_estructura",
    family="momentum",
    rationale="Efecto de manada de segundo orden. Una tendencia que ACELERA indica que siguen entrando "
              "participantes nuevos —apalancamiento creciente, financiación de perpetuos al alza, compra "
              "por FOMO—, mientras que una que se desacelera indica que el comprador marginal se ha "
              "agotado y solo queda rotación entre los que ya están dentro. Comparamos la tasa por vela "
              "de 10 velas contra la de 20 para que la comparación sea de escala homogénea; no es una "
              "variante de roc10, es una condición sobre su derivada.",
    prior="Esperamos ventaja positiva y mayor por barra que roc10 mientras la tendencia acelera, y "
          "esperamos que FALLE justo en los blow-off tops, donde la aceleración es máxima en la víspera "
          "del giro (diciembre 2017, abril y noviembre 2021). Queda falsada si su ventaja es "
          "indistinguible de la de roc10: la aceleración no añadiría información sobre el nivel.",
    fn=_roc_estructura,
    params={"corto": _ROC_CORTO, "largo": _ROC_LARGO},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 5. Signo del Awesome Oscillator
# --------------------------------------------------------------------------------------------
def _ao_signo(s: Series) -> np.ndarray:
    c, out = _base(s, _AO_LENTA + 1)
    if c is None:
        return out
    ao = _ao(s)
    ok = np.isfinite(ao)
    out[ok & (ao > 0)] = 1
    out[ok & (ao < 0)] = -1
    return out


register(Hypothesis(
    name="momentum.ao_signo",
    family="momentum",
    rationale="El AO resta el precio medio de las últimas 5 velas al de las últimas 34, es decir compara "
              "el coste medio de la cohorte compradora reciente con el de la cohorte previa. Si es "
              "positivo, los que compraron hace poco están en beneficio y no tienen prisa por vender, de "
              "modo que la oferta en cada subida es fina; si es negativo, cada rebote se encuentra con "
              "gente saliendo a la par, que es oferta densa y renovada. Usa (H+L)/2 en vez del cierre, "
              "lo que reduce el peso del precio de subasta de cierre frente al rango realmente operado.",
    prior="Esperamos ventaja positiva en tendencia y en torno a cero o negativa en lateral, donde el "
          "cruce de las dos medias se produce con retraso sistemático. Esperamos que NO supere a un cruce "
          "de medias equivalente de la familia trend: si lo hace, el precio medio aporta algo que el "
          "cierre no, y eso sería un resultado en sí mismo. Queda falsada si el signo del AO no separa "
          "retornos futuros ni siquiera en régimen tendencial.",
    fn=_ao_signo,
    params={"rapida": _AO_RAPIDA, "lenta": _AO_LENTA},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 6. AO que CONFIRMA el extremo nuevo (folclore: onda 3)
# --------------------------------------------------------------------------------------------
def _ao_confirma(s: Series) -> np.ndarray:
    c, out = _base(s, _DIV_VENTANA + _AO_LENTA + 1)
    if c is None:
        return out
    ao = _ao(s)
    jmax = _extremo_previo(c, buscar_max=True)
    jmin = _extremo_previo(c, buscar_max=False)

    hay_max = jmax >= 0
    jm = np.where(hay_max, jmax, 0)
    fin_max = hay_max & np.isfinite(c) & np.isfinite(c[jm]) & np.isfinite(ao) & np.isfinite(ao[jm])
    out[fin_max & (c > c[jm]) & (ao > ao[jm]) & (ao > 0)] = 1

    hay_min = jmin >= 0
    jn = np.where(hay_min, jmin, 0)
    fin_min = hay_min & np.isfinite(c) & np.isfinite(c[jn]) & np.isfinite(ao) & np.isfinite(ao[jn])
    out[fin_min & (c < c[jn]) & (ao < ao[jn]) & (ao < 0)] = -1
    return out


register(Hypothesis(
    name="momentum.ao_confirma_extremo",
    family="momentum",
    rationale="Un máximo de precio nuevo acompañado de un AO MAYOR que en el máximo previo de la ventana "
              "significa que el desplazamiento del coste medio es más grande que la vez anterior: la "
              "cohorte que compra es más numerosa, no la misma gente rotando el mismo capital. Ese es el "
              "patrón que los practicantes de Elliott llaman tercera onda, la de mayor participación y "
              "menor resistencia. El mecanismo real es de manada: la ruptura confirmada atrae al "
              "seguidor sistemático y al minorista a la vez, y ambos compran en el mismo lado del libro.",
    prior="Esperamos ventaja positiva pero muy concentrada: la señal debe estar plana la mayor parte del "
          "tiempo y activarse en pocas barras por año. Esperamos que sea PEOR que ao_signo en ventaja "
          "total (por estar mucho menos tiempo en mercado) y MEJOR en ventaja por barra. Queda falsada si "
          "su ventaja por barra no supera a la de ao_divergencia_extremo: en ese caso la distinción "
          "onda 3 / onda 5 del folclore de Elliott carece de contenido empírico.",
    fn=_ao_confirma,
    params={"ventana": _DIV_VENTANA, "separacion": _DIV_SEPARACION,
            "rapida": _AO_RAPIDA, "lenta": _AO_LENTA},
    timeframes=("4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 7. AO que DIVERGE del extremo nuevo (folclore: onda 5)
# --------------------------------------------------------------------------------------------
def _ao_divergencia(s: Series) -> np.ndarray:
    c, out = _base(s, _DIV_VENTANA + _AO_LENTA + 1)
    if c is None:
        return out
    ao = _ao(s)
    jmax = _extremo_previo(c, buscar_max=True)
    jmin = _extremo_previo(c, buscar_max=False)

    hay_max = jmax >= 0
    jm = np.where(hay_max, jmax, 0)
    fin_max = hay_max & np.isfinite(c) & np.isfinite(c[jm]) & np.isfinite(ao) & np.isfinite(ao[jm])
    # "Twin peaks" bajista de Bill Williams: ambos picos por encima de cero.
    out[fin_max & (c > c[jm]) & (ao < ao[jm]) & (ao > 0) & (ao[jm] > 0)] = -1

    hay_min = jmin >= 0
    jn = np.where(hay_min, jmin, 0)
    fin_min = hay_min & np.isfinite(c) & np.isfinite(c[jn]) & np.isfinite(ao) & np.isfinite(ao[jn])
    out[fin_min & (c < c[jn]) & (ao > ao[jn]) & (ao < 0) & (ao[jn] < 0)] = 1
    return out


register(Hypothesis(
    name="momentum.ao_divergencia_extremo",
    family="momentum",
    rationale="Un máximo de precio nuevo con un AO MENOR que en el máximo previo dice que el precio se ha "
              "hecho con menos desplazamiento del coste medio: el máximo lo producen cierres de cortos y "
              "minorista rezagado, con un tamaño agregado menor, mientras la cohorte que empujó el tramo "
              "anterior ya está distribuyendo contra ellos. Se exige que ambos picos estén por encima de "
              "cero (twin peaks de Bill Williams) para no confundir agotamiento con rebote dentro de una "
              "caída. Emitimos la señal contraria al precio.",
    prior="Nuestro prior es DESFAVORABLE y lo registramos igualmente. Esperamos ventaja ligeramente "
          "negativa en el conjunto del histórico y claramente negativa dentro de las tendencias largas de "
          "2017-Q4 y 2020-Q4, donde la divergencia aparece y el precio sigue subiendo semanas; solo "
          "esperamos ventaja positiva en la vecindad inmediata de los techos de ciclo, que son cuatro "
          "eventos y por tanto n insuficiente. Queda falsada como herramienta operativa si su ventaja "
          "global no es positiva; se registra para dejar documentado el fracaso esperado de un artefacto "
          "muy popular, no para respaldarlo.",
    fn=_ao_divergencia,
    params={"ventana": _DIV_VENTANA, "separacion": _DIV_SEPARACION,
            "rapida": _AO_RAPIDA, "lenta": _AO_LENTA},
    timeframes=("4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 8. Aceleración por pendiente de regresión
# --------------------------------------------------------------------------------------------
def _linreg_aceleracion(s: Series) -> np.ndarray:
    c, out = _base(s, 2 * _LINREG + 2)
    if c is None:
        return out
    seguro = np.where(np.isfinite(c) & (c > 0), c, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_c = np.log(seguro)
    if not np.isfinite(log_c).any():
        return out
    pend = talib.LINEARREG_SLOPE(_f(log_c), _LINREG)
    delta = pend - _atras(pend, _LINREG)          # cambio en una ventana no solapada
    ok = np.isfinite(pend) & np.isfinite(delta)
    out[ok & (pend > 0) & (delta > 0)] = 1
    out[ok & (pend < 0) & (delta < 0)] = -1
    return out


register(Hypothesis(
    name="momentum.linreg_aceleracion",
    family="momentum",
    rationale="La pendiente por mínimos cuadrados de veinte log-precios usa las veinte observaciones, no "
              "solo los extremos, así que una única mecha de liquidación en cascada —que es oferta "
              "forzada, no demanda informada— no la domina. Su cambio en veinte velas mide si la fuerza "
              "que sostiene la tendencia crece o se agota. La hipótesis concreta es que el momento medido "
              "con un estimador robusto al ruido de liquidaciones sobrevive donde el medido con "
              "endpoints no.",
    prior="Solapa a propósito con roc_estructura y se registra igual: si la ventaja de linreg_aceleracion "
          "es materialmente MAYOR, la diferencia se debe al ruido de mechas y liquidaciones y no al "
          "concepto de aceleración; si son indistinguibles, el estimador robusto no compra nada en BTC. "
          "Esperamos diferencia pequeña. Queda falsada si ninguna de las dos versiones tiene ventaja, y "
          "esperamos que ambas fallen en los reinicios de tendencia tras capitulación, donde la pendiente "
          "aún es negativa cuando el suelo ya está puesto.",
    fn=_linreg_aceleracion,
    params={"periodo": _LINREG},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 9. Persistencia del signo del retorno
# --------------------------------------------------------------------------------------------
def _persistencia_signo(s: Series) -> np.ndarray:
    c, out = _base(s, _PERSIST_VENTANA + 2)
    if c is None:
        return out
    r = np.diff(c, prepend=np.nan)
    sube = (r > 0).astype(np.float64)             # NaN -> False -> 0.0, sin inventar señal
    baja = (r < 0).astype(np.float64)
    n_sube = talib.SUM(_f(sube), _PERSIST_VENTANA)
    n_baja = talib.SUM(_f(baja), _PERSIST_VENTANA)
    ok = np.isfinite(n_sube) & np.isfinite(n_baja)
    out[ok & (n_sube >= _PERSIST_ALTO)] = 1
    out[ok & (n_baja >= _PERSIST_ALTO)] = -1
    return out


register(Hypothesis(
    name="momentum.persistencia_signo",
    family="momentum",
    rationale="Cuenta cuántas de las últimas veinte velas cerraron al alza y descarta por completo la "
              "magnitud. El motivo es que los dos mecanismos dejan huellas distintas: la manada y el "
              "seguimiento sistemático producen una sucesión de compras pequeñas y repetidas —deriva con "
              "muchas velas del mismo signo—, mientras que una noticia o una liquidación produce magnitud "
              "enorme en una sola vela y ninguna persistencia. Separar frecuencia de tamaño es "
              "exactamente lo que contrasta esta hipótesis, y de paso la hace inmune a las colas gruesas "
              "de BTC.",
    prior="Esperamos ventaja positiva MENOR que la de roc10 pero más estable entre regímenes, por ser un "
          "estadístico de conteo. Esperamos que falle explícitamente en los tramos de deriva alcista "
          "lenta seguida de caídas verticales (segunda mitad de 2021): allí la frecuencia de velas verdes "
          "es alta y el retorno acumulado negativo, y la señal estará larga todo el trayecto. Queda "
          "falsada si el conteo de signos no separa retornos futuros en ningún timeframe.",
    fn=_persistencia_signo,
    params={"ventana": _PERSIST_VENTANA, "umbral": _PERSIST_ALTO},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 10. Momento filtrado por el ratio de eficiencia de Kaufman
# --------------------------------------------------------------------------------------------
def _er_filtrado(s: Series) -> np.ndarray:
    c, out = _base(s, _ER_PERIODO + 2)
    if c is None:
        return out
    neto = c - _atras(c, _ER_PERIODO)
    recorrido = talib.SUM(_f(np.abs(np.diff(c, prepend=np.nan))), _ER_PERIODO)
    ok = np.isfinite(neto) & np.isfinite(recorrido) & (recorrido > 0)
    er = np.full(c.size, np.nan)
    er[ok] = np.abs(neto[ok]) / recorrido[ok]
    eficiente = np.isfinite(er) & (er >= _ER_UMBRAL)
    out[eficiente & (neto > 0)] = 1
    out[eficiente & (neto < 0)] = -1
    return out


register(Hypothesis(
    name="momentum.er_kaufman_filtrado",
    family="momentum",
    rationale="El ratio de eficiencia divide el desplazamiento neto entre el recorrido total, así que es "
              "alto cuando el precio va de A a B casi en línea recta. Un camino recto es la firma de un "
              "desequilibrio sostenido del libro —alguien grande ejecutando en una dirección durante "
              "días—, y un camino en zigzag es la firma de compradores y vendedores alternándose contra "
              "un proveedor de liquidez. La hipótesis es que el momento solo paga en el primer caso, "
              "porque en el segundo el que persigue la tendencia le compra caro al que la provee y paga "
              "el diferencial una y otra vez.",
    prior="Esperamos que la ventaja condicionada a ER≥0,30 sea claramente superior a la de roc10 y que el "
          "filtro reduzca mucho el número de barras en posición. Queda falsada si la ventaja filtrada NO "
          "supera a la de roc10: eso significaría que la rectitud del camino no informa sobre la "
          "persistencia futura, que es justo el mecanismo postulado. El umbral 0,30 se fija a priori; si "
          "falla no se prueba otro umbral, la hipótesis se da por refutada.",
    fn=_er_filtrado,
    params={"periodo": _ER_PERIODO, "umbral": _ER_UMBRAL},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 11. Cruce del KAMA
# --------------------------------------------------------------------------------------------
def _kama_cruce(s: Series) -> np.ndarray:
    c, out = _base(s, _KAMA_ER + 32)
    if c is None:
        return out
    kama = talib.KAMA(c, _KAMA_ER)
    ok = np.isfinite(kama) & np.isfinite(c)
    out[ok & (c > kama)] = 1
    out[ok & (c < kama)] = -1
    return out


register(Hypothesis(
    name="momentum.kama_cruce",
    family="momentum",
    rationale="El KAMA usa el mismo ratio de eficiencia para interpolar entre una media de 2 y una de 30: "
              "se pega al precio cuando el camino es eficiente y se aplana cuando es ruidoso. El "
              "participante al que esto pretende evitar es concreto: el seguidor de tendencia que compra "
              "cada ruptura falsa en un lateral y le va entregando el diferencial al creador de mercado "
              "vela tras vela. Frente a er_kaufman_filtrado, que decide si operar, aquí la eficiencia "
              "decide la VELOCIDAD del filtro y nunca deja de tener opinión.",
    prior="Esperamos menos cambios de señal y mejor ventaja por barra que un cruce de media fija, y "
          "esperamos que falle en el arranque de las tendencias explosivas, donde la adaptación va por "
          "detrás y entra tarde. Queda falsada si su ventaja no supera a la del cruce de EMA fija de la "
          "familia trend: en ese caso adaptar la velocidad no compra nada y solo añade una pieza móvil.",
    fn=_kama_cruce,
    params={"periodo_er": _KAMA_ER, "rapida": 2, "lenta": 30},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------------
# 12. Momento confirmado por participación
# --------------------------------------------------------------------------------------------
def _momento_con_volumen(s: Series) -> np.ndarray:
    c, out = _base(s, _ROC_LARGO + _VOL_LARGA + 2)
    if c is None:
        return out
    v = _f(s.volume)
    if not np.isfinite(v).any():
        return out
    roc = talib.ROC(c, _ROC_LARGO)
    v_corta = talib.SMA(v, _VOL_CORTA)
    v_larga = talib.SMA(v, _VOL_LARGA)
    ok = np.isfinite(roc) & np.isfinite(v_corta) & np.isfinite(v_larga) & (v_larga > 0)
    participa = ok & (v_corta > v_larga)
    out[participa & (roc > 0)] = 1
    out[participa & (roc < 0)] = -1
    return out


register(Hypothesis(
    name="momentum.momento_con_volumen",
    family="momentum",
    rationale="La manada deja huella en el volumen: un tramo con participación creciente implica dinero "
              "nuevo entrando, mientras que uno con volumen decreciente es el mismo capital rotando entre "
              "los que ya están dentro, y ese se agota solo. Exigimos que la media de volumen de 5 velas "
              "supere a la de 20 al mismo tiempo que el retorno de 20 velas tiene signo definido, de modo "
              "que la señal solo esté activa cuando dirección y participación coinciden. Las dos ventanas "
              "de volumen no son dos variantes de un parámetro: hacen falta las dos para definir "
              "'creciente', y son el par convencional de volumen relativo.",
    prior="Esperamos ventaja superior a la del ROC solo en 1d y 4h, y FALLO en 1h por la estacionalidad "
          "intradía del volumen (sesiones asiática y americana) que dispara el filtro por hora del día y "
          "no por convicción. Esta es además la primera hipótesis de la familia que debería romperse si "
          "los datos vienen de un exchange con volumen inflado o lavado. Queda falsada si el filtro de "
          "volumen no mejora al ROC de 20 sin filtrar. "
          "NOTA DE AUDITORÍA (2026-09-08): ese control —el ROC de 20 sin filtrar— no estaba "
          "registrado en ninguna parte, de modo que el criterio de falsación de arriba no se podía "
          "ejecutar; se ha registrado como `flow.roc20_sin_filtro` y es contra él contra quien debe "
          "medirse esta hipótesis. Queda declarado además que "
          "`flow.tendencia_con_participacion` es la misma construcción (ROC-20 condicionado a "
          "expansión de volumen) con otra pareja de medias de volumen, y que las dos son ensayos "
          "dependientes aunque compartan pocas señales.",
    fn=_momento_con_volumen,
    params={"roc": _ROC_LARGO, "volumen_corto": _VOL_CORTA, "volumen_largo": _VOL_LARGA},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))
