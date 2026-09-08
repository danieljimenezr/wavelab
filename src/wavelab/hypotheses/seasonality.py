"""Familia ESTACIONALIDAD: el reloj y el calendario como variable explicativa.

La tesis común a toda la familia es una sola: **la liquidez y la participación no son uniformes en
el tiempo**. Hay horas en las que el libro lo sostienen mesas americanas y horas en las que lo
sostiene flujo minorista apalancado; hay días en los que los raíles fiat están abiertos y días en
los que no; hay fechas en las que vence interés abierto y fechas en las que se reconstruye. Si el
retorno de BTC dependiese solo de la información, el reloj sería irrelevante. Si depende también de
QUIÉN puede operar en cada momento, no lo es.

Es también la familia más propensa al hallazgo espurio, y conviene decir por qué antes de mirar
nada: el calendario ofrece un número enorme de particiones (24 horas, 7 días, 31 días del mes, 4
trimestres, fases de ciclo) y casi todas son gratis de probar. Con suficientes cortes siempre
aparece uno "significativo". Las defensas que se adoptan aquí, todas ANTES de ejecutar nada:

1. **Ninguna ventana se elige por su rendimiento.** Cada una está anclada a un hecho institucional
   exógeno y verificable (apertura del contado americano, sesión de Tokio, cierre semanal del CME,
   sello de funding de los perpetuos, vencimiento de Deribit, halving). Si el ancla existe fuera
   del gráfico, la ventana no es un grado de libertad.
2. **Cada hipótesis declara cómo espera fracasar**, y en varios casos el criterio de falsación es
   más exigente que "gana dinero": si el mismo efecto aparece en ventanas de control equivalentes,
   la afirmación *estacional* es falsa aunque el P&L sea positivo.
3. **Estas hipótesis NO son ensayos independientes.** Varias comparten velas (fin de mes con
   vencimiento mensual; fin de semana con el viernes tarde; horas americanas con horas asiáticas,
   que son casi complementarias). Una corrección de Bonferroni sobre 11 ensayos es conservadora
   aquí, y una corrección que asuma independencia es sencillamente incorrecta: quien ejecute el
   contraste debería usar FDR bajo dependencia, o correlacionar las series de señal antes de
   decidir el número efectivo de ensayos. Se deja escrito para que no se decida después.
4. **Deriva incondicional.** BTC sube en la muestra. Cualquier hipótesis mayoritariamente larga
   heredará esa deriva. Ninguna de las de esta familia puede considerarse confirmada por su
   expectativa absoluta: solo cuenta el delta contra el brazo nulo (entradas aleatorias con la
   misma exposición), que es justo lo que el arnés de backtest ya calcula.

--------------------------------------------------------------------------------------------------
CONVENIOS ASUMIDOS (declarados aquí para que el resultado sea interpretable, y para que si el arnés
usa otro convenio se sepa exactamente qué hay que reinterpretar)

*   ``s.ts`` es la marca de APERTURA de la vela en ms epoch UTC, alineada a la rejilla del
    timeframe. Es el convenio del proyecto: ``resample_from_1m`` indexa por ``open_time_ms`` y
    ``Bar.__post_init__`` exige ``open_time_ms % tf.ms == 0``.

*   La señal en la vela ``i`` se decide con información hasta el cierre de ``i`` y la posición se
    mantiene durante la vela ``i+1``. Por eso las ventanas horarias se evalúan sobre
    ``ts + duración_del_timeframe``: la hora que interesa es la de la vela que se va a MANTENER, no
    la de la que se acaba de cerrar.

    Esto **no es mirar al futuro**. No se lee ni un dato de la vela ``i+1``: solo se calcula qué
    hora marcará el reloj, que es información conocida desde 1970 y disponible para cualquier
    participante en cualquier instante anterior. El mismo argumento vale para el calendario (cuándo
    cae el último viernes de marzo de 2019 se sabía en 2018) y para las fechas de halving, de las
    que solo se usan las ya ocurridas respecto a la vela evaluada.

    Si el arnés aplicase en cambio la señal ``i`` al retorno de la propia vela ``i``, TODAS las
    hipótesis de este fichero quedarían desplazadas una vela hacia atrás y habría que leerlas como
    ventanas adelantadas un periodo. Se declara ahora, no después de ver los números.

*   Las hipótesis de hora del día se registran solo en ``15m`` y ``1h``, donde la apertura y el
    cierre de la vela caen en la misma hora UTC y la asignación es inequívoca. En ``4h`` una vela
    abarca cuatro horas distintas y "la hora de la vela" no significa nada; registrarlas ahí sería
    fabricar una ambigüedad y luego atribuírsela al mercado.

*   El horario de verano (DST) desplaza en una hora los anclajes americanos (apertura del contado,
    cierre semanal del CME) durante parte del año. Se acepta ese ±1h en vez de introducir un
    calendario de DST: las ventanas se eligen lo bastante anchas para contener el ancla en ambos
    regímenes. Es una imprecisión declarada, no un parámetro ajustable.

CAUSALIDAD. Todas las funciones de este módulo usan exclusivamente datos hasta ``i`` incluido. El
relleno hacia adelante de referencias pasadas (apertura del día, cierre del viernes) se hace con
``np.maximum.accumulate`` sobre índices, que es una operación de prefijo: el valor en ``i`` solo
puede provenir de un índice ``<= i``. No hay ``shift`` negativos, ni ``np.roll``, ni detección de
extremos definida contra el array entero.
"""

from __future__ import annotations

import numpy as np
import talib

from wavelab.hypotheses.base import Hypothesis, Series, register

_MS_HOUR = 3_600_000
_MS_DAY = 86_400_000

#: Duración de cada timeframe. Es metadato estático (la rejilla del reloj), no un dato de mercado.
_TF_MS: dict[str, int] = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000,
    "1w": 604_800_000,
}

#: Halvings ya ocurridos, en ms epoch UTC. Solo se usa el ÚLTIMO ANTERIOR a cada vela: nunca se
#: consulta una fecha posterior a la vela evaluada, así que la tabla no introduce anticipación.
_HALVINGS_MS: np.ndarray = np.array(
    [
        np.datetime64("2012-11-28T15:24", "ms"),  # bloque 210.000
        np.datetime64("2016-07-09T16:46", "ms"),  # bloque 420.000
        np.datetime64("2020-05-11T19:23", "ms"),  # bloque 630.000
        np.datetime64("2024-04-20T00:09", "ms"),  # bloque 840.000
    ],
    dtype="datetime64[ms]",
).astype(np.int64)


# --------------------------------------------------------------------------------------
# Utilidades de reloj y calendario. Todas devuelven arrays alineados con la serie.
# --------------------------------------------------------------------------------------

def _hold_ts(s: Series) -> np.ndarray:
    """Marca temporal de la vela DURANTE la que estará abierta la posición decidida en ``i``.

    Es ``ts[i] + duración``: aritmética de reloj, no lectura de datos futuros (véase el docstring
    del módulo). El respaldo por diferencia de marcas solo actúa si aparece un timeframe que no
    esté en la tabla, y sigue siendo información de calendario.
    """
    ts = np.asarray(s.ts, dtype=np.int64)
    paso = _TF_MS.get(s.tf)
    if paso is None:
        paso = int(ts[1] - ts[0]) if ts.size > 1 else 0
    return ts + paso


def _hora_utc(t: np.ndarray) -> np.ndarray:
    """Hora UTC (0-23)."""
    return ((t // _MS_HOUR) % 24).astype(np.int64)


def _dia_semana(t: np.ndarray) -> np.ndarray:
    """Día de la semana con el convenio de Python: 0 = lunes … 6 = domingo.

    El 1970-01-01 (día epoch 0) fue jueves, que en este convenio es 3; de ahí el desplazamiento.
    """
    return (((t // _MS_DAY) + 3) % 7).astype(np.int64)


def _fecha(t: np.ndarray) -> np.ndarray:
    """Fecha UTC de cada marca, como ``datetime64[D]``."""
    return t.astype("datetime64[ms]").astype("datetime64[D]")


def _dia_del_mes(fechas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (día del mes 1..31, días que tiene ese mes)."""
    meses = fechas.astype("datetime64[M]")
    dom = (fechas - meses.astype("datetime64[D]")).astype(np.int64) + 1
    largo = ((meses + np.timedelta64(1, "M")).astype("datetime64[D]")
             - meses.astype("datetime64[D]")).astype(np.int64)
    return dom, largo


def _ultimo_viernes(meses: np.ndarray) -> np.ndarray:
    """Fecha del último viernes de cada mes ``datetime64[M]``.

    Es el vencimiento de Deribit (08:00 UTC) y el del CME. Se calcula del calendario, que es
    exógeno y conocido con años de antelación.
    """
    ultimo = (meses + np.timedelta64(1, "M")).astype("datetime64[D]") - np.timedelta64(1, "D")
    wd = (ultimo.astype(np.int64) + 3) % 7          # 0 = lunes, 4 = viernes
    return ultimo - ((wd - 4) % 7).astype("timedelta64[D]")


def _ffill_idx(marca: np.ndarray) -> np.ndarray:
    """Índice de la última posición marcada en ``<= i``, o -1 si aún no hubo ninguna.

    ``np.maximum.accumulate`` es una operación de PREFIJO: el resultado en ``i`` no puede depender
    de ninguna posición posterior. Es la pieza que mantiene causal todo relleno hacia adelante.
    """
    n = marca.size
    idx = np.where(marca, np.arange(n, dtype=np.int64), np.int64(-1))
    return np.maximum.accumulate(idx)


def _f64(x: np.ndarray) -> np.ndarray:
    """talib exige float64 contiguo; una vista de otro dtype lanza o calcula basura."""
    return np.ascontiguousarray(x, dtype=np.float64)


# --------------------------------------------------------------------------------------
# 1. Horas del contado americano
# --------------------------------------------------------------------------------------

def _horas_americanas(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    h = _hora_utc(_hold_ts(s))
    out[(h >= 13) & (h <= 20)] = 1
    return out


register(Hypothesis(
    name="seasonality.us_cash_hours_long",
    family="seasonality",
    rationale="Desde la aprobación de los ETF de contado, el comprador marginal de BTC opera en "
              "horario de Nueva York: las creaciones y reembolsos de los ETF se calculan contra el "
              "cierre de las 16:00 ET, la liquidez del CME se concentra en la sesión americana y "
              "las mesas que arbitran la base contado-futuro solo tienen personal delante de la "
              "pantalla en esas horas. Un flujo comprador recurrente, institucional y poco "
              "sensible al precio, concentrado en una ventana horaria fija, tiene que dejar huella "
              "en el retorno medio de esas horas. En el resto del día el libro lo sostienen "
              "creadores de mercado que por definición no tienen dirección.",
    prior="Esperamos deriva media POSITIVA en las velas mantenidas entre las 13:00 y las 21:00 "
          "UTC, y superior a la del brazo nulo con la misma exposición. Falsable por dos vías, y "
          "ambas cuentan como fracaso: (a) si la ventaja no sobrevive a la comparación con el "
          "brazo nulo, lo único que estamos midiendo es la deriva incondicional de BTC repartida "
          "por horas; (b) el mecanismo propuesto (flujo ETF/CME) es reciente, así que esperamos "
          "efecto débil o nulo antes de 2021 y más nítido desde enero de 2024 — si el efecto "
          "aparece uniforme en toda la muestra, el mecanismo que proponemos NO es el que lo "
          "produce y la hipótesis está mal explicada aunque acierte el signo.",
    fn=_horas_americanas,
    params={"hora_inicio_utc": 13, "hora_fin_utc": 20, "ancla": "sesión de contado de EE.UU."},
    timeframes=("15m", "1h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------
# 2. Horas asiáticas
# --------------------------------------------------------------------------------------

def _horas_asiaticas(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    h = _hora_utc(_hold_ts(s))
    out[(h >= 0) & (h <= 7)] = -1
    return out


register(Hypothesis(
    name="seasonality.asia_hours_short",
    family="seasonality",
    rationale="Entre las 00:00 y las 08:00 UTC (09:00-17:00 en Tokio) el libro está en su punto "
              "más fino: las mesas europeas y americanas no están, y lo que queda es sobre todo "
              "flujo minorista de perpetuos muy apalancado. Con un libro fino, una orden del mismo "
              "tamaño mueve más el precio y ese desplazamiento es mecánico, no informativo, así "
              "que tiende a devolverse cuando entra la profundidad de Londres. La participación de "
              "Asia oriental además se contrajo tras la prohibición china de 2021, con lo que el "
              "flujo marginal de esas horas se parece más al ruido que a una acumulación "
              "informada.",
    prior="Esperamos que la deriva de las horas 00:00-08:00 UTC quede POR DEBAJO de la "
          "incondicional, y en concreto que un corto en esa ventana tenga expectativa positiva "
          "neta de costes. Fracasa si: (a) la expectativa del corto es negativa, que es el "
          "resultado por defecto al ponerse corto de un activo con deriva secular positiva; (b) el "
          "resultado es un espejo exacto del de `us_cash_hours_long` — las dos ventanas son casi "
          "complementarias, así que si sus series de señal están correlacionadas cerca de -1 no "
          "son dos hallazgos sino uno, y hay que contarlas como UN ensayo. Esperamos además que el "
          "efecto sea más débil antes de 2021, cuando la participación asiática era mayor y su "
          "flujo sí era informativo.",
    fn=_horas_asiaticas,
    params={"hora_inicio_utc": 0, "hora_fin_utc": 7, "ancla": "sesión de Tokio"},
    timeframes=("15m", "1h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------
# 3. Solape Londres-Nueva York: continuación de la dirección del día
# --------------------------------------------------------------------------------------

def _solape_continuacion(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n == 0:
        return out

    ts = np.asarray(s.ts, dtype=np.int64)
    cierre = np.asarray(s.close, dtype=np.float64)
    apertura = np.asarray(s.open, dtype=np.float64)

    # Apertura del día UTC en curso: la de la primera vela del día, rellenada hacia adelante.
    dia = ts // _MS_DAY
    inicio_dia = np.empty(n, dtype=bool)
    inicio_dia[0] = True
    inicio_dia[1:] = dia[1:] != dia[:-1]
    idx = _ffill_idx(inicio_dia)                 # inicio_dia[0] es True ⇒ nunca queda en -1
    ancla = apertura[idx]

    h = _hora_utc(_hold_ts(s))
    ventana = (h >= 13) & (h <= 16)

    mov = cierre - ancla
    valido = ventana & ~np.isnan(mov)
    out[valido & (mov > 0)] = 1
    out[valido & (mov < 0)] = -1
    return out


register(Hypothesis(
    name="seasonality.overlap_day_trend",
    family="seasonality",
    rationale="El solape de la tarde de Londres con la mañana de Nueva York (13:00-16:30 UTC) es "
              "la única franja en la que mesas europeas y americanas están presentes a la vez, y "
              "por tanto la de mayor profundidad agregada del día. Una intención direccional "
              "formada durante las horas finas no se puede ejecutar en tamaño hasta que llega esa "
              "profundidad: los algoritmos de troceo institucionales (VWAP/TWAP) hacen ahí la "
              "mayor parte de su trabajo. El mecanismo predice continuación, no reversión: la "
              "dirección acumulada desde la apertura de las 00:00 UTC debería EXTENDERSE en esa "
              "ventana.",
    prior="Esperamos expectativa positiva al seguir el signo del movimiento del día en curso "
          "durante las 13:00-17:00 UTC. Fracasa si la expectativa es indistinguible de cero o del "
          "brazo nulo, y esperamos concretamente que fracase en régimen lateral de baja "
          "volatilidad, donde el precio orbita la apertura del día y el signo es una moneda al "
          "aire. Criterio adicional y decisivo: si el mismo seguimiento del signo del día funciona "
          "igual de bien en una ventana de control de cuatro horas cualquiera, entonces lo que hay "
          "es momento intradía genérico y la afirmación ESTACIONAL es falsa, gane lo que gane.",
    fn=_solape_continuacion,
    params={"hora_inicio_utc": 13, "hora_fin_utc": 16, "ancla": "apertura del día UTC (00:00)"},
    timeframes=("15m", "1h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------
# 4. Fin de semana
# --------------------------------------------------------------------------------------

def _fin_de_semana(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    wd = _dia_semana(_hold_ts(s))
    out[wd >= 5] = -1                     # sábado (5) y domingo (6)
    return out


register(Hypothesis(
    name="seasonality.weekend_short",
    family="seasonality",
    rationale="El fin de semana los raíles fiat están cerrados: no hay transferencias bancarias, "
              "no hay creación ni reembolso de participaciones de ETF y el CME no cotiza. Queda el "
              "perpetuo 24/7 con creadores de mercado que recortan inventario porque no pueden "
              "cubrirse en el mercado regulado, así que el libro adelgaza y las horquillas se "
              "abren. El flujo comprador que tiene calendario (institucional, nóminas, ETF) "
              "desaparece durante 48 horas, mientras que el flujo vendedor forzoso —liquidaciones "
              "de apalancamiento— no descansa nunca. La asimetría entre un comprador con horario "
              "de oficina y un vendedor automático es el mecanismo.",
    prior="Esperamos deriva de sábado y domingo menor o igual que cero y claramente por debajo de "
          "la de los días laborables. Fracasa si el fin de semana muestra deriva positiva "
          "comparable a la laborable. Esperamos además que el efecto se haya ATENUADO desde 2021, "
          "cuando los perpetuos y la creación de mercado automatizada pasaron a dominar el "
          "volumen: si solo aparece en la submuestra 2017-2020, es un fósil de un mercado que ya "
          "no existe y lo declararemos como tal en vez de venderlo como regularidad operable.",
    fn=_fin_de_semana,
    params={"dias": "sábado y domingo UTC"},
    timeframes=("1h", "4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------
# 5. Reducción de riesgo del viernes por la tarde
# --------------------------------------------------------------------------------------

def _viernes_tarde(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    t = _hold_ts(s)
    out[(_dia_semana(t) == 4) & (_hora_utc(t) >= 18)] = -1
    return out


register(Hypothesis(
    name="seasonality.friday_derisk_short",
    family="seasonality",
    rationale="El cierre semanal del CME (21:00 UTC en horario de verano, 22:00 en invierno) "
              "obliga a las mesas de base a aplanar o rodar lo que no pueden mantener contra un "
              "mercado cerrado, y los largos apalancados que pagan funding recortan antes de dos "
              "días sin raíles fiat ni cobertura regulada. Los dos flujos apuntan al mismo lado "
              "—vender— y se concentran en las últimas horas del viernes. Es una hipótesis "
              "distinta de la del fin de semana: aquí el mecanismo es el aplanamiento ANTES del "
              "cierre del venue, no la delgadez del libro DURANTE los dos días siguientes.",
    prior="Esperamos expectativa negativa para un largo (positiva para el corto) entre las 18:00 y "
          "las 24:00 UTC del viernes, de magnitud menor que el propio efecto de fin de semana. "
          "Fracasa si la expectativa no es negativa, y también —esto es lo importante— si la misma "
          "deriva negativa aparece en las horas 18-24 del resto de días laborables: en ese caso "
          "sería un efecto de hora del día, ya cubierto por otras hipótesis de esta familia, y no "
          "un efecto del viernes. La ventana se declara ancha (seis horas) a propósito para "
          "contener el cierre del CME en horario de verano y de invierno sin tener que elegir.",
    fn=_viernes_tarde,
    params={"dia": "viernes", "hora_inicio_utc": 18, "hora_fin_utc": 23,
            "ancla": "cierre semanal del CME"},
    timeframes=("15m", "1h"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------
# 6. Hueco del CME: reversión hacia el cierre del viernes
# --------------------------------------------------------------------------------------

def _hueco_cme(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n == 0:
        return out

    ts = np.asarray(s.ts, dtype=np.int64)
    cierre = np.asarray(s.close, dtype=np.float64)

    # Referencia: cierre de la vela que ABRE el viernes a las 21:00 UTC (última hora completa antes
    # del cierre semanal del CME). Se marca con el reloj de la PROPIA vela,
    # no con el de la siguiente.
    marca = (_dia_semana(ts) == 4) & (_hora_utc(ts) == 21)
    idx = _ffill_idx(marca)
    hay_ref = idx >= 0
    ref = cierre[np.clip(idx, 0, None)]

    # Frescura: entre el viernes 21:00 y el lunes 12:00 hay ~63 horas. Si la última referencia es
    # más vieja que eso, hubo un hueco de datos y la referencia ya no describe este fin de semana.
    fresca = (np.arange(n, dtype=np.int64) - idx) <= 96

    t = _hold_ts(s)
    wd, h = _dia_semana(t), _hora_utc(t)
    ventana = ((wd == 6) & (h >= 22)) | ((wd == 0) & (h <= 11))

    atr = talib.ATR(_f64(s.high), _f64(s.low), _f64(s.close), timeperiod=14)
    atr_ok = ~np.isnan(atr)

    hueco = cierre - ref
    valido = (ventana & hay_ref & fresca & atr_ok & ~np.isnan(hueco)
              & (np.abs(hueco) > np.where(atr_ok, atr, np.inf)))
    out[valido & (hueco > 0)] = -1   # por encima del cierre del viernes ⇒ hueco a la baja
    out[valido & (hueco < 0)] = 1
    return out


register(Hypothesis(
    name="seasonality.cme_gap_monday",
    family="seasonality",
    rationale="Los futuros de BTC del CME dejan de cotizar el viernes por la tarde y reabren el "
              "domingo por la noche, mientras el contado no para: cualquier movimiento del fin de "
              "semana abre un hueco en el gráfico del CME. Al reabrir, las mesas que arbitran la "
              "base tienen que reconstruir la cobertura entre ambos mercados, y ese flujo tira del "
              "contado hacia el último precio negociado en el venue regulado. El segundo "
              "ingrediente es reflexivo pero real: el hueco es un nivel que mira todo el mercado, "
              "y las órdenes se apilan donde la gente mira. Se exige que el hueco supere un "
              "ATR(14) porque por debajo de eso no hay nivel, hay ruido.",
    prior="Esperamos expectativa ligeramente positiva al desvanecer el movimiento del fin de "
          "semana (corto si el precio está por encima del cierre del viernes, largo si está por "
          "debajo) entre la reapertura del domingo y el mediodía del lunes. Fracasa si el "
          "movimiento del fin de semana resulta informativo en vez de ruido, es decir, si "
          "continúa; esperamos exactamente ese fracaso en tendencias fuertes y en fines de semana "
          "con noticias macro o quiebras de contrapartes. La dirección del fracaso es tan "
          "informativa como el acierto: si pierde de forma sistemática, lo aprendido es que el fin "
          "de semana SÍ mueve información.",
    fn=_hueco_cme,
    params={"atr": 14, "umbral_en_atr": 1.0, "ancla": "cierre del viernes 21:00 UTC",
            "ventana": "domingo 22:00 UTC - lunes 12:00 UTC", "frescura_max_velas": 96},
    timeframes=("1h",),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------
# 7. Hora previa al sello de funding
# --------------------------------------------------------------------------------------

def _pre_funding(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)

    roc = talib.ROC(_f64(s.close), timeperiod=8)      # retorno de la ventana de funding (8 h)
    finito = ~np.isnan(roc)

    h = _hora_utc(_hold_ts(s))
    ventana = (h == 7) | (h == 15) | (h == 23)

    valido = ventana & finito
    out[valido & (roc > 0)] = -1                      # se desvanece el movimiento de la ventana
    out[valido & (roc < 0)] = 1
    return out


register(Hypothesis(
    name="seasonality.pre_funding_fade",
    family="seasonality",
    rationale="Binance y el resto de venues de perpetuos liquidan el funding a las 00:00, 08:00 y "
              "16:00 UTC. Cuando la ventana de ocho horas ha sido muy direccional, el lado "
              "abarrotado es el que paga, y parte de ese lado cierra en la hora previa al sello "
              "justo para no pagarlo; simétricamente, el creador de mercado que lleva ocho horas "
              "acumulando el inventario contrario lo reduce antes del corte. Los dos flujos "
              "empujan contra el movimiento de la ventana y se concentran en esa hora concreta. El "
              "periodo de 8 horas no es un parámetro elegido: es la longitud del intervalo de "
              "funding.",
    prior="Esperamos autocorrelación NEGATIVA (desvanecimiento) en las horas 07, 15 y 23 UTC, y no "
          "en las otras veintiuna. El criterio de falsación es más duro que la rentabilidad y es "
          "la razón de ser de la hipótesis: si el mismo desvanecimiento de la ROC de 8 velas "
          "funciona igual en el resto de horas, lo que hay es reversión a la media genérica y la "
          "afirmación estacional es FALSA aunque el P&L sea positivo. Esperamos además que fracase "
          "en tendencia sostenida, donde el lado abarrotado es también el lado correcto y quien "
          "paga funding sigue ganando.",
    fn=_pre_funding,
    params={"ventana_funding_h": 8, "horas_utc_previas": (7, 15, 23)},
    timeframes=("1h",),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------
# 8. Cambio de mes
# --------------------------------------------------------------------------------------

def _cambio_de_mes(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    fechas = _fecha(_hold_ts(s))
    dom, largo = _dia_del_mes(fechas)
    out[(dom <= 3) | (dom == largo)] = 1
    return out


register(Hypothesis(
    name="seasonality.turn_of_month_long",
    family="seasonality",
    rationale="Las compras recurrentes tienen calendario: nóminas, programas automáticos de DCA en "
              "los exchanges y aportaciones periódicas de fondos y tesorerías se concentran a "
              "final de mes y en los primeros días del siguiente. Ese flujo es insensible al "
              "precio —compra el importe, cueste lo que cueste— y llega sincronizado a un libro "
              "cuyos vendedores no comparten esa sincronía. Es el mismo efecto de cambio de mes "
              "documentado en renta variable desde los años ochenta, trasplantado a un activo cuya "
              "base minorista compra por importe fijo y no por convicción puntual.",
    prior="Esperamos una ventaja pequeña y positiva en el último día natural del mes y en los tres "
          "primeros del siguiente, por encima del brazo nulo. Fracasa si la ventaja es nula, que "
          "es lo que cabe esperar si el flujo cripto no está sincronizado con ninguna nómina por "
          "ser una base global sin día de cobro común. Segundo criterio, y más exigente: si la "
          "ventaja se concentra en UNO solo de los cuatro días, es una coincidencia y no un flujo "
          "— el mecanismo predice un BLOQUE de días, no una fecha. Esperamos que se debilite en "
          "los meses cuyo día 1 cae en fin de semana, con los raíles fiat cerrados.",
    fn=_cambio_de_mes,
    params={"dias": "último día natural del mes y días 1, 2 y 3 del siguiente"},
    timeframes=("4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------
# 9. Semana previa al vencimiento trimestral
# --------------------------------------------------------------------------------------

def _previo_vencimiento_trimestral(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    fechas = _fecha(_hold_ts(s))
    meses = fechas.astype("datetime64[M]")
    trimestral = np.isin((meses.astype(np.int64) % 12), (2, 5, 8, 11))   # mar, jun, sep, dic
    dias_para = (_ultimo_viernes(meses) - fechas).astype(np.int64)
    out[trimestral & (dias_para >= 1) & (dias_para <= 7)] = -1
    return out


register(Hypothesis(
    name="seasonality.pre_quarterly_expiry_short",
    family="seasonality",
    rationale="El mayor interés abierto en opciones y futuros (Deribit y CME) vence el último "
              "viernes de marzo, junio, septiembre y diciembre. En la semana previa ese interés se "
              "cierra o se rueda: la base contado-futuro se comprime, las mesas que financiaban el "
              "carry deshacen el largo de contado que lo cubría y los tenedores de opciones "
              "ajustan delta contra un vencimiento cada vez más próximo. El neto de esas "
              "operaciones retira apalancamiento LARGO, que es el lado estructuralmente dominante "
              "en cripto, y retirar el lado dominante presiona el precio a la baja.",
    prior="Esperamos deriva levemente negativa en los siete días previos al vencimiento "
          "trimestral. Fracasa si la deriva es positiva o indistinguible del resto del trimestre. "
          "Advertencia que forma parte del registro: hay cuatro vencimientos trimestrales al año, "
          "así que la muestra efectiva son unas decenas de EVENTOS, no miles de velas; la "
          "corrección por contraste múltiple debe aplicarse sobre eventos y los intervalos de "
          "confianza calcularse por bloques, no por vela, o el error estándar saldrá falsamente "
          "diminuto. Si el efecto solo aparece en los trimestres de un año alcista concreto, lo "
          "que hay es ese año y no el vencimiento.",
    fn=_previo_vencimiento_trimestral,
    params={"meses": (3, 6, 9, 12), "dias_previos": 7,
            "ancla": "último viernes del trimestre, 08:00 UTC"},
    timeframes=("4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------
# 10. Días posteriores al vencimiento mensual
# --------------------------------------------------------------------------------------

def _posterior_vencimiento_mensual(s: Series) -> np.ndarray:
    out = np.zeros(len(s), dtype=np.int8)
    fechas = _fecha(_hold_ts(s))
    meses = fechas.astype("datetime64[M]")

    # El vencimiento vigente es el de este mes si ya pasó; si no, el del mes anterior. Sin esto,
    # los primeros días de mes (cuando el último viernes cae el 30 o el 31) quedarían fuera.
    v_actual = _ultimo_viernes(meses)
    v_previo = _ultimo_viernes(meses - np.timedelta64(1, "M"))
    vigente = np.where(fechas >= v_actual, v_actual, v_previo)

    dias_desde = (fechas - vigente).astype(np.int64)
    out[(dias_desde >= 1) & (dias_desde <= 3)] = 1
    return out


register(Hypothesis(
    name="seasonality.post_monthly_expiry_long",
    family="seasonality",
    rationale="Pasado el vencimiento mensual (último viernes, 08:00 UTC en Deribit) desaparece la "
              "cobertura de delta que ancla el precio a los strikes con más interés abierto, y el "
              "posicionamiento se reconstruye: las calls vencidas se ruedan al mes siguiente, las "
              "mesas vuelven a abrir el carry y el apalancamiento que se recortó regresa. Esa "
              "reconstrucción del posicionamiento largo es, mecánicamente, flujo comprador "
              "repartido en los días inmediatamente posteriores.",
    prior="Esperamos deriva positiva pequeña en los tres días siguientes a cada vencimiento "
          "mensual. Fracasa si es nula, y muy especialmente si resulta ser el espejo simétrico de "
          "`pre_quarterly_expiry_short`: si lo que medimos es un único pivote alrededor del "
          "vencimiento, no son dos hallazgos sino uno y hay que contarlo como UN ensayo. Esperamos "
          "que fracase en los meses en que el vencimiento cae junto al cambio de mes, donde su "
          "ventana se confunde con la de `turn_of_month_long` y ninguna de las dos es atribuible.",
    fn=_posterior_vencimiento_mensual,
    params={"dias_posteriores": 3, "ancla": "último viernes de cada mes, 08:00 UTC"},
    timeframes=("4h", "1d"),
    min_warmup=200,
))


# --------------------------------------------------------------------------------------
# 11. Fase del ciclo de halving
# --------------------------------------------------------------------------------------

def _fase_halving(s: Series) -> np.ndarray:
    n = len(s)
    out = np.zeros(n, dtype=np.int8)
    if n == 0:
        return out

    t = _hold_ts(s)
    # `side="right"` ⇒ solo halvings ESTRICTAMENTE anteriores o iguales a la vela. Nunca el próximo.
    k = np.searchsorted(_HALVINGS_MS, t, side="right") - 1
    hay = k >= 0
    dias = (t - _HALVINGS_MS[np.clip(k, 0, None)]) // _MS_DAY

    out[hay & (dias >= 180) & (dias <= 540)] = 1      # meses 6-18: expansión
    out[hay & (dias > 540) & (dias <= 900)] = -1      # meses 18-30: contracción
    return out


register(Hypothesis(
    name="seasonality.halving_cycle_phase",
    family="seasonality",
    rationale="El halving parte en dos la emisión de los mineros de un bloque para el siguiente. "
              "Los mineros son vendedores estructurales —pagan energía en moneda fiduciaria— así "
              "que el flujo vendedor diario que el mercado tiene que absorber se reduce a la mitad "
              "en una fecha conocida de antemano. El modelo folclórico del ciclo sostiene que el "
              "efecto no es instantáneo sino que se acumula durante los 6-18 meses siguientes, a "
              "medida que el déficit de oferta se manifiesta, y que da paso a una contracción de "
              "unos doce meses cuando el precio ya lo ha descontado y se agota el comprador "
              "marginal.",
    prior="Esperamos expectativa positiva en los meses 6-18 posteriores al halving y negativa en "
          "los meses 18-30. ADVERTENCIA, y forma parte del registro previo tanto como el signo: la "
          "muestra efectiva son DOS ciclos completos en una serie de Binance que empieza en 2017 "
          "—n≈2, no miles de velas—, y ninguna corrección por contraste múltiple arregla un n de "
          "2. Peor aún, los límites de 6, 18 y 30 meses proceden de una descripción escrita "
          "MIRANDO esos mismos ciclos: es contaminación dentro de muestra que ya no se puede "
          "deshacer. Declaramos por adelantado que leeremos el resultado como descriptivo y jamás "
          "como evidencia, y la registramos precisamente para que su resultado —éxito o fracaso— "
          "conste en el acta en vez de aparecer luego como una idea que 'siempre habíamos tenido'. "
          "Falsación adicional del mecanismo: si el ciclo iniciado en 2024, en el que la emisión "
          "ya es insignificante frente al volumen diario de los ETF, no reproduce el patrón, el "
          "mecanismo está muerto aunque la media histórica sobreviva.",
    fn=_fase_halving,
    params={"expansion_dias": (180, 540), "contraccion_dias": (541, 900),
            "halvings": ("2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20")},
    timeframes=("1d",),
    min_warmup=200,
))
