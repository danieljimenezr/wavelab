# El umbral del ZigZag: qué controla `k` en realidad

Medido sobre BTCUSDT, 2025-01 → 2026-09, con `min_pct=0` para aislar el efecto del ATR.

| tf | k | pivotes | velas/pivote | \|tramo\| mediano | retardo mediano |
|---|---|---|---|---|---|
| 15m | **1,5** | 10.441 | 5,7 | 0,75% | **2** |
| 15m | 3,0 | 2.730 | 21,7 | 1,50% | 5 |
| 15m | 5,0 | 856 | 69,1 | 2,79% | 15 |
| 15m | 8,0 | 281 | 210,4 | 4,80% | 47 |
| 1h | **1,5** | 2.496 | 5,9 | 1,65% | **1** |
| 1h | 5,0 | 221 | 66,9 | 5,39% | 16 |
| 4h | **1,5** | 601 | 6,1 | 3,65% | **2** |
| 4h | 3,0 | 169 | 21,9 | 6,49% | 5 |
| 4h | 5,0 | 54 | 68,4 | 12,43% | 19 |
| 1d | **1,5** | 100 | 6,2 | 9,46% | **2** |
| 1d | 5,0 | 6 | 102,5 | 32,01% | 24 |

## Lo que dicen los números

**1. `k` es adimensional de verdad.** La densidad de pivotes es prácticamente la misma en todos los
timeframes para un mismo `k`: ~6 velas por pivote con k=1,5, ~22 con k=3, ~68 con k=5, ~200 con k=8.
No es casualidad: el umbral es `k·ATR`, y el ATR escala con el timeframe. Esto es lo que hace real
el requisito multi-activo — el mismo k=1,5 significa lo mismo en BTC, en EURUSD y en AAPL.

**2. `k` fija el GRADO, el timeframe fija la ESCALA.** Con k=1,5 el tramo mediano es 0,75% en 15m,
1,65% en 1h, 3,65% en 4h y 9,46% en 1d. Es la propiedad fractal que Elliott necesita: la misma
estructura a distintas escalas.

**3. El coste de subir `k` NO es visual, es latencia.** En 15m, pasar de k=1,5 a k=8 baja de 10.441
pivotes a 281 —estructura mucho más limpia— pero el retardo mediano de confirmación pasa de **2 velas
a 47**, unas doce horas. Te enteras de que hubo un techo medio día después de que ocurriera.

Ese es el intercambio real, y no tiene solución: es un tiempo de primer paso a una barrera. Cuanto
más lejos pones la barrera, más tardas en cruzarla. Un ZigZag "limpio" es un ZigZag que te informa
tarde, y la mitad del valor de un pivote está en enterarte a tiempo.

## Decisión actual

`k = 1,5` y `min_pct = 0,005`, que son los valores del plan. **No se han ajustado mirando gráficos**:
esta tabla existe para que la elección futura sea informada, no para justificar un cambio ahora.

Si algún día se cambia, es un ENSAYO y va a `trials.sqlite` con su hash de configuración. La política
de "no buscamos parámetros" no elimina la búsqueda: la mueve a ajuste a ojo sin registrar, y entonces
el N efectivo del Deflated Sharpe cuenta ~1 mientras la carga real de contraste múltiple está en los
cientos.

## Nota sobre los grados

Un solo `k` produce **un grado** de pivotes. Los grados superiores NO salen de subir `k`: salen de la
recursión de la gramática (`Motive` y `Corr` a través de sí mismos), donde una onda 3 se parsea a su
vez como un impulso completo. Subir `k` para "ver ondas más grandes" sería sustituir estructura por
suavizado, y pagarlo en latencia.
