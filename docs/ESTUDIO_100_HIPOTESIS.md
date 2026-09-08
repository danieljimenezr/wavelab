# Estudio exhaustivo: 100 hipótesis, 3 timeframes, 9 años. Ninguna sobrevive.

BTCUSDT · 2017-08 → 2026-09 · 4.757.335 velas de 1m · **238 combinaciones evaluadas**

## Cómo se hizo, y por qué importa el orden

1. **Registro previo.** Ocho agentes escribieron 100 hipótesis en 8 familias declarando `rationale`
   (por qué debería funcionar, con el mecanismo de mercado concreto) y `prior` (qué efecto se espera
   y **en qué condiciones quedaría falsada**) **antes de ver un solo resultado**.
2. **Auditoría independiente.** Prueba de invariancia de prefijo sobre las 100 × todos sus
   timeframes: calcular las señales sobre la serie completa y sobre un prefijo, y exigir igualdad
   vela a vela, con un shock de régimen inyectado tras el corte para delatar cualquier estadístico
   definido contra el array entero. **Cero fugas.** Además: 1 duplicado eliminado, 1 bug de signo
   corregido (afectaba al 2,7% de las velas) y 4 brazos de control añadidos porque seis hipótesis
   citaban un control que no existía.
3. **Evaluación con exceso sobre la deriva.** El estadístico NO es el retorno bruto sino
   `señal × (retorno − deriva media)`. Con el bruto, la ganadora es siempre la que más tiempo pasa
   larga, porque BTC subió un 1.748%: eso es beta, y comprarla cuesta cero.
4. **Reality Check de White** con bootstrap de bloques, sobre TODAS las hipótesis incluidas las
   fracasadas, más corrección de Šidák por haber probado 3 timeframes.

## El resultado

| timeframe | mejor del Reality Check | p corregido por 100 hipótesis |
|---|---|---|
| 1h | `trend.ema200_filter` | 0,1235 — no supera |
| **4h** | `trend.ema200_filter` | **0,0200 — supera** |
| 1d | `seasonality.halving_cycle_phase` | 0,0688 — no supera |

Corrección de Šidák por los 3 timeframes: **1 − (1 − 0,0200)³ = 0,0588**.

**No supera el 5%. Nada sobrevive.**

## Y lo mejor: la ganadora se falsa a sí misma

El prior de `trend.ema200_filter`, escrito antes de ver ningún número:

> «si el mecanismo es reflexivo, el efecto debe ser MAYOR en 1d, que es donde el nivel se publica y
> se mira, y casi nulo en 15m y 1h. Un edge igual o superior en 15m falsaría la explicación aunque
> el número saliera rentable.»

Lo medido:

| timeframe | exceso | fuera de muestra |
|---|---|---|
| 1h | +0,026% | **−0,023%** |
| 4h | **+0,139%** | +0,122% |
| 1d | **−0,063%** | **−0,119%** |

Es **negativa en 1d**, justo donde su propia teoría predecía el efecto más fuerte. Un efecto real no
cambia de signo entre timeframes contiguos: eso es la firma del ruido.

**La ganadora del contraste múltiple queda falsada por el criterio que ella misma escribió.** Sin el
registro previo, el titular habría sido «el filtro de EMA200 funciona en 4h, p=0,02».

## Degradación fuera de muestra

| timeframe | con exceso positivo EN muestra | FUERA de muestra |
|---|---|---|
| 1h | 19% | 12% |
| 4h | 45% | **25%** |
| 1d | 42% | 38% |

Si los efectos fueran reales, persistirían. En 4h casi la mitad desaparece al salir de muestra.

Detalle revelador: en 1h **solo el 19%** de las estrategias tiene exceso positivo. Menos de la mitad
de lo que darían monedas al aire — porque una señal equivocada no es neutral: paga el diferencial de
estar en el lado malo.

## Lo único que merece seguimiento (que NO es lo mismo que "funciona")

Dos hipótesis retienen su ventaja casi intacta fuera de muestra en 4h, que es la prueba más difícil
de pasar por azar:

| | en muestra | fuera | retiene | n efectivo |
|---|---|---|---|---|
| `candles.flujo_cuerpos_20` | +1,150% | +1,123% | **98%** | 149 |
| `momentum.ao_confirma_extremo` | +0,337% | +0,379% | **112%** | 387 |

No superan el contraste múltiple, así que **no son un hallazgo**. Son candidatas para verificación
FORWARD, con datos que nadie ha visto todavía. Seguir buscando hacia atrás sobre los mismos nueve
años no puede resolverlo: cada mirada adicional gasta más grados de libertad sin aportar
información nueva.

`candles.flujo_cuerpos_20` mide qué proporción del rango negociado en 20 velas se convirtió en
avance neto de apertura a cierre: alto cuando el rango se recorre en vez de ir y volver.
`momentum.ao_confirma_extremo` exige que un máximo de precio nuevo venga con un Awesome Oscillator
mayor que en el máximo anterior — la firma de que compra una cohorte nueva y no la misma rotando.

## Ensayos registrados

238 combinaciones + 3 Reality Checks. Todo a `trials.sqlite`. Cualquier resultado futuro sobre estos
mismos datos arrastra este número.
