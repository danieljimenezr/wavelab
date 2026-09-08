# El primer resultado honesto: los arquetipos de Elliott NO tienen ventaja

Fecha: 2026-09-08 · BTCUSDT 4h · 2017-08 → 2026-09 · 4.757.255 velas de 1m · 19.820 decisiones
evaluadas · **1.449 operaciones resueltas**

## El número

| costes | n | esperanza | acierto | profit factor | brazo nulo | **ventaja** |
|---|---|---|---|---|---|---|
| ×1,0 | 1.449 | +0,007R | 33,3% | 1,01 | +0,015R | **−0,008R** |
| ×1,5 | 1.449 | −0,023R | 33,3% | 0,97 | −0,015R | −0,008R |
| ×2,0 | 1.449 | −0,053R | 33,1% | 0,93 | −0,045R | −0,008R |

Intervalo de confianza bootstrap sobre la esperanza: **[−0,076R, +0,090R]**. Contiene el cero por
ambos lados. El mínimo exigido para admitir una operación era +0,15R, y el límite inferior ni se
acerca.

## La prueba decisiva: el brazo nulo

Cada señal generó una operación gemela con entrada en un instante aleatorio cercano, **misma
geometría en R y misma barrera temporal**. Aísla «¿aporta algo el conteo?» de «¿subió BTC?», que es
lo que en realidad mide un backtest sin control.

| | n | señal | nulo | ventaja | IC95 de la ventaja |
|---|---|---|---|---|---|
| largos | 658 | +0,144R | **+0,165R** | −0,021R | [−0,146, +0,105] |
| cortos | 791 | −0,107R | **−0,109R** | +0,003R | [−0,116, +0,122] |
| todos | 1.449 | +0,007R | +0,015R | −0,008R | [−0,095, +0,081] |

**Aquí está todo.** A primera vista parecía haber señal: los largos daban +0,144R y los cortos
−0,107R. Pero **el brazo nulo reproduce exactamente la misma asimetría** (+0,165R y −0,109R). Es
decir: en un periodo en que BTC subió un 1.748%, comprar funcionaba y vender no, y daba igual que la
entrada la eligiera Elliott o un dado. El conteo de ondas no aportó nada.

Sin el brazo nulo, este resultado se habría leído como «los largos de onda 2 funcionan» y habría
sido falso.

## La puntuación de Fibonacci tampoco predice nada

Correlación de rangos entre el ajuste de guías y el resultado: **rho = +0,0038, p = 0,886**.

Es decir: que un conteo encaje maravillosamente con las proporciones de Fibonacci **no dice nada**
sobre si esa operación va a ganar. Toda la función de puntuación —retroceso de la onda 2, extensión
de la 3, alternancia— no tiene poder predictivo detectable sobre 1.449 casos.

## Contexto

- Comprar y mantener: **+1.748%** ($4.261 → $78.734)
- La estrategia: **+10,2R** en 1.449 operaciones, es decir, ruido

## Qué NO significa esto

- **No significa que el código esté mal.** El motor causal, el arnés de determinismo y el brazo
  nulo funcionaron exactamente como debían: detectaron que no hay ventaja en vez de fabricar una.
- **No significa que Elliott sea inútil como marco.** Sigue produciendo un precio de invalidación
  objetivo, y eso vale como instrumento de definición de riesgo aunque la dirección sea ruido.
- **No es un resultado sorprendente.** Coincide con van Ginneken (Tilburg), el único estudio
  riguroso y abierto sobre Elliott algorítmico, que tampoco encontró rentabilidad significativa. Y
  el plan de este proyecto lo anticipaba textualmente: «la puerta de validación probablemente
  rechace la estrategia en su primera ejecución».

## Qué significa

**Que la maquinaria de honestidad se ha ganado su coste el primer día que se usó.** Sin el brazo
nulo emparejado, sin el arnés de causalidad y sin el filtro de costes, este sistema habría enseñado
señales con aspecto profesional y esperanza cero, y habrían hecho falta meses de operar en real para
descubrirlo.

## Ensayos registrados

Los cinco análisis de esta sesión están en `trials.sqlite`. N efectivo = 2 hashes de configuración.
Cualquier Sharpe futuro habrá de deflactarse por ese número: he mirado estos datos cinco veces, y
eso queda registrado en vez de olvidarse convenientemente.
