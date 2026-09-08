# El día que me salté mi propia arquitectura

2026-09-08. Este proyecto tiene: un tipo `AsOf` que lanza si lees antes de tiempo, un `PivotStore`
cuyo `__getitem__` lanza, un decorador `@causal` que verifica cada argumento, un tipo
`ProvisionalWindow` estructuralmente incapaz de alimentar la ruta causal, y un arnés de replay que
detecta lookahead comparando prefijos.

Y aun así escribí esto en un script de análisis suelto:

```python
r = np.diff(np.log(c))        # r[i] = retorno del día i al i+1
w = c[1:] > ema[1:]           # w[i] usa el cierre del día i+1
rendimiento = r * w           # decide con el dato que aún no existe
```

Resultado: CAGR del **99,3%**, Sharpe 1,62, caída máxima del −33%. Espectacular. Y falso.

La versión causal —cambiar `c[1:]` por `c[:-1]`— da CAGR **26,1%**, Sharpe 0,55, caída −56%. Es
decir: **peor que comprar y mantener** en retorno.

## Cómo se detectó

No por leer el código. Por una prueba de robustez rutinaria: *¿qué pasa si retraso la señal un día?*

| retraso | CAGR |
|---|---|
| 0 días | 99,3% |
| **1 día** | **26,1%** |
| 2 días | 23,5% |

Un efecto real no se evapora por ejecutar un día más tarde. **Si una estrategia muere al retrasarla
una barra, no es una estrategia: es una fuga de información.** Es la prueba más barata y más
brutal que existe, y debería ejecutarse siempre.

## La lección que importa

La arquitectura de causalidad protege el código que pasa POR ella. Un script de análisis rápido,
escrito para "solo echar un vistazo", no pasa por ninguna guarda — y es exactamente donde se toman
las decisiones sobre qué construir.

Y ocurrió DOS VECES el mismo día:
1. El Reality Check alimentado con retorno bruto en vez de exceso sobre la deriva, que premiaba a
   la estrategia que más tiempo pasaba larga en un activo que subió un 1.748%.
2. Este desalineamiento de un índice.

Si alguien que ha construido toda esta maquinaria comete el error dos veces en una tarde, la
conclusión razonable es que **la mayoría de las estrategias publicadas y vendidas tienen este bug**.
Y no por mala fe: porque el bug produce números preciosos y no da ningún error.

## Regla que se añade al proyecto

Todo análisis que produzca un número que influya en una decisión debe pasar, sin excepción:

1. **Prueba de retraso.** Retrasar la señal una barra. Si el resultado se desploma, hay fuga.
2. **Control aleatorio con la misma exposición.** Si un filtro aleatorio con la misma exposición
   media rinde parecido, no hay habilidad: hay beta.
3. **Exceso sobre la deriva, jamás retorno bruto.**
