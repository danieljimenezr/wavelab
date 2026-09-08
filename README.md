# WaveLab

Analizador de BTC en tiempo real: ondas de Elliott **causales**, régimen de mercado, y puntos de
entrada/salida con un precio de invalidación explícito.

Herramienta **visual y consultiva**. No ejecuta órdenes, no guarda claves de exchange, no se conecta a
ninguna cuenta. Tú decides y tú operas.

---

## Léeme antes de usarlo

### 1. El producto es el precio de invalidación, no la etiqueta de onda

En cada cierre de vela del timeframe de disparo, WaveLab responde a tres preguntas de forma falsable:

1. **¿En qué estructura está el mercado?** Un conjunto *ordenado* de hipótesis, nunca una sola.
2. **¿A qué precio exacto esa estructura es falsa?** Un precio que sale del motor de reglas y lleva
   escrito el nombre de la regla que lo produce (R1, R2b, R3, línea 2-4).
3. **¿El R:R resultante justifica la operación?** Un filtro que muestra la aritmética, diga sí o no.

Las etiquetas de onda son el **medio** para calcular el precio de invalidación. El precio de
invalidación es el **producto**.

### 2. Elliott puede no tener ventaja direccional

El único estudio riguroso y abierto sobre Elliott algorítmico (van Ginneken, Universidad de Tilburg:
tres algoritmos objetivos y replicables sobre AUD/USD, EUR/USD, USD/CAD y USD/JPY, 2005-2013, con
bootstrap de paseo aleatorio y ajuste por costes, riesgo y diferenciales de interés) **no encontró
evidencia concluyente de rentabilidad significativa**. Es un estudio no replicado, sobre otra clase de
activo y otra era — no es la última palabra, pero es la mejor evidencia pública disponible.

El valor defendible de esta herramienta no es que Elliott prediga. Es que Elliott produce un precio de
invalidación **objetivo**, y eso es un instrumento de definición de riesgo aunque la dirección sea ruido.

### 3. Va a decir NO_TRADE la mayor parte del tiempo

Es el comportamiento correcto y va a sentirse como que está rota. Cuando rechaza, muestra la aritmética
del rechazo.

### 4. La autonomía tiene un techo teórico, no de ingeniería

BTC desde 2017 son ~9 años. Un sistema swing da 40-80 operaciones al año, de las que solo ~1/7 quedan
fuera de muestra bajo walk-forward, repartidas en 4 regímenes → **13-35 observaciones por régimen**. A
esa n, un 70% y un 40% de acierto **no son distinguibles**.

Consecuencia: «el sistema decide qué indicadores usar o descartar» es satisfacible **a nivel de vela**
(decenas de miles de observaciones) y **no lo es a nivel de operación de arquetipo**. Por eso el diseño
separa las dos cosas:

- el **coeficiente de información inclina** los pesos usando datos abundantes de nivel de vela,
- el **Hedge selecciona** usando datos escasos de nivel de operación,

y la interfaz muestra los dos términos por separado para que veas cuál está actuando. Cualquier
herramienta que prometa selección autónoma de indicadores a este ritmo de datos está mintiendo.

### 5. Todo número de backtest está contaminado, incluidos los que pasan la puerta

Quien diseñó esto ya había visto la historia de BTC. Cada umbral, cada regla y cada nivel de Fibonacci
elegido es un ensayo implícito ajustado a ese mismo histórico. Por eso `trials.sqlite` registra cada
hash de configuración jamás evaluado, y el Deflated Sharpe se calcula con ese N efectivo real.

---

## Principios de diseño

**La causalidad es un tipo, no una convención.** `PivotStore.__getitem__` lanza `CausalityError`, de
modo que `as_of(t)` es el único camino alcanzable. `AsOf[T]` transporta `available_at`. El decorador
`@causal` verifica cada argumento. Una regla de linter se puede silenciar; un `raise` no. El repintado
es la única clase de bug que **falla hacia arriba** (el backtest sale *mejor*, no peor), así que tiene
que ser imposible de escribir, no meramente testeado.

**Una sola función, viva y en backtest.** El motor expone `on_bar(state, bar) -> (state, signals,
decision)` y el backtester ejecuta exactamente esa misma función. **No hay ruta vectorizada.** Si
backtest y live fueran dos implementaciones, su divergencia reintroduciría lookahead en silencio.

**Regla anti-astronauta.** Hay cinco costuras (`FeedAdapter`, `FeatureProvider`, `Production`,
`SignalSource`, `Veto`), cada una con un solo método que importa. **Si una costura necesita un segundo
método significativo para ser útil, está dibujada en el sitio equivocado.** Sin contenedor de inyección
de dependencias, sin descubrimiento por entry-points, sin framework de plugins.

**Ningún estadístico como número pelado.** Siempre la terna *(valor, n del que sale, estado de la
precondición)*. Si la precondición falla se muestra «no computable — faltan N», nunca un número. Un
estadístico que devuelve un valor tranquilizador a partir de datos insuficientes es peor que no tenerlo:
fabrica confianza.

---

## Lo que deliberadamente NO se construye

Está aquí para que no erosione dentro de tres meses:

- Búsqueda en rejilla, Optuna o genética sobre parámetros de indicadores. Los parámetros están fijados
  (RSI-14, ATR-14, EMA 21/55/200, MACD 12/26/9) y no se tocan. La búsqueda de parámetros **es** el
  generador de sobreajuste de backtest.
- Aprendizaje por refuerzo. Predicción neuronal de precio. Bandits sobre estrategias.
- Re-optimización continua y auto-descubrimiento de indicadores.
- Ejecución de órdenes, claves de exchange, o cualquier despliegue multiusuario.

---

## Estado

En construcción. Ver `docs/` y el orden de hitos M0 → M9.

## Licencia

MIT. Ver `LICENSE` y `NOTICE` para atribuciones de terceros.
