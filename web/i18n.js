/**
 * A very small i18n layer: three plain objects and four functions.
 *
 * No framework and no build step, because this project vendors every dependency and has no Node
 * toolchain. Two pages, ~200 strings: anything heavier would cost more than it saves.
 *
 * English is the SOURCE language and the fallback. A key missing from es/ca renders in English
 * instead of showing the raw key: a page in mixed languages is still usable, a page full of
 * `assay.foot` is not.
 *
 * Three kinds of string are handled, and they are genuinely different problems:
 *   1. Static page copy      → `data-i18n` attributes in the HTML, replaced by `applyStatic()`.
 *   2. Strings built in JS   → `t(key, vars)`.
 *   3. Text the SERVER sends → `tx(text)`: the API speaks English only (test titles, verdict
 *      summaries, import reports), so it is translated on arrival by exact match first and then
 *      by pattern, keeping every number the server computed untouched. Anything unrecognised
 *      falls through in English, which is why this can never show a broken page.
 */

const STORE_KEY = 'wavelab.lang';

export const LANGS = ['en', 'es', 'ca'];
export const LANG_NAMES = { en: 'EN', es: 'ES', ca: 'CA' };

//: Number and date formatting follows the chosen language, so 1.234,5 / 1,234.5 come out the way
//: the reader expects. Lightweight Charts takes the same tag through `localization.locale`.
const LOCALE = { en: 'en-GB', es: 'es-ES', ca: 'ca-ES' };

// ---------------------------------------------------------------------------- dictionaries

const EN = {
  'lang.aria': 'Language',
  'nav.chart': 'Chart',
  'nav.assay': 'Assay',

  // ---- chart page
  'idx.analysing': 'analysing…',
  'family.candles': 'Candle shapes',
  'family.flow': 'Volume and flow',
  'family.mean_reversion': 'Mean reversion',
  'family.momentum': 'Momentum',
  'family.seasonality': 'Calendar and seasonality',
  'family.structure': 'Market structure',
  'family.trend': 'Trend following',
  'family.volatility': 'Volatility',
  'idx.public_server': "You are looking at the public server, and this chart is not on it.\n\n"
    + 'The chart is the owner\'s private view: /api/history, /api/decide and /ws answer 404 here '
    + 'on purpose. That is the guarantee working, not a fault.\n\n'
    + 'On macOS `localhost` resolves to ::1 before 127.0.0.1, so an ssh tunnel left open on port '
    + '8000 sends the browser to production without saying so. Open http://127.0.0.1:8000 instead, '
    + 'or close the tunnel.',
  'idx.chart_error': 'The chart could not load its history ($1). The panel would otherwise sit here '
    + 'saying "analysing…" for ever, which would be worse than telling you.',
  'banner.public_server': 'Public server — the chart lives on your local one (127.0.0.1:8000)',
  'banner.chart_error': 'The chart is not loading. See the panel.',
  'idx.connecting': 'connecting…',
  'idx.foot': 'view only · no orders · <kbd>1m</kbd> is the only series stored, everything else '
    + 'is resampled from it',
  'role.gate': 'veto',
  'role.regime': 'regime',
  'role.trigger': 'entries',
  'role.source': 'source',
  'feed.down': 'feed down',
  'feed.mute': 'connected but MUTE {s}s',
  'feed.live': 'feed live',
  'mode.live': 'up to date',
  'mode.catch_up': 'catching up',
  'mode.warmup': 'warming up',
  'health.status': 'status: {mode}',
  'health.gaps': 'gaps in window: {n}',
  'health.lag': 'lag: {n} bars',
  'health.reconnects': 'reconnects: {n} · bars healed: {m}',
  'banner.catch_up': 'Catching up after an outage. The state is being brought forward but NO '
    + 'decision is issued: an entry zone computed on stale bars describes a price that has '
    + 'already gone.',
  'banner.warmup': 'Warming up: loading history. No analysis yet.',
  'waves.line': 'pivots: {n} · ATR {atr} · {tail}',
  'waves.none': 'no structure',
  'waves.tentative': 'tentative {kind}, confirms at ${price}',
  'kind.high': 'high',
  'kind.low': 'low',
  'confirm.below': 'confirms below',
  'confirm.above': 'confirms above',
  'verdict.no_trade': 'NO TRADE',
  'verdict.watch': 'WATCH',
  'verdict.actionable': 'ACTIONABLE',
  'plan.zone': 'zone',
  'hyp.long': 'long',
  'hyp.short': 'short',
  'hyp.fit': 'fit {score}',
  'hyp.truncated': 'truncated',
  'hyp.invalidation': 'invalidation',
  'hyp.entry_zone': 'entry zone',
  'hyp.stop': 'stop',
  'hyp.targets': 'targets',
  'hyp.arith': 'R:R to T2 <b>{rr}</b> · stop at <b>{atr} ATR</b><br>'
    + 'cost <b>{cost}%</b> of R · size <b>{size}%</b><br>'
    + 'this R:R needs you to be right <b>{p}%</b> of the time',
  'panel.maturity': 'level {n} · prior',
  'panel.ranked': 'ranked hypotheses',
  'panel.footer': 'A <b>visual</b> tool. It places no orders.<br>'
    + 'The invalidation price is the only thing this framework produces objectively: it is where '
    + 'your idea is false.',

  // ---- assay page
  'assay.title': 'Assay — is your strategy real?',
  'assay.tagline': 'An <i>assay</i> is the test that decides whether an ore is gold or pyrite. '
    + 'This does the same to a trading strategy: five tests that almost nothing passes. It does '
    + 'not tell you when to buy — it tells you whether what you believe works holds up.',
  'tab.catalog': 'Catalogue',
  'tab.csv': 'My CSV',
  'tab.rule': 'Write a rule',
  'sel.loading': 'loading…',
  'tf.1d': 'daily',
  'tf.4h': '4 hours',
  'tf.1h': '1 hour',
  'btn.battery': 'Put it through the battery',
  'btn.validate_csv': 'Validate my strategy',
  'btn.validate_rule': 'Validate my rule',
  'catalog.count': '{n} strategies registered',
  'csv.hint': 'Upload a CSV with <b>a date column</b> and <b>a signal column</b>. It takes ISO '
    + 'dates or epochs, and signals like <code>1/0/-1</code>, <code>long/flat/short</code> or '
    + '<code>buy/sell</code>. A position is held until you change it.<br>'
    + '<b>Add a price column</b> and it is validated against <b>your</b> asset — equities, FX, '
    + 'whatever crypto. Without one, our BTC series is used.',
  'csv.ex1': '<b>signals only</b> (validated against BTC)',
  'csv.ex2': '<b>with your own asset</b>',
  'csv.code1': 'date,signal\n2024-01-15,long\n2024-03-02,flat\n2024-04-10,short',
  'csv.code2': 'date,price,signal\n2024-01-15,185.50,long\n2024-01-16,187.20,long\n'
    + '2024-01-17,183.10,flat',
  'rule.hint': 'Write when to be <b>long</b> and, if you want, when to be <b>short</b>. '
    + 'When neither holds, you are out of the market.',
  'rule.start': 'Start from an example:',
  'ex.trend': 'simple trend',
  'ex.rsi': 'RSI extremes',
  'ex.cross': 'moving-average cross',
  'ex.breakout': 'channel breakout',
  'ex.range': 'range expansion',
  'rule.long': 'Long when…',
  'rule.short': 'Short when… <span style="color:var(--dim)">(optional)</span>',
  'rule.short_ph': 'e.g. close < ema(close, 200)',
  'rule.help': 'Series and functions available',
  'assay.foot': '<b>Assay</b> is a <b>validation</b> tool, not investment advice. A strategy '
    + 'surviving the battery does not prove it makes money: it proves it is not one of the five '
    + 'known mistakes that make a beautiful backtest lose money live.',
  'run.step1': 'Computing the signal over the whole series…',
  'run.step2': 'Re-running it one bar late (look-ahead detector)…',
  'run.step3': "Comparing against the market's base rate…",
  'run.step4': '<b>Generating 250 random filters with the same exposure…</b> '
    + '(this is the part that takes the time)',
  'run.step5': 'Cutting the history with purging to measure out of sample…',
  'err.title': 'Could not validate',
  'err.no_file': 'pick a CSV file first',
  'v.survives': 'SURVIVES',
  'v.doubtful': 'DOUBTFUL',
  'v.fails': 'DOES NOT SURVIVE',
  'v.lookahead': 'IT READS THE FUTURE',
  'tag.pass': 'PASS',
  'tag.fail': 'FAIL',
  'tag.inconclusive': 'NO DATA',
  'kpi.cagr': 'CAGR',
  'kpi.sharpe': 'Sharpe',
  'kpi.dd': 'Worst drawdown',
  'kpi.exposure': 'Exposure',
  'kpi.episodes': 'Episodes',
  'kpi.bh': 'buy and hold {v}',
  'kpi.in_market': 'of the time inside the market',
  'kpi.of_signals': 'of {n} signals',
  'report.heading': 'This is what I understood from your data:',
  'meta.rationale': '<b>Why it should work</b> (stated before seeing any result)',
  'meta.prior': 'When it would be proved wrong',
  'btn.copy': 'Copy report',
  'btn.copied': 'copied ✓',
  'btn.download': 'Download report (.md)',
  'chart.strategy': 'strategy',
  'chart.bh': 'buy and hold',
  'md.strategy': 'strategy',
  'md.bh': 'buy and hold',
  'md.dd': 'Worst drawdown',
  'md.exposure': 'Exposure',
  'md.episodes': 'Independent episodes',
  'md.tests': 'The five tests',
  'md.reading': 'Reading of your data',
  'md.rationale': 'Reasoning stated before measuring',
  'md.prior': 'When it would be proved wrong:',
  'md.footer': 'Generated by **Assay** (DR Markets). A validation tool, not investment advice.\n'
    + 'A strategy surviving does not prove it makes money: it proves it is not one of the five '
    + 'known mistakes that make a beautiful backtest lose money live.',
};

const ES = {
  'lang.aria': 'Idioma',
  'nav.chart': 'Gráfico',
  'nav.assay': 'Assay',

  'idx.analysing': 'analizando…',
  'family.candles': 'Formas de vela',
  'family.flow': 'Volumen y flujo',
  'family.mean_reversion': 'Reversión a la media',
  'family.momentum': 'Momento',
  'family.seasonality': 'Calendario y estacionalidad',
  'family.structure': 'Estructura de mercado',
  'family.trend': 'Seguimiento de tendencia',
  'family.volatility': 'Volatilidad',
  'idx.public_server': 'Estás viendo el servidor público, y este gráfico no está ahí.\n\n'
    + 'El gráfico es la vista privada del propietario: /api/history, /api/decide y /ws responden '
    + '404 aquí a propósito. Es la garantía funcionando, no un fallo.\n\n'
    + 'En macOS «localhost» se resuelve a ::1 antes que a 127.0.0.1, así que un túnel ssh abierto '
    + 'en el puerto 8000 manda el navegador a producción sin decírtelo. Abre '
    + 'http://127.0.0.1:8000 o cierra el túnel.',
  'idx.chart_error': 'El gráfico no ha podido cargar su histórico ($1). Si no, el panel se quedaría '
    + 'aquí diciendo «analizando…» para siempre, que sería peor que contártelo.',
  'banner.public_server': 'Servidor público — el gráfico está en el tuyo local (127.0.0.1:8000)',
  'banner.chart_error': 'El gráfico no carga. Mira el panel.',
  'idx.connecting': 'conectando…',
  'idx.foot': 'solo visual · sin órdenes · <kbd>1m</kbd> es la única serie almacenada, el resto '
    + 'se resamplea',
  'role.gate': 'veto',
  'role.regime': 'régimen',
  'role.trigger': 'entradas',
  'role.source': 'fuente',
  'feed.down': 'feed caído',
  'feed.mute': 'conectado pero MUDO {s}s',
  'feed.live': 'feed en vivo',
  'mode.live': 'al día',
  'mode.catch_up': 'poniéndose al día',
  'mode.warmup': 'calentando',
  'health.status': 'estado: {mode}',
  'health.gaps': 'huecos en ventana: {n}',
  'health.lag': 'retraso: {n} velas',
  'health.reconnects': 'reconexiones: {n} · velas curadas: {m}',
  'banner.catch_up': 'Poniéndose al día tras un corte. El estado se está actualizando pero NO se '
    + 'emite ninguna decisión: una zona de entrada calculada sobre velas antiguas describe un '
    + 'precio que ya pasó.',
  'banner.warmup': 'Calentando: cargando histórico. Aún no hay análisis.',
  'waves.line': 'pivotes: {n} · ATR {atr} · {tail}',
  'waves.none': 'sin estructura',
  'waves.tentative': '{kind} tentativo, confirma en ${price}',
  'kind.high': 'máximo',
  'kind.low': 'mínimo',
  'confirm.below': 'confirma por debajo',
  'confirm.above': 'confirma por encima',
  'verdict.no_trade': 'NO OPERAR',
  'verdict.watch': 'VIGILAR',
  'verdict.actionable': 'ACCIONABLE',
  'plan.zone': 'zona',
  'hyp.long': 'largo',
  'hyp.short': 'corto',
  'hyp.fit': 'ajuste {score}',
  'hyp.truncated': 'truncada',
  'hyp.invalidation': 'invalidación',
  'hyp.entry_zone': 'zona de entrada',
  'hyp.stop': 'stop',
  'hyp.targets': 'objetivos',
  'hyp.arith': 'R:R a T2 <b>{rr}</b> · stop a <b>{atr} ATR</b><br>'
    + 'coste <b>{cost}%</b> de R · tamaño <b>{size}%</b><br>'
    + 'este R:R exige acertar el <b>{p}%</b> de las veces',
  'panel.maturity': 'nivel {n} · prior',
  'panel.ranked': 'hipótesis ordenadas',
  'panel.footer': 'Herramienta <b>visual</b>. No ejecuta órdenes.<br>'
    + 'El precio de invalidación es lo único que este marco produce de forma objetiva: es dónde '
    + 'tu idea es falsa.',

  'assay.title': 'Assay — ¿tu estrategia es real?',
  'assay.tagline': 'Un <i>assay</i> es el ensayo que determina si un mineral es oro o pirita. '
    + 'Esto hace lo mismo con una estrategia de trading: cinco pruebas que casi ninguna supera. '
    + 'No te dice cuándo comprar — te dice si lo que crees que funciona resiste.',
  'tab.catalog': 'Catálogo',
  'tab.csv': 'Mi CSV',
  'tab.rule': 'Escribir una regla',
  'sel.loading': 'cargando…',
  'tf.1d': 'diario',
  'tf.4h': '4 horas',
  'tf.1h': '1 hora',
  'btn.battery': 'Someter a la batería',
  'btn.validate_csv': 'Validar mi estrategia',
  'btn.validate_rule': 'Validar mi regla',
  'catalog.count': '{n} estrategias registradas',
  'csv.hint': 'Sube un CSV con <b>una columna de fecha</b> y <b>una de señal</b>. Acepta fechas '
    + 'ISO o epoch, y señales como <code>1/0/-1</code>, <code>largo/fuera/corto</code> o '
    + '<code>buy/sell</code>. La posición se mantiene hasta que la cambies.<br>'
    + '<b>Si añades una columna de precio</b>, se valida contra <b>tu</b> activo — acciones, '
    + 'divisas, la cripto que sea. Sin ella se usa nuestra serie de BTC.',
  'csv.ex1': '<b>solo señales</b> (se valida contra BTC)',
  'csv.ex2': '<b>con tu propio activo</b>',
  'csv.code1': 'fecha,señal\n2024-01-15,largo\n2024-03-02,fuera\n2024-04-10,corto',
  'csv.code2': 'fecha,precio,señal\n2024-01-15,185.50,largo\n2024-01-16,187.20,largo\n'
    + '2024-01-17,183.10,fuera',
  'rule.hint': 'Escribe cuándo estar <b>largo</b> y, si quieres, cuándo estar <b>corto</b>. '
    + 'Si ninguna se cumple, estás fuera del mercado.',
  'rule.start': 'Empieza por un ejemplo:',
  'ex.trend': 'tendencia simple',
  'ex.rsi': 'RSI extremos',
  'ex.cross': 'cruce de medias',
  'ex.breakout': 'ruptura de canal',
  'ex.range': 'expansión de rango',
  'rule.long': 'Largo cuando…',
  'rule.short': 'Corto cuando… <span style="color:var(--dim)">(opcional)</span>',
  'rule.short_ph': 'p. ej.: close < ema(close, 200)',
  'rule.help': 'Series y funciones disponibles',
  'assay.foot': '<b>Assay</b> es una herramienta de <b>validación</b>, no de recomendación de '
    + 'inversión. Que una estrategia sobreviva a la batería no demuestra que gane dinero: '
    + 'demuestra que no es ninguno de los cinco errores conocidos que hacen que un backtest '
    + 'bonito pierda dinero en real.',
  'run.step1': 'Calculando la señal sobre toda la serie…',
  'run.step2': 'Reejecutándola con un retraso de una barra (detector de fugas)…',
  'run.step3': 'Comparando contra la tasa base del mercado…',
  'run.step4': '<b>Generando 250 filtros aleatorios de la misma exposición…</b> '
    + '(esto es lo que tarda)',
  'run.step5': 'Partiendo el histórico con purga para medir fuera de muestra…',
  'err.title': 'No se pudo validar',
  'err.no_file': 'elige primero un fichero CSV',
  'v.survives': 'SOBREVIVE',
  'v.doubtful': 'DUDOSA',
  'v.fails': 'NO SOBREVIVE',
  'v.lookahead': 'MIRA AL FUTURO',
  'tag.pass': 'PASA',
  'tag.fail': 'FALLA',
  'tag.inconclusive': 'SIN DATOS',
  'kpi.cagr': 'CAGR',
  'kpi.sharpe': 'Sharpe',
  'kpi.dd': 'Peor caída',
  'kpi.exposure': 'Exposición',
  'kpi.episodes': 'Episodios',
  'kpi.bh': 'comprar y mantener {v}',
  'kpi.in_market': 'del tiempo dentro del mercado',
  'kpi.of_signals': 'de {n} señales',
  'report.heading': 'Esto es lo que he entendido de tus datos:',
  'meta.rationale': '<b>Por qué debería funcionar</b> (declarado antes de ver ningún resultado)',
  'meta.prior': 'Cuándo quedaría desmentida',
  'btn.copy': 'Copiar informe',
  'btn.copied': 'copiado ✓',
  'btn.download': 'Descargar informe (.md)',
  'chart.strategy': 'estrategia',
  'chart.bh': 'comprar y mantener',
  'md.strategy': 'estrategia',
  'md.bh': 'comprar y mantener',
  'md.dd': 'Peor caída',
  'md.exposure': 'Exposición',
  'md.episodes': 'Episodios independientes',
  'md.tests': 'Las cinco pruebas',
  'md.reading': 'Lectura de los datos',
  'md.rationale': 'Razonamiento declarado antes de medir',
  'md.prior': 'Cuándo quedaría desmentida:',
  'md.footer': 'Generado por **Assay** (DR Markets). Herramienta de validación, no de '
    + 'recomendación de inversión.\nQue una estrategia sobreviva no demuestra que gane dinero: '
    + 'demuestra que no es ninguno de los cinco errores conocidos que hacen que un backtest '
    + 'bonito pierda dinero en real.',
};

const CA = {
  'lang.aria': 'Idioma',
  'nav.chart': 'Gràfic',
  'nav.assay': 'Assay',

  'idx.analysing': 'analitzant…',
  'family.candles': 'Formes d\'espelma',
  'family.flow': 'Volum i flux',
  'family.mean_reversion': 'Reversió a la mitjana',
  'family.momentum': 'Moment',
  'family.seasonality': 'Calendari i estacionalitat',
  'family.structure': 'Estructura de mercat',
  'family.trend': 'Seguiment de tendència',
  'family.volatility': 'Volatilitat',
  'idx.public_server': 'Estàs veient el servidor públic, i aquest gràfic no hi és.\n\n'
    + 'El gràfic és la vista privada del propietari: /api/history, /api/decide i /ws responen 404 '
    + 'aquí expressament. És la garantia funcionant, no pas una errada.\n\n'
    + 'A macOS «localhost» es resol a ::1 abans que a 127.0.0.1, així que un túnel ssh obert al '
    + 'port 8000 envia el navegador a producció sense dir-t\'ho. Obre http://127.0.0.1:8000 o '
    + 'tanca el túnel.',
  'idx.chart_error': 'El gràfic no ha pogut carregar el seu històric ($1). Altrament el panell es '
    + 'quedaria aquí dient «analitzant…» per sempre, cosa que seria pitjor que dir-t\'ho.',
  'banner.public_server': 'Servidor públic — el gràfic és al teu local (127.0.0.1:8000)',
  'banner.chart_error': 'El gràfic no carrega. Mira el panell.',
  'idx.connecting': 'connectant…',
  'idx.foot': 'només visual · sense ordres · <kbd>1m</kbd> és l\'única sèrie desada, la resta es '
    + 'resamplea a partir d\'ella',
  'role.gate': 'veto',
  'role.regime': 'règim',
  'role.trigger': 'entrades',
  'role.source': 'font',
  'feed.down': 'feed caigut',
  'feed.mute': 'connectat però MUT {s}s',
  'feed.live': 'feed en directe',
  'mode.live': 'al dia',
  'mode.catch_up': 'posant-se al dia',
  'mode.warmup': 'escalfant',
  'health.status': 'estat: {mode}',
  'health.gaps': 'buits a la finestra: {n}',
  'health.lag': 'retard: {n} espelmes',
  'health.reconnects': 'reconnexions: {n} · espelmes curades: {m}',
  'banner.catch_up': 'Posant-se al dia després d\'un tall. L\'estat s\'està actualitzant però NO '
    + 's\'emet cap decisió: una zona d\'entrada calculada sobre espelmes antigues descriu un preu '
    + 'que ja ha passat.',
  'banner.warmup': 'Escalfant: carregant l\'històric. Encara no hi ha anàlisi.',
  'waves.line': 'pivots: {n} · ATR {atr} · {tail}',
  'waves.none': 'sense estructura',
  'waves.tentative': '{kind} temptatiu, confirma a ${price}',
  'kind.high': 'màxim',
  'kind.low': 'mínim',
  'confirm.below': 'confirma per sota',
  'confirm.above': 'confirma per sobre',
  'verdict.no_trade': 'NO OPERAR',
  'verdict.watch': 'VIGILAR',
  'verdict.actionable': 'ACCIONABLE',
  'plan.zone': 'zona',
  'hyp.long': 'llarg',
  'hyp.short': 'curt',
  'hyp.fit': 'ajust {score}',
  'hyp.truncated': 'truncada',
  'hyp.invalidation': 'invalidació',
  'hyp.entry_zone': 'zona d\'entrada',
  'hyp.stop': 'stop',
  'hyp.targets': 'objectius',
  'hyp.arith': 'R:R a T2 <b>{rr}</b> · stop a <b>{atr} ATR</b><br>'
    + 'cost <b>{cost}%</b> de R · mida <b>{size}%</b><br>'
    + 'aquest R:R exigeix encertar el <b>{p}%</b> de les vegades',
  'panel.maturity': 'nivell {n} · prior',
  'panel.ranked': 'hipòtesis ordenades',
  'panel.footer': 'Eina <b>visual</b>. No executa ordres.<br>'
    + 'El preu d\'invalidació és l\'única cosa que aquest marc produeix de manera objectiva: és '
    + 'on la teva idea és falsa.',

  'assay.title': 'Assay — la teva estratègia és real?',
  'assay.tagline': 'Un <i>assay</i> és l\'assaig que determina si un mineral és or o pirita. '
    + 'Això fa el mateix amb una estratègia de trading: cinc proves que gairebé cap no supera. '
    + 'No et diu quan comprar — et diu si allò que creus que funciona aguanta.',
  'tab.catalog': 'Catàleg',
  'tab.csv': 'El meu CSV',
  'tab.rule': 'Escriure una regla',
  'sel.loading': 'carregant…',
  'tf.1d': 'diari',
  'tf.4h': '4 hores',
  'tf.1h': '1 hora',
  'btn.battery': 'Sotmetre a la bateria',
  'btn.validate_csv': 'Validar la meva estratègia',
  'btn.validate_rule': 'Validar la meva regla',
  'catalog.count': '{n} estratègies registrades',
  'csv.hint': 'Puja un CSV amb <b>una columna de data</b> i <b>una de senyal</b>. Accepta dates '
    + 'ISO o epoch, i senyals com <code>1/0/-1</code>, <code>llarg/fora/curt</code> o '
    + '<code>buy/sell</code>. La posició es manté fins que la canviïs.<br>'
    + '<b>Si hi afegeixes una columna de preu</b>, es valida contra el <b>teu</b> actiu — '
    + 'accions, divises, la cripto que sigui. Sense ella s\'usa la nostra sèrie de BTC.',
  'csv.ex1': '<b>només senyals</b> (es valida contra BTC)',
  'csv.ex2': '<b>amb el teu propi actiu</b>',
  'csv.code1': 'data,senyal\n2024-01-15,llarg\n2024-03-02,fora\n2024-04-10,curt',
  'csv.code2': 'data,preu,senyal\n2024-01-15,185.50,llarg\n2024-01-16,187.20,llarg\n'
    + '2024-01-17,183.10,fora',
  'rule.hint': 'Escriu quan estar <b>llarg</b> i, si vols, quan estar <b>curt</b>. '
    + 'Si no es compleix cap de les dues, ets fora del mercat.',
  'rule.start': 'Comença per un exemple:',
  'ex.trend': 'tendència simple',
  'ex.rsi': 'RSI extrems',
  'ex.cross': 'creuament de mitjanes',
  'ex.breakout': 'ruptura de canal',
  'ex.range': 'expansió de rang',
  'rule.long': 'Llarg quan…',
  'rule.short': 'Curt quan… <span style="color:var(--dim)">(opcional)</span>',
  'rule.short_ph': 'p. ex.: close < ema(close, 200)',
  'rule.help': 'Sèries i funcions disponibles',
  'assay.foot': '<b>Assay</b> és una eina de <b>validació</b>, no de recomanació d\'inversió. '
    + 'Que una estratègia sobrevisqui a la bateria no demostra que guanyi diners: demostra que no '
    + 'és cap dels cinc errors coneguts que fan que un backtest bonic perdi diners en real.',
  'run.step1': 'Calculant el senyal sobre tota la sèrie…',
  'run.step2': 'Reexecutant-lo amb un retard d\'una barra (detector de fuites)…',
  'run.step3': 'Comparant contra la taxa base del mercat…',
  'run.step4': '<b>Generant 250 filtres aleatoris de la mateixa exposició…</b> '
    + '(això és el que triga)',
  'run.step5': 'Partint l\'històric amb purga per mesurar fora de mostra…',
  'err.title': 'No s\'ha pogut validar',
  'err.no_file': 'tria primer un fitxer CSV',
  'v.survives': 'SOBREVIU',
  'v.doubtful': 'DUBTOSA',
  'v.fails': 'NO SOBREVIU',
  'v.lookahead': 'MIRA AL FUTUR',
  'tag.pass': 'PASSA',
  'tag.fail': 'FALLA',
  'tag.inconclusive': 'SENSE DADES',
  'kpi.cagr': 'CAGR',
  'kpi.sharpe': 'Sharpe',
  'kpi.dd': 'Pitjor caiguda',
  'kpi.exposure': 'Exposició',
  'kpi.episodes': 'Episodis',
  'kpi.bh': 'comprar i mantenir {v}',
  'kpi.in_market': 'del temps dins del mercat',
  'kpi.of_signals': 'de {n} senyals',
  'report.heading': 'Això és el que he entès de les teves dades:',
  'meta.rationale': '<b>Per què hauria de funcionar</b> (declarat abans de veure cap resultat)',
  'meta.prior': 'Quan quedaria desmentida',
  'btn.copy': 'Copiar l\'informe',
  'btn.copied': 'copiat ✓',
  'btn.download': 'Descarregar l\'informe (.md)',
  'chart.strategy': 'estratègia',
  'chart.bh': 'comprar i mantenir',
  'md.strategy': 'estratègia',
  'md.bh': 'comprar i mantenir',
  'md.dd': 'Pitjor caiguda',
  'md.exposure': 'Exposició',
  'md.episodes': 'Episodis independents',
  'md.tests': 'Les cinc proves',
  'md.reading': 'Lectura de les dades',
  'md.rationale': 'Raonament declarat abans de mesurar',
  'md.prior': 'Quan quedaria desmentida:',
  'md.footer': 'Generat per **Assay** (DR Markets). Eina de validació, no de recomanació '
    + 'd\'inversió.\nQue una estratègia sobrevisqui no demostra que guanyi diners: demostra que '
    + 'no és cap dels cinc errors coneguts que fan que un backtest bonic perdi diners en real.',
};

const DICTS = { en: EN, es: ES, ca: CA };

// ------------------------------------------------------- text that arrives from the API in English
//
// The API contract is English and stays English: translating it server-side would mean a `lang`
// parameter on every endpoint and a second copy of the product's most carefully written prose
// living inside Python. These are the strings the user actually reads — the five test titles,
// their explanations, the four verdicts — matched exactly against what the server sends.
// A miss falls through in English rather than breaking anything.

const SERVER_ES = {
  // battery verdict summaries
  'IT READS THE FUTURE. The result is not reachable in real time.':
    'MIRA AL FUTURO. El resultado no es alcanzable en tiempo real.',
  'DOES NOT SURVIVE. It fails several independent tests.':
    'NO SOBREVIVE. Falla varias pruebas independientes.',
  'DOUBTFUL. Passes most of them but fails one that matters.':
    'DUDOSA. Pasa la mayoría pero falla una que importa.',
  ['SURVIVES the battery. That is not proof it makes money: it is proof it is not one of the '
  + 'known mistakes.']:
    'SOBREVIVE la batería. Eso no demuestra que gane dinero: demuestra que no es ninguno de los '
    + 'errores conocidos.',

  // the five test titles
  'Does it survive being executed one bar later?':
    '¿Sobrevive si se ejecuta una barra más tarde?',
  'Is it right more often than any old moment?':
    '¿Acierta más veces que un momento cualquiera?',
  'Does it beat RANDOM filters with the same exposure?':
    '¿Gana a filtros ALEATORIOS con la misma exposición?',
  'Does it hold up outside the sample it was found in?':
    '¿Aguanta fuera de la muestra en la que se encontró?',
  'Does it hold up out of sample?': '¿Aguanta fuera de muestra?',
  'Are there enough INDEPENDENT observations?':
    '¿Hay suficientes observaciones INDEPENDIENTES?',

  // and their explanations, which are the argument the product is making
  ['If the result collapses when the signal is delayed by a single bar, the strategy is using '
  + 'information you would not have had in real time. It is the cheapest test there is and the '
  + 'one that knocks over the most strategies.']:
    'Si el resultado se desploma al retrasar la señal una sola barra, la estrategia está usando '
    + 'información que no habrías tenido en tiempo real. Es la prueba más barata que existe y la '
    + 'que tumba más estrategias.',
  ['Hundreds of filters are generated that enter and exit at random with the same frequency and '
  + 'the same time spent inside the market. If your strategy is not clearly above them, what you '
  + 'have is exposure, not judgement.']:
    'Se generan cientos de filtros que entran y salen al azar con la misma frecuencia y el mismo '
    + 'tiempo dentro del mercado. Si tu estrategia no está claramente por encima de ellos, lo que '
    + 'tienes es exposición, no criterio.',
  ['The history is cut into segments and the horizon is purged at every boundary, so that no open '
  + 'trade crosses from one segment into the next. The edge has to survive where nobody was '
  + 'looking.']:
    'El histórico se corta en segmentos y se purga el horizonte en cada frontera, para que ninguna '
    + 'operación abierta cruce de un segmento al siguiente. La ventaja tiene que sobrevivir donde '
    + 'nadie estaba mirando.',
  'Series too short to cut up.': 'Serie demasiado corta para trocearla.',
  ['Signals that overlap inside the holding horizon are the SAME observation. An n=500 that is '
  + 'really 12 episodes produces p-values that are wrong by orders of magnitude. It is López de '
  + "Prado's label overlap problem."]:
    'Las señales que se solapan dentro del horizonte de mantenimiento son la MISMA observación. Un '
    + 'n=500 que en realidad son 12 episodios produce p-valores equivocados por órdenes de '
    + 'magnitud. Es el problema de solapamiento de etiquetas de López de Prado.',

  // fixed lines from the CSV importer and the endpoints
  'far too few signals': 'demasiado pocas señales',
  'signal read from text (long/short/flat)': 'señal leída del texto (largo/corto/fuera)',
  'time read as an epoch in NANOseconds': 'tiempo leído como epoch en NANOsegundos',
  'time read as an epoch in MICROseconds': 'tiempo leído como epoch en MICROsegundos',
  'time read as an epoch in milliseconds': 'tiempo leído como epoch en milisegundos',
  'time read as an epoch in SECONDS': 'tiempo leído como epoch en SEGUNDOS',
  ['your CSV has no price column: it is validated against OUR BTCUSDT series. If you trade a '
  + 'different asset, add a `price` column.']:
    'tu CSV no trae columna de precio: se valida contra NUESTRA serie de BTCUSDT. Si operas otro '
    + 'activo, añade una columna `precio`.',
  'the file has no rows': 'el fichero no tiene ninguna fila',
  'empty file': 'fichero vacío',
  'the file is larger than 12 MB': 'el fichero pesa más de 12 MB',

  // decision card
  'not enough structure yet': 'todavía no hay suficiente estructura',
  'no structure satisfies the hard rules right now':
    'ninguna estructura cumple las reglas duras ahora mismo',
  'there is structure, but the price is not inside any entry zone':
    'hay estructura, pero el precio no está dentro de ninguna zona de entrada',
  ['PRIOR level (n=0): expectancies from an expert table, unvalidated. The verdict cannot go '
  + 'above WATCH.']:
    'Nivel PRIOR (n=0): esperanzas de una tabla experta, sin validar. El veredicto no puede pasar '
    + 'de VIGILAR.',
  'the entry zone is on the far side of the stop.':
    'la zona de entrada está al otro lado del stop.',
  // The w4 zone is truncated so it can never invade wave 1's territory, and the truncation can
  // consume it entirely. Rare, but it is the ONLY rejection that carries no number, so it cannot
  // be reached by any pattern: without this line it is the one refusal a Spanish reader gets in
  // English.
  ['the entry zone is empty once truncated against the invalidation: there is nowhere left to '
  + 'enter above the stop.']:
    'la zona de entrada queda vacía al truncarla contra la invalidación: no queda sitio donde '
    + 'entrar por encima del stop.',
  // The chart label for the hypothesis, from `Hypothesis.terminal_label` in waves/matcher.py.
  // It is the single most prominent server string on the decision page — the heading of every
  // hypothesis card — and it rendered in English in every language until now, because app.js
  // interpolated `h.label` raw instead of through tx(). It also leaks INTO an otherwise
  // translated sentence: the "state «…» offers no entry" refusal captures this label in $1, so a
  // Spanish reader was getting "el estado «w2 complete → inside w3» no ofrece entrada".
  'w2 complete → inside w3': 'onda 2 completa → dentro de la onda 3',
  'w3 complete → inside w4': 'onda 3 completa → dentro de la onda 4',
  'w4 complete → inside w5': 'onda 4 completa → dentro de la onda 5',
  'five complete → ABC expected': 'cinco completas → se espera ABC',
  'not available': 'no disponible',

  // the rule editor's help: the KEYS are the DSL and never change, only these descriptions
  'closing price': 'precio de cierre',
  'opening price': 'precio de apertura',
  'high of the bar': 'máximo de la barra',
  'low of the bar': 'mínimo de la barra',
  volume: 'volumen',
  '14-bar ATR': 'ATR de 14 barras',
  'high − low of the bar': 'máximo − mínimo de la barra',
  'close − open': 'cierre − apertura',
  'simple moving average over n bars': 'media móvil simple de n barras',
  'exponential moving average over n bars': 'media móvil exponencial de n barras',
  'RSI over n bars (14 by default)': 'RSI de n barras (14 por defecto)',
  'standard deviation over n bars': 'desviación típica de n barras',
  'highest value of the last n bars': 'valor más alto de las últimas n barras',
  'lowest value of the last n bars': 'valor más bajo de las últimas n barras',
  'percent change against n bars ago': 'cambio porcentual respecto a hace n barras',
  'the value n bars ago (negative n FORBIDDEN)': 'el valor de hace n barras (n negativa PROHIBIDA)',
  '1 on the bar where a crosses above b': '1 en la barra en la que a cruza por encima de b',
  '1 on the bar where a crosses below b': '1 en la barra en la que a cruza por debajo de b',
  'absolute value': 'valor absoluto',

  // What the battery calls the thing it just tested. A catalogue hypothesis arrives here as its
  // identifier (`candles.body_flow_20`) and must stay that way; these three are the only names
  // the server writes as prose, and they head the Markdown report the reader downloads.
  'your strategy (CSV)': 'tu estrategia (CSV)',
  'your strategy (CSV with prices)': 'tu estrategia (CSV con precios)',
  'your rule': 'tu regla',

  // The rule editor's refusals that carry NO computed number, so they are exact matches and not
  // patterns. All four fell through in English, and the group LOOKED covered because the one
  // refusal beside them — `expression not allowed` — has a pattern: an entry that is present for
  // its noisiest neighbour is not evidence for the quiet ones.
  'write comparisons one at a time: `a > b and b > c`, not `a > b > c`':
    'escribe las comparaciones de una en una: `a > b and b > c`, no `a > b > c`',
  'functions can only be called by name': 'las funciones solo se pueden llamar por su nombre',
  'functions do not take keyword arguments': 'las funciones no aceptan argumentos con nombre',
  ['the rule has to produce a series the same size as the prices (did you write a constant '
  + 'instead of a comparison?)']:
    'la regla tiene que producir una serie del mismo tamaño que los precios (¿has escrito una '
    + 'constante en lugar de una comparación?)',

  // the two fixed refusals: the concurrency limiter, and the look-ahead guard in shift()
  ['there are too many validations under way right now. Try again in a minute: each one takes a '
  + 'few seconds and only two run at a time so that nobody\'s result gets degraded.']:
    'ahora mismo hay demasiadas validaciones en marcha. Inténtalo dentro de un minuto: cada una '
    + 'tarda unos segundos y solo se ejecutan dos a la vez para que no se degrade el resultado de '
    + 'nadie.',
  ['shift() does not accept negative values. A negative shift brings in data from the FUTURE, and '
  + 'that is the bug that turns a beautiful backtest into money lost live. If you really do want '
  + 'to look forward, this tool is not for you.']:
    'shift() no acepta valores negativos. Un desplazamiento negativo trae datos del FUTURO, y ese '
    + 'es el fallo que convierte un backtest precioso en dinero perdido en real. Si de verdad '
    + 'quieres mirar hacia delante, esta herramienta no es para ti.',
};

const SERVER_CA = {
  'IT READS THE FUTURE. The result is not reachable in real time.':
    'MIRA AL FUTUR. El resultat no és assolible en temps real.',
  'DOES NOT SURVIVE. It fails several independent tests.':
    'NO SOBREVIU. Falla diverses proves independents.',
  'DOUBTFUL. Passes most of them but fails one that matters.':
    'DUBTOSA. Passa la majoria però en falla una que importa.',
  ['SURVIVES the battery. That is not proof it makes money: it is proof it is not one of the '
  + 'known mistakes.']:
    'SOBREVIU la bateria. Això no demostra que guanyi diners: demostra que no és cap dels errors '
    + 'coneguts.',

  'Does it survive being executed one bar later?':
    'Sobreviu si s\'executa una barra més tard?',
  'Is it right more often than any old moment?':
    'Encerta més vegades que un moment qualsevol?',
  'Does it beat RANDOM filters with the same exposure?':
    'Guanya filtres ALEATORIS amb la mateixa exposició?',
  'Does it hold up outside the sample it was found in?':
    'Aguanta fora de la mostra on es va trobar?',
  'Does it hold up out of sample?': 'Aguanta fora de mostra?',
  'Are there enough INDEPENDENT observations?':
    'Hi ha prou observacions INDEPENDENTS?',

  ['If the result collapses when the signal is delayed by a single bar, the strategy is using '
  + 'information you would not have had in real time. It is the cheapest test there is and the '
  + 'one that knocks over the most strategies.']:
    'Si el resultat s\'esfondra en retardar el senyal una sola barra, l\'estratègia està fent '
    + 'servir informació que no hauries tingut en temps real. És la prova més barata que hi ha i '
    + 'la que en tomba més.',
  ['Hundreds of filters are generated that enter and exit at random with the same frequency and '
  + 'the same time spent inside the market. If your strategy is not clearly above them, what you '
  + 'have is exposure, not judgement.']:
    'Es generen centenars de filtres que entren i surten a l\'atzar amb la mateixa freqüència i el '
    + 'mateix temps dins del mercat. Si la teva estratègia no està clarament per damunt d\'ells, '
    + 'el que tens és exposició, no criteri.',
  ['The history is cut into segments and the horizon is purged at every boundary, so that no open '
  + 'trade crosses from one segment into the next. The edge has to survive where nobody was '
  + 'looking.']:
    'L\'històric es talla en segments i es purga l\'horitzó a cada frontera, perquè cap operació '
    + 'oberta no creuï d\'un segment al següent. L\'avantatge ha de sobreviure allà on ningú no '
    + 'mirava.',
  'Series too short to cut up.': 'Sèrie massa curta per trossejar-la.',
  ['Signals that overlap inside the holding horizon are the SAME observation. An n=500 that is '
  + 'really 12 episodes produces p-values that are wrong by orders of magnitude. It is López de '
  + "Prado's label overlap problem."]:
    'Els senyals que se solapen dins de l\'horitzó de manteniment són la MATEIXA observació. Una '
    + 'n=500 que en realitat són 12 episodis produeix p-valors equivocats per ordres de magnitud. '
    + 'És el problema de solapament d\'etiquetes de López de Prado.',

  'far too few signals': 'massa pocs senyals',
  'signal read from text (long/short/flat)': 'senyal llegit del text (llarg/curt/fora)',
  'time read as an epoch in NANOseconds': 'temps llegit com a epoch en NANOsegons',
  'time read as an epoch in MICROseconds': 'temps llegit com a epoch en MICROsegons',
  'time read as an epoch in milliseconds': 'temps llegit com a epoch en mil·lisegons',
  'time read as an epoch in SECONDS': 'temps llegit com a epoch en SEGONS',
  ['your CSV has no price column: it is validated against OUR BTCUSDT series. If you trade a '
  + 'different asset, add a `price` column.']:
    'el teu CSV no porta columna de preu: es valida contra LA NOSTRA sèrie de BTCUSDT. Si operes '
    + 'un altre actiu, afegeix-hi una columna `preu`.',
  'the file has no rows': 'el fitxer no té cap fila',
  'empty file': 'fitxer buit',
  'the file is larger than 12 MB': 'el fitxer pesa més de 12 MB',

  'not enough structure yet': 'encara no hi ha prou estructura',
  'no structure satisfies the hard rules right now':
    'cap estructura no compleix les regles dures ara mateix',
  'there is structure, but the price is not inside any entry zone':
    'hi ha estructura, però el preu no és dins de cap zona d\'entrada',
  ['PRIOR level (n=0): expectancies from an expert table, unvalidated. The verdict cannot go '
  + 'above WATCH.']:
    'Nivell PRIOR (n=0): esperances d\'una taula experta, sense validar. El veredicte no pot '
    + 'passar de VIGILAR.',
  'the entry zone is on the far side of the stop.':
    'la zona d\'entrada és a l\'altra banda del stop.',
  ['the entry zone is empty once truncated against the invalidation: there is nowhere left to '
  + 'enter above the stop.']:
    'la zona d\'entrada queda buida en truncar-la contra la invalidació: no hi queda cap lloc on '
    + 'entrar per damunt del stop.',
  // The chart label for the hypothesis, from `Hypothesis.terminal_label` in waves/matcher.py.
  // It is the single most prominent server string on the decision page — the heading of every
  // hypothesis card — and it rendered in English in every language until now, because app.js
  // interpolated `h.label` raw instead of through tx(). It also leaks INTO an otherwise
  // translated sentence: the "state «…» offers no entry" refusal captures this label in $1, so a
  // Spanish reader was getting "el estado «w2 complete → inside w3» no ofrece entrada".
  'w2 complete → inside w3': 'ona 2 completa → dins de l\'ona 3',
  'w3 complete → inside w4': 'ona 3 completa → dins de l\'ona 4',
  'w4 complete → inside w5': 'ona 4 completa → dins de l\'ona 5',
  'five complete → ABC expected': 'cinc completes → s\'espera ABC',
  'not available': 'no disponible',

  'closing price': 'preu de tancament',
  'opening price': 'preu d\'obertura',
  'high of the bar': 'màxim de la barra',
  'low of the bar': 'mínim de la barra',
  volume: 'volum',
  '14-bar ATR': 'ATR de 14 barres',
  'high − low of the bar': 'màxim − mínim de la barra',
  'close − open': 'tancament − obertura',
  'simple moving average over n bars': 'mitjana mòbil simple de n barres',
  'exponential moving average over n bars': 'mitjana mòbil exponencial de n barres',
  'RSI over n bars (14 by default)': 'RSI de n barres (14 per defecte)',
  'standard deviation over n bars': 'desviació típica de n barres',
  'highest value of the last n bars': 'valor més alt de les últimes n barres',
  'lowest value of the last n bars': 'valor més baix de les últimes n barres',
  'percent change against n bars ago': 'canvi percentual respecte a fa n barres',
  'the value n bars ago (negative n FORBIDDEN)': 'el valor de fa n barres (n negativa PROHIBIDA)',
  '1 on the bar where a crosses above b': '1 a la barra on a creua per damunt de b',
  '1 on the bar where a crosses below b': '1 a la barra on a creua per sota de b',
  'absolute value': 'valor absolut',

  // What the battery calls the thing it just tested. A catalogue hypothesis arrives here as its
  // identifier (`candles.body_flow_20`) and must stay that way; these three are the only names
  // the server writes as prose, and they head the Markdown report the reader downloads.
  'your strategy (CSV)': 'la teva estratègia (CSV)',
  'your strategy (CSV with prices)': 'la teva estratègia (CSV amb preus)',
  'your rule': 'la teva regla',

  'write comparisons one at a time: `a > b and b > c`, not `a > b > c`':
    'escriu les comparacions d\'una en una: `a > b and b > c`, no pas `a > b > c`',
  'functions can only be called by name': 'les funcions només es poden cridar pel seu nom',
  'functions do not take keyword arguments': 'les funcions no accepten arguments amb nom',
  ['the rule has to produce a series the same size as the prices (did you write a constant '
  + 'instead of a comparison?)']:
    'la regla ha de produir una sèrie de la mateixa mida que els preus (has escrit una constant '
    + 'en comptes d\'una comparació?)',

  // the two fixed refusals: the concurrency limiter, and the look-ahead guard in shift()
  ['there are too many validations under way right now. Try again in a minute: each one takes a '
  + 'few seconds and only two run at a time so that nobody\'s result gets degraded.']:
    'ara mateix hi ha massa validacions en marxa. Torna-ho a provar d\'aquí a un minut: cadascuna '
    + 'triga uns segons i només se n\'executen dues alhora perquè no es degradi el resultat de '
    + 'ningú.',
  ['shift() does not accept negative values. A negative shift brings in data from the FUTURE, and '
  + 'that is the bug that turns a beautiful backtest into money lost live. If you really do want '
  + 'to look forward, this tool is not for you.']:
    'shift() no accepta valors negatius. Un desplaçament negatiu porta dades del FUTUR, i aquest '
    + 'és l\'error que converteix un backtest preciós en diners perduts en real. Si de debò vols '
    + 'mirar cap endavant, aquesta eina no és per a tu.',
};

const SERVER = { es: SERVER_ES, ca: SERVER_CA };

//: The server's dynamic lines: a fixed shape with the numbers IT computed dropped into it. The
//: captured groups are copied over verbatim — every figure, threshold and date the user sees is
//: still exactly the one the battery produced, only the words around them change.
const PATTERNS = [
  [/^CAGR (.+?) → (.+?) when delayed \(keeps (.+?) of the excess over buy and hold\)$/, {
    es: 'CAGR $1 → $2 al retrasarla (conserva $3 del exceso sobre comprar y mantener)',
    ca: 'CAGR $1 → $2 en retardar-la (conserva $3 de l\'excés sobre comprar i mantenir)',
  }],
  [/^strategy (.+?) vs base (.+?) per period$/, {
    es: 'estrategia $1 vs base $2 por periodo',
    ca: 'estratègia $1 vs base $2 per període',
  }],
  [/^percentile (.+?) against (.+?) random filters \(median (.+?) CAGR\)$/, {
    es: 'percentil $1 frente a $2 filtros aleatorios (mediana $3 CAGR)',
    ca: 'percentil $1 davant de $2 filtres aleatoris (mediana $3 CAGR)',
  }],
  [/^in-sample excess (.+?) → out of sample (.+?) \(keeps (.+?)\)$/, {
    es: 'exceso dentro de muestra $1 → fuera de muestra $2 (conserva $3)',
    ca: 'excés dins de mostra $1 → fora de mostra $2 (conserva $3)',
  }],
  [new RegExp('^not applicable: the excess is already negative in sample \\((.+?)\\)\\. '
    + 'There is no edge that could survive outside it\\.$'), {
    es: 'no aplicable: el exceso ya es negativo dentro de muestra ($1). No hay ninguna ventaja que '
      + 'pueda sobrevivir fuera de ella.',
    ca: 'no aplicable: l\'excés ja és negatiu dins de mostra ($1). No hi ha cap avantatge que pugui '
      + 'sobreviure a fora.',
  }],
  [new RegExp('^(.+?) bars holding a position across (.+?) runs \\(mean duration (.+?) bars\\) '
    + '→ (.+?) independent episodes$'), {
    es: '$1 barras con posición en $2 tramos (duración media $3 barras) → $4 episodios '
      + 'independientes',
    ca: '$1 barres amb posició en $2 trams (durada mitjana $3 barres) → $4 episodis independents',
  }],

  // CSV importer report
  [/^(.+?) rows and (.+?) columns: (.*)$/, {
    es: '$1 filas y $2 columnas: $3',
    ca: '$1 files i $2 columnes: $3',
  }],
  [/^time column: «(.+?)» · signal column: «(.+?)»$/, {
    es: 'columna de tiempo: «$1» · columna de señal: «$2»',
    ca: 'columna de temps: «$1» · columna de senyal: «$2»',
  }],
  [/^dates read as text \(e\.g\. «(.*)»\)$/, {
    es: 'fechas leídas como texto (p. ej. «$1»)',
    ca: 'dates llegides com a text (p. ex. «$1»)',
  }],
  [/^the signal column has (.+?) distinct values \(max (.+?)\): only the SIGN of the position is used$/, {
    es: 'la columna de señal tiene $1 valores distintos (máx $2): solo se usa el SIGNO de la '
      + 'posición',
    ca: 'la columna de senyal té $1 valors diferents (màx $2): només es fa servir el SIGNE de la '
      + 'posició',
  }],
  [/^numeric signal with values (.*)$/, {
    es: 'señal numérica con valores $1',
    ca: 'senyal numèric amb valors $1',
  }],
  [/^(.+?) duplicate timestamps: the last one is kept$/, {
    es: '$1 marcas de tiempo duplicadas: se conserva la última',
    ca: '$1 marques de temps duplicades: es conserva l\'última',
  }],
  [/^price column: «(.+?)» — YOUR price series will be used, not ours \((.+?) to (.+?)\)$/, {
    es: 'columna de precio: «$1» — se usará TU serie de precios, no la nuestra ($2 a $3)',
    ca: 'columna de preu: «$1» — s\'usarà LA TEVA sèrie de preus, no la nostra ($2 a $3)',
  }],
  [/^column «(.+?)» discarded as a price: it has invalid or non-positive values$/, {
    es: 'columna «$1» descartada como precio: tiene valores inválidos o no positivos',
    ca: 'columna «$1» descartada com a preu: té valors no vàlids o no positius',
  }],

  // errors the importer raises. These are the FIRST thing a user with a messy spreadsheet sees,
  // so leaving them in English would greet a Spanish speaker with English at the exact moment
  // something went wrong.
  // The figure is measured from the series in hand, not fixed: it was 1,748% over 2017-2026 and it
  // moves with every bar that arrives, and it is different again for a CSV of the user's own
  // prices. $1 goes through localiseNumber, so `1,748` reads `1.748` here.
  [new RegExp('^In an asset that rose (.+?)%, any mostly-long rule looks right\\. What counts is '
    + 'whether it is right MORE often than being in the market at a random instant\\.$'), {
    es: 'En un activo que ha subido un $1%, cualquier regla casi siempre larga parece acertar. Lo '
      + 'que cuenta es si acierta MÁS veces que estar en el mercado en un instante al azar.',
    ca: 'En un actiu que ha pujat un $1%, qualsevol regla gairebé sempre llarga sembla encertar. '
      + 'El que compta és si encerta MÉS vegades que ser al mercat en un instant a l\'atzar.',
  }],
  [new RegExp("^I can't find the time column\\. Columns: (.+?)\\. "
    + "It should be called something like: (.+?)$"), {
    es: 'No encuentro la columna de tiempo. Columnas: $1. Debería llamarse algo como: $2',
    ca: 'No trobo la columna de temps. Columnes: $1. Hauria d\'anomenar-se una cosa com: $2',
  }],
  [new RegExp("^I can't find the signal column\\. Columns: (.+?)\\. "
    + "It should be called something like: (.+?)$"), {
    es: 'No encuentro la columna de señal. Columnas: $1. Debería llamarse algo como: $2',
    ca: 'No trobo la columna de senyal. Columnes: $1. Hauria d\'anomenar-se una cosa com: $2',
  }],
  [new RegExp("^I don't understand these signal values: (.+?)\\. "
    + "Use numbers \\(-1/0/1\\) or words: (.+?) / (.+?) / (.+?)$"), {
    es: 'No entiendo estos valores de señal: $1. Usa números (-1/0/1) o palabras: $2 / $3 / $4',
    ca: 'No entenc aquests valors de senyal: $1. Fes servir nombres (-1/0/1) o paraules: '
      + '$2 / $3 / $4',
  }],
  [new RegExp('^the time column holds values around (.+?), which look like neither a timestamp '
    + 'nor a date\\. Is this the right column\\?$'), {
    es: 'la columna de tiempo tiene valores en torno a $1, que no parecen ni una marca de tiempo '
      + 'ni una fecha. ¿Es esta la columna correcta?',
    ca: 'la columna de temps té valors al voltant de $1, que no semblen ni una marca de temps ni '
      + 'una data. És aquesta la columna correcta?',
  }],
  [new RegExp('^the dates could not be interpreted\\. The first one reads «(.+?)»\\. Use a shape '
    + 'like 2024-03-14, 2024-03-14 18:00, or an epoch in seconds or milliseconds\\.$'), {
    es: 'no se han podido interpretar las fechas. La primera dice «$1». Usa un formato como '
      + '2024-03-14, 2024-03-14 18:00, o una marca de tiempo en segundos o milisegundos.',
    ca: 'no s\'han pogut interpretar les dates. La primera diu «$1». Fes servir un format com '
      + '2024-03-14, 2024-03-14 18:00, o una marca de temps en segons o mil·lisegons.',
  }],
  [/^could not read the CSV: (.+?)$/, {
    es: 'no se ha podido leer el CSV: $1',
    ca: 'no s\'ha pogut llegir el CSV: $1',
  }],

  // lines the endpoints add to the report
  [/^(.+?) long, (.+?) short, (.+?) flat in your file$/, {
    es: '$1 largo, $2 corto, $3 fuera en tu fichero',
    ca: '$1 llarg, $2 curt, $3 fora al teu fitxer',
  }],
  [/^(.+?) long, (.+?) short, (.+?) flat$/, {
    es: '$1 largo, $2 corto, $3 fuera',
    ca: '$1 llarg, $2 curt, $3 fora',
  }],
  [/^validated on YOUR series of (.+?) rows, between (.+?) and (.+?)$/, {
    es: 'validado sobre TU serie de $1 filas, entre $2 y $3',
    ca: 'validat sobre LA TEVA sèrie de $1 files, entre $2 i $3',
  }],
  [/^aligned to (.+?) (\S+?) bars between (.+?) and (.+?)$/, {
    es: 'alineado a $1 barras de $2 entre $3 y $4',
    ca: 'alineat a $1 barres de $2 entre $3 i $4',
  }],
  [/^(.+?) long bars, (.+?) short out of (.+?)$/, {
    es: '$1 barras en largo, $2 en corto de $3',
    ca: '$1 barres en llarg, $2 en curt de $3',
  }],

  // errors the user is most likely to meet
  [/^not enough data on (.+?)$/, {
    es: 'no hay suficientes datos en $1',
    ca: 'no hi ha prou dades en $1',
  }],
  [/^unknown hypothesis: (.+?)$/, {
    es: 'hipótesis desconocida: $1',
    ca: 'hipòtesi desconeguda: $1',
  }],
  [/^the service is loading nine years of history \((.+?) months\)\. Try again in a minute\.$/, {
    es: 'el servicio está cargando nueve años de histórico ($1 meses). Inténtalo dentro de un '
      + 'minuto.',
    ca: 'el servei està carregant nou anys d\'històric ($1 mesos). Torna-ho a provar d\'aquí a un '
      + 'minut.',
  }],
  [/^at most (.+?) validations per minute\. Wait a few seconds\.$/, {
    es: 'como máximo $1 validaciones por minuto. Espera unos segundos.',
    ca: 'com a màxim $1 validacions per minut. Espera uns segons.',
  }],
  [/^no data for (.+?)$/, {
    es: 'no hay datos para $1',
    ca: 'no hi ha dades per a $1',
  }],
  [/^timeframe (.+?) is not configured$/, {
    es: 'el timeframe $1 no está configurado',
    ca: 'el timeframe $1 no està configurat',
  }],
  [new RegExp('^after aligning your CSV with our (.+?) bars only (.+?) bars are left holding a '
    + 'position\\. Check that the dates fall inside 2017-2026 and that the timeframe you picked is '
    + 'the one you meant\\.$'), {
    es: 'tras alinear tu CSV con nuestras barras de $1 solo quedan $2 barras con posición. '
      + 'Comprueba que las fechas caen dentro de 2017-2026 y que el timeframe elegido es el que '
      + 'querías.',
    ca: 'després d\'alinear el teu CSV amb les nostres barres de $1 només queden $2 barres amb '
      + 'posició. Comprova que les dates cauen dins de 2017-2026 i que el timeframe triat és el '
      + 'que volies.',
  }],
  [new RegExp('^with your own prices we need at least 120 rows and there are (.+?)\\. With fewer, '
    + 'none of the five tests can conclude anything\\.$'), {
    es: 'con tus propios precios hacen falta al menos 120 filas y hay $1. Con menos, ninguna de '
      + 'las cinco pruebas puede concluir nada.',
    ca: 'amb els teus propis preus calen com a mínim 120 files i n\'hi ha $1. Amb menys, cap de '
      + 'les cinc proves no pot concloure res.',
  }],
  [new RegExp('^the rule only holds on (.+?) bars out of (.+?)\\. With that few, nothing can be '
    + 'concluded\\.$'), {
    es: 'la regla solo se cumple en $1 barras de $2. Con tan pocas no se puede concluir nada.',
    ca: 'la regla només es compleix en $1 barres de $2. Amb tan poques no es pot concloure res.',
  }],
  [new RegExp('^you have run (.+?) validations in the last hour, which is the limit\\. Come back '
    + 'in (.+?) min\\. Every validation runs 250 random controls and that costs real CPU\\.$'), {
    es: 'has hecho $1 validaciones en la última hora, que es el límite. Vuelve dentro de $2 min. '
      + 'Cada validación ejecuta 250 controles aleatorios y eso cuesta CPU de verdad.',
    ca: 'has fet $1 validacions en l\'última hora, que és el límit. Torna d\'aquí a $2 min. Cada '
      + 'validació executa 250 controls aleatoris i això costa CPU de debò.',
  }],

  // The rule editor's refusals. Every one of these is a sentence the user reads WHILE their rule
  // is broken, which is the worst moment to be handed a language they do not read. The captured
  // groups are Python type names, identifiers and the function/series lists: those are the
  // vocabulary of the expression language itself and are copied over untouched on purpose —
  // translating `close` or `BitXor` would name something the editor does not accept.
  // This one tracks `_Interpreter.visit()` in expr.py word for word, and it has already drifted
  // once: expr.py grew "`and`/`or`/`not` and the functions on the list are accepted" while the
  // regex here still said "and the listed functions". Nothing broke and nothing was reported —
  // the anchored regex simply stopped matching, so the commonest error in the editor rendered in
  // English on a Spanish page while this entry sat here looking present.
  [new RegExp('^expression not allowed: (.+?)\\. Only comparisons, arithmetic, `and`/`or`/`not` '
    + 'and the functions on the list are accepted\\. No imports, no attributes, no indexing, no '
    + 'lambdas\\.$'), {
    es: 'expresión no permitida: $1. Solo se aceptan comparaciones, aritmética, `and`/`or`/`not` '
      + 'y las funciones de la lista. Sin imports, sin atributos, sin indexación, sin lambdas.',
    ca: 'expressió no permesa: $1. Només s\'accepten comparacions, aritmètica, `and`/`or`/`not` i '
      + 'les funcions de la llista. Sense imports, sense atributs, sense indexació, sense lambdes.',
  }],
  [/^only numbers are accepted, not (.+?)$/, {
    es: 'solo se aceptan números, no $1',
    ca: 'només s\'accepten nombres, no pas $1',
  }],
  [/^unknown name: (.+?)\. Series: (.+?)\. Functions: (.+?)$/, {
    es: 'nombre desconocido: $1. Series: $2. Funciones: $3',
    ca: 'nom desconegut: $1. Sèries: $2. Funcions: $3',
  }],
  [/^operator not allowed: (.+?)$/, {
    es: 'operador no permitido: $1',
    ca: 'operador no permès: $1',
  }],
  [/^unary operator not allowed: (.+?)$/, {
    es: 'operador unario no permitido: $1',
    ca: 'operador unari no permès: $1',
  }],
  [/^comparison not allowed: (.+?)$/, {
    es: 'comparación no permitida: $1',
    ca: 'comparació no permesa: $1',
  }],
  [/^unknown function: (.+?)\(\)\. Available: (.+?)$/, {
    es: 'función desconocida: $1(). Disponibles: $2',
    ca: 'funció desconeguda: $1(). Disponibles: $2',
  }],
  [/^syntax error: (.+?)$/, {
    es: 'error de sintaxis: $1',
    ca: 'error de sintaxi: $1',
  }],
  [/^error while evaluating: (.+?): (.+)$/, {
    es: 'error al evaluar: $1: $2',
    ca: 'error en avaluar: $1: $2',
  }],

  // the plan's rejection arithmetic, which is half of what the decision card is for
  [/^stop at (.+?) ATR \(>(.+?)\): size cut to (.+?), the stop is NOT tightened$/, {
    es: 'stop a $1 ATR (>$2): tamaño recortado al $3, el stop NO se aprieta',
    ca: 'stop a $1 ATR (>$2): mida retallada al $3, el stop NO s\'estreny',
  }],
  [new RegExp('^stop at (.+?) ATR: it sits inside the noise floor \\(<(.+?) ATR\\) and any wick '
    + 'would sweep it\\.$'), {
    es: 'stop a $1 ATR: queda dentro del suelo de ruido (<$2 ATR) y cualquier mecha lo barrería.',
    ca: 'stop a $1 ATR: queda dins del sòl de soroll (<$2 ATR) i qualsevol metxa l\'escombraria.',
  }],
  [new RegExp('^the round trip in fees takes (.+?) of R \\(maximum (.+?)\\): the stop is too '
    + 'close for any plausible edge to survive the fees\\.$'), {
    es: 'la ida y vuelta en comisiones se lleva el $1 de R (máximo $2): el stop está demasiado '
      + 'cerca para que ninguna ventaja plausible sobreviva a las comisiones.',
    ca: 'l\'anada i tornada en comissions s\'emporta el $1 de R (màxim $2): el stop és massa a '
      + 'prop perquè cap avantatge plausible sobrevisqui a les comissions.',
  }],
  [new RegExp("^state '(.+?)' offers no entry; inside wave 3 you manage, you do not enter\\.$"), {
    es: 'el estado «$1» no ofrece entrada; dentro de la onda 3 se gestiona, no se entra.',
    ca: 'l\'estat «$1» no ofereix entrada; dins de l\'ona 3 es gestiona, no s\'entra.',
  }],
  // The dollar signs live INSIDE the captured groups: `$` followed by a digit is the replacement
  // syntax, and a template written as `$$1` is a bug waiting for the first user with a price.
  [/^price (\$.+?) outside the zone (\$.+?)-(\$.+?)$/, {
    es: 'precio $1 fuera de la zona $2-$3',
    ca: 'preu $1 fora de la zona $2-$3',
  }],
];

// ---------------------------------------------------------------------------- state

function detect() {
  try {
    const saved = localStorage.getItem(STORE_KEY);
    if (LANGS.includes(saved)) return saved;
  } catch {
    // Private browsing, or storage blocked. Not a reason to fail: fall through to the browser's
    // own preference.
  }
  for (const tag of navigator.languages ?? [navigator.language ?? '']) {
    const base = String(tag).toLowerCase().split('-')[0];
    if (LANGS.includes(base)) return base;
  }
  return 'en';
}

let lang = detect();
const listeners = [];

export const getLang = () => lang;
export const locale = () => LOCALE[lang];

/** A key's text in the current language, with `{placeholders}` filled in. */
export function t(key, vars) {
  const s = DICTS[lang]?.[key] ?? EN[key] ?? key;
  return vars ? s.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? vars[k] : m)) : s;
}

/** Text that came from the API (English): exact match first, then by pattern, then untouched. */
/**
 * A captured group that is ENTIRELY a number gets the reader's decimal separator.
 *
 * The server writes `2.9%` and the stat tiles are formatted with Intl, so the Spanish page was
 * showing `2,9%` in the tile and `2.9%` in the sentence explaining it, three lines apart. Same
 * number, two spellings; it reads as two different figures.
 *
 * Strict on purpose. Only a bare number, optionally signed, optionally a percentage — so a date
 * (2024-03-14), an identifier (`close`), a column list and a pandas message all pass through
 * untouched. Anything looser and this would start rewriting the user's own data back at them.
 */
function localiseNumber(g) {
  if (LOCALE[lang].startsWith('en')) return g;
  // Anchored on a WHOLE well-formed English number: optional sign, optional 3-digit grouping,
  // optional decimals, optional percent. That shape cannot be a date (2024-03-14), an identifier
  // (`close`), a column list or a pandas message, so nothing else can be caught by accident.
  if (!/^-?\d{1,3}(,\d{3})*(\.\d+)?%?$/.test(g) && !/^-?\d+(\.\d+)?%?$/.test(g)) return g;
  return g.replace(/,/g, '\u0000').replace(/\./g, ',').replace(/\u0000/g, '.');
}

export function tx(text) {
  if (!text || lang === 'en') return text;
  const table = SERVER[lang];
  if (table && table[text]) return table[text];
  for (const [re, tr] of PATTERNS) {
    const m = re.exec(text);
    // A capture can itself be a translatable sentence. The chart label is: the "state «…» offers
    // no entry" refusal embeds `terminal_label`, so a Catalan reader was getting a Catalan
    // sentence with «w3 complete → inside w4» sitting in the middle of it. Look each group up in
    // the table before substituting; anything not in it — a number, `close`, a pandas message —
    // passes through untouched, which is what the $-groups are for.
    if (m && tr[lang]) return tr[lang].replace(/\$(\d)/g, (_, i) => {
      const g = m[Number(i)] ?? '';
      return (table && table[g]) || localiseNumber(g);
    });
  }
  return text;
}

// ------------------------------------------------------------ the hypotheses' pre-registration
//
// `rationale` and `prior` are the longest prose in the product —110 KB per language— and the
// English reader must not pay for them. So they do NOT live in the dictionaries above: they sit
// in /hyp.es.json and /hyp.ca.json and are fetched the first time a reader actually asks for
// Spanish or Catalan. Once per language, per page load.
//
// The Python side stays English-only on purpose: the pre-registration is the scientific record
// and the record is in English. These files are a reading aid layered over it, which is exactly
// why a fetch that fails must show the English text and never an empty panel.

const HYP = {};          // lang -> { name: {rationale, prior} }; {} after a failed fetch
const HYP_PENDING = {};  // lang -> in-flight promise, so two callers share one request

/**
 * Makes sure the current language's prose is in memory. Resolves immediately for English and for
 * an already-loaded language; NEVER rejects — a failure caches an empty table, and `hypProse`
 * then falls back to the API's English.
 */
export function ensureHypProse(which = lang) {
  if (which === 'en' || HYP[which]) return Promise.resolve();
  if (!HYP_PENDING[which]) {
    HYP_PENDING[which] = fetch(`./hyp.${which}.json`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d) => { HYP[which] = d; })
      .catch(() => {
        // Cached as empty rather than left undefined: retrying on every repaint would hammer a
        // server that has already said no, and the page reads perfectly well in English.
        HYP[which] = {};
      })
      .finally(() => { delete HYP_PENDING[which]; });
  }
  return HYP_PENDING[which];
}

/**
 * The translated `{rationale, prior}` for a hypothesis, or `null` when there is none to use:
 * English, prose not loaded yet, fetch failed, or a result that is not a catalogue hypothesis
 * at all (an uploaded CSV, a hand-written rule). `null` means "render what the API sent".
 */
export function hypProse(name) {
  if (!name || lang === 'en') return null;
  return HYP[lang]?.[name] ?? null;
}

/** Registers a callback to re-render whatever `applyStatic` cannot reach. */
export function onLangChange(fn) {
  listeners.push(fn);
}

export function applyStatic(root = document) {
  document.documentElement.lang = lang;
  for (const el of root.querySelectorAll('[data-i18n]')) el.textContent = t(el.dataset.i18n);
  for (const el of root.querySelectorAll('[data-i18n-html]')) el.innerHTML = t(el.dataset.i18nHtml);
  for (const el of root.querySelectorAll('[data-i18n-placeholder]')) {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  }
  for (const el of root.querySelectorAll('[data-i18n-label]')) {
    el.setAttribute('aria-label', t(el.dataset.i18nLabel));
  }
}

export function setLang(next) {
  if (!LANGS.includes(next) || next === lang) return;
  lang = next;
  try {
    localStorage.setItem(STORE_KEY, lang);
  } catch {
    // The choice will not survive a reload, but the page still switches. Failing here would be
    // absurd: nobody cares less about storage quotas than someone who just clicked "CA".
  }
  applyStatic();
  paintSwitcher();
  for (const fn of listeners) fn(lang);
}

function paintSwitcher() {
  for (const b of document.querySelectorAll('[data-lang]')) {
    b.setAttribute('aria-pressed', String(b.dataset.lang === lang));
  }
}

function mountSwitcher() {
  const host = document.getElementById('lang');
  if (!host) return;
  host.innerHTML = '';
  LANGS.forEach((code, i) => {
    if (i) host.appendChild(Object.assign(document.createElement('span'),
      { className: 'sep', textContent: '·' }));
    const b = document.createElement('button');
    b.type = 'button';
    b.dataset.lang = code;
    b.textContent = LANG_NAMES[code];
    b.lang = code;
    b.onclick = () => setLang(code);
    host.appendChild(b);
  });
  paintSwitcher();
}

// Module scripts are deferred, so the document is already parsed here: the page can be translated
// before it is ever painted and there is no flash of English.
applyStatic();
mountSwitcher();
