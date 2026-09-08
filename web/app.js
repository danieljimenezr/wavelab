import { createChart, CandlestickSeries, HistogramSeries, LineSeries, LineStyle }
  from './vendor/lightweight-charts.standalone.production.mjs';

const $ = (id) => document.getElementById(id);
const fmt = (n, d = 2) =>
  n == null ? '—' : n.toLocaleString('es-ES', { minimumFractionDigits: d, maximumFractionDigits: d });

// El rol de cada timeframe es un invariante del sistema, no una preferencia de la interfaz:
// 1d y 4h VETAN la dirección, 1h clasifica el régimen, y 15m es el único que puede dibujar
// entradas. Mostrarlo evita que el usuario espere una señal en un timeframe que nunca la dará.
const ROLES = { gate: 'veto', regime: 'régimen', trigger: 'entradas', source: 'fuente' };

const chart = createChart($('chart'), {
  // autoSize NO es opcional. Sin él, createChart mide el contenedor UNA vez, en el momento de la
  // creación; si en ese instante aún no tiene tamaño, los canvas se quedan en 0 px y el gráfico
  // queda en negro sin ningún error de consola. Un ResizeObserver que solo llama a fitContent()
  // no arregla nada: fitContent ajusta el rango temporal, no el tamaño del lienzo.
  autoSize: true,
  layout: { background: { color: '#0d1117' }, textColor: '#8b949e', attributionLogo: false },
  grid: { vertLines: { color: '#161b22' }, horzLines: { color: '#161b22' } },
  rightPriceScale: { borderColor: '#21262d', scaleMargins: { top: 0.06, bottom: 0.26 } },
  timeScale: { borderColor: '#21262d', timeVisible: true, secondsVisible: false },
  crosshair: { mode: 0 },
  localization: { locale: 'es-ES' },
});

const candles = chart.addSeries(CandlestickSeries, {
  upColor: '#26a69a', downColor: '#ef5350', borderVisible: false,
  wickUpColor: '#26a69a', wickDownColor: '#ef5350',
  priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
});

const volume = chart.addSeries(HistogramSeries, {
  priceFormat: { type: 'volume' }, priceScaleId: 'vol',
});
chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });

// Tramos CONFIRMADOS: sólidos. Son estructura que ya no puede cambiar.
const legs = chart.addSeries(LineSeries, {
  color: '#c9a227', lineWidth: 2, priceLineVisible: false, lastValueVisible: false,
  crosshairMarkerVisible: false,
});
// Tramo PROVISIONAL: discontinuo. Cambia legítimamente según se mueve el precio, y pintarlo
// igual que uno confirmado sería afirmar una certeza que no se tiene. La separación visual no
// es estética: es la diferencia entre informar y mentir.
const legProv = chart.addSeries(LineSeries, {
  color: '#8b7500', lineWidth: 2, lineStyle: LineStyle.Dashed, priceLineVisible: false,
  lastValueVisible: false, crosshairMarkerVisible: false,
});

let confirmLine = null;


let tfActual = '15m';
let ultimaVela = null;
let primeraVela = null;

function pintarVolumen(bars) {
  volume.setData(bars.map((b) => ({
    time: b.time, value: b.volume,
    color: b.close >= b.open ? 'rgba(38,166,154,.35)' : 'rgba(239,83,80,.35)',
  })));
}

function actualizarPrecio(b) {
  if (!b) return;
  ultimaVela = b;
  $('px').textContent = '$' + fmt(b.close);
  if (primeraVela) {
    const pct = ((b.close - primeraVela.open) / primeraVela.open) * 100;
    const el = $('chg');
    el.textContent = (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%';
    el.style.background = pct >= 0 ? 'rgba(38,166,154,.16)' : 'rgba(239,83,80,.16)';
    el.style.color = pct >= 0 ? '#26a69a' : '#ef5350';
  }
  document.title = `$${fmt(b.close, 0)} · wavelab`;
}

async function cargar(tf) {
  const r = await fetch(`/api/history?tf=${tf}&limit=1500`);
  const d = await r.json();
  if (d.error) { console.error(d); return; }
  tfActual = tf;
  $('sym').textContent = d.symbol;
  candles.setData(d.bars);
  pintarVolumen(d.bars);
  primeraVela = d.bars[0];
  actualizarPrecio(d.bars.at(-1));
  chart.timeScale().fitContent();
  if (!$('tfs').children.length) construirSelector(d.timeframes, d.roles);
  pintarOndas(d.waves);
  salud(d.health, d.gaps);
}

function construirSelector(tfs, roles) {
  const nav = $('tfs');
  nav.innerHTML = '';
  // 1m se almacena pero no se muestra: es la serie FUENTE, no un timeframe de análisis.
  for (const tf of tfs.filter((t) => t !== '1m')) {
    const b = document.createElement('button');
    b.className = 'tf' + (tf === tfActual ? ' on' : '');
    b.innerHTML = `${tf}<span class="role">${ROLES[roles[tf]] ?? ''}</span>`;
    b.onclick = () => {
      [...nav.children].forEach((c) => c.classList.remove('on'));
      b.classList.add('on');
      cargar(tf);
    };
    nav.appendChild(b);
  }
}

function pintarOndas(w) {
  if (!w) return;
  const conf = w.legs.filter((l) => !l.tentative).map((l) => ({ time: l.ts / 1000, value: l.price }));
  legs.setData(conf);

  const prov = w.legs.find((l) => l.tentative);
  legProv.setData(
    prov && conf.length
      ? [conf.at(-1), { time: prov.ts / 1000, value: prov.price }]
      : [],
  );

  // ★ La línea gris del precio de confirmación.
  // Casi nadie la implementa, y es lo que convierte la debilidad de repintado de Elliott en la
  // línea más accionable del gráfico: dejas de ver "esto podría ser el techo" y pasas a ver el
  // precio EXACTO en el que deja de ser un quizá.
  if (confirmLine) { candles.removePriceLine(confirmLine); confirmLine = null; }
  if (w.confirm_price != null && prov) {
    const arriba = prov.kind === 1;
    confirmLine = candles.createPriceLine({
      price: w.confirm_price,
      color: '#8b949e',
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      axisLabelVisible: true,
      title: arriba ? 'confirma por debajo' : 'confirma por encima',
    });
  }
  $('s-piv').textContent =
    `pivotes: ${w.n_confirmed} · ATR ${fmt(w.atr, 0)} · ` +
    (w.confirm_price != null
      ? `${prov?.kind === 1 ? 'máximo' : 'mínimo'} tentativo, confirma en $${fmt(w.confirm_price, 0)}`
      : 'sin estructura');
}

function salud(h, gaps) {
  if (!h) return;
  const conectado = h.connected;
  const mudo = h.silent_seconds > 180;
  $('d-feed').className = 'dot ' + (!conectado || mudo ? 'bad' : 'ok');
  $('s-feed').textContent = !conectado
    ? 'feed caído'
    : mudo
      // Una conexión ABIERTA que no entrega nada es un fallo real y observado (el WebSocket de
      // futuros de Binance hace exactamente eso). Se vigila el silencio, no el socket.
      ? `conectado pero MUDO ${Math.round(h.silent_seconds)}s`
      : 'feed en vivo';

  const modos = { live: 'al día', catch_up: 'poniéndose al día', warmup: 'calentando' };
  $('s-mode').textContent = 'estado: ' + (modos[h.mode] ?? h.mode);
  $('s-gaps').textContent = `huecos en ventana: ${gaps ?? h.gaps_in_window ?? 0}`;
  $('s-lag').textContent = `retraso: ${fmt(h.lag_bars, 1)} velas`;
  $('s-recon').textContent = `reconexiones: ${h.reconnects} · velas curadas: ${h.healed_bars}`;

  const banner = $('banner');
  if (h.mode === 'catch_up') {
    banner.textContent =
      'Poniéndose al día tras un corte. El estado se está actualizando pero NO se emite ninguna ' +
      'decisión: una zona de entrada calculada sobre velas antiguas describe un precio que ya pasó.';
    banner.classList.add('show');
  } else if (h.mode === 'warmup') {
    banner.textContent = 'Calentando: cargando histórico. Aún no hay análisis.';
    banner.classList.add('show');
  } else {
    banner.classList.remove('show');
  }
}

function conectar() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  const latido = setInterval(() => ws.readyState === 1 && ws.send('.'), 25000);

  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === 'health') { salud(m); return; }
    if (m.type === 'waves') { if (m.tf === tfActual) pintarOndas(m); return; }
    // Solo interesan los mensajes del timeframe visible. El servidor manda DIFFS de todos:
    // filtrar aquí es más barato que abrir un socket por timeframe.
    if (m.tf !== tfActual) return;
    if (m.type === 'bar' || m.type === 'tick') {
      candles.update(m.bar);
      volume.update({
        time: m.bar.time, value: m.bar.volume,
        color: m.bar.close >= m.bar.open ? 'rgba(38,166,154,.35)' : 'rgba(239,83,80,.35)',
      });
      actualizarPrecio(m.bar);
    }
  };
  ws.onclose = () => { clearInterval(latido); setTimeout(conectar, 2000); };
  ws.onerror = () => ws.close();
}

// El timeframe de disparo es 15m: el único que puede dibujar entradas.
cargar('15m').then(conectar);
