import { createChart, CandlestickSeries, HistogramSeries, LineSeries, LineStyle }
  from './vendor/lightweight-charts.standalone.production.mjs';
import { locale, onLangChange, t, tx } from './i18n.js';

const $ = (id) => document.getElementById(id);
const fmt = (n, d = 2) =>
  n == null ? '—' : n.toLocaleString(locale(), { minimumFractionDigits: d, maximumFractionDigits: d });

//: For values the SERVER controls (timeframe roles, health modes): if the backend ever adds one
//: the dictionaries do not know about, show the raw value rather than the key `role.whatever`.
const tOr = (key, fallback) => (t(key) === key ? fallback : t(key));

const chart = createChart($('chart'), {
  // autoSize is NOT optional. Without it, createChart measures the container ONCE, at creation
  // time; if the container has no size at that instant the canvases stay at 0 px and the chart
  // renders black without a single console error. A ResizeObserver that only calls fitContent()
  // fixes nothing: fitContent adjusts the time range, not the size of the canvas.
  autoSize: true,
  layout: { background: { color: '#0d1117' }, textColor: '#8b949e', attributionLogo: false },
  grid: { vertLines: { color: '#161b22' }, horzLines: { color: '#161b22' } },
  rightPriceScale: { borderColor: '#21262d', scaleMargins: { top: 0.06, bottom: 0.26 } },
  timeScale: { borderColor: '#21262d', timeVisible: true, secondsVisible: false },
  crosshair: { mode: 0 },
  localization: { locale: locale() },
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

// CONFIRMED legs: solid. This is structure that can no longer change.
const legs = chart.addSeries(LineSeries, {
  color: '#c9a227', lineWidth: 2, priceLineVisible: false, lastValueVisible: false,
  crosshairMarkerVisible: false,
});
// PROVISIONAL leg: dashed. It legitimately changes as price moves, and drawing it like a
// confirmed one would assert a certainty nobody has. The visual separation is not decoration:
// it is the difference between informing and lying.
const legProv = chart.addSeries(LineSeries, {
  color: '#8b7500', lineWidth: 2, lineStyle: LineStyle.Dashed, priceLineVisible: false,
  lastValueVisible: false, crosshairMarkerVisible: false,
});

let confirmLine = null;
let planLines = [];


let currentTf = '15m';
let lastBar = null;
let firstBar = null;
//: Kept so a language switch can repaint everything on screen without going back to the network.
let lastWaves = null;
let lastHealth = null;
let lastGaps = null;
let lastDecision = null;
let lastRoles = null;
let lastTfs = null;

function paintVolume(bars) {
  volume.setData(bars.map((b) => ({
    time: b.time, value: b.volume,
    color: b.close >= b.open ? 'rgba(38,166,154,.35)' : 'rgba(239,83,80,.35)',
  })));
}

function updatePrice(b) {
  if (!b) return;
  lastBar = b;
  $('px').textContent = '$' + fmt(b.close);
  if (firstBar) {
    const pct = ((b.close - firstBar.open) / firstBar.open) * 100;
    const el = $('chg');
    el.textContent = (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%';
    el.style.background = pct >= 0 ? 'rgba(38,166,154,.16)' : 'rgba(239,83,80,.16)';
    el.style.color = pct >= 0 ? '#26a69a' : '#ef5350';
  }
  document.title = `$${fmt(b.close, 0)} · wavelab`;
}

async function load(tf) {
  const r = await fetch(`/api/history?tf=${tf}&limit=1500`);
  const d = await r.json();
  if (d.error) { console.error(d); return; }
  currentTf = tf;
  $('sym').textContent = d.symbol;
  candles.setData(d.bars);
  paintVolume(d.bars);
  firstBar = d.bars[0];
  updatePrice(d.bars.at(-1));
  chart.timeScale().fitContent();
  lastTfs = d.timeframes; lastRoles = d.roles;
  if (!$('tfs').children.length) buildSelector(d.timeframes, d.roles);
  paintWaves(d.waves);
  paintHealth(d.health, d.gaps);
  decide();
}

function buildSelector(tfs, roles) {
  const nav = $('tfs');
  nav.innerHTML = '';
  // 1m is stored but never shown: it is the SOURCE series, not a timeframe of analysis.
  for (const tf of tfs.filter((x) => x !== '1m')) {
    const b = document.createElement('button');
    b.className = 'tf' + (tf === currentTf ? ' on' : '');
    // The role of each timeframe is an invariant of the system, not an interface preference:
    // 1d and 4h VETO the direction, 1h classifies the regime, and 15m is the only one allowed to
    // draw entries. Showing it stops the user waiting for a signal on a timeframe that will
    // never produce one.
    b.innerHTML = `${tf}<span class="role">${roles[tf] ? tOr('role.' + roles[tf], roles[tf]) : ''}</span>`;
    b.onclick = () => {
      [...nav.children].forEach((c) => c.classList.remove('on'));
      b.classList.add('on');
      load(tf);
    };
    nav.appendChild(b);
  }
}

function paintWaves(w) {
  if (!w) return;
  lastWaves = w;
  const conf = w.legs.filter((l) => !l.tentative).map((l) => ({ time: l.ts / 1000, value: l.price }));
  legs.setData(conf);

  const prov = w.legs.find((l) => l.tentative);
  legProv.setData(
    prov && conf.length
      ? [conf.at(-1), { time: prov.ts / 1000, value: prov.price }]
      : [],
  );

  // ★ The grey confirmation-price line.
  // Almost nobody implements it, and it is what turns Elliott's repainting weakness into the most
  // actionable line on the chart: you stop seeing "this might be the top" and start seeing the
  // EXACT price at which it stops being a maybe.
  if (confirmLine) { candles.removePriceLine(confirmLine); confirmLine = null; }
  if (w.confirm_price != null && prov) {
    const up = prov.kind === 1;
    confirmLine = candles.createPriceLine({
      price: w.confirm_price,
      color: '#8b949e',
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      axisLabelVisible: true,
      title: t(up ? 'confirm.below' : 'confirm.above'),
    });
  }
  $('s-piv').textContent = t('waves.line', {
    n: w.n_confirmed,
    atr: fmt(w.atr, 0),
    tail: w.confirm_price != null
      ? t('waves.tentative', {
        kind: t(prov?.kind === 1 ? 'kind.high' : 'kind.low'),
        price: fmt(w.confirm_price, 0),
      })
      : t('waves.none'),
  });
}

function paintHealth(h, gaps) {
  if (!h) return;
  lastHealth = h;
  if (gaps !== undefined) lastGaps = gaps;
  const connected = h.connected;
  const mute = h.silent_seconds > 180;
  $('d-feed').className = 'dot ' + (!connected || mute ? 'bad' : 'ok');
  $('s-feed').textContent = !connected
    ? t('feed.down')
    : mute
      // An OPEN connection that delivers nothing is a real, observed failure (Binance's futures
      // WebSocket does exactly that). What is watched is the silence, not the socket.
      ? t('feed.mute', { s: Math.round(h.silent_seconds) })
      : t('feed.live');

  $('s-mode').textContent = t('health.status', { mode: tOr('mode.' + h.mode, h.mode) });
  $('s-gaps').textContent = t('health.gaps', { n: lastGaps ?? h.gaps_in_window ?? 0 });
  $('s-lag').textContent = t('health.lag', { n: fmt(h.lag_bars, 1) });
  $('s-recon').textContent = t('health.reconnects', { n: h.reconnects, m: h.healed_bars });

  const banner = $('banner');
  if (h.mode === 'catch_up') {
    banner.textContent = t('banner.catch_up');
    banner.classList.add('show');
  } else if (h.mode === 'warmup') {
    banner.textContent = t('banner.warmup');
    banner.classList.add('show');
  } else {
    banner.classList.remove('show');
  }
}

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  const heartbeat = setInterval(() => ws.readyState === 1 && ws.send('.'), 25000);

  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === 'health') { paintHealth(m); return; }
    if (m.type === 'waves') { if (m.tf === currentTf) { paintWaves(m); decide(); } return; }
    // Only messages for the visible timeframe matter. The server broadcasts DIFFS for all of
    // them: filtering here is cheaper than opening one socket per timeframe.
    if (m.tf !== currentTf) return;
    if (m.type === 'bar' || m.type === 'tick') {
      candles.update(m.bar);
      volume.update({
        time: m.bar.time, value: m.bar.volume,
        color: m.bar.close >= m.bar.open ? 'rgba(38,166,154,.35)' : 'rgba(239,83,80,.35)',
      });
      updatePrice(m.bar);
    }
  };
  ws.onclose = () => { clearInterval(heartbeat); setTimeout(connect, 2000); };
  ws.onerror = () => ws.close();
}

// The trigger timeframe is 15m: the only one allowed to draw entries.
load('15m').then(connect);


// ---------------------------------------------------------------------------- decision
const VERDICT = {
  no_trade: ['v-no', 'verdict.no_trade'],
  watch: ['v-watch', 'verdict.watch'],
  actionable: ['v-act', 'verdict.actionable'],
};

function clearPlan() {
  planLines.forEach((l) => candles.removePriceLine(l));
  planLines = [];
}

function drawPlan(h) {
  clearPlan();
  if (!h || !h.viable) return;
  const line = (price, color, title, style = LineStyle.Solid, w = 1) =>
    planLines.push(candles.createPriceLine({
      price, color, lineWidth: w, lineStyle: style, axisLabelVisible: true, title,
    }));
  // Invalidation IS the product: the thickest line and the only red one.
  line(h.invalidation_price, '#f85149', h.invalidation_rule.split(' ')[0], LineStyle.Solid, 2);
  line(h.entry_lo, '#58a6ff', t('plan.zone'), LineStyle.Dotted);
  line(h.entry_hi, '#58a6ff', t('plan.zone'), LineStyle.Dotted);
  h.targets.forEach((x, i) => line(x, '#26a69a', `T${i + 1}`, LineStyle.Dashed));
}

function hypothesisRow(h, top) {
  const money = (v) => '$' + fmt(v, 0);
  const head = `<h4>${h.label} <span class="lbl">· ${t(h.direction === 'LONG' ? 'hyp.long' : 'hyp.short')}
    · ${t('hyp.fit', { score: h.score })}</span>${h.truncated ? `<span class="badge">${t('hyp.truncated')}</span>` : ''}</h4>`;

  if (!h.viable) {
    return `<div class="hyp${top ? ' top' : ''}">${head}
      <div class="row"><span class="lbl">${t('hyp.invalidation')}</span>
        <b class="inval">${money(h.invalidation_price)}</b></div>
      <div class="why">${h.reasons.map((r) => `<div>· ${tx(r)}</div>`).join('')}</div></div>`;
  }

  // The arithmetic of a REJECTION is shown exactly like the arithmetic of an acceptance. A "no"
  // without numbers is an opinion; with numbers it is an argument you can argue back against.
  return `<div class="hyp${top ? ' top' : ''}">${head}
    <div class="row"><span class="lbl">${t('hyp.entry_zone')}</span>
      <b class="zone">${money(h.entry_lo)} – ${money(h.entry_hi)}</b></div>
    <div class="row"><span class="lbl">${t('hyp.stop')}</span><b>${money(h.stop)}</b></div>
    <div class="row"><span class="lbl">${t('hyp.invalidation')} · ${h.invalidation_rule.split(' ')[0]}</span>
      <b class="inval">${money(h.invalidation_price)}</b></div>
    <div class="row"><span class="lbl">${t('hyp.targets')}</span>
      <b>${h.targets.map(money).join(' · ')}</b></div>
    <div class="arith">${t('hyp.arith', {
      rr: h.rr_t2,
      atr: h.stop_atr,
      cost: (h.cost_r * 100).toFixed(1),
      size: (h.size_factor * 100).toFixed(0),
      p: (h.p_required * 100).toFixed(1),
    })}</div>
    ${h.reasons.length ? `<div class="why">${h.reasons.map((r) => `<div>· ${tx(r)}</div>`).join('')}</div>` : ''}
  </div>`;
}

function paintDecision(d) {
  const panel = $('panel');
  if (d.error) { panel.innerHTML = `<div class="lbl">${tx(d.error)}</div>`; clearPlan(); return; }

  const [cls, key] = VERDICT[d.verdict] ?? ['v-no', null];
  const hs = d.hypotheses ?? [];
  const top = hs.find((h) => h.viable && h.in_zone) ?? hs.find((h) => h.viable) ?? hs[0];
  drawPlan(top);

  panel.innerHTML = `
    <div><span class="verdict ${cls}">${key ? t(key) : d.verdict}</span>
      <span class="badge" style="background:rgba(88,166,255,.14);color:#58a6ff">
        ${t('panel.maturity', { n: d.maturity })}</span></div>
    <div class="why">${(d.reasons ?? []).map((x) => `<div>· ${tx(x)}</div>`).join('')}</div>
    ${hs.length ? `<h3 class="sec">${t('panel.ranked')}</h3>` : ''}
    ${hs.map((h, i) => hypothesisRow(h, i === 0)).join('')}
    <div class="why" style="margin-top:14px">${t('panel.footer')}</div>`;
}

async function decide() {
  const r = await fetch(`/api/decide?tf=${currentTf}`);
  lastDecision = await r.json();
  paintDecision(lastDecision);
}

// A language switch repaints from the state already in memory: no refetch, and nothing on screen
// is left half-translated.
onLangChange(() => {
  chart.applyOptions({ localization: { locale: locale() } });
  if (lastTfs) buildSelector(lastTfs, lastRoles);
  if (lastWaves) paintWaves(lastWaves);
  if (lastHealth) paintHealth(lastHealth, lastGaps);
  if (lastDecision) paintDecision(lastDecision);
  if (lastBar) updatePrice(lastBar);
});
