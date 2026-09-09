import { createChart, LineSeries } from './vendor/lightweight-charts.standalone.production.mjs';
import { ensureHypProse, hypProse, locale, onLangChange, t, tx } from './i18n.js';

const $ = (id) => document.getElementById(id);
const pct = (v, d = 1) => (v * 100).toLocaleString(locale(),
  { minimumFractionDigits: d, maximumFractionDigits: d }) + '%';

//: The four verdicts the battery can return, mapped to their label key and their CSS class. The
//: values are the API's, and they are the product: `lookahead` is not a shade of `fails`, it is
//: the accusation that the strategy is reading the future.
const VERDICT = {
  survives: ['v.survives', 'v-survives'],
  doubtful: ['v.doubtful', 'v-doubtful'],
  fails: ['v.fails', 'v-fails'],
  lookahead: ['v.lookahead', 'v-lookahead'],
};
const TAG = { pass: 'tag.pass', fail: 'tag.fail', inconclusive: 'tag.inconclusive' };

let chart = null, sStrategy = null, sBH = null;
//: Everything on screen is kept so a language switch can repaint it without asking the server to
//: run 250 random controls a second time.
let lastResult = null;
let lastHelp = null;
let lastCount = null;

//: The pre-registration prose. The API always speaks English; in es/ca the translated version is
//: substituted here, by hypothesis name. `??` and not `||`: a translation is either there or it
//: is not, and the English text is the fallback in both this panel and the exported report.
function prose(d) {
  const tr = hypProse(d.name);
  return { rationale: tr?.rationale ?? d.rationale, prior: tr?.prior ?? d.prior };
}

async function loadCatalogue() {
  const d = await (await fetch('/api/hypotheses')).json();
  const sel = $('hyp');
  sel.innerHTML = '';
  let family = null;
  for (const h of d.hypotheses) {
    if (h.family !== family) {
      family = h.family;
      sel.appendChild(Object.assign(document.createElement('optgroup'), { label: family }));
    }
    sel.lastChild.appendChild(
      Object.assign(document.createElement('option'), { value: h.name, textContent: h.name }));
  }
  lastCount = d.hypotheses.length;
  $('status').textContent = t('catalog.count', { n: lastCount });
}

function paintCurve(curve) {
  if (!chart) {
    chart = createChart($('chart'), {
      autoSize: true,
      layout: { background: { color: '#161b22' }, textColor: '#8b949e', attributionLogo: false },
      grid: { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
      rightPriceScale: { borderColor: '#21262d', mode: 1 },   // 1 = logarithmic scale
      timeScale: { borderColor: '#21262d', timeVisible: false },
      localization: { locale: locale() },
    });
    sBH = chart.addSeries(LineSeries,
      { color: '#8b949e', lineWidth: 1, title: t('chart.bh'), priceLineVisible: false });
    sStrategy = chart.addSeries(LineSeries,
      { color: '#58a6ff', lineWidth: 2, title: t('chart.strategy'), priceLineVisible: false });
  }
  sStrategy.setData(curve.map((p) => ({ time: p.t, value: p.e })));
  sBH.setData(curve.map((p) => ({ time: p.t, value: p.b })));
  chart.timeScale().fitContent();
}

async function request(mode, tf) {
  if (mode === 'catalog') {
    const hyp = $('hyp').value;
    return (await fetch(`/api/validate?hyp=${encodeURIComponent(hyp)}&tf=${tf}`)).json();
  }
  if (mode === 'csv') {
    const f = $('file').files[0];
    if (!f) return { error: t('err.no_file') };
    const text = await f.text();
    return (await fetch(`/api/validate_csv?tf=${tf}`,
      { method: 'POST', headers: { 'Content-Type': 'text/plain' }, body: text })).json();
  }
  return (await fetch('/api/validate_rule', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tf, long: $('long').value, short: $('short').value }),
  })).json();
}

async function validate(ev) {
  const mode = ev.currentTarget.dataset.mode;
  const tf = ev.currentTarget.closest('.pane').querySelector('.tf').value;
  document.querySelectorAll('.run').forEach((b) => (b.disabled = true));
  // HONEST progress: not an invented bar that advances on its own, but what is actually
  // happening. What takes the time is the 250 random control filters, and saying so stops ten
  // seconds of silence from looking like a hang.
  $('out').innerHTML = `<div class="loading">
    <div class="step">${t('run.step1')}</div>
    <div class="step">${t('run.step2')}</div>
    <div class="step">${t('run.step3')}</div>
    <div class="step">${t('run.step4')}</div>
    <div class="step">${t('run.step5')}</div>
    <div class="bar"><i></i></div></div>`;
  let d;
  try { d = await request(mode, tf); } catch (e) { d = { error: String(e) }; }
  document.querySelectorAll('.run').forEach((b) => (b.disabled = false));
  lastResult = d;
  // Normally already resolved —the fetch was started at page load— but a first run in es/ca on a
  // cold cache waits for it here rather than painting English prose and swapping it a beat later.
  await ensureHypProse();
  render(d);
}

function render(d) {
  if (d.error) {
    // An import error is useful information, not a failure: it is shown in full.
    $('out').innerHTML =
      `<div class="verdict v-doubtful"><b>${t('err.title')}</b>${tx(d.error)}</div>`;
    return;
  }

  const [labelKey, cls] = VERDICT[d.verdict] ?? [null, ''];
  const p = prose(d);
  const kpi = (l, v, c) => `<div class="kpi"><div class="l">${l}</div>
    <div class="v">${v}</div><div class="c">${c}</div></div>`;

  $('out').innerHTML = `
    ${d.report ? `<div class="report"><b>${t('report.heading')}</b><br>
       ${d.report.map((x) => `· ${tx(x)}`).join('<br>')}</div>` : ''}
    <div class="verdict ${cls}"><b>${labelKey ? t(labelKey) : d.verdict}</b>${tx(d.summary)}</div>
    <div class="grid">
      ${kpi(t('kpi.cagr'), pct(d.cagr), t('kpi.bh', { v: pct(d.cagr_bh) }))}
      ${kpi(t('kpi.sharpe'), d.sharpe.toFixed(2), t('kpi.bh', { v: d.sharpe_bh.toFixed(2) }))}
      ${kpi(t('kpi.dd'), pct(d.max_dd, 0), t('kpi.bh', { v: pct(d.max_dd_bh, 0) }))}
      ${kpi(t('kpi.exposure'), pct(d.exposure, 0), t('kpi.in_market'))}
      ${kpi(t('kpi.episodes'), d.n_effective, t('kpi.of_signals', { n: d.n_signals }))}
    </div>
    <div id="chart"></div>
    ${p.rationale ? `<div class="meta">
      ${t('meta.rationale')}<br>${p.rationale}
      <br><br><b>${t('meta.prior')}</b><br>${p.prior}</div>` : ''}
    <div class="actions">
      <button id="copy">${t('btn.copy')}</button>
      <button id="download">${t('btn.download')}</button>
    </div>
    ${d.tests.map((x) => `
      <div class="test ${x.status}">
        <h3>${tx(x.title)}<span class="tag">${t(TAG[x.status] ?? x.status)}</span></h3>
        <div class="det">${tx(x.detail)}</div>
        <div class="exp">${tx(x.explanation)}</div>
      </div>`).join('')}`;

  chart = null;
  paintCurve(d.equity_curve);

  // A report you cannot show anybody is no use for deciding as a team, and no use for selling.
  const md = markdownReport(d);
  $('copy').onclick = async () => {
    await navigator.clipboard.writeText(md);
    $('copy').textContent = t('btn.copied');
    setTimeout(() => ($('copy').textContent = t('btn.copy')), 1800);
  };
  $('download').onclick = () => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([md], { type: 'text/markdown' }));
    a.download = `assay-${d.name.replace(/[^\w.-]+/g, '_')}.md`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
}

function markdownReport(d) {
  const ICON = { pass: '✅', fail: '❌', inconclusive: '➖' };
  const label = VERDICT[d.verdict] ? t(VERDICT[d.verdict][0]) : d.verdict;
  const p = prose(d);
  return `# Assay — ${tx(d.name)}

**${label}** — ${tx(d.summary)}

| | ${t('md.strategy')} | ${t('md.bh')} |
|---|---|---|
| CAGR | ${pct(d.cagr)} | ${pct(d.cagr_bh)} |
| Sharpe | ${d.sharpe.toFixed(2)} | ${d.sharpe_bh.toFixed(2)} |
| ${t('md.dd')} | ${pct(d.max_dd, 0)} | ${pct(d.max_dd_bh, 0)} |
| ${t('md.exposure')} | ${pct(d.exposure, 0)} | 100% |
| ${t('md.episodes')} | ${d.n_effective} | — |

## ${t('md.tests')}

${d.tests.map((x) => `### ${ICON[x.status]} ${tx(x.title)}
${tx(x.detail)}

> ${tx(x.explanation)}`).join('\n\n')}
${d.report ? `\n## ${t('md.reading')}\n\n${d.report.map((x) => `- ${tx(x)}`).join('\n')}` : ''}
${p.rationale ? `\n## ${t('md.rationale')}\n\n${p.rationale}\n\n**${t('md.prior')}** ${p.prior}` : ''}

---
${t('md.footer')}
`;
}

document.querySelectorAll('.run').forEach((b) => (b.onclick = validate));
document.querySelectorAll('.tab').forEach((tab) => (tab.onclick = () => {
  document.querySelectorAll('.tab').forEach((x) => x.classList.toggle('on', x === tab));
  document.querySelectorAll('.pane').forEach((p) =>
    p.classList.toggle('hidden', p.id !== 'p-' + tab.dataset.t));
}));

async function loadHelp() {
  lastHelp = await (await fetch('/api/rule_help')).json();
  paintHelp();
}

function paintHelp() {
  // `series` has to be there, not just the object: if /api/rule_help ever answers with an error
  // payload, a bare `if (!lastHelp)` lets it through and the language switch dies half way,
  // leaving the page in two languages at once.
  if (!lastHelp?.series) return;
  // The keys are the DSL itself and stay in English in every language: they are code the user
  // types. Only the description beside them is translated.
  const row = ([k, v]) => `<div><code>${k}</code> — ${tx(v)}</div>`;
  $('help').innerHTML = Object.entries(lastHelp.series).map(row).join('')
    + Object.entries(lastHelp.functions).map(row).join('');
}

document.querySelectorAll('.ex').forEach((b) => (b.onclick = () => {
  $('long').value = b.dataset.l || '';
  $('short').value = b.dataset.c || '';
}));

onLangChange(() => {
  if (lastCount != null) $('status').textContent = t('catalog.count', { n: lastCount });
  paintHelp();
  if (!lastResult) return;
  // THE case this layer exists for: a result already on screen when the language changes. The
  // panel is repainted ONCE, after the prose for the new language is in hand, because `render`
  // rebuilds the chart from scratch — painting twice would make it blink. `ensureHypProse` never
  // rejects, so a dead file still repaints, with the English rationale.
  const at = lastResult;
  ensureHypProse().then(() => { if (lastResult === at) render(at); });
});

loadCatalogue();
loadHelp();
// Warmed at load, not at the first result: in es/ca the 110 KB is on its way while the reader is
// still choosing a hypothesis. In English this is a no-op and nothing is downloaded.
ensureHypProse();
