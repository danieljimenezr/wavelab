import { createChart, LineSeries } from './vendor/lightweight-charts.standalone.production.mjs';

const $ = (id) => document.getElementById(id);
const pct = (v, d = 1) => (v * 100).toLocaleString('es-ES',
  { minimumFractionDigits: d, maximumFractionDigits: d }) + '%';

const VERDICTO = {
  sobrevive: ['SOBREVIVE', 'v-sobrevive'],
  dudosa: ['DUDOSA', 'v-dudosa'],
  no_sobrevive: ['NO SOBREVIVE', 'v-no_sobrevive'],
  fuga: ['MIRA AL FUTURO', 'v-fuga'],
};
const ETIQ = { pasa: 'PASA', falla: 'FALLA', no_concluyente: 'SIN DATOS' };

let chart = null, sEstr = null, sBH = null;

async function cargarCatalogo() {
  const d = await (await fetch('/api/hipotesis')).json();
  const sel = $('hyp');
  sel.innerHTML = '';
  let familia = null;
  for (const h of d.hipotesis) {
    if (h.family !== familia) {
      familia = h.family;
      sel.appendChild(Object.assign(document.createElement('optgroup'), { label: familia }));
    }
    sel.lastChild.appendChild(
      Object.assign(document.createElement('option'), { value: h.name, textContent: h.name }));
  }
  $('status').textContent = `${d.hipotesis.length} estrategias registradas`;
}

function pintarCurva(curva) {
  if (!chart) {
    chart = createChart($('chart'), {
      autoSize: true,
      layout: { background: { color: '#161b22' }, textColor: '#8b949e', attributionLogo: false },
      grid: { vertLines: { color: '#21262d' }, horzLines: { color: '#21262d' } },
      rightPriceScale: { borderColor: '#21262d', mode: 1 },   // 1 = escala logarítmica
      timeScale: { borderColor: '#21262d', timeVisible: false },
      localization: { locale: 'es-ES' },
    });
    sBH = chart.addSeries(LineSeries,
      { color: '#8b949e', lineWidth: 1, title: 'comprar y mantener', priceLineVisible: false });
    sEstr = chart.addSeries(LineSeries,
      { color: '#58a6ff', lineWidth: 2, title: 'estrategia', priceLineVisible: false });
  }
  sEstr.setData(curva.map((p) => ({ time: p.t, value: p.e })));
  sBH.setData(curva.map((p) => ({ time: p.t, value: p.b })));
  chart.timeScale().fitContent();
}

async function pedir(modo, tf) {
  if (modo === 'catalogo') {
    const hyp = $('hyp').value;
    return (await fetch(`/api/validar?hyp=${encodeURIComponent(hyp)}&tf=${tf}`)).json();
  }
  if (modo === 'csv') {
    const f = $('fichero').files[0];
    if (!f) return { error: 'elige primero un fichero CSV' };
    const texto = await f.text();
    return (await fetch(`/api/validar_csv?tf=${tf}`,
      { method: 'POST', headers: { 'Content-Type': 'text/plain' }, body: texto })).json();
  }
  return (await fetch('/api/validar_regla', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tf, largo: $('largo').value, corto: $('corto').value }),
  })).json();
}

async function validar(ev) {
  const modo = ev.currentTarget.dataset.modo;
  const tf = ev.currentTarget.closest('.pane').querySelector('.tf').value;
  document.querySelectorAll('.run').forEach((b) => (b.disabled = true));
  // Progreso HONESTO: no una barra inventada que avanza sola, sino lo que de verdad está
  // ocurriendo. Lo que tarda son los 250 filtros aleatorios de control, y decirlo evita que
  // diez segundos de silencio parezcan un cuelgue.
  $('out').innerHTML = `<div class="cargando">
    <div class="paso">Calculando la señal sobre toda la serie…</div>
    <div class="paso">Reejecutándola con un retraso de una barra (detector de fugas)…</div>
    <div class="paso">Comparando contra la tasa base del mercado…</div>
    <div class="paso"><b>Generando 250 filtros aleatorios de la misma exposición…</b>
      (esto es lo que tarda)</div>
    <div class="paso">Partiendo el histórico con purga para medir fuera de muestra…</div>
    <div class="barra"><i></i></div></div>`;
  let d;
  try { d = await pedir(modo, tf); } catch (e) { d = { error: String(e) }; }
  document.querySelectorAll('.run').forEach((b) => (b.disabled = false));
  if (d.error) {
    // Un error de importación es información útil, no un fallo: se enseña entero.
    $('out').innerHTML = `<div class="verdict v-dudosa"><b>No se pudo validar</b>${d.error}</div>`;
    return;
  }

  const [txt, cls] = VERDICTO[d.veredicto] ?? ['?', ''];
  const kpi = (l, v, c) => `<div class="kpi"><div class="l">${l}</div>
    <div class="v">${v}</div><div class="c">${c}</div></div>`;

  $('out').innerHTML = `
    ${d.informe ? `<div class="informe"><b>Esto es lo que he entendido de tus datos:</b><br>
       ${d.informe.map((x) => `· ${x}`).join('<br>')}</div>` : ''}
    <div class="verdict ${cls}"><b>${txt}</b>${d.resumen}</div>
    <div class="grid">
      ${kpi('CAGR', pct(d.cagr), `comprar y mantener ${pct(d.cagr_bh)}`)}
      ${kpi('Sharpe', d.sharpe.toFixed(2), `comprar y mantener ${d.sharpe_bh.toFixed(2)}`)}
      ${kpi('Peor caída', pct(d.max_dd, 0), `comprar y mantener ${pct(d.max_dd_bh, 0)}`)}
      ${kpi('Exposición', pct(d.exposure, 0), 'del tiempo dentro del mercado')}
      ${kpi('Episodios', d.n_effective, `de ${d.n_signals} señales`)}
    </div>
    <div id="chart"></div>
    ${d.rationale ? `<div class="meta">
      <b>Por qué debería funcionar</b> (declarado antes de ver ningún resultado)<br>${d.rationale}
      <br><br><b>Cuándo quedaría desmentida</b><br>${d.prior}</div>` : ''}
    <div class="acciones">
      <button id="copiar">Copiar informe</button>
      <button id="descargar">Descargar informe (.md)</button>
    </div>
    ${d.tests.map((t) => `
      <div class="test ${t.estado}">
        <h3>${t.titulo}<span class="tag">${ETIQ[t.estado]}</span></h3>
        <div class="det">${t.detalle}</div>
        <div class="exp">${t.explicacion}</div>
      </div>`).join('')}`;

  chart = null;
  pintarCurva(d.curva);

  // Un informe que no se puede enseñar a nadie no sirve para decidir en equipo ni para vender.
  const md = informeMarkdown(d);
  $('copiar').onclick = async () => {
    await navigator.clipboard.writeText(md);
    $('copiar').textContent = 'copiado ✓';
    setTimeout(() => ($('copiar').textContent = 'Copiar informe'), 1800);
  };
  $('descargar').onclick = () => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([md], { type: 'text/markdown' }));
    a.download = `assay-${d.nombre.replace(/[^\w.-]+/g, '_')}.md`;
    a.click();
    URL.revokeObjectURL(a.href);
  };
}

function informeMarkdown(d) {
  const ICO = { pasa: '✅', falla: '❌', no_concluyente: '➖' };
  return `# Assay — ${d.nombre}

**${VERDICTO[d.veredicto]?.[0] ?? d.veredicto}** — ${d.resumen}

| | estrategia | comprar y mantener |
|---|---|---|
| CAGR | ${pct(d.cagr)} | ${pct(d.cagr_bh)} |
| Sharpe | ${d.sharpe.toFixed(2)} | ${d.sharpe_bh.toFixed(2)} |
| Peor caída | ${pct(d.max_dd, 0)} | ${pct(d.max_dd_bh, 0)} |
| Exposición | ${pct(d.exposure, 0)} | 100% |
| Episodios independientes | ${d.n_effective} | — |

## Las cinco pruebas

${d.tests.map((t) => `### ${ICO[t.estado]} ${t.titulo}
${t.detalle}

> ${t.explicacion}`).join('\n\n')}
${d.informe ? `\n## Lectura de los datos\n\n${d.informe.map((x) => `- ${x}`).join('\n')}` : ''}
${d.rationale ? `\n## Razonamiento declarado antes de medir\n\n${d.rationale}\n\n**Cuándo quedaría desmentida:** ${d.prior}` : ''}

---
Generado por **Assay** (DR Markets). Herramienta de validación, no de recomendación
de inversión.
Que una estrategia sobreviva no demuestra que gane dinero: demuestra que no es ninguno de los
cinco errores conocidos que hacen que un backtest bonito pierda dinero en real.
`;
}

document.querySelectorAll('.run').forEach((b) => (b.onclick = validar));
document.querySelectorAll('.tab').forEach((t) => (t.onclick = () => {
  document.querySelectorAll('.tab').forEach((x) => x.classList.toggle('on', x === t));
  document.querySelectorAll('.pane').forEach((p) =>
    p.classList.toggle('oculto', p.id !== 'p-' + t.dataset.t));
}));

async function cargarAyuda() {
  const a = await (await fetch('/api/ayuda_regla')).json();
  $('ayuda').innerHTML =
    Object.entries(a.series).map(([k, v]) => `<div><code>${k}</code> — ${v}</div>`).join('') +
    Object.entries(a.funciones).map(([k, v]) => `<div><code>${k}</code> — ${v}</div>`).join('');
}

document.querySelectorAll('.ej').forEach((b) => (b.onclick = () => {
  $('largo').value = b.dataset.l || '';
  $('corto').value = b.dataset.c || '';
}));

cargarCatalogo();
cargarAyuda();
