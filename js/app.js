(() => {
const E = Engine;
const $ = id => document.getElementById(id);
const MODELS = {
  lr: { name: "Logistic regression", color: "var(--lr)" },
  forest: { name: "Random forest", color: "var(--rf)" },
  tree: { name: "Decision tree", color: "var(--dt)" },
};
const state = { dsKey: "bc", ds: DATASETS.bc, csv: null, testFrac: 0.2, nTrees: 25, seed: 20260928, version: 0 };

// ---------- formatting
const pct = (v, d = 2) => (v * 100).toFixed(d) + "%";
const pp = (v, d = 2) => (v * 100).toFixed(d) + " pp";
const fmtP = p => (p < 1e-4 ? p.toExponential(2) : p.toFixed(4));
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const nTestOf = (n, f) => Math.ceil(f * n);

// ---------- seeds derived from the master seed
function hash(str) { let h = 2166136261; for (const c of str) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); } return h >>> 0; }
function seedList(tag, n) { const r = E.rng((state.seed ^ hash(tag)) >>> 0); return Array.from({ length: n }, () => 1 + Math.floor(r() * 2147483646)); }

// ---------- worker pool (falls back to the main thread if workers are blocked)
const Pool = {
  workers: [], useWorkers: false, url: null,
  size: Math.max(1, Math.min(6, (navigator.hardwareConcurrency || 2) - 1)),
  async init() {
    try {
      this.url = "js/worker.js";
      const w = new Worker(this.url);
      const ok = await new Promise(res => {
        const t = setTimeout(() => res(false), 2000);
        w.onmessage = e => { if (e.data.type === "pong") { clearTimeout(t); res(true); } };
        w.onerror = () => { clearTimeout(t); res(false); };
        w.postMessage({ type: "ping" });
      });
      w.terminate();
      this.useWorkers = ok;
    } catch (e) { this.useWorkers = false; }
    $("engine-note").textContent = this.useWorkers
      ? `Models run in your browser on ${this.size} parallel worker${this.size > 1 ? "s" : ""}. Nothing is uploaded.`
      : "Models run in your browser. Nothing is uploaded.";
  },
  spawn() {
    this.kill();
    if (!this.useWorkers) return;
    for (let i = 0; i < this.size; i++) {
      const w = new Worker(this.url);
      w.postMessage({ type: "data", X: state.ds.X, y: state.ds.y });
      this.workers.push(w);
    }
  },
  kill() { this.workers.forEach(w => w.terminate()); this.workers = []; },
  token: 0,
  run(jobs, onResult) {
    const token = ++this.token;
    if (!this.useWorkers) return this.runLocal(jobs, onResult, token);
    if (!this.workers.length) this.spawn();
    return new Promise(resolve => {
      let next = 0, done = 0;
      if (!jobs.length) return resolve(true);
      const feed = w => {
        if (token !== this.token) return;
        if (next >= jobs.length) return;
        const id = next++;
        w.postMessage({ ...jobs[id], id });
      };
      this.workers.forEach(w => {
        w.onmessage = e => {
          if (token !== this.token || e.data.type !== "res") return;
          onResult(e.data.id, e.data.acc, e.data.nTest);
          done++;
          if (done === jobs.length) resolve(true); else feed(w);
        };
        feed(w);
      });
      this.cancel = () => resolve(false);
    });
  },
  async runLocal(jobs, onResult, token) {
    let t0 = performance.now();
    for (let id = 0; id < jobs.length; id++) {
      if (token !== this.token) return false;
      const j = jobs[id], d = state.ds;
      const s = E.split(d.y, j.testFrac, j.splitSeed);
      const m = E.fit(j.model, d.X, d.y, s.tr, j.modelSeed, j.nTrees);
      onResult(id, E.accuracy(m, d.X, d.y, s.te), s.te.length);
      if (performance.now() - t0 > 30) { await new Promise(r => setTimeout(r, 0)); t0 = performance.now(); }
    }
    return true;
  },
  stop() { this.token++; if (this.useWorkers) { this.kill(); } if (this.cancel) this.cancel(); },
};

// ---------- charts (hand-drawn SVG, redrawn on resize)
const ro = new ResizeObserver(entries => entries.forEach(e => e.target._draw && e.target._draw()));
function mount(el, draw) { el._draw = draw; draw(); if (!el._observed) { ro.observe(el); el._observed = true; } }
function ticks(min, max, n = 5) {
  if (max === min) { max = min + 1; }
  const span = max - min, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || 10 * mag;
  const out = []; for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
function frame(el, opt) {
  const W = Math.max(280, el.clientWidth), H = opt.height || 250;
  const m = { l: 52, r: 14, t: 10, b: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const xs = v => m.l + (opt.x1 === opt.x0 ? iw / 2 : (v - opt.x0) / (opt.x1 - opt.x0) * iw);
  const ys = v => m.t + ih - (v - opt.y0) / (opt.y1 - opt.y0) * ih;
  let g = "";
  for (const t of opt.yt) g += `<line x1="${m.l}" x2="${W - m.r}" y1="${ys(t)}" y2="${ys(t)}" style="stroke:var(--line)" stroke-width="1"/><text x="${m.l - 8}" y="${ys(t) + 4}" text-anchor="end">${opt.yf(t)}</text>`;
  if (opt.xt) for (const t of opt.xt) g += `<text x="${xs(t)}" y="${H - m.b + 18}" text-anchor="middle">${opt.xf ? opt.xf(t) : t}</text>`;
  g += `<line x1="${m.l}" x2="${W - m.r}" y1="${m.t + ih}" y2="${m.t + ih}" style="stroke:var(--muted)" stroke-width="1"/>`;
  if (opt.xl) g += `<text x="${m.l + iw / 2}" y="${H - 4}" text-anchor="middle">${opt.xl}</text>`;
  if (opt.yl) g += `<text transform="translate(12 ${m.t + ih / 2}) rotate(-90)" text-anchor="middle">${opt.yl}</text>`;
  return { W, H, m, iw, ih, xs, ys, g };
}
function svgWrap(f, inner, label) { return `<svg viewBox="0 0 ${f.W} ${f.H}" height="${f.H}" role="img" aria-label="${esc(label)}">${f.g}${inner}</svg>`; }
function empty(el, msg) { el._draw = null; el.innerHTML = `<div style="height:240px;display:grid;place-items:center" class="muted">${msg}</div>`; }
function yRange(vals, padFrac = 0.08, clamp = [0, 1]) {
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  const pad = Math.max((hi - lo) * padFrac, 0.005);
  return [Math.max(clamp[0], lo - pad), Math.min(clamp[1], hi + pad)];
}
// series: [{label,color,pts:[[x,y]],band:[[x,lo,hi]],dots,dash}]
function lineChart(el, spec) {
  mount(el, () => {
    const all = [];
    spec.series.forEach(s => { s.pts.forEach(p => all.push(p[1])); (s.band || []).forEach(b => { if (b[0] >= (spec.bandFrom || 0)) all.push(b[1], b[2]); }); });
    if (spec.hline != null) all.push(spec.hline.y);
    if (!all.length) return empty(el, "Waiting for results");
    const [y0, y1] = spec.yRange || yRange(all, 0.1, spec.clamp || [0, 1]);
    const xsAll = spec.series.flatMap(s => s.pts.map(p => p[0]));
    const x0 = spec.x0 ?? Math.min(...xsAll), x1 = spec.x1 ?? Math.max(...xsAll);
    let xt = ticks(x0, x1, Math.max(3, Math.floor(el.clientWidth / 90)));
    if (spec.xInt) xt = xt.filter(v => Number.isInteger(v));
    const f = frame(el, { x0, x1, y0, y1, yt: ticks(y0, y1, 5), yf: spec.yf, xt, xf: spec.xf, xl: spec.xl, yl: spec.yl, height: spec.height });
    let inner = "";
    if (spec.hline) {
      const y = f.ys(spec.hline.y);
      inner += `<line x1="${f.m.l}" x2="${f.W - f.m.r}" y1="${y}" y2="${y}" style="stroke:${spec.hline.color}" stroke-width="1.5" stroke-dasharray="5 4"/><text x="${f.W - f.m.r}" y="${y - 6}" text-anchor="end" style="fill:${spec.hline.color}">${spec.hline.label}</text>`;
    }
    for (const s of spec.series) {
      if (s.band && s.band.length > 1) {
        const cl = v => f.ys(Math.max(y0, Math.min(y1, v)));
        const top = s.band.map(b => `${f.xs(b[0])},${cl(b[2])}`), bot = s.band.slice().reverse().map(b => `${f.xs(b[0])},${cl(b[1])}`);
        inner += `<polygon points="${top.concat(bot).join(" ")}" style="fill:${s.color}" fill-opacity="0.16" stroke="none"/>`;
      }
    }
    for (const s of spec.series) {
      if (s.pts.length > 1) inner += `<polyline points="${s.pts.map(p => `${f.xs(p[0])},${f.ys(p[1])}`).join(" ")}" fill="none" style="stroke:${s.color}" stroke-width="2" ${s.dash ? 'stroke-dasharray="6 4"' : ""} stroke-linejoin="round"/>`;
      if (s.dots || s.pts.length === 1) inner += s.pts.map(p => `<circle cx="${f.xs(p[0])}" cy="${f.ys(p[1])}" r="${s.r || 3}" style="fill:${s.color}"><title>${esc(s.label)}: ${spec.tip ? spec.tip(p) : p[1]}</title></circle>`).join("");
      if (s.pts.length && s.endLabel) { const p = s.pts[s.pts.length - 1]; inner += `<circle cx="${f.xs(p[0])}" cy="${f.ys(p[1])}" r="4.5" style="fill:${s.color};stroke:var(--surface)" stroke-width="2"/>`; }
    }
    el.innerHTML = svgWrap(f, inner, spec.label || "chart");
  });
}
// cats: x labels; series: [{label,color,vals}] ; vals are heights
function barChart(el, spec) {
  mount(el, () => {
    const all = spec.series.flatMap(s => s.vals);
    if (!all.length) return empty(el, "Waiting for results");
    const y1 = Math.max(...all) * 1.12 || 1;
    const f = frame(el, { x0: 0, x1: 1, y0: 0, y1, yt: ticks(0, y1, 5), yf: spec.yf, xl: spec.xl, yl: spec.yl, height: spec.height });
    const n = spec.cats.length, k = spec.series.length;
    const slot = f.iw / n, bw = Math.max(2, Math.min(36, slot * 0.8 / k));
    const every = Math.ceil(n / Math.max(1, Math.floor(f.iw / 64)));
    let inner = "";
    spec.cats.forEach((c, i) => {
      const cx = f.m.l + slot * (i + 0.5);
      spec.series.forEach((s, j) => {
        const v = s.vals[i] || 0, x = cx - (k * bw) / 2 + j * bw, y = f.ys(v);
        if (v > 0) inner += `<rect x="${x}" y="${y}" width="${bw - 1}" height="${f.m.t + f.ih - y}" style="fill:${s.color}" rx="1"><title>${esc(s.label)} · ${esc(c)}: ${spec.tip ? spec.tip(v) : v}</title></rect>`;
      });
      if (i % every === 0) inner += `<text x="${cx}" y="${f.H - f.m.b + 18}" text-anchor="middle">${esc(c)}</text>`;
    });
    if (spec.vline != null) {
      const x = f.m.l + slot * spec.vline;
      inner += `<line x1="${x}" x2="${x}" y1="${f.m.t}" y2="${f.m.t + f.ih}" style="stroke:var(--ink)" stroke-width="1.5" stroke-dasharray="4 4"/><text x="${x + 6}" y="${f.m.t + 12}" text-anchor="start" style="fill:var(--ink)">${spec.vlineLabel || ""}</text>`;
    }
    el.innerHTML = svgWrap(f, inner, spec.label || "bar chart");
  });
}
function quantile(s, q) { const p = (s.length - 1) * q, b = Math.floor(p); return s[b + 1] !== undefined ? s[b] + (p - b) * (s[b + 1] - s[b]) : s[b]; }
function boxChart(el, spec) {
  mount(el, () => {
    const all = spec.groups.flatMap(g => g.vals);
    if (all.length < 2) return empty(el, "Waiting for results");
    const [y0, y1] = yRange(all, 0.1);
    const f = frame(el, { x0: 0, x1: 1, y0, y1, yt: ticks(y0, y1, 5), yf: spec.yf, yl: spec.yl, height: spec.height });
    const n = spec.groups.length, slot = f.iw / n, bw = Math.min(70, slot * 0.42);
    let inner = "";
    spec.groups.forEach((g, i) => {
      const s = g.vals.slice().sort((a, b) => a - b); if (!s.length) return;
      const q1 = quantile(s, 0.25), q2 = quantile(s, 0.5), q3 = quantile(s, 0.75), iqr = q3 - q1;
      const lo = s.find(v => v >= q1 - 1.5 * iqr), hi = [...s].reverse().find(v => v <= q3 + 1.5 * iqr);
      const cx = f.m.l + slot * (i + 0.5), x = cx - bw / 2;
      inner += `<line x1="${cx}" x2="${cx}" y1="${f.ys(hi)}" y2="${f.ys(q3)}" style="stroke:${g.color}" stroke-width="1.5"/><line x1="${cx}" x2="${cx}" y1="${f.ys(q1)}" y2="${f.ys(lo)}" style="stroke:${g.color}" stroke-width="1.5"/>`;
      inner += `<line x1="${cx - bw / 4}" x2="${cx + bw / 4}" y1="${f.ys(hi)}" y2="${f.ys(hi)}" style="stroke:${g.color}" stroke-width="1.5"/><line x1="${cx - bw / 4}" x2="${cx + bw / 4}" y1="${f.ys(lo)}" y2="${f.ys(lo)}" style="stroke:${g.color}" stroke-width="1.5"/>`;
      inner += `<rect x="${x}" y="${f.ys(q3)}" width="${bw}" height="${Math.max(1, f.ys(q1) - f.ys(q3))}" style="fill:${g.color};stroke:${g.color}" fill-opacity="0.18" stroke-width="1.5" rx="2"/>`;
      inner += `<line x1="${x}" x2="${x + bw}" y1="${f.ys(q2)}" y2="${f.ys(q2)}" style="stroke:${g.color}" stroke-width="2.5"/>`;
      s.filter(v => v < lo || v > hi).forEach(v => inner += `<circle cx="${cx}" cy="${f.ys(v)}" r="3.5" style="fill:var(--surface);stroke:${g.color}" stroke-width="1.5"/>`);
      inner += `<text x="${cx}" y="${f.H - f.m.b + 18}" text-anchor="middle">${esc(g.short || g.label)}</text>`;
      inner += `<title>${esc(g.label)}: median ${spec.yf(q2)}</title>`;
    });
    el.innerHTML = svgWrap(f, inner, spec.label || "box plot");
  });
}
function legend(el, items) { el.innerHTML = items.map(i => `<span><i class="dot" style="background:${i.color}${i.dash ? ";border-radius:0;height:3px;width:16px" : ""}"></i>${esc(i.label)}</span>`).join(""); }

// ---------- shared UI wiring
function modelChecks(el, ids, defaults) {
  el.innerHTML = ids.map(id => `<label class="chip"><input type="checkbox" id="${el.id}-${id}" value="${id}" ${defaults.includes(id) ? "checked" : ""}><i class="dot" style="background:${MODELS[id].color}"></i>${MODELS[id].name}</label>`).join("");
}
const checked = el => [...el.querySelectorAll("input:checked")].map(i => i.value);
function bindRange(id, fmt) { const el = $(id), v = $(id + "-v"); const up = () => (v.textContent = fmt(+el.value)); el.addEventListener("input", up); up(); return el; }

const runs = {}; // per-tab run state
function setBusy(tab, busy) {
  $(tab + "-run").disabled = busy; $(tab + "-stop").disabled = !busy;
  runs[tab] = runs[tab] || {}; runs[tab].busy = busy;
}
function progress(tab, done, total, label) {
  $(tab + "-bar").style.width = (total ? (100 * done) / total : 0) + "%";
  $(tab + "-prog").textContent = label;
}
function markRun(tab) { runs[tab] = runs[tab] || {}; runs[tab].version = state.version; $(tab + "-stale").hidden = true; }
function markStale() {
  state.version++;
  for (const t of ["a", "b", "c", "t"]) if (runs[t] && runs[t].version !== undefined && runs[t].version !== state.version) $(t + "-stale").hidden = false;
}
let raf = 0;
const schedule = fn => { if (raf) return; raf = requestAnimationFrame(() => { raf = 0; fn(); }); };

// ---------- dataset + settings
function setDataset(key) {
  for (const b of document.querySelectorAll(".seg button")) b.setAttribute("aria-pressed", String(b.dataset.ds === key));
  $("upload").hidden = key !== "csv";
  if (key === "csv" && !state.csv) { state.dsKey = "csv"; renderDsInfo(); return; }
  stopAll();
  state.dsKey = key; state.ds = key === "csv" ? state.csv : DATASETS[key];
  Pool.spawn(); renderDsInfo(); markStale();
}
function renderDsInfo() {
  const d = state.dsKey === "csv" ? state.csv : state.ds;
  if (!d) { $("dsinfo").textContent = "Load a CSV file to use your own data."; $("stepv").textContent = "–"; return; }
  const n1 = d.y.reduce((s, v) => s + v, 0), n = d.y.length, nt = nTestOf(n, state.testFrac);
  $("dsinfo").innerHTML = `${esc(d.desc)}<br><span class="mono">${n.toLocaleString()} rows · ${d.features.length} features · ${esc(d.classes[0])} ${n - n1} / ${esc(d.classes[1])} ${n1}</span>`;
  $("testfrac-v").textContent = `${Math.round(state.testFrac * 100)}% · ${nt} cases`;
  $("stepv").textContent = `${(100 / nt).toFixed(3)} pp`;
}
function stopAll() { Pool.stop(); for (const t of ["a", "b", "c", "t"]) if (runs[t] && runs[t].busy) { setBusy(t, false); progress(t, 0, 1, "Stopped"); } }

$("testfrac").addEventListener("input", e => { state.testFrac = +e.target.value / 100; renderDsInfo(); markStale(); });
bindRange("ntrees", v => v).addEventListener("input", e => { state.nTrees = +e.target.value; markStale(); });
$("seed").addEventListener("change", e => { const v = parseInt(e.target.value, 10); state.seed = isFinite(v) ? Math.abs(v) : 1; e.target.value = state.seed; markStale(); });
$("reroll").addEventListener("click", () => { state.seed = 1 + Math.floor(Math.random() * 99999999); $("seed").value = state.seed; markStale(); });
for (const b of document.querySelectorAll(".seg button")) b.addEventListener("click", () => setDataset(b.dataset.ds));

// CSV upload
function parseCSV(text) {
  const rows = []; let row = [], cur = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { cur += '"'; i++; } else q = false; } else cur += c; }
    else if (c === '"') q = true;
    else if (c === "," || c === ";" || c === "\t") { row.push(cur); cur = ""; }
    else if (c === "\n" || c === "\r") { if (c === "\r" && text[i + 1] === "\n") i++; row.push(cur); cur = ""; if (row.some(v => v.trim() !== "")) rows.push(row); row = []; }
    else cur += c;
  }
  row.push(cur); if (row.some(v => v.trim() !== "")) rows.push(row);
  return rows;
}
let csvRaw = null;
function buildCSV(target) {
  const { head, body, name } = csvRaw;
  const numeric = head.map((_, j) => body.every(r => r[j] === undefined || r[j].trim() === "" || isFinite(+r[j])));
  const feats = head.map((_, j) => j).filter(j => j !== target && numeric[j]);
  const labels = [...new Set(body.map(r => (r[target] || "").trim()).filter(v => v !== ""))].sort();
  if (labels.length !== 2) throw new Error(`The label column “${head[target]}” has ${labels.length} different values; it needs exactly 2.`);
  if (!feats.length) throw new Error("No numeric feature columns found.");
  const X = [], y = [];
  for (const r of body) {
    const lab = (r[target] || "").trim(); if (!labels.includes(lab)) continue;
    const row = feats.map(j => +r[j]); if (row.some(v => r === undefined || !isFinite(v))) continue;
    if (feats.some(j => r[j] === undefined || r[j].trim() === "")) continue;
    X.push(row); y.push(labels.indexOf(lab));
  }
  const n1 = y.reduce((s, v) => s + v, 0);
  if (X.length < 30) throw new Error(`Only ${X.length} complete rows; at least 30 are needed.`);
  if (n1 < 5 || X.length - n1 < 5) throw new Error("Each class needs at least 5 rows.");
  const skipped = head.length - 1 - feats.length;
  return { title: name, desc: `${name} · label “${head[target]}”${skipped ? ` · ${skipped} non-numeric column${skipped > 1 ? "s" : ""} skipped` : ""}`, features: feats.map(j => head[j]), classes: labels, X, y };
}
function useCSV(target) {
  try {
    state.csv = buildCSV(target);
    const big = state.csv.X.length > 4000;
    $("csvmsg").innerHTML = `<span class="mono">${state.csv.X.length.toLocaleString()} rows loaded.</span>${big ? " Large file: forests will be slow, so start with fewer splits." : ""}`;
    setDataset("csv");
  } catch (err) { $("csvmsg").innerHTML = `<span class="err">${esc(err.message)}</span>`; }
}
function loadFile(file) {
  if (!file) return;
  const fr = new FileReader();
  fr.onload = () => {
    const rows = parseCSV(String(fr.result));
    if (rows.length < 2) { $("csvmsg").innerHTML = '<span class="err">The file has no data rows.</span>'; return; }
    const head = rows[0].map((h, j) => h.trim() || `column ${j + 1}`), body = rows.slice(1);
    csvRaw = { head, body, name: file.name.replace(/\.[^.]+$/, "") };
    const cand = head.map((_, j) => j).filter(j => new Set(body.map(r => (r[j] || "").trim()).filter(Boolean)).size === 2);
    if (!cand.length) { $("csvmsg").innerHTML = '<span class="err">No column has exactly two values to use as the label.</span>'; return; }
    $("targetsel").innerHTML = cand.map(j => `<option value="${j}">${esc(head[j])}</option>`).join("");
    $("targetsel").value = cand[cand.length - 1];
    $("targetrow").hidden = false;
    useCSV(cand[cand.length - 1]);
  };
  fr.readAsText(file);
}
$("csvfile").addEventListener("change", e => loadFile(e.target.files[0]));
$("targetsel").addEventListener("change", e => useCSV(+e.target.value));
const up = $("upload");
up.addEventListener("dragover", e => { e.preventDefault(); up.classList.add("drag"); });
up.addEventListener("dragleave", () => up.classList.remove("drag"));
up.addEventListener("drop", e => { e.preventDefault(); up.classList.remove("drag"); loadFile(e.dataTransfer.files[0]); });

// ---------- tabs with hash routing
const TABS = ["splits", "seeds", "aa", "testsize"];
function showTab(name) {
  if (!TABS.includes(name)) name = "splits";
  for (const t of TABS) { $("tab-" + t).setAttribute("aria-selected", String(t === name)); $("panel-" + t).hidden = t !== name; }
  document.querySelectorAll(".chart").forEach(el => el._draw && el._draw());
}
for (const t of TABS) $("tab-" + t).addEventListener("click", () => { if (location.hash !== "#" + t) location.hash = t; else showTab(t); });
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

function guard() { if (!state.ds || (state.dsKey === "csv" && !state.csv)) { alertMsg("Load a CSV file first, or pick a built-in dataset."); return false; } return true; }
function alertMsg(msg) { for (const t of ["a", "b", "c", "t"]) $(t + "-prog").textContent = msg; }

// =====================================================================
// Experiment B: split count
// =====================================================================
modelChecks($("b-models"), ["lr", "forest", "tree"], ["lr", "forest", "tree"]);
bindRange("b-k", v => v);
$("b-stop").addEventListener("click", () => { Pool.stop(); setBusy("b", false); progress("b", 0, 1, "Stopped"); });
$("b-run").addEventListener("click", runB);

async function runB() {
  if (!guard()) return;
  const models = checked($("b-models"));
  if (!models.length) { progress("b", 0, 1, "Pick at least one model."); return; }
  stopAll();
  const K = +$("b-k").value, tf = state.testFrac, nt = state.nTrees;
  const splitSeeds = seedList("splits", K), mSeeds = seedList("split-models", K * models.length);
  const jobs = [];
  for (let j = 0; j < K; j++) models.forEach((m, mi) => jobs.push({ model: m, splitSeed: splitSeeds[j], testFrac: tf, modelSeed: mSeeds[j * models.length + mi], nTrees: nt }));
  const Y = Array.from({ length: K }, () => Array(models.length).fill(null));
  let done = 0, nTest = nTestOf(state.ds.y.length, tf);
  setBusy("b", true); markRun("b");
  const t0 = performance.now();
  const draw = () => renderB(models, Y, K, nTest, tf);
  draw();
  const ok = await Pool.run(jobs, (id, acc, n) => {
    Y[Math.floor(id / models.length)][id % models.length] = acc; nTest = n; done++;
    progress("b", done, jobs.length, `${done} of ${jobs.length} model fits · ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    schedule(draw);
  });
  setBusy("b", false);
  if (ok) { progress("b", 1, 1, `Done: ${K} splits × ${models.length} model${models.length > 1 ? "s" : ""} in ${((performance.now() - t0) / 1000).toFixed(1)} s`); draw(); }
}

function renderB(models, Y, K, nTest, tf) {
  // contiguous completed splits, so running stats are in split order
  let c = 0; while (c < K && Y[c].every(v => v !== null)) c++;
  const rows = Y.slice(0, c);
  const col = i => rows.map(r => r[i]);
  const tieTop = rows.map(r => { const mx = Math.max(...r); return r.map(v => Math.abs(v - mx) < 1e-12); });
  // KPIs
  $("b-kpis").innerHTML = models.map((m, i) => {
    const v = col(i); if (!v.length) return `<div class="kpi" style="--c:${MODELS[m].color}"><span class="name">${MODELS[m].name}</span><span class="big">–</span></div>`;
    const ci = E.tCI(v), lead = tieTop.filter(t => t[i]).length;
    return `<div class="kpi" style="--c:${MODELS[m].color}"><span class="name">${MODELS[m].name}</span><span class="big">${pct(E.mean(v))}</span>
      <span class="sub">95% CI ${pct(ci[0])} – ${pct(ci[1])}</span><span class="sub">SD across splits ${v.length > 1 ? pp(E.sd(v)) : "–"}</span>
      ${models.length > 1 ? `<span class="lead">Top score on <b class="num">${lead}</b> of ${c} splits</span>` : ""}</div>`;
  }).join("");
  // Running mean with CI band
  const series = models.map((m, i) => {
    const v = col(i), pts = [], band = [];
    let s = 0;
    v.forEach((a, k) => { s += a; const sub = v.slice(0, k + 1); pts.push([k + 1, s / (k + 1)]); if (k >= 1) { const ci = E.tCI(sub); band.push([k + 1, ci[0], ci[1]]); } });
    return { label: MODELS[m].name, color: MODELS[m].color, pts, band, endLabel: true };
  });
  legend($("b-leg1"), series); legend($("b-leg2"), series);
  lineChart($("b-running"), { series, x0: 1, x1: K, bandFrom: Math.min(5, Math.max(2, c)), xInt: true, xl: "number of splits", yl: "mean accuracy", yf: v => (v * 100).toFixed(1) + "%", label: "Running mean accuracy by number of splits" });
  lineChart($("b-persplit"), { series: models.map((m, i) => ({ label: MODELS[m].name, color: MODELS[m].color, pts: col(i).map((a, k) => [k + 1, a]), dots: true, r: 2.5 })), x0: 1, x1: K, xInt: true, xl: "split number", yl: "accuracy", yf: v => (v * 100).toFixed(1) + "%", tip: p => `split ${p[0]}, ${pct(p[1])}`, label: "Accuracy on every split" });
  boxChart($("b-box"), { groups: models.map((m, i) => ({ label: MODELS[m].name, short: MODELS[m].name, color: MODELS[m].color, vals: col(i) })), yf: v => (v * 100).toFixed(1) + "%", yl: "accuracy", label: "Accuracy distribution by model" });
  // ANOVA
  if (models.length >= 2 && c >= 3) {
    const a = E.anova(rows);
    $("b-re").textContent = a.re.toFixed(1) + "%";
    $("b-re").style.color = a.re >= 100 ? "var(--good)" : "var(--warn)";
    $("b-reform").textContent = `MSE_CRD / MSE_RBD × 100 = ${a.crd.mse.toExponential(3)} / ${a.rbd.mse.toExponential(3)} × 100 · partial η² = ${a.eta.toFixed(3)}`;
    const r = (src, ss, df, ms, F, p) => `<tr><td>${src}</td><td>${ss.toFixed(6)}</td><td>${df}</td><td>${ms.toExponential(3)}</td><td>${F == null ? "–" : F.toFixed(2)}</td><td>${p == null ? "–" : fmtP(p)}</td></tr>`;
    $("b-anova").innerHTML = `<tr><th>Source</th><th>SS</th><th>df</th><th>MS</th><th>F</th><th>p</th></tr>
      <tr><td colspan="6" class="muted" style="font-family:var(--body);font-size:12px">CRD (splits ignored)</td></tr>
      ${r("Model", a.crd.ssm, a.crd.dfm, a.crd.msm, a.crd.F, a.crd.p)}${r("Residual", a.crd.sse, a.crd.dfe, a.crd.mse)}
      <tr><td colspan="6" class="muted" style="font-family:var(--body);font-size:12px">RBD (split = block)</td></tr>
      ${r("Model", a.rbd.ssm, a.rbd.dfm, a.rbd.msm, a.rbd.F, a.rbd.p)}${r("Split", a.rbd.sss, a.rbd.dfs, a.rbd.mss, a.rbd.Fs, a.rbd.ps)}${r("Residual", a.rbd.sse, a.rbd.dfe, a.rbd.mse)}`;
    // paired tests
    let pr = `<tr><th>Difference</th><th>Mean</th><th>Naive 95% CI</th><th>Corrected CI</th><th>t (df ${c - 1})</th><th>p</th><th>Clear?</th></tr>`;
    for (let i = 0; i < models.length; i++) for (let j = i + 1; j < models.length; j++) {
      let A = col(i), B = col(j), ni = i, nj = j;
      if (E.mean(A) < E.mean(B)) { [A, B] = [B, A]; [ni, nj] = [nj, ni]; }
      const t = E.paired(A, B, tf), clear = t.cci[0] > 0 || t.cci[1] < 0;
      pr += `<tr><td>${MODELS[models[ni]].name} − ${MODELS[models[nj]].name}</td><td>${(t.m * 100 >= 0 ? "+" : "") + (t.m * 100).toFixed(2)} pp</td><td>[${(t.ci[0] * 100).toFixed(2)}, ${(t.ci[1] * 100).toFixed(2)}]</td><td>[${(t.cci[0] * 100).toFixed(2)}, ${(t.cci[1] * 100).toFixed(2)}]</td><td>${t.t.toFixed(2)}</td><td>${fmtP(t.p)}</td><td><span class="pill ${clear ? "yes" : "no"}">${clear ? "Yes" : "Not clear"}</span></td></tr>`;
    }
    $("b-paired").innerHTML = pr;
    // insight
    const means = models.map((_, i) => E.mean(col(i)));
    const bi = means.indexOf(Math.max(...means)), lead = tieTop.filter(t => t[bi]).length;
    const others = models.map((m, i) => ({ m, n: tieTop.filter(t => t[i] && !t[bi]).length })).filter(o => o.n > 0);
    $("b-insight").innerHTML = `Over ${c} splits, <b>${MODELS[models[bi]].name}</b> had the highest mean accuracy (${pct(means[bi])}) and tied for the top score on ${lead} of ${c} splits.`
      + (others.length ? ` On ${others.map(o => `${o.n} split${o.n > 1 ? "s" : ""} ${MODELS[o.m].name.toLowerCase()}`).join(" and ")} won outright, so a single run could have crowned a different model.` : " No other model won a split outright on this run.")
      + ` Blocking by split gave a relative efficiency of <b>${a.re.toFixed(1)}%</b>${a.re >= 100 ? `, so ignoring the splits would need about ${Math.round(a.re - 100)}% more of them for the same precision.` : ", so blocking did not reduce the error on this run."}`;
  } else {
    $("b-re").textContent = "–"; $("b-re").style.color = "";
    $("b-reform").textContent = models.length < 2 ? "Select two or more models to compare them." : "Needs at least 3 completed splits.";
    $("b-anova").innerHTML = ""; $("b-paired").innerHTML = "";
    const v = models.length ? col(0) : [];
    $("b-insight").innerHTML = v.length > 1 ? `${MODELS[models[0]].name} averaged ${pct(E.mean(v))} over ${c} splits, with an SD of ${pp(E.sd(v))} from split to split. One test case on ${nTest} is ${(100 / nTest).toFixed(2)} pp.` : "";
  }
}

// =====================================================================
// Experiment A: seed only
// =====================================================================
modelChecks($("a-models"), ["tree", "forest", "lr"], ["tree", "forest"]);
bindRange("a-n", v => v);
$("a-stop").addEventListener("click", () => { Pool.stop(); setBusy("a", false); progress("a", 0, 1, "Stopped"); });
$("a-run").addEventListener("click", runA);

async function runA() {
  if (!guard()) return;
  const models = checked($("a-models"));
  if (!models.length) { progress("a", 0, 1, "Pick at least one model."); return; }
  stopAll();
  const N = +$("a-n").value, tf = state.testFrac, nt = state.nTrees;
  const splitSeed = seedList("fixed-split", 1)[0], seeds = seedList("seed-runs", N * models.length);
  const jobs = []; models.forEach((m, mi) => { for (let i = 0; i < N; i++) jobs.push({ model: m, splitSeed, testFrac: tf, modelSeed: seeds[mi * N + i], nTrees: nt }); });
  const R = models.map(() => Array(N).fill(null));
  let done = 0, nTest = nTestOf(state.ds.y.length, tf);
  setBusy("a", true); markRun("a");
  const t0 = performance.now();
  const draw = () => renderA(models, R, N, nTest);
  draw();
  const ok = await Pool.run(jobs, (id, acc, n) => { R[Math.floor(id / N)][id % N] = acc; nTest = n; done++; progress("a", done, jobs.length, `${done} of ${jobs.length} model fits`); schedule(draw); });
  setBusy("a", false);
  if (ok) { progress("a", 1, 1, `Done: ${N} seeds × ${models.length} model${models.length > 1 ? "s" : ""} on one ${nTest}-case test set in ${((performance.now() - t0) / 1000).toFixed(1)} s`); draw(); }
}
function renderA(models, R, N, nTest) {
  const vals = R.map(r => { const o = []; for (const v of r) { if (v === null) break; o.push(v); } return o; });
  const counts = vals.map(v => v.map(a => Math.round(a * nTest)));
  const allK = counts.flat();
  if (!allK.length) { ["a-hist", "a-hw"].forEach(id => empty($(id), "Waiting for results")); $("a-kpis").innerHTML = ""; $("a-table").innerHTML = ""; $("a-insight").textContent = ""; return; }
  const kmin = Math.min(...allK), kmax = Math.max(...allK), cats = [];
  for (let k = kmin; k <= kmax; k++) cats.push(k);
  const series = models.map((m, i) => ({ label: MODELS[m].name, color: MODELS[m].color, vals: cats.map(k => counts[i].filter(v => v === k).length) }));
  legend($("a-leg1"), series);
  barChart($("a-hist"), { cats: cats.map(k => ((k / nTest) * 100).toFixed(2) + "%"), series, yl: "number of seeds", yf: v => Math.round(v) === v ? v : "", tip: v => `${v} seeds`, label: "Accuracy counts across seeds" });
  $("a-histcap").textContent = `Scores sit on a grid of ${(100 / nTest).toFixed(3)} pp: each step is one more of the ${nTest} test cases right.`;
  const hwS = models.map((m, i) => ({ label: MODELS[m].name, color: MODELS[m].color, pts: vals[i].map((_, k) => k).slice(1).map(k => [k + 1, E.tCI(vals[i].slice(0, k + 1))[2]]) }));
  legend($("a-leg2"), hwS);
  lineChart($("a-hw"), { series: hwS, x0: 2, x1: N, xInt: true, xl: "number of seeds", yl: "half-width (pp)", yf: v => (v * 100).toFixed(2), clamp: [0, 1], hline: { y: 0.005, label: "0.5 pp", color: "var(--warn)" }, label: "CI half-width against number of seeds" });
  $("a-kpis").innerHTML = models.map((m, i) => {
    const v = vals[i]; if (!v.length) return "";
    const u = new Set(counts[i]).size;
    return `<div class="kpi" style="--c:${MODELS[m].color}"><span class="name">${MODELS[m].name}</span><span class="big">${v.length > 1 ? pp(E.sd(v)) : "–"}</span><span class="sub">SD across ${v.length} seeds</span><span class="sub">${u} distinct score${u > 1 ? "s" : ""} · range ${pct(Math.min(...v))} – ${pct(Math.max(...v))}</span></div>`;
  }).join("");
  $("a-table").innerHTML = `<tr><th>Model</th><th>Runs</th><th>Mean</th><th>SD</th><th>95% CI for mean</th><th>Min</th><th>Max</th><th>Distinct</th></tr>` + models.map((m, i) => {
    const v = vals[i]; if (v.length < 2) return "";
    const ci = E.tCI(v);
    return `<tr><td>${MODELS[m].name}</td><td>${v.length}</td><td>${pct(v.length ? E.mean(v) : 0)}</td><td>${pp(E.sd(v))}</td><td>[${pct(ci[0])}, ${pct(ci[1])}]</td><td>${pct(Math.min(...v))}</td><td>${pct(Math.max(...v))}</td><td>${new Set(counts[i]).size}</td></tr>`;
  }).join("");
  const parts = models.map((m, i) => {
    const k = counts[i]; if (!k.length) return "";
    const lo = Math.min(...k), hi = Math.max(...k);
    return hi === lo ? `<b>${MODELS[m].name}</b> gave the same score every time (${pct(lo / nTest)}): it has no randomness to vary`
      : `<b>${MODELS[m].name}</b> got between ${lo} and ${hi} of the ${nTest} test cases right, a swing of ${hi - lo} case${hi - lo > 1 ? "s" : ""} (${((hi - lo) / nTest * 100).toFixed(2)} pp)`;
  }).filter(Boolean);
  $("a-insight").innerHTML = `With the data held fixed, ${parts.join("; ")}. The seed alone moves the score.`;
}

// =====================================================================
// Experiment C: A/A
// =====================================================================
bindRange("c-n", v => v);
$("c-stop").addEventListener("click", () => { Pool.stop(); setBusy("c", false); progress("c", 0, 1, "Stopped"); });
$("c-run").addEventListener("click", runC);
let lastC = null;
$("c-thr").addEventListener("change", () => lastC && renderC(lastC));

async function runC() {
  if (!guard()) return;
  stopAll();
  const model = $("c-model").value, N = +$("c-n").value, tf = state.testFrac, nt = state.nTrees;
  const fixed = seedList("fixed-split", 1)[0], ms = seedList("aa-models", 4 * N), ss = seedList("aa-splits", 2 * N);
  const jobs = [];
  for (let k = 0; k < N; k++) {
    jobs.push({ model, splitSeed: fixed, testFrac: tf, modelSeed: ms[4 * k], nTrees: nt }, { model, splitSeed: fixed, testFrac: tf, modelSeed: ms[4 * k + 1], nTrees: nt });
    jobs.push({ model, splitSeed: ss[2 * k], testFrac: tf, modelSeed: ms[4 * k + 2], nTrees: nt }, { model, splitSeed: ss[2 * k + 1], testFrac: tf, modelSeed: ms[4 * k + 3], nTrees: nt });
  }
  const A = Array(jobs.length).fill(null);
  const st = { model, N, A, nTest: nTestOf(state.ds.y.length, tf) };
  lastC = st;
  let done = 0; setBusy("c", true); markRun("c");
  const t0 = performance.now();
  renderC(st);
  const ok = await Pool.run(jobs, (id, acc, n) => { A[id] = acc; st.nTest = n; done++; progress("c", done, jobs.length, `${done} of ${jobs.length} model fits`); schedule(() => renderC(st)); });
  setBusy("c", false);
  if (ok) { progress("c", 1, 1, `Done: ${N} pairs per condition in ${((performance.now() - t0) / 1000).toFixed(1)} s`); renderC(st); }
}
function renderC(st) {
  const { A, N, nTest, model } = st, thr = +$("c-thr").value / 100;
  const gaps = [[], []];
  for (let k = 0; k < N; k++) {
    const b = 4 * k;
    if (A[b] === null || A[b + 1] === null || A[b + 2] === null || A[b + 3] === null) break;
    gaps[0].push(Math.round(Math.abs(A[b] - A[b + 1]) * nTest));
    gaps[1].push(Math.round(Math.abs(A[b + 2] - A[b + 3]) * nTest));
  }
  const n = gaps[0].length;
  const thrCases = Math.ceil(thr * nTest - 1e-9);
  const cond = [{ label: "D1 · same split", color: "var(--accent)" }, { label: "D2 · new split each arm", color: "var(--dt)" }];
  if (!n) { empty($("c-hist"), "Waiting for results"); empty($("c-run-ch"), "Waiting for results"); $("c-kpis").innerHTML = ""; $("c-insight").textContent = ""; return; }
  const gmax = Math.max(...gaps.flat(), thrCases), cats = []; for (let g = 0; g <= gmax; g++) cats.push(g);
  const hs = cond.map((c, i) => ({ ...c, vals: cats.map(g => gaps[i].filter(v => v === g).length / n) }));
  legend($("c-leg1"), hs); legend($("c-leg2"), cond);
  barChart($("c-hist"), { cats: cats.map(g => `${g} · ${(g / nTest * 100).toFixed(1)}`), series: hs, xl: "gap: test cases · percentage points", yl: "share of pairs", yf: v => Math.round(v * 100) + "%", tip: v => pct(v, 1), vline: thrCases, vlineLabel: `≥ ${thr * 100} pp`, label: "Distribution of A/A gaps" });
  const runS = cond.map((c, i) => {
    const pts = [], band = []; let hit = 0;
    gaps[i].forEach((g, k) => { if (g >= thrCases) hit++; const m = k + 1, p = hit / m; pts.push([m, p]); if (m >= 10) { const ci = E.zCI(hit, m); band.push([m, ci[0], ci[1]]); } });
    return { ...c, pts, band, endLabel: true };
  });
  lineChart($("c-run-ch"), { series: runS, x0: 1, x1: N, xInt: true, xl: "pairs so far", yl: "proportion", yf: v => Math.round(v * 100) + "%", label: "Running proportion of gaps over threshold" });
  const props = gaps.map(g => g.filter(v => v >= thrCases).length);
  $("c-kpis").innerHTML = cond.map((c, i) => {
    const ci = E.zCI(props[i], n);
    return `<div class="kpi" style="--c:${c.color}"><span class="name">${c.label}</span><span class="big">${pct(props[i] / n, 1)}</span><span class="sub">${props[i]} of ${n} pairs with gap ≥ ${thr * 100} pp</span><span class="sub">95% Z CI ${pct(ci[0], 1)} – ${pct(ci[1], 1)}</span></div>`;
  }).join("") + `<div class="kpi"><span class="name">Smallest gap that counts</span><span class="big">${thrCases} case${thrCases > 1 ? "s" : ""}</span><span class="sub">= ${(thrCases / nTest * 100).toFixed(2)} pp on ${nTest} test cases</span></div>`;
  const nm = MODELS[model].name.toLowerCase();
  $("c-insight").innerHTML = (model === "lr" && props[0] === 0
    ? `Logistic regression has no random seed, so on the same split the two copies always agree. With a new split for each arm, <b>${pct(props[1] / n, 1)}</b> of pairs still differed by at least ${thr * 100} pp.`
    : `Two identical ${nm} runs differed by at least ${thr * 100} pp in <b>${pct(props[0] / n, 1)}</b> of pairs on the same split, and in <b>${pct(props[1] / n, 1)}</b> when each arm got its own split. A one-run gap of that size can be pure noise.`);
}

// =====================================================================
// Test-set size sweep
// =====================================================================
bindRange("t-k", v => v);
$("t-stop").addEventListener("click", () => { Pool.stop(); setBusy("t", false); progress("t", 0, 1, "Stopped"); });
$("t-run").addEventListener("click", runT);
const FRACS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5];
async function runT() {
  if (!guard()) return;
  stopAll();
  const model = $("t-model").value, K = +$("t-k").value, nt = state.nTrees;
  const ss = seedList("size-splits", K * FRACS.length), ms = seedList("size-models", K * FRACS.length);
  const jobs = []; FRACS.forEach((f, fi) => { for (let j = 0; j < K; j++) jobs.push({ model, splitSeed: ss[fi * K + j], testFrac: f, modelSeed: ms[fi * K + j], nTrees: nt }); });
  const R = FRACS.map(() => []), NT = FRACS.map(f => nTestOf(state.ds.y.length, f));
  let done = 0; setBusy("t", true); markRun("t");
  const t0 = performance.now();
  const draw = () => renderT(model, R, NT, K);
  draw();
  const ok = await Pool.run(jobs, (id, acc, n) => { const fi = Math.floor(id / K); R[fi].push(acc); NT[fi] = n; done++; progress("t", done, jobs.length, `${done} of ${jobs.length} model fits`); schedule(draw); });
  setBusy("t", false);
  if (ok) { progress("t", 1, 1, `Done: ${FRACS.length} test sizes × ${K} splits in ${((performance.now() - t0) / 1000).toFixed(1)} s`); draw(); }
}
function renderT(model, R, NT, K) {
  const rows = FRACS.map((f, i) => ({ f, n: NT[i], v: R[i] })).filter(r => r.v.length >= 2);
  if (!rows.length) { empty($("t-sd"), "Waiting for results"); empty($("t-mean"), "Waiting for results"); $("t-table").innerHTML = ""; $("t-insight").textContent = ""; return; }
  const obs = rows.map(r => [r.n, E.sd(r.v)]), bin = rows.map(r => { const p = E.mean(r.v); return [r.n, Math.sqrt(p * (1 - p) / r.n)]; });
  const sd = [{ label: `Observed SD (${MODELS[model].name})`, color: MODELS[model].color, pts: obs, dots: true }, { label: "Binomial √(p(1−p)/n)", color: "var(--muted)", pts: bin, dots: true, dash: true }];
  legend($("t-leg1"), sd.map(s => ({ ...s, dash: s.dash })));
  const xmin = Math.min(...NT), xmax = Math.max(...NT);
  lineChart($("t-sd"), { series: sd, x0: xmin, x1: xmax, xl: "test-set size (cases)", yl: "SD (pp)", yf: v => (v * 100).toFixed(1), clamp: [0, 1], yRange: [0, Math.max(...obs.map(o => o[1]), ...bin.map(b => b[1])) * 1.15], tip: p => `n=${p[0]}, ${pp(p[1])}`, label: "Observed and Binomial SD against test-set size" });
  lineChart($("t-mean"), { series: [{ label: MODELS[model].name, color: MODELS[model].color, pts: rows.map(r => [r.n, E.mean(r.v)]), band: rows.map(r => { const c = E.tCI(r.v); return [r.n, c[0], c[1]]; }), dots: true }], x0: xmin, x1: xmax, xl: "test-set size (cases)", yl: "mean accuracy", yf: v => (v * 100).toFixed(1) + "%", tip: p => `n=${p[0]}, ${pct(p[1])}`, label: "Mean accuracy against test-set size" });
  $("t-table").innerHTML = `<tr><th>Test share</th><th>Test cases</th><th>Step 1/n</th><th>Splits</th><th>Mean</th><th>Observed SD</th><th>Binomial SD</th><th>Ratio</th></tr>` + rows.map((r, i) => {
    const p = E.mean(r.v), b = Math.sqrt(p * (1 - p) / r.n), s = E.sd(r.v);
    return `<tr><td>${Math.round(r.f * 100)}%</td><td>${r.n}</td><td>${(100 / r.n).toFixed(3)} pp</td><td>${r.v.length}</td><td>${pct(p)}</td><td>${pp(s)}</td><td>${pp(b)}</td><td>${b > 0 ? (s / b).toFixed(2) + "×" : "–"}</td></tr>`;
  }).join("");
  const a = rows[0], z = rows[rows.length - 1];
  $("t-insight").innerHTML = rows.length > 1 ? `Growing the test set from <b>${a.n}</b> to <b>${z.n}</b> cases changed the split-to-split SD from ${pp(E.sd(a.v))} to ${pp(E.sd(z.v))}, and each test case now counts ${(100 / z.n).toFixed(2)} pp instead of ${(100 / a.n).toFixed(2)} pp. Fewer training cases can lower the mean, so test size is a trade-off.` : "";
}

// ---------- start
renderDsInfo();
showTab(location.hash.slice(1));
Pool.init().then(() => { Pool.spawn(); runB(); });
})();
