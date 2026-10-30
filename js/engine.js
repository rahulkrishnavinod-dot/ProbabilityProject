// ---------- Engine: seeded RNG, models, statistics (shared by page and tests) ----------
const Engine = (() => {
  // Mulberry32 PRNG: same seed -> same sequence
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function shuffle(arr, r) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(r() * (i + 1));
      const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
    return arr;
  }
  // Stratified split: same class mix in train and test
  function split(y, testFrac, seed) {
    const r = rng(seed);
    const nTest = Math.ceil(testFrac * y.length);
    const byClass = [[], []];
    y.forEach((c, i) => byClass[c].push(i));
    const te = [], tr = [];
    const t1 = Math.round(nTest * byClass[1].length / y.length);
    const want = [nTest - t1, t1];
    byClass.forEach((ids, c) => {
      shuffle(ids, r);
      ids.forEach((id, k) => (k < want[c] ? te : tr).push(id));
    });
    return { tr, te };
  }

  // ---- Logistic regression: standardize, then Newton's method with L2 (C = 1, like scikit-learn)
  function fitLR(X, y, idx) {
    const p = X[0].length, n = idx.length;
    const mu = new Float64Array(p), sd = new Float64Array(p);
    for (const i of idx) for (let j = 0; j < p; j++) mu[j] += X[i][j];
    for (let j = 0; j < p; j++) mu[j] /= n;
    for (const i of idx) for (let j = 0; j < p; j++) sd[j] += (X[i][j] - mu[j]) ** 2;
    for (let j = 0; j < p; j++) sd[j] = Math.sqrt(sd[j] / n) || 1;
    const Z = idx.map(i => { const z = new Float64Array(p + 1); z[0] = 1; for (let j = 0; j < p; j++) z[j + 1] = (X[i][j] - mu[j]) / sd[j]; return z; });
    const yy = idx.map(i => y[i]);
    const d = p + 1;
    let w = new Float64Array(d);
    for (let it = 0; it < 25; it++) {
      const g = new Float64Array(d), H = Array.from({ length: d }, () => new Float64Array(d));
      for (let k = 0; k < n; k++) {
        const z = Z[k]; let s = 0;
        for (let j = 0; j < d; j++) s += w[j] * z[j];
        const pr = 1 / (1 + Math.exp(-s)), e = pr - yy[k], v = pr * (1 - pr);
        for (let a = 0; a < d; a++) { g[a] += e * z[a]; const va = v * z[a]; for (let b = a; b < d; b++) H[a][b] += va * z[b]; }
      }
      for (let a = 0; a < d; a++) for (let b = 0; b < a; b++) H[a][b] = H[b][a];
      for (let j = 1; j < d; j++) { g[j] += w[j]; H[j][j] += 1; }   // penalty 0.5||w||^2, intercept free
      H[0][0] += 1e-8;
      const step = solve(H, g);
      let mx = 0;
      for (let j = 0; j < d; j++) { w[j] -= step[j]; mx = Math.max(mx, Math.abs(step[j])); }
      if (mx < 1e-7) break;
    }
    return { predict(row) { let s = w[0]; for (let j = 0; j < p; j++) s += w[j + 1] * (row[j] - mu[j]) / sd[j]; return s > 0 ? 1 : 0; } };
  }
  function solve(A, b) { // Gaussian elimination with partial pivoting
    const n = b.length, M = A.map((r, i) => [...r, b[i]]);
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      [M[c], M[piv]] = [M[piv], M[c]];
      for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; if (f) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    const x = new Float64Array(n);
    for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
    return x;
  }

  // ---- Decision tree (Gini, grown until pure). Features are tried in a seeded random order and
  // ties keep the first best, so the seed matters, as in scikit-learn.
  function buildTree(X, y, idx, maxF, r) {
    const p = X[0].length;
    const feats = Array.from({ length: p }, (_, j) => j);
    function node(ids) {
      let n1 = 0; for (const i of ids) n1 += y[i];
      const n = ids.length;
      if (n1 === 0 || n1 === n || n < 2) return { leaf: n1 / n };
      shuffle(feats, r);
      let best = null, bestImp = 2 * n1 * (n - n1) / n - 1e-12; // weighted Gini of parent
      let tried = 0;
      for (let fi = 0; fi < p && (tried < maxF || best === null); fi++) {
        const f = feats[fi]; tried++;
        const s = ids.slice().sort((a, b) => X[a][f] - X[b][f]);
        let l1 = 0;
        for (let k = 0; k < n - 1; k++) {
          l1 += y[s[k]];
          const v = X[s[k]][f], v2 = X[s[k + 1]][f];
          if (v === v2) continue;
          const nl = k + 1, nr = n - nl, r1 = n1 - l1;
          const imp = 2 * l1 * (nl - l1) / nl + 2 * r1 * (nr - r1) / nr;
          if (imp < bestImp) { bestImp = imp; best = { f, t: (v + v2) / 2, k: nl, s }; }
        }
      }
      if (!best) return { leaf: n1 / n };
      return { f: best.f, t: best.t, L: node(best.s.slice(0, best.k)), R: node(best.s.slice(best.k)) };
    }
    const root = node(idx);
    return row => { let nd = root; while (nd.leaf === undefined) nd = row[nd.f] <= nd.t ? nd.L : nd.R; return nd.leaf; };
  }
  function fitTree(X, y, idx, seed) {
    const t = buildTree(X, y, idx, X[0].length, rng(seed));
    return { predict: row => (t(row) > 0.5 ? 1 : 0) };
  }
  function fitForest(X, y, idx, seed, nTrees) {
    const r = rng(seed), p = X[0].length, maxF = Math.max(1, Math.floor(Math.sqrt(p)));
    const trees = [];
    for (let t = 0; t < nTrees; t++) {
      const boot = idx.map(() => idx[Math.floor(r() * idx.length)]);
      trees.push(buildTree(X, y, boot, maxF, r));
    }
    return { predict(row) { let s = 0; for (const t of trees) s += t(row); return s / trees.length > 0.5 ? 1 : 0; } };
  }
  function fit(model, X, y, idx, seed, nTrees) {
    if (model === "lr") return fitLR(X, y, idx);
    if (model === "tree") return fitTree(X, y, idx, seed);
    return fitForest(X, y, idx, seed, nTrees);
  }
  function accuracy(m, X, y, te) { let c = 0; for (const i of te) if (m.predict(X[i]) === y[i]) c++; return c / te.length; }

  // ---- Statistics
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  const sd = a => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1)); };
  function lgamma(x) {
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, t = x + 5.5; t -= (x + 0.5) * Math.log(t);
    let s = 1.000000000190015; for (const k of c) s += k / ++y;
    return -t + Math.log(2.5066282746310005 * s / x);
  }
  function betacf(a, b, x) {
    let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < 1e-300) d = 1e-300; d = 1 / d; let h = d;
    for (let m = 1; m <= 300; m++) {
      const m2 = 2 * m;
      let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d;
      const del = d * c; h *= del; if (Math.abs(del - 1) < 3e-14) break;
    }
    return h;
  }
  function ibeta(x, a, b) { // regularized incomplete beta I_x(a,b)
    if (x <= 0) return 0; if (x >= 1) return 1;
    const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
  }
  const tTwoSided = (t, df) => ibeta(df / (df + t * t), df / 2, 0.5);           // P(|T| > |t|)
  const fSf = (F, d1, d2) => (F <= 0 ? 1 : ibeta(d2 / (d2 + d1 * F), d2 / 2, d1 / 2)); // P(F > f)
  function tInv(p, df) { // t quantile for upper-tail two-sided p, via bisection
    let lo = 0, hi = 1000;
    for (let i = 0; i < 200; i++) { const m = (lo + hi) / 2; if (tTwoSided(m, df) > p) lo = m; else hi = m; }
    return (lo + hi) / 2;
  }
  function tCI(a, conf = 0.95) {
    const n = a.length, m = mean(a);
    if (n < 2) return [m, m, 0];
    const h = tInv(1 - conf, n - 1) * sd(a) / Math.sqrt(n);
    return [m - h, m + h, h];
  }
  function zCI(k, n) { const p = k / n, h = 1.959963984540054 * Math.sqrt(p * (1 - p) / n); return [Math.max(0, p - h), Math.min(1, p + h)]; }

  // CRD and RBD ANOVA from sums of squares. Y[j][i] = split j, model i
  function anova(Y) {
    const b = Y.length, t = Y[0].length, N = b * t;
    const g = Y.flat().reduce((s, v) => s + v, 0) / N;
    const colM = Array.from({ length: t }, (_, i) => Y.reduce((s, r) => s + r[i], 0) / b);
    const rowM = Y.map(r => r.reduce((s, v) => s + v, 0) / t);
    const sst = Y.flat().reduce((s, v) => s + (v - g) ** 2, 0);
    const ssm = b * colM.reduce((s, v) => s + (v - g) ** 2, 0);
    const sss = t * rowM.reduce((s, v) => s + (v - g) ** 2, 0);
    const dfm = t - 1, dfc = N - t, dfs = b - 1, dfr = (t - 1) * (b - 1);
    const mseC = (sst - ssm) / dfc, sse = sst - ssm - sss, mseR = sse / dfr;
    const msm = ssm / dfm, mss = sss / dfs;
    return {
      crd: { ssm, sse: sst - ssm, dfm, dfe: dfc, msm, mse: mseC, F: msm / mseC, p: fSf(msm / mseC, dfm, dfc) },
      rbd: { ssm, sss, sse, dfm, dfs, dfe: dfr, msm, mss, mse: mseR, F: msm / mseR, p: fSf(msm / mseR, dfm, dfr), Fs: mss / mseR, ps: fSf(mss / mseR, dfs, dfr) },
      re: 100 * mseC / mseR, eta: ssm / (ssm + sse), sst
    };
  }
  function paired(a, b, testFrac) {
    const d = a.map((v, i) => v - b[i]), n = d.length, m = mean(d), s = sd(d);
    const t = s > 0 ? m / (s / Math.sqrt(n)) : 0;
    const tq = tInv(0.05, n - 1);
    const se = s / Math.sqrt(n), cse = Math.sqrt((1 / n + testFrac / (1 - testFrac))) * s;
    return { m, t, p: s > 0 ? tTwoSided(t, n - 1) : 1, ci: [m - tq * se, m + tq * se], cci: [m - tq * cse, m + tq * cse] };
  }
  return { rng, split, fit, accuracy, mean, sd, tCI, zCI, anova, paired, tInv, tTwoSided, fSf };
})();
if (typeof module !== "undefined") module.exports = Engine;
