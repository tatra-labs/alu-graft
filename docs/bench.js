// alu-graft circuit engine: runs the grafted heads + neurons (logical units) on a token list.
// Used by index.html in the browser and by scripts/check_web.mjs under Node.
(function (root) {
  const INV = Array.from({ length: 64 }, (_, i) => Math.pow(1e6, -i / 64));

  function rng(s) {
    return () => { s |= 0; s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }

  // spaces attach to a following + - = (Qwen3's BPE does the same for these strings)
  function tokenize(s) {
    const out = [];
    for (let i = 0; i < s.length; i++) {
      if (s[i] === " " && i + 1 < s.length && "+-=".includes(s[i + 1])) { out.push(" " + s[i + 1]); i++; continue; }
      out.push(s[i]);
    }
    return out;
  }

  function prep(d) {
    const layers = [];
    for (let L = 0; L < 15; L++) {
      const lay = { L };
      const hs = d.heads[L];
      if (hs) {
        const kv = {};
        lay.heads = hs.map((h) => {
          const q = Object.entries(h.q).map(([w, v]) => [+w, Float64Array.from(v)]);
          if (!kv[h.kv]) kv[h.kv] = { k: Object.entries(h.k).map(([w, v]) => [+w, Float64Array.from(v)]),
            v: Object.entries(h.v).map(([vd, t]) => [+vd, Object.entries(t).map(([w, c]) => [+w, c])]) };
          return { kv: h.kv, q, o: Object.entries(h.o).map(([vd, t]) => [+vd, Object.entries(t).map(([w, c]) => [+w, c])]) };
        });
        lay.kv = kv; lay.wq = Float64Array.from(d.qk[L].q); lay.wk = Float64Array.from(d.qk[L].k);
      }
      const ns = d.neurons[L];
      if (ns) {
        lay.neurons = ns.map(([g, u, dn]) => ({
          gi: Int32Array.from(g.map((x) => x[0])), gc: Float64Array.from(g.map((x) => x[1])),
          ui: Int32Array.from(u.map((x) => x[0])), uc: Float64Array.from(u.map((x) => x[1])),
          di: Int32Array.from(dn.map((x) => x[0])), dc: Float64Array.from(dn.map((x) => x[1])) }));
      }
      layers.push(lay);
    }
    d.layers = layers;
    return d;
  }

  const silu = (x) => x / (1 + Math.exp(-x));

  function normVec(x, w) {
    let s = 0; for (let i = 0; i < 128; i++) s += x[i] * x[i];
    const r = 1 / Math.sqrt(s / 128 + 1e-6); const o = new Float64Array(128);
    for (let i = 0; i < 128; i++) o[i] = w[i] * x[i] * r;
    return o;
  }

  function rope(x, p) {
    const o = new Float64Array(128);
    for (let i = 0; i < 64; i++) {
      const a = p * INV[i], c = Math.cos(a), s = Math.sin(a);
      o[i] = x[i] * c - x[i + 64] * s; o[i + 64] = x[i + 64] * c + x[i] * s;
    }
    return o;
  }

  // Returns snapshots[layer][position] = Float64Array(K) of logical wire values after each layer.
  function forward(C, toks, jitter, seed) {
    const K = C.K, T = toks.length, rand = rng(seed);
    const h = toks.map((t) => { const v = new Float64Array(K); const row = C.emb[C.mini[t]] || {};
      for (const [w, x] of Object.entries(row)) v[+w] = x; return v; });
    const rho = (name) => { const r = C.ranges[name]; const lo = r[0] * jitter, hi = r[1] * jitter;
      return Math.exp(lo + (hi - lo) * rand()); };
    const snaps = [];
    for (const lay of C.layers) {
      if (lay.heads) {
        const rs = h.map(() => rho("in" + lay.L));
        const Ks = {}, Vs = {};
        for (const [slot, kv] of Object.entries(lay.kv)) {
          Ks[slot] = h.map((x, s) => { const k = new Float64Array(128);
            for (const [w, vec] of kv.k) { const a = rs[s] * x[w]; if (a) for (let i = 0; i < 128; i++) k[i] += a * vec[i]; }
            return rope(normVec(k, lay.wk), s); });
          Vs[slot] = h.map((x, s) => { const v = new Float64Array(128);
            for (const [vd, terms] of kv.v) { let a = 0; for (const [w, c] of terms) a += c * rs[s] * x[w]; v[vd] = a; }
            return v; });
        }
        const adds = h.map(() => new Float64Array(K));
        for (const hd of lay.heads) {
          for (let t = 0; t < T; t++) {
            const q = new Float64Array(128);
            for (const [w, vec] of hd.q) { const a = rs[t] * h[t][w]; if (a) for (let i = 0; i < 128; i++) q[i] += a * vec[i]; }
            const qr = rope(normVec(q, lay.wq), t);
            const sc = new Float64Array(t + 1); let m = -Infinity;
            for (let s = 0; s <= t; s++) { let d = 0; const k = Ks[hd.kv][s]; for (let i = 0; i < 128; i++) d += qr[i] * k[i];
              d /= Math.sqrt(128); sc[s] = d; if (d > m) m = d; }
            let z = 0; for (let s = 0; s <= t; s++) { sc[s] = Math.exp(sc[s] - m); z += sc[s]; }
            for (const [vd, terms] of hd.o) {
              let o = 0; for (let s = 0; s <= t; s++) o += sc[s] / z * Vs[hd.kv][s][vd];
              for (const [w, c] of terms) adds[t][w] += c * o;
            }
          }
        }
        for (let t = 0; t < T; t++) for (let i = 0; i < K; i++) h[t][i] += adds[t][i];
      }
      if (lay.neurons) {
        for (let t = 0; t < T; t++) {
          const r = rho("post" + lay.L), x = h[t], add = new Float64Array(K);
          for (const n of lay.neurons) {
            let g = 0, u = 0;
            for (let j = 0; j < n.gi.length; j++) g += n.gc[j] * r * x[n.gi[j]];
            for (let j = 0; j < n.ui.length; j++) u += n.uc[j] * r * x[n.ui[j]];
            const a = silu(g) * u;
            if (a === 0) continue;
            for (let j = 0; j < n.di.length; j++) add[n.di[j]] += n.dc[j] * a;
          }
          for (let i = 0; i < K; i++) x[i] += add[i];
        }
      }
      snaps.push(h.map((x) => Float64Array.from(x)));
    }
    return snaps;
  }

  const LABEL = (name) => name === "out_minus" ? "-" : name === "out_sminus" ? " -" : name === "out_space" ? " "
    : name === "out_end" ? "END" : name.slice(3);

  function readout(C, state) {
    let best = null;
    C.readouts.forEach(([p], i) => { const v = state[p]; if (v > 0.5 && (!best || v > best.v)) best = { name: C.readout_names[i], v }; });
    return best;
  }

  // Autoregressive run: the circuit appends the tokens it forces until END or silence.
  function solve(C, text, jitter = 1, seed = 1) {
    let toks = ["\n", ...tokenize(text)];
    const n0 = toks.length, steps = [];
    let snaps = null, first = null;
    for (let step = 0; step < 10; step++) {
      snaps = forward(C, toks, jitter, seed + step * 7919);
      if (!first) first = snaps;
      const f = readout(C, snaps[14][toks.length - 1]);
      if (!f) break;
      if (f.name === "out_end") { steps.push({ end: true, v: f.v }); break; }
      const tok = LABEL(f.name);
      steps.push({ tok, v: f.v });
      toks = [...toks, tok];
    }
    return { prompt: toks.slice(1, n0), steps, answer: steps.filter((s) => !s.end).map((s) => s.tok).join(""),
      ended: steps.length > 0 && steps[steps.length - 1].end, snaps: first, eqPos: n0 - 1 };
  }

  root.ALU = { prep, forward, readout, solve, tokenize };
})(typeof globalThis !== "undefined" ? globalThis : this);
