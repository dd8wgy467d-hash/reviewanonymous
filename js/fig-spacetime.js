/* N3 — Figure 1 of the paper, live (§3, Theorems 3.1–3.2).
   A 1-D caricature of ℋ split into five cells U_q (bands). One step maps the cell
   U_q onto U_{δ_x(q)} by  Φ_x(h) = c_{δ_x(q)} + λ (h − c_q),  then adds e ∈ [−η, η].
   λ = 1: executive organ only (a translation, nothing restores).
   λ < 1: the same map also restores (contracts the cell about its code point). */
(function () {
  "use strict";
  const NB = window.NB;
  const EDGES = [0.4, 1.2, 2.1, 3.1, 3.6, 4.4];
  const NQ = 5;
  const CEN = [0, 1, 2, 3, 4].map((k) => (EDGES[k] + EDGES[k + 1]) / 2);
  const HALF = [0, 1, 2, 3, 4].map((k) => (EDGES[k + 1] - EDGES[k]) / 2);
  const SYMS = [
    { label: "a", title: "advance: q → q+1", d: (q) => (q + 1) % NQ },
    { label: "b", title: "hold: q → q", d: (q) => q },
  ];
  const P = 28; // random trajectories
  const WIN = 26; // visible steps

  const cellOf = (h) => {
    if (h < EDGES[0] || h > EDGES[NQ]) return -1;
    for (let k = 0; k < NQ; k++) if (h <= EDGES[k + 1]) return k;
    return -1;
  };

  NB.register("fig-spacetime", function (root) {
    let lambda = 1;
    let eta = 0.06;
    let worst = true;
    let rand;
    let hist; // [{q, x, iv:[[lo,hi]...], escaped, parts:[h|NaN]}]

    const ctr = NB.controls(root);
    NB.slider(ctr, {
      label: "cell contraction λ",
      min: 0,
      max: 1,
      step: 0.01,
      value: lambda,
      format: (v) => (v === 1 ? "1 (none)" : v === 0 ? "0 (exact)" : v.toFixed(2)),
      onInput: (v) => ((lambda = v), replay()),
    });
    NB.slider(ctr, { label: "perturbation η", min: 0, max: 0.2, step: 0.005, value: eta, format: (v) => v.toFixed(3), onInput: (v) => ((eta = v), replay()) });
    NB.worstToggle(ctr, (v) => ((worst = v), replay()));
    const symRow = NB.controls(root, "nb-symbols");
    symRow.appendChild(NB.h("span", { class: "nb-group-label", text: "Read" }));
    SYMS.forEach((s, x) => NB.button(symRow, `${s.label} <small>${s.title}</small>`, () => (step(x), draw()), { cls: "sym" }));
    NB.button(symRow, "+1 random", () => (step(randSym()), draw()));
    const player = NB.player(symRow, () => (step(randSym()), draw(), hist.length < 400), { delay: 260, root });
    NB.button(symRow, "Reset", () => reset(), { cls: "ghost" });

    const panel = NB.panel(root, "Space-time: the run of the automaton (top) and the hidden state in ℋ, one column per step", "wide");
    const W = 760;
    const H = 360;
    const svg = NB.svgRoot(panel, W, H);
    NB.legend(panel, [
      { label: "exact run through the code points ι(q<sub>t</sub>)", color: "var(--ink)" },
      { label: "random η-trajectories", color: "var(--muted)" },
      { label: "all η-trajectories: the reachable set at step t", color: "var(--ink-2)", shape: "bar" },
      { label: "… straddling a wrong cell", color: "var(--bad)", shape: "bar" },
    ]);
    const XL = 64;
    const XR = W - 14;
    const YT = 58;
    const YB = H - 22;
    const HMIN = 0.2;
    const HMAX = 4.6;
    const sy = (h) => YB - ((h - HMIN) / (HMAX - HMIN)) * (YB - YT);
    const gBands = NB.s("g", {}, svg);
    const gData = NB.s("g", {}, svg);
    const gTop = NB.s("g", {}, svg);

    // static bands
    for (let k = 0; k < NQ; k++) {
      NB.s("rect", { x: XL, y: sy(EDGES[k + 1]), width: XR - XL, height: sy(EDGES[k]) - sy(EDGES[k + 1]), class: "band", style: `--c:${NB.stateColor(k)}` }, gBands);
      NB.subText(gBands, { x: XL - 8, y: sy(CEN[k]) + 4, class: "cell-label", "text-anchor": "end" }, "U", `q${k + 1}`);
    }
    for (const e of EDGES) NB.s("line", { x1: XL, x2: XR, y1: sy(e), y2: sy(e), class: "band-edge" }, gBands);
    NB.s("text", { x: 10, y: YT - 8, class: "svg-note", text: "ℋ" }, gBands);

    const randSym = () => (rand() < 0.7 ? 0 : 1);

    function reset() {
      player.stop();
      seedWord = [];
      replay();
    }
    let seedWord = [];

    // Recompute the whole history from the word (so sliders act on the same word).
    function replay() {
      NB.live("fig-spacetime", { lambda, eta, worst: worst ? "True" : "False" });
      rand = NB.rng(11);
      const word = seedWord.slice();
      hist = [{ q: 0, x: null, iv: [[CEN[0], CEN[0]]], escaped: false, parts: new Array(P).fill(CEN[0]) }];
      seedWord = [];
      for (const x of word) step(x);
      draw();
    }

    function step(x) {
      seedWord.push(x);
      const prev = hist[hist.length - 1];
      const q = SYMS[x].d(prev.q);
      // exact reachable set: split each interval at the cell edges, map each piece, fatten by η
      let escaped = prev.escaped;
      let pieces = [];
      for (const [lo, hi] of prev.iv) {
        if (lo < EDGES[0] || hi > EDGES[NQ]) escaped = true;
        for (let k = 0; k < NQ; k++) {
          const a = Math.max(lo, EDGES[k]);
          const b = Math.min(hi, EDGES[k + 1]);
          if (a > b) continue;
          const tgt = SYMS[x].d(k);
          pieces.push([CEN[tgt] + lambda * (a - CEN[k]) - eta, CEN[tgt] + lambda * (b - CEN[k]) + eta]);
        }
      }
      pieces.sort((u, v) => u[0] - v[0]);
      const iv = [];
      for (const p of pieces) {
        const last = iv[iv.length - 1];
        if (last && p[0] <= last[1] + 1e-12) last[1] = Math.max(last[1], p[1]);
        else iv.push(p.slice());
      }
      const parts = prev.parts.map((h) => {
        const k = cellOf(h);
        if (k < 0 || !isFinite(h)) return NaN;
        return CEN[SYMS[x].d(k)] + lambda * (h - CEN[k]) + NB.perturb(rand, eta, worst);
      });
      const bad = escaped || iv.some(([lo, hi]) => lo < EDGES[q] || hi > EDGES[q + 1]);
      hist.push({ q, x, iv, escaped, parts, bad });
    }

    function draw() {
      NB.clear(gData);
      NB.clear(gTop);
      const T = hist.length - 1;
      const t0 = Math.max(0, T - WIN + 1);
      const cols = WIN;
      const dx = (XR - XL - 20) / (cols - 1);
      const sx = (s) => XL + 10 + (s - t0) * dx;

      // top strip: automaton run
      for (let s = t0; s <= T; s++) {
        const q = hist[s].q;
        NB.s("rect", { x: sx(s) - 11, y: 10, width: 22, height: 20, rx: 4, class: "chip", style: `--c:${NB.stateColor(q)}` }, gTop);
        NB.s("text", { x: sx(s), y: 24, class: "chip-label", "text-anchor": "middle", text: `q${q + 1}` }, gTop);
        if (s > t0 || s > 0) {
          const x = hist[s].x;
          if (x != null) NB.s("text", { x: sx(s) - dx / 2, y: 44, class: "svg-note", "text-anchor": "middle", text: SYMS[x].label }, gTop);
        }
      }
      NB.s("text", { x: 10, y: 24, class: "svg-note", text: "run" }, gTop);

      // random trajectories
      for (let i = 0; i < P; i++) {
        let d = "";
        let pen = false;
        for (let s = t0; s <= T; s++) {
          const h = hist[s].parts[i];
          if (!isFinite(h)) {
            pen = false;
            continue;
          }
          const y = sy(Math.max(HMIN, Math.min(HMAX, h)));
          d += (pen ? "L" : "M") + sx(s).toFixed(1) + "," + y.toFixed(1);
          pen = true;
        }
        NB.s("path", { d, class: "traj" }, gData);
      }
      // exact run
      let d = "";
      for (let s = t0; s <= T; s++) d += (s === t0 ? "M" : "L") + sx(s) + "," + sy(CEN[hist[s].q]);
      NB.s("path", { d, class: "nominal-line" }, gData);
      // reachable sets
      for (let s = t0; s <= T; s++) {
        for (const [lo, hi] of hist[s].iv) {
          const a = Math.max(HMIN, lo);
          const b = Math.min(HMAX, hi);
          if (b < a) continue;
          NB.s("line", { x1: sx(s), x2: sx(s), y1: sy(a), y2: sy(b) - (b - a < 0.02 ? 2 : 0), class: "reach-bar" + (hist[s].bad ? " bad" : "") }, gData);
        }
        NB.s("circle", { cx: sx(s), cy: sy(CEN[hist[s].q]), r: 3.2, class: "codept" }, gData);
      }
      NB.s("text", { x: XR, y: H - 6, class: "svg-note", "text-anchor": "end", text: `t = ${t0} … ${T}` }, gData);

    }

    // initial word: the one of Figure 1 in spirit (a a b a a a …)
    rand = NB.rng(3);
    seedWord = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    replay();
  });
})();
