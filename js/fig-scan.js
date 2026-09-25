/* N7 — Scan compatibility is a budget (§5: Def. 5.1, Prop. 5.2; App. D, Figure 2; Prop. G.1).
   (a) A parallel (Hillis–Steele) scan over transition tables: every merge is a gather, exact.
   (b) The same scan over affine pairs (A, b) in a low-precision format: the tree rounds in a
       different order than the sequential loop, and the two trajectories drift apart. */
(function () {
  "use strict";
  const NB = window.NB;

  // ---------------------------------------------------------------- (a) table scan
  NB.register("fig-scan-tree", function (root) {
    const L = 8;
    let taskKey = "Z";
    let task = NB.tasks.cyclic(5);
    let rounds = 3;
    let word = [];
    let rand = NB.rng(5);

    const ctr = NB.controls(root);
    NB.segmented(ctr, {
      label: "Task",
      options: [
        { value: "Z", label: "ℤ<sub>5</sub>" },
        { value: "S3", label: "S<sub>3</sub>" },
        { value: "FF", label: "flip-flop" },
      ],
      value: taskKey,
      onChange: (v) => {
        taskKey = v;
        task = v === "Z" ? NB.tasks.cyclic(5) : v === "S3" ? NB.tasks.s3() : NB.tasks.flipflop();
        newWord();
      },
    });
    const rs = NB.slider(ctr, { label: "rounds shown", min: 0, max: 3, step: 1, value: rounds, onInput: (v) => ((rounds = v), draw()) });
    NB.button(ctr, "New word", newWord, { cls: "ghost" });

    const panel = NB.panel(root, "Hillis–Steele scan over 8 transition tables. A box holds the composite table of the span [i..j]; a table lists δ(q) for q = 0, 1, …", "wide");
    const W = 780;
    const H = 380;
    const svg = NB.svgRoot(panel, W, H);
    const stats = NB.h("div", { class: "nb-stats" });
    root.appendChild(stats);
    const sDepth = NB.stat(stats, "depth");
    const sBud = NB.stat(stats, "budget per composite (Prop. G.1)");
    const sEq = NB.stat(stats, "scan vs. sequential run");

    function newWord() {
      NB.live("fig-scan-tree", { task: taskKey === "Z" ? "cyclic(N=5)" : taskKey === "S3" ? "s3()" : "flipflop()" });
      word = Array.from({ length: L }, () => NB.randInt(rand, task.symbols.length));
      draw();
    }

    function draw() {
      NB.clear(svg);
      // levels[r][i] = {tab, lo, hi}
      const levels = [word.map((x, i) => ({ tab: NB.table(task, x), lo: i, hi: i }))];
      for (let r = 1; r <= 3; r++) {
        const prev = levels[r - 1];
        const d = 1 << (r - 1);
        levels.push(prev.map((p, i) => (i >= d ? { tab: NB.compose(p.tab, prev[i - d].tab), lo: prev[i - d].lo, hi: p.hi, from: i - d } : { ...p, carried: true })));
      }
      const XL = 70;
      const dx = (W - XL - 20) / L;
      const bw = Math.min(84, dx - 8);
      const rowY = (r) => 60 + r * 72;
      const cx = (i) => XL + dx * (i + 0.5);
      const fmtTab = (tab) => tab.map((v) => (task.id === "S3" ? v : task.states[v])).join(" ");
      // symbol headers
      word.forEach((x, i) => NB.s("text", { x: cx(i), y: 22, class: "svg-note strong", "text-anchor": "middle", text: `x${sub(i + 1)} = ${task.symbols[x].label}` }, svg));
      for (let r = 0; r <= rounds; r++) {
        NB.s("text", { x: 8, y: rowY(r) + 4, class: "svg-note", text: r === 0 ? "tables" : `round ${r}` }, svg);
        levels[r].forEach((b, i) => {
          if (r > 0 && !b.carried) {
            NB.s("line", { x1: cx(b.from), y1: rowY(r - 1) + 14, x2: cx(i) - 6, y2: rowY(r) - 14, class: "scan-edge" }, svg);
            NB.s("line", { x1: cx(i), y1: rowY(r - 1) + 14, x2: cx(i), y2: rowY(r) - 14, class: "scan-edge" }, svg);
          } else if (r > 0) NB.s("line", { x1: cx(i), y1: rowY(r - 1) + 14, x2: cx(i), y2: rowY(r) - 14, class: "scan-edge carried" }, svg);
          const g = NB.s("g", { class: "scan-box" + (r === 3 || (r === rounds && b.lo === 0) ? " done" : "") }, svg);
          NB.s("rect", { x: cx(i) - bw / 2, y: rowY(r) - 14, width: bw, height: 28, rx: 5 }, g);
          NB.s("text", { x: cx(i), y: rowY(r) + 1, class: "scan-tab", "text-anchor": "middle", text: fmtTab(b.tab) }, g);
          NB.s("text", { x: cx(i), y: rowY(r) + 11, class: "scan-span", "text-anchor": "middle", text: b.lo === b.hi ? `[${b.lo + 1}]` : `[${b.lo + 1}..${b.hi + 1}]` }, g);
          g.appendChild(NB.s("title", { text: `composite of x${b.lo + 1} … x${b.hi + 1}: table (${fmtTab(b.tab)})` }));
        });
      }
      // states read off the prefixes vs the sequential run
      const yS = rowY(3) + 50;
      let q = task.start;
      let allEq = true;
      const final = levels[Math.min(rounds, 3)];
      NB.s("text", { x: 8, y: yS - 6, class: "svg-note", text: "scan q_t" }, svg);
      NB.s("text", { x: 8, y: yS + 18, class: "svg-note", text: "sequential" }, svg);
      word.forEach((x, i) => {
        q = task.delta(q, x);
        const complete = final[i].lo === 0;
        const qs = complete ? final[i].tab[task.start] : null;
        if (complete && qs !== q) allEq = false;
        NB.s("rect", { x: cx(i) - 18, y: yS - 20, width: 36, height: 20, rx: 4, class: "chip" + (complete ? "" : " pending"), style: `--c:${NB.stateColor(complete ? qs : 0)}` }, svg);
        NB.s("text", { x: cx(i), y: yS - 6, class: "chip-label", "text-anchor": "middle", text: complete ? task.states[qs] : "…" }, svg);
        NB.s("rect", { x: cx(i) - 18, y: yS + 4, width: 36, height: 20, rx: 4, class: "chip", style: `--c:${NB.stateColor(q)}` }, svg);
        NB.s("text", { x: cx(i), y: yS + 18, class: "chip-label", "text-anchor": "middle", text: task.states[q] }, svg);
      });
      const n = task.n;
      sDepth.set(`${rounds} of log₂ 8 = 3 rounds (a sequential loop needs 8)`);
      sBud.set(`b = n⌈log₂ n⌉ = ${n * Math.ceil(Math.log2(n))} bits, merge = ${n} lookups`);
      const done = final.filter((b) => b.lo === 0).length;
      sEq.set(done === L ? "identical at every position, whatever the tree" : `${done}/8 prefixes complete; the rest need more rounds`, done === L && allEq ? "good" : null);
    }
    const sub = (k) => String(k).split("").map((c) => "₀₁₂₃₄₅₆₇₈₉"[c]).join("");
    newWord();
  });

  // ---------------------------------------------------------------- (b) seq vs par
  NB.register("fig-scan-gap", function (root) {
    let fmt = "fp16";
    let logL = 12;
    let seed = 0;
    const CH = 8;
    const WORDS = 2;
    const LET = 5;

    const ctr = NB.controls(root);
    NB.segmented(ctr, {
      label: "Number format",
      options: ["fp32", "bf16", "fp16"].map((v) => ({ value: v, label: v })),
      value: fmt,
      onChange: (v) => ((fmt = v), schedule()),
    });
    NB.slider(ctr, { label: "length L", min: 8, max: 13, step: 1, value: logL, format: (v) => `2${NB.sup(v)} = ${NB.fmtInt(2 ** v)}`, onInput: (v) => ((logL = v), schedule()) });
    NB.button(ctr, "Resample model", () => (seed++, schedule()), { cls: "ghost" });

    const panel = NB.panel(root, "Gap ‖h<sub>t</sub><sup>par</sup> − h<sub>t</sub><sup>seq</sup>‖<sub>∞</sub> between the parallel scan and the sequential loop, same model, same word", "wide");
    const plot = new NB.Plot(panel, { width: 760, height: 280, x: [1, 4096], y: [1e-9, 1e3], ylog: true, margin: { l: 52, b: 40 } });
    NB.legend(panel, [
      { label: "affine, contracting (ρ<sub>x</sub> ∈ [0.9, 0.999])", color: "var(--c-contract)" },
      { label: "affine, isometric (ρ<sub>x</sub> = 1)", color: "var(--c-iso)" },
      { label: "NFSM tables: the gap is exactly 0 at every step", color: "var(--c-nfsm)" },
    ]);
    let res = null;

    function simulate() {
      const R = NB.formats[fmt].round;
      const L = 2 ** logL;
      const rand = NB.rng(77 + seed);
      const out = {};
      for (const kind of ["contract", "iso"]) {
        // per letter, per channel: A = ρ e^{iθ}, b complex
        const par = [];
        for (let x = 0; x < LET; x++) {
          const row = [];
          for (let c = 0; c < CH; c++) {
            const r = kind === "iso" ? 1 : 0.9 + 0.099 * rand();
            const th = NB.TWO_PI * rand();
            row.push([R(r * Math.cos(th)), R(r * Math.sin(th)), R(0.3 * NB.gauss(rand)), R(0.3 * NB.gauss(rand))]);
          }
          par.push(row);
        }
        const gap = new Float64Array(L);
        for (let wv = 0; wv < WORDS; wv++) {
          const w = Array.from({ length: L }, () => NB.randInt(rand, LET));
          for (let c = 0; c < CH; c++) {
            // sequential
            const seqRe = new Float64Array(L);
            const seqIm = new Float64Array(L);
            let hr = 0;
            let hi = 0;
            for (let t = 0; t < L; t++) {
              const [ar, ai, br, bi] = par[w[t]][c];
              const nr = R(R(R(ar * hr) - R(ai * hi)) + br);
              const ni = R(R(R(ar * hi) + R(ai * hr)) + bi);
              hr = nr;
              hi = ni;
              seqRe[t] = hr;
              seqIm[t] = hi;
            }
            // parallel (Hillis–Steele) over (A, b)
            let Ar = new Float64Array(L);
            let Ai = new Float64Array(L);
            let Br = new Float64Array(L);
            let Bi = new Float64Array(L);
            for (let t = 0; t < L; t++) [Ar[t], Ai[t], Br[t], Bi[t]] = par[w[t]][c];
            for (let d = 1; d < L; d <<= 1) {
              const nAr = Ar.slice();
              const nAi = Ai.slice();
              const nBr = Br.slice();
              const nBi = Bi.slice();
              for (let t = d; t < L; t++) {
                // (A2,b2)∘(A1,b1) = (A2 A1, A2 b1 + b2), with (A1,b1) the earlier span
                const a2r = Ar[t];
                const a2i = Ai[t];
                const a1r = Ar[t - d];
                const a1i = Ai[t - d];
                const b1r = Br[t - d];
                const b1i = Bi[t - d];
                nAr[t] = R(R(a2r * a1r) - R(a2i * a1i));
                nAi[t] = R(R(a2r * a1i) + R(a2i * a1r));
                nBr[t] = R(R(R(a2r * b1r) - R(a2i * b1i)) + Br[t]);
                nBi[t] = R(R(R(a2r * b1i) + R(a2i * b1r)) + Bi[t]);
              }
              Ar = nAr;
              Ai = nAi;
              Br = nBr;
              Bi = nBi;
            }
            for (let t = 0; t < L; t++) {
              const g = Math.max(Math.abs(Br[t] - seqRe[t]), Math.abs(Bi[t] - seqIm[t]));
              if (g > gap[t] || !isFinite(g)) gap[t] = isFinite(g) ? g : 1e3;
            }
          }
        }
        // block-average for plotting
        const pts = [];
        const nb = Math.min(300, L);
        for (let k = 0; k < nb; k++) {
          const a = Math.floor((k * L) / nb);
          const b = Math.max(a + 1, Math.floor(((k + 1) * L) / nb));
          let s = 0;
          for (let t = a; t < b; t++) s += gap[t];
          const m = s / (b - a);
          pts.push([(a + b) / 2 + 1, m > 0 ? m : NaN]);
        }
        out[kind] = pts;
      }
      out.L = L;
      return out;
    }

    function schedule() {
      NB.live("fig-scan-gap", { fmt, L: 2 ** logL });
      clearTimeout(schedule.t);
      root.classList.add("busy");
      schedule.t = setTimeout(() => {
        res = simulate();
        root.classList.remove("busy");
        draw();
      }, 40);
    }
    function draw() {
      plot.x = [1, res ? res.L : 4096];
      plot.axes({ xlabel: "step t", ylabel: "gap (log)", yticks: [1e-9, 1e-6, 1e-3, 1, 1e3], nx: 6 });
      NB.clear(plot.gData);
      if (!res) return;
      plot.line(res.contract, { class: "series", style: "stroke:var(--c-contract)" });
      plot.line(res.iso, { class: "series", style: "stroke:var(--c-iso)" });
      NB.s("line", { x1: plot.x0, x2: plot.x1, y1: plot.y0 - 3, y2: plot.y0 - 3, class: "series thick", style: "stroke:var(--c-nfsm)" }, plot.gData);
      NB.s("text", { x: plot.x1 - 4, y: plot.y0 - 9, class: "svg-note", "text-anchor": "end", text: "NFSM: 0 (log scale floor)" }, plot.gData);
    }
    plot.crosshair(
      () =>
        res
          ? [
              { name: "contracting", color: "var(--c-contract)", pts: res.contract },
              { name: "isometric", color: "var(--c-iso)", pts: res.iso },
            ]
          : [],
      (x) => `t ≈ ${NB.fmtInt(x)}`,
      (y) => (isFinite(y) ? NB.fmtSci(y) : "0"),
    );
    draw();
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((es) => {
        if (es.some((e) => e.isIntersecting)) {
          io.disconnect();
          schedule();
        }
      });
      io.observe(root);
    } else schedule();
  });
})();
