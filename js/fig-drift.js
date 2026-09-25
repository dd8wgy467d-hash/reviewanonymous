/* N5 — A perfectly trained ρ = 1 recurrence still drifts (App. E, Figure 3).
   The exact rotation of R² by 2π/7 realizes ℤ_7 under the constant word. Its entries cos and sin are
   rounded once to each number format, then the recurrence runs in fp64 from h_0 = (1, 0). The rounded
   matrix is still a scaled rotation, so the run has a closed form:
   h_t = r^t (cos tθ̃, sin tθ̃), with r = ‖(c̃, s̃)‖ and θ̃ = atan2(s̃, c̃).
   Left: one running realization on the circle. Right: the whole length sweep for every format.
   Mirrors assets/drift_curves.py of the paper. */
(function () {
  "use strict";
  const NB = window.NB;
  const N = 7;
  const THETA = NB.TWO_PI / N;
  const BOUND = Math.PI / N;
  const T_MAX = 1e8;
  const LN_MAX = Math.log(Number.MAX_VALUE);
  const LN_MIN = Math.log(Number.MIN_VALUE);

  // Round-to-nearest-even to a binary format with p significant bits and smallest normal
  // exponent emin (subnormals included). |x| ≤ 1 here, so overflow never occurs.
  function roundTo(x, p, emin) {
    if (x === 0 || !isFinite(x)) return x;
    const ax = Math.abs(x);
    let e = Math.floor(Math.log2(ax));
    if (Math.pow(2, e) > ax) e--;
    else if (Math.pow(2, e + 1) <= ax) e++;
    const q = Math.pow(2, Math.max(e, emin) - p + 1);
    const v = x / q;
    let r = Math.round(v);
    if (Math.abs(v - Math.trunc(v)) === 0.5) r = 2 * Math.round(v / 2);
    return r * q;
  }

  const FORMATS = [
    { key: "fp32", label: "fp32", p: 24, emin: -126, color: "var(--s6)" },
    { key: "bf16", label: "bf16", p: 8, emin: -126, color: "var(--s5)" },
    { key: "fp16", label: "fp16", p: 11, emin: -14, color: "var(--s7)" },
    { key: "e4m3", label: "fp8 E4M3", p: 4, emin: -6, color: "var(--s3)" },
    { key: "e5m2", label: "fp8 E5M2", p: 3, emin: -14, color: "var(--s4)" },
  ];
  // stored weights, log radius, angle error per step, first wrong readout
  const ROWS = FORMATS.map((f) => {
    const c = roundTo(Math.cos(THETA), f.p, f.emin);
    const s = roundTo(Math.sin(THETA), f.p, f.emin);
    const lnr = 0.5 * Math.log1p(c * c + s * s - 1); // accurate for r ≈ 1
    const dth = Math.atan2(s, c) - THETA;
    const tFail = dth === 0 ? Infinity : Math.ceil(BOUND / Math.abs(dth));
    return { f, c, s, lnr, dth, tFail };
  });

  NB.register("fig-drift", function (root) {
    let fi = 1; // bf16: fails within a few hundred steps
    let t = 0;
    let speed = 3;

    // ---------------------------------------------------------------- controls
    const ctr = NB.controls(root);
    NB.segmented(ctr, {
      label: "Weights stored in",
      options: FORMATS.map((f, i) => ({ value: i, label: f.label })),
      value: fi,
      onChange: (v) => ((fi = Number(v)), draw()),
    });
    const ctr2 = NB.controls(root);
    const player = NB.player(
      ctr2,
      () => {
        t = Math.min(T_MAX, t + Math.round(speed));
        draw();
        return t < T_MAX;
      },
      { delay: 60, root },
    );
    NB.slider(ctr2, { label: "steps per frame", min: 1, max: 1e6, log: true, value: speed, format: (v) => NB.fmtInt(Math.round(v)), onInput: (v) => (speed = v) });
    // slow time down: the pause between two frames, from 30 ms to 2 s
    NB.slider(ctr2, { label: "time per frame", min: 30, max: 2000, log: true, value: 60, format: (v) => (v < 1000 ? `${Math.round(v)} ms` : `${(v / 1000).toFixed(1)} s`), onInput: (v) => player.setDelay(v) });
    NB.button(ctr2, "Skip to just before the first error", () => {
      const r = ROWS[fi];
      if (isFinite(r.tFail)) t = Math.max(0, r.tFail - Math.max(5, 5 * Math.round(speed)));
      draw();
    });
    NB.button(ctr2, "Reset t = 0", () => (player.stop(), (t = 0), draw()), { cls: "ghost" });

    // ---------------------------------------------------------------- panels
    const panels = NB.panels(root);
    const pC = NB.panel(panels, "One run: the state h<sub>t</sub> (dot) and where it should be, ι(t mod 7) (ring)");
    const W = 340;
    const csvg = NB.svgRoot(pC, W, 395);
    csvg.style.overflow = "hidden";
    const pS = NB.panel(panels, "Length sweep, every format: phase error Δθ<sub>t</sub> (top) and log ‖h<sub>t</sub>‖ (bottom). × first wrong readout", "grow");
    const pp = new NB.Plot(pS, { width: 480, height: 200, x: [1, T_MAX], xlog: true, y: [-1.35 * BOUND, 1.35 * BOUND], margin: { l: 52, b: 34, t: 10 } });
    const rp = new NB.Plot(pS, { width: 480, height: 170, x: [1, T_MAX], xlog: true, y: [-1, 1], margin: { l: 52, b: 34, t: 10 } });
    NB.legend(pS, FORMATS.map((f) => ({ label: f.label, color: f.color })));

    const stats = NB.h("div", { class: "nb-stats" });
    root.appendChild(stats);
    const sT = NB.stat(stats, "step t");
    const sQ = NB.stat(stats, "true state / readout");
    const sP = NB.stat(stats, "phase error Δθ<sub>t</sub> (fails at ±π/7)");
    const sR = NB.stat(stats, "‖h<sub>t</sub>‖ (1 for the exact rotation)");
    const sF = NB.stat(stats, "first wrong readout");

    // ---------------------------------------------------------------- the circle
    const C = W / 2;
    const S = 112; // px per unit
    const X = (r, a) => C + r * S * Math.cos(a);
    const Y = (r, a) => C - r * S * Math.sin(a);
    const gStatic = NB.s("g", {}, csvg);
    const gDyn = NB.s("g", {}, csvg);
    for (let k = 0; k < N; k++) {
      const a0 = (k - 0.5) * THETA;
      const a1 = (k + 0.5) * THETA;
      const R = 1.45;
      NB.s("path", { d: `M${C},${C} L${X(R, a0)},${Y(R, a0)} A${R * S},${R * S} 0 0 0 ${X(R, a1)},${Y(R, a1)} Z`, class: "cell", style: `--c:${NB.stateColor(k)}` }, gStatic);
      NB.s("line", { x1: C, y1: C, x2: X(R, a1), y2: Y(R, a1), class: "cell-edge" }, gStatic);
    }
    NB.s("circle", { cx: C, cy: C, r: S, class: "unit" }, gStatic);
    for (let k = 0; k < N; k++) {
      const a = k * THETA;
      const x = X(1, a);
      const y = Y(1, a);
      NB.s("path", { d: `M${x},${y - 5} L${x + 5},${y} L${x},${y + 5} L${x - 5},${y}Z`, class: "code" }, gStatic);
      NB.s("text", { x: X(1.27, a), y: Y(1.27, a) + 4, class: "cell-label", "text-anchor": "middle", text: String(k) }, gStatic);
    }
    // phase gauge under the circle
    const GY = 362;
    const GX0 = 40;
    const GX1 = W - 40;
    const gx = (v) => GX0 + ((v + BOUND) / (2 * BOUND)) * (GX1 - GX0);
    NB.s("line", { x1: GX0, x2: GX1, y1: GY, y2: GY, class: "baseline" }, gStatic);
    for (const [v, lab] of [[-BOUND, "−π/7"], [0, "0"], [BOUND, "+π/7"]]) {
      NB.s("line", { x1: gx(v), x2: gx(v), y1: GY - 6, y2: GY + 6, class: "baseline" }, gStatic);
      NB.s("text", { x: gx(v), y: GY + 20, class: "tick", "text-anchor": "middle", text: lab }, gStatic);
    }
    NB.s("text", { x: C, y: GY - 12, class: "svg-note", "text-anchor": "middle", text: "phase error Δθ_t" }, gStatic);

    // state at step t, in closed form (the angle is reduced exactly: tθ ≡ (t mod 7)θ)
    function stateAt(r, tt) {
      const q = tt % N;
      const err = tt * r.dth;
      const logn = tt * r.lnr;
      const slip = Math.round(err / THETA); // how many cells the state has slipped
      const read = (((q + slip) % N) + N) % N;
      return { q, err, logn, read, angle: q * THETA + err };
    }

    function drawCircle() {
      NB.clear(gDyn);
      const r = ROWS[fi];
      const st = stateAt(r, t);
      const ok = st.read === st.q;
      // squashed radius so that the dot stays in view: 1 + 0.35 tanh(log ‖h‖)
      const rr = 1 + 0.35 * Math.tanh(st.logn);
      NB.s("circle", { cx: X(1, st.q * THETA), cy: Y(1, st.q * THETA), r: 11, class: "drift-target" }, gDyn);
      NB.s("line", { x1: C, y1: C, x2: X(rr, st.angle), y2: Y(rr, st.angle), class: "drift-ray" }, gDyn);
      NB.s("circle", { cx: X(rr, st.angle), cy: Y(rr, st.angle), r: 7, class: "drift-state" + (ok ? "" : " bad"), style: `--c:${r.f.color}` }, gDyn);
      // gauge marker
      const inside = Math.abs(st.err) < BOUND;
      const v = Math.max(-1.08 * BOUND, Math.min(1.08 * BOUND, st.err));
      NB.s("circle", { cx: gx(v), cy: GY, r: 6, class: "drift-state" + (inside ? "" : " bad"), style: `--c:${r.f.color}` }, gDyn);
      return { st, ok };
    }

    // ---------------------------------------------------------------- the sweep
    const GRID = [];
    for (let i = 0; i <= 400; i++) {
      const tt = Math.round(Math.pow(T_MAX, i / 400));
      if (!GRID.length || tt > GRID[GRID.length - 1]) GRID.push(tt);
    }
    function series(r, val) {
      const pts = GRID.map((tt) => [tt, tt < r.tFail ? val(tt) : NaN]);
      if (r.tFail <= T_MAX) {
        const k = pts.findIndex((p) => p[0] >= r.tFail);
        pts.splice(k, 0, [r.tFail, val(r.tFail)]);
      }
      return pts;
    }
    for (const r of ROWS) {
      r.ph = series(r, (tt) => tt * r.dth);
      r.rd = series(r, (tt) => tt * r.lnr);
    }
    let m = 0;
    for (const r of ROWS) m = Math.max(m, Math.abs(r.lnr) * Math.min(r.tFail, T_MAX));
    rp.y = [-1.15 * m, 1.15 * m];
    const cross = (pl, x, y, col) => {
      const px = pl.sx(x);
      const py = pl.sy(y);
      NB.s("path", { d: `M${px - 5},${py - 5}L${px + 5},${py + 5}M${px - 5},${py + 5}L${px + 5},${py - 5}`, class: "series thick", style: `stroke:${col}` }, pl.gData);
    };

    function drawSweep() {
      pp.axes({ xlabel: "", ylabel: "Δθ_t", yticks: [-BOUND, 0, BOUND], yfmt: (v) => (v === 0 ? "0" : v > 0 ? "π/7" : "−π/7"), nx: 8 });
      const mt = Number(m.toPrecision(2));
      rp.axes({ xlabel: "step t (log)", ylabel: "log ‖h_t‖", yticks: [-mt, 0, mt], yfmt: (v) => String(v).replace("-", "−"), nx: 8 });
      for (const pl of [pp, rp]) NB.clear(pl.gData);
      for (const v of [BOUND, -BOUND]) NB.s("line", { x1: pp.x0, x2: pp.x1, y1: pp.sy(v), y2: pp.sy(v), class: "marker-line" }, pp.gData);
      ROWS.forEach((r, i) => {
        const cls = "series" + (i === fi ? " thick" : " faded");
        pp.line(r.ph, { class: cls, style: `stroke:${r.f.color}` });
        rp.line(r.rd, { class: cls, style: `stroke:${r.f.color}` });
        if (r.tFail <= T_MAX) {
          cross(pp, r.tFail, r.tFail * r.dth, r.f.color);
          cross(rp, r.tFail, r.tFail * r.lnr, r.f.color);
        }
      });
      // where the running realization is
      if (t >= 1) {
        const r = ROWS[fi];
        for (const [pl, y] of [[pp, t * r.dth], [rp, t * r.lnr]]) {
          NB.s("line", { x1: pl.sx(t), x2: pl.sx(t), y1: pl.y0, y2: pl.y1, class: "now-line" }, pl.gData);
          if (t < r.tFail) NB.s("circle", { cx: pl.sx(t), cy: pl.sy(y), r: 4.5, class: "hover-dot", style: `fill:${r.f.color}` }, pl.gData);
        }
      }
    }

    function draw() {
      const r = ROWS[fi];
      NB.live("fig-drift", { fmt: r.f.label, t });
      const { st, ok } = drawCircle();
      drawSweep();
      sT.set(t.toLocaleString("en-US"));
      sQ.set(`${st.q} / ${st.read}`, ok ? "good" : "bad");
      sP.set(`${st.err < 0 ? "−" : st.err > 0 ? "+" : ""}${NB.fmtSci(Math.abs(st.err))} rad`, Math.abs(st.err) < BOUND ? null : "bad");
      sR.set(st.logn > LN_MAX ? "overflow (> 10³⁰⁸)" : st.logn < LN_MIN ? "0 (underflow)" : NB.fmtSci(Math.exp(st.logn)));
      sF.set(isFinite(r.tFail) ? `t = ${r.tFail.toLocaleString("en-US")}` : "never", "bad");
    }

    pp.crosshair(
      () => ROWS.map((r) => ({ name: r.f.label, color: r.f.color, pts: r.ph })),
      (x) => `t ≈ ${NB.fmtInt(x)}`,
      (y) => (isFinite(y) ? NB.fmt(y, 4) + " rad" : "already failed"),
    );
    rp.crosshair(
      () => ROWS.map((r) => ({ name: r.f.label, color: r.f.color, pts: r.rd })),
      (x) => `t ≈ ${NB.fmtInt(x)}`,
      (y) => (isFinite(y) ? NB.fmtSci(y) : "already failed"),
    );
    draw();
  });
})();
