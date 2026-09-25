/* N2 — Realizing ℤ_5 with an affine RNN at finite precision (paper §2, Defs 2.1–2.2).
   h_t = ρ R(2π x_t / N) h_{t-1} + e_t,  ‖e_t‖ ≤ η (= η in the worst case);  ι(q) = (cos 2πq/N, sin 2πq/N);
   π(h) = the sector containing h, undefined near the origin. */
(function () {
  "use strict";
  const NB = window.NB;
  const M = 200; // random η-trajectories drawn
  const RMIN = 0.2; // π is undefined for ‖h‖ < RMIN

  NB.register("fig-dial", function (root) {
    const N = 5;
    let rho = 1.0;
    let eta = 0.03;
    let worst = false;
    let rand = NB.rng(7);
    let t, q, nominal, parts, reach;

    const ctr = NB.controls(root);
    NB.slider(ctr, { label: "spectral radius ρ", min: 0.9, max: 1, step: 0.001, value: rho, format: (v) => v.toFixed(3), onInput: (v) => ((rho = v), reset()) });
    NB.slider(ctr, { label: "perturbation η", min: 0, max: 0.1, step: 0.002, value: eta, format: (v) => v.toFixed(3), onInput: (v) => ((eta = v), reset()) });
    NB.worstToggle(ctr, (v) => ((worst = v), reset()));
    const symRow = NB.controls(root, "nb-symbols");

    const panels = NB.panels(root);
    const pDial = NB.panel(panels, "Hidden state space ℋ = ℝ², cells U<sub>q</sub> and code points ι(q)");
    const svg = NB.svgRoot(pDial, 340, 340);
    svg.style.overflow = "hidden"; // a large reachable disk must not spill over the controls
    NB.legend(pDial, [
      { label: "ι(q)", color: "var(--code)", shape: "diamond" },
      { label: "exact run", color: "var(--ink)", shape: "dot" },
      { label: "random η-trajectories", color: "var(--muted)", shape: "dot" },
      { label: "misread", color: "var(--bad)", shape: "dot" },
      { label: "all η-trajectories (U* of this word)", color: "var(--ink)", shape: "ring", dash: true },
    ]);
    const pPlot = NB.panel(panels, "Why the guarantee breaks: reachable radius vs. distance to the cell boundary", "grow");
    const plot = new NB.Plot(pPlot, { width: 460, height: 300, x: [0, 100], y: [0, 1.05], margin: { l: 46, b: 40 } });
    NB.legend(pPlot, [
      { label: "radius r<sub>t</sub> of the reachable disk", color: "var(--c-contract)" },
      { label: "distance from the exact run to its cell boundary", color: "var(--ink-2)", dash: true },
    ]);
    const stats = NB.h("div", { class: "nb-stats" });
    root.appendChild(stats);
    const sT = NB.stat(stats, "length t");
    const sQ = NB.stat(stats, "true state q<sub>t</sub>");
    const sWorst = NB.stat(stats, "(T1) at this length, all η-trajectories");
    const sRand = NB.stat(stats, "random η-trajectories read correctly");
    const sL = NB.stat(stats, "first failing length L*");

    const gCells = NB.s("g", {}, svg);
    const gReach = NB.s("g", {}, svg);
    const gParts = NB.s("g", {}, svg);
    const gTop = NB.s("g", {}, svg);
    const C = 170;
    const S = 120; // px per unit
    const X = (x) => C + x * S;
    const Y = (y) => C - y * S;

    function buildSymbols() {
      NB.clear(symRow);
      symRow.appendChild(NB.h("span", { class: "nb-group-label", text: "Read" }));
      for (let k = 1; k < N; k++) NB.button(symRow, "+" + k, () => (step(k), draw()), { cls: "sym" });
      NB.button(symRow, "+10 random", () => run(10));
      NB.button(symRow, "+100 random", () => run(100));
      player = NB.player(symRow, () => (step(NB.randInt(Math.random, N)), draw(), t < 5000), { delay: 60, root });
      NB.button(symRow, "Reset", reset, { cls: "ghost" });
    }
    let player;

    function run(k) {
      for (let i = 0; i < k; i++) step(NB.randInt(Math.random, N));
      draw();
    }

    function reset() {
      NB.live("fig-dial", { N, rho, eta, worst: worst ? "True" : "False" });
      if (player) player.stop();
      rand = NB.rng(7);
      t = 0;
      q = 0;
      nominal = [1, 0];
      reach = 0;
      parts = Array.from({ length: M }, () => [1, 0]);
      drawCells();
      draw();
    }

    function step(x) {
      const th = (NB.TWO_PI * x) / N;
      const c = rho * Math.cos(th);
      const s = rho * Math.sin(th);
      const rot = (h) => [c * h[0] - s * h[1], s * h[0] + c * h[1]];
      nominal = rot(nominal);
      for (let i = 0; i < M; i++) {
        const h = rot(parts[i]);
        const e = NB.perturb2(rand, eta, worst);
        parts[i] = [h[0] + e[0], h[1] + e[1]];
      }
      reach = rho * reach + eta; // ρR maps a disk of radius r onto a disk of radius ρr
      q = (q + x) % N;
      t++;
    }

    function decode(h) {
      if (Math.hypot(h[0], h[1]) < RMIN) return -1;
      let a = Math.atan2(h[1], h[0]);
      if (a < 0) a += NB.TWO_PI;
      return Math.round((a / NB.TWO_PI) * N) % N;
    }

    // Distance from the exact run (radius ρ^t, on the code ray) to the boundary of its cell.
    const margin = (tt) => {
      const r = Math.pow(rho, tt);
      return Math.max(0, Math.min(r * Math.sin(Math.PI / N), r - RMIN));
    };
    const radius = (tt) => (rho === 1 ? eta * tt : (eta * (1 - Math.pow(rho, tt))) / (1 - rho));
    function failingLength() {
      if (eta === 0) return rho === 1 ? Infinity : Math.ceil(Math.log(RMIN) / Math.log(rho));
      for (let tt = 1; tt < 1e6; tt++) if (radius(tt) >= margin(tt)) return tt;
      return Infinity;
    }

    function drawCells() {
      NB.clear(gCells);
      const R = 1.32;
      for (let k = 0; k < N; k++) {
        const a0 = (NB.TWO_PI * (k - 0.5)) / N;
        const a1 = (NB.TWO_PI * (k + 0.5)) / N;
        const large = a1 - a0 > Math.PI ? 1 : 0;
        const d =
          N === 1
            ? ""
            : `M${X(RMIN * Math.cos(a0))},${Y(RMIN * Math.sin(a0))} L${X(R * Math.cos(a0))},${Y(R * Math.sin(a0))} ` +
              `A${R * S},${R * S} 0 ${large} 0 ${X(R * Math.cos(a1))},${Y(R * Math.sin(a1))} ` +
              `L${X(RMIN * Math.cos(a1))},${Y(RMIN * Math.sin(a1))} A${RMIN * S},${RMIN * S} 0 ${large} 1 ${X(RMIN * Math.cos(a0))},${Y(RMIN * Math.sin(a0))}Z`;
        NB.s("path", { d, class: "cell", style: `--c:${NB.stateColor(k)}` }, gCells);
      }
      for (let k = 0; k < N; k++) {
        const a = (NB.TWO_PI * (k + 0.5)) / N;
        NB.s("line", { x1: X(RMIN * Math.cos(a)), y1: Y(RMIN * Math.sin(a)), x2: X(1.32 * Math.cos(a)), y2: Y(1.32 * Math.sin(a)), class: "cell-edge" }, gCells);
      }
      NB.s("circle", { cx: C, cy: C, r: RMIN * S, class: "deadzone" }, gCells);
      NB.s("text", { x: C, y: C + 4, class: "svg-note", "text-anchor": "middle", text: "π undef." }, gCells);
      NB.s("circle", { cx: C, cy: C, r: S, class: "unit" }, gCells);
      for (let k = 0; k < N; k++) {
        const a = (NB.TWO_PI * k) / N;
        const x = X(Math.cos(a));
        const y = Y(Math.sin(a));
        NB.s("path", { d: `M${x},${y - 6} L${x + 6},${y} L${x},${y + 6} L${x - 6},${y}Z`, class: "code" }, gCells);
        NB.s("text", { x: X(1.2 * Math.cos(a)), y: Y(1.2 * Math.sin(a)) + 5, class: "cell-label", "text-anchor": "middle", text: `U${sub(k)}` }, gCells);
      }
    }
    const sub = (k) => String(k).split("").map((c) => "₀₁₂₃₄₅₆₇₈₉"[c]).join("");

    function draw() {
      // reachable disk
      NB.clear(gReach);
      const rr = Math.min(reach, 3);
      NB.s("circle", { cx: X(nominal[0]), cy: Y(nominal[1]), r: Math.max(rr * S, 0.5), class: "reach" }, gReach);
      // particles
      NB.clear(gParts);
      let ok = 0;
      for (const h of parts) {
        const good = decode(h) === q;
        ok += good;
        NB.s("circle", { cx: X(Math.max(-1.4, Math.min(1.4, h[0]))), cy: Y(Math.max(-1.4, Math.min(1.4, h[1]))), r: 2.1, class: "pt " + (good ? "ok" : "bad") }, gParts);
      }
      NB.clear(gTop);
      NB.s("circle", { cx: X(nominal[0]), cy: Y(nominal[1]), r: 5.5, class: "nominal" }, gTop);

      // right plot
      const Ls = failingLength();
      const Tmax = Math.max(40, Math.min(3000, isFinite(Ls) ? Math.ceil(Ls * 1.6) : 200), t + 5);
      plot.x = [0, Tmax];
      plot.axes({ xlabel: "sequence length t", ylabel: "distance in ℋ", nx: 5 });
      NB.clear(plot.gData);
      const n = 240;
      const ptsM = [];
      const ptsR = [];
      for (let i = 0; i <= n; i++) {
        const tt = (Tmax * i) / n;
        ptsM.push([tt, margin(tt)]);
        ptsR.push([tt, radius(tt)]);
      }
      plot.line(ptsM, { class: "series dash", style: "stroke:var(--ink-2)" });
      plot.line(ptsR, { class: "series", style: "stroke:var(--c-contract)" });
      if (isFinite(Ls) && Ls <= Tmax) {
        NB.s("line", { x1: plot.sx(Ls), x2: plot.sx(Ls), y1: plot.y0, y2: plot.y1, class: "marker-line bad" }, plot.gData);
        NB.s("text", { x: plot.sx(Ls) + 5, y: plot.y1 + 12, class: "svg-note bad", text: `L* = ${Ls}` }, plot.gData);
      }
      NB.s("line", { x1: plot.sx(t), x2: plot.sx(t), y1: plot.y0, y2: plot.y1, class: "marker-line" }, plot.gData);
      NB.s("circle", { cx: plot.sx(t), cy: plot.sy(Math.min(radius(t), 1.05)), r: 4, class: "hover-dot", style: "fill:var(--c-contract)" }, plot.gData);

      sT.set(String(t));
      sQ.set(String(q));
      const worstOK = reach < margin(t);
      sWorst.set(worstOK ? "every η-trajectory is read correctly" : "some η-trajectory is misread", worstOK ? "good" : "bad");
      const frac = ok / M;
      sRand.set(`${Math.round(100 * frac)}%`, frac === 1 ? "good" : frac > 0.9 ? "warn" : "bad");
      sL.set(isFinite(Ls) ? String(Ls) : "none (η = 0 and ρ = 1)", isFinite(Ls) ? null : "good");
    }

    plot.crosshair(
      () => {
        const pts = (f) => Array.from({ length: 241 }, (_, i) => [(plot.x[1] * i) / 240, f((plot.x[1] * i) / 240)]);
        return [
          { name: "reachable radius r<sub>t</sub>", color: "var(--c-contract)", pts: pts(radius) },
          { name: "distance to boundary", color: "var(--ink-2)", pts: pts(margin) },
        ];
      },
      (x) => `t = ${Math.round(x)}`,
      (y) => y.toFixed(3),
    );

    buildSymbols();
    reset();
  });
})();
