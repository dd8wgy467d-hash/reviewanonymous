/* N4 — Two rates in one map (§3 Theorem 3.3; §4.2 multistability; App. F.3).
   Target: a symbol e that must *hold* one bit (δ_e = id on {0,1}); every non-definite
   automaton contains such a non-constant idempotent (Lemma F.11). Codes ι(0) = −1, ι(1) = +1,
   readout π(h) = [h > 0]. We iterate the 1-D map Φ_e under perturbations |e_t| ≤ η. */
(function () {
  "use strict";
  const NB = window.NB;
  const LIM = 1.8;

  NB.register("fig-tworates", function (root) {
    let kind = "affine";
    let a = 0.95;
    let beta = 2.0;
    let eta = 0.05;
    let worst = true;
    let K = 150;
    let seed = 1;

    const maps = {
      affine: { f: (h) => a * h, df: () => a },
      tanh: { f: (h) => Math.tanh(beta * h) / Math.tanh(beta), df: (h) => (beta * (1 - Math.tanh(beta * h) ** 2)) / Math.tanh(beta) },
      exact: { f: (h) => (h > 0 ? 1 : h < 0 ? -1 : 0), df: (h) => (Math.abs(h) < 1e-9 ? Infinity : 0) },
    };

    const ctr = NB.controls(root);
    NB.segmented(ctr, {
      label: "Update map of the hold symbol",
      options: [
        { value: "affine", label: "affine  a·h" },
        { value: "tanh", label: "multistable  tanh(βh)/tanh β" },
        { value: "exact", label: "exactly restoring  argmax" },
      ],
      value: kind,
      onChange: (v) => ((kind = v), paramVis(), draw()),
    });
    const sA = NB.slider(ctr, { label: "slope a", min: 0.5, max: 1.05, step: 0.005, value: a, format: (v) => v.toFixed(3), onInput: (v) => ((a = v), draw()) });
    const sB = NB.slider(ctr, { label: "steepness β", min: 0.3, max: 6, step: 0.05, value: beta, format: (v) => v.toFixed(2), onInput: (v) => ((beta = v), draw()) });
    NB.slider(ctr, { label: "perturbation η", min: 0, max: 0.4, step: 0.005, value: eta, format: (v) => v.toFixed(3), onInput: (v) => ((eta = v), draw()) });
    NB.worstToggle(ctr, (v) => ((worst = v), draw()));
    NB.slider(ctr, { label: "steps k", min: 20, max: 600, step: 10, value: K, onInput: (v) => ((K = v), draw()) });
    NB.button(ctr, "New noise", () => (seed++, draw()), { cls: "ghost" });
    function paramVis() {
      sA.el.style.display = kind === "affine" ? "" : "none";
      sB.el.style.display = kind === "tanh" ? "" : "none";
    }

    const panels = NB.panels(root);
    const pMap = NB.panel(panels, "The map Φ<sub>e</sub> and its cobweb (first 30 steps)");
    const mp = new NB.Plot(pMap, { width: 320, height: 300, x: [-LIM, LIM], y: [-LIM, LIM], margin: { l: 40, b: 34, r: 10 } });
    const rp = new NB.Plot(pMap, { width: 320, height: 130, x: [-LIM, LIM], y: [0.01, 100], ylog: true, margin: { l: 40, b: 34, r: 10, t: 8 } });
    const pTime = NB.panel(panels, "Both runs of the word e<sup>k</sup>, from ι(0) and ι(1)", "grow");
    const tp = new NB.Plot(pTime, { width: 460, height: 300, x: [0, K], y: [-LIM, LIM], margin: { l: 40, b: 36 } });
    NB.legend(pTime, [
      { label: "run from ι(1) = +1", color: NB.stateColor(1) },
      { label: "run from ι(0) = −1", color: NB.stateColor(0) },
      { label: "all η-trajectories (worst case)", color: "var(--muted)", shape: "bar" },
    ]);
    function fixedPoints(m) {
      const out = [];
      const n = 2400;
      let prev = null;
      for (let i = 0; i <= n; i++) {
        const h = -3 + (6 * i) / n;
        const g = m.f(h) - h;
        if (prev != null && Math.sign(g) !== Math.sign(prev.g) && prev.g !== 0) {
          const hs = (prev.h + h) / 2;
          out.push(hs);
        } else if (g === 0) out.push(h);
        prev = { h, g };
      }
      // dedupe
      const pts = out.filter((v, i) => i === 0 || Math.abs(v - out[i - 1]) > 1e-3);
      return pts.map((h) => ({ h, stable: Math.abs(m.df(h)) < 1 }));
    }

    function draw() {
      NB.live("fig-tworates", { kind, a, beta, eta, K, worst: worst ? "True" : "False" });
      const m = maps[kind];
      const rand = NB.rng(1000 + seed);
      // runs
      const runs = [1, -1].map((h0) => {
        const hs = [h0];
        let h = h0;
        for (let k = 1; k <= K; k++) {
          h = m.f(h) + NB.perturb(rand, eta, worst);
          hs.push(h);
        }
        return hs;
      });
      // worst-case reachable interval from each code (all maps here are monotone)
      const tubes = [1, -1].map((h0) => {
        let lo = h0;
        let hi = h0;
        const out = [[lo, hi]];
        let fail = null;
        for (let k = 1; k <= K; k++) {
          lo = m.f(lo) - eta;
          hi = m.f(hi) + eta;
          out.push([lo, hi]);
          if (fail == null && (h0 > 0 ? lo <= 0 : hi >= 0)) fail = k;
        }
        return { out, fail };
      });

      // ---- map plot
      mp.axes({ xlabel: "h", ylabel: "Φₑ(h)", xticks: [-1, 0, 1], yticks: [-1, 0, 1], xgrid: true });
      NB.clear(mp.gData);
      NB.s("rect", { x: mp.sx(-LIM), y: mp.y1, width: mp.sx(0) - mp.sx(-LIM), height: mp.y0 - mp.y1, class: "band", style: `--c:${NB.stateColor(0)}` }, mp.gData);
      NB.s("rect", { x: mp.sx(0), y: mp.y1, width: mp.sx(LIM) - mp.sx(0), height: mp.y0 - mp.y1, class: "band", style: `--c:${NB.stateColor(1)}` }, mp.gData);
      mp.line([[-LIM, -LIM], [LIM, LIM]], { class: "series dash thin", style: "stroke:var(--muted)" });
      const pts = [];
      for (let i = 0; i <= 400; i++) {
        const h = -LIM + (2 * LIM * i) / 400;
        pts.push([h, m.f(h)]);
      }
      if (kind === "exact") {
        mp.line([[-LIM, -1], [0, -1]], { class: "series", style: "stroke:var(--ink)" });
        mp.line([[0, 1], [LIM, 1]], { class: "series", style: "stroke:var(--ink)" });
        mp.line([[0, -1], [0, 1]], { class: "series dash thin", style: "stroke:var(--ink)" });
      } else mp.line(pts, { class: "series", style: "stroke:var(--ink)" });
      [1, 0].forEach((ri) => {
        const hs = runs[ri];
        let d = [];
        for (let k = 0; k < Math.min(30, K); k++) {
          const h = clamp(hs[k]);
          const fh = clamp(m.f(hs[k]));
          d.push([h, k === 0 ? h : clamp(hs[k])], [h, fh], [clamp(hs[k + 1]), clamp(hs[k + 1])]);
        }
        mp.line(d, { class: "series thin", style: `stroke:${NB.stateColor(ri === 0 ? 1 : 0)}` });
      });
      for (const fp of fixedPoints(m)) {
        if (Math.abs(fp.h) > LIM) continue;
        NB.s("circle", { cx: mp.sx(fp.h), cy: mp.sy(fp.h), r: 4.5, class: fp.stable ? "fp stable" : "fp unstable" }, mp.gData);
      }

      // ---- rate plot
      rp.axes({ xlabel: "local rate |Φ′(h)|  (1 = dashed)", xticks: [-1, 0, 1], yticks: [0.01, 1, 100], yfmt: (v) => (v === 1 ? "1" : v < 1 ? "0.01" : "100") });
      NB.clear(rp.gData);
      rp.line([[-LIM, 1], [LIM, 1]], { class: "series dash thin", style: "stroke:var(--muted)" });
      const rpts = pts.map(([h]) => [h, Math.max(0.011, Math.min(90, Math.abs(m.df(h))))]);
      if (kind === "exact") {
        rp.line([[-LIM, 0.011], [-0.02, 0.011]], { class: "series", style: "stroke:var(--c-nfsm)" });
        rp.line([[0.02, 0.011], [LIM, 0.011]], { class: "series", style: "stroke:var(--c-nfsm)" });
        rp.line([[0, 0.011], [0, 90]], { class: "series", style: "stroke:var(--c-nfsm)" });
        NB.s("text", { x: rp.sx(0) + 6, y: rp.sy(40), class: "svg-note", text: "jump" }, rp.gData);
      } else rp.line(rpts, { class: "series", style: `stroke:${kind === "affine" ? "var(--c-contract)" : "var(--c-nfsm)"}` });

      // ---- time plot
      tp.x = [0, K];
      tp.axes({ xlabel: "step k (the word e^k)", ylabel: "h", yticks: [-1, 0, 1] });
      NB.clear(tp.gData);
      NB.s("rect", { x: tp.x0, y: tp.sy(0), width: tp.x1 - tp.x0, height: tp.y0 - tp.sy(0), class: "band", style: `--c:${NB.stateColor(0)}` }, tp.gData);
      NB.s("rect", { x: tp.x0, y: tp.y1, width: tp.x1 - tp.x0, height: tp.sy(0) - tp.y1, class: "band", style: `--c:${NB.stateColor(1)}` }, tp.gData);
      tubes.forEach((tb) => {
        const up = tb.out.map((iv, k) => [k, clamp(iv[1])]);
        const dn = tb.out.map((iv, k) => [k, clamp(iv[0])]).reverse();
        NB.s("path", { d: tp.pathD(up) + tp.pathD(dn).replace("M", "L") + "Z", class: "tube" }, tp.gData);
      });
      runs.forEach((hs, ri) => {
        tp.line(hs.map((h, k) => [k, clamp(h)]), { class: "series", style: `stroke:${NB.stateColor(ri === 0 ? 1 : 0)}` });
      });
      const fail = Math.min(tubes[0].fail || Infinity, tubes[1].fail || Infinity);
      if (isFinite(fail)) {
        NB.s("line", { x1: tp.sx(fail), x2: tp.sx(fail), y1: tp.y0, y2: tp.y1, class: "marker-line bad" }, tp.gData);
        NB.s("text", { x: tp.sx(fail) + 5, y: tp.y1 + 12, class: "svg-note bad", text: `worst case misreads at k = ${fail}` }, tp.gData);
      }
    }
    const clamp = (h) => Math.max(-LIM, Math.min(LIM, h));

    paramVis();
    draw();
  });
})();
