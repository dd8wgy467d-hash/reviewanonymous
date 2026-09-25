/* N6 — Definite vs non-definite: the flip-flop (§4.1, Theorem 4.1(ii), Prop. F.13; §7, Table 1).
   Affine tracker:  set: h ← +1 + e,  reset: h ← −1 + e,  id: h ← κh + e,  |e| ≤ η,  π(h) = [h > 0].
   κ is the memory factor of the identity: the fraction of h an identity step keeps.
   Restoring tracker: the same step followed by a snap to the nearest code point ±1. */
(function () {
  "use strict";
  const NB = window.NB;
  const BATCH = 200;
  const ID = 0;
  const RESET = 1;
  const SET = 2;

  NB.register("fig-flipflop", function (root) {
    let dist = "DFF";
    let pid = 0.97;
    let kappa = 0.9;
    let eta = 0.05;
    let worst = false;
    let model = "affine";
    let seed = 3;
    let L = 320; // word length, also used for the batch accuracy

    const ctr = NB.controls(root);
    NB.segmented(ctr, {
      label: "Words drawn from",
      options: [
        { value: "DFF", label: "DFF<sub>5</sub>: ≤ 4 identities in a row", title: "definite words" },
        { value: "FF", label: "FF: identity runs of any length", title: "non-definite words" },
      ],
      value: dist,
      onChange: (v) => ((dist = v), (sP.el.style.display = v === "FF" ? "" : "none"), draw()),
    });
    const sP = NB.slider(ctr, { label: "P(identity)", min: 0.8, max: 0.995, step: 0.005, value: pid, format: (v) => v.toFixed(3), onInput: (v) => ((pid = v), draw()) });
    sP.el.style.display = "none";
    const ctr2 = NB.controls(root);
    NB.segmented(ctr2, {
      label: "Tracker",
      options: [
        { value: "affine", label: "affine (contracting identity)" },
        { value: "restore", label: "restoring (snap to ±1)" },
      ],
      value: model,
      onChange: (v) => ((model = v), draw()),
    });
    NB.slider(ctr2, { label: "memory factor κ (on id: h ← κ·h)", min: 0.5, max: 1, step: 0.005, value: kappa, format: (v) => v.toFixed(3), onInput: (v) => ((kappa = v), draw()) });
    NB.slider(ctr2, { label: "perturbation η", min: 0, max: 0.3, step: 0.005, value: eta, format: (v) => v.toFixed(3), onInput: (v) => ((eta = v), draw()) });
    NB.worstToggle(ctr2, (v) => ((worst = v), draw()));
    NB.slider(ctr2, { label: "word length L", min: 20, max: 2000, log: true, value: L, format: (v) => String(Math.round(v)), onInput: (v) => ((L = Math.round(v)), draw()) });
    NB.button(ctr2, "New words", () => (seed++, draw()), { cls: "ghost" });

    const panel = NB.panel(root, "One word of length <span class=\"n6-len\">320</span>: symbols (top) and the hidden state h<sub>t</sub> (bottom). <code>set</code> writes h = +1, <code>reset</code> writes h = −1, <code>id</code> keeps only the fraction κ of h; the readout is the sign of h.", "wide");
    const W = 760;
    const H = 290;
    const svg = NB.svgRoot(panel, W, H);
    NB.legend(panel, [
      { label: "set", color: NB.stateColor(1), shape: "bar" },
      { label: "reset", color: NB.stateColor(0), shape: "bar" },
      { label: "identity run longer than the window k*", color: "var(--bad)", shape: "bar" },
      { label: "hidden state h<sub>t</sub> (one η-trajectory)", color: "var(--c-contract)" },
      { label: "all η-trajectories (η-band)", color: "var(--muted)", shape: "bar" },
      { label: "… some of them misread", color: "var(--bad)", shape: "bar" },
      { label: "misread step of this η-trajectory", color: "var(--bad)", shape: "bar" },
    ]);
    const stats = NB.h("div", { class: "nb-stats" });
    root.appendChild(stats);
    const sK = NB.stat(stats, "undeclared context window k* (worst case)");
    const sRun = NB.stat(stats, "longest identity run in this word");
    const sErr = NB.stat(stats, "misread positions in this word");
    const sAcc = NB.stat(stats, `sequence accuracy over ${BATCH} words of length <span class="n6-len">320</span>`, "wide");

    function word(rand, n, d) {
      const w = [];
      let run = 0;
      for (let t = 0; t < n; t++) {
        let x;
        if (d === "DFF") {
          x = run >= 4 ? 1 + Math.floor(rand() * 2) : Math.floor(rand() * 3);
        } else x = rand() < pid ? ID : 1 + Math.floor(rand() * 2);
        run = x === ID ? run + 1 : 0;
        w.push(x);
      }
      return w;
    }
    function run(w, rand) {
      let h = -1;
      let q = 0;
      const hs = [];
      const qs = [];
      let errs = 0;
      for (const x of w) {
        const e = NB.perturb(rand, eta, worst);
        if (x === SET) {
          h = 1 + e;
          q = 1;
        } else if (x === RESET) {
          h = -1 + e;
          q = 0;
        } else h = kappa * h + e;
        const d = h > 0 ? 1 : 0;
        if (d !== q) errs++;
        if (model === "restore") h = d ? 1 : -1;
        hs.push(h);
        qs.push(q);
      }
      return { hs, qs, errs };
    }
    // Exact interval of h_t over all η-trajectories of the word (every map here is monotone).
    // For the restoring tracker, the interval is the step's image before the snap; the snap then
    // sends it to the code points on either side of 0 that it touches.
    function band(w) {
      let lo = -1;
      let hi = -1;
      let codes = [-1];
      const out = [];
      for (const x of w) {
        if (x === SET) (lo = 1 - eta), (hi = 1 + eta);
        else if (x === RESET) (lo = -1 - eta), (hi = -1 + eta);
        else if (model === "affine") (lo = kappa * lo - eta), (hi = kappa * hi + eta);
        else (lo = kappa * Math.min(...codes) - eta), (hi = kappa * Math.max(...codes) + eta);
        out.push([lo, hi]);
        if (model === "restore") codes = [...(lo <= 0 ? [-1] : []), ...(hi > 0 ? [1] : [])];
      }
      return out;
    }

    function kStar() {
      if (model === "restore") return eta < 1 ? Infinity : 0;
      if (eta === 0) return kappa > 0 ? Infinity : 0;
      for (let k = 0; k < 1e7; k++) {
        const geo = kappa === 1 ? k + 1 : (1 - Math.pow(kappa, k + 1)) / (1 - kappa);
        if (Math.pow(kappa, k) - eta * geo <= 0) return k;
      }
      return Infinity;
    }

    function draw() {
      NB.live("fig-flipflop", { L, kappa, eta, model, worst: worst ? "True" : "False", dist: dist === "DFF" ? "DFF5" : `FF, P(id) = ${pid}` });
      const rand = NB.rng(100 + seed);
      root.querySelectorAll(".n6-len").forEach((el) => (el.textContent = String(L)));
      const w = word(rand, L, dist);
      const r = run(w, rand);
      const ks = kStar();
      NB.clear(svg);
      const XL = 40;
      const XR = W - 10;
      const sx = (t) => XL + ((t + 0.5) / L) * (XR - XL);
      const YT = 70;
      const YB = H - 26;
      const sy = (h) => YT + ((1.6 - Math.max(-1.6, Math.min(1.6, h))) / 3.2) * (YB - YT);
      // tape
      NB.s("text", { x: 4, y: 26, class: "svg-note", text: "word" }, svg);
      let runStart = null;
      let longest = 0;
      for (let t = 0; t <= L; t++) {
        const x = t < L ? w[t] : null;
        if (x === ID) {
          if (runStart == null) runStart = t;
        } else if (runStart != null) {
          const len = t - runStart;
          longest = Math.max(longest, len);
          if (len > ks) NB.s("rect", { x: sx(runStart) - 1, y: 36, width: sx(t - 1) - sx(runStart) + 2, height: 5, rx: 2, class: "run-bad" }, svg);
          runStart = null;
        }
        if (x === SET) NB.s("line", { x1: sx(t), x2: sx(t), y1: 12, y2: 32, class: "tick-sym", style: `stroke:${NB.stateColor(1)}` }, svg);
        if (x === RESET) NB.s("line", { x1: sx(t), x2: sx(t), y1: 12, y2: 32, class: "tick-sym", style: `stroke:${NB.stateColor(0)}` }, svg);
      }
      // bands
      NB.s("rect", { x: XL, y: YT, width: XR - XL, height: sy(0) - YT, class: "band", style: `--c:${NB.stateColor(1)}` }, svg);
      NB.s("rect", { x: XL, y: sy(0), width: XR - XL, height: YB - sy(0), class: "band", style: `--c:${NB.stateColor(0)}` }, svg);
      NB.s("text", { x: XL - 6, y: sy(1) + 4, class: "cell-label", "text-anchor": "end", text: "U₁" }, svg);
      NB.s("text", { x: XL - 6, y: sy(-1) + 4, class: "cell-label", "text-anchor": "end", text: "U₀" }, svg);
      NB.s("line", { x1: XL, x2: XR, y1: sy(0), y2: sy(0), class: "band-edge" }, svg);
      // η-band: all η-trajectories, red where some of them are misread
      const bd = band(w);
      const up = bd.map((iv, t) => `${sx(t).toFixed(1)},${sy(iv[1]).toFixed(1)}`);
      const dn = bd.map((iv, t) => `${sx(t).toFixed(1)},${sy(iv[0]).toFixed(1)}`).reverse();
      NB.s("path", { d: "M" + up.join("L") + "L" + dn.join("L") + "Z", class: "tube" }, svg);
      const cw = (XR - XL) / L;
      bd.forEach(([lo, hi], t) => {
        if (r.qs[t] ? lo <= 0 : hi > 0) NB.s("rect", { x: sx(t) - cw / 2 - 0.3, y: sy(hi), width: cw + 0.6, height: Math.max(1, sy(lo) - sy(hi)), class: "tube-bad" }, svg);
      });
      // target
      let d = "";
      r.qs.forEach((q, t) => (d += (t ? "L" : "M") + sx(t).toFixed(1) + "," + sy(q ? 1 : -1).toFixed(1)));
      NB.s("path", { d, class: "target-line" }, svg);
      d = "";
      r.hs.forEach((h, t) => (d += (t ? "L" : "M") + sx(t).toFixed(1) + "," + sy(h).toFixed(1)));
      NB.s("path", { d, class: "series", style: `stroke:${model === "affine" ? "var(--c-contract)" : "var(--c-nfsm)"}` }, svg);
      // one tick per misread step, exactly one step wide, so that long words do not smear them
      const tw = Math.max(1, cw);
      r.hs.forEach((h, t) => {
        if ((h > 0 ? 1 : 0) !== r.qs[t]) NB.s("rect", { x: sx(t) - tw / 2, y: YB + 5, width: tw, height: 9, class: "misread-tick" }, svg);
      });
      NB.s("text", { x: XR, y: H - 2, class: "svg-note", "text-anchor": "end", text: "t →" }, svg);

      sK.set(isFinite(ks) ? `${ks} identities` : "none: the state is restored at every step", isFinite(ks) ? "warn" : "good");
      sRun.set(`${longest}` + (dist === "DFF" ? " (never more than 4)" : ""), longest > ks ? "bad" : "good");
      sErr.set(String(r.errs), r.errs ? "bad" : "good");

      // batch accuracy on both distributions (mirrors the DFF5 / FF columns of Table 1)
      const acc = {};
      for (const dd of ["DFF", "FF"]) {
        const br = NB.rng(9000 + seed);
        let ok = 0;
        for (let b = 0; b < BATCH; b++) {
          const ww = word(br, L, dd);
          if (run(ww, br).errs === 0) ok++;
        }
        acc[dd] = ok / BATCH;
      }
      const tag = (v) => `${Math.round(100 * v)}%`;
      sAcc.set(
        `DFF<sub>5</sub>: <b>${tag(acc.DFF)}</b> · FF (P(id) = ${pid.toFixed(3)}): <b>${tag(acc.FF)}</b>`,
        acc.DFF === 1 && acc.FF === 1 ? "good" : acc.DFF === 1 ? "warn" : "bad",
      );
    }

    draw();
  });
})();
