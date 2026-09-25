/* N8 — One NFSM head (§6, eq. (6); Props. G.1–G.3).
   Φ_x(h) = rd(θ_x rd(h)),  rd(h) = one-hot of argmax_j h_j.
   σ_x(k) = argmax_j (θ_x)_{jk}: the table is the column-wise argmax of the logits.
   Perturbations only reach the logits; a perturbation below half the smallest column
   margin γ leaves every table unchanged, at every length (Prop. G.3). */
(function () {
  "use strict";
  const NB = window.NB;

  NB.register("fig-nfsm", function (root) {
    let taskKey = "Z5";
    let task;
    let d;
    let theta; // theta[x][j][k]
    let cur = 0; // symbol shown
    let k = 0; // head index
    let q = 0; // target state
    let steps = 0;
    let errors = 0;
    let firstErr = null;
    let last = null; // details of the last step
    let delta = 0; // logit perturbation
    let worst = false; // perturbations of size exactly η
    let etaS = 0.3; // state perturbation (sup-norm)
    const rand = NB.rng(31);

    const ctr = NB.controls(root);
    NB.segmented(ctr, {
      label: "Target",
      options: [
        { value: "Z5", label: "ℤ<sub>5</sub> (d = 5)" },
        { value: "FF", label: "flip-flop (d = 2)" },
      ],
      value: taskKey,
      onChange: (v) => ((taskKey = v), setup("trained")),
    });
    NB.button(ctr, "Logits of a trained head", () => setup("trained"));
    NB.button(ctr, "Random logits (untrained)", () => setup("random"), { cls: "ghost" });

    const panels = NB.panels(root);
    const pH = NB.panel(panels, "Logits θ<sub>x</sub> (row j = to, column k = from). Click a cell to make it the largest of its column.");
    const tabs = NB.controls(pH, "nb-tabs");
    const hsvg = NB.svgRoot(pH, 330, 320);
    pH.appendChild(NB.h("p", { class: "nb-caption", html: "Solid outline: σ<sub>x</sub>(k), the largest entry of column k. Dashed: the target δ<sub>x</sub>(k), when they differ. γ: the column margin." }));
    const pR = NB.panel(panels, "One step of the head: restoring organ, then executive organ", "grow");
    const rsvg = NB.svgRoot(pR, 440, 250);
    const runRow = NB.controls(pR, "nb-symbols");
    const sl = NB.controls(pR);
    NB.slider(sl, { label: "state perturbation ‖e‖<sub>∞</sub>", min: 0, max: 0.6, step: 0.01, value: etaS, format: (v) => v.toFixed(2), onInput: (v) => ((etaS = v), NB.live("fig-nfsm", { eta_s: v })) });
    const sD = NB.slider(sl, { label: "logit perturbation |Δ<sub>jk</sub>|", min: 0, max: 3, step: 0.01, value: delta, format: (v) => v.toFixed(2), onInput: (v) => ((delta = v), paintStats(), NB.live("fig-nfsm", { delta: v })) });
    NB.worstToggle(sl, (v) => ((worst = v), NB.live("fig-nfsm", { worst: v ? "True" : "False" })));
    const stats = NB.h("div", { class: "nb-stats" });
    root.appendChild(stats);
    const sTab = NB.stat(stats, "table σ<sub>x</sub> vs target δ<sub>x</sub>", "wide");
    const sGam = NB.stat(stats, "smallest column margin γ (all symbols)");
    const sGuar = NB.stat(stats, "Prop. G.3: |Δ| &lt; γ/2?");
    const sRun = NB.stat(stats, "steps run / misread positions");

    function setup(mode) {
      task = taskKey === "Z5" ? NB.tasks.cyclic(5) : NB.tasks.flipflop();
      d = task.n;
      theta = task.symbols.map((_, x) => {
        const m = [];
        for (let j = 0; j < d; j++) {
          m.push([]);
          for (let kk = 0; kk < d; kk++) {
            const on = task.delta(kk, x) === j;
            m[j].push(mode === "trained" ? (on ? 2.2 : 0) + 0.7 * NB.gauss(rand) : 1.2 * NB.gauss(rand));
          }
        }
        if (mode === "trained") for (let kk = 0; kk < d; kk++) promote(m, task.delta(kk, x), kk, 0.15);
        return m;
      });
      cur = 0;
      resetRun();
      NB.clear(tabs);
      tabs.appendChild(NB.h("span", { class: "nb-group-label", text: "Symbol x" }));
      tabBtns = task.symbols.map((s, x) => NB.button(tabs, s.label, () => x < theta.length && ((cur = x), draw()), { cls: "tab" }));
      NB.clear(runRow);
      runRow.appendChild(NB.h("span", { class: "nb-group-label", text: "Step" }));
      task.symbols.forEach((s, x) => NB.button(runRow, s.label, () => x < theta.length && (stepHead(x), (cur = x), draw()), { cls: "sym" }));
      NB.button(runRow, "+1,000 random", () => bulk(1000));
      NB.button(runRow, "+100,000 random", () => bulk(100000));
      NB.button(runRow, "Reset run", () => (resetRun(), draw()), { cls: "ghost" });
      stepHead(0);
      draw();
    }
    let tabBtns = [];
    function promote(m, j, kk, gap) {
      let mx = -Infinity;
      for (let r = 0; r < d; r++) if (r !== j) mx = Math.max(mx, m[r][kk]);
      if (m[j][kk] < mx + gap) m[j][kk] = mx + gap + 0.4 * Math.abs(NB.gauss(rand));
    }
    const table = (m) => Array.from({ length: d }, (_, kk) => argmaxCol(m, kk));
    function argmaxCol(m, kk, noise) {
      let best = 0;
      let bv = -Infinity;
      for (let j = 0; j < d; j++) {
        const v = m[j][kk] + (noise ? noise() : 0);
        if (v > bv) {
          bv = v;
          best = j;
        }
      }
      return best;
    }
    function margins(m) {
      return Array.from({ length: d }, (_, kk) => {
        const s = argmaxCol(m, kk);
        let second = -Infinity;
        for (let j = 0; j < d; j++) if (j !== s) second = Math.max(second, m[j][kk]);
        return m[s][kk] - second;
      });
    }
    function resetRun() {
      k = task.start;
      q = task.start;
      steps = 0;
      errors = 0;
      firstErr = null;
      last = null;
    }
    const noise = () => NB.perturb(rand, delta, worst);
    function stepHead(x) {
      // restoring organ: the state is perturbed, rd reads its largest coordinate
      const h = Array.from({ length: d }, (_, j) => (j === k ? 1 : 0) + NB.perturb(rand, etaS, worst));
      const kr = h.indexOf(Math.max(...h));
      // executive organ: column kr of the (perturbed) logits, then rd
      const col = Array.from({ length: d }, (_, j) => theta[x][j][kr] + noise());
      const kn = col.indexOf(Math.max(...col));
      last = { x, h, kr, col, kn };
      k = kn;
      q = task.delta(q, x);
      steps++;
      if (k !== q) {
        errors++;
        if (firstErr == null) firstErr = steps;
      }
    }
    function bulk(n) {
      for (let i = 0; i < n; i++) stepHead(NB.randInt(rand, task.symbols.length));
      draw();
    }

    function drawHeat() {
      NB.clear(hsvg);
      const m = theta[cur];
      const cs = Math.min(56, 260 / d);
      const X0 = 40;
      const Y0 = 34;
      let lo = Infinity;
      let hi = -Infinity;
      for (const r of m) for (const v of r) (lo = Math.min(lo, v)), (hi = Math.max(hi, v));
      const tab = table(m);
      const mar = margins(m);
      for (let kk = 0; kk < d; kk++) NB.s("text", { x: X0 + kk * cs + cs / 2, y: Y0 - 8, class: "svg-note", "text-anchor": "middle", text: `k=${task.states[kk]}` }, hsvg);
      for (let j = 0; j < d; j++) {
        NB.s("text", { x: X0 - 6, y: Y0 + j * cs + cs / 2 + 4, class: "svg-note", "text-anchor": "end", text: `j=${task.states[j]}` }, hsvg);
        for (let kk = 0; kk < d; kk++) {
          const v = m[j][kk];
          const f = (v - lo) / (hi - lo || 1);
          const isMax = tab[kk] === j;
          const target = task.delta(kk, cur) === j;
          const g = NB.s("g", { class: "heat-cell", tabindex: 0, role: "button", "aria-label": `row ${j} column ${kk}: ${v.toFixed(2)}` }, hsvg);
          NB.s("rect", { x: X0 + kk * cs + 1, y: Y0 + j * cs + 1, width: cs - 2, height: cs - 2, rx: 4, class: "heat", style: `--f:${(0.08 + 0.85 * f).toFixed(3)}` }, g);
          if (isMax) NB.s("rect", { x: X0 + kk * cs + 2.5, y: Y0 + j * cs + 2.5, width: cs - 5, height: cs - 5, rx: 4, class: "argmax" + (target ? "" : " wrong") }, g);
          else if (target) NB.s("rect", { x: X0 + kk * cs + 4, y: Y0 + j * cs + 4, width: cs - 8, height: cs - 8, rx: 3, class: "target-cell" }, g);
          NB.s("text", { x: X0 + kk * cs + cs / 2, y: Y0 + j * cs + cs / 2 + 4, class: "heat-val" + (f > 0.55 ? " inv" : ""), "text-anchor": "middle", text: v.toFixed(1) }, g);
          const act = () => {
            promote(theta[cur], j, kk, 0.3);
            draw();
          };
          g.addEventListener("click", act);
          g.addEventListener("keydown", (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), act()));
        }
      }
      const yF = Y0 + d * cs + 16;
      for (let kk = 0; kk < d; kk++) {
        const ok = tab[kk] === task.delta(kk, cur);
        NB.s("text", { x: X0 + kk * cs + cs / 2, y: yF, class: "svg-note " + (ok ? "good" : "bad"), "text-anchor": "middle", text: ok ? "✓" : "✗" }, hsvg);
        NB.s("text", { x: X0 + kk * cs + cs / 2, y: yF + 14, class: "svg-note", "text-anchor": "middle", text: `γ=${mar[kk].toFixed(2)}` }, hsvg);
      }
    }

    function bars(g, x0, y0, vals, hl, label, sub) {
      const bw = Math.min(16, 90 / d);
      const hmax = 60;
      const vmin = Math.min(0, ...vals);
      const vmax = Math.max(1, ...vals);
      const sy = (v) => y0 + hmax - ((v - vmin) / (vmax - vmin)) * hmax;
      NB.s("line", { x1: x0 - 2, x2: x0 + d * (bw + 3), y1: sy(0), y2: sy(0), class: "baseline" }, g);
      vals.forEach((v, j) => {
        const top = Math.min(sy(v), sy(0));
        NB.s("rect", { x: x0 + j * (bw + 3), y: top, width: bw, height: Math.max(1, Math.abs(sy(v) - sy(0))), rx: 2, class: "bar" + (j === hl ? " hl" : "") }, g);
        NB.s("text", { x: x0 + j * (bw + 3) + bw / 2, y: y0 + hmax + 12, class: "svg-note tiny", "text-anchor": "middle", text: task.states[j] }, g);
      });
      NB.s("text", { x: x0, y: y0 - 18, class: "svg-note strong", text: label }, g);
      if (sub) NB.s("text", { x: x0, y: y0 - 6, class: "svg-note", text: sub }, g);
    }

    function drawRun() {
      NB.clear(rsvg);
      if (!last) {
        NB.s("text", { x: 20, y: 120, class: "svg-note", text: "Press a symbol below to run one step of the head." }, rsvg);
        return;
      }
      const g = NB.s("g", {}, rsvg);
      const colW = 106;
      bars(g, 10, 44, last.h, last.kr, "h + e", "perturbed state");
      bars(g, 10 + colW, 44, last.h.map((_, j) => (j === last.kr ? 1 : 0)), last.kr, "rd(h + e)", "restoring organ");
      bars(g, 10 + 2 * colW, 44, last.col, last.kn, `θ_x rd(·)`, `column ${task.states[last.kr]} of θ_${task.symbols[last.x].label}`);
      bars(g, 10 + 3 * colW, 44, last.col.map((_, j) => (j === last.kn ? 1 : 0)), last.kn, "rd(·) = h_t", "executive organ");
      for (let i = 0; i < 3; i++) NB.s("text", { x: 10 + (i + 1) * colW - 14, y: 84, class: "svg-note", text: "→" }, g);
      const ok = last.kn === q;
      NB.s("text", { x: 10, y: 196, class: "svg-note " + (ok ? "good" : "bad"), text: `index ${task.states[last.kn]}, target ${task.states[q]} ${ok ? "✓" : "✗"}` }, g);
      NB.s("text", { x: 10, y: 214, class: "svg-note", text: "The state is carried as an index; e is erased by the first rd as long as ‖e‖∞ < 1/2." }, g);
    }

    function paintStats() {
      const tab = table(theta[cur]);
      const tgt = NB.table(task, cur);
      const same = tab.every((v, i) => v === tgt[i]);
      sTab.set(`σ = (${tab.map((v) => task.states[v]).join(", ")})  vs  δ = (${tgt.map((v) => task.states[v]).join(", ")})`, same ? "good" : "bad");
      let allOk = true;
      let gam = Infinity;
      task.symbols.forEach((_, x) => {
        const t = table(theta[x]);
        if (!t.every((v, i) => v === task.delta(i, x))) allOk = false;
        gam = Math.min(gam, ...margins(theta[x]));
      });
      sGam.set(`${gam.toFixed(3)}${allOk ? "" : "  (some table is wrong: R3 fails)"}`, allOk ? "good" : "bad");
      sGuar.set(
        delta < gam / 2 ? `yes (${delta.toFixed(2)} &lt; ${(gam / 2).toFixed(2)}): tables unchanged at every length` : `no (${delta.toFixed(2)} ≥ ${(gam / 2).toFixed(2)}): a column may be reordered`,
        delta < gam / 2 ? "good" : "warn",
      );
      sRun.set(`${NB.fmtInt(steps)} / ${NB.fmtInt(errors)}` + (firstErr ? ` (first at step ${firstErr})` : ""), errors ? "bad" : "good");
    }

    function draw() {
      NB.live("fig-nfsm", { d, eta_s: etaS, delta, worst: worst ? "True" : "False" });
      tabBtns.forEach((b, x) => b.classList.toggle("on", x === cur));
      drawHeat();
      drawRun();
      paintStats();
    }
    setup("trained");
  });
})();
