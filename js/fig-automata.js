/* N1 — State tracking as a semiautomaton (paper §2, "State tracking and automata"). */
(function () {
  "use strict";
  const NB = window.NB;

  NB.register("fig-automata", function (root) {
    const N = 5;
    let task = NB.tasks.cyclic(N);
    let q, prevQ, word, lastX, dw;

    const ctr = NB.controls(root);
    const taskSeg = NB.segmented(ctr, {
      label: "Task",
      options: [
        { value: "Z", label: "ℤ<sub>5</sub> (count mod 5)" },
        { value: "FF", label: "flip-flop" },
        { value: "S3", label: "S<sub>3</sub> (swap cups)" },
      ],
      value: "Z",
      onChange: () => rebuild(),
    });
    const symRow = NB.controls(root, "nb-symbols");

    const panels = NB.panels(root);
    const pGraph = NB.panel(panels, "The automaton: arrows show δ<sub>x</sub> for the last symbol read", "grow");
    const gsvg = NB.svgRoot(pGraph, 380, 300);
    const pMap = NB.panel(panels, "The word read so far, as a map δ<sub>w</sub> : Q → Q");
    const tape = NB.h("div", { class: "nb-tape" });
    pMap.appendChild(tape);
    const msvg = NB.svgRoot(pMap, 260, 290);
    const stats = NB.h("div", { class: "nb-stats" });
    root.appendChild(stats);
    const sLen = NB.stat(stats, "|w|");
    const sMap = NB.stat(stats, "δ<sub>w</sub> is");
    const sMon = NB.stat(stats, "transition monoid");
    const arrow = NB.arrowMarker(gsvg, "n1-arrow", "arrowhead");
    const arrowHot = NB.arrowMarker(gsvg, "n1-arrow-hot", "arrowhead hot");
    const arrowMap = NB.arrowMarker(msvg, "n1-arrow-map", "arrowhead");
    const arrowMapHot = NB.arrowMarker(msvg, "n1-arrow-map-hot", "arrowhead hot");

    function rebuild() {
      NB.live("fig-automata", { task: taskSeg.value === "Z" ? `cyclic(N=${N})` : taskSeg.value === "FF" ? "flipflop()" : "s3()" });
      const v = taskSeg.value;
      task = v === "Z" ? NB.tasks.cyclic(N) : v === "FF" ? NB.tasks.flipflop() : NB.tasks.s3();
      NB.clear(symRow);
      symRow.appendChild(NB.h("span", { class: "nb-group-label", text: "Read a symbol" }));
      task.symbols.forEach((s, x) => NB.button(symRow, s.label, () => step(x), { cls: "sym" }));
      NB.button(symRow, "+10 random", () => {
        for (let i = 0; i < 10; i++) step(Math.floor(Math.random() * task.symbols.length), true);
        draw();
      });
      NB.button(symRow, "Reset", reset, { cls: "ghost" });
      reset();
    }

    function reset() {
      q = task.start;
      prevQ = null;
      word = [];
      lastX = 0;
      dw = NB.identityTable(task.n);
      draw();
    }

    function step(x, silent) {
      if (x >= task.symbols.length) return;
      prevQ = q;
      q = task.delta(q, x);
      word.push(x);
      lastX = x;
      dw = NB.compose(NB.table(task, x), dw);
      if (!silent) draw();
    }

    function positions() {
      const n = task.n;
      const cx = 190;
      const cy = 150;
      if (n === 2) return [
        [110, 150],
        [270, 150],
      ];
      const R = 108;
      return Array.from({ length: n }, (_, i) => {
        // Z_N: state q at angle 2πq/N, counter-clockwise, as on the dial of N2.
        const a = task.id.startsWith("Z") ? (NB.TWO_PI * i) / n : -Math.PI / 2 + (NB.TWO_PI * i) / n;
        return [cx + R * Math.cos(a), cy - R * Math.sin(a) * (task.id.startsWith("Z") ? 1 : -1)];
      });
    }

    function edge(g, p0, p1, r, hot) {
      const dx = p1[0] - p0[0];
      const dy = p1[1] - p0[1];
      const L = Math.hypot(dx, dy);
      const ux = dx / L;
      const uy = dy / L;
      const bend = Math.min(0.22 * L, 34);
      const mx = (p0[0] + p1[0]) / 2 - uy * bend;
      const my = (p0[1] + p1[1]) / 2 + ux * bend;
      // start/end shortened along the direction to the control point
      const s = shorten(p0, [mx, my], r);
      const e = shorten(p1, [mx, my], r + 2);
      NB.s(
        "path",
        { d: `M${s[0]},${s[1]} Q${mx},${my} ${e[0]},${e[1]}`, class: "edge" + (hot ? " hot" : ""), "marker-end": hot ? arrowHot : arrow },
        g,
      );
    }
    function shorten(p, toward, r) {
      const dx = toward[0] - p[0];
      const dy = toward[1] - p[1];
      const L = Math.hypot(dx, dy) || 1;
      return [p[0] + (dx / L) * r, p[1] + (dy / L) * r];
    }
    function selfLoop(g, p, r, hot) {
      const cx = 190;
      const cy = 150;
      let ox = p[0] - cx;
      let oy = p[1] - cy;
      const L = Math.hypot(ox, oy);
      if (L < 1) {
        ox = 0;
        oy = -1;
      } else {
        ox /= L;
        oy /= L;
      }
      const lr = 11;
      const c = [p[0] + ox * (r + lr - 3), p[1] + oy * (r + lr - 3)];
      NB.s("circle", { cx: c[0], cy: c[1], r: lr, class: "edge loop" + (hot ? " hot" : "") }, g);
    }

    function draw() {
      NB.clear(gsvg.querySelector("g.content") || NB.s("g", { class: "content" }, gsvg));
      const g = gsvg.querySelector("g.content");
      const pos = positions();
      const r = task.n > 6 ? 16 : 19;
      const t = NB.table(task, lastX);
      for (let s = 0; s < task.n; s++) {
        const hot = prevQ === s && word.length > 0;
        if (t[s] === s) selfLoop(g, pos[s], r, hot);
        else edge(g, pos[s], pos[t[s]], r, hot);
      }
      for (let s = 0; s < task.n; s++) {
        const cur = s === q;
        NB.s("circle", { cx: pos[s][0], cy: pos[s][1], r, class: "node" + (cur ? " current" : ""), style: `--c:${NB.stateColor(s)}` }, g);
        NB.s(
          "text",
          { x: pos[s][0], y: pos[s][1] + 4.5, class: "node-label" + (cur ? " current" : ""), "text-anchor": "middle", text: task.states[s] },
          g,
        );
      }
      NB.s("text", { x: 8, y: 292, class: "svg-note", text: `x = ${task.symbols[lastX].label}` }, g);

      // tape
      NB.clear(tape);
      tape.appendChild(NB.h("span", { class: "tape-label", text: "w =" }));
      const shown = word.slice(-18);
      if (word.length > 18) tape.appendChild(NB.h("span", { class: "tape-more", text: `…${word.length - 18} more` }));
      if (!word.length) tape.appendChild(NB.h("span", { class: "tape-empty", text: "ε (empty word)" }));
      shown.forEach((x) => tape.appendChild(NB.h("span", { class: "tape-chip", text: task.symbols[x].label })));

      // map δ_w
      const mg = msvg.querySelector("g.content") || NB.s("g", { class: "content" }, msvg);
      NB.clear(mg);
      const n = task.n;
      const H = 290;
      const gap = Math.min(34, (H - 50) / n);
      const y0 = 30 + (H - 40 - gap * (n - 1)) / 2;
      const xl = 60;
      const xr = 200;
      NB.s("text", { x: xl, y: 16, class: "svg-note", "text-anchor": "middle", text: "q" }, mg);
      NB.s("text", { x: xr, y: 16, class: "svg-note", "text-anchor": "middle", text: "δ_w(q)" }, mg);
      for (let s = 0; s < n; s++) {
        const ya = y0 + s * gap;
        const yb = y0 + dw[s] * gap;
        const hot = s === task.start;
        NB.s(
          "line",
          { x1: xl + 12, y1: ya, x2: xr - 14, y2: yb, class: "edge" + (hot ? " hot" : ""), "marker-end": hot ? arrowMapHot : arrowMap },
          mg,
        );
      }
      for (let s = 0; s < n; s++) {
        for (const x of [xl, xr]) {
          const cur = x === xr && s === q;
          NB.s("circle", { cx: x, cy: y0 + s * gap, r: 10, class: "node small" + (cur ? " current" : ""), style: `--c:${NB.stateColor(s)}` }, mg);
          NB.s(
            "text",
            { x: x + (x === xl ? -16 : 16), y: y0 + s * gap + 4, class: "node-label small", "text-anchor": x === xl ? "end" : "start", text: task.states[s] },
            mg,
          );
        }
      }

      sLen.set(String(word.length));
      const perm = new Set(dw).size === n;
      const cst = NB.isConstant(dw);
      sMap.set(
        word.length === 0 ? "the identity (δ<sub>ε</sub>)" : cst ? "a constant map: the past is forgotten" : perm ? "a permutation of Q" : "neither constant nor a permutation",
      );
      sMon.set(`${task.monoidSize} elements (${task.kind})`);
    }

    rebuild();
  });
})();
