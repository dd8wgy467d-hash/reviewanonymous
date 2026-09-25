/* N9 — Heads as memory channels (§6; Prop. G.4, Example G.7; App. H.6.2, the S_3 model of seed 42).
   S_3 as three cups. Head 1 (3 indices) carries where ball 1 is; head 2 (2 indices) carries the
   parity of the arrangement. Together they determine it: c(j, k) is unique in S_3. */
(function () {
  "use strict";
  const NB = window.NB;

  NB.register("fig-heads", function (root) {
    let arr = [1, 2, 3]; // arr[p] = ball in cup p
    // the two heads run on their own tables (App. H.6.2); they never look at the cups
    const SIGMA1 = [[2, 1, 3], [1, 3, 2]]; // head 1, for x1 = (1 2) and x2 = (2 3)
    const SIGMA2 = [[2, 1], [2, 1]]; // head 2 flips at every symbol
    let j = 1; // head 1 index
    let kk = 1; // head 2 index
    let lastSwap = null;
    const history = [];

    const ctr = NB.controls(root, "nb-symbols");
    ctr.appendChild(NB.h("span", { class: "nb-group-label", text: "Read" }));
    NB.button(ctr, "x₁ = (1 2) <small>swap cups 1 and 2</small>", () => swap(0), { cls: "sym" });
    NB.button(ctr, "x₂ = (2 3) <small>swap cups 2 and 3</small>", () => swap(1), { cls: "sym" });
    NB.button(ctr, "+5 random", () => {
      for (let i = 0; i < 5; i++) swap(Math.random() < 0.5 ? 0 : 1, true);
      draw();
    });
    NB.button(ctr, "Reset", () => ((arr = [1, 2, 3]), (j = 1), (kk = 1), (lastSwap = null), (history.length = 0), draw()), { cls: "ghost" });

    const panels = NB.panels(root);
    const pC = NB.panel(panels, "The state: an arrangement of three balls (6 states)", "grow");
    const csvg = NB.svgRoot(pC, 420, 220);
    const pD = NB.panel(panels, "Reading the joint index back: c(j, k)");
    const dwrap = NB.h("div", { class: "nb-table-wrap" });
    pD.appendChild(dwrap);

    function swap(which, silent) {
      const a = which === 0 ? 0 : 1;
      [arr[a], arr[a + 1]] = [arr[a + 1], arr[a]];
      j = SIGMA1[which][j - 1];
      kk = SIGMA2[which][kk - 1];
      lastSwap = which;
      history.push(which);
      if (!silent) draw();
    }

    function draw() {
      NB.clear(csvg);
      // cups
      const cx = (p) => 60 + p * 80;
      for (let p = 0; p < 3; p++) {
        const hot = lastSwap != null && (p === lastSwap || p === lastSwap + 1);
        NB.s("path", { d: `M${cx(p) - 30},50 L${cx(p) - 22},110 L${cx(p) + 22},110 L${cx(p) + 30},50`, class: "cup" + (hot ? " hot" : "") }, csvg);
        NB.s("circle", { cx: cx(p), cy: 88, r: 17, class: "ball" + (arr[p] === 1 ? " one" : "") }, csvg);
        NB.s("text", { x: cx(p), y: 93, class: "ball-label" + (arr[p] === 1 ? " one" : ""), "text-anchor": "middle", text: String(arr[p]) }, csvg);
        NB.s("text", { x: cx(p), y: 130, class: "svg-note", "text-anchor": "middle", text: `cup ${p + 1}` }, csvg);
      }
      // heads
      const hx = 280;
      NB.s("text", { x: hx, y: 30, class: "svg-note strong", text: "head 1: where is ball 1?" }, csvg);
      for (let i = 1; i <= 3; i++) {
        NB.s("rect", { x: hx + (i - 1) * 40, y: 40, width: 32, height: 28, rx: 5, class: "idx" + (i === j ? " on" : "") }, csvg);
        NB.s("text", { x: hx + (i - 1) * 40 + 16, y: 59, class: "idx-label" + (i === j ? " on" : ""), "text-anchor": "middle", text: String(i) }, csvg);
      }
      NB.s("text", { x: hx, y: 98, class: "svg-note strong", text: "head 2: parity" }, csvg);
      ["even", "odd"].forEach((lab, i) => {
        NB.s("rect", { x: hx + i * 60, y: 108, width: 52, height: 28, rx: 5, class: "idx" + (i + 1 === kk ? " on" : "") }, csvg);
        NB.s("text", { x: hx + i * 60 + 26, y: 127, class: "idx-label" + (i + 1 === kk ? " on" : ""), "text-anchor": "middle", text: lab }, csvg);
      });
      NB.s("text", { x: 20, y: 170, class: "svg-note", text: "Head 1 applies each swap to its index: σ¹(1 2) = (2, 1, 3),  σ¹(2 3) = (1, 3, 2)." }, csvg);
      NB.s("text", { x: 20, y: 188, class: "svg-note", text: "Head 2 flips at every symbol: σ²(1 2) = σ²(2 3) = (2, 1).  These are the tables" }, csvg);
      NB.s("text", { x: 20, y: 206, class: "svg-note", text: "extracted from the trained S₃ model (App. H.6.2)." }, csvg);

      // decoder table
      NB.clear(dwrap);
      const tb = NB.h("table", { class: "nb-table compact" });
      tb.appendChild(NB.h("thead", {}, NB.h("tr", {}, NB.h("th", { text: "head 1  j" }), NB.h("th", { text: "head 2  k" }), NB.h("th", { text: "arrangement" }))));
      const body = NB.h("tbody");
      const all = [];
      const perms = [
        [1, 2, 3],
        [2, 1, 3],
        [2, 3, 1],
        [3, 2, 1],
        [3, 1, 2],
        [1, 3, 2],
      ];
      const par = (p) => {
        let inv = 0;
        for (let a = 0; a < 3; a++) for (let b = a + 1; b < 3; b++) inv += p[a] > p[b];
        return inv % 2;
      };
      for (const p of perms) all.push({ j: p.indexOf(1) + 1, k: par(p) + 1, p });
      all.sort((u, v) => u.j - v.j || u.k - v.k);
      for (const r of all) {
        const on = r.j === j && r.k === kk;
        body.appendChild(NB.h("tr", { class: on ? "on" : "" }, NB.h("td", { class: "num", text: String(r.j) }), NB.h("td", { text: r.k === 1 ? "even" : "odd" }), NB.h("td", { class: "mono", text: r.p.join(" ") })));
      }
      tb.appendChild(body);
      dwrap.appendChild(tb);
      dwrap.appendChild(NB.h("p", { class: "nb-caption", html: "Every one of the 3 × 2 joint indices is reached, and each names exactly one arrangement: the heads store the state without redundancy." }));
    }

    // budget calculator (Example G.7)
    const pB = NB.panel(root, "Why channels: tracking a running product in S<sub>n</sub> (Example G.7)", "wide");
    const bc = NB.controls(pB);
    let n = 5;
    NB.slider(bc, { label: "n", min: 3, max: 9, step: 1, value: n, onInput: (v) => ((n = v), budget()) });
    const bwrap = NB.h("div", { class: "nb-table-wrap" });
    pB.appendChild(bwrap);
    function fact(m) {
      let f = 1;
      for (let i = 2; i <= m; i++) f *= i;
      return f;
    }
    function budget() {
      NB.live("fig-heads", { n });
      const f = fact(n);
      const lg = (v) => Math.ceil(Math.log2(v));
      const rows = [
        [`one head of n! = ${f.toLocaleString("en-US")} indices`, (f * lg(f)).toLocaleString("en-US"), (f * f).toLocaleString("en-US")],
        [`n − 1 = ${n - 1} heads of n = ${n} indices (one per point), sharing one logit matrix`, ((n - 1) * n * lg(n)).toLocaleString("en-US"), (n * n).toLocaleString("en-US")],
      ];
      NB.clear(bwrap);
      const tb = NB.h("table", { class: "nb-table" });
      tb.appendChild(NB.h("thead", {}, NB.h("tr", {}, NB.h("th", { text: "Encoding of the state" }), NB.h("th", { text: "scan budget b (bits)" }), NB.h("th", { text: "logits per step" }))));
      const body = NB.h("tbody");
      rows.forEach((r, i) => body.appendChild(NB.h("tr", { class: i === 1 ? "on" : "" }, NB.h("td", { text: r[0] }), NB.h("td", { class: "num", text: r[1] }), NB.h("td", { class: "num", text: r[2] }))));
      tb.appendChild(body);
      bwrap.appendChild(tb);
    }
    budget();
    draw();
  });
})();
