/* N12: Streamed keyword spotting (§7.2, Figure 2A; App. I.1).
   Digit accuracy (%) of the trained models, median and range over three seeds, with the digit at the
   end of the stream (noise before it) or at the start (noise after it). Test noise filtered as in
   App. I.1. Values copied from the summary written by the script that draws Figure 2A. */
(function () {
  "use strict";
  const NB = window.NB;

  // noise duration in seconds (T - 101 frames at 100 frames per second)
  const T = [0.99, 2.99, 6.99, 14.99, 30.99, 62.99, 126.99, 254.99, 510.99, 1022.99, 2046.99, 3070.99, 4094.99, 8190.99];
  // each row: [median, min, max] at every T
  const P = (s) => s.split(";").map((r) => r.trim().split(/\s+/).map(Number));
  const MODELS = [
    {
      key: "nfsm", pattern: "nondef", label: "Neural-FSM", note: "restoring", color: "var(--m-nfsm)",
      before: P("92.8 92.5 93.3; 92.9 92.5 93.2; 93.0 92.5 93.2; 92.7 92.5 93.1; 92.7 92.2 93.1; 92.8 92.4 93.4; 92.8 92.5 93.2; 92.9 92.5 93.3; 92.7 92.4 93.3; 93.0 92.5 93.1; 92.9 92.5 93.6; 92.6 92.5 93.0; 92.9 92.3 93.1; 92.8 92.3 93.1"),
      after: P("92.8 92.6 93.2; 92.9 92.6 93.1; 92.7 92.7 93.3; 92.7 92.6 92.9; 92.6 92.5 93.1; 92.4 92.4 92.8; 92.6 92.1 92.9; 92.5 92.2 92.8; 92.6 92.0 92.9; 92.6 92.2 93.2; 92.8 91.9 93.2; 92.4 92.0 93.0; 92.7 92.1 92.8; 92.5 92.1 92.6"),
    },
    {
      key: "mingru", pattern: "kdef", label: "minGRU<sup>−</sup>", note: "ρ &lt; 1", color: "var(--m-mingru)",
      before: P("96.4 96.3 96.9; 96.5 96.2 96.7; 96.4 96.2 96.6; 96.5 96.2 96.5; 96.4 96.3 96.5; 96.4 96.1 96.5; 96.2 96.1 96.4; 96.3 95.9 96.5; 96.3 95.9 96.5; 96.3 95.9 96.4; 96.3 95.8 96.4; 96.3 95.7 96.4; 96.0 95.8 96.5; 96.0 95.7 96.6"),
      after: P("96.5 96.5 96.7; 96.6 96.4 97.0; 96.6 96.5 96.7; 96.5 96.4 96.6; 96.2 95.8 96.5; 95.0 94.0 96.3; 88.7 88.3 94.3; 75.0 53.6 85.2; 47.8 3.9 77.7; 20.5 0.0 65.7; 6.0 0.0 39.7; 2.4 0.0 22.5; 0.6 0.0 13.6; 0.0 0.0 7.4"),
    },
    {
      key: "mamba", pattern: "kdef", label: "Mamba<sup>−</sup>", note: "ρ &lt; 1", color: "var(--m-mamba)",
      before: P("96.3 95.5 96.9; 96.4 95.5 97.0; 96.2 95.4 97.0; 96.2 95.4 96.9; 96.3 95.3 96.9; 96.4 95.5 97.0; 96.3 95.6 96.9; 96.3 95.6 97.0; 96.4 95.4 96.9; 96.4 95.5 96.9; 96.3 95.5 97.0; 96.3 95.3 96.9; 96.3 95.3 97.0; 96.3 95.4 96.9"),
      after: P("96.3 95.5 96.8; 96.2 95.4 97.0; 96.2 95.1 96.8; 94.4 93.9 95.7; 86.1 10.1 89.8; 30.1 0.0 51.5; 3.2 0.0 17.4; 0.0 0.0 5.1; 0.0 0.0 0.6; 0.0 0.0 0.2; 0.0 0.0 0.3; 0.0 0.0 0.5; 0.1 0.0 0.2; 0.1 0.0 0.4"),
    },
    {
      key: "aussm", pattern: "collapse", label: "AUSSM", note: "ρ = 1", color: "var(--m-aussm)",
      before: P("39.7 37.1 46.7; 30.3 26.6 42.3; 24.3 14.9 35.7; 18.4 10.9 28.4; 15.0 10.8 24.3; 13.4 11.0 20.1; 10.7 10.4 16.5; 10.5 10.3 13.2; 9.9 9.7 11.6; 10.0 9.8 11.3; 10.4 9.4 10.5; 10.3 10.2 10.7; 10.2 9.7 10.4; 9.9 9.9 10.3"),
      after: P("42.8 36.6 43.2; 36.6 31.3 43.4; 30.3 27.2 43.6; 28.2 26.0 40.2; 24.6 22.4 37.3; 22.2 21.0 31.8; 19.8 18.2 26.4; 17.8 15.5 21.2; 15.6 13.1 18.1; 13.2 13.1 13.6; 11.6 11.3 12.8; 11.4 10.8 11.7; 10.9 10.6 11.3; 10.6 10.1 10.7"),
    },
    {
      key: "deltaproduct", pattern: "collapse", label: "DeltaProduct", note: "ρ = 1", color: "var(--m-deltaproduct)",
      before: P("96.7 96.5 96.8; 96.6 96.5 96.7; 96.4 96.3 96.6; 96.3 96.2 96.3; 95.9 95.9 96.0; 95.3 95.1 95.4; 93.9 92.9 94.0; 90.3 87.7 91.2; 62.6 59.4 70.1; 17.1 17.0 22.0; 15.1 10.5 15.4; 12.7 8.7 14.7; 13.8 8.3 13.8; 12.3 9.4 14.4"),
      after: P("96.7 96.4 96.9; 96.7 96.2 96.8; 96.6 96.2 96.8; 96.5 96.0 96.7; 95.9 95.3 96.4; 94.6 94.4 95.7; 92.2 91.9 94.3; 80.3 74.4 86.3; 56.7 55.6 57.5; 35.7 28.6 39.4; 20.2 19.0 26.9; 17.2 15.6 23.7; 15.5 14.0 20.6; 12.4 12.2 13.3"),
    },
  ];
  const XT = [1, 4, 16, 64, 256, 1024, 8192];
  const fmtS = (v) => (v < 60 ? Math.round(v) + " s" : v < 3600 ? Math.round(v / 60) + " min" : (v / 3600).toFixed(1) + " h");

  // The three patterns Theorem 4.1 predicts (Figure 2A, left), and what the data show for each.
  const PATTERNS = {
    nondef: {
      name: "non-definite",
      models: "Neural-FSM",
      why:
        "A restoring recurrence puts the state back on an attractor at every step, so the digit, once heard, is kept, and noise on either side changes nothing. Accuracy should stay at its trained level at every length.",
      seen: "The Neural-FSM stays between 92.4% and 93.0% from 1 s to 8192 s of noise, on both sides.",
    },
    kdef: {
      name: "k-definite",
      models: "Mamba<sup>−</sup>, minGRU<sup>−</sup> (ρ &lt; 1)",
      why:
        "A contracting recurrence depends only on roughly its last k inputs. Noise before the digit has faded by the time the digit arrives, so it is harmless. Noise after the digit, once longer than k, pushes the digit out of that window: the model has heard only noise and answers \"none\", so digit accuracy falls to 0, below chance.",
      seen: "Both stay near 96% for any amount of noise before the digit. With 8192 s of noise after it, Mamba<sup>−</sup> drops to 0.1% and minGRU<sup>−</sup> to 0.0%; Mamba<sup>−</sup> forgets sooner.",
    },
    collapse: {
      name: "collapsing",
      models: "DeltaProduct, AUSSM (ρ = 1)",
      why:
        "Without contraction, perturbations pile up in the state as the stream grows, wherever the digit sits, until the state no longer tells the digits apart. Accuracy falls toward chance on both sides, and only the total length of the stream matters.",
      seen: "DeltaProduct falls from 96.7% to 12.3% (noise before) and 12.4% (noise after) at 8192 s. AUSSM, which learns the task only partially, falls from 39.7% and 42.8% to 9.9% and 10.6%.",
    },
  };

  NB.register("fig-kws", function (root) {
    const shown = new Set(MODELS.map((m) => m.key));
    let pattern = "all";
    const ctr = NB.controls(root);
    NB.segmented(ctr, {
      label: "Compare with the theory",
      options: [
        { value: "all", label: "all models" },
        { value: "nondef", label: "non-definite" },
        { value: "kdef", label: "k-definite" },
        { value: "collapse", label: "collapsing" },
      ],
      value: pattern,
      onChange: (v) => ((pattern = v), draw()),
    });
    const body = NB.group(ctr, "Models");
    for (const m of MODELS) {
      const b = NB.h("button", {
        type: "button",
        class: "nb-seg-btn on",
        html: `<i class="dot" style="--sw:${m.color}"></i>${m.label} <small>${m.note}</small>`,
        on: {
          click: () => {
            shown.has(m.key) ? shown.delete(m.key) : shown.add(m.key);
            b.classList.toggle("on", shown.has(m.key));
            draw();
          },
        },
      });
      body.appendChild(b);
    }

    // A test stream to listen to: 16 s of noise, the word "four" at about 4.9 s.
    const pL = NB.panel(NB.panels(root), 'Listen: one 16 s test stream (answer: "four")', "grow");
    const audio = NB.h("audio", { controls: "", preload: "metadata", src: "assets/kws_example.wav", class: "nb-audio" });
    pL.appendChild(audio);
    const LW = 820, LX0 = 10, LX1 = 810, DUR = 16, D0 = 4.9, D1 = 5.9;
    const lsvg = NB.svgRoot(pL, LW, 44);
    const lx = (t) => LX0 + (t / DUR) * (LX1 - LX0);
    NB.s("rect", { x: LX0, y: 6, width: LX1 - LX0, height: 18, rx: 3, style: "fill:var(--panel);stroke:var(--axis)" }, lsvg);
    NB.s("rect", { x: lx(D0), y: 6, width: lx(D1) - lx(D0), height: 18, style: "fill:var(--m-mingru);opacity:0.35" }, lsvg);
    NB.s("text", { x: (lx(D0) + lx(D1)) / 2, y: 38, class: "svg-note", "text-anchor": "middle", text: "four" }, lsvg);
    NB.s("text", { x: lx(D0) / 2 + LX0 / 2, y: 19, class: "svg-note", "text-anchor": "middle", text: "noise" }, lsvg);
    NB.s("text", { x: (lx(D1) + LX1) / 2, y: 19, class: "svg-note", "text-anchor": "middle", text: "noise" }, lsvg);
    const head = NB.s("line", { x1: LX0, x2: LX0, y1: 2, y2: 28, style: "stroke:var(--ink);stroke-width:2" }, lsvg);
    const move = () => {
      const x = lx(Math.min(audio.currentTime || 0, DUR));
      head.setAttribute("x1", x);
      head.setAttribute("x2", x);
      if (!audio.paused) requestAnimationFrame(move);
    };
    audio.addEventListener("play", move);
    audio.addEventListener("seeked", move);
    lsvg.addEventListener("click", (e) => {
      const pt = lsvg.createSVGPoint();
      pt.x = e.clientX;
      pt.y = e.clientY;
      const loc = pt.matrixTransform(lsvg.getScreenCTM().inverse());
      audio.currentTime = Math.max(0, Math.min(DUR, ((loc.x - LX0) / (LX1 - LX0)) * DUR));
      move();
    });

    const pT = NB.panel(NB.panels(root), "Theory: the three patterns Theorem 4.1 predicts (shape only)", "grow");
    const tsvg = NB.svgRoot(pT, 820, 250);
    const panels = NB.panels(root);
    const pB = NB.panel(panels, "Data: noise before the digit", "grow");
    const pA = NB.panel(panels, "Data: noise after the digit", "grow");
    const plotB = new NB.Plot(pB, { width: 400, height: 260, x: [8191, 0.99], xlog: true, y: [0, 100], margin: { l: 44, r: 12 } });
    const plotA = new NB.Plot(pA, { width: 400, height: 260, x: [0.99, 8191], xlog: true, y: [0, 100], margin: { l: 44, r: 12 } });
    NB.legend(root, MODELS.map((m) => ({ label: m.label, color: m.color })));
    const explain = NB.h("div", { class: "nb-explain" });
    root.appendChild(explain);

    // schematic: x runs from t_before (left) through 0 (centre) to t_after (right)
    const X0 = 90, X1 = 800, C = (X0 + X1) / 2, HI = 40, CH = 168, FG = 196, BASE = 214, K = 110, ELL = 220;
    const TH = {
      nondef: { d: `M${X0},${HI} L${X1},${HI}`, cls: "" },
      kdef: { d: `M${X0},${HI - 4} L${C + K - 18},${HI - 4} C${C + K},${HI - 4} ${C + K},${FG} ${C + K + 18},${FG} L${X1},${FG}`, cls: "dash" },
      collapse: { d: `M${X0},${CH} L${C - ELL - 18},${CH} C${C - ELL},${CH} ${C - ELL},${HI + 4} ${C - ELL + 18},${HI + 4} L${C + ELL - 18},${HI + 4} C${C + ELL},${HI + 4} ${C + ELL},${CH} ${C + ELL + 18},${CH} L${X1},${CH}`, cls: "dot" },
    };
    function drawTheory() {
      NB.clear(tsvg);
      NB.s("line", { x1: X0, x2: X1, y1: BASE, y2: BASE, class: "baseline" }, tsvg);
      NB.s("line", { x1: C, x2: C, y1: 24, y2: BASE + 4, class: "baseline" }, tsvg);
      [["trained", HI], ["chance", CH], ["forgot", FG]].forEach(([t, y]) => NB.s("text", { x: X0 - 6, y: y + 4, class: "tick", "text-anchor": "end", text: t }, tsvg));
      NB.s("text", { x: C, y: BASE + 17, class: "tick", "text-anchor": "middle", text: "0" }, tsvg);
      NB.s("text", { x: (X0 + C) / 2, y: BASE + 30, class: "axis-label", "text-anchor": "middle", text: "← noise before" }, tsvg);
      NB.s("text", { x: (C + X1) / 2, y: BASE + 30, class: "axis-label", "text-anchor": "middle", text: "noise after →" }, tsvg);
      NB.s("text", { x: C + K, y: BASE + 17, class: "tick", "text-anchor": "middle", text: "k" }, tsvg);
      NB.s("text", { x: C - ELL, y: BASE + 17, class: "tick", "text-anchor": "middle", text: "ℓ" }, tsvg);
      NB.s("text", { x: C + ELL, y: BASE + 17, class: "tick", "text-anchor": "middle", text: "ℓ" }, tsvg);
      for (const [key, t] of Object.entries(TH)) {
        const on = pattern === "all" || pattern === key;
        NB.s("path", { d: t.d, fill: "none", class: "series " + t.cls + (pattern === key ? " thick" : ""), style: `stroke:var(--ink);opacity:${on ? 1 : 0.15}` }, tsvg);
      }
      const lab = { nondef: [X1, HI - 10, "end"], kdef: [X1, FG - 8, "end"], collapse: [X0 + 2, CH - 8, "start"] };
      for (const [key, [x, y, anc]] of Object.entries(lab)) {
        const on = pattern === "all" || pattern === key;
        NB.s("text", { x, y, class: "svg-note" + (pattern === key ? " strong" : ""), "text-anchor": anc, style: `opacity:${on ? 1 : 0.3}`, text: PATTERNS[key].name }, tsvg);
      }
    }

    function drawOne(plot, side) {
      NB.clear(plot.gData);
      plot.axes({ xticks: XT, xfmt: (v) => String(v), yticks: [0, 20, 40, 60, 80, 100], xlabel: `noise ${side} the digit (s, log scale)`, ylabel: "digit accuracy (%)" });
      for (const m of MODELS) {
        if (!shown.has(m.key)) continue;
        const op = pattern === "all" || pattern === m.pattern ? 1 : 0.12;
        const g = NB.s("g", { style: `opacity:${op}` }, plot.gData);
        const rows = m[side];
        const top = T.map((t, i) => [t, rows[i][2]]);
        const bot = T.map((t, i) => [t, rows[i][1]]).reverse();
        const d = plot.pathD(top) + "L" + plot.pathD(bot).slice(1) + "Z";
        NB.s("path", { d, class: "band", style: `fill:${m.color};opacity:var(--band-alpha);stroke:none` }, g);
        plot.line(T.map((t, i) => [t, rows[i][0]]), { class: "series", style: `stroke:${m.color}` }, g);
        T.forEach((t, i) => NB.s("circle", { cx: plot.sx(t), cy: plot.sy(rows[i][0]), r: 2.6, style: `fill:${m.color}` }, g));
      }
    }
    function series(side) {
      return MODELS.filter((m) => shown.has(m.key) && (pattern === "all" || pattern === m.pattern)).map((m) => ({ name: m.label, color: m.color, pts: T.map((t, i) => [t, m[side][i][0]]) }));
    }
    function drawExplain() {
      const keys = pattern === "all" ? Object.keys(PATTERNS) : [pattern];
      explain.innerHTML =
        '<table class="nb-table compact explain"><thead><tr><th>Pattern</th><th>Expected for</th><th>Why</th><th>What the data show</th></tr></thead><tbody>' +
        keys.map((k) => `<tr><td>${PATTERNS[k].name}</td><td>${PATTERNS[k].models}</td><td>${PATTERNS[k].why}</td><td>${PATTERNS[k].seen}</td></tr>`).join("") +
        "</tbody></table>";
    }
    function draw() {
      drawTheory();
      drawOne(plotB, "before");
      drawOne(plotA, "after");
      drawExplain();
    }
    draw();
    plotB.crosshair(() => series("before"), (x) => fmtS(x) + " of noise before", (y) => y.toFixed(1) + "%");
    plotA.crosshair(() => series("after"), (x) => fmtS(x) + " of noise after", (y) => y.toFixed(1) + "%");
  });
})();
