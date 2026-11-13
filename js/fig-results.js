/* N10: Table 1 and Table 5 (§7.1; App. H.4): failing length per task, model and seed.
   Values: a number is the failing length L_max; ">=X" means no failure up to X;
   "x" means the seed did not learn the task. Seeds 42..46. */
(function () {
  "use strict";
  const NB = window.NB;

  const TASKS = [
    { key: "Z2", label: "ℤ<sub>2</sub>", states: 2, group: "abelian" },
    { key: "Z16", label: "ℤ<sub>16</sub>", states: 16, group: "abelian" },
    { key: "S3", label: "S<sub>3</sub>", states: 6, group: "non-abelian" },
    { key: "S4", label: "S<sub>4</sub>", states: 24, group: "non-abelian" },
    { key: "A5", label: "A<sub>5</sub>", states: 60, group: "non-abelian, non-solvable" },
    { key: "M11", label: "M<sub>11</sub>", states: 7920, group: "non-abelian, non-solvable" },
    { key: "DFF5", label: "DFF<sub>5</sub>", states: 3, group: "flip-flop, definite words", definite: true },
    { key: "FF", label: "FF", states: 3, group: "flip-flop" },
    { key: "TSO3", label: "TSO<sub>3</sub>", states: 44, group: "textual" },
    { key: "TSO4", label: "TSO<sub>4</sub>", states: 202, group: "textual" },
    { key: "TSO5", label: "TSO<sub>5</sub>", states: 1132, group: "textual" },
  ];
  const MODELS = [
    { key: "nfsm", label: "Neural-FSM", sub: "1 layer (2 on TSO), restoring", color: "var(--m-nfsm)" },
    { key: "mamba", label: "Mamba<sup>−</sup>", sub: "4 × 256, affine, ρ &lt; 1", color: "var(--m-mamba)" },
    { key: "pdssm", label: "PD-SSM", sub: "4 × 256, affine, ρ &lt; 1", color: "var(--m-pdssm)" },
    { key: "mingru", label: "minGRU<sup>−</sup>", sub: "4 × 256, affine, ρ &lt; 1", color: "var(--m-mingru)" },
    { key: "aussm", label: "AUSSM", sub: "4 × 256, affine, ρ = 1", color: "var(--m-aussm)" },
    { key: "deltaproduct", label: "DeltaProduct", sub: "4 × 256, n<sub>h</sub> = 2, affine, ρ = 1", color: "var(--m-deltaproduct)" },
  ];
  const G = (s) => s.split(" ");
  // Table 5 of the paper, seeds 42 to 46
  const DATA = {
    Z2: { nfsm: G(">=1M >=1M >=1M >=1M >=1M"), mamba: G("4352 5176 326 655 >=512k"), pdssm: G("102 117 105 119 110"), mingru: G("4122 1211 2134 5448 951"), aussm: G("274 115 144 101 151"), deltaproduct: G("824 193 586 144 711") },
    Z16: { nfsm: G(">=1M >=1M >=1M >=1M >=1M"), mamba: G("78 91 83 87 x"), pdssm: G("98 117 94 98 97"), mingru: G("101 119 128 79 117"), aussm: G("110 108 145 135 142"), deltaproduct: G("96 88 85 92 97") },
    S3: { nfsm: G(">=1M >=1M >=1M >=1M >=1M"), mamba: G("155 163 186 197 200"), pdssm: G("130 89 133 139 149"), mingru: G("209 257 246 291 240"), aussm: G("102 168 128 216 100"), deltaproduct: G("240 319 815 159 205") },
    S4: { nfsm: G(">=1M >=1M >=1M >=1M >=1M"), mamba: G("103 x 113 92 107"), pdssm: G("113 144 101 131 171"), mingru: G("113 81 93 87 99"), aussm: G("103 83 83 89 98"), deltaproduct: G("400 384 214 222 257") },
    A5: { nfsm: G(">=1M >=1M >=1M >=1M >=1M"), mamba: G("x x x x x"), pdssm: G("187 179 288 189 141"), mingru: G("x x x x x"), aussm: G("81 82 81 82 86"), deltaproduct: G("242 252 194 197 232") },
    M11: { nfsm: G(">=32k >=32k >=32k >=32k >=32k"), mamba: G("x x x x x"), pdssm: G("x x x x x"), mingru: G("x x x x x"), aussm: G("x x x x x"), deltaproduct: G("x x x x x") },
    DFF5: { nfsm: G(">=1M >=1M >=1M >=1M >=1M"), mamba: G(">=1M >=1M >=1M >=512k >=512k"), pdssm: G(">=256k >=256k >=256k >=256k >=256k"), mingru: G(">=1M >=1M >=1M >=1M >=1M"), aussm: G("102 120 93 96 87"), deltaproduct: G("336 588 507 297 280") },
    FF: { nfsm: G(">=1M >=1M >=1M >=1M >=1M"), mamba: G("178 173 143 181 1738"), pdssm: G("192 105 133 115 169"), mingru: G("236 191 324 227 300"), aussm: G("70 119 148 116 146"), deltaproduct: G("5633 4461 28223 6145 31366") },
    TSO3: { nfsm: G(">=1M >=1M >=1M >=1M >=1M"), mamba: G("1583 x x 1420 x"), pdssm: G("358 575 363 286 503"), mingru: G("341 342 342 342 1459"), aussm: G("x x x x x"), deltaproduct: G("1212 1428 1332 1350 1537") },
    TSO4: { nfsm: G(">=1M >=1M >=1M >=1M >=1M"), mamba: G("x x x x x"), pdssm: G("x 1922 x x x"), mingru: G("x x x x x"), aussm: G("x x x x x"), deltaproduct: G("1988 1453 1463 1140 1573") },
    TSO5: { nfsm: G(">=128k >=128k >=128k >=128k >=128k"), mamba: G("x x x x x"), pdssm: G("x x x 321 x"), mingru: G("x x x x x"), aussm: G("x x x x x"), deltaproduct: G("1232 1368 1732 1324 1099") },
  };
  // medians exactly as reported in Table 1, in the order of MODELS
  const MED = {
    Z2: [">=1M", "4352", "110", "2134", "144", "586"],
    Z16: [">=1M", "85", "98", "117", "135", "92"],
    S3: [">=1M", "186", "133", "246", "128", "240"],
    S4: [">=1M", "105", "131", "93", "89", "257"],
    A5: [">=1M", "x", "187", "x", "82", "232"],
    M11: [">=32k", "x", "x", "x", "x", "x"],
    DFF5: [">=1M", ">=1M", ">=256k", ">=1M", "96", "336"],
    FF: [">=1M", "178", "133", "236", "119", "6145"],
    TSO3: [">=1M", "1501", "363", "342", "x", "1350"],
    TSO4: [">=1M", "x", "1922", "x", "x", "1463"],
    TSO5: [">=128k", "x", "321", "x", "x", "1324"],
  };
  const CERT = new Set(["Z2", "Z16", "S3", "S4", "A5", "M11", "DFF5", "FF"]); // tables extracted and matched

  function parse(v) {
    if (v === "x") return { x: true };
    const ge = v.startsWith(">=");
    let s = ge ? v.slice(2) : v;
    let mult = 1;
    if (/[kK]$/.test(s)) (mult = 1024), (s = s.slice(0, -1));
    if (/M$/.test(s)) (mult = 1048576), (s = s.slice(0, -1));
    return { ge, val: Number(s) * mult, text: (ge ? "≥ " : "") + v.replace(">=", "") };
  }

  NB.register("fig-results", function (root) {
    let view = "median";
    const ctr = NB.controls(root);
    NB.segmented(ctr, {
      label: "Show",
      options: [
        { value: "median", label: "median over seeds (Table 1)" },
        { value: "seeds", label: "every seed (Table 5)" },
      ],
      value: view,
      onChange: (v) => ((view = v), draw()),
    });
    const wrap = NB.h("div", { class: "nb-table-wrap" });
    root.appendChild(wrap);
    const note = NB.h("p", { class: "nb-caption" });
    root.appendChild(note);
    note.innerHTML =
      "Cell shade: log of the failing length L<sub>max</sub> (longer is darker). ≥ L: no failure up to the longest length that fits on one GPU. " +
      "† the extracted tables match the target, which certifies correctness at every length provided the logit margin exceeds the error of the logit computation. " +
      "× no seed learned the task. k/5: seeds that learned it. Hover a cell for the seeds. " +
      "Baselines fail below 90% accuracy, the Neural-FSM below 100%. Algebraic tasks trained at L = 64, TSO at L = 256.";

    function shade(p) {
      if (p.x) return "";
      const f = Math.max(0, Math.min(1, (Math.log10(p.val) - 1.5) / (6 - 1.5)));
      return `--f:${(0.08 + 0.8 * f).toFixed(3)}`;
    }

    function draw() {
      NB.clear(wrap);
      const tb = NB.h("table", { class: "nb-table results" });
      const h1 = NB.h("tr", {}, NB.h("th", { text: "" }));
      for (const t of TASKS)
        h1.appendChild(NB.h("th", { class: "task" + (t.definite ? " definite" : ""), title: t.group, html: `${t.label}<span class="sub">[${NB.fmtInt(t.states)}]${t.definite ? " definite" : ""}</span>` }));
      tb.appendChild(NB.h("thead", {}, h1));
      const body = NB.h("tbody");
      MODELS.forEach((m, mi) => {
        const tr = NB.h("tr", { class: m.key === "nfsm" ? "nfsm" : "" }, NB.h("th", { class: "model", html: `<i class="dot" style="--sw:${m.color}"></i>${m.label}<span class="sub">${m.sub}</span>` }));
        for (const t of TASKS) {
          const seeds = DATA[t.key][m.key];
          const learned = seeds.filter((s) => s !== "x").length;
          let content;
          let style = "";
          if (view === "median") {
            const p = parse(MED[t.key][mi]);
            style = shade(p);
            content = p.x ? "×" : p.text + (m.key === "nfsm" && CERT.has(t.key) ? "†" : "");
            content = `<span class="v">${content}</span><span class="sub">${learned}/5</span>`;
          } else {
            content = seeds.map((s) => `<span class="seed ${s === "x" ? "x" : ""}" style="${shade(parse(s))}">${s === "x" ? "×" : parse(s).text}</span>`).join("");
          }
          const deep = view === "median" && !parse(MED[t.key][mi]).x && Math.log10(parse(MED[t.key][mi]).val) > 4.2;
          const td = NB.h("td", { class: "cell" + (view === "seeds" ? " seeds" : "") + (deep ? " deep" : ""), style, html: content });
          td.addEventListener("pointermove", (e) =>
            NB.showTip(
              `<div class="tip-head">${m.label} on ${t.label}</div>` + seeds.map((s, i) => `<div class="tip-row">seed ${42 + i}<b>${s === "x" ? "did not learn" : parse(s).text}</b></div>`).join(""),
              e.clientX,
              e.clientY,
            ),
          );
          td.addEventListener("pointerleave", NB.hideTip);
          tr.appendChild(td);
        }
        body.appendChild(tr);
      });
      tb.appendChild(body);
      wrap.appendChild(tb);
    }
    draw();
  });
})();
