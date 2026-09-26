/* N11 — Tracking Shuffled Objects (§7; App. H.1 and H.6.3).
   The token-level automaton of App. H.1: state (p, g), with p the first name of the sentence in
   progress (or idle) and g the holder of every item (or unset). The trained 2-layer NFSM keeps one
   second-layer head per item, holding exactly the holder register shown here (Figure 5). */
(function () {
  "use strict";
  const NB = window.NB;
  const PEOPLE = ["alice", "bob", "charlie", "dana", "eve"];
  const ITEMS = ["ring", "key", "coin", "stamp", "card"];

  NB.register("fig-tso", function (root) {
    let N = 4;
    let S = 8;
    let seed = 1;
    let tokens = [];
    let cur = 0;

    const ctr = NB.controls(root);
    NB.slider(ctr, { label: "people and items N", min: 3, max: 5, step: 1, value: N, onInput: (v) => ((N = v), gen()) });
    NB.slider(ctr, { label: "swap sentences", min: 1, max: 30, step: 1, value: S, onInput: (v) => ((S = v), gen()) });
    NB.button(ctr, "New story", () => (seed++, gen()), { cls: "ghost" });
    const tctr = NB.controls(root);
    const tS = NB.slider(tctr, { label: "tokens read", min: 0, max: 10, step: 1, value: 0, onInput: (v) => ((cur = v), draw()) });
    const player = NB.player(tctr, () => {
      if (cur >= tokens.length) return false;
      cur++;
      tS.set(cur, true);
      draw();
      return cur < tokens.length;
    }, { delay: 220, root });

    const panels = NB.panels(root);
    const pT = NB.panel(panels, "The stream, one token at a time", "grow");
    const text = NB.h("div", { class: "nb-story" });
    pT.appendChild(text);
    const pR = NB.panel(panels, "The automaton state (p, g) after the tokens read");
    const reg = NB.h("div", { class: "nb-table-wrap" });
    pR.appendChild(reg);

    function gen() {
      NB.live("fig-tso", { N, S });
      player.stop();
      const rand = NB.rng(500 + seed);
      const perm = PEOPLE.slice(0, N).map((p, i) => [p, i]);
      const items = ITEMS.slice(0, N).map((it) => [it, rand()]).sort((a, b) => a[1] - b[1]).map((a) => a[0]);
      tokens = [];
      for (let i = 0; i < N; i++) tokens.push(PEOPLE[i], "has", "the", items[i], ".");
      for (let s = 0; s < S; s++) {
        const a = NB.randInt(rand, N);
        let b = NB.randInt(rand, N - 1);
        if (b >= a) b++;
        tokens.push(PEOPLE[a], "swaps", "with", PEOPLE[b], ".");
      }
      const asked = ITEMS[NB.randInt(rand, N)];
      tokens.push("who", "has", "the", asked, "?");
      void perm;
      // open halfway through the story: every sentence is 5 tokens long
      cur = 5 * Math.floor((N + S + 1) / 2);
      tS.el.querySelector("input").max = tokens.length;
      tS.set(cur, true);
      draw();
    }

    function state(upto) {
      let p = null;
      const g = {};
      for (let i = 0; i < N; i++) g[ITEMS[i]] = null;
      for (let t = 0; t < upto; t++) {
        const tok = tokens[t];
        if (PEOPLE.includes(tok)) {
          if (p == null) p = tok;
          else {
            // a second name swaps the items of the two people
            for (const it in g) {
              if (g[it] === p) g[it] = tok;
              else if (g[it] === tok) g[it] = p;
            }
            p = null;
          }
        } else if (ITEMS.includes(tok) && tokens[t - 3] !== "who" && p != null) {
          g[tok] = p;
          p = null;
        }
      }
      return { p, g };
    }

    function draw() {
      NB.clear(text);
      tokens.forEach((tok, t) => {
        const cls = "tok" + (t < cur ? " read" : "") + (t === cur - 1 ? " last" : "") + (t >= tokens.length - 5 ? " q" : "");
        const s = NB.h("span", { class: cls, text: tok });
        s.addEventListener("click", () => ((cur = t + 1), tS.set(cur, true), draw()));
        text.appendChild(s);
        if (tok === "." || tok === "?") text.appendChild(NB.h("br"));
      });
      const st = state(cur);
      NB.clear(reg);
      const tb = NB.h("table", { class: "nb-table compact" });
      tb.appendChild(NB.h("thead", {}, NB.h("tr", {}, NB.h("th", { text: "register" }), NB.h("th", { text: "value" }))));
      const body = NB.h("tbody");
      body.appendChild(NB.h("tr", {}, NB.h("td", { html: "p <small>(first name of the sentence)</small>" }), NB.h("td", { class: "mono", text: st.p || "idle" })));
      for (let i = 0; i < N; i++) {
        const who = st.g[ITEMS[i]];
        const col = who ? NB.stateColor(PEOPLE.indexOf(who)) : "var(--muted)";
        body.appendChild(NB.h("tr", {}, NB.h("td", { html: `holder of the ${ITEMS[i]}` }), NB.h("td", { class: "mono", html: `<i class="dot" style="--sw:${col}"></i>${who || "unset"}` })));
      }
      tb.appendChild(body);
      reg.appendChild(tb);
      if (cur >= tokens.length) {
        const asked = tokens[tokens.length - 2];
        reg.appendChild(NB.h("p", { class: "nb-answer", html: `who has the ${asked}? <b>${st.g[asked]}</b>` }));
      }
    }
    gen();
  });
})();
