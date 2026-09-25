/* =====================================================================
   tasks.js — the semiautomata used across the notebook.
   A task is (Q, Σ, δ): n states, a list of symbols, delta(q, xIndex).
   ===================================================================== */
(function () {
  "use strict";
  const NB = window.NB;

  NB.tasks = {
    // Counting modulo N: the running example of the paper (Z_5 in Appendix H.5).
    cyclic(N) {
      const symbols = [];
      for (let k = 1; k < N; k++) symbols.push({ label: "+" + k });
      symbols.push({ label: "+0" });
      const shifts = symbols.map((s) => Number(s.label.slice(1)));
      return {
        id: "Z" + N,
        name: `ℤ<sub>${N}</sub>`,
        n: N,
        states: Array.from({ length: N }, (_, i) => String(i)),
        symbols,
        delta: (q, x) => (q + shifts[x]) % N,
        start: 0,
        kind: "abelian group",
        definite: false,
        monoidSize: N,
      };
    },

    // The flip-flop monoid: identity, reset (constant 0), set (constant 1).
    flipflop() {
      return {
        id: "FF",
        name: "flip-flop",
        n: 2,
        states: ["0", "1"],
        symbols: [{ label: "id" }, { label: "reset" }, { label: "set" }],
        delta: (q, x) => (x === 0 ? q : x === 1 ? 0 : 1),
        start: 0,
        kind: "non-invertible monoid",
        definite: false,
        monoidSize: 3,
      };
    },

    // S_3 acting on three cups: a state is the arrangement of balls 1,2,3,
    // a symbol swaps the contents of two cup positions.
    s3() {
      const perms = ["123", "213", "231", "321", "312", "132"]; // Cayley 6-cycle order
      const idx = Object.fromEntries(perms.map((p, i) => [p, i]));
      const swap = (p, a, b) => {
        const arr = p.split("");
        [arr[a], arr[b]] = [arr[b], arr[a]];
        return arr.join("");
      };
      return {
        id: "S3",
        name: "S<sub>3</sub>",
        n: 6,
        states: perms,
        symbols: [{ label: "(1 2)" }, { label: "(2 3)" }],
        delta: (q, x) => idx[x === 0 ? swap(perms[q], 0, 1) : swap(perms[q], 1, 2)],
        start: 0,
        kind: "non-abelian group",
        definite: false,
        monoidSize: 6,
      };
    },
  };

  // Transition table of symbol x as an array: table[q] = δ_x(q).
  NB.table = (task, x) => Array.from({ length: task.n }, (_, q) => task.delta(q, x));
  // Compose tables: (after ∘ before)[q] = after[before[q]].
  NB.compose = (after, before) => before.map((q) => after[q]);
  NB.identityTable = (n) => Array.from({ length: n }, (_, q) => q);
  // Is the map constant?
  NB.isConstant = (t) => t.every((v) => v === t[0]);
})();
