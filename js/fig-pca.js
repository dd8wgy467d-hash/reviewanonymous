/* N12 — Geometry of trained recurrent states on ℤ_5 under the constant word (+1)^T (App. H.5, Figure 4). */
(function () {
  "use strict";
  const NB = window.NB;

  NB.register("fig-pca", function (root) {
    let seed = 46;
    let layers = 2;
    const ctr = NB.controls(root);
    NB.segmented(ctr, {
      label: "Seed",
      options: [42, 43, 44, 45, 46].map((s) => ({ value: s, label: String(s) })),
      value: seed,
      onChange: (v) => ((seed = v), load()),
    });
    NB.segmented(ctr, {
      label: "Baseline layers B",
      options: [1, 2, 4].map((l) => ({ value: l, label: String(l) })),
      value: layers,
      onChange: (v) => ((layers = v), load()),
    });
    const img = NB.h("img", { class: "nb-gif", loading: "lazy", alt: "" });
    root.appendChild(NB.h("div", { class: "nb-gif-wrap" }, img));
    function load() {
      NB.live("fig-pca", { seed, layers });
      img.src = `assets/pca_projections/fsa_hidden_pca_mod5_L${layers}_seed${seed}.gif`;
      img.alt = `Recurrent states on Z5 projected on two principal components, B = ${layers} layers, seed ${seed}`;
    }
    load();
  });
})();
