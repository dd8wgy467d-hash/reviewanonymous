/* =====================================================================
   core.js — shared helpers for the interactive notebook.
   Everything hangs off the global namespace NB (classic scripts, so the
   page also works when opened from the file system).
   ===================================================================== */
(function () {
  "use strict";
  const NB = (window.NB = window.NB || {});
  const SVGNS = "http://www.w3.org/2000/svg";
  NB.TWO_PI = Math.PI * 2;

  // ------------------------------------------------------------------
  // Random numbers
  // ------------------------------------------------------------------
  NB.rng = function (seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };
  NB.gauss = function (rand) {
    const u = 1 - rand();
    const v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(NB.TWO_PI * v);
  };
  // Uniform sample in the 2-D disk of radius r.
  NB.ball2 = function (rand, r) {
    const a = NB.TWO_PI * rand();
    const s = r * Math.sqrt(rand());
    return [s * Math.cos(a), s * Math.sin(a)];
  };
  NB.randInt = (rand, n) => Math.floor(rand() * n);
  // Scalar perturbation: uniform in [−η, η], or exactly ±η in the worst case.
  NB.perturb = (rand, eta, worst) => (worst ? (rand() < 0.5 ? -eta : eta) : eta * (2 * rand() - 1));
  // 2-D perturbation: uniform in the disk of radius η, or on its boundary in the worst case.
  NB.perturb2 = function (rand, eta, worst) {
    if (!worst) return NB.ball2(rand, eta);
    const a = NB.TWO_PI * rand();
    return [eta * Math.cos(a), eta * Math.sin(a)];
  };
  // The "worst case" checkbox shared by every figure that samples perturbations.
  NB.worstToggle = function (parent, onChange) {
    return NB.toggle(parent, { label: "worst case: every perturbation has size exactly η", value: true, onChange });
  };

  // ------------------------------------------------------------------
  // Finite-precision emulation (round-to-nearest-even on fp32 bits)
  // ------------------------------------------------------------------
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  function dropBits(x, drop) {
    f32[0] = x;
    let u = u32[0];
    const lsb = (u >>> drop) & 1;
    u = (u + ((1 << (drop - 1)) - 1) + lsb) >>> 0;
    u32[0] = (u >>> drop) << drop;
    return f32[0];
  }
  NB.formats = {
    fp32: { label: "fp32", u: Math.pow(2, -24), round: (x) => Math.fround(x) },
    bf16: { label: "bf16", u: Math.pow(2, -8), round: (x) => dropBits(x, 16) },
    fp16: {
      label: "fp16",
      u: Math.pow(2, -11),
      round: (x) => {
        const ax = Math.abs(x);
        if (ax >= 65520) return x > 0 ? Infinity : -Infinity;
        if (ax < 6.103515625e-5) return Math.round(x * 16777216) / 16777216; // subnormals
        return dropBits(x, 13);
      },
    },
  };

  // ------------------------------------------------------------------
  // DOM builders
  // ------------------------------------------------------------------
  // NB.h("div", {class: "x", text: "hi", on: {click: fn}}, child1, ...)
  NB.h = function (tag, attrs, ...children) {
    const el = document.createElement(tag);
    applyAttrs(el, attrs);
    for (const c of children.flat()) {
      if (c == null || c === false) continue;
      el.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return el;
  };
  // NB.s("circle", {cx: 1, cy: 2, r: 3, class: "dot"}, parentSvgGroup)
  NB.s = function (tag, attrs, parent) {
    const el = document.createElementNS(SVGNS, tag);
    applyAttrs(el, attrs);
    if (parent) parent.appendChild(el);
    return el;
  };
  function applyAttrs(el, attrs) {
    if (!attrs) return;
    for (const k in attrs) {
      const v = attrs[k];
      if (v == null || v === false) continue;
      if (k === "text") el.textContent = v;
      else if (k === "html") el.innerHTML = v;
      else if (k === "on") for (const ev in v) el.addEventListener(ev, v[ev]);
      else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
      else el.setAttribute(k, v);
    }
  }
  // SVG text "base" with a subscript: NB.subText(g, {x, y, class}, "U", "q1")
  NB.subText = function (parent, attrs, base, subscript) {
    const t = NB.s("text", attrs, parent);
    t.appendChild(document.createTextNode(base));
    NB.s("tspan", { "baseline-shift": "sub", "font-size": "75%", text: subscript }, t);
    return t;
  };
  NB.clear = function (el) {
    while (el.firstChild) el.removeChild(el.firstChild);
    return el;
  };
  NB.svgRoot = function (parent, w, h, cls) {
    return NB.s(
      "svg",
      { viewBox: `0 0 ${w} ${h}`, class: "nb-svg " + (cls || ""), role: "img" },
      parent,
    );
  };
  NB.fmt = function (x, d) {
    if (!isFinite(x)) return x > 0 ? "∞" : "−∞";
    return x.toFixed(d == null ? 2 : d).replace("-", "−");
  };
  NB.fmtInt = function (x) {
    if (!isFinite(x)) return "∞";
    if (x >= 1e6) return (x / 1e6).toFixed(x >= 1e7 ? 0 : 1) + "M";
    if (x >= 1e4) return (x / 1e3).toFixed(0) + "k";
    return Math.round(x).toLocaleString("en-US");
  };
  NB.fmtSci = function (x) {
    if (x === 0) return "0";
    if (!isFinite(x)) return "∞";
    const e = Math.floor(Math.log10(Math.abs(x)));
    if (e >= -2 && e <= 3) return x.toPrecision(2);
    const m = x / Math.pow(10, e);
    return `${m.toFixed(1)}·10${sup(e)}`;
  };
  function sup(n) {
    const map = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
    return String(n).split("").map((c) => map[c]).join("");
  }
  NB.sup = sup;
  NB.stateColor = (i) => `var(--s${((i % 8) + 8) % 8})`;

  // ------------------------------------------------------------------
  // Controls
  // ------------------------------------------------------------------
  NB.controls = function (parent, cls) {
    return parent.appendChild(NB.h("div", { class: "nb-controls " + (cls || "") }));
  };

  // Slider. opts: {label, min, max, step, value, format, onInput, log}
  NB.slider = function (parent, opts) {
    const fmt = opts.format || ((v) => String(v));
    const log = !!opts.log;
    const toPos = (v) => (log ? (1000 * Math.log(v / opts.min)) / Math.log(opts.max / opts.min) : v);
    const fromPos = (p) => (log ? opts.min * Math.pow(opts.max / opts.min, p / 1000) : Number(p));
    const input = NB.h("input", {
      type: "range",
      min: log ? 0 : opts.min,
      max: log ? 1000 : opts.max,
      step: log ? 1 : opts.step || "any",
      value: toPos(opts.value),
      "aria-label": opts.label,
    });
    const val = NB.h("span", { class: "nb-slider-val" });
    const wrap = NB.h(
      "label",
      { class: "nb-slider" },
      NB.h("span", { class: "nb-slider-head" }, NB.h("span", { class: "nb-slider-name", html: opts.label }), val),
      input,
    );
    parent.appendChild(wrap);
    let value = opts.value;
    val.textContent = fmt(value);
    input.addEventListener("input", () => {
      value = fromPos(input.value);
      if (opts.step && !log) value = Math.round(value / opts.step) * opts.step;
      value = Number(value.toPrecision(12));
      val.textContent = fmt(value);
      if (opts.onInput) opts.onInput(value);
    });
    return {
      el: wrap,
      get value() {
        return value;
      },
      set(v, silent) {
        value = v;
        input.value = toPos(v);
        val.textContent = fmt(v);
        if (!silent && opts.onInput) opts.onInput(v);
      },
    };
  };

  // A titled panel inside a figure. Returns the panel body.
  NB.panel = function (parent, title, cls) {
    const p = NB.h("div", { class: "nb-panel " + (cls || "") });
    if (title) p.appendChild(NB.h("div", { class: "nb-panel-title", html: title }));
    parent.appendChild(p);
    return p;
  };
  NB.panels = function (parent, cls) {
    return parent.appendChild(NB.h("div", { class: "nb-panels " + (cls || "") }));
  };

  NB.button = function (parent, label, onClick, opts) {
    opts = opts || {};
    const b = NB.h("button", {
      type: "button",
      class: "nb-btn " + (opts.cls || ""),
      title: opts.title,
      html: label,
      on: { click: onClick },
    });
    parent.appendChild(b);
    return b;
  };

  NB.group = function (parent, label, cls) {
    const g = NB.h("div", { class: "nb-group " + (cls || "") });
    if (label) g.appendChild(NB.h("span", { class: "nb-group-label", html: label }));
    const body = NB.h("div", { class: "nb-group-body" });
    g.appendChild(body);
    parent.appendChild(g);
    return body;
  };

  // Segmented control. opts: {label, options:[{value,label}], value, onChange}
  NB.segmented = function (parent, opts) {
    const body = NB.group(parent, opts.label, "nb-seg");
    body.setAttribute("role", "radiogroup");
    let value = opts.value;
    const btns = opts.options.map((o) => {
      const b = NB.h("button", {
        type: "button",
        class: "nb-seg-btn",
        role: "radio",
        html: o.label,
        title: o.title,
        on: {
          click: () => {
            api.set(o.value);
          },
        },
      });
      b.dataset.value = String(o.value);
      body.appendChild(b);
      return b;
    });
    function paint() {
      for (const b of btns) {
        const on = b.dataset.value === String(value);
        b.classList.toggle("on", on);
        b.setAttribute("aria-checked", on ? "true" : "false");
      }
    }
    const api = {
      get value() {
        return value;
      },
      set(v, silent) {
        value = v;
        paint();
        if (!silent && opts.onChange) opts.onChange(v);
      },
    };
    paint();
    return api;
  };

  NB.toggle = function (parent, opts) {
    const input = NB.h("input", { type: "checkbox", class: "nb-switch" });
    input.checked = !!opts.value;
    const lab = NB.h("label", { class: "nb-toggle" }, input, NB.h("span", { html: opts.label }));
    input.addEventListener("change", () => opts.onChange && opts.onChange(input.checked));
    parent.appendChild(lab);
    return {
      get value() {
        return input.checked;
      },
      set(v, silent) {
        input.checked = !!v;
        if (!silent && opts.onChange) opts.onChange(input.checked);
      },
    };
  };

  // A labelled read-out. set(text, status) with status "good" | "bad" | "warn" | null.
  NB.stat = function (parent, label, cls) {
    const v = NB.h("span", { class: "nb-stat-val" });
    const el = NB.h("div", { class: "nb-stat " + (cls || "") }, NB.h("span", { class: "nb-stat-label", html: label }), v);
    parent.appendChild(el);
    return {
      el,
      set(html, status) {
        const icon = status === "good" ? "✓ " : status === "bad" ? "✗ " : status === "warn" ? "! " : "";
        v.innerHTML = icon + html;
        el.dataset.status = status || "";
      },
    };
  };

  // Play / pause helper. onStep() returns false to stop.
  NB.player = function (parent, onStep, opts) {
    opts = opts || {};
    let timer = null;
    const btn = NB.button(parent, "▶ Play", toggle, { cls: "nb-play" });
    let delay = opts.delay || 120;
    function tick() {
      const cont = onStep();
      if (cont === false) return stop();
      timer = setTimeout(tick, delay);
    }
    function start() {
      if (timer) return;
      btn.innerHTML = "❚❚ Pause";
      btn.classList.add("on");
      tick();
    }
    function stop() {
      if (timer) clearTimeout(timer);
      timer = null;
      btn.innerHTML = "▶ Play";
      btn.classList.remove("on");
    }
    function toggle() {
      timer ? stop() : start();
    }
    // pause when scrolled out of view
    if (opts.root && "IntersectionObserver" in window) {
      new IntersectionObserver((es) => es.forEach((e) => !e.isIntersecting && stop())).observe(opts.root);
    }
    return {
      start,
      stop,
      get playing() {
        return !!timer;
      },
      setDelay(d) {
        delay = d;
      },
      btn,
    };
  };

  // ------------------------------------------------------------------
  // Tooltip (one per page)
  // ------------------------------------------------------------------
  let tip = null;
  NB.showTip = function (html, x, y) {
    if (!tip) {
      tip = NB.h("div", { class: "nb-tip", role: "tooltip" });
      document.body.appendChild(tip);
    }
    tip.innerHTML = html;
    tip.style.display = "block";
    const r = tip.getBoundingClientRect();
    let left = x + 14;
    let top = y + 14;
    if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
    if (top + r.height > window.innerHeight - 8) top = y - r.height - 14;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = Math.max(8, top) + "px";
  };
  NB.hideTip = function () {
    if (tip) tip.style.display = "none";
  };

  // ------------------------------------------------------------------
  // Plot: linear / log axes in an SVG viewBox
  // ------------------------------------------------------------------
  class Plot {
    constructor(parent, o) {
      this.W = o.width || 520;
      this.H = o.height || 300;
      this.m = Object.assign({ l: 52, r: 16, t: 14, b: 40 }, o.margin || {});
      this.svg = NB.svgRoot(parent, this.W, this.H, o.cls);
      if (o.title) this.svg.setAttribute("aria-label", o.title);
      this.x = o.x;
      this.y = o.y;
      this.xlog = !!o.xlog;
      this.ylog = !!o.ylog;
      this.gAxes = NB.s("g", { class: "axes" }, this.svg);
      this.gData = NB.s("g", { class: "data" }, this.svg);
      this.gOver = NB.s("g", { class: "over" }, this.svg);
      const clipId = "clip" + Math.random().toString(36).slice(2);
      const defs = NB.s("defs", null, this.svg);
      const cp = NB.s("clipPath", { id: clipId }, defs);
      this.clipRect = NB.s("rect", {}, cp);
      this.gData.setAttribute("clip-path", `url(#${clipId})`);
      this.updateClip();
    }
    updateClip() {
      this.clipRect.setAttribute("x", this.m.l);
      this.clipRect.setAttribute("y", this.m.t - 2);
      this.clipRect.setAttribute("width", this.W - this.m.l - this.m.r);
      this.clipRect.setAttribute("height", this.H - this.m.t - this.m.b + 4);
    }
    get x0() {
      return this.m.l;
    }
    get x1() {
      return this.W - this.m.r;
    }
    get y0() {
      return this.H - this.m.b;
    }
    get y1() {
      return this.m.t;
    }
    sx(v) {
      const [a, b] = this.x;
      const f = this.xlog ? Math.log(v / a) / Math.log(b / a) : (v - a) / (b - a);
      return this.x0 + f * (this.x1 - this.x0);
    }
    sy(v) {
      const [a, b] = this.y;
      const f = this.ylog ? Math.log(v / a) / Math.log(b / a) : (v - a) / (b - a);
      return this.y0 - f * (this.y0 - this.y1);
    }
    ix(px) {
      const [a, b] = this.x;
      const f = (px - this.x0) / (this.x1 - this.x0);
      return this.xlog ? a * Math.pow(b / a, f) : a + f * (b - a);
    }
    ticks(dom, log, n) {
      const [a, b] = dom;
      if (log) {
        const out = [];
        for (let e = Math.ceil(Math.log10(a) - 1e-9); e <= Math.floor(Math.log10(b) + 1e-9); e++) out.push(Math.pow(10, e));
        return out;
      }
      const span = b - a;
      const step0 = span / (n || 5);
      const mag = Math.pow(10, Math.floor(Math.log10(step0)));
      const step = [1, 2, 2.5, 5, 10].map((s) => s * mag).find((s) => s >= step0);
      const out = [];
      for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(Number(v.toPrecision(10)));
      return out;
    }
    axes(o) {
      o = o || {};
      NB.clear(this.gAxes);
      const g = this.gAxes;
      const xt = o.xticks || this.ticks(this.x, this.xlog, o.nx);
      const yt = o.yticks || this.ticks(this.y, this.ylog, o.ny);
      const fx = o.xfmt || ((v) => (this.xlog ? "10" + sup(Math.round(Math.log10(v))) : String(v)));
      const fy = o.yfmt || ((v) => (this.ylog ? "10" + sup(Math.round(Math.log10(v))) : String(v)));
      for (const v of yt) {
        const y = this.sy(v);
        if (o.ygrid !== false) NB.s("line", { x1: this.x0, x2: this.x1, y1: y, y2: y, class: "grid" }, g);
        NB.s("text", { x: this.x0 - 6, y: y + 4, class: "tick", "text-anchor": "end", text: fy(v) }, g);
      }
      for (const v of xt) {
        const x = this.sx(v);
        if (o.xgrid) NB.s("line", { x1: x, x2: x, y1: this.y0, y2: this.y1, class: "grid" }, g);
        NB.s("line", { x1: x, x2: x, y1: this.y0, y2: this.y0 + 4, class: "baseline" }, g);
        NB.s("text", { x, y: this.y0 + 17, class: "tick", "text-anchor": "middle", text: fx(v) }, g);
      }
      NB.s("line", { x1: this.x0, x2: this.x1, y1: this.y0, y2: this.y0, class: "baseline" }, g);
      if (o.xlabel)
        NB.s("text", { x: (this.x0 + this.x1) / 2, y: this.H - 5, class: "axis-label", "text-anchor": "middle", text: o.xlabel }, g);
      if (o.ylabel)
        NB.s(
          "text",
          {
            x: -(this.y0 + this.y1) / 2,
            y: 13,
            transform: "rotate(-90)",
            class: "axis-label",
            "text-anchor": "middle",
            text: o.ylabel,
          },
          g,
        );
    }
    pathD(pts) {
      let d = "";
      let pen = false;
      for (const [x, y] of pts) {
        if (!isFinite(x) || !isFinite(y) || (this.ylog && y <= 0) || (this.xlog && x <= 0)) {
          pen = false;
          continue;
        }
        d += (pen ? "L" : "M") + this.sx(x).toFixed(1) + "," + this.sy(y).toFixed(1);
        pen = true;
      }
      return d;
    }
    line(pts, attrs, parent) {
      return NB.s("path", Object.assign({ d: this.pathD(pts), fill: "none" }, attrs), parent || this.gData);
    }
    // Crosshair + tooltip. getSeries() -> [{name, color, pts:[[x,y],...] (x ascending)}]
    crosshair(getSeries, fx, fy) {
      const over = NB.s(
        "rect",
        { x: this.x0, y: this.y1, width: this.x1 - this.x0, height: this.y0 - this.y1, class: "hit" },
        this.gOver,
      );
      const vline = NB.s("line", { class: "crosshair", y1: this.y1, y2: this.y0, visibility: "hidden" }, this.gOver);
      const dots = NB.s("g", {}, this.gOver);
      const move = (ev) => {
        const pt = this.svg.createSVGPoint();
        pt.x = ev.clientX;
        pt.y = ev.clientY;
        const loc = pt.matrixTransform(this.svg.getScreenCTM().inverse());
        const xv = this.ix(loc.x);
        NB.clear(dots);
        let html = "";
        let xShown = null;
        for (const s of getSeries()) {
          if (!s.pts.length) continue;
          let lo = 0;
          let hi = s.pts.length - 1;
          while (hi - lo > 1) {
            const mid = (lo + hi) >> 1;
            s.pts[mid][0] < xv ? (lo = mid) : (hi = mid);
          }
          const p = Math.abs(s.pts[lo][0] - xv) < Math.abs(s.pts[hi][0] - xv) ? s.pts[lo] : s.pts[hi];
          if (xShown == null) xShown = p[0];
          if (isFinite(p[1]) && (!this.ylog || p[1] > 0))
            NB.s("circle", { cx: this.sx(p[0]), cy: this.sy(p[1]), r: 4, class: "hover-dot", style: `fill:${s.color}` }, dots);
          html += `<div class="tip-row"><i style="background:${s.color}"></i>${s.name}<b>${fy(p[1])}</b></div>`;
        }
        if (xShown == null) return;
        vline.setAttribute("x1", this.sx(xShown));
        vline.setAttribute("x2", this.sx(xShown));
        vline.setAttribute("visibility", "visible");
        NB.showTip(`<div class="tip-head">${fx(xShown)}</div>` + html, ev.clientX, ev.clientY);
      };
      const leave = () => {
        vline.setAttribute("visibility", "hidden");
        NB.clear(dots);
        NB.hideTip();
      };
      over.addEventListener("pointermove", move);
      over.addEventListener("pointerleave", leave);
    }
  }
  NB.Plot = Plot;

  // Legend (HTML) — items: [{label, color, dash}]
  NB.legend = function (parent, items) {
    const el = NB.h("div", { class: "nb-legend" });
    for (const it of items) {
      const sw = NB.h("i", { class: "sw " + (it.shape || "line") });
      sw.style.setProperty("--sw", it.color);
      if (it.dash) sw.classList.add("dash");
      if (it.dot) sw.classList.add("dotted");
      el.appendChild(NB.h("span", {}, sw, NB.h("span", { html: it.label })));
    }
    parent.appendChild(el);
    return el;
  };

  // Arrowhead marker helper: returns url(#id) for a given class.
  NB.arrowMarker = function (svg, id, cls) {
    let defs = svg.querySelector("defs");
    if (!defs) defs = NB.s("defs", null, svg);
    const m = NB.s(
      "marker",
      { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" },
      defs,
    );
    NB.s("path", { d: "M0,1 L10,5 L0,9 z", class: cls || "arrowhead" }, m);
    return `url(#${id})`;
  };

  // ------------------------------------------------------------------
  // Figure registry: each figure registers an init function run on load.
  // ------------------------------------------------------------------
  NB.figures = [];
  NB.register = function (id, init) {
    NB.figures.push({ id, init });
  };
  // ------------------------------------------------------------------
  // Flip side: the code behind each figure (entries in js/code.js)
  // ------------------------------------------------------------------
  const KW = new Set("def return for in if elif else while and or not lambda None True False global break import from as is".split(" "));
  const BI = new Set(
    "range min max abs round sum len all any enumerate zip print set list dict sorted count mean median log log2 exp sqrt sin cos tanh sign ceil factorial argmax norm angle inf pi".split(" "),
  );
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  NB.highlight = function (src) {
    const re = /(\{\{\w+\}\})|(#[^\n]*)|("[^"\n]*"|'[^'\n]*')|(\b\d+(?:\.\d+)?(?:e-?\d+)?\b)|([A-Za-z_]\w*)/g;
    let out = "";
    let last = 0;
    let m;
    while ((m = re.exec(src))) {
      out += esc(src.slice(last, m.index));
      last = re.lastIndex;
      const t = esc(m[0]);
      if (m[1]) out += t;
      else if (m[2]) out += `<span class="tk-com">${t}</span>`;
      else if (m[3]) out += `<span class="tk-str">${t}</span>`;
      else if (m[4]) out += `<span class="tk-num">${t}</span>`;
      else if (KW.has(m[5])) out += `<span class="tk-kw">${t}</span>`;
      else if (BI.has(m[5])) out += `<span class="tk-bi">${t}</span>`;
      else if (src[re.lastIndex] === "(") out += `<span class="tk-fn">${t}</span>`;
      else out += t;
    }
    out += esc(src.slice(last));
    out = out.replace(/\{\{(\w+)\}\}/g, '<span class="live" data-k="$1">$1</span>');
    return out
      .split("\n")
      .map((l) => `<span class="ln">${l || " "}</span>`)
      .join("");
  };

  NB._live = {};
  const fmtLive = (v) => (typeof v === "number" ? (Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(4)))) : String(v));
  function refreshLive(id) {
    const body = document.getElementById(id);
    const fig = body && body.closest(".nb-figure");
    if (!fig) return;
    const vals = NB._live[id] || {};
    const flipped = fig.classList.contains("flipped");
    fig.querySelectorAll(".nb-face.back .live[data-k]").forEach((s) => {
      const v = vals[s.dataset.k];
      const txt = v == null ? s.dataset.k : fmtLive(v);
      if (s.textContent !== txt) {
        s.textContent = txt;
        if (flipped) {
          s.classList.remove("bump");
          void s.offsetWidth;
          s.classList.add("bump");
        }
      }
    });
  }
  // Figures report the values their code is running with: NB.live("fig-dial", {rho: 0.99}).
  NB.live = function (id, params) {
    NB._live[id] = Object.assign(NB._live[id] || {}, params);
    refreshLive(id);
  };

  function buildFlip(fig) {
    const body = fig.querySelector(".nb-fig-body");
    const id = body && body.id;
    const code = NB.code && NB.code[id];
    if (!code) return;
    const inner = NB.h("div", { class: "nb-flip" });
    const front = NB.h("div", { class: "nb-face front" });
    const back = NB.h("div", { class: "nb-face back", inert: "", "aria-label": "Code behind this figure" });
    while (fig.firstChild) front.appendChild(fig.firstChild);
    inner.append(front, back);
    fig.appendChild(inner);
    fig.classList.add("has-code");

    const cap = front.querySelector("figcaption");
    const openBtn = NB.h("button", {
      type: "button",
      class: "nb-btn code-btn",
      title: "Flip the figure to see the code it runs",
      "aria-pressed": "false",
      html: '<span aria-hidden="true">&lt;/&gt;</span> Code',
      on: { click: () => flip(true) },
    });
    cap.insertBefore(openBtn, cap.firstChild);

    const figNo = cap.querySelector(".fig-no");
    const head = NB.h(
      "div",
      { class: "code-head" },
      NB.h("span", { class: "fig-no", text: figNo ? figNo.textContent : "" }),
      NB.h("span", { class: "code-title", text: "The code behind this figure" }),
    );
    const closeBtn = NB.h("button", {
      type: "button",
      class: "nb-btn code-btn",
      html: '<span aria-hidden="true">↩</span> Back to the figure',
      on: { click: () => flip(false) },
    });
    head.appendChild(closeBtn);
    back.appendChild(head);
    for (const b of code.blocks) {
      back.appendChild(NB.h("div", { class: "code-block-title", text: b.title }));
      back.appendChild(NB.h("pre", { class: "code", html: NB.highlight(b.src) }));
    }
    back.appendChild(
      NB.h("p", {
        class: "code-foot",
        html: `Python-style pseudocode of what <code>${code.file}</code> executes. <span class="live">Highlighted</span> values are your current settings and follow the controls on the other side.`,
      }),
    );
    back.addEventListener("keydown", (e) => e.key === "Escape" && flip(false));

    function flip(on) {
      // let the card grow if the code is longer than the figure
      front.style.minHeight = on ? Math.max(front.offsetHeight, back.scrollHeight) + "px" : "";
      fig.classList.toggle("flipped", on);
      front.inert = on;
      back.inert = !on;
      openBtn.setAttribute("aria-pressed", String(on));
      if (on) refreshLive(id);
      setTimeout(() => (on ? closeBtn : openBtn).focus({ preventScroll: true }), 50);
    }
  }

  NB.boot = function () {
    document.querySelectorAll(".nb-figure").forEach(buildFlip);
    for (const f of NB.figures) {
      const root = document.getElementById(f.id);
      if (!root) continue;
      try {
        f.init(root);
      } catch (e) {
        console.error("Figure " + f.id + " failed:", e);
        root.appendChild(NB.h("p", { class: "nb-error", text: "This figure failed to load: " + e.message }));
      }
    }
  };

  // Theme toggle (light / dark), remembered per viewer when storage allows.
  NB.initTheme = function () {
    const btn = document.getElementById("theme-toggle");
    let saved = null;
    try {
      saved = localStorage.getItem("nb-theme");
    } catch (e) {}
    if (saved) document.documentElement.dataset.theme = saved;
    const isDark = () =>
      document.documentElement.dataset.theme === "dark" ||
      (!document.documentElement.dataset.theme && window.matchMedia("(prefers-color-scheme: dark)").matches);
    const paint = () => btn && (btn.textContent = isDark() ? "☀ Light" : "☾ Dark");
    paint();
    if (btn)
      btn.addEventListener("click", () => {
        const next = isDark() ? "light" : "dark";
        document.documentElement.dataset.theme = next;
        try {
          localStorage.setItem("nb-theme", next);
        } catch (e) {}
        paint();
      });
  };
})();
