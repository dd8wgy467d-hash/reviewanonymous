/* =====================================================================
   code.js — the back of every figure: Python-style pseudocode that
   mirrors what the figure's JavaScript executes.
   {{name}} is a live value, filled from NB.live(figureId, {name: value}).
   Keep these in sync with js/fig-*.js.
   ===================================================================== */
(function () {
  "use strict";
  const NB = window.NB;
  const R = String.raw;

  NB.code = {
    // ------------------------------------------------------------ N1
    "fig-automata": {
      file: "js/fig-automata.js",
      blocks: [
        {
          title: "A semiautomaton, and a word as a map (§2)",
          src: R`task = {{task}}                  # (Q, Σ, δ)
Q, Sigma, delta = task.states, task.symbols, task.delta

q = q0                              # the current state
delta_w = list(Q)                   # δ_ε = id_Q, stored as a table: delta_w[p] = δ_w(p)

def read(x):
    global q, delta_w
    q = delta(q, x)                           # one transition
    delta_w = [delta(p, x) for p in delta_w]  # δ_{wx} = δ_x ∘ δ_w

# what the panel reports about δ_w
is_constant    = len(set(delta_w)) == 1       # the word has forgotten its start
is_permutation = len(set(delta_w)) == len(Q)`,
        },
      ],
    },

    // ------------------------------------------------------------ N2
    "fig-dial": {
      file: "js/fig-dial.js",
      blocks: [
        {
          title: "An affine RNN realizing ℤ_5 (eqs. 2–3)",
          src: R`N, rho, eta = 5, {{rho}}, {{eta}}

def iota(q):                        # encoding: code points on the unit circle
    return [cos(2*pi*q/N), sin(2*pi*q/N)]

def readout(h):                     # π, partial
    if norm(h) < 0.2:
        return None                 # undefined near the origin
    return round(angle(h) / (2*pi/N)) % N    # index of the sector

def Phi(h, x):                      # A_x = ρ R(2πx/N),  b_x = 0
    return rho * rotate(h, 2*pi*x/N)`,
        },
        {
          title: "Running a word under perturbations of size η (Def. 2.1)",
          src: R`worst = {{worst}}                    # the "worst case" checkbox

def perturbation(eta):              # ‖e‖ = η exactly in the worst case
    return eta * unit_vector(random_angle()) if worst else uniform_in_ball(eta)

h_exact = iota(0)                   # the unperturbed run
cloud   = [iota(0)] * 200           # 200 random η-trajectories
r       = 0.0                       # radius of U*, all η-trajectories of this word
q       = 0

for x in word:
    h_exact = Phi(h_exact, x)
    cloud   = [Phi(h, x) + perturbation(eta) for h in cloud]
    r       = rho * r + eta         # ρR maps a disk of radius r onto one of radius ρr
    q       = (q + x) % N

# (T1) at length t holds iff the whole disk stays in the cell of q_t
def margin(t):                      # distance from the exact run to its cell boundary
    return min(rho**t * sin(pi/N), rho**t - 0.2)

L_star = min(t for t in count(1) if r_t(t) >= margin(t))`,
        },
      ],
    },

    // ------------------------------------------------------------ N3
    "fig-spacetime": {
      file: "js/fig-spacetime.js",
      blocks: [
        {
          title: "One map that executes and restores (Thm 3.1, Figure 1)",
          src: R`edges  = [0.4, 1.2, 2.1, 3.1, 3.6, 4.4]      # the five cells U_q1 … U_q5 (bands)
center = [(edges[k] + edges[k+1]) / 2 for k in range(5)]
half   = [(edges[k+1] - edges[k]) / 2 for k in range(5)]
lam, eta, worst = {{lambda}}, {{eta}}, {{worst}}

def Phi(h, x):
    q = cell_of(h)                               # the readout π(h)
    return center[delta(q, x)] + lam * (h - center[q])
    #      executive organ: go to the next cell
    #                                  restoring organ: shrink by λ (λ = 1: none)

def perturbation(eta):                           # |e| = η exactly in the worst case
    return choice([-eta, eta]) if worst else uniform(-eta, eta)

h = center[0]                                    # one random η-trajectory (grey lines)
for x in word:
    h = Phi(h, x) + perturbation(eta)`,
        },
        {
          title: "The exact η-reachable set, by interval arithmetic",
          src: R`reach = [(center[0], center[0])]    # start at ι(q1)

def step(reach, x):
    pieces = []
    for lo, hi in reach:
        for q in range(5):                       # split at the cell edges
            a, b = max(lo, edges[q]), min(hi, edges[q+1])
            if a <= b:                           # Φ is affine on each cell
                pieces.append((Phi(a, x) - eta, Phi(b, x) + eta))
    return merge_overlapping(pieces)             # pieces in different cells branch

# (T1) fails at t if the reachable set leaves the cell of q_t
# cell inclusion, eq. (4):  Φ_x(U_q) ⊕ B̄_η ⊆ U_δx(q)
holds = all(lam * half[q] + eta <= half[delta(q, x)]
            for q in range(5) for x in ["a", "b"])`,
        },
      ],
    },

    // ------------------------------------------------------------ N4
    "fig-tworates": {
      file: "js/fig-tworates.js",
      blocks: [
        {
          title: "The update map of a symbol that must hold one bit",
          src: R`a, beta, eta, K, worst = {{a}}, {{beta}}, {{eta}}, {{K}}, {{worst}}

maps = {
    "affine": lambda h: a * h,                       # one rate everywhere
    "tanh":   lambda h: tanh(beta * h) / tanh(beta), # multistable, fixed points ±1
    "exact":  lambda h: sign(h),                     # argmax: exact restoration
}
Phi = maps["{{kind}}"]

runs = {"from ι(1)": [+1.0], "from ι(0)": [-1.0]}
for k in range(K):
    for run in runs.values():
        e = choice([-eta, eta]) if worst else uniform(-eta, eta)   # |e| = η in the worst case
        run.append(Phi(run[-1]) + e)`,
        },
        {
          title: "Worst case over all η-trajectories, and the two rates",
          src: R`# Φ is monotone, so the image of an interval is an interval: exact tubes
lo, hi = +1.0, +1.0
for k in range(1, K + 1):
    lo, hi = Phi(lo) - eta, Phi(hi) + eta
    if lo <= 0:
        print("some η-trajectory is misread at k =", k)   # (T1) fails
        break

local_rate = abs(derivative(Phi))      # plotted under the map:
                                       # < 1 inside the cells  → contraction
                                       # > 1 between the cells → separation
attractors = [h for h in fixed_points(Phi) if local_rate(h) < 1]
capacity   = log2(len(attractors))     # robust capacity (Def. F.20, Prop. F.21)`,
        },
      ],
    },

    // ------------------------------------------------------------ N5
    "fig-drift": {
      file: "js/fig-drift.js",
      blocks: [
        {
          title: "A perfectly trained rotation, rounded once (App. E, Figure 3)",
          src: R`N = 7
theta = 2*pi/N                           # exact solution: rotate by 2π/7 (ρ = 1)
bound = pi/N                             # half the angle between two code points

for fmt in ["fp32", "bf16", "fp16", "fp8_e4m3", "fp8_e5m2"]:
    c = round_to(fmt, cos(theta))        # the best weights the format can store,
    s = round_to(fmt, sin(theta))        # rounded once, never again
    # R = [[c, -s], [s, c]] is still a scaled rotation
    r, dtheta = sqrt(c*c + s*s), atan2(s, c) - theta
    t_fail = ceil(bound / abs(dtheta))   # first t with |t·dtheta| >= pi/N`,
        },
        {
          title: "The running realization (left panel), for {{fmt}}",
          src: R`# h_t = R^t h_0 in fp64 from h_0 = (1, 0), in closed form:
t     = {{t}}
angle = t * (theta + dtheta)             # where the state is
q_t   = t % N                            # where it should be: code point ι(q_t)
phase_error = t * dtheta                 # grows linearly in t
norm        = r ** t                     # grows or decays geometrically
read  = round(angle / theta) % N         # the readout: nearest code point
correct = (read == q_t)                  # fails once |phase_error| >= pi/N`,
        },
      ],
    },

    // ------------------------------------------------------------ N6
    "fig-flipflop": {
      file: "js/fig-flipflop.js",
      blocks: [
        {
          title: "Two trackers for the flip-flop",
          src: R`kappa, eta, model, worst = {{kappa}}, {{eta}}, "{{model}}", {{worst}}
# κ: memory factor, the fraction of h that an identity step keeps

h, q = -1.0, 0                        # ι(0) = −1
for x in word:                        # {{L}} letters drawn from {{dist}}
    e = choice([-eta, eta]) if worst else uniform(-eta, eta)
    if   x == "set":   h, q = +1 + e, 1     # A = 0, b = +1
    elif x == "reset": h, q = -1 + e, 0     # A = 0, b = −1
    else:              h = kappa * h + e    # identity: A = κ, this is the only memory
    read = 1 if h > 0 else 0          # π(h)
    errors += (read != q)
    if model == "restore":
        h = +1 if read else -1        # snap back onto the code point`,
        },
        {
          title: "The η-band: all η-trajectories of the word at once",
          src: R`lo, hi = -1.0, -1.0                   # exact interval of h_t (every map is monotone)
for t, x in enumerate(word):
    if   x == "set":   lo, hi = +1 - eta, +1 + eta
    elif x == "reset": lo, hi = -1 - eta, -1 + eta
    else:              lo, hi = kappa * lo - eta, kappa * hi + eta
    misread_by_some = (lo <= 0) if q[t] == 1 else (hi > 0)    # shaded red
    # restoring tracker: the snap then sends [lo, hi] to the code points it touches`,
        },
        {
          title: "The undeclared context window",
          src: R`# after a set, k identities later, the worst-case state is
#   κ^k − η·(1 + κ + … + κ^k)
k_star = min(k for k in count() if kappa**k - eta * sum(kappa**i for i in range(k + 1)) <= 0)

# DFF5 never draws more than 4 identities in a row: definite words, fine if k* > 4
# FF draws runs of any length: some run is always longer than k*`,
        },
      ],
    },

    // ------------------------------------------------------------ N7a
    "fig-scan-tree": {
      file: "js/fig-scan.js",
      blocks: [
        {
          title: "A parallel scan over transition tables (Prop. G.1)",
          src: R`task = {{task}}
tables = [[delta(q, x) for q in Q] for x in word]    # σ_x as a list: σ_x[q]

def compose(later, earlier):                         # a gather: n integer lookups, exact
    return [later[q] for q in earlier]

P, d = tables[:], 1                                  # Hillis–Steele inclusive scan
while d < len(P):                                    # log2(8) = 3 rounds
    P = [compose(P[i], P[i - d]) if i >= d else P[i] for i in range(len(P))]
    d *= 2

states = [P[t][q0] for t in range(len(P))]           # identical to the sequential run,
                                                     # whatever the shape of the tree`,
        },
      ],
    },

    // ------------------------------------------------------------ N7b
    "fig-scan-gap": {
      file: "js/fig-scan.js",
      blocks: [
        {
          title: "One affine model, two evaluation orders (App. D, Figure 2)",
          src: R`rnd = round_to_{{fmt}}                 # applied after EVERY multiply and add
L   = {{L}}
# per letter x and per channel: A_x = ρ_x·e^{iθ_x} (complex), b_x ~ N(0, 0.3²)

# sequential loop
h = 0
for t, x in enumerate(word):
    h = rnd(rnd(A[x] * h) + b[x])
    h_seq[t] = h

# parallel scan over pairs (A, b):  (A2, b2) ∘ (A1, b1) = (A2·A1, A2·b1 + b2)
P, d = [(A[x], b[x]) for x in word], 1
while d < L:
    P = [(rnd(A2 * A1), rnd(rnd(A2 * b1) + b2)) if i >= d else P[i]
         for i, ((A2, b2), (A1, b1)) in enumerate(zip(P, shift(P, d)))]
    d *= 2
h_par = [b for (A, b) in P]              # h_0 = 0, so h_t is the b-part

gap[t] = max over channels |h_par[t] - h_seq[t]|     # mean over 2 words
# NFSM: composites are integer tables → gap is exactly 0`,
        },
      ],
    },

    // ------------------------------------------------------------ N8
    "fig-nfsm": {
      file: "js/fig-nfsm.js",
      blocks: [
        {
          title: "One head, as it runs (eq. 6, eq. 12)",
          src: R`d = {{d}}                                  # attractor indices of the head
theta = {x: logits(x) for x in Sigma}      # d×d each, from the input alone

worst = {{worst}}                          # the "worst case" checkbox
def noise(a, size):                        # entries of size exactly a in the worst case
    return random_signs(size) * a if worst else uniform(-a, a, size=size)

def step(k, x, eta_s={{eta_s}}, Delta={{delta}}):
    # restoring organ: rd(h) reads the largest coordinate of the perturbed state
    h  = one_hot(k, d) + noise(eta_s, d)
    kr = argmax(h)                         # == k whenever eta_s < 1/2
    # executive organ: rd(θ_x · one_hot(kr)) is the argmax of column kr
    col = theta[x][:, kr] + noise(Delta, d)  # logit perturbation
    return argmax(col)                     # = σ_x(kr)`,
        },
        {
          title: "The table, its margin, and the guarantee (Props. G.2, G.3)",
          src: R`sigma = {x: [argmax(theta[x][:, k]) for k in range(d)] for x in Sigma}
realizes = all(sigma[x][q] == delta(q, x) for x in Sigma for q in Q)   # (R3)

def margin(col):                           # gap between the two largest entries
    top2 = sorted(col)[-2:]
    return top2[1] - top2[0]
gamma = min(margin(theta[x][:, k]) for x in Sigma for k in range(d))

guaranteed = Delta < gamma / 2             # tables unchanged, at every length

# clicking a cell (j, k) of θ_x:
theta[x][j, k] = max(theta[x][i, k] for i != j) + 0.3 + noise   # make it the argmax`,
        },
      ],
    },

    // ------------------------------------------------------------ N9
    "fig-heads": {
      file: "js/fig-heads.js",
      blocks: [
        {
          title: "S₃ carried by two heads (Prop. G.4, App. H.6.2)",
          src: R`sigma1 = {"(1 2)": (2, 1, 3), "(2 3)": (1, 3, 2)}   # head 1: 3 indices
sigma2 = {"(1 2)": (2, 1),    "(2 3)": (2, 1)}      # head 2: 2 indices

cups, j, k = [1, 2, 3], 1, 1                 # the arrangement, and the two heads

def read(x):                                 # x swaps two cup positions
    global j, k
    swap_positions(cups, x)                  # the task (drawn on the left)
    j = sigma1[x][j - 1]                     # each head applies its own table,
    k = sigma2[x][k - 1]                     # never looking at the cups

def decode(j, k):                            # c(j, k): unique in S3
    return the arrangement with ball 1 in cup j and parity (k == 2)`,
        },
        {
          title: "Example G.7: one head per point",
          src: R`from math import factorial, ceil, log2
n = {{n}}
one_head  = dict(bits=factorial(n) * ceil(log2(factorial(n))), logits=factorial(n)**2)
n_1_heads = dict(bits=(n - 1) * n * ceil(log2(n)),           logits=n**2)  # shared logits`,
        },
      ],
    },

    // ------------------------------------------------------------ N10
    "fig-results": {
      file: "js/fig-results.js",
      blocks: [
        {
          title: "How a cell of the table is computed (§7, App. H.3)",
          src: R`def sequence_accuracy(model, L):
    words = sample_words(512, length=L)
    return mean(all(model(w)[t] == target(w)[t] for t in range(L)) for w in words)

def failing_length(model, threshold):        # 0.9 for baselines, 1.0 for the NFSM
    for L in tested_lengths:                 # up to what fits on one GPU
        if sequence_accuracy(model, L) < threshold:
            return L
    return ">= " + str(tested_lengths[-1])

cell = median(failing_length(m) for m in seeds_that_learned)   # "k/5" under it`,
        },
        {
          title: "What † means: extraction certifies every length (App. H.6.1)",
          src: R`def certify(model, task):
    sigma = {x: column_argmax(model.logits(x)) for x in Sigma}   # the joint tables
    c, todo = {s0: q0}, [s0]                 # joint index -> task state
    while todo:
        s = todo.pop()
        for x in Sigma:
            s2, q2 = sigma[x](s), delta(c[s], x)
            if s2 in c and c[s2] != q2: return False    # one index, two states
            if model.readout(s2) != q2: return False
            if s2 not in c:
                c[s2] = q2
                todo.append(s2)
    return True    # by induction on |w|: correct on every word, of every length`,
        },
      ],
    },

    // ------------------------------------------------------------ N11
    "fig-tso": {
      file: "js/fig-tso.js",
      blocks: [
        {
          title: "The token-level automaton of TSO (App. H.1)",
          src: R`N, swaps = {{N}}, {{S}}
p = None                                  # first name of the sentence in progress
g = {item: None for item in items[:N]}    # holder of every item

for tok in stream:
    if tok in people:
        if p is None:
            p = tok                       # first name of a sentence
        else:                             # second name: "p swaps with tok ."
            for it in g:
                if   g[it] == p:   g[it] = tok
                elif g[it] == tok: g[it] = p
            p = None
    elif tok in items and not in_question and p is not None:
        g[tok] = p                        # "p has the item ."
        p = None
    # every other token leaves (p, g) unchanged

answer = g[asked_item]                    # read off at the last token`,
        },
      ],
    },

    // ------------------------------------------------------------ N12
    "fig-pca": {
      file: "js/fig-pca.js",
      blocks: [
        {
          title: "How each animation was made (App. H.5, offline)",
          src: R`model = load(task="Z5", seed={{seed}}, layers={{layers}})
word  = ["+1"] * T                               # the constant word (+1)^T
H     = model.last_recurrent_states(word)        # one state per step
Z     = PCA(n_components=2).fit_transform(H)     # fit on this run alone

for window in three_windows(T):                  # early, middle, late
    scatter(Z[window], color=target_state[window])        # t mod 5
    shade(decision_regions(model.classifier, plane=PCA))  # what the readout sees`,
        },
      ],
    },
  };
})();
