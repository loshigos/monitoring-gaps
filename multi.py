"""Experiment 3: multi-action metacognitive control of information acquisition.

Actions: stop (commit to the preferred answer), cheap probe, strong probe,
setup (enables a better strong probe; usable once).  A hidden opportunity
regime z changes channel reliabilities; a cue about z (validity v) is observed
at t = 0.  Utility = correctness - operation costs.

Sufficient state: (p-index, cue, tool, counts per channel configuration).
"""
import numpy as np, time, itertools
from collections import defaultdict

PRIORS = (0.55, 0.70, 0.85, 0.95)
BASE_COST = {'cheap': 0.006, 'strong': 0.035, 'setup': 0.015}
ACTIONS = ('stop', 'cheap', 'strong', 'setup')


def reliabilities(family):
    """q[(channel_config, z)] ; channel configs: 'cheap', 'pre' (strong, tool off), 'post'."""
    if family == 'het':
        return {('cheap', 0): 0.57, ('pre', 0): 0.64, ('post', 0): 0.64,
                ('cheap', 1): 0.72, ('pre', 1): 0.91, ('post', 1): 0.91}
    if family == 'setup':
        return {('cheap', 0): 0.54, ('pre', 0): 0.54, ('post', 0): 0.88,
                ('cheap', 1): 0.58, ('pre', 1): 0.58, ('post', 1): 0.96}
    raise ValueError


CFG = ('cheap', 'pre', 'post')


class Acq:
    def __init__(self, family, H, mult, v, q=None):
        self.family, self.H, self.mult, self.v = family, H, mult, v
        self.q = reliabilities(family) if q is None else q
        self.cost = {a: BASE_COST[a] * mult for a in BASE_COST}
        # latents (pidx, z, y)
        self.latents = [(i, z, y) for i in range(4) for z in (0, 1) for y in (0, 1)]
        self.prior = np.array([0.25 * 0.5 * (PRIORS[i] if y == 1 else 1 - PRIORS[i])
                               for i, z, y in self.latents])

    def init(self):
        out = []
        for i in range(4):
            for cue in (0, 1):
                lik = np.array([(1.0 if pi == i else 0.0) *
                                ((self.v + (1 - self.v) / 2) if z == cue else (1 - self.v) / 2)
                                for pi, z, y in self.latents])
                out.append(((i, cue, 0, (0,) * 6), lik))
        return out

    def ops(self, s):
        i, cue, tool, cnt = s
        acts = ['cheap', 'strong']
        if tool == 0:
            acts.append('setup')
        return acts

    def step(self, s, a, q=None):
        """list of (child, likelihood vector) for action a."""
        q = self.q if q is None else q
        i, cue, tool, cnt = s
        if a == 'setup':
            return [((i, cue, 1, cnt), np.ones(len(self.latents)))]
        cfg = 'cheap' if a == 'cheap' else ('post' if tool else 'pre')
        ci = CFG.index(cfg)
        out = []
        for o in (1, 0):
            lik = np.array([(q[(cfg, z)] if o == y else 1 - q[(cfg, z)]) for pi, z, y in self.latents])
            c = list(cnt)
            c[2 * ci + (0 if o == 1 else 1)] += 1
            out.append(((i, cue, tool, tuple(c)), lik))
        return out


class AcqGraph:
    """Layered DAG of sufficient states under all actions. Layer t = number of ops used."""

    def __init__(self, P: Acq, q_model=None):
        self.P = P
        H = P.H
        nl = len(P.latents)
        q = P.q if q_model is None else q_model
        self.layers = []
        J = []  # joint under the *model* used for beliefs is not needed; store per-state latent likelihood
        lay = {}
        rows = []
        for st, lik in P.init():
            lay[st] = len(rows); rows.append(P.prior * lik)
        self.layers.append(lay)
        self.W = [np.array(rows)]  # unnormalised joint (prior * likelihood of the state's history)
        self.edges = []  # per t: list of (parent, action, child, lik_vector)
        for t in range(H):
            nxt, nrows, E = {}, [], []
            for st, pi in self.layers[t].items():
                for a in P.ops(st):
                    for ch, lik in P.step(st, a, q):
                        if ch not in nxt:
                            nxt[ch] = len(nrows); nrows.append(None)
                        E.append((pi, a, nxt[ch], lik))
            # joint for children: W(child) = W(parent) * lik  (well-defined: same for all parents)
            W = np.zeros((len(nrows), nl))
            done = np.zeros(len(nrows), bool)
            for pi, a, ci, lik in E:
                if not done[ci]:
                    W[ci] = self.W[t][pi] * lik; done[ci] = True
            self.layers.append(nxt); self.W.append(W); self.edges.append(E)
        self.post = [np.where(w.sum(1, keepdims=True) > 0, w / np.maximum(w.sum(1, keepdims=True), 1e-300), 1.0 / w.shape[1]) for w in self.W]
        ymask = np.array([1.0 if y == 1 else 0.0 for i, z, y in P.latents])
        self.p1 = [b @ ymask for b in self.post]
        self.conf = [np.maximum(p, 1 - p) for p in self.p1]
        self.states = [list(l.keys()) for l in self.layers]

    def n(self, t):
        return len(self.layers[t])


def transitions(G: AcqGraph, Ptrue: Acq):
    """True predictive transition probabilities per (t, parent, action) -> list (child, prob),
    using posterior under the TRUE model.  G must be built with the true model."""
    T = {}
    for t, E in enumerate(G.edges):
        for pi, a, ci, lik in E:
            p = float(G.post[t][pi] @ lik)
            T.setdefault((t, pi, a), []).append((ci, p))
    return T


def solve_full(G: AcqGraph, P: Acq, trans, depth=None):
    """Bayes-optimal values (depth=None) or receding-horizon depth-limited action choice.
    Returns Q dict per t: array (n, len(ACTIONS)) with -inf for unavailable."""
    H = P.H
    V = [None] * (H + 1)
    Q = [None] * (H + 1)
    V[H] = G.conf[H].copy()
    Q[H] = np.full((G.n(H), 4), -np.inf); Q[H][:, 0] = G.conf[H]
    for t in range(H - 1, -1, -1):
        q = np.full((G.n(t), 4), -np.inf)
        q[:, 0] = G.conf[t]
        for (tt, pi, a), lst in trans.items():
            if tt != t:
                continue
            q[pi, ACTIONS.index(a)] = -P.cost[a] + sum(p * V[t + 1][ci] for ci, p in lst)
        Q[t] = q
        V[t] = q.max(1)
    return Q, V


def lookahead_policy(G, P, trans, depth):
    """Receding-horizon controller: at each state evaluate sequences of at most `depth`
    operations followed by stopping, choose the first action of the best sequence."""
    H = P.H
    memo = {}

    def val(t, pi, d):
        key = (t, pi, d)
        if key in memo:
            return memo[key]
        best = G.conf[t][pi]
        if d > 0 and t < H:
            for a in ('cheap', 'strong', 'setup'):
                lst = trans.get((t, pi, a))
                if lst is None:
                    continue
                v = -P.cost[a] + sum(p * val(t + 1, ci, d - 1) for ci, p in lst)
                best = max(best, v)
        memo[key] = best
        return best
    act = []
    for t in range(H):
        a_t = np.zeros(G.n(t), int)
        for pi in range(G.n(t)):
            best, ba = G.conf[t][pi], 0
            for a in ('cheap', 'strong', 'setup'):
                lst = trans.get((t, pi, a))
                if lst is None:
                    continue
                v = -P.cost[a] + sum(p * val(t + 1, ci, depth - 1) for ci, p in lst)
                if v > best + 1e-12:
                    best, ba = v, ACTIONS.index(a)
            a_t[pi] = ba
        act.append(a_t)
    return act


def greedy_from_Q(Q, H):
    return [np.argmax(np.where(np.isfinite(Q[t]), Q[t] + np.array([1e-12, 0, 0, 0]), -np.inf), 1) for t in range(H)]


def evaluate_policy(G, P, trans, act):
    """Exact expected utility of a node-action policy under true transitions."""
    H = P.H
    r = np.zeros(G.n(0))
    init = P.init()
    for st, lik in init:
        r[G.layers[0][st]] += float(P.prior @ lik)
    val = 0.0
    for t in range(H):
        nr = np.zeros(G.n(t + 1))
        for pi in np.nonzero(r)[0]:
            a = ACTIONS[act[t][pi]]
            if a == 'stop':
                val += r[pi] * G.conf_true[t][pi]
            else:
                val -= r[pi] * P.cost[a]
                for ci, p in trans[(t, pi, a)]:
                    nr[ci] += r[pi] * p
        r = nr
    val += float(r @ G.conf_true[H])
    return val


def occupancy(G, P, trans, act):
    H = P.H
    occ = []
    r = np.zeros(G.n(0))
    for st, lik in P.init():
        r[G.layers[0][st]] += float(P.prior @ lik)
    for t in range(H):
        occ.append(r.copy())
        nr = np.zeros(G.n(t + 1))
        for pi in np.nonzero(r)[0]:
            a = ACTIONS[act[t][pi]]
            if a != 'stop':
                for ci, p in trans[(t, pi, a)]:
                    nr[ci] += r[pi] * p
        r = nr
    occ.append(r)
    return occ


# ---------------------------------------------------------------------------
# Monitors
# ---------------------------------------------------------------------------

def labels(G, kind, nbins=50):
    out = []
    for t in range(G.P.H + 1):
        cb = np.minimum(np.floor((G.conf[t] - 0.5) / (0.5 / nbins) + 1e-9).astype(int), nbins - 1)
        tool = np.array([s[2] for s in G.states[t]])
        cue = np.array([s[1] for s in G.states[t]])
        if kind == 'conf':
            out.append(cb * 2 + tool)
        elif kind == 'conf+cue':
            out.append((cb * 2 + tool) * 2 + cue)
        else:
            out.append(np.arange(G.n(t)))
    return out


def certify_multi(G, P, trans, lab, time_limit=300, warm=None):
    """MILP: best deterministic monitor-restricted policy for multi-action control."""
    import highspy
    from scipy.sparse import coo_matrix
    H = P.H
    # max-reach bounds R
    R = [None] * (H + 1)
    r0 = np.zeros(G.n(0))
    for st, lik in P.init():
        r0[G.layers[0][st]] += float(P.prior @ lik)
    R[0] = r0
    for t in range(H):
        acc = defaultdict(lambda: defaultdict(float))
        for (tt, pi, a), lst in trans.items():
            if tt == t:
                for ci, p in lst:
                    acc[ci][pi] = max(acc[ci][pi], p)
        nr = np.zeros(G.n(t + 1))
        for ci, d in acc.items():
            nr[ci] = sum(R[t][pi] * p for pi, p in d.items())
        R[t + 1] = np.minimum(nr, 1.0)
    var = {}
    nv = 0
    for t in range(H):
        for pi in range(G.n(t)):
            if R[t][pi] <= 0:
                continue
            avail = ['stop'] + P.ops(G.states[t][pi])
            for a in avail:
                var[(t, pi, a)] = nv; nv += 1
    zv = {}
    for t in range(H):
        for pi in range(G.n(t)):
            if R[t][pi] <= 0:
                continue
            k = int(lab[t][pi])
            for a in ['stop'] + P.ops(G.states[t][pi]):
                if (t, k, a) not in zv:
                    zv[(t, k, a)] = nv; nv += 1
    c = np.zeros(nv)
    for (t, pi, a), j in var.items():
        if a == 'stop':
            c[j] = -G.conf_true[t][pi]
        else:
            v = -P.cost[a]
            if t == H - 1:
                v += sum(p * G.conf_true[H][ci] for ci, p in trans[(t, pi, a)])
            c[j] = -v
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    # flow
    inflow = defaultdict(list)
    for (t, pi, a), j in var.items():
        if a != 'stop' and t + 1 < H:
            for ci, p in trans[(t, pi, a)]:
                inflow[(t + 1, ci)].append((j, p))
    for t in range(H):
        for pi in range(G.n(t)):
            if R[t][pi] <= 0:
                continue
            for a in ['stop'] + P.ops(G.states[t][pi]):
                rows.append(r); cols.append(var[(t, pi, a)]); vals.append(1.0)
            for j, p in inflow.get((t, pi), []):
                rows.append(r); cols.append(j); vals.append(-p)
            val0 = R[0][pi] if t == 0 else 0.0
            lo.append(val0); hi.append(val0); r += 1
    # linking x <= R z
    for (t, pi, a), j in var.items():
        k = int(lab[t][pi])
        rows += [r, r]; cols += [j, zv[(t, k, a)]]; vals += [1.0, -R[t][pi]]
        lo.append(-np.inf); hi.append(0.0); r += 1
    # one action per fiber
    fib = defaultdict(list)
    for (t, k, a), j in zv.items():
        fib[(t, k)].append(j)
    for key, js in fib.items():
        for j in js:
            rows.append(r); cols.append(j); vals.append(1.0)
        lo.append(1.0); hi.append(1.0); r += 1
    A = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsc()
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    h.setOptionValue('mip_rel_gap', 1e-9)
    h.setOptionValue('mip_abs_gap', 1e-12)
    h.setOptionValue('time_limit', float(time_limit))
    h.setOptionValue('primal_feasibility_tolerance', 1e-9)
    h.setOptionValue('mip_feasibility_tolerance', 1e-9)
    h.setOptionValue('threads', 1)
    lp = highspy.HighsLp()
    lp.num_col_ = nv; lp.num_row_ = r
    lp.col_cost_ = c
    ub = np.full(nv, np.inf)
    for j in zv.values():
        ub[j] = 1
    lp.col_lower_ = np.zeros(nv); lp.col_upper_ = ub
    lp.row_lower_ = np.array(lo); lp.row_upper_ = np.array(hi)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices; lp.a_matrix_.value_ = A.data
    integ = [highspy.HighsVarType.kContinuous] * nv
    for j in zv.values():
        integ[j] = highspy.HighsVarType.kInteger
    lp.integrality_ = integ
    h.passModel(lp)
    t0 = time.time()
    h.run()
    info = h.getInfo()
    x = np.array(h.getSolution().col_value)
    pol = {}
    for (t, k, a), j in zv.items():
        if x[j] > 0.5:
            pol[(t, k)] = a
    act = []
    for t in range(H):
        a_t = np.zeros(G.n(t), int)
        for pi in range(G.n(t)):
            k = int(lab[t][pi])
            a_t[pi] = ACTIONS.index(pol.get((t, k), 'stop'))
        act.append(a_t)
    return dict(status=h.modelStatusToString(h.getModelStatus()), time=time.time() - t0,
                value=evaluate_policy(G, P, trans, act), dual=-info.mip_dual_bound,
                act=act, nbin=len(zv))


def brute_force_restricted(G, P, trans, lab):
    """Exhaustive enumeration over deterministic monitor-restricted policies (small cases)."""
    H = P.H
    fibers = []
    for t in range(H):
        for k in np.unique(lab[t]):
            pis = np.where(lab[t] == k)[0]
            avail = ['stop'] + P.ops(G.states[t][pis[0]])
            fibers.append((t, int(k), avail))
    best = -np.inf
    n = 1
    for f in fibers:
        n *= len(f[2])
    if n > 2_000_000:
        return None, n
    for combo in itertools.product(*[f[2] for f in fibers]):
        pol = {(f[0], f[1]): a for f, a in zip(fibers, combo)}
        act = [np.array([ACTIONS.index(pol[(t, int(lab[t][pi]))]) for pi in range(G.n(t))]) for t in range(H)]
        v = evaluate_policy(G, P, trans, act)
        best = max(best, v)
    return best, n
