"""Exact ideal-observer models for the three reducibility families (A, B, C),
monitor-restricted control, MILP certification and information quantities.

Conventions
-----------
Binary answer y in {+,-} (index 0 = '+', 1 = '-'), uniform prior.
Horizon T: at t = 0..T-1 the meta-level chooses continue (cost lam, one more
observation) or stop. At t = T the agent must stop.
Stop value  S(c) = max(c*1 + (1-c)*(-1.5), 0) = max(2.5c - 1.5, 0).
"""
import numpy as np
from collections import defaultdict
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
import time

ERR = 1.5
T_DEFAULT = 25


def stop_value(c):
    return np.maximum(c - ERR * (1 - c), 0.0)


# ---------------------------------------------------------------------------
# Family definitions.  Each family gives
#   latents: list of (type, y)
#   prior:   array over latents
#   init():  list of (state, likelihood vector over latents)   [t = 0]
#   step(state): list of (obs, child_state, likelihood vector over latents)
#   rtype(type): 'R' or 'I' (reducible / irreducible) for info measures
# ---------------------------------------------------------------------------

class FamilyA:
    """Side channel: blank rate differs between reducible and irreducible items."""
    name = 'A'

    def __init__(self, rho, delta, q=0.65, beta=0.3):
        self.rho, self.delta, self.q, self.beta = rho, delta, q, beta
        self.types = ['R', 'I']
        self.latents = [(tp, y) for tp in self.types for y in (0, 1)]
        pt = {'R': 1 - rho, 'I': rho}
        self.prior = np.array([pt[tp] * 0.5 for tp, y in self.latents])
        # per-observation likelihoods; obs 'b','p','m'
        L = {}
        for i, (tp, y) in enumerate(self.latents):
            if tp == 'R':
                pc = (1 - beta) * q
                pw = (1 - beta) * (1 - q)
                L[i] = {'b': beta, 'p': pc if y == 0 else pw, 'm': pw if y == 0 else pc}
            else:
                bb = beta + delta
                L[i] = {'b': bb, 'p': (1 - bb) / 2, 'm': (1 - bb) / 2}
        self.L = {o: np.array([L[i][o] for i in range(len(self.latents))]) for o in 'bpm'}

    def init(self):
        return [((0, 0, 0), np.ones(len(self.latents)))]

    def step(self, s):
        nb, n_p, nm = s
        return [(o, (nb + (o == 'b'), n_p + (o == 'p'), nm + (o == 'm')), self.L[o]) for o in 'bpm']


class FamilyB:
    """Temporal statistics: irreducible items come from a two-state Markov source."""
    name = 'B'

    def __init__(self, rho, kappa, q=0.65):
        self.rho, self.kappa, self.q = rho, kappa, q
        self.types = ['R', 'I']
        self.latents = [(tp, y) for tp in self.types for y in (0, 1)]
        pt = {'R': 1 - rho, 'I': rho}
        self.prior = np.array([pt[tp] * 0.5 for tp, y in self.latents])

    def init(self):
        # state: (n_plus, n_minus, switches, last) ; last = -1 before any obs
        return [((0, 0, 0, -1), np.ones(len(self.latents)))]

    def step(self, s):
        n_p, nm, sw, last = s
        out = []
        for o in (0, 1):  # 0 = '+', 1 = '-'
            lik = np.zeros(len(self.latents))
            for i, (tp, y) in enumerate(self.latents):
                if tp == 'R':
                    lik[i] = self.q if o == y else 1 - self.q
                else:
                    if last < 0:
                        lik[i] = 0.5
                    else:
                        lik[i] = self.kappa if o == last else 1 - self.kappa
            child = (n_p + (o == 0), nm + (o == 1), sw + (last >= 0 and o != last), o)
            out.append((o, child, lik))
        return out


class FamilyC:
    """Static cue at t=0 about item type (easy / hard / hopeless)."""
    name = 'C'

    def __init__(self, rho, v, q_easy=0.75, q_hard=0.6):
        self.rho, self.v = rho, v
        self.types = ['E', 'H', 'X']  # X = hopeless (irreducible)
        self.qs = {'E': q_easy, 'H': q_hard, 'X': 0.5}
        pt = {'E': (1 - rho) / 2, 'H': (1 - rho) / 2, 'X': rho}
        self.latents = [(tp, y) for tp in self.types for y in (0, 1)]
        self.prior = np.array([pt[tp] * 0.5 for tp, y in self.latents])

    def init(self):
        out = []
        for cue in self.types:
            lik = np.array([(self.v + (1 - self.v) / 3) if tp == cue else (1 - self.v) / 3
                            for tp, y in self.latents])
            out.append(((cue, 0, 0), lik))
        return out

    def step(self, s):
        cue, n_p, nm = s
        out = []
        for o in (0, 1):
            lik = np.array([self.qs[tp] if o == y else 1 - self.qs[tp] for tp, y in self.latents])
            out.append((o, (cue, n_p + (o == 0), nm + (o == 1)), lik))
        return out


def is_irreducible(fam, tp):
    return tp in ('I', 'X')


# ---------------------------------------------------------------------------
# Build the layered state graph with joint probabilities P(s, theta)
# ---------------------------------------------------------------------------

class Graph:
    """Layered graph of sufficient states.

    layers[t]: dict state -> index
    J[t]: array (n_t, n_latent) joint probability P(H_t in s, theta) (always-continue)
    edges[t]: list of (parent_idx, child_idx, lik_vector)  between t and t+1
    """

    def __init__(self, fam, T=T_DEFAULT, prior=None):
        self.fam, self.T = fam, T
        prior = fam.prior if prior is None else prior
        nl = len(fam.latents)
        layers, J, edges = [], [], []
        idx, rows = {}, []
        for st, lik in fam.init():
            if st not in idx:
                idx[st] = len(rows)
                rows.append(np.zeros(nl))
            rows[idx[st]] += prior * lik
        layers.append(idx)
        J.append(np.array(rows))
        for t in range(T):
            nidx, nrows, E = {}, [], []
            states = list(layers[t].keys())
            for st in states:
                pi = layers[t][st]
                for o, ch, lik in fam.step(st):
                    if ch not in nidx:
                        nidx[ch] = len(nrows)
                        nrows.append(np.zeros(nl))
                    nrows[nidx[ch]] += J[t][pi] * lik
                    E.append((pi, nidx[ch], lik))
            layers.append(nidx)
            J.append(np.array(nrows))
            edges.append(E)
        self.layers, self.J, self.edges = layers, J, edges
        # derived
        self.P = [j.sum(1) for j in J]                     # P(s) under always-continue
        self.B = [j / np.maximum(p[:, None], 1e-300) for j, p in zip(J, self.P)]  # posterior
        yplus = np.array([1.0 if y == 0 else 0.0 for tp, y in fam.latents])
        self.pplus = [b @ yplus for b in self.B]
        self.conf = [np.maximum(pp, 1 - pp) for pp in self.pplus]
        self.S = [stop_value(c) for c in self.conf]
        # transition matrices (sparse lists): P(child | parent)
        self.trans = []
        for t in range(T):
            E = self.edges[t]
            pa = np.array([e[0] for e in E])
            ch = np.array([e[1] for e in E])
            # P(child|parent) = sum_theta b(theta|parent) lik(theta)
            pr = np.array([self.B[t][e[0]] @ e[2] for e in E])
            self.trans.append((pa, ch, pr))

    def n(self, t):
        return len(self.layers[t])


def solve_optimal(G, lam):
    """Full-information Bayes-optimal values. Returns Qstop, Qcont, V lists."""
    T = G.T
    V = [None] * (T + 1)
    Qs, Qc = [None] * (T + 1), [None] * (T + 1)
    V[T] = G.S[T].copy()
    Qs[T] = G.S[T].copy()
    Qc[T] = np.full(G.n(T), -np.inf)
    for t in range(T - 1, -1, -1):
        pa, ch, pr = G.trans[t]
        EV = np.bincount(pa, weights=pr * V[t + 1][ch], minlength=G.n(t))
        Qc[t] = -lam + EV
        Qs[t] = G.S[t]
        V[t] = np.maximum(Qs[t], Qc[t])
    return Qs, Qc, V


def start_value(G, V0):
    return float(G.P[0] @ V0)


def oracle_value(fam, lam, T=T_DEFAULT):
    """Agent told the type: average over types of the type-conditioned optimum."""
    tot = 0.0
    for tp in fam.types:
        mask = np.array([1.0 if l[0] == tp else 0.0 for l in fam.latents])
        pt = (fam.prior * mask).sum()
        if pt <= 0:
            continue
        G = Graph(fam, T, prior=fam.prior * mask / pt)
        _, _, V = solve_optimal(G, lam)
        tot += pt * start_value(G, V[0])
    return tot


# ---------------------------------------------------------------------------
# Monitors
# ---------------------------------------------------------------------------

def conf_bin(c, nbins=50):
    return np.minimum(np.floor((c - 0.5) / (0.5 / nbins) + 1e-9).astype(int), nbins - 1)


def irr_post(G, t):
    m = np.array([1.0 if is_irreducible(G.fam, l[0]) else 0.0 for l in G.fam.latents])
    return G.B[t] @ m


def monitor_labels(G, kind='conf', nbins=50, rbins=None):
    """Return list over t of integer monitor labels for each state (non-augmented)."""
    out = []
    for t in range(G.T + 1):
        k = conf_bin(G.conf[t], nbins)
        if kind == 'conf':
            lab = k
        elif kind == 'conf+red':  # confidence + quantised P(irreducible)
            r = np.minimum(np.floor(irr_post(G, t) * rbins + 1e-12).astype(int), rbins - 1)
            lab = k * rbins + r
        elif kind == 'full':
            lab = np.arange(G.n(t))
        else:
            raise ValueError(kind)
        out.append(lab)
    return out


# ---------------------------------------------------------------------------
# Generic layered "node" model used by policy evaluation and MILP.
# Nodes may be sufficient states or augmented states (parent, state).
# ---------------------------------------------------------------------------

class NodeModel:
    """Layered model: init mass p0 over layer-0 nodes, transitions, values.

    S[t]   stop value per node; lam continue cost
    trans[t] (pa, ch, pr) transitions from layer t to t+1
    lab[t] monitor label per node
    sid[t] underlying sufficient-state index per node (for V*, Q*)
    """

    def __init__(self, p0, S, trans, lab, sid, T, lam):
        self.p0, self.S, self.trans, self.lab, self.sid, self.T, self.lam = p0, S, trans, lab, sid, T, lam

    def n(self, t):
        return len(self.S[t])


def node_model_plain(G, lam, labels):
    return NodeModel(G.P[0], G.S, G.trans, labels, [np.arange(G.n(t)) for t in range(G.T + 1)], G.T, lam)


def node_model_prevconf(G, lam, nbins=50):
    """Augment each state with its parent so that previous confidence is observable.

    Node at t>=1 = (parent sufficient state, state). Monitor = (bin(c_t), bin(c_{t-1})).
    Augmented node probability transitions: from (p,s) to (s,c) with P(c|s).
    """
    T = G.T
    # layer 0: plain states
    nodes = [[(None, i) for i in range(G.n(0))]]
    for t in range(T):
        pa, ch, pr = G.trans[t]
        pairs = sorted(set(zip(pa.tolist(), ch.tolist())))
        nodes.append(pairs)
    index = [{nd: j for j, nd in enumerate(L)} for L in nodes]
    S, lab, sid, trans = [], [], [], []
    for t in range(T + 1):
        s_idx = np.array([nd[1] for nd in nodes[t]])
        sid.append(s_idx)
        S.append(G.S[t][s_idx])
        kb = conf_bin(G.conf[t][s_idx], nbins)
        if t == 0:
            lab.append(kb * (nbins + 1) + nbins)
        else:
            p_idx = np.array([nd[0] for nd in nodes[t]])
            kp = conf_bin(G.conf[t - 1][p_idx], nbins)
            lab.append(kb * (nbins + 1) + kp)
    # transitions between augmented layers
    for t in range(T):
        pa, ch, pr = G.trans[t]
        # aggregate P(child | state) over edges (several obs can map to same child)
        agg = defaultdict(float)
        for a, c, p in zip(pa.tolist(), ch.tolist(), pr.tolist()):
            agg[(a, c)] += p
        by_state = defaultdict(list)
        for (a, c), p in agg.items():
            by_state[a].append((c, p))
        A, C, Pp = [], [], []
        for j, (prev, s) in enumerate(nodes[t]):
            for c, p in by_state[s]:
                A.append(j)
                C.append(index[t + 1][(s, c)])
                Pp.append(p)
        trans.append((np.array(A), np.array(C), np.array(Pp)))
    p0 = G.P[0].copy()
    return NodeModel(p0, S, trans, lab, sid, T, lam), nodes


# ---------------------------------------------------------------------------
# Policy evaluation for monitor-restricted policies
# policy: list over t<T of dict label -> 1 (continue) / 0 (stop); default stop
# ---------------------------------------------------------------------------

def occupancy(M, act):
    """Forward running occupancy r_t over nodes given per-node actions act[t] (bool cont)."""
    r = [None] * (M.T + 1)
    r[0] = M.p0.copy()
    for t in range(M.T):
        pa, ch, pr = M.trans[t]
        flow = r[t][pa] * act[t][pa] * pr
        r[t + 1] = np.bincount(ch, weights=flow, minlength=M.n(t + 1))
    return r


def node_actions(M, g):
    return [np.array([g[t].get(int(k), 0) for k in M.lab[t]], dtype=float) for t in range(M.T)]


def evaluate(M, act):
    r = occupancy(M, act)
    val = 0.0
    for t in range(M.T):
        val += float(r[t] @ ((1 - act[t]) * M.S[t])) - M.lam * float(r[t] @ act[t])
    val += float(r[M.T] @ M.S[M.T])
    return val


def backward_values(M, act):
    """Value-to-go under fixed node actions."""
    W = [None] * (M.T + 1)
    W[M.T] = M.S[M.T].copy()
    for t in range(M.T - 1, -1, -1):
        pa, ch, pr = M.trans[t]
        EV = np.bincount(pa, weights=pr * W[t + 1][ch], minlength=M.n(t))
        W[t] = np.where(act[t] > 0, -M.lam + EV, M.S[t])
    return W


def projected_policy(M, Qs, Qc):
    """Forward projected policy: in each label choose action maximizing occupancy-weighted Q*."""
    g = []
    act = []
    r = M.p0.copy()
    for t in range(M.T):
        qs = Qs[t][M.sid[t]]
        qc = Qc[t][M.sid[t]]
        labs = M.lab[t]
        ws = defaultdict(float)
        wc = defaultdict(float)
        for k, rr, a, b in zip(labs.tolist(), r.tolist(), qs.tolist(), qc.tolist()):
            if rr > 0:
                ws[k] += rr * a
                wc[k] += rr * b
        gt = {k: int(wc[k] > ws[k] + 1e-15) for k in ws}
        g.append(gt)
        a_t = np.array([gt.get(int(k), 0) for k in labs], dtype=float)
        act.append(a_t)
        pa, ch, pr = M.trans[t]
        r = np.bincount(ch, weights=r[pa] * a_t[pa] * pr, minlength=M.n(t + 1))
    return g, act


def coordinate_ascent(M, starts, max_sweeps=50):
    """Best-response coordinate ascent over steps (the local search used previously)."""
    best_val, best_act = -np.inf, None
    for act0 in starts:
        act = [a.copy() for a in act0]
        val = evaluate(M, act)
        for sweep in range(max_sweeps):
            changed = False
            for t in range(M.T - 1, -1, -1):
                r = occupancy(M, act)
                W = backward_values(M, act)
                pa, ch, pr = M.trans[t]
                EV = np.bincount(pa, weights=pr * W[t + 1][ch], minlength=M.n(t))
                qc = -M.lam + EV
                qs = M.S[t]
                labs = M.lab[t]
                ws = defaultdict(float)
                wc = defaultdict(float)
                for k, rr, a, b in zip(labs.tolist(), r[t].tolist(), qs.tolist(), qc.tolist()):
                    ws[k] += rr * a
                    wc[k] += rr * b
                new = act[t].copy()
                cur = {}
                for k in ws:
                    cur[k] = 1 if wc[k] > ws[k] + 1e-13 else (0 if ws[k] > wc[k] + 1e-13 else None)
                for j, k in enumerate(labs.tolist()):
                    if cur[k] is not None:
                        new[j] = cur[k]
                if np.any(new != act[t]):
                    act[t] = new
                    changed = True
            nv = evaluate(M, act)
            if nv <= val + 1e-14 and not changed:
                break
            val = nv
            if not changed:
                break
        val = evaluate(M, act)
        if val > best_val:
            best_val, best_act = val, act
    return best_val, best_act


def threshold_start(M, thr):
    # continue while confidence below thr ; uses stop values monotone in conf
    return None


# ---------------------------------------------------------------------------
# MILP certification of the best monitor-restricted deterministic policy
# ---------------------------------------------------------------------------

def certify_milp(M, Pac=None, time_limit=600, rel_gap=1e-9, warm=None):
    """Exact best monitor-restricted deterministic policy by MILP.

    Scaled formulation: for each reachable node n at step t, a_t(n) and b_t(n) are the
    fractions of its always-continue mass U_t(n) that continue / stop.  Then
    a + b = sum_p w(p->n) a_{t-1}(p) with convex weights w = U_p P(n|p) / U_n,
    a_t(n) <= z_t(k(n)),  b_t(n) <= 1 - z_t(k(n)),  z binary.
    Returns dict with exact value of the returned policy and the solver's dual bound.
    """
    T = M.T
    if Pac is None:
        allc = [np.ones(M.n(t)) for t in range(T)]
        Pac = occupancy(M, allc)
    reach = [Pac[t] > 0 for t in range(T)]
    off = 0
    A_, B_ = [], []
    for t in range(T):
        n = int(reach[t].sum())
        ids = -np.ones(M.n(t), int)
        ids[reach[t]] = off + np.arange(n); A_.append(ids.copy()); off += n
        ids2 = -np.ones(M.n(t), int)
        ids2[reach[t]] = off + np.arange(n); B_.append(ids2); off += n
    zmap = []
    for t in range(T):
        labs = np.unique(M.lab[t][reach[t]])
        zmap.append({int(k): off + j for j, k in enumerate(labs)}); off += len(labs)
    nvar = off
    c = np.zeros(nvar)
    for t in range(T):
        U = Pac[t][reach[t]]
        c[B_[t][reach[t]]] -= U * M.S[t][reach[t]]
        if t < T - 1:
            c[A_[t][reach[t]]] -= U * (-M.lam)
        else:
            pa, ch, pr = M.trans[t]
            ev = np.bincount(pa, weights=pr * M.S[T][ch], minlength=M.n(t))
            c[A_[t][reach[t]]] -= U * (-M.lam + ev[reach[t]])
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    rowid = []
    for t in range(T):
        rid = -np.ones(M.n(t), int)
        js = np.where(reach[t])[0]
        rid[js] = r + np.arange(len(js))
        rows += np.repeat(rid[js], 2).tolist()
        cols += np.ravel(np.column_stack([A_[t][js], B_[t][js]])).tolist()
        vals += [1.0] * (2 * len(js))
        if t == 0:
            lo += [1.0] * len(js); hi += [1.0] * len(js)
        else:
            lo += [0.0] * len(js); hi += [0.0] * len(js)
            pa, ch, pr = M.trans[t - 1]
            m = reach[t - 1][pa] & reach[t][ch]
            w = Pac[t - 1][pa[m]] * pr[m] / Pac[t][ch[m]]
            rows += rid[ch[m]].tolist(); cols += A_[t - 1][pa[m]].tolist(); vals += (-w).tolist()
        r += len(js)
        rowid.append(rid)
    for t in range(T):
        js = np.where(reach[t])[0]
        zs = np.array([zmap[t][int(k)] for k in M.lab[t][js]])
        n = len(js)
        rr = r + np.arange(n)
        rows += np.repeat(rr, 2).tolist(); cols += np.ravel(np.column_stack([A_[t][js], zs])).tolist()
        vals += [1.0, -1.0] * n; lo += [-np.inf] * n; hi += [0.0] * n; r += n
        rr = r + np.arange(n)
        rows += np.repeat(rr, 2).tolist(); cols += np.ravel(np.column_stack([B_[t][js], zs])).tolist()
        vals += [1.0, 1.0] * n; lo += [-np.inf] * n; hi += [1.0] * n; r += n
    Amat = coo_matrix((vals, (rows, cols)), shape=(r, nvar)).tocsc()
    integrality = np.zeros(nvar)
    for t in range(T):
        for k, v in zmap[t].items():
            integrality[v] = 1
    import highspy
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    h.setOptionValue('mip_rel_gap', rel_gap)
    h.setOptionValue('mip_abs_gap', 1e-12)
    h.setOptionValue('time_limit', float(time_limit))
    h.setOptionValue('primal_feasibility_tolerance', 1e-9)
    h.setOptionValue('mip_feasibility_tolerance', 1e-9)
    h.setOptionValue('threads', 1)
    lp = highspy.HighsLp()
    lp.num_col_ = nvar; lp.num_row_ = r
    lp.col_cost_ = c
    lp.col_lower_ = np.zeros(nvar); lp.col_upper_ = np.ones(nvar)
    lp.row_lower_ = np.array(lo, float); lp.row_upper_ = np.array(hi, float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = Amat.indptr; lp.a_matrix_.index_ = Amat.indices; lp.a_matrix_.value_ = Amat.data
    lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous for v in integrality]
    h.passModel(lp)
    if warm is not None:
        # warm start from node actions: build full solution vector
        x0 = np.zeros(nvar)
        r_occ = occupancy(M, warm)
        for t in range(T):
            js = np.where(reach[t])[0]
            frac = r_occ[t][js] / Pac[t][js]
            x0[A_[t][js]] = frac * warm[t][js]
            x0[B_[t][js]] = frac * (1 - warm[t][js])
            for k, v in zmap[t].items():
                sel = (M.lab[t] == k) & reach[t]
                x0[v] = warm[t][sel][0]
        sol = highspy.HighsSolution(); sol.col_value = list(x0); sol.value_valid = True
        h.setSolution(sol)
    t0 = time.time()
    h.run()
    el = time.time() - t0
    info = h.getInfo()
    st = h.modelStatusToString(h.getModelStatus())
    out = dict(status=st, time=el, nbin=int(integrality.sum()))
    x = np.array(h.getSolution().col_value)
    if len(x) == nvar:
        g = [{k: int(round(x[v])) for k, v in zmap[t].items()} for t in range(T)]
        act = node_actions(M, g)
        out['value'] = evaluate(M, act)
        out['milp_obj'] = -info.objective_function_value
        out['dual'] = -info.mip_dual_bound
        out['gap'] = info.mip_gap
        out['g'] = g
        out['act'] = act
    return out


def lp_relaxation_bound(M, Pac=None):
    """Upper bound on restricted value from the LP relaxation (for reporting)."""
    pass


# ---------------------------------------------------------------------------
# Information quantities and bounds under a given policy's occupancy
# ---------------------------------------------------------------------------

def reflect_latent(G, t):
    """Posterior over control-relevant latent Theta' = (type, z) where z = 1 iff y equals
    current answer (answer '+' when p+ >= 0.5). Family B additionally gets the
    observed last symbol relative to the answer (degenerate component)."""
    fam = G.fam
    B = G.B[t]
    ans = (G.pplus[t] < 0.5).astype(int)  # 0 => '+'
    types = fam.types
    out = np.zeros((G.n(t), len(types) * 2))
    for i, (tp, y) in enumerate(fam.latents):
        ti = types.index(tp)
        z_correct = (y == ans)  # vector over states
        out[np.arange(G.n(t)), ti * 2 + 0] += B[:, i] * z_correct
        out[np.arange(G.n(t)), ti * 2 + 1] += B[:, i] * (~z_correct)
    if fam.name == 'B':
        states = list(G.layers[t].keys())
        last = np.array([s[3] for s in states])
        agree = np.where(last < 0, 2, (last == ans).astype(int))  # 0/1, 2 = no obs yet
        ext = np.zeros((G.n(t), out.shape[1] * 3))
        for j in range(G.n(t)):
            ext[j, agree[j] * out.shape[1]:(agree[j] + 1) * out.shape[1]] = out[j]
        out = ext
    return out


def kl(p, q):
    m = p > 0
    return float(np.sum(p[m] * (np.log(p[m]) - np.log(np.maximum(q[m], 1e-300)))))


def h2(p):
    p = np.clip(p, 0, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        v = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return np.nan_to_num(v)


def fiber_analysis(G, M, act, Qs, Qc, lam, L=None):
    """For a policy (node actions), compute per (t, label):
    occupancy, conflict C, residual latent info (nats), residual reducibility info,
    action info (bits), action-gap max/min, TV-based bound terms.
    Returns list of dict rows."""
    T = M.T
    r = occupancy(M, act)
    if L is None:
        L = 1.0 + ERR + lam * T
    rows = []
    for t in range(T):
        sid = M.sid[t]
        beta = reflect_latent(G, t)[sid]
        irr = irr_post(G, t)[sid]
        D = (Qc[t] - Qs[t])[sid]
        Vs = np.maximum(Qc[t], Qs[t])[sid]
        labs = M.lab[t]
        w_all = r[t]
        for k in np.unique(labs[w_all > 0]):
            m = (labs == k) & (w_all > 0)
            w = w_all[m]
            P = w.sum()
            wn = w / P
            Dk = D[m]
            Ep = float(wn @ np.maximum(Dk, 0)); Em = float(wn @ np.maximum(-Dk, 0))
            C = min(Ep, Em)
            bbar = wn @ beta[m]
            Ires = max(0.0, float(sum(wi * kl(b, bbar) for wi, b in zip(wn, beta[m]))))
            tv = float(wn @ (0.5 * np.abs(beta[m] - bbar).sum(1)))
            # reducibility residual: I(tau; H | k) with binary irr posterior
            ibar = float(wn @ irr[m])
            Ired = float(wn @ (_bkl(irr[m], ibar)))
            tol = 1e-12
            pplus = float(wn @ (Dk > tol))
            pminus = float(wn @ (Dk < -tol))
            p0 = max(0.0, 1.0 - pplus - pminus)
            Iact = float(h2(pplus))          # ties broken towards stop
            absD = np.abs(Dk)
            Gmax = float(absD.max())
            nz = absD[absD > tol]
            gmin = float(nz.min()) if len(nz) else 0.0
            mu = min(pplus, pminus)          # minority mass among non-tied histories
            qn = pplus / (pplus + pminus) if pplus + pminus > 0 else 0.0
            Iact_nt = float(h2(qn))          # decision information among non-tied histories
            # local Lipschitz constant: spread of conditional plan values -> use
            rows.append(dict(t=t, k=int(k), P=P, C=C, Ires=Ires, Ired=Ired, TV=tv, Iact=Iact, Iact_nt=Iact_nt, p0=p0,
                             Gmax=Gmax, gmin=gmin, mu=mu, Ep=Ep, Em=Em,
                             chosen=int(act[t][m][0]),
                             regret_chosen=(Em if act[t][m][0] > 0 else Ep)))
    return rows


def _bkl(p, q):
    p = np.clip(p, 1e-300, 1); q = min(max(q, 1e-300), 1 - 1e-16)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(p > 0, p * np.log(p / q), 0.0)
        b = np.where(p < 1, (1 - p) * np.log(np.maximum(1 - p, 1e-300) / (1 - q)), 0.0)
    return np.nan_to_num(a + b)


def bounds_from_rows(rows, T, lam, L=None):
    if L is None:
        L = 1.0 + ERR + lam * T
    gap_exact = sum(r['P'] * r['regret_chosen'] for r in rows)
    conflict_sum = sum(r['P'] * r['C'] for r in rows)
    # Theorem-2 style (per time: Jensen over k)
    by_t = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    for r in rows:
        bt = by_t[r['t']]
        bt[0] += r['P']               # P(run at t)
        bt[1] += r['P'] * r['Ires']   # P(run) * I(k)  (unnormalised)
        bt[2] += r['P'] * r['TV']
        bt[3] += r['P'] * r['Gmax'] * r['Iact'] / 2
        bt[4] += r['P'] * r['Gmax'] * r['mu']
    pinsker = 0.0
    tvb = 0.0
    for t, (pr, pi, ptv, pa, pmu) in by_t.items():
        if pr > 0:
            pinsker += L * pr * np.sqrt(2 * max(pi, 0.0) / pr)
            tvb += 2 * L * ptv
    act_bound = sum(v[3] for v in by_t.values())
    mu_bound = sum(v[4] for v in by_t.values())
    lower = sum(r['P'] * r['gmin'] * (1 - r['p0']) * (r['Iact_nt'] / 2) ** 2 for r in rows)
    lower_mu = sum(r['P'] * r['gmin'] * r['mu'] for r in rows)
    return dict(gap=gap_exact, conflict=conflict_sum, pinsker=pinsker, tv=tvb,
                act=act_bound, mu=mu_bound, lower=lower, lower_mu=lower_mu)
