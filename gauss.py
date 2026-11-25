"""Family D: Gaussian evidence accumulation with resolvable and unresolvable items.

x_t | y, r ~ N(mu*y, 1) if resolvable, N(0, sigma_u^2) if unresolvable.
Belief (pi, q): pi = P(resolvable | x_1:t), q = P(y = + | resolvable, x_1:t), reflected so q >= 1/2.
The continuous belief process is discretised into a finite MDP on a (pi, q) grid with
bilinear (convex-combination) rounding of the exact Bayesian successor belief, which gives a
proper Markov chain on grid nodes. Restricted-controller optimisation and certification are
exact for this discretised model; transfer to the continuous task is checked by simulation.
"""
import numpy as np
from collections import defaultdict
from scipy.stats import norm

MU, W, T = 0.30, 1.5, 20


def stop_value(c):
    return np.maximum(c - W * (1 - c), 0.0)


class GaussModel:
    def __init__(self, rho, sigma_u=1.8, n_pi=41, n_q=21, n_x=161, T=T):
        self.rho, self.su, self.T = rho, sigma_u, T
        self.pis = np.linspace(0, 1, n_pi)
        self.qs = np.linspace(0.5, 1, n_q)
        xs = np.linspace(-8, 8, n_x)
        self.xs = xs
        self.w = np.gradient(xs)  # quadrature weights (trapezoid-like)
        self.fp = norm.pdf(xs, MU, 1); self.fm = norm.pdf(xs, -MU, 1); self.fu = norm.pdf(xs, 0, sigma_u)
        self._build()

    def _cell(self, pi, q):
        """bilinear rounding: returns list of (i, j, weight)."""
        npi, nq = len(self.pis), len(self.qs)
        fi = pi * (npi - 1); i0 = min(int(np.floor(fi)), npi - 2); a = fi - i0
        fj = (q - 0.5) / 0.5 * (nq - 1); j0 = min(int(np.floor(fj)), nq - 2); b = fj - j0
        out = []
        for di, wi in ((0, 1 - a), (1, a)):
            for dj, wj in ((0, 1 - b), (1, b)):
                ww = wi * wj
                if ww > 1e-12:
                    out.append((i0 + di, j0 + dj, ww))
        return out

    def _succ(self, i, j):
        """transition distribution from grid node (i,j) to grid nodes (independent of t)."""
        pi, q = self.pis[i], self.qs[j]
        lr = q * self.fp + (1 - q) * self.fm
        px = pi * lr + (1 - pi) * self.fu
        wx = px * self.w
        wx = wx / wx.sum()
        acc = defaultdict(float)
        for k in range(len(self.xs)):
            if wx[k] < 1e-14:
                continue
            den = px[k]
            pi2 = pi * lr[k] / den
            q2 = q * self.fp[k] / lr[k] if lr[k] > 0 else 0.5
            if q2 < 0.5:
                q2 = 1 - q2
            for a, b, ww in self._cell(pi2, q2):
                acc[(a, b)] += wx[k] * ww
        return acc

    def _build(self):
        npi, nq = len(self.pis), len(self.qs)
        self.succ = {}
        i0 = int(round((1 - self.rho) * (npi - 1)))
        assert abs(self.pis[i0] - (1 - self.rho)) < 1e-9
        layers = [{(i0, 0): 0}]
        trans = []
        for t in range(self.T):
            nxt = {}
            pa, ch, pr = [], [], []
            for (i, j), idx in layers[t].items():
                if (i, j) not in self.succ:
                    self.succ[(i, j)] = self._succ(i, j)
                for key, p in self.succ[(i, j)].items():
                    if key not in nxt:
                        nxt[key] = len(nxt)
                    pa.append(idx); ch.append(nxt[key]); pr.append(p)
            layers.append(nxt)
            trans.append((np.array(pa), np.array(ch), np.array(pr)))
        self.layers, self.trans = layers, trans
        self.nodes = [sorted(l, key=l.get) for l in layers]
        self.PI = [np.array([self.pis[i] for i, j in n]) for n in self.nodes]
        self.Q = [np.array([self.qs[j] for i, j in n]) for n in self.nodes]
        self.conf = [p * q + (1 - p) / 2 for p, q in zip(self.PI, self.Q)]
        self.S = [stop_value(c) for c in self.conf]
        self.P = [None] * (self.T + 1)
        self.P[0] = np.array([1.0])
        for t in range(self.T):
            pa, ch, pr = trans[t]
            self.P[t + 1] = np.bincount(ch, weights=self.P[t][pa] * pr, minlength=len(layers[t + 1]))

    def n(self, t):
        return len(self.layers[t])


def solve(G, lam):
    T = G.T
    V = [None] * (T + 1); Qs = [None] * (T + 1); Qc = [None] * (T + 1)
    V[T] = G.S[T].copy(); Qs[T] = G.S[T].copy(); Qc[T] = np.full(G.n(T), -np.inf)
    for t in range(T - 1, -1, -1):
        pa, ch, pr = G.trans[t]
        Qc[t] = -lam + np.bincount(pa, weights=pr * V[t + 1][ch], minlength=G.n(t))
        Qs[t] = G.S[t]
        V[t] = np.maximum(Qs[t], Qc[t])
    return Qs, Qc, V


def conf_labels(G, nbins=50):
    return [np.minimum(np.floor((c - 0.5) / (0.5 / nbins) + 1e-9).astype(int), nbins - 1) for c in G.conf]


# ---------------------------------------------------------------------------
# continuous-task simulation with exact Bayesian beliefs
# ---------------------------------------------------------------------------

def simulate_items(rho, sigma_u, n, rng, T=T):
    y = np.where(rng.random(n) < 0.5, 1, -1)
    res = rng.random(n) >= rho
    x = np.where(res[:, None], rng.normal(MU * y[:, None], 1.0, (n, T)), rng.normal(0, sigma_u, (n, T)))
    # beliefs before each decision t = 0..T (after t samples)
    lp = norm.logpdf(x, MU, 1).cumsum(1); lm = norm.logpdf(x, -MU, 1).cumsum(1); lu = norm.logpdf(x, 0, sigma_u).cumsum(1)
    z = np.zeros((n, 1))
    lp = np.concatenate([z, lp], 1); lm = np.concatenate([z, lm], 1); lu = np.concatenate([z, lu], 1)
    a = np.log(max(1 - rho, 1e-300)) + np.log(0.5) + lp
    b = np.log(max(1 - rho, 1e-300)) + np.log(0.5) + lm
    c = np.log(max(rho, 1e-300)) + lu
    m = np.maximum(np.maximum(a, b), c)
    ea, eb, ec = np.exp(a - m), np.exp(b - m), np.exp(c - m)
    s = ea + eb + ec
    pplus = (ea + 0.5 * ec) / s
    conf = np.maximum(pplus, 1 - pplus)
    pires = (ea + eb) / s
    return dict(x=x, y=y, res=res, conf=conf, pplus=pplus, pires=pires)


def run_bin_policy(items, g, lam, nbins=50, T=T):
    """Apply a (t, confidence-bin) stop/continue policy with optimal terminal answer/abstain.
    g: list over t<T of dict bin -> 1 continue. Returns per-item return and stopping time."""
    conf = items['conf']; n = conf.shape[0]
    tau = np.full(n, T)
    alive = np.ones(n, bool)
    for t in range(T):
        b = np.minimum(np.floor((conf[:, t] - 0.5) / (0.5 / nbins) + 1e-9).astype(int), nbins - 1)
        keys = np.array(sorted(int(k) for k in g[t]))
        full = np.array([g[t][str(keys[np.argmin(np.abs(keys - kk))])] if isinstance(next(iter(g[t])), str)
                         else g[t][keys[np.argmin(np.abs(keys - kk))]] for kk in range(nbins)])
        cont = full[b] > 0
        stop_now = alive & ~cont
        tau[stop_now] = t
        alive &= cont
    c = conf[np.arange(n), tau]; pp = items['pplus'][np.arange(n), tau]
    ans = np.where(pp >= 0.5, 1, -1)
    answer = (c - W * (1 - c)) > 0
    correct = ans == items['y']
    rew = np.where(answer, np.where(correct, 1.0, -W), 0.0)
    return rew - lam * tau, tau


def full_grid_voc(G, lam):
    """VOC_t on the full (pi, q) grid, for interpolation in continuous simulations."""
    npi, nq = len(G.pis), len(G.qs)
    cells = [(i, j) for i in range(npi) for j in range(nq)]
    for c in cells:
        if c not in G.succ:
            G.succ[c] = G._succ(*c)
    PI, Qg = np.meshgrid(G.pis, G.qs, indexing='ij')
    S = stop_value(PI * Qg + (1 - PI) / 2)
    V = S.copy()
    vocs = [None] * G.T
    for t in range(G.T - 1, -1, -1):
        Qc = np.full((npi, nq), -lam)
        for (i, j) in cells:
            Qc[i, j] += sum(p * V[a, b] for (a, b), p in G.succ[(i, j)].items())
        vocs[t] = Qc - S
        V = np.maximum(S, Qc)
    return vocs


def interp(grid, G, pi, q):
    npi, nq = len(G.pis), len(G.qs)
    fi = np.clip(pi, 0, 1) * (npi - 1); i0 = np.minimum(np.floor(fi).astype(int), npi - 2); a = fi - i0
    fj = (np.clip(q, 0.5, 1) - 0.5) / 0.5 * (nq - 1); j0 = np.minimum(np.floor(fj).astype(int), nq - 2); b = fj - j0
    return ((1 - a) * (1 - b) * grid[i0, j0] + a * (1 - b) * grid[i0 + 1, j0] + (1 - a) * b * grid[i0, j0 + 1] + a * b * grid[i0 + 1, j0 + 1])


def run_voc_policy(items, G, vocs, lam, T=T):
    pires = items['pires']; pp = items['pplus']; n = pires.shape[0]
    # q = P(y=+ | resolvable) reflected
    tau = np.full(n, T); alive = np.ones(n, bool)
    for t in range(T):
        pi = pires[:, t]
        # P(y=+) = pi*q + (1-pi)/2  ->  q = (p - (1-pi)/2)/pi
        q = np.where(pi > 1e-9, (pp[:, t] - (1 - pi) / 2) / np.maximum(pi, 1e-9), 0.5)
        q = np.maximum(q, 1 - q)
        v = interp(vocs[t], G, pi, q)
        stop_now = alive & (v <= 0)
        tau[stop_now] = t; alive &= (v > 0)
    c = items['conf'][np.arange(n), tau]; p = pp[np.arange(n), tau]
    ans = np.where(p >= 0.5, 1, -1); answer = (c - W * (1 - c)) > 0
    rew = np.where(answer, np.where(ans == items['y'], 1.0, -W), 0.0)
    return rew - lam * tau, tau


def stop_reward(items, t_idx):
    n = items['conf'].shape[0]
    c = items['conf'][np.arange(n), t_idx]; p = items['pplus'][np.arange(n), t_idx]
    ans = np.where(p >= 0.5, 1, -1); answer = (c - W * (1 - c)) > 0
    return np.where(answer, np.where(ans == items['y'], 1.0, -W), 0.0)


def fit_bin_policy(items, lam, nbins=25, T=T):
    """Fitted confidence-restricted controller: backward choice per (t, bin)
    of the action with the higher mean realised return, continuing under the fitted later policy."""
    n = items['conf'].shape[0]
    G = stop_reward(items, np.full(n, T))
    g = [None] * T
    for t in range(T - 1, -1, -1):
        b = np.minimum(np.floor((items['conf'][:, t] - 0.5) / (0.5 / nbins) + 1e-9).astype(int), nbins - 1)
        R = stop_reward(items, np.full(n, t))
        C = -lam + G
        gt = {}
        for k in range(nbins):
            m = b == k
            gt[k] = int(m.any() and C[m].mean() > R[m].mean())
        cont = np.array([gt[k] for k in b]) > 0
        G = np.where(cont, C, R)
        g[t] = gt
    return g
