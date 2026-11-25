"""Experiment 3: four sources of metalevel loss in multi-action acquisition."""
import json, os, sys, itertools, time
import numpy as np
from collections import defaultdict
from multi import *

OUT = 'results/exp3'
os.makedirs(OUT, exist_ok=True)


def oracle(P):
    tot = 0.0
    for z in (0, 1):
        P2 = Acq(P.family, P.H, P.mult, 1.0)  # cue fully valid == told regime
        # restrict prior to regime z
        mask = np.array([1.0 if l[1] == z else 0.0 for l in P2.latents])
        P2.prior = P2.prior * mask / (P2.prior * mask).sum()
        G = AcqGraph(P2); G.conf_true = G.conf
        tr = transitions(G, P2)
        Q, V = solve_full(G, P2, tr)
        act = greedy_from_Q(Q, P2.H)
        tot += 0.5 * evaluate_policy(G, P2, tr, act)
    return tot


def projected_multi(G, P, tr, Q, lab):
    H = P.H
    act = []
    r = np.zeros(G.n(0))
    for st, lik in P.init():
        r[G.layers[0][st]] += float(P.prior @ lik)
    rows = []
    for t in range(H):
        a_t = np.zeros(G.n(t), int)
        for k in np.unique(lab[t]):
            m = np.where(lab[t] == k)[0]
            w = r[m]
            q = Q[t][m]
            if w.sum() <= 0:
                a_t[m] = 0
                continue
            fin = np.isfinite(q).all(0)
            avg = np.where(fin, (w[:, None] * np.where(np.isfinite(q), q, 0)).sum(0) / w.sum(), -np.inf)
            a = int(np.argmax(avg + np.array([1e-12, 0, 0, 0])))
            a_t[m] = a
            # fiber diagnostics
            wn = w / w.sum()
            vstar = q.max(1)
            C = float(wn @ vstar) - float(avg.max())
            astar = np.argmax(q + np.array([1e-12, 0, 0, 0]), 1)
            pa = np.array([wn[astar == j].sum() for j in range(4)])
            Hbits = float(-(pa[pa > 0] * np.log2(pa[pa > 0])).sum())
            regret_mode = vstar - np.where(np.isfinite(q[:, int(np.argmax(pa))]), q[:, int(np.argmax(pa))], -1e9)
            Gk = float(regret_mode.max())
            rows.append(dict(t=t, P=float(w.sum()), C=C, Hbits=Hbits, pmax=float(pa.max()), G=Gk,
                             chosen_regret=float(wn @ (vstar - q[:, a]))))
        act.append(a_t)
        nr = np.zeros(G.n(t + 1))
        for pi in np.nonzero(r)[0]:
            a = ACTIONS[a_t[pi]]
            if a != 'stop':
                for ci, p in tr[(t, pi, a)]:
                    nr[ci] += r[pi] * p
        r = nr
    return act, rows


def threshold_family(G, P, tr):
    """Tuned confidence threshold with a fixed acquisition schedule (as in common practice)."""
    H = P.H
    best = (-np.inf, None)
    for sched in ('cheap', 'strong', 'setup-strong'):
        for th in np.linspace(0.5, 1.0, 26):
            act = []
            for t in range(H):
                a_t = np.zeros(G.n(t), int)
                for pi, st in enumerate(G.states[t]):
                    if G.conf[t][pi] >= th - 1e-12:
                        a_t[pi] = 0
                    else:
                        if sched == 'cheap':
                            a_t[pi] = 1
                        elif sched == 'strong':
                            a_t[pi] = 2
                        else:
                            a_t[pi] = 3 if st[2] == 0 else 2
                act.append(a_t)
            v = evaluate_policy(G, P, tr, act)
            if v > best[0]:
                best = (v, (sched, float(th)))
    return best


def estimated_planner(P, n, seed):
    rng = np.random.default_rng(seed)
    qhat = {}
    for key, qv in P.q.items():
        k = rng.binomial(n, qv)
        qhat[key] = (k + 1) / (n + 2)
    # model graph (beliefs under qhat) -- same indexing as true graph
    Gm = AcqGraph(P, q_model=qhat)
    Pm = Acq(P.family, P.H, P.mult, P.v, q=qhat)
    trm = transitions(Gm, Pm)
    Gm.conf_true = Gm.conf
    Q, V = solve_full(Gm, Pm, trm)
    return greedy_from_Q(Q, P.H), Gm


def run(cfg):
    fam, H, mult, v = cfg
    fn = f'{OUT}/{fam}_{H}_{mult}_{v}.json'
    if os.path.exists(fn):
        return json.load(open(fn))
    P = Acq(fam, H, mult, v)
    G = AcqGraph(P); G.conf_true = G.conf
    tr = transitions(G, P)
    Q, V = solve_full(G, P, tr)
    act_star = greedy_from_Q(Q, H)
    Vstar = evaluate_policy(G, P, tr, act_star)
    out = dict(family=fam, H=H, mult=mult, v=v, Vstar=Vstar, Vorc=oracle(P))
    for kind in ('conf', 'conf+cue'):
        lab = labels(G, kind)
        actp, rows = projected_multi(G, P, tr, Q, lab)
        vp = evaluate_policy(G, P, tr, actp)
        res = certify_multi(G, P, tr, lab)
        out[kind] = dict(Vproj=vp, Vcert=res['value'], dual=res['dual'], status=res['status'], time=res['time'],
                         conflict=sum(r['P'] * r['C'] for r in rows),
                         bound_mode=sum(r['P'] * r['G'] * (1 - r['pmax']) for r in rows),
                         bound_ent=sum(r['P'] * r['G'] * r['Hbits'] for r in rows),
                         proj_gap=Vstar - vp)
    # planning depth
    for d in (1, 2):
        a = lookahead_policy(G, P, tr, d)
        out[f'Vdepth{d}'] = evaluate_policy(G, P, tr, a)
    # tuned threshold (restricted to confidence + tool via schedule)
    vt, par = threshold_family(G, P, tr)
    out['Vthr'] = vt; out['thr'] = par
    # estimation
    est = {}
    for n in (32, 128, 512):
        vals = []
        for s in range(20):
            a, Gm = estimated_planner(P, n, 1000 * n + s)
            # true correctness of model-preferred answer
            G.conf_true = [np.where(Gm.p1[t] >= 0.5, G.p1[t], 1 - G.p1[t]) for t in range(H + 1)]
            vals.append(evaluate_policy(G, P, tr, a))
            G.conf_true = G.conf
        est[n] = vals
    out['Vest'] = {str(k): v for k, v in est.items()}
    json.dump(out, open(fn, 'w'))
    print(cfg, 'V*=%.4f conf-gap=%.5f cue-gap=%.5f d1=%.4f d2=%.4f thr=%.4f est512=%.4f' % (
        Vstar, Vstar - out['conf']['Vcert'], Vstar - out['conf+cue']['Vcert'], out['Vdepth1'], out['Vdepth2'], vt,
        np.mean(est[512])), flush=True)
    return out


if __name__ == '__main__':
    cfgs = list(itertools.product(('het', 'setup'), (2, 4, 6), (0.5, 1.0, 2.0), (0.0, 0.5, 1.0)))
    for c in cfgs:
        run(c)
