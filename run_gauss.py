"""Family D: certified or bracketed phase diagram over the cost per step (lambda), the share of unresolvable items (rho) and sigma_u."""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from gauss import *
import core

OUT = 'results/gauss'; os.makedirs(OUT, exist_ok=True)
NPI, NQ, NB, TL = 21, 11, 25, 200
LAMS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08)
RHOS = (0.0, 0.2, 0.4, 0.6, 0.8)


def h2(p):
    p = np.clip(p, 1e-15, 1 - 1e-15); return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def bkl(p, q):
    p = np.clip(p, 1e-15, 1 - 1e-15); q = min(max(q, 1e-15), 1 - 1e-15)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def info(G, M, act, Qs, Qc):
    r = core.occupancy(M, act)
    dinfo = rinfo = 0.0
    for t in range(G.T):
        D = Qc[t] - Qs[t]; lab = M.lab[t]; w = r[t]; irr = 1 - G.PI[t]
        for k in np.unique(lab[w > 0]):
            m = (lab == k) & (w > 0); P = w[m].sum(); wn = w[m] / P
            dinfo += P * h2(float(wn @ (D[m] > 1e-12)))
            ib = float(wn @ irr[m]); rinfo += P * float(wn @ bkl(irr[m], ib))
    return dinfo, rinfo


def run(cond):
    su, rho, lam = cond
    fn = f'{OUT}/{su}_{rho}_{lam}.json'
    if os.path.exists(fn):
        return
    t0 = time.time()
    G = GaussModel(rho, sigma_u=su, n_pi=NPI, n_q=NQ)
    Qs, Qc, V = solve(G, lam); Vstar = float(V[0][0])
    M = core.node_model_plain(G, lam, conf_labels(G, NB))
    g, actp = core.projected_policy(M, Qs, Qc); vp = core.evaluate(M, actp)
    starts = [actp, [np.ones(M.n(t)) for t in range(M.T)]] + [[(G.conf[t] < th).astype(float) for t in range(M.T)] for th in (0.6, 0.7, 0.8, 0.9)]
    vca, actca = core.coordinate_ascent(M, starts)
    res = core.certify_milp(M, time_limit=TL, warm=actca)
    best_act = res['act'] if res['value'] >= vca else actca
    vbest = max(res['value'], vca)
    gbest = [{int(k): int(best_act[t][M.lab[t] == k][0]) for k in np.unique(M.lab[t])} for t in range(M.T)]
    dual = min(res['dual'], Vstar)
    di, ri = info(G, M, actp, Qs, Qc)
    # continuous task (exact Bayesian beliefs): Bayes policy via interpolated VOC, fitted restricted controller
    rng = np.random.default_rng(7)
    train = simulate_items(rho, su, 100000, rng); test = simulate_items(rho, su, 50000, rng)
    gfit = fit_bin_policy(train, lam, nbins=NB)
    rc, _ = run_bin_policy(test, gfit, lam, nbins=NB)
    vocs = full_grid_voc(G, lam)
    ro, _ = run_voc_policy(test, G, vocs, lam)
    d = ro - rc
    out = dict(sigma_u=su, rho=rho, lam=lam, Vstar=Vstar, Vproj=vp, Vca=vca, Vbest=vbest, dual=dual, status=res['status'],
               gap_upper=Vstar - vbest, gap_lower=Vstar - dual, gap_proj=Vstar - vp, gap_ca=Vstar - vca,
               milp_time=res['time'], dinfo=di, rinfo=ri,
               mc_opt=float(ro.mean()), mc_conf=float(rc.mean()), mc_gap=float(d.mean()), mc_gap_se=float(d.std() / np.sqrt(len(d))),
               policy=[{str(k): v for k, v in gt.items()} for gt in gbest], fitted=[{str(k): v for k, v in gt.items()} for gt in gfit], wall=time.time() - t0)
    json.dump(out, open(fn, 'w'))
    print(cond, 'gap in [%.4f, %.4f] proj %.4f mc %.4f±%.4f %s %.0fs' % (out['gap_lower'], out['gap_upper'], out['gap_proj'], out['mc_gap'], out['mc_gap_se'], res['status'], out['wall']), flush=True)


if __name__ == '__main__':
    conds = [(1.8, r, l) for l in LAMS for r in RHOS]
    conds += [(1.0, r, l) for l in (0.01, 0.02, 0.04) for r in (0.2, 0.4, 0.6)]
    for c in conds:
        run(c)
