"""Experiment 1: exact, certified analysis of 256 conditions."""
import json, sys, os, time
import numpy as np
from multiprocessing import Pool
from core import *
from core import _bkl

OUT = 'results/exp1'
os.makedirs(OUT, exist_ok=True)


def conditions():
    out = []
    for rho in (0.2, 0.4):
        for lam in (0.01, 0.02, 0.04, 0.06):
            for d in np.round(np.linspace(0, 0.7, 11), 3):
                out.append(('A', rho, lam, float(d)))
            for k in np.round(np.arange(0.05, 1.0, 0.1), 3):
                out.append(('B', rho, lam, float(k)))
            for v in np.round(np.linspace(0, 1, 11), 3):
                out.append(('C', rho, lam, float(v)))
    return out


def make_family(f, rho, par, q=0.65):
    if f == 'A':
        return FamilyA(rho, par, q=q)
    if f == 'B':
        return FamilyB(rho, par)
    return FamilyC(rho, par)


def type_entropy(fam):
    pr = {}
    for (tp, y), p in zip(fam.latents, fam.prior):
        pr[tp] = pr.get(tp, 0) + p
    p = np.array(list(pr.values()))
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def red_entropy(fam):
    pI = sum(p for (tp, y), p in zip(fam.latents, fam.prior) if is_irreducible(fam, tp))
    return float(-(pI * np.log(pI) + (1 - pI) * np.log(1 - pI)))


def identifiability(G, t):
    """I(tau_red; H_t) under always-continue (nats)."""
    irr = irr_post(G, t)
    P = G.P[t]
    pI = float(P @ irr)
    return float(P @ _bkl(irr, pI))


def uniform_residual(G, labels):
    """Policy-free residual reducibility info: mean over t of I(tau;H_t|phi_t) under always-continue."""
    vals = []
    for t in range(1, G.T + 1):
        irr = irr_post(G, t); P = G.P[t]; lab = labels[t]
        tot = 0.0
        for k in np.unique(lab):
            m = lab == k
            w = P[m]; W = w.sum()
            if W <= 0:
                continue
            ib = float(w @ irr[m]) / W
            tot += float(w @ _bkl(irr[m], ib))
        vals.append(tot)
    return float(np.mean(vals))


def analyse(G, M, act, Qs, Qc, lam, label):
    rows = fiber_analysis(G, M, act, Qs, Qc, lam)
    b = bounds_from_rows(rows, M.T, lam)
    Pr = defaultdict(float)
    Ired = 0.0
    Iact = 0.0
    for r in rows:
        Ired += r['P'] * r['Ired']
        Iact += r['P'] * r['Iact']
    b['Ired_cum'] = Ired      # sum_t P(run_t) I(tau;H_t|phi_t,run)  (nats)
    b['Iact_cum'] = Iact      # sum_t P(run_t) I(A*;H_t|phi_t,run) (bits)
    b['Ires_cum'] = sum(r['P'] * r['Ires'] for r in rows)
    return {label + '_' + k: v for k, v in b.items()}


def run(cond, time_limit=300):
    f, rho, lam, par = cond
    fn = f'{OUT}/{f}_{rho}_{lam}_{par}.json'
    if os.path.exists(fn):
        return json.load(open(fn))
    t0 = time.time()
    fam = make_family(f, rho, par)
    G = Graph(fam)
    Qs, Qc, V = solve_optimal(G, lam)
    Vstar = start_value(G, V[0])
    Vorc = oracle_value(fam, lam)
    labels = monitor_labels(G)
    M = node_model_plain(G, lam, labels)
    gproj, actp = projected_policy(M, Qs, Qc)
    Vproj = evaluate(M, actp)
    starts = [actp, [np.ones(M.n(t)) for t in range(M.T)]]
    for th in (0.6, 0.7, 0.8, 0.9):
        starts.append([(G.conf[t] < th).astype(float) for t in range(M.T)])
    Vca, actca = coordinate_ascent(M, starts)
    res = certify_milp(M, time_limit=time_limit, warm=actca)
    Vm = res['value']
    out = dict(family=f, rho=rho, lam=lam, par=par, Vstar=Vstar, Vorc=Vorc, Vproj=Vproj,
               Vca=Vca, Vmilp=Vm, dual=res['dual'], status=res['status'], milp_time=res['time'],
               nbin=res['nbin'])
    Vconf = max(Vm, Vca)
    out['Vconf'] = Vconf
    out['gap'] = Vstar - Vconf
    out['gap_lower'] = Vstar - min(res['dual'], Vstar)  # certified lower bound on gap
    out['gap_ca'] = Vstar - Vca
    out['gap_proj'] = Vstar - Vproj
    den = Vorc - Vconf
    out['gamma'] = out['gap'] / den if den > 1e-12 else 0.0
    out['gamma_ca'] = out['gap_ca'] / (Vorc - Vca) if Vorc - Vca > 1e-12 else 0.0
    out['H_tau'] = type_entropy(fam)
    out['H_red'] = red_entropy(fam)
    out['ident15'] = identifiability(G, 15)
    out['uniform_res'] = uniform_residual(G, labels)
    act_opt = res['act'] if Vm >= Vca else actca
    out.update(analyse(G, M, actp, Qs, Qc, lam, 'proj'))
    out.update(analyse(G, M, act_opt, Qs, Qc, lam, 'opt'))
    # first-step policy-free lower bound on gap: sum_k P(phi_0=k) C_0(k)
    rows0 = fiber_analysis(G, M, [np.ones(M.n(t)) for t in range(M.T)], Qs, Qc, lam)
    out['lb_t0'] = sum(r['P'] * r['C'] for r in rows0 if r['t'] == 0)
    out['wall'] = time.time() - t0
    json.dump(out, open(fn, 'w'))
    print(f, rho, lam, par, 'gap %.5f ca %.5f proj %.5f  milp %s %.1fs' % (out['gap'], out['gap_ca'], out['gap_proj'], res['status'], res['time']), flush=True)
    return out


if __name__ == '__main__':
    conds = conditions()
    # family A last (slowest) so that quick ones finish first
    conds = sorted(conds, key=lambda c: {'C': 0, 'B': 1, 'A': 2}[c[0]])
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as p:
        p.map(run, conds, chunksize=1)
