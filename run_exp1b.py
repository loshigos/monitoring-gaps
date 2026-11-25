"""Experiment 1b: certified gaps for refined monitors (monitor relativity / metacognitive bits)."""
import json, os, sys, time
import numpy as np
from multiprocessing import Pool
from core import *
from run_exp1 import make_family

OUT = 'results/exp1b'
os.makedirs(OUT, exist_ok=True)


def conditions():
    out = []
    for lam in (0.02, 0.06):
        for d in np.round(np.linspace(0, 0.7, 11), 3):
            out.append(('A', 0.4, lam, float(d)))
        for k in np.round(np.arange(0.05, 1.0, 0.1), 3):
            out.append(('B', 0.4, lam, float(k)))
        for v in np.round(np.linspace(0, 1, 11), 3):
            out.append(('C', 0.4, lam, float(v)))
    return out


def run(cond):
    f, rho, lam, par = cond
    fn = f'{OUT}/{f}_{rho}_{lam}_{par}.json'
    if os.path.exists(fn):
        return
    fam = make_family(f, rho, par)
    G = Graph(fam)
    Qs, Qc, V = solve_optimal(G, lam)
    Vstar = start_value(G, V[0])
    out = dict(family=f, rho=rho, lam=lam, par=par, Vstar=Vstar)
    mons = {
        'red1': node_model_plain(G, lam, monitor_labels(G, 'conf+red', rbins=2)),
        'red2': node_model_plain(G, lam, monitor_labels(G, 'conf+red', rbins=4)),
        'red3': node_model_plain(G, lam, monitor_labels(G, 'conf+red', rbins=8)),
    }
    Mp, _ = node_model_prevconf(G, lam)
    mons['prev'] = Mp
    for name, M in mons.items():
        g, actp = projected_policy(M, Qs, Qc)
        res = certify_milp(M, time_limit=300, warm=actp)
        out[name] = dict(value=res['value'], dual=res['dual'], status=res['status'], time=res['time'],
                         gap=Vstar - res['value'], gap_lower=Vstar - min(res['dual'], Vstar))
    json.dump(out, open(fn, 'w'))
    print(cond, {k: round(v['gap'], 5) for k, v in out.items() if isinstance(v, dict)}, flush=True)


if __name__ == '__main__':
    conds = sorted(conditions(), key=lambda c: {'C': 0, 'B': 1, 'A': 2}[c[0]])
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 1) as p:
        p.map(run, conds, chunksize=1)
