"""Recompute the analysis of the projected controller for every Experiment 1 result in results/exp1/."""
import json,glob
import numpy as np
from core import *
from run_exp1 import make_family, analyse
for f in glob.glob('results/exp1/*.json'):
    r=json.load(open(f))
    fam=make_family(r['family'],r['rho'],r['par']); G=Graph(fam); lam=r['lam']
    Qs,Qc,V=solve_optimal(G,lam); M=node_model_plain(G,lam,monitor_labels(G))
    g,act=projected_policy(M,Qs,Qc)
    r.update(analyse(G,M,act,Qs,Qc,lam,'proj'))
    json.dump(r,open(f,'w'),default=float)
print('done')
