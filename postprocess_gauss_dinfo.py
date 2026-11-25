"""Add the decision information and the decision-information bound under the projected controller to each family D result in results/gauss/."""
import json, glob, numpy as np
from gauss import *
import core
from run_gauss import NPI, NQ, NB, bkl
def h2(p):
    p=np.clip(p,1e-15,1-1e-15); return float(-(p*np.log2(p)+(1-p)*np.log2(1-p)))
for f in glob.glob('results/gauss/*.json'):
    r=json.load(open(f))
    G=GaussModel(r['rho'],sigma_u=r['sigma_u'],n_pi=NPI,n_q=NQ); Qs,Qc,V=solve(G,r['lam'])
    M=core.node_model_plain(G,r['lam'],conf_labels(G,NB)); g,act=core.projected_policy(M,Qs,Qc)
    occ=core.occupancy(M,act); di=0.0; bound=0.0
    for t in range(G.T):
        D=Qc[t]-Qs[t]; lab=M.lab[t]; w=occ[t]
        for k in np.unique(lab[w>0]):
            m=(lab==k)&(w>0); P=w[m].sum(); wn=w[m]/P
            pp=float(wn@(D[m]>1e-12)); pm=float(wn@(D[m]<-1e-12))
            if pp+pm>0:
                I=h2(pp/(pp+pm))*(pp+pm)
            else: I=0.0
            di+=P*I
            # decision-information bound term (ties broken towards stop)
            bound+=P*np.abs(D[m]).max()*h2(pp)/2
    r['dinfo']=di; r['bound_dec']=bound
    json.dump(r,open(f,'w'))
print('ok')
