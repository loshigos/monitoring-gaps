"""Brute-force validation of the MILP certificates on small instances."""
import numpy as np, itertools, json
from multi import *
import core

rows=[]
# multi-action
for fam in ('het','setup'):
  for H in (2,3):
    for v in (0.0,0.5):
      for nb in (2,3):
        P=Acq(fam,H,1.0,v); G=AcqGraph(P); G.conf_true=G.conf; tr=transitions(G,P)
        lab=labels(G,'conf',nbins=nb)
        res=certify_multi(G,P,tr,lab)
        # reachable fibers under any policy
        R=[np.zeros(G.n(t)) for t in range(H+1)]
        for st,lik in P.init(): R[0][G.layers[0][st]]+=float(P.prior@lik)
        for t in range(H):
            for (tt,pi,a),lst in tr.items():
                if tt==t and R[t][pi]>0:
                    for ci,p in lst: R[t+1][ci]+=p
        fibers=[]
        for t in range(H):
            for k in np.unique(lab[t][R[t]>0]):
                pis=np.where((lab[t]==k)&(R[t]>0))[0]
                fibers.append((t,int(k),['stop']+P.ops(G.states[t][pis[0]])))
        n=int(np.prod([len(f[2]) for f in fibers]))
        if n>300000: continue
        best=-np.inf
        for combo in itertools.product(*[f[2] for f in fibers]):
            pol={(f[0],f[1]):a for f,a in zip(fibers,combo)}
            act=[np.array([ACTIONS.index(pol.get((t,int(lab[t][pi])),'stop')) for pi in range(G.n(t))]) for t in range(H)]
            best=max(best,evaluate_policy(G,P,tr,act))
        rows.append(dict(model='acq-'+fam,H=H,v=v,bins=nb,n_policies=n,milp=res['value'],brute=best,diff=res['value']-best))
        print(rows[-1])
# binary families, short horizon
for name,fam in (('A',core.FamilyA(0.4,0.6)),('B',core.FamilyB(0.4,0.85)),('C',core.FamilyC(0.4,0.6))):
    for T in (3,4):
      for nb in (3,4):
        G=core.Graph(fam,T); lam=0.04
        Qs,Qc,V=core.solve_optimal(G,lam)
        labs=core.monitor_labels(G,nbins=nb)
        M=core.node_model_plain(G,lam,labs)
        res=core.certify_milp(M)
        Pac=core.occupancy(M,[np.ones(M.n(t)) for t in range(T)])
        fib=[(t,int(k)) for t in range(T) for k in np.unique(labs[t][Pac[t]>0])]
        if 2**len(fib)>300000: continue
        best=-np.inf
        for combo in itertools.product((0,1),repeat=len(fib)):
            g=[{} for _ in range(T)]
            for (t,k),a in zip(fib,combo): g[t][k]=a
            best=max(best,core.evaluate(M,core.node_actions(M,g)))
        rows.append(dict(model=name,H=T,v=None,bins=nb,n_policies=2**len(fib),milp=res['value'],brute=best,diff=res['value']-best))
        print(rows[-1])
json.dump(rows,open('results/milp_validation.json','w'),indent=1)
print('max |diff| =',max(abs(r['diff']) for r in rows),'n cases',len(rows))
