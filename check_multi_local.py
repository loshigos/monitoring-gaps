"""Check that no single-fiber deviation improves the certified multi-action controllers."""
import json, itertools, numpy as np
from multi import *
worst=0; n=0
for fam,H,mult,v in itertools.product(('het','setup'),(2,4,6),(0.5,1.0,2.0),(0.0,0.5,1.0)):
    P=Acq(fam,H,mult,v); G=AcqGraph(P); G.conf_true=G.conf; tr=transitions(G,P)
    for kind in ('conf','conf+cue'):
        lab=labels(G,kind); res=certify_multi(G,P,tr,lab); base=res['value']; act=res['act']
        occ=occupancy(G,P,tr,act)
        for t in range(H):
            for k in np.unique(lab[t]):
                m=lab[t]==k
                if occ[t][m].sum()<=0: continue
                st=G.states[t][np.where(m)[0][0]]
                for a in ['stop']+P.ops(st):
                    ai=ACTIONS.index(a)
                    if ai==act[t][m][0]: continue
                    new=[x.copy() for x in act]; new[t][m]=ai
                    d=evaluate_policy(G,P,tr,new)-base; worst=max(worst,d); n+=1
print('deviations tested',n,'max improvement',worst)
json.dump(dict(n=n,max_improvement=worst),open('results/multi_local_check.json','w'))
