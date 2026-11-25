import json,glob,numpy as np
from core import *
from run_exp1 import make_family
N=60000
def bayes_dstop(f,par,lam,rho=0.4):
    fam=make_family(f,rho,par); G=Graph(fam); Qs,Qc,V=solve_optimal(G,lam)
    act=[(Qc[t]>Qs[t]).astype(float) for t in range(G.T)]
    M=node_model_plain(G,lam,monitor_labels(G)); r=occupancy(M,act)
    num=den=0.0
    for t in range(G.T):
        irr=irr_post(G,t); lab=monitor_labels(G)[t]
        for k in np.unique(lab):
            m=lab==k
            mI=(r[t][m]*irr[m]).sum(); mR=(r[t][m]*(1-irr[m])).sum()
            if mI*N>=20 and mR*N>=20:
                sI=(r[t][m]*irr[m]*(1-act[t][m])).sum()/mI; sR=(r[t][m]*(1-irr[m])*(1-act[t][m])).sum()/mR
                w=min(mI,mR); num+=w*(sI-sR); den+=w
    return num/den if den>0 else float('nan'), start_value(G,V[0])
E1={(r['family'],r['rho'],r['lam'],r['par']):r for r in map(lambda f:json.load(open(f)),glob.glob('results/exp1/*.json')+glob.glob('results/exp2_refs/*.json'))}
out=[]
conds=[('A',p,0.02) for p in (0.0,0.2,0.4,0.6)]+[('C',p,0.06) for p in (0.0,0.3,0.6,1.0)]
ref={}
for f,p,lam in conds:
    bd,vs=bayes_dstop(f,p,lam)
    key=(f,0.4,lam,p)
    if key in E1: e=E1[key]; vconf=e['Vconf']; gap=e['gap']
    else:
        print('missing',key); continue
    ref[(f,p)]=dict(bayes_dstop=bd,Vstar=vs,Vconf=vconf,gap=gap)
agents=[json.load(open(x)) for x in glob.glob('results/exp2/*.json')]
for a in agents:
    rf=ref[(a['fam'],a['par'])]
    a.update(rf)
    a['z_vs_conf']=(a['ret']-rf['Vconf'])/a['se'] if a['se']>0 else 0.0
    a['capture']=(a['ret']-rf['Vconf'])/rf['gap'] if rf['gap']>1e-9 else None
json.dump(dict(ref={f'{k[0]}_{k[1]}':v for k,v in ref.items()},agents=agents),open('results/summary_exp2.json','w'),indent=1)
for k,v in ref.items(): print(k,{kk:round(vv,5) for kk,vv in v.items()})
for a in sorted(agents,key=lambda a:(a['arch'],a['fam'],a['par'],a['seed'])):
    print(a['arch'],a['fam'],a['par'],a['seed'],'ret %.4f se %.4f Vconf %.4f V* %.4f z %.1f cap %s dstop %.4f [%.4f,%.4f] bayes %.4f abst0 %.3f'%(a['ret'],a['se'],a['Vconf'],a['Vstar'],a['z_vs_conf'],None if a['capture'] is None else round(a['capture'],2),a['dstop'],*a['dstop_ci'],a['bayes_dstop'],a['abstain_t0']))
