import json,glob,numpy as np
from scipy.stats import spearmanr
R=[json.load(open(f)) for f in glob.glob('results/exp1/*.json')]
R=sorted(R,key=lambda r:(r['family'],r['rho'],r['lam'],r['par']))
A=lambda k: np.array([r[k] for r in R])
fam=A('family'); gap=A('gap'); gam=A('gamma')
res_proj=A('proj_Ired_cum')/A('H_red'); res_opt=A('opt_Ired_cum')/A('H_red')
ident=A('ident15')/A('H_red'); unif=A('uniform_res')/A('H_red')
iact=A('proj_Iact_cum'); iact_opt=A('opt_Iact_cum')
S={}
def sp(x,y): return spearmanr(x,y).correlation
rng=np.random.default_rng(0)
def boot_diff(x1,x2,y,B=2000):
    n=len(y); d=[]
    for b in range(B):
        i=rng.integers(0,n,n); d.append(sp(x1[i],y[i])-sp(x2[i],y[i]))
    return np.percentile(d,[2.5,97.5]).tolist()
S['rho_res_proj']=sp(res_proj,gam); S['rho_res_opt']=sp(res_opt,gam); S['rho_ident']=sp(ident,gam); S['rho_unif']=sp(unif,gam); S['rho_iact']=sp(iact,gam)
S['diff_ci_proj_vs_ident']=boot_diff(res_proj,ident,gam)
S['diff_ci_iact_vs_ident']=boot_diff(iact,ident,gam)
for f in 'ABC':
    m=fam==f
    S['fam_'+f]=dict(res_proj=sp(res_proj[m],gam[m]),ident=sp(ident[m],gam[m]),iact=sp(iact[m],gam[m]),unif=sp(unif[m],gam[m]),
        max_gap=float(gap[m].max()), n=int(m.sum()))
# CA vs certified
ca=A('gap_ca'); pr=A('gap_proj'); d=ca-gap
S['n']=len(R); S['n_certified']=int(sum(r['status']=='Optimal' for r in R))
S['ca_overstates']=int((d>1e-7).sum()); S['ca_max_abs']=float(d.max())
big=gap>1e-4
S['ca_max_rel']=float(np.max(d[big]/gap[big])); 
S['ca_overstates_by_fam']={f:int(((d>1e-7)&(fam==f)).sum()) for f in 'ABC'}
S['gamma_ca_vs_cert_spearman']=sp(A('gamma_ca'),gam)
S['milp_time_median']=float(np.median(A('milp_time'))); S['milp_time_max']=float(A('milp_time').max())
S['milp_time_median_by_fam']={f:float(np.median(A('milp_time')[fam==f])) for f in 'ABC'}
# zero residual -> zero gap
zr=A('proj_Ired_cum')<1e-10
S['zero_res_n']=int(zr.sum()); S['zero_res_maxgap']=float(gap[zr].max()) if zr.any() else None
za=A('proj_Iact_cum')<1e-12
S['zero_iact_n']=int(za.sum()); S['zero_iact_maxgap']=float(gap[za].max()) if za.any() else None
S['n_gap_pos']=int(big.sum()); S['n_gap_zero']=int((gap<1e-9).sum())
# bound anatomy (conditions with certified gap > 1e-4)
def med(x): return float(np.median(x))
pins=A('proj_pinsker'); tv=A('proj_tv'); act=A('proj_act'); mu=A('proj_mu'); low=A('proj_lower')
S['anat']=dict(pinsker_over_gap=med(pins[big]/gap[big]), tv_over_gap=med(tv[big]/gap[big]), proj_over_gap=med(pr[big]/gap[big]),
   pinsker_over_tv=med(pins[big]/tv[big]), tv_over_proj=med(tv[big]/pr[big]),
   act_over_gap=med(act[big]/gap[big]), mu_over_gap=med(mu[big]/gap[big]), act_over_proj=med(act[big]/pr[big]),
   pinsker_over_proj=med(pins[big]/pr[big]),
   act_q=[float(np.percentile(act[big]/gap[big],q)) for q in (10,25,75,90)],
   pins_q=[float(np.percentile(pins[big]/gap[big],q)) for q in (10,25,75,90)],
   lower_over_proj=med(low[big]/pr[big]), lower_le_proj=bool(np.all(low<=pr+1e-12)), act_ge_proj=bool(np.all(act>=pr-1e-12)),
   pins_ge_gap=bool(np.all(pins>=gap-1e-12)), act_ge_gap=bool(np.all(act>=gap-1e-12)))
S['anat_by_fam']={f:dict(pins=med((pins/gap)[big&(fam==f)]), act=med((act/gap)[big&(fam==f)])) for f in 'ABC'}
# steering under the certified optimum
oc=A('opt_conflict')
steer=1-oc[big]/gap[big]
S['steer_median']=med(steer); S['steer_q']=[float(np.percentile(steer,q)) for q in (10,25,75,90)]
S['steer_all']=int((oc[big]<1e-9).sum()); S['steer_gt_half']=int((steer>0.5).sum())
S['steer_by_fam']={f:med((1-oc/np.where(gap>0,gap,1))[big&(fam==f)]) for f in 'ABC'}
lb=A('lb_t0'); S['lb_t0_pos']=int((lb>1e-9).sum()); S['lb_t0_frac_median']=med(lb[lb>1e-9]/gap[lb>1e-9]) if (lb>1e-9).any() else None
S['lb_t0_valid']=bool(np.all(lb<=gap+1e-9))
# examples for text
def get(f,rho,lam,par): return [r for r in R if r['family']==f and r['rho']==rho and r['lam']==lam and abs(r['par']-par)<1e-9][0]
S['ex_A_hi']=get('A',0.4,0.02,0.7); S['ex_A_0']=get('A',0.4,0.02,0.0)
S['ex_C_peak']={str(v):get('C',0.4,0.06,v)['gap'] for v in np.round(np.linspace(0,1,11),3)}
S['ex_B']={str(k):get('B',0.4,0.02,k)['gap'] for k in np.round(np.arange(0.05,1,0.1),3)}
S['max_gap_rel']=max((r['gap']/abs(r['Vconf']) if abs(r['Vconf'])>1e-9 else 0, r['family'],r['par'],r['lam'],r['rho']) for r in R)
S['max_gamma']=float(gam.max())
json.dump(S,open('results/summary_exp1.json','w'),indent=1,default=float)
for k,v in S.items():
    if not k.startswith('ex_'): print(k,v)
print(S['ex_A_hi']['gap'],S['ex_A_hi']['Vconf'],S['ex_A_0']['gap']); print(S['ex_C_peak']); print(S['ex_B'])
