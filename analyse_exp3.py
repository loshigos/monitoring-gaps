import json,glob,numpy as np
R=[json.load(open(f)) for f in glob.glob('results/exp3/*.json')]
S={'n':len(R),'all_certified':all(r['conf']['status']=='Optimal' and r['conf+cue']['status']=='Optimal' for r in R),
   'max_time':max(max(r['conf']['time'],r['conf+cue']['time']) for r in R)}
tab=[]
for fam in ('het','setup'):
  for v in (0.0,0.5,1.0):
    T=[r for r in R if r['family']==fam and r['v']==v]
    m=lambda f: float(np.mean([f(r) for r in T]))
    row=dict(family=fam,v=v,Vstar=m(lambda r:r['Vstar']),Vorc=m(lambda r:r['Vorc']),
      rep_conf=m(lambda r:r['Vstar']-r['conf']['Vcert']), rep_cue=m(lambda r:r['Vstar']-r['conf+cue']['Vcert']),
      depth1=m(lambda r:r['Vstar']-r['Vdepth1']), depth2=m(lambda r:r['Vstar']-r['Vdepth2']),
      est32=m(lambda r:r['Vstar']-np.mean(r['Vest']['32'])), est128=m(lambda r:r['Vstar']-np.mean(r['Vest']['128'])), est512=m(lambda r:r['Vstar']-np.mean(r['Vest']['512'])),
      thr=m(lambda r:r['Vstar']-r['Vthr']), opt_loss=m(lambda r:r['conf']['Vcert']-r['Vthr']),
      proj_conf=m(lambda r:r['conf']['proj_gap']))
    tab.append(row)
S['table']=tab
gaps=np.array([r['Vstar']-r['conf']['Vcert'] for r in R]); thr=np.array([r['conf']['Vcert']-r['Vthr'] for r in R])
pos=gaps>1e-5
S['ratio_opt_to_rep_median']=float(np.median(thr[pos]/gaps[pos])); S['n_pos']=int(pos.sum())
S['bound_ent_over_gap_median']=float(np.median([r['conf']['bound_ent']/(r['Vstar']-r['conf']['Vcert']) for r in R if r['Vstar']-r['conf']['Vcert']>1e-5]))
S['bound_mode_over_gap_median']=float(np.median([r['conf']['bound_mode']/(r['Vstar']-r['conf']['Vcert']) for r in R if r['Vstar']-r['conf']['Vcert']>1e-5]))
S['bounds_valid']=all(r['conf']['bound_ent']>=r['Vstar']-r['conf']['Vcert']-1e-12 for r in R)
S['max_rep_cue']=float(max(r['Vstar']-r['conf+cue']['Vcert'] for r in R))
S['max_rep_conf']=float(max(r['Vstar']-r['conf']['Vcert'] for r in R))
# estimation seed CI for n=512 overall
for n in ('32','128','512'):
    per_seed=np.array([[r['Vstar']-r['Vest'][n][s] for s in range(20)] for r in R]).mean(0)
    S['est_'+n]=[float(per_seed.mean()),float(per_seed.std(ddof=1)/np.sqrt(20))]
json.dump(S,open('results/summary_exp3.json','w'),indent=1)
for r in tab: print({k:(round(v,5) if isinstance(v,float) else v) for k,v in r.items()})
print({k:v for k,v in S.items() if k!='table'})
