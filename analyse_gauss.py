"""Analysis of family D: phase-diagram figure and comparison of the recurrent agents with the certified gaps."""
import json, glob, numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figs
from gauss import simulate_items, run_bin_policy, T

R = {(r['sigma_u'], r['rho'], r['lam']): r for r in map(lambda f: json.load(open(f)), glob.glob('results/gauss/*.json'))}
LAMS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08); RHOS = (0.0, 0.2, 0.4, 0.6, 0.8)
S = {}
r18 = [R[(1.8, r, l)] for l in LAMS for r in RHOS]
S['n_cells'] = len(r18); S['n_closed'] = sum(r['status'] == 'Optimal' for r in r18)
S['max_width'] = max(r['gap_upper'] - r['gap_lower'] for r in r18)
S['peak'] = max(r18, key=lambda r: r['gap_lower'])
S['peak'] = {k: S['peak'][k] for k in ('lam', 'rho', 'gap_lower', 'gap_upper', 'Vstar', 'mc_gap', 'mc_gap_se')}
S['axis_lam0_max'] = max(r['gap_upper'] for r in r18 if r['lam'] == 0.0)
S['axis_rho0_max'] = max(r['gap_upper'] for r in r18 if r['rho'] == 0.0)
S['pos_lower'] = sum(r['gap_lower'] > 1e-4 for r in r18)
S['zero_dinfo_zero_gap'] = all((r['dinfo'] < 1e-9) == (r['gap_upper'] < 1e-9) for r in r18)
S['n_zero'] = sum(r['gap_upper'] < 1e-9 for r in r18)
S['proj_over_ca_max'] = max((r['gap_proj'] / r['gap_upper']) for r in r18 if r['gap_upper'] > 1e-3)
S['steer_example'] = {k: R[(1.8, 0.8, 0.02)][k] for k in ('gap_proj', 'gap_upper', 'gap_lower')}
r10 = [r for k, r in R.items() if k[0] == 1.0]
S['su1'] = dict(max_upper=max(r['gap_upper'] for r in r10), max_rinfo=max(r['rinfo'] for r in r10),
                min_rinfo18=min(r['rinfo'] for r in r18 if r['gap_upper'] > 1e-3))
pairs = [(R[(1.0, r, l)], R[(1.8, r, l)]) for l in (0.01, 0.02, 0.04) for r in (0.2, 0.4, 0.6)]
S['su_pairs'] = [(a['lam'], a['rho'], a['gap_upper'], b['gap_upper']) for a, b in pairs]
S['mc_vs_grid'] = [(r['lam'], r['rho'], r['mc_gap'], r['gap_upper']) for r in r18 if r['gap_upper'] > 1e-3]
S['mc_opt_vs_grid_Vstar_maxdiff'] = max(abs(r['mc_opt'] - r['Vstar']) for r in r18)
# figure 8: phase diagram (lower and upper bounds) + sigma contrast
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.9), gridspec_kw=dict(wspace=0.45))
for ax, key, title in zip(axs[:2], ('gap_lower', 'gap_upper'), ('certified lower bound', 'best controller (upper bound)')):
    Z = np.array([[R[(1.8, r, l)][key] for l in LAMS] for r in RHOS])
    im = ax.imshow(np.maximum(Z, 0), origin='lower', cmap='Blues', vmin=0, vmax=0.046, aspect='auto')
    ax.set_xticks(range(len(LAMS))); ax.set_xticklabels([f'{l:g}' for l in LAMS], rotation=45, fontsize=7)
    ax.set_yticks(range(len(RHOS))); ax.set_yticklabels([f'{r:g}' for r in RHOS])
    ax.set_xlabel('cost per step $\\lambda$'); ax.set_title(title, fontsize=8.5); ax.grid(False)
    for i in range(len(RHOS)):
        for j in range(len(LAMS)):
            v = Z[i, j]
            ax.text(j, i, f'{1000*max(v,0):.0f}', ha='center', va='center', fontsize=6, color='white' if v > 0.025 else '#0b0b0b')
axs[0].set_ylabel('share of unresolvable items $\\rho$')
axs[1].set_yticklabels([])
ax = axs[2]
x = np.arange(len(pairs))
ax.bar(x - 0.2, [p[3] for p in S['su_pairs']], 0.38, color='#2a78d6', label='$\\sigma_u=1.8$', zorder=3)
ax.bar(x + 0.2, [p[2] for p in S['su_pairs']], 0.38, color='#eb6834', label='$\\sigma_u=1.0$', zorder=3)
ax.set_xticks(x); ax.set_xticklabels([f'{p[0]:g} / {p[1]:g}' for p in S['su_pairs']], fontsize=6, rotation=60)
ax.set_xlabel('$\\lambda$ / $\\rho$', fontsize=8); ax.set_ylabel('gap (upper bound)'); ax.legend(fontsize=7)
ax.grid(axis='x', visible=False); ax.set_title('variance channel', fontsize=8.5)
for i, a in enumerate(axs):
    a.text(-0.12 if i else -0.25, 1.06, 'abc'[i], transform=a.transAxes, fontweight='bold', fontsize=11)
[fig.savefig(f'figures/fig6_gauss.{ext}', dpi=300, bbox_inches='tight') for ext in ('png', 'pdf')]

# ----------------------------------------------------------------- agents
A = [json.load(open(f)) for f in glob.glob('results/gauss_agents/*.json')]
rows = []
cache = {}
for a in A:
    key = (a['lam'], a['rho'])
    ref = R[(1.8, a['rho'], a['lam'])]
    if key not in cache:
        items = simulate_items(a['rho'], 1.8, 20000, np.random.default_rng(a['test_seed']))
        g = [{int(k): v for k, v in gt.items()} for gt in ref['fitted']]
        rc, tc = run_bin_policy(items, g, a['lam'], nbins=25)
        cache[key] = (items, rc)
    items, rc = cache[key]
    z = np.load(f"results/gauss_agents/{a['kind']}_{a['lam']}_{a['rho']}_{a['seed']}.npz")
    d = z['ret'] - rc
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(300)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    # matched-confidence stopping difference
    tau = z['tau']; res = items['res']; conf = items['conf']
    num = den = 0.0
    for t in range(T):
        run = tau >= t
        b = np.minimum(np.floor((conf[run, t] - 0.5) / 0.01 + 1e-9).astype(int), 49)
        st = tau[run] == t; un = ~res[run]
        for k in np.unique(b):
            m = b == k
            nI = (m & un).sum(); nR = (m & ~un).sum()
            if nI >= 20 and nR >= 20:
                w = min(nI, nR); num += w * (st[m & un].mean() - st[m & ~un].mean()); den += w
    dstop = num / den if den > 0 else np.nan
    nonlearner = a['abstain_all'] and ref['Vstar'] > 0.01
    rows.append(dict(kind=a['kind'], lam=a['lam'], rho=a['rho'], seed=a['seed'], adv=float(d.mean()), lo=float(lo), hi=float(hi),
                     dstop=float(dstop), gap_upper=ref['gap_upper'], gap_lower=ref['gap_lower'], nonlearner=bool(nonlearner), ret=a['ret']))
json.dump(rows, open('results/gauss_agents_summary.json', 'w'), indent=1)
for kind in ('gru', 'lstm'):
    K = [r for r in rows if r['kind'] == kind]
    Kl = [r for r in K if not r['nonlearner']]
    out = dict(n=len(K), nonlearners=len(K) - len(Kl))
    for lab, X in (('all', K), ('learners', Kl)):
        out[f'rho_adv_{lab}'] = spearmanr([r['gap_upper'] for r in X], [r['adv'] for r in X]).correlation
        out[f'rho_adv_lower_{lab}'] = spearmanr([r['gap_lower'] for r in X], [r['adv'] for r in X]).correlation
        Xd = [r for r in X if r['rho'] > 0 and not np.isnan(r['dstop'])]
        out[f'rho_dstop_{lab}'] = spearmanr([r['gap_upper'] for r in Xd], [r['dstop'] for r in Xd]).correlation
    hi_gap = [r for r in Kl if r['gap_lower'] > 0.015]
    out['hi_gap_n'] = len(hi_gap); out['hi_gap_sig_pos'] = sum(r['lo'] > 0 for r in hi_gap)
    zero = [r for r in Kl if r['gap_upper'] < 1e-9]
    out['zero_n'] = len(zero); out['zero_sig_pos'] = sum(r['lo'] > 0 for r in zero); out['zero_sig_neg'] = sum(r['hi'] < 0 for r in zero)
    out['zero_max_abs_adv'] = max(abs(r['adv']) for r in zero) if zero else None
    S[kind] = out
    print(kind, out)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
for r in rows:
    if r['nonlearner']:
        continue
    c = '#2a78d6' if r['kind'] == 'gru' else '#eb6834'; mk = 'o' if r['kind'] == 'gru' else '^'
    axs[0].errorbar(r['gap_upper'] + (0.0006 if r['kind'] == 'lstm' else 0), r['adv'], yerr=[[r['adv'] - r['lo']], [r['hi'] - r['adv']]], fmt=mk, color=c, ms=4, lw=0.7)
    if r['rho'] > 0 and not np.isnan(r['dstop']):
        axs[1].scatter(r['gap_upper'], r['dstop'], marker=mk, color=c, s=16, zorder=3)
axs[0].axhline(0, color='#52514e', lw=0.8)
axs[0].set_xlabel('certified gap (upper bound)'); axs[0].set_ylabel('agent return $-$ fitted conf. controller')
axs[1].set_xlabel('certified gap (upper bound)'); axs[1].set_ylabel('matched-confidence $\\Delta_{\\mathrm{stop}}$')
from matplotlib.lines import Line2D
axs[1].legend(handles=[Line2D([], [], marker='o', ls='', color='#2a78d6', label='GRU'), Line2D([], [], marker='^', ls='', color='#eb6834', label='LSTM')], fontsize=7)
for i, a in enumerate(axs):
    a.text(0.02, 0.96, 'ab'[i], transform=a.transAxes, fontweight='bold', fontsize=11, va='top')
fig.tight_layout(); [fig.savefig(f'figures/figS2_gauss_agents.{ext}', dpi=300) for ext in ('png', 'pdf')]
P = json.load(open('results/probe_steer.json'))
S['probe'] = {s: dict(r2=v['r2_resvoc'], r2u=v['r2_resvoc_untrained'], r2c=v['r2_conf']) for s, v in P.items()}
def slope(d):
    a = np.array([-4, -2, 0, 2, 4]); y = np.array([d[str(x)][0] for x in a]); return float(np.polyfit(a, y, 1)[0])
S['steer'] = {s: dict(voc=slope(v['steer']['voc']), conf=slope(v['steer']['conf']),
                      rand=[slope(v['steer'][f'rand{i}']) for i in range(8)]) for s, v in P.items()}
json.dump(S, open('results/summary_gauss.json', 'w'), indent=1, default=float)
print({k: v for k, v in S.items() if k not in ('mc_vs_grid', 'su_pairs')})
