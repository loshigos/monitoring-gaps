"""Experiment 1b summary and Figure 5 (monitor refinements), Figure 7 (learning agents), and Tables 2 and 3.

Run after run_exp1.py, run_exp1b.py, agents.py, analyse_exp2.py, run_exp3.py and analyse_exp3.py.
"""
import json, glob, os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figs  # shared style
from figs import COL, NAME

os.makedirs('tables', exist_ok=True); os.makedirs('figures', exist_ok=True)
E1 = [json.load(open(f)) for f in glob.glob('results/exp1/*.json')]

# ---------------- Experiment 1b
B = [json.load(open(f)) for f in glob.glob('results/exp1b/*.json')]
E1d = {(r['family'], r['rho'], r['lam'], r['par']): r for r in E1}
mons = ['prev', 'red1', 'red2', 'red3']
rem = {f: {m: [] for m in mons} for f in 'ABC'}
tot = {f: {m: [0.0, 0.0] for m in mons} for f in 'ABC'}
closed = sum(all(b[m]['status'] == 'Optimal' for m in mons) for b in B)
maxw = max(b[m]['gap'] - b[m]['gap_lower'] for b in B for m in mons)
for b in B:
    base = E1d[(b['family'], b['rho'], b['lam'], b['par'])]['gap']
    for m in mons:
        tot[b['family']][m][0] += max(b[m]['gap'], 0); tot[b['family']][m][1] += max(base, 0)
removed = {f: {m: 1 - tot[f][m][0] / tot[f][m][1] for m in mons} for f in 'ABC'}
print('removed', removed, 'closed', closed, len(B), 'maxw', maxw)
viol = sum(b[m]['gap'] > E1d[(b['family'], b['rho'], b['lam'], b['par'])]['gap'] + 1e-6 for b in B for m in mons)
fig, ax = plt.subplots(figsize=(6.0, 2.8))
labs = [('prev', 'progress\n(previous conf.)'), ('red1', 'solvability\n1 bit'), ('red2', 'solvability\n2 bits'), ('red3', 'solvability\n3 bits')]
w = 0.26
for j, f in enumerate('ABC'):
    ax.bar(np.arange(4) + (j - 1) * w, [100 * (1 - removed[f][m]) for m, _ in labs], width=w * 0.88, color=COL[f], label=NAME[f], zorder=3)
ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in labs]); ax.set_ylabel('gap remaining (% of confidence gap)')
ax.set_ylim(0, 105); ax.grid(axis='x', visible=False); ax.legend(loc='upper right', fontsize=7)
fig.tight_layout(); [fig.savefig(f'figures/fig5_refinements.{ext}', dpi=300) for ext in ('png', 'pdf')]
json.dump(dict(removed=removed, closed=closed, n=len(B), maxw=maxw), open('results/summary_exp1b.json', 'w'), indent=1)

# ---------------- Experiment 2
S2 = json.load(open('results/summary_exp2.json'))
ref = S2['ref']; ag = S2['agents']
rows = ['| Family | Parameter | $V^*$ | $V_{\\text{conf}}$ (certified) | Gap | $\\Delta_{\\text{stop}}^{*}$ |', '|:--|--:|--:|--:|--:|--:|']
for k in ['A_0.0', 'A_0.2', 'A_0.4', 'A_0.6', 'C_0.0', 'C_0.3', 'C_0.6', 'C_1.0']:
    f, p = k.split('_'); r = ref[k]
    rows.append(f"| {f} | {'$\\delta$' if f == 'A' else '$v$'} = {float(p):g} | {r['Vstar']:.4f} | {r['Vconf']:.4f} | {max(r['gap'], 0):.4f} | {r['bayes_dstop']:.4f} |")
open('tables/table2.md', 'w').write('\n'.join(rows) + '\n')

big = [a for a in ag if ref[f"{a['fam']}_{a['par']}"]['gap'] > 0.02]
sig = [a for a in big if a['z_vs_conf'] > 2.58]
neg = [a for a in big if a['z_vs_conf'] < -2.58]
cap = lambda arch: np.mean([a['capture'] for a in big if a['arch'] == arch])
inf = [a for a in ag if not (a['fam'] == 'C' and a['par'] == 0.0) and a['abstain'] < 0.999]
collapsed = [a for a in ag if a['abstain'] >= 0.999 and not (a['fam'] == 'C' and a['par'] == 0.0)]
inf2 = [a for a in inf if a not in collapsed]
from scipy.stats import pearsonr, spearmanr
x = [a['bayes_dstop'] for a in inf2]; y = [a['dstop'] for a in inf2]
pr = pearsonr(x, y)[0]; sr = spearmanr(x, y).correlation
A0 = [a for a in ag if a['fam'] == 'A' and a['par'] == 0.0 and a not in collapsed]
A2 = [a for a in ag if a['fam'] == 'A' and a['par'] == 0.2]
C0 = [a for a in ag if a['fam'] == 'C' and a['par'] == 0.0]
C0_abst = sum(a['abstain_t0'] > 0.99 for a in C0)
ratios_C = [a['dstop'] / a['bayes_dstop'] for a in inf2 if a['fam'] == 'C' and a['par'] in (0.3, 0.6)]
print('Experiment 2: %d agents with informative behaviour; Pearson r = %.2f, Spearman rho = %.2f between agent and Bayes-optimal stopping differences' % (len(inf2), pr, sr))
print('Experiment 2: %d of %d agent-condition pairs with certified gap > 0.02 earned significantly more than the best confidence-restricted controller' % (len(sig), len(big)))

# Figure 7 (learning agents)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
mk = {'gru': 'o', 'transformer': '^'}
for a in ag:
    if a in collapsed or (a['fam'] == 'C' and a['par'] == 0.0):
        continue
    c = COL[a['fam']]
    axs[0].errorbar(a['bayes_dstop'], a['dstop'], yerr=[[a['dstop'] - a['dstop_ci'][0]], [a['dstop_ci'][1] - a['dstop']]], fmt=mk[a['arch']],
                    color=c, ms=5, mfc=c if a['arch'] == 'gru' else 'white', lw=0.8, zorder=3)
m = 0.55
axs[0].plot([0, m], [0, m], ls='--', color='#8a8984', lw=1)
axs[0].set_xlabel('Bayes-optimal $\\Delta_{\\mathrm{stop}}$'); axs[0].set_ylabel('agent $\\Delta_{\\mathrm{stop}}$')
for a in ag:
    if a['fam'] == 'C' and a['par'] == 0.0:
        continue
    r = ref[f"{a['fam']}_{a['par']}"]
    c = COL[a['fam']]
    axs[1].errorbar(max(r['gap'], 0), a['ret'] - r['Vconf'], yerr=1.96 * a['se'], fmt=mk[a['arch']], color=c, ms=5,
                    mfc=c if a['arch'] == 'gru' else 'white', lw=0.8, zorder=3)
axs[1].plot([0, 0.055], [0, 0.055], ls='--', color='#8a8984', lw=1); axs[1].axhline(0, color='#52514e', lw=0.8)
axs[1].set_xlabel('certified gap $\\Delta_{\\mathrm{conf}}$'); axs[1].set_ylabel('return $-\\ V_{\\mathrm{conf}}$')
from matplotlib.lines import Line2D
h = [Line2D([], [], marker='o', ls='', color='#52514e', label='GRU'), Line2D([], [], marker='^', ls='', color='#52514e', mfc='white', label='transformer'),
     Line2D([], [], marker='s', ls='', color=COL['A'], label='family A'), Line2D([], [], marker='s', ls='', color=COL['C'], label='family C')]
axs[1].legend(handles=h, loc='lower right', fontsize=7)
for i, ax in enumerate(axs):
    ax.text(0.02, 0.96, 'ab'[i], transform=ax.transAxes, fontweight='bold', fontsize=11, va='top')
fig.tight_layout(); [fig.savefig(f'figures/fig7_agents.{ext}', dpi=300) for ext in ('png', 'pdf')]

# ---------------- Experiment 3
S3 = json.load(open('results/summary_exp3.json'))
rows = ['| Family | $v$ | $V^*$ | Repr. conf. | Repr. +cue | Depth 1 | Depth 2 | Est. 512 | Thresh. | Optim. |',
        '|:-----|---:|------:|------:|------:|------:|------:|------:|------:|------:|']
fmt = lambda x: f"{x:.4f}" if x >= 5e-5 else ("<0.0001" if x > 1e-9 else "0")
for r in S3['table']:
    rows.append(f"| {'het.' if r['family']=='het' else 'setup'} | {r['v']:g} | {r['Vstar']:.4f} | {fmt(r['rep_conf'])} | {fmt(r['rep_cue'])} | {fmt(r['depth1'])} | {fmt(r['depth2'])} | {fmt(r['est512'])} | {fmt(r['thr'])} | {fmt(r['opt_loss'])} |")
open('tables/table3.md', 'w').write('\n'.join(rows) + '\n')
