import json, glob, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.grid': True,
                     'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'legend.frameon': False})
COL = {'A': '#2a78d6', 'B': '#eb6834', 'C': '#1baf7a'}
MRK = {'A': 'o', 'B': 's', 'C': '^'}
NAME = {'A': 'A side channel', 'B': 'B temporal', 'C': 'C static cue'}
R = [json.load(open(f)) for f in glob.glob('results/exp1/*.json')]


def fig2():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for ax, key, lab in zip(axs, ('gap_ca', 'gap_proj'), ('local search (coordinate ascent)', 'projected controller')):
        for f in 'ABC':
            X = np.array([r['gap'] for r in R if r['family'] == f]); Y = np.array([r[key] for r in R if r['family'] == f])
            ax.scatter(X, Y, s=16, c=COL[f], marker=MRK[f], edgecolors='white', linewidths=0.5, label=NAME[f], zorder=3)
        m = max(r[key] for r in R) * 1.05
        ax.plot([0, m], [0, m], color='#8a8984', lw=1, ls='--', zorder=2)
        ax.set_xlabel('certified gap $\\Delta_{\\mathrm{conf}}$'); ax.set_ylabel('gap from ' + lab)
        ax.set_xlim(-0.002, m); ax.set_ylim(-0.002, m)
    axs[0].text(0.02, 0.95, 'a', transform=axs[0].transAxes, fontweight='bold', fontsize=11, va='top')
    axs[1].text(0.02, 0.95, 'b', transform=axs[1].transAxes, fontweight='bold', fontsize=11, va='top')
    axs[0].legend(loc='lower right')
    fig.tight_layout(); [fig.savefig(f'figures/fig2_certified.{ext}', dpi=300) for ext in ('png', 'pdf')]


def fig3():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    xs = [('proj_Iact_cum', None, 'decision information (bits)', 0.89),
          ('proj_Ired_cum', 'H_red', 'residual reducibility info (norm.)', 0.81),
          ('ident15', 'H_red', 'identifiability $I(\\tau;H_{15})$ (norm.)', 0.61)]
    for i, (ax, (k, norm, lab, rho)) in enumerate(zip(axs, xs)):
        for f in 'ABC':
            X = np.array([r[k] / (r[norm] if norm else 1) for r in R if r['family'] == f])
            Y = np.array([r['gamma'] for r in R if r['family'] == f])
            ax.scatter(X, Y, s=11, c=COL[f], marker=MRK[f], edgecolors='white', linewidths=0.4, label=NAME[f], zorder=3)
        ax.set_xlabel(lab)
        ax.set_title(f'Spearman $\\rho$ = {rho:.2f}', fontsize=8.5, color='#0b0b0b')
        ax.text(0.03, 0.97, 'abc'[i], transform=ax.transAxes, fontweight='bold', fontsize=11, va='top')
    axs[0].set_ylabel('necessity index $\\gamma$')
    axs[2].legend(loc='upper left', bbox_to_anchor=(0.05, 0.93), fontsize=7)
    fig.tight_layout(); [fig.savefig(f'figures/fig3_predictors.{ext}', dpi=300) for ext in ('png', 'pdf')]


def fig4():
    big = [r for r in R if r['gap'] > 1e-4]
    data = [('Latent\n(Thm 4)', [r['proj_pinsker'] / r['gap'] for r in big]),
            ('TV step\n(Eq. 12)', [r['proj_tv'] / r['gap'] for r in big]),
            ('Decision\n(Thm 3)', [r['proj_act'] / r['gap'] for r in big]),
            ('Minority\nmass', [r['proj_mu'] / r['gap'] for r in big]),
            ('Projected\ncontroller', [r['gap_proj'] / r['gap'] for r in big])]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    rng = np.random.default_rng(0)
    for i, (lab, v) in enumerate(data):
        v = np.array(v)
        fams = [r['family'] for r in big]
        for f in 'ABC':
            m = np.array([x == f for x in fams])
            ax.scatter(i + rng.uniform(-0.18, 0.18, m.sum()), v[m], s=7, c=COL[f], alpha=0.7, edgecolors='none',
                       label=NAME[f] if i == 0 else None, zorder=3)
        med = np.median(v)
        ax.plot([i - 0.3, i + 0.3], [med, med], color='#0b0b0b', lw=2, zorder=4)
        ax.text(i + 0.33, med, f'{med:,.0f}' if med >= 10 else f'{med:.2f}', va='center', fontsize=7.5, color='#0b0b0b')
    ax.set_yscale('log'); ax.axhline(1, color='#8a8984', lw=1, ls='--')
    ax.set_xlim(-0.5, len(data) - 0.1); ax.set_xticks(range(len(data))); ax.set_xticklabels([d[0] for d in data])
    ax.set_ylabel('bound / certified gap'); ax.legend(loc='upper right', fontsize=7, markerscale=2)
    ax.grid(axis='x', visible=False)
    fig.tight_layout(); [fig.savefig(f'figures/fig4_bounds.{ext}', dpi=300) for ext in ('png', 'pdf')]




def figS4():
    S = json.load(open('results/summary_exp3.json'))
    series = [('rep_conf', 'representation: confidence monitor'), ('rep_cue', 'representation: + cue bit'),
              ('depth1', 'planning depth: one step'), ('depth2', 'planning depth: two steps'),
              ('est32', 'estimation: n = 32'), ('est512', 'estimation: n = 512'),
              ('opt_loss', 'optimisation: tuned threshold')]
    cols = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6']
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    floor = 1e-6
    for ax, fam, title in zip(axs, ('het', 'setup'), ('heterogeneous family', 'setup family')):
        rows = [r for r in S['table'] if r['family'] == fam]
        w = 0.11
        for j, ((k, lab), c) in enumerate(zip(series, cols)):
            vals = [max(r[k], floor) for r in rows]
            ax.bar(np.arange(3) + (j - 3) * w, vals, width=w * 0.85, color=c, label=lab, zorder=3)
        ax.set_yscale('log'); ax.set_ylim(floor, 0.3)
        ax.set_xticks(range(3)); ax.set_xticklabels([f'v = {r["v"]:g}' for r in rows]); ax.set_xlabel('cue validity')
        ax.set_title(title, fontsize=9); ax.grid(axis='x', visible=False)
    axs[0].set_ylabel('loss vs. full-information optimum')
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=7)
    fig.tight_layout(); [fig.savefig(f'figures/figS4_sources.{ext}', dpi=300) for ext in ('png', 'pdf')]


if __name__ == '__main__':
    for f in sys.argv[1:]:
        globals()[f]()
