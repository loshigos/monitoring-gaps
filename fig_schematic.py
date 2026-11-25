import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
INK = '#1f2933'; MUTED = '#6b7280'; BLUE = '#2563eb'; ORANGE = '#d97706'; GREEN = '#059669'; RED = '#dc2626'

fig = plt.figure(figsize=(7.2, 3.3))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis('off')


def box(x, y, w, h, text, ec=INK, fc='white', fs=8.5, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', ec=ec, fc=fc, lw=1.1)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK,
            fontweight='bold' if bold else 'normal')


def arrow(x1, y1, x2, y2, c=INK, ls='-'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=9, color=c, lw=1.1, linestyle=ls))

# panel a: histories -> monitor -> controller
ax.text(1, 44, 'a', fontweight='bold', fontsize=11)
ax.text(5, 44, 'Monitor-restricted control', fontsize=9, color=INK)
hist = [('$h_1$: reducible, $c$=0.75', GREEN, 'continue'), ('$h_2$: irreducible, $c$=0.75', RED, 'stop'),
        ('$h_3$: reducible, $c$=0.75', GREEN, 'continue')]
for i, (t, col, a) in enumerate(hist):
    y = 33 - i * 9
    box(2, y, 19, 5.5, t, ec=col, fs=7.5)
    arrow(22, y + 2.75, 30, 21.5, c=MUTED)
    ax.text(12, y - 1.6, '$A^*$ = ' + a, ha='center', fontsize=7, color=col)
box(30, 18, 14, 7, 'monitor\n$\\varphi_t$ = (bin($c_t$), $t$)', ec=BLUE, fs=8)
arrow(44.5, 21.5, 50, 21.5)
box(50, 18, 11, 7, 'controller\n$g_t(\\varphi_t)$', fs=8)
ax.text(36.5, 13, 'pools histories with\nincompatible $A^*$', ha='center', fontsize=7.2, color=BLUE)
ax.text(36.5, 5.5, 'conflict $C_t(k)>0$  iff  $H(A^*_t\\mid\\varphi_t=k)>0$', ha='center', fontsize=7.5, color=INK)

# panel b: conflict vs steering
ax.text(65, 44, 'b', fontweight='bold', fontsize=11)
ax.text(69, 44, 'Where the gap comes from', fontsize=9, color=INK)
box(66, 31, 15, 7, 'conflict $C$\nshared action in a\nmixed monitor state', ec=ORANGE, fs=7.2)
box(83, 31, 15, 7, 'steering $\\chi$\nlocal regret paid to\navoid mixed states', ec=GREEN, fs=7.2)
ax.text(82, 26, '$\\Delta_\\varphi=\\min_g\\sum P_g\\,[C^g+\\chi^g]$', ha='center', fontsize=8.5)
box(66, 7, 32, 13, '', ec=MUTED)
ax.text(82, 17.2, 'metacognitive bit  $\\psi_t=\\mathrm{sign}\\,\\mathrm{VOC}_t$', ha='center', fontsize=8, color=INK)
ax.text(82, 13.2, 'always sufficient (1 bit per decision);', ha='center', fontsize=7.3, color=MUTED)
ax.text(82, 9.8, 'the cost is computing it, not transmitting it', ha='center', fontsize=7.3, color=MUTED)
[fig.savefig(f'figures/fig1_schematic.{ext}', dpi=300) for ext in ('png', 'pdf')]
print('ok')
