"""Render the oracle-free Figure 1. Minimal ink: explanation lives in the caption."""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, 'logs', 'figure1_no_oracle')
OUT = os.path.join(ROOT, 'figures', 'figure1_no_oracle')
os.makedirs(OUT, exist_ok=True)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
z = np.load(os.path.join(IN, f'fig1_no_oracle_seed{seed}.npz'))

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#dedcd7'

plt.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10.5,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.8,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.labelcolor': INK, 'text.color': INK,
    'grid.color': GRID, 'grid.linewidth': 0.6,
    'legend.frameon': False, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'savefig.facecolor': 'white',
})

step, s1 = z['step'], float(z['steps_stage1'])
fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.9))

# ---- (a) held-out DSM residual split by the model's own subspace ---------
ax = axes[0]
ax.grid(True, alpha=0.55, zorder=0); ax.set_axisbelow(True)
ax.plot(step, z['dsm_holdout'], color=INK, lw=1.1, ls=(0, (5, 2)), zorder=5,
        label='Total')
ax.plot(step, z['res_tangent'], color=BLUE, lw=1.9, zorder=3,
        label=r'Tangent $\widehat{P}$')
ax.plot(step, z['res_normal'], color=ORANGE, lw=1.9, zorder=4,
        label=r'Normal $I-\widehat{P}$')
ax.axvline(s1, color=MUTED, ls=':', lw=1.1, zorder=1)
ax.text(s1 - 45, 2.1e-3, 'Stage 1 ', color=INK2, fontsize=8.5, ha='right')
ax.text(s1 + 45, 2.1e-3, ' Stage 2', color=INK2, fontsize=8.5, ha='left')
ax.set_yscale('log')
ax.set_xlabel('Training step')
ax.set_ylabel('Held-out DSM residual')
ax.set_title('(a)  Residual by learned subspace', loc='left', pad=8)
ax.set_ylim(1.6e-3, 4e3); ax.set_xlim(-60, 2060)
ax.set_xticks([0, 500, 1000, 1500, 2000])
ax.legend(loc='upper right', bbox_to_anchor=(1.01, 1.02), handlelength=1.6,
          borderaxespad=0.2, labelspacing=0.3)

axin = ax.inset_axes([0.46, 0.20, 0.40, 0.235])
m = step >= 200
axin.plot(step[m], z['res_tangent'][m], color=BLUE, lw=1.4)
axin.axvline(s1, color=MUTED, ls=':', lw=1.0)
axin.set_ylim(4.80, 5.00); axin.set_yticks([4.85, 4.95])
axin.set_xticks([1000]); axin.set_xticklabels([])
axin.tick_params(labelsize=7, length=2, pad=1.5)
for sp in axin.spines.values():
    sp.set_color(MUTED); sp.set_linewidth(0.6)
axin.set_title('tangent, linear', fontsize=7.5, color=INK2, pad=2)

# ---- (b) spectrum of WW^T ------------------------------------------------
ax = axes[1]
ax.grid(True, axis='y', alpha=0.55, zorder=0); ax.set_axisbelow(True)
idx = np.arange(1, 21)
ax.plot(idx, z['spec_init'][:20], 'o-', color=ORANGE, lw=1.7, ms=4,
        zorder=3, label='Init')
ax.plot(idx, z['spec_stage1_end'][:20], 'o-', color=BLUE, lw=1.9, ms=4.5,
        zorder=4, label='After Stage 1')
d_hat, done = int(z['final_d_hat']), z['spec_stage1_end']
ax.annotate('', xy=(d_hat, done[d_hat - 1]), xytext=(d_hat + 1, done[d_hat]),
            arrowprops=dict(arrowstyle='<->', color=INK, lw=1.2,
                            shrinkA=2.5, shrinkB=2.5))
ax.text(d_hat + 1.3, 0.50, rf'$\widehat{{d}}={d_hat}$', color=INK, fontsize=10,
        va='center')
ax.set_xlabel(r'Eigenvalue index of $WW^{\top}$')
ax.set_ylabel('Eigenvalue')
ax.set_title(r'(b)  Spectrum of $WW^{\top}$', loc='left', pad=8)
ax.set_xticks([1, 5, 10, 15, 20]); ax.set_ylim(-0.05, 1.12)
ax.legend(loc='center left', bbox_to_anchor=(0.02, 0.72), handlelength=1.6,
          labelspacing=0.3)

# ---- (c) response along the model's own inferred normal ------------------
ax = axes[2]
ax.grid(True, alpha=0.55, zorder=0); ax.set_axisbelow(True)
a, r, h = z['alphas'], z['normal_response'], float(z['h_t'])
ax.scatter(a, r, s=10, color=BLUE, alpha=0.3, linewidths=0, zorder=3,
           label='Learned')
ax.plot(a, -a / h, color=INK, ls='--', lw=1.5, zorder=4, label=r'$-\alpha/h$')
slope = np.polyfit(a, r, 1)[0]
ax.text(0.03, 0.05, f'slope ${slope:.1f}$  (theory ${-1/h:.0f}$)',
        transform=ax.transAxes, fontsize=9, color=INK, va='bottom')
ax.set_xlabel(r'Displacement $\alpha$ along $\hat{u}_{\min}$')
ax.set_ylabel(r'Score component along $\hat{u}_{\min}$')
ax.set_title(r'(c)  Response along $\hat{u}_{\min}$', loc='left', pad=8)
ax.legend(loc='upper right', handlelength=1.6, labelspacing=0.3)

fig.tight_layout(w_pad=2.2)
for ext in ('png', 'pdf'):
    p = os.path.join(OUT, f'figure1_no_oracle_seed{seed}.{ext}')
    fig.savefig(p, dpi=220, bbox_inches='tight')
    print('saved', p)
