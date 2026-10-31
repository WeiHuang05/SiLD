"""Appendix figure: Stacked MNIST samples across the Stage-1 objective family.

Replaces the earlier three-panel real/SiLD/LDM figure, whose labels predate
the repositioning of the clean autoencoder as the sigma -> 0 member of the
SiLD family. The four panels are exactly the arms of the main-text Stacked
MNIST table, plus real data.

The VP-schedule arm is deliberately NOT shown here. It uses the same nominal
t0 = 5e-3 as the additive arm but draws Stage-1 noise under the VP schedule,
which induces an effective std of 2.74e-2 -- so it varies the noise *form*
and the noise *magnitude* at once, and does not belong on a sigma axis. It is
a parameterization check and is reported separately in the appendix table.

Panels are the uncurated 8x8 sample grids written by the training runs; this
script only crops the stale baked-in titles, relabels, and composes them.
All arms are seed 42 and share Stage-2 interval t2 in [0.001, 0.999].
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, 'figures', 'stacked_mnist_stage1_family')

# (label, FID string, source png, family)
#   family: 'real' | 'kl' (stochastic KL bottleneck) | 'det' (deterministic)
PANELS = [
    ('Real data', None,
     'figures/sild_matches_ldm/stacked_mnist_real_t005_seed42.png', 'real'),
    (r'$\beta$-VAE, KL-regularized', '16.15',
     'figures/sild_matches_ldm/stacked_mnist_gen_ldm_t005_seed42.png', 'kl'),
    (r'SiLD, clean AE  ($\sigma_{\mathrm{geom}}=0$)', '8.20',
     'figures/clean_ae_ldm_stacked/stacked_mnist_gen_clean_ae_ldm_tmin001_seed42.png', 'det'),
    (r'SiLD  ($\sigma_{\mathrm{geom}}=5\times10^{-3}$)', '7.95',
     'figures/sild_matches_ldm/stacked_mnist_gen_uld_t005_seed42.png', 'det'),
]

EDGE = {'real': '#6b6b6b', 'kl': '#c2591a', 'det': '#1f6fb4'}


def crop_grid(path):
    """Drop the stale baked-in title and white margin, keep the sample grid."""
    arr = np.array(Image.open(os.path.join(ROOT, path)).convert('RGB')).astype(int)
    dark = arr.sum(2) < 120           # the grid tiles have black backgrounds
    # Threshold on a large fraction of the width/height so that the stale
    # title text, however long, can never be mistaken for a row of tiles.
    thr_r = 0.30 * arr.shape[1]
    thr_c = 0.30 * arr.shape[0]
    rows = np.where(dark.sum(1) > thr_r)[0]
    cols = np.where(dark.sum(0) > thr_c)[0]
    box = (cols.min(), rows.min(), cols.max() + 1, rows.max() + 1)
    return Image.open(os.path.join(ROOT, path)).convert('RGB').crop(box)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 9.3),
                             gridspec_kw={'hspace': 0.17, 'wspace': 0.04})
    for ax, (label, fid, path, fam) in zip(axes.ravel(), PANELS):
        ax.imshow(np.asarray(crop_grid(path)), interpolation='lanczos')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(EDGE[fam]); sp.set_linewidth(2.2)
        title = label if fid is None else f'{label}\nFID {fid}'
        ax.set_title(title, fontsize=12, pad=6,
                     color='black' if fam == 'real' else EDGE[fam])

    # group legend: what the colours mean
    handles = [
        Rectangle((0, 0), 1, 1, fc='none', ec=EDGE['real'], lw=2.2,
                  label='Training data'),
        Rectangle((0, 0), 1, 1, fc='none', ec=EDGE['kl'], lw=2.2,
                  label='Stochastic KL-regularized Stage 1'),
        Rectangle((0, 0), 1, 1, fc='none', ec=EDGE['det'], lw=2.2,
                  label='Deterministic Stage 1 (SiLD family)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 0.004))
    fig.subplots_adjust(left=0.014, right=0.986, top=0.945, bottom=0.062)
    for ext in ('pdf', 'png'):
        out = os.path.join(OUT_DIR, f'stacked_mnist_stage1_family_seed42.{ext}')
        fig.savefig(out, dpi=200)
        print('wrote', out)


if __name__ == '__main__':
    main()
