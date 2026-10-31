"""
Build the 3-panel paper figure for Stacked MNIST:
  Real | SiLD (ours) | LDM-CNN
with FID labels in the panel titles. Loads the t0=0.005 seed=42 checkpoints
that produced the numbers in the paper Table 1 (mean from 3-seed
aggregation; the seed=42 checkpoints are used as a representative draw).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.image_data import StackedMNISTDataset
from models.two_stage_latent import TwoStageManifoldDiffusion
from models.latent_diffusion_cnn import LatentDiffusionCNN


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
FIG_DIR = os.path.join(PROJECT_ROOT, 'figures')
CKPT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')


def make_grid(images_flat, nrow, ncol, gap=1):
    """Compose nrow x ncol grid from (n, 3*28*28) flat tensor."""
    H = nrow * 28 + (nrow - 1) * gap
    W = ncol * 28 + (ncol - 1) * gap
    canvas = np.ones((H, W, 3), dtype=np.float32)
    n = min(len(images_flat), nrow * ncol)
    for i in range(n):
        r, c = i // ncol, i % ncol
        img = images_flat[i].detach().cpu().clamp(0, 1).view(3, 28, 28)
        img = img.permute(1, 2, 0).numpy()
        y = r * (28 + gap)
        x = c * (28 + gap)
        canvas[y:y + 28, x:x + 28] = img
    return canvas


def aggregated_fid(t_int):
    """Pull mean and std FID from the aggregator output, for the requested t group."""
    agg_path = os.path.join(LOG_DIR, 'stacked_mnist_results_agg.json')
    if not os.path.exists(agg_path):
        return None, None, None, None
    with open(agg_path) as f:
        data = json.load(f)
    for g in data['groups']:
        if abs(g['t0'] - t_int / 1000.0) < 1e-9:
            uld = g['fid']['uld']; ldm = g['fid']['ldm']
            return uld['mean'], uld['std'], ldm['mean'], ldm['std']
    return None, None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--t0', type=float, default=0.005,
                    help='Stage-1 noise scale (default 0.005, the paper Table 1 setting).')
    ap.add_argument('--seed', type=int, default=42,
                    help='Seed of the checkpoint to draw samples from.')
    ap.add_argument('--nrow', type=int, default=8)
    ap.add_argument('--ncol', type=int, default=24)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--out', default=os.path.join(FIG_DIR, 'stacked_mnist_composite_sild.png'))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_show = args.nrow * args.ncol

    tag = f"t{int(round(args.t0 * 1000)):03d}_seed{args.seed}"
    print(f"Loading checkpoints with tag: {tag}")

    image_shape = (3, 28, 28)
    d = 3 * 28 * 28
    common = dict(d=d, latent_dim=64, hidden_dim_enc=512,
                  hidden_dim_score=512, n_layers=6,
                  image_shape=image_shape, base_ch=64)

    model_ts = TwoStageManifoldDiffusion(t0=args.t0, **common).to(device)
    state = torch.load(os.path.join(CKPT_DIR, f'stacked_mnist_uld_full_{tag}.pt'),
                       map_location=device)
    model_ts.load_state_dict(state['model'])
    model_ts.freeze_stage1()

    model_ldm = LatentDiffusionCNN(kl_weight=1e-4, **common).to(device)
    state = torch.load(os.path.join(CKPT_DIR, f'stacked_mnist_ldm_full_{tag}.pt'),
                       map_location=device)
    model_ldm.load_state_dict(state['model'])
    model_ldm.freeze_ae()

    print(f"Sampling {n_show} from each model ({args.steps} steps)...")
    with torch.no_grad():
        x_sild = model_ts.sample(n_show, n_steps=args.steps, device=device, clamp=(0, 1)).cpu()
        x_ldm = model_ldm.sample(n_show, n_steps=args.steps, device=device, clamp=(0, 1)).cpu()
    dataset = StackedMNISTDataset(n_pool=60000, seed=args.seed)
    x_real, _ = dataset.sample_clean(n_show)

    sild_grid = make_grid(x_sild, args.nrow, args.ncol)
    ldm_grid = make_grid(x_ldm, args.nrow, args.ncol)
    real_grid = make_grid(x_real, args.nrow, args.ncol)

    sild_mu, sild_sd, ldm_mu, ldm_sd = aggregated_fid(int(round(args.t0 * 1000)))
    if sild_mu is not None:
        sild_label = f"SiLD (ours), FID = {sild_mu:.2f} ± {sild_sd:.2f}"
        ldm_label = f"LDM-CNN, FID = {ldm_mu:.2f} ± {ldm_sd:.2f}"
    else:
        sild_label, ldm_label = "SiLD (ours)", "LDM-CNN"

    fig, axes = plt.subplots(1, 3, figsize=(3 * (args.ncol * 0.45), args.nrow * 0.45 + 0.6))
    panels = [(real_grid, "Real"), (sild_grid, sild_label), (ldm_grid, ldm_label)]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(args.out, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == '__main__':
    main()
