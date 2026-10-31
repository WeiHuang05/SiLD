"""
CelebA-HQ U-Net v2 — both-sampler comparison.

Loads the existing celebahq_unetv2_{twostage,ldm}_full.pt checkpoints and
runs each through TWO samplers:

  - "legacy"  : the old buggy sampler that integrated to t = T/n_steps
                (essentially t -> 0), without a Tweedie final step.
  - "fixed"   : the post-bug-fix sampler that integrates to t = t0 and
                takes z = x0_pred at the final step.

Reports FID for each (method x sampler) pair so we have a definitive
delta on the SAME checkpoint, isolating the sampler change from any
training differences.

For LDM the bug fix did NOT touch sampling, so legacy and fixed should
give identical numbers (sanity check).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import shutil
import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')

from data.image_data import CelebAHQDataset
from experiments.compare_celebahq_unet_v2 import (
    TwoStageSpatial, LDMSpatial, load_checkpoint,
)

try:
    import torch_fidelity
    TORCH_FIDELITY_AVAILABLE = True
except ImportError:
    TORCH_FIDELITY_AVAILABLE = False

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Legacy sampler (the OLD buggy one) — re-implemented inline
# ----------------------------------------------------------------------

@torch.no_grad()
def legacy_twostage_sample(model, n, n_steps=500, device='cpu', clamp=None):
    """Old TwoStageSpatial.sample BEFORE the bug fix.
       dt = T/n_steps, integrates from T down to dt, no Tweedie final step.
    """
    _, h, w = model.spatial_shape
    z = torch.randn(n, model.latent_ch, h, w, device=device)
    dt = model.T / n_steps
    for i in range(n_steps, 0, -1):
        t = torch.full((n,), i * dt, device=device)
        alpha_bar, sigma = model.noise_schedule(t)
        eps_pred = model.score_net(z, t)
        ab = alpha_bar.view(-1, 1, 1, 1)
        sig = sigma.view(-1, 1, 1, 1)
        x0_pred = (z - sig * eps_pred) / torch.sqrt(ab).clamp(min=1e-8)
        alpha_bar_prev, sigma_prev = model.noise_schedule(t - dt)
        ab_p = alpha_bar_prev.view(-1, 1, 1, 1)
        sig_p = sigma_prev.view(-1, 1, 1, 1)
        z = torch.sqrt(ab_p) * x0_pred + sig_p * torch.randn_like(z)
    x = model.decode(z)
    if clamp:
        x = x.clamp(*clamp)
    return x


# ----------------------------------------------------------------------
# torch-fidelity helpers
# ----------------------------------------------------------------------

def _save_uint8_pngs(images, out_dir, prefix):
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images):
        arr = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        arr = np.transpose(arr, (1, 2, 0))
        Image.fromarray(arr).save(os.path.join(out_dir, f'{prefix}_{i:05d}.png'))


def compute_fid(real_dir, gen_dir):
    if not TORCH_FIDELITY_AVAILABLE:
        return float('nan')
    metrics = torch_fidelity.calculate_metrics(
        input1=gen_dir, input2=real_dir,
        cuda=torch.cuda.is_available(),
        isc=False, fid=True, kid=False, verbose=False,
        samples_find_deep=False,
    )
    return float(metrics.get('frechet_inception_distance', float('nan')))


# ----------------------------------------------------------------------
# Sampling utility
# ----------------------------------------------------------------------

def big_sample(sample_fn, n_total, batch=32, device='cpu'):
    out = []
    for i in range(0, n_total, batch):
        k = min(batch, n_total - i)
        x = sample_fn(k, n_steps=500, device=device, clamp=(0, 1)).cpu()
        out.append(x.view(-1, 3, 64, 64))
        if (i // batch + 1) % 16 == 0:
            print(f"    {i + batch}/{n_total}")
    return torch.cat(out)[:n_total]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def run(seed=42, n_fid=2048):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    img_size = 64
    image_shape = (3, img_size, img_size)
    d = 3 * img_size * img_size
    latent_ch = 8
    base_ch = 128
    unet_base_ch = 256
    t0 = 0.005

    print(f"Loading CelebA-HQ dataset (n_fid={n_fid})...")
    dataset = CelebAHQDataset(img_size=img_size, max_samples=20000)

    # ------------------ Load TwoStage ------------------
    print("\nLoading ULD checkpoint...")
    model_ts = TwoStageSpatial(
        d=d, image_shape=image_shape, latent_ch=latent_ch,
        base_ch=base_ch, unet_base_ch=unet_base_ch, t0=t0,
    ).to(device)
    ckpt = load_checkpoint(model_ts, 'celebahq_unetv2_twostage_full', device=device)
    if ckpt is None:
        print("ERROR: ULD checkpoint missing")
        return
    model_ts.eval()

    # ------------------ Load LDM ------------------
    print("\nLoading LDM checkpoint...")
    model_ldm = LDMSpatial(
        d=d, image_shape=image_shape, latent_ch=latent_ch,
        base_ch=base_ch, unet_base_ch=unet_base_ch, kl_weight=1e-4,
    ).to(device)
    ckpt = load_checkpoint(model_ldm, 'celebahq_unetv2_ldm_full', device=device)
    if ckpt is None:
        print("ERROR: LDM checkpoint missing")
        return
    model_ldm.eval()

    # ------------------ Save real images ------------------
    scratch = os.path.join('/tmp', f'celebahq_2sampler_{os.getpid()}')
    real_dir = os.path.join(scratch, 'real')
    print(f"\nSaving {n_fid} real images to {real_dir} ...")
    real_imgs = []
    for _ in range(n_fid // 64):
        x0, _ = dataset.sample_clean(64)
        real_imgs.append(x0.view(-1, 3, img_size, img_size))
    real_imgs = torch.cat(real_imgs)[:n_fid]
    _save_uint8_pngs(real_imgs, real_dir, 'r')

    # ------------------ Sample 4 ways ------------------
    fid_results = {}
    for label, model, fn in [
        ('uld_legacy', model_ts, lambda n, **kw: legacy_twostage_sample(model_ts, n, **kw)),
        ('uld_fixed',  model_ts, lambda n, **kw: model_ts.sample(n, **kw)),
        ('ldm_legacy', model_ldm, lambda n, **kw: model_ldm.sample(n, **kw)),  # LDM unchanged
        ('ldm_fixed',  model_ldm, lambda n, **kw: model_ldm.sample(n, **kw)),  # identical
    ]:
        print(f"\n>>> Sampling {label} ({n_fid} samples)")
        gen = big_sample(fn, n_fid, batch=32, device=device)
        out_dir = os.path.join(scratch, label)
        _save_uint8_pngs(gen, out_dir, label[0])
        print(f"  Computing FID for {label}...")
        fid = compute_fid(real_dir, out_dir)
        print(f"  {label}: FID = {fid:.2f}")
        fid_results[label] = fid

    shutil.rmtree(scratch, ignore_errors=True)

    summary = {
        'n_fid': n_fid,
        'fid': fid_results,
        'delta_uld_fix_minus_legacy': fid_results['uld_fixed'] - fid_results['uld_legacy'],
        'delta_ldm_fix_minus_legacy': fid_results['ldm_fixed'] - fid_results['ldm_legacy'],
    }
    out_path = os.path.join(LOG_DIR, 'celebahq_both_samplers.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 70)
    print("CELEBA-HQ — both-sampler comparison (same checkpoints)")
    print("=" * 70)
    print(f"  {'method/sampler':<20} {'FID':<10}")
    print("  " + "-" * 35)
    for k, v in fid_results.items():
        print(f"  {k:<20} {v:<10.2f}")
    print(f"\n  ULD delta (fixed - legacy): "
          f"{summary['delta_uld_fix_minus_legacy']:+.2f}")
    print(f"  LDM delta (fixed - legacy): "
          f"{summary['delta_ldm_fix_minus_legacy']:+.2f}  "
          f"(should be 0 — sanity check)")


if __name__ == '__main__':
    run()
