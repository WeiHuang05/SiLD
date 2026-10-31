"""
Experiment: CelebA (64x64x3, d=12288) — Two-Stage with MMD regularization

Key idea: Add lightweight MMD (Maximum Mean Discrepancy) regularization
to Stage 1 of Two-Stage, encouraging the latent distribution to be smooth
without the strong N(0,I) assumption of KL divergence.

This addresses the reconstruction-generation trade-off:
  - KL (LDM): forces p(z) ≈ N(0,I), good for diffusion but hurts reconstruction
  - MMD (ours): gently smooths p(z), preserves manifold structure while
    eliminating dead zones in latent space

We compare 3 models:
  1. Two-Stage (no regularization) — baseline
  2. Two-Stage + MMD — our proposed improvement
  3. LDM (KL regularization) — standard baseline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.image_data import CelebADataset
from models.two_stage_latent import TwoStageManifoldDiffusion
from models.latent_diffusion_cnn import LatentDiffusionCNN

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available, using MSE only")

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: torchmetrics not available, skipping FID")

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# ================================================================
# MMD kernel
# ================================================================

def gaussian_kernel(x, y, sigma=1.0):
    """RBF kernel between batches x and y."""
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    # Auto bandwidth: use median heuristic if sigma is None
    x = x.unsqueeze(1)  # (B1, 1, D)
    y = y.unsqueeze(0)  # (1, B2, D)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel = torch.exp(-((tiled_x - tiled_y) ** 2).sum(2) / (2 * sigma ** 2 * dim))
    return kernel


def mmd_loss(z, sigma=1.0):
    """
    Compute MMD between z and samples from N(0, I).
    Uses RBF kernel with bandwidth scaled by dimension.
    """
    batch_size = z.size(0)
    dim = z.size(1)

    # Sample from prior
    z_prior = torch.randn_like(z)

    # Multi-scale kernel for robustness
    mmd = torch.tensor(0.0, device=z.device)
    for s in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        k_xx = gaussian_kernel(z, z, sigma=s)
        k_yy = gaussian_kernel(z_prior, z_prior, sigma=s)
        k_xy = gaussian_kernel(z, z_prior, sigma=s)

        # Unbiased MMD^2 estimate
        diag_mask = 1.0 - torch.eye(batch_size, device=z.device)
        mmd += (k_xx * diag_mask).sum() / (batch_size * (batch_size - 1))
        mmd += (k_yy * diag_mask).sum() / (batch_size * (batch_size - 1))
        mmd -= 2.0 * k_xy.mean()

    return mmd


# ================================================================
# Helpers
# ================================================================

def save_checkpoint(model, name, extra=None):
    path = os.path.join(CKPT_DIR, f'{name}.pt')
    state = {'model': model.state_dict()}
    if extra:
        state.update(extra)
    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(model, name, device='cpu'):
    path = os.path.join(CKPT_DIR, f'{name}.pt')
    if os.path.exists(path):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state['model'])
        print(f"  Checkpoint loaded: {path}")
        return state
    return None


def plot_image_grid(images, nrow=8, title='', save_name=None, img_size=64):
    n = min(len(images), nrow * nrow)
    ncol = nrow
    nrow_actual = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow_actual, ncol, figsize=(ncol * 1.5, nrow_actual * 1.5))
    if nrow_actual == 1:
        axes = axes.reshape(1, -1)
    for i in range(nrow_actual):
        for j in range(ncol):
            idx = i * ncol + j
            ax = axes[i, j]
            if idx < n:
                img = images[idx].detach().cpu().clamp(0, 1)
                img = img.view(3, img_size, img_size).permute(1, 2, 0).numpy()
                ax.imshow(img)
            ax.axis('off')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_name:
        path = os.path.join(FIG_DIR, save_name)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")
    else:
        plt.close()


# ================================================================
# Training functions
# ================================================================

def train_stage1_lpips(model, dataset, n_steps, lr=1e-4, batch_size=64,
                       lpips_weight=0.5, device='cpu'):
    """Stage 1: MSE + LPIPS, no regularization (baseline Two-Stage)."""
    img_size = model.image_shape[1]
    sigma_t0 = model.t0

    loss_fn_lpips = None
    if LPIPS_AVAILABLE:
        loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
        loss_fn_lpips.eval()
        for p in loss_fn_lpips.parameters():
            p.requires_grad = False

    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    print(f"  Stage 1 (no reg): {n_steps} steps, lpips_w={lpips_weight}")

    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)
        x_t = x0 + sigma_t0 * torch.randn_like(x0)
        z, x_hat = model.project(x_t)

        mse_loss = F.mse_loss(x_hat, x0)
        lpips_loss = torch.tensor(0.0, device=device)
        if loss_fn_lpips is not None:
            x0_img = x0.view(-1, 3, img_size, img_size) * 2 - 1
            xh_img = x_hat.view(-1, 3, img_size, img_size) * 2 - 1
            lpips_loss = loss_fn_lpips(xh_img, x0_img).mean()

        loss = mse_loss + lpips_weight * lpips_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 200 == 0:
            print(f"  S1 [{step+1}/{n_steps}] total={loss.item():.6f} "
                  f"mse={mse_loss.item():.6f} lpips={lpips_loss.item():.6f}")

    # Compute latent stats
    with torch.no_grad():
        zs = []
        for _ in range(10):
            x0, _ = dataset.sample_clean(min(batch_size, 2000))
            x0 = x0.to(device)
            zs.append(model.encoder(x0))
        zs = torch.cat(zs, dim=0)
        model.z_mean = zs.mean(dim=0)
        model.z_std = zs.std(dim=0).clamp(min=1e-6)


def train_stage1_mmd(model, dataset, n_steps, lr=1e-4, batch_size=64,
                     lpips_weight=0.5, mmd_weight=10.0, device='cpu'):
    """Stage 1: MSE + LPIPS + MMD regularization."""
    img_size = model.image_shape[1]
    sigma_t0 = model.t0

    loss_fn_lpips = None
    if LPIPS_AVAILABLE:
        loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
        loss_fn_lpips.eval()
        for p in loss_fn_lpips.parameters():
            p.requires_grad = False

    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    print(f"  Stage 1 (MMD): {n_steps} steps, lpips_w={lpips_weight}, mmd_w={mmd_weight}")

    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)
        x_t = x0 + sigma_t0 * torch.randn_like(x0)
        z, x_hat = model.project(x_t)

        mse_loss = F.mse_loss(x_hat, x0)
        lpips_loss = torch.tensor(0.0, device=device)
        if loss_fn_lpips is not None:
            x0_img = x0.view(-1, 3, img_size, img_size) * 2 - 1
            xh_img = x_hat.view(-1, 3, img_size, img_size) * 2 - 1
            lpips_loss = loss_fn_lpips(xh_img, x0_img).mean()

        # MMD on normalized latent
        z_norm = (z - z.mean(dim=0)) / z.std(dim=0).clamp(min=1e-6)
        mmd = mmd_loss(z_norm)

        loss = mse_loss + lpips_weight * lpips_loss + mmd_weight * mmd

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 200 == 0:
            print(f"  S1 [{step+1}/{n_steps}] total={loss.item():.6f} "
                  f"mse={mse_loss.item():.6f} lpips={lpips_loss.item():.6f} "
                  f"mmd={mmd.item():.6f}")

    # Compute latent stats
    with torch.no_grad():
        zs = []
        for _ in range(10):
            x0, _ = dataset.sample_clean(min(batch_size, 2000))
            x0 = x0.to(device)
            zs.append(model.encoder(x0))
        zs = torch.cat(zs, dim=0)
        model.z_mean = zs.mean(dim=0)
        model.z_std = zs.std(dim=0).clamp(min=1e-6)


def train_vae_lpips(model, dataset, n_steps, lr=1e-4, batch_size=64,
                    lpips_weight=0.5, kl_weight=1e-4, device='cpu'):
    """VAE training: MSE + LPIPS + KL (standard LDM)."""
    img_size = int(np.sqrt(model.d // 3))

    loss_fn_lpips = None
    if LPIPS_AVAILABLE:
        loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
        loss_fn_lpips.eval()
        for p in loss_fn_lpips.parameters():
            p.requires_grad = False

    params = list(model.vae_encoder.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    print(f"  VAE (KL): {n_steps} steps, lpips_w={lpips_weight}, kl_w={kl_weight}")

    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)
        mu, logvar = model.vae_encoder(x0)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        x_recon = model.decoder(z)

        mse_loss = F.mse_loss(x_recon, x0)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        lpips_loss = torch.tensor(0.0, device=device)
        if loss_fn_lpips is not None:
            x0_img = x0.view(-1, 3, img_size, img_size) * 2 - 1
            xr_img = x_recon.view(-1, 3, img_size, img_size) * 2 - 1
            lpips_loss = loss_fn_lpips(xr_img, x0_img).mean()

        loss = mse_loss + lpips_weight * lpips_loss + kl_weight * kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 200 == 0:
            print(f"  VAE [{step+1}/{n_steps}] total={loss.item():.6f} "
                  f"mse={mse_loss.item():.6f} lpips={lpips_loss.item():.6f} "
                  f"kl={kl_loss.item():.4f}")

    with torch.no_grad():
        zs = []
        for _ in range(10):
            x0, _ = dataset.sample_clean(min(batch_size, 2000))
            x0 = x0.to(device)
            zs.append(model.vae_encoder.encode_deterministic(x0))
        zs = torch.cat(zs, dim=0)
        model.z_mean = zs.mean(dim=0)
        model.z_std = zs.std(dim=0).clamp(min=1e-6)


def evaluate_model(model, dataset, is_twostage=True, device='cpu'):
    """Evaluate eps prediction at multiple time points."""
    eval_times = [0.01, 0.1, 0.3, 0.5, 0.8]
    total = 0.0
    for t_eval in eval_times:
        with torch.no_grad():
            x0, _ = dataset.sample_clean(2000)
            x0 = x0.to(device)
            if is_twostage:
                z0 = model.encode(x0)
                t_batch = torch.full((2000,), t_eval, device=device)
                z_t, eps_z = model.forward_diffusion_latent(z0, t_batch)
                eps_pred = model.predict_eps_latent(z_t, t_batch)
            else:
                z0 = (model.vae_encoder.encode_deterministic(x0) -
                      model.z_mean) / model.z_std.clamp(min=1e-6)
                t_batch = torch.full((2000,), t_eval, device=device)
                alpha_bar, sigma = model.noise_schedule(t_batch)
                eps_z = torch.randn_like(z0)
                z_t = torch.sqrt(alpha_bar.unsqueeze(-1)) * z0 + \
                      sigma.unsqueeze(-1) * eps_z
                eps_pred = model.score_net(z_t, t_batch)
            mse = F.mse_loss(eps_pred, eps_z).item()
            total += mse
            print(f"  t={t_eval:.2f}: eps_MSE={mse:.6f}")
    return total / len(eval_times)


def compute_latent_stats(model, dataset, is_twostage=True, device='cpu'):
    """Compute statistics of the latent distribution for analysis."""
    with torch.no_grad():
        zs = []
        for _ in range(20):
            x0, _ = dataset.sample_clean(500)
            x0 = x0.to(device)
            if is_twostage:
                z = model.encode(x0)
            else:
                z = (model.vae_encoder.encode_deterministic(x0) -
                     model.z_mean) / model.z_std.clamp(min=1e-6)
            zs.append(z.cpu())
        zs = torch.cat(zs, dim=0)

        # Gaussianity metrics
        mean = zs.mean(dim=0)
        std = zs.std(dim=0)
        skew = ((zs - mean) ** 3).mean(dim=0) / (std ** 3 + 1e-8)
        kurt = ((zs - mean) ** 4).mean(dim=0) / (std ** 4 + 1e-8) - 3.0

        print(f"    Mean of means: {mean.mean().item():.4f} "
              f"(ideal: 0.0)")
        print(f"    Mean of stds:  {std.mean().item():.4f} "
              f"(ideal: 1.0)")
        print(f"    Mean |skewness|: {skew.abs().mean().item():.4f} "
              f"(ideal: 0.0)")
        print(f"    Mean |excess kurt|: {kurt.abs().mean().item():.4f} "
              f"(ideal: 0.0)")

        # MMD to N(0,I) on a subsample
        z_sub = zs[:512].to(device)
        mmd_val = mmd_loss(z_sub).item()
        print(f"    MMD to N(0,I): {mmd_val:.6f} (ideal: 0.0)")

    return {'mean': mean.mean().item(), 'std': std.mean().item(),
            'skew': skew.abs().mean().item(), 'kurt': kurt.abs().mean().item(),
            'mmd': mmd_val}


# ================================================================
# Main
# ================================================================

def run_experiment(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    img_size = 64
    image_shape = (3, img_size, img_size)
    d = 3 * img_size * img_size
    latent_dim = 256
    t0 = 0.005

    base_ch = 128
    hidden_dim_score = 1024
    n_score_layers = 8

    ae_steps = 20000
    diff_steps = 60000
    batch_size = 64
    lpips_weight = 0.5
    mmd_weight = 10.0  # MMD regularization strength

    print("=" * 70)
    print(f"CelebA MMD EXPERIMENT (d={d}, {img_size}x{img_size}x3)")
    print(f"  latent_dim={latent_dim}")
    print(f"  base_ch={base_ch}, score_hidden={hidden_dim_score}, "
          f"score_layers={n_score_layers}")
    print(f"  AE={ae_steps}, Diff={diff_steps}, batch={batch_size}")
    print(f"  LPIPS={lpips_weight}, MMD={mmd_weight}")
    print("=" * 70)

    dataset = CelebADataset(img_size=img_size, max_samples=20000)

    print("\n--- PCA Analysis ---")
    explained_var = dataset.pca_explained_variance
    cumvar = torch.cumsum(explained_var, dim=0) / explained_var.sum()
    for threshold in [0.90, 0.95, 0.99]:
        k_est = (cumvar < threshold).sum().item() + 1
        print(f"  {threshold*100:.0f}% variance at k={k_est}")
    k_95 = (cumvar < 0.95).sum().item() + 1

    results = {}

    # ================================================================
    # MODEL 1: Two-Stage (no regularization) — baseline
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODEL 1: Two-Stage (no regularization)")
    print("=" * 70)

    model_ts = TwoStageManifoldDiffusion(
        d=d, latent_dim=latent_dim, hidden_dim_enc=512,
        hidden_dim_score=hidden_dim_score, n_layers=n_score_layers,
        t0=t0, image_shape=image_shape, base_ch=base_ch,
    ).to(device)

    ckpt = load_checkpoint(model_ts, 'celeba_mmd_ts_full', device=device)
    if ckpt is not None:
        print("  Resumed from checkpoint.")
        model_ts.freeze_stage1()
        recon_ts = ckpt.get('recon_mse', 0.0)
        eps_ts = ckpt.get('avg_loss', 0.0)
    else:
        ckpt_s1 = load_checkpoint(model_ts, 'celeba_mmd_ts_stage1', device=device)
        if ckpt_s1 is not None:
            model_ts.freeze_stage1()
            recon_ts = ckpt_s1.get('recon_mse', 0.0)
        else:
            print("\n--- Stage 1: MSE + LPIPS ---")
            train_stage1_lpips(model_ts, dataset, n_steps=ae_steps, lr=1e-4,
                               batch_size=batch_size, lpips_weight=lpips_weight,
                               device=device)
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                _, x_hat = model_ts.project(x0)
                recon_ts = F.mse_loss(x_hat, x0).item()
                print(f"\n  Reconstruction MSE: {recon_ts:.6f}")
            model_ts.freeze_stage1()
            save_checkpoint(model_ts, 'celeba_mmd_ts_stage1',
                            {'recon_mse': recon_ts})

        print("\n--- Stage 2: Latent Diffusion ---")
        model_ts.train_stage2(dataset, n_steps=diff_steps, lr=3e-4,
                              batch_size=batch_size)
        eps_ts = evaluate_model(model_ts, dataset, is_twostage=True,
                                device=device)
        save_checkpoint(model_ts, 'celeba_mmd_ts_full',
                        {'recon_mse': recon_ts, 'avg_loss': eps_ts})

    print("\n  Latent distribution stats (Two-Stage, no reg):")
    stats_ts = compute_latent_stats(model_ts, dataset, is_twostage=True,
                                    device=device)
    results['ts'] = {'recon': recon_ts, 'eps': eps_ts, 'stats': stats_ts}

    # ================================================================
    # MODEL 2: Two-Stage + MMD
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODEL 2: Two-Stage + MMD")
    print("=" * 70)

    model_mmd = TwoStageManifoldDiffusion(
        d=d, latent_dim=latent_dim, hidden_dim_enc=512,
        hidden_dim_score=hidden_dim_score, n_layers=n_score_layers,
        t0=t0, image_shape=image_shape, base_ch=base_ch,
    ).to(device)

    ckpt = load_checkpoint(model_mmd, 'celeba_mmd_tsmmd_full', device=device)
    if ckpt is not None:
        print("  Resumed from checkpoint.")
        model_mmd.freeze_stage1()
        recon_mmd = ckpt.get('recon_mse', 0.0)
        eps_mmd = ckpt.get('avg_loss', 0.0)
    else:
        ckpt_s1 = load_checkpoint(model_mmd, 'celeba_mmd_tsmmd_stage1',
                                  device=device)
        if ckpt_s1 is not None:
            model_mmd.freeze_stage1()
            recon_mmd = ckpt_s1.get('recon_mse', 0.0)
        else:
            print(f"\n--- Stage 1: MSE + LPIPS + MMD (weight={mmd_weight}) ---")
            train_stage1_mmd(model_mmd, dataset, n_steps=ae_steps, lr=1e-4,
                             batch_size=batch_size, lpips_weight=lpips_weight,
                             mmd_weight=mmd_weight, device=device)
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                _, x_hat = model_mmd.project(x0)
                recon_mmd = F.mse_loss(x_hat, x0).item()
                print(f"\n  Reconstruction MSE: {recon_mmd:.6f}")
            model_mmd.freeze_stage1()
            save_checkpoint(model_mmd, 'celeba_mmd_tsmmd_stage1',
                            {'recon_mse': recon_mmd})

        print("\n--- Stage 2: Latent Diffusion ---")
        model_mmd.train_stage2(dataset, n_steps=diff_steps, lr=3e-4,
                               batch_size=batch_size)
        eps_mmd = evaluate_model(model_mmd, dataset, is_twostage=True,
                                 device=device)
        save_checkpoint(model_mmd, 'celeba_mmd_tsmmd_full',
                        {'recon_mse': recon_mmd, 'avg_loss': eps_mmd})

    print("\n  Latent distribution stats (Two-Stage + MMD):")
    stats_mmd = compute_latent_stats(model_mmd, dataset, is_twostage=True,
                                     device=device)
    results['mmd'] = {'recon': recon_mmd, 'eps': eps_mmd, 'stats': stats_mmd}

    # ================================================================
    # MODEL 3: LDM (KL regularization)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODEL 3: LDM (KL regularization)")
    print("=" * 70)

    model_ldm = LatentDiffusionCNN(
        d=d, latent_dim=latent_dim, hidden_dim_enc=512,
        hidden_dim_score=hidden_dim_score, n_layers=n_score_layers,
        kl_weight=1e-4, image_shape=image_shape, base_ch=base_ch,
    ).to(device)

    ckpt = load_checkpoint(model_ldm, 'celeba_mmd_ldm_full', device=device)
    if ckpt is not None:
        print("  Resumed from checkpoint.")
        model_ldm.freeze_ae()
        recon_ldm = ckpt.get('recon_mse', 0.0)
        eps_ldm = ckpt.get('avg_loss', 0.0)
    else:
        ckpt_ae = load_checkpoint(model_ldm, 'celeba_mmd_ldm_ae', device=device)
        if ckpt_ae is not None:
            model_ldm.freeze_ae()
            recon_ldm = ckpt_ae.get('recon_mse', 0.0)
        else:
            print("\n--- VAE: MSE + LPIPS + KL ---")
            train_vae_lpips(model_ldm, dataset, n_steps=ae_steps, lr=1e-4,
                            batch_size=batch_size, lpips_weight=lpips_weight,
                            kl_weight=1e-4, device=device)
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                z = model_ldm.vae_encoder.encode_deterministic(x0)
                x_recon = model_ldm.decoder(z)
                recon_ldm = F.mse_loss(x_recon, x0).item()
                print(f"\n  VAE Recon MSE: {recon_ldm:.6f}")
            model_ldm.freeze_ae()
            save_checkpoint(model_ldm, 'celeba_mmd_ldm_ae',
                            {'recon_mse': recon_ldm})

        print("\n--- Diffusion Training ---")
        model_ldm.train_diffusion(dataset, n_steps=diff_steps, lr=3e-4,
                                  batch_size=batch_size)
        eps_ldm = evaluate_model(model_ldm, dataset, is_twostage=False,
                                 device=device)
        save_checkpoint(model_ldm, 'celeba_mmd_ldm_full',
                        {'recon_mse': recon_ldm, 'avg_loss': eps_ldm})

    print("\n  Latent distribution stats (LDM, KL):")
    stats_ldm = compute_latent_stats(model_ldm, dataset, is_twostage=False,
                                     device=device)
    results['ldm'] = {'recon': recon_ldm, 'eps': eps_ldm, 'stats': stats_ldm}

    # ================================================================
    # GENERATE SAMPLES
    # ================================================================
    print("\n--- Generating Samples ---")
    with torch.no_grad():
        print("  Two-Stage (no reg): sampling...")
        x_gen_ts = model_ts.sample(64, n_steps=500, device=device,
                                   clamp=(0.0, 1.0)).cpu()
        plot_image_grid(x_gen_ts, nrow=8, title='Two-Stage (no reg)',
                        save_name='celeba_mmd_gen_ts.png', img_size=img_size)

        print("  Two-Stage + MMD: sampling...")
        x_gen_mmd = model_mmd.sample(64, n_steps=500, device=device,
                                     clamp=(0.0, 1.0)).cpu()
        plot_image_grid(x_gen_mmd, nrow=8, title='Two-Stage + MMD',
                        save_name='celeba_mmd_gen_tsmmd.png', img_size=img_size)

        print("  LDM (KL): sampling...")
        x_gen_ldm = model_ldm.sample(64, n_steps=500, device=device,
                                     clamp=(0.0, 1.0)).cpu()
        plot_image_grid(x_gen_ldm, nrow=8, title='LDM (KL)',
                        save_name='celeba_mmd_gen_ldm.png', img_size=img_size)

        x_real, _ = dataset.sample_clean(64)
        plot_image_grid(x_real, nrow=8, title='Real CelebA',
                        save_name='celeba_mmd_real.png', img_size=img_size)

    # ================================================================
    # FID
    # ================================================================
    fid_ts = fid_mmd = fid_ldm = float('nan')
    if FID_AVAILABLE:
        print("\n--- FID ---")
        n_fid = 2048
        fb = 64

        real_imgs = []
        for _ in range(n_fid // fb):
            x0, _ = dataset.sample_clean(fb)
            real_imgs.append((x0.view(-1, 3, img_size, img_size).clamp(0, 1)
                              * 255).to(torch.uint8))
        real_imgs = torch.cat(real_imgs)[:n_fid]

        models_for_fid = [
            ('Two-Stage', model_ts),
            ('Two-Stage+MMD', model_mmd),
            ('LDM', model_ldm),
        ]
        fid_results = {}

        with torch.no_grad():
            for name, model in models_for_fid:
                print(f"  Generating {n_fid} for {name}...")
                gen_imgs = []
                for i in range(n_fid // fb):
                    x = model.sample(fb, n_steps=500, device=device,
                                     clamp=(0.0, 1.0)).cpu()
                    gen_imgs.append((x.view(-1, 3, img_size, img_size)
                                    .clamp(0, 1) * 255).to(torch.uint8))
                    if (i + 1) % 8 == 0:
                        print(f"    {(i+1)*fb}/{n_fid}")
                gen_imgs = torch.cat(gen_imgs)[:n_fid]

                fid = FrechetInceptionDistance(feature=2048,
                                              normalize=False).to(device)
                fid.update(real_imgs.to(device), real=True)
                fid.update(gen_imgs.to(device), real=False)
                fid_val = fid.compute().item()
                fid_results[name] = fid_val
                print(f"  {name} FID: {fid_val:.2f}")
                del fid, gen_imgs
                torch.cuda.empty_cache()

        fid_ts = fid_results.get('Two-Stage', float('nan'))
        fid_mmd = fid_results.get('Two-Stage+MMD', float('nan'))
        fid_ldm = fid_results.get('LDM', float('nan'))
        del real_imgs

    # ================================================================
    # RECONSTRUCTION COMPARISON
    # ================================================================
    print("\n--- Reconstruction ---")
    with torch.no_grad():
        x_test, _ = dataset.sample_clean(8)
        x_d = x_test.to(device)
        _, xr_ts = model_ts.project(x_d)
        _, xr_mmd = model_mmd.project(x_d)
        z_l = model_ldm.vae_encoder.encode_deterministic(x_d)
        xr_ldm = model_ldm.decoder(z_l)

        fig, axes = plt.subplots(4, 8, figsize=(20, 10))
        labels = ['Original', 'Two-Stage', 'TS+MMD', 'LDM']
        images = [x_test, xr_ts.cpu(), xr_mmd.cpu(), xr_ldm.cpu()]
        for j in range(8):
            for row, imgs in enumerate(images):
                axes[row, j].imshow(imgs[j].view(3, img_size, img_size)
                                    .permute(1, 2, 0).clamp(0, 1).numpy())
                axes[row, j].axis('off')
        for row, label in enumerate(labels):
            axes[row, 0].set_ylabel(label, fontsize=11, rotation=0,
                                    labelpad=60, va='center')
        plt.suptitle('CelebA Reconstruction: no-reg vs MMD vs KL', fontsize=14)
        plt.tight_layout()
        path = os.path.join(FIG_DIR, 'celeba_mmd_reconstruction.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'=' * 70}")
    print("CelebA SUMMARY (Regularization Comparison)")
    print("=" * 70)
    print(f"  Intrinsic dim (95% PCA): k={k_95}")

    print(f"\n  {'Model':<25} {'Reg':<8} {'Eps Loss':<12} {'Recon MSE':<12} "
          f"{'FID':<10} {'|Skew|':<8} {'|Kurt|':<8} {'MMD':<10}")
    print("  " + "-" * 95)
    for name, key, fid_val in [
        ('Two-Stage', 'ts', fid_ts),
        ('Two-Stage+MMD', 'mmd', fid_mmd),
        ('LDM', 'ldm', fid_ldm),
    ]:
        r = results[key]
        reg = {'ts': 'none', 'mmd': 'MMD', 'ldm': 'KL'}[key]
        print(f"  {name:<25} {reg:<8} {r['eps']:<12.6f} {r['recon']:<12.6f} "
              f"{fid_val:<10.2f} {r['stats']['skew']:<8.4f} "
              f"{r['stats']['kurt']:<8.4f} {r['stats']['mmd']:<10.6f}")


if __name__ == '__main__':
    run_experiment()
