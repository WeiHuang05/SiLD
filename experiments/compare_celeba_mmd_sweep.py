"""
Experiment: CelebA — MMD weight sweep

Sweeps mmd_weight ∈ {0.1, 1.0, 5.0} for Two-Stage.
Reuses existing baselines (Two-Stage no-reg, LDM) from checkpoints.
Previous run showed mmd_weight=10.0 was too strong.
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.image_data import CelebADataset
from models.two_stage_latent import TwoStageManifoldDiffusion
from models.latent_diffusion_cnn import LatentDiffusionCNN
from utils.experiment_variant import (
    output_dir, experiment_variant, paired_stage2_t_min,
)

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False

FIG_DIR = output_dir(__file__, 'figures')
CKPT_DIR = output_dir(__file__, 'checkpoints')
LOG_DIR = output_dir(__file__, 'logs')


# ================================================================
# MMD
# ================================================================

def gaussian_kernel(x, y, sigma=1.0):
    dim = x.size(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    return torch.exp(-((x - y) ** 2).sum(2) / (2 * sigma ** 2 * dim))


def mmd_loss(z):
    batch_size = z.size(0)
    z_prior = torch.randn_like(z)
    mmd = torch.tensor(0.0, device=z.device)
    diag_mask = 1.0 - torch.eye(batch_size, device=z.device)
    for s in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        k_xx = gaussian_kernel(z, z, sigma=s)
        k_yy = gaussian_kernel(z_prior, z_prior, sigma=s)
        k_xy = gaussian_kernel(z, z_prior, sigma=s)
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
    fig, axes = plt.subplots(nrow_actual, ncol,
                             figsize=(ncol * 1.5, nrow_actual * 1.5))
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


def train_stage1_mmd(model, dataset, n_steps, lr, batch_size,
                     lpips_weight, mmd_weight, device):
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

    print(f"  Stage 1 (MMD={mmd_weight}): {n_steps} steps")

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

    with torch.no_grad():
        zs = []
        for _ in range(10):
            x0, _ = dataset.sample_clean(min(batch_size, 2000))
            x0 = x0.to(device)
            zs.append(model.encoder(x0))
        zs = torch.cat(zs, dim=0)
        model.z_mean = zs.mean(dim=0)
        model.z_std = zs.std(dim=0).clamp(min=1e-6)


def train_stage1_lpips(model, dataset, n_steps, lr, batch_size,
                       lpips_weight, device):
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

    print(f"  Stage 1 (no reg): {n_steps} steps")

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

    with torch.no_grad():
        zs = []
        for _ in range(10):
            x0, _ = dataset.sample_clean(min(batch_size, 2000))
            x0 = x0.to(device)
            zs.append(model.encoder(x0))
        zs = torch.cat(zs, dim=0)
        model.z_mean = zs.mean(dim=0)
        model.z_std = zs.std(dim=0).clamp(min=1e-6)


def train_vae_lpips(model, dataset, n_steps, lr, batch_size,
                    lpips_weight, kl_weight, device):
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

    print(f"  VAE (KL={kl_weight}): {n_steps} steps")

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


def evaluate_eps(model, dataset, is_twostage, device):
    eval_times = [0.01, 0.1, 0.3, 0.5, 0.8]
    total = 0.0
    for t_eval in eval_times:
        with torch.no_grad():
            x0, _ = dataset.sample_clean(2000)
            x0 = x0.to(device)
            if is_twostage:
                z0 = model.encode(x0)
                t_b = torch.full((2000,), t_eval, device=device)
                z_t, eps = model.forward_diffusion_latent(z0, t_b)
                eps_pred = model.predict_eps_latent(z_t, t_b)
            else:
                z0 = (model.vae_encoder.encode_deterministic(x0) -
                      model.z_mean) / model.z_std.clamp(min=1e-6)
                t_b = torch.full((2000,), t_eval, device=device)
                ab, sig = model.noise_schedule(t_b)
                eps = torch.randn_like(z0)
                z_t = torch.sqrt(ab.unsqueeze(-1)) * z0 + sig.unsqueeze(-1) * eps
                eps_pred = model.score_net(z_t, t_b)
            mse = F.mse_loss(eps_pred, eps).item()
            total += mse
            print(f"  t={t_eval:.2f}: eps_MSE={mse:.6f}")
    return total / len(eval_times)


def compute_latent_stats(model, dataset, is_twostage, device):
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
        mean = zs.mean(dim=0)
        std = zs.std(dim=0)
        skew = ((zs - mean) ** 3).mean(dim=0) / (std ** 3 + 1e-8)
        kurt = ((zs - mean) ** 4).mean(dim=0) / (std ** 4 + 1e-8) - 3.0
        z_sub = zs[:512].to(device)
        mmd_val = mmd_loss(z_sub).item()
        print(f"    |skew|={skew.abs().mean().item():.4f} "
              f"|kurt|={kurt.abs().mean().item():.4f} "
              f"mmd={mmd_val:.6f}")
    return {'skew': skew.abs().mean().item(),
            'kurt': kurt.abs().mean().item(), 'mmd': mmd_val}


def compute_fid(model, dataset, n_fid, fb, img_size, device):
    if not FID_AVAILABLE:
        return float('nan')
    real_imgs = []
    for _ in range(n_fid // fb):
        x0, _ = dataset.sample_clean(fb)
        real_imgs.append((x0.view(-1, 3, img_size, img_size).clamp(0, 1)
                          * 255).to(torch.uint8))
    real_imgs = torch.cat(real_imgs)[:n_fid]

    gen_imgs = []
    with torch.no_grad():
        for i in range(n_fid // fb):
            x = model.sample(fb, n_steps=500, device=device,
                              clamp=(0.0, 1.0)).cpu()
            gen_imgs.append((x.view(-1, 3, img_size, img_size).clamp(0, 1)
                             * 255).to(torch.uint8))
        gen_imgs = torch.cat(gen_imgs)[:n_fid]

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid.update(real_imgs.to(device), real=True)
    fid.update(gen_imgs.to(device), real=False)
    fid_val = fid.compute().item()
    del fid, real_imgs, gen_imgs
    torch.cuda.empty_cache()
    return fid_val


# ================================================================
# Main
# ================================================================

def run_experiment(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Experiment variant: {experiment_variant()}")

    img_size = 64
    image_shape = (3, img_size, img_size)
    d = 3 * img_size * img_size
    latent_dim = 256
    t0 = 0.005
    stage2_t_min = paired_stage2_t_min(t0)
    print(f"Paired Stage-2 time range: [{stage2_t_min}, 0.999]")

    base_ch = 128
    hidden_dim_score = 1024
    n_score_layers = 8

    ae_steps = 20000
    diff_steps = 60000
    batch_size = 64
    lpips_weight = 0.5

    mmd_weights = [0.1, 1.0, 5.0]  # sweep these

    print("=" * 70)
    print(f"CelebA MMD SWEEP (d={d}, {img_size}x{img_size}x3)")
    print(f"  latent_dim={latent_dim}, base_ch={base_ch}")
    print(f"  AE={ae_steps}, Diff={diff_steps}")
    print(f"  MMD weights to sweep: {mmd_weights}")
    print("=" * 70)

    dataset = CelebADataset(img_size=img_size, max_samples=20000)

    print("\n--- PCA Analysis ---")
    explained_var = dataset.pca_explained_variance
    cumvar = torch.cumsum(explained_var, dim=0) / explained_var.sum()
    for threshold in [0.90, 0.95, 0.99]:
        k_est = (cumvar < threshold).sum().item() + 1
        print(f"  {threshold*100:.0f}% variance at k={k_est}")
    k_95 = (cumvar < 0.95).sum().item() + 1

    all_results = {}

    def make_model():
        return TwoStageManifoldDiffusion(
            d=d, latent_dim=latent_dim, hidden_dim_enc=512,
            hidden_dim_score=hidden_dim_score, n_layers=n_score_layers,
            t0=t0, image_shape=image_shape, base_ch=base_ch,
            diffusion_t_min=stage2_t_min,
        ).to(device)

    # ================================================================
    # Baseline: Two-Stage (no reg)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Two-Stage (no reg)")
    print("=" * 70)

    model_ts = make_model()
    ckpt = load_checkpoint(model_ts, 'celeba_mmd_ts_full', device=device)
    if ckpt is not None:
        model_ts.freeze_stage1()
        recon = ckpt.get('recon_mse', 0.0)
        eps = ckpt.get('avg_loss', 0.0)
    else:
        train_stage1_lpips(model_ts, dataset, ae_steps, 1e-4, batch_size,
                           lpips_weight, device)
        with torch.no_grad():
            x0, _ = dataset.sample_clean(2000)
            x0 = x0.to(device)
            _, xh = model_ts.project(x0)
            recon = F.mse_loss(xh, x0).item()
            print(f"  Recon MSE: {recon:.6f}")
        model_ts.freeze_stage1()
        save_checkpoint(model_ts, 'celeba_mmd_ts_stage1', {'recon_mse': recon})
        model_ts.train_stage2(dataset, n_steps=diff_steps, lr=3e-4,
                              batch_size=batch_size)
        eps = evaluate_eps(model_ts, dataset, True, device)
        save_checkpoint(model_ts, 'celeba_mmd_ts_full',
                        {'recon_mse': recon, 'avg_loss': eps})

    stats = compute_latent_stats(model_ts, dataset, True, device)
    print(f"\n  Computing FID...")
    fid = compute_fid(model_ts, dataset, 2048, 64, img_size, device)
    print(f"  FID: {fid:.2f}")
    all_results['no_reg'] = {'recon': recon, 'eps': eps, 'fid': fid,
                             'stats': stats, 'model': model_ts}

    # ================================================================
    # Sweep: Two-Stage + MMD at different weights
    # ================================================================
    for mw in mmd_weights:
        tag = f'mmd{mw}'
        ckpt_name = f'celeba_mmdsweep_{tag}'
        print(f"\n{'=' * 70}")
        print(f"Two-Stage + MMD (weight={mw})")
        print("=" * 70)

        model = make_model()
        ckpt = load_checkpoint(model, f'{ckpt_name}_full', device=device)
        if ckpt is not None:
            model.freeze_stage1()
            recon = ckpt.get('recon_mse', 0.0)
            eps = ckpt.get('avg_loss', 0.0)
        else:
            ckpt_s1 = load_checkpoint(model, f'{ckpt_name}_stage1',
                                      device=device)
            if ckpt_s1 is not None:
                model.freeze_stage1()
                recon = ckpt_s1.get('recon_mse', 0.0)
            else:
                train_stage1_mmd(model, dataset, ae_steps, 1e-4, batch_size,
                                 lpips_weight, mw, device)
                with torch.no_grad():
                    x0, _ = dataset.sample_clean(2000)
                    x0 = x0.to(device)
                    _, xh = model.project(x0)
                    recon = F.mse_loss(xh, x0).item()
                    print(f"  Recon MSE: {recon:.6f}")
                model.freeze_stage1()
                save_checkpoint(model, f'{ckpt_name}_stage1',
                                {'recon_mse': recon})

            model.train_stage2(dataset, n_steps=diff_steps, lr=3e-4,
                               batch_size=batch_size)
            eps = evaluate_eps(model, dataset, True, device)
            save_checkpoint(model, f'{ckpt_name}_full',
                            {'recon_mse': recon, 'avg_loss': eps})

        stats = compute_latent_stats(model, dataset, True, device)
        print(f"\n  Computing FID...")
        fid = compute_fid(model, dataset, 2048, 64, img_size, device)
        print(f"  FID: {fid:.2f}")
        all_results[tag] = {'recon': recon, 'eps': eps, 'fid': fid,
                            'stats': stats, 'model': model}

    # ================================================================
    # Baseline: LDM (KL)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("LDM (KL)")
    print("=" * 70)

    model_ldm = LatentDiffusionCNN(
        d=d, latent_dim=latent_dim, hidden_dim_enc=512,
        hidden_dim_score=hidden_dim_score, n_layers=n_score_layers,
        kl_weight=1e-4, image_shape=image_shape, base_ch=base_ch,
        diffusion_t_min=stage2_t_min,
    ).to(device)

    ckpt = load_checkpoint(model_ldm, 'celeba_mmd_ldm_full', device=device)
    if ckpt is not None:
        model_ldm.freeze_ae()
        recon = ckpt.get('recon_mse', 0.0)
        eps = ckpt.get('avg_loss', 0.0)
    else:
        train_vae_lpips(model_ldm, dataset, ae_steps, 1e-4, batch_size,
                        lpips_weight, 1e-4, device)
        with torch.no_grad():
            x0, _ = dataset.sample_clean(2000)
            x0 = x0.to(device)
            z = model_ldm.vae_encoder.encode_deterministic(x0)
            xr = model_ldm.decoder(z)
            recon = F.mse_loss(xr, x0).item()
            print(f"  Recon MSE: {recon:.6f}")
        model_ldm.freeze_ae()
        save_checkpoint(model_ldm, 'celeba_mmd_ldm_ae', {'recon_mse': recon})
        model_ldm.train_diffusion(dataset, n_steps=diff_steps, lr=3e-4,
                                  batch_size=batch_size)
        eps = evaluate_eps(model_ldm, dataset, False, device)
        save_checkpoint(model_ldm, 'celeba_mmd_ldm_full',
                        {'recon_mse': recon, 'avg_loss': eps})

    stats = compute_latent_stats(model_ldm, dataset, False, device)
    print(f"\n  Computing FID...")
    fid = compute_fid(model_ldm, dataset, 2048, 64, img_size, device)
    print(f"  FID: {fid:.2f}")
    all_results['ldm'] = {'recon': recon, 'eps': eps, 'fid': fid,
                          'stats': stats}

    # ================================================================
    # Generate samples for all
    # ================================================================
    print("\n--- Generating Sample Grids ---")
    with torch.no_grad():
        for key in ['no_reg'] + [f'mmd{w}' for w in mmd_weights]:
            m = all_results[key]['model']
            x = m.sample(64, n_steps=500, device=device,
                         clamp=(0.0, 1.0)).cpu()
            plot_image_grid(x, nrow=8, title=f'TS {key}',
                            save_name=f'celeba_sweep_{key}.png',
                            img_size=img_size)
        x_ldm = model_ldm.sample(64, n_steps=500, device=device,
                                 clamp=(0.0, 1.0)).cpu()
        plot_image_grid(x_ldm, nrow=8, title='LDM (KL)',
                        save_name='celeba_sweep_ldm.png',
                        img_size=img_size)

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 70}")
    print("CelebA MMD SWEEP SUMMARY")
    print("=" * 70)
    print(f"  Intrinsic dim (95% PCA): k={k_95}")
    print(f"\n  {'Model':<25} {'MMD_w':<8} {'Eps Loss':<12} {'Recon MSE':<12} "
          f"{'FID':<10} {'|Skew|':<8} {'|Kurt|':<8} {'MMD→N':<10}")
    print("  " + "-" * 100)

    rows = [('TS (no reg)', 'no_reg', '-'),
            ('TS + MMD', 'mmd0.1', '0.1'),
            ('TS + MMD', 'mmd1.0', '1.0'),
            ('TS + MMD', 'mmd5.0', '5.0'),
            ('LDM (KL)', 'ldm', 'KL')]

    for name, key, mw in rows:
        r = all_results[key]
        print(f"  {name:<25} {mw:<8} {r['eps']:<12.6f} {r['recon']:<12.6f} "
              f"{r['fid']:<10.2f} {r['stats']['skew']:<8.4f} "
              f"{r['stats']['kurt']:<8.4f} {r['stats']['mmd']:<10.6f}")

    serializable = {
        key: {
            'recon_mse': value['recon'],
            'eps_loss': value['eps'],
            'fid': value['fid'],
            'latent_stats': value['stats'],
        }
        for key, value in all_results.items()
    }
    out_path = os.path.join(LOG_DIR, 'celeba_mmd_sweep_results.json')
    with open(out_path, 'w') as f:
        json.dump({
            'experiment_variant': experiment_variant(),
            'stage2_time_range': [stage2_t_min, 0.999],
            'config': {
                't0': t0, 'ae_steps': ae_steps, 'diff_steps': diff_steps,
                'batch_size': batch_size, 'mmd_weights': mmd_weights,
            },
            'results': serializable,
        }, f, indent=2)
    print(f"  Saved results: {out_path}")


if __name__ == '__main__':
    run_experiment()
