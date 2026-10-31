"""
Experiment: CelebA (64x64x3, d=12288) — with PatchGAN + EMA

Fair comparison: Two-Stage vs LDM-CNN with identical architecture.
  - PatchGAN discriminator for adversarial AE training
  - EMA (Exponential Moving Average) for both AE and diffusion
  - LPIPS weight = 1.0
  - FID evaluation
"""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import copy
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
from utils.experiment_variant import (
    output_dir, experiment_variant, paired_stage2_t_min,
)

# LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available, using MSE only")

# FID
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: torchmetrics not available, skipping FID")

FIG_DIR = output_dir(__file__, 'figures')
CKPT_DIR = output_dir(__file__, 'checkpoints')
LOG_DIR = output_dir(__file__, 'logs')


# ================================================================
# PatchGAN Discriminator
# ================================================================
class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator (from pix2pix / VQ-GAN).
    Classifies whether each NxN patch is real or fake.
    """
    def __init__(self, in_channels=3, base_ch=64, n_layers=3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_ch
        for i in range(1, n_layers):
            ch_next = min(ch * 2, 512)
            layers += [
                nn.Conv2d(ch, ch_next, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ch_next),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = ch_next
        # Final layer: 1-channel prediction map
        layers += [
            nn.Conv2d(ch, ch, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, 1, 4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def hinge_loss_d(real_pred, fake_pred):
    """Discriminator hinge loss."""
    return (F.relu(1.0 - real_pred).mean() + F.relu(1.0 + fake_pred).mean()) * 0.5


def hinge_loss_g(fake_pred):
    """Generator hinge loss."""
    return -fake_pred.mean()


# ================================================================
# EMA helper
# ================================================================
class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)
        # Also update buffers (e.g., batch norm running stats)
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.data.copy_(m_buf.data)

    def apply(self, model):
        """Copy EMA weights into model."""
        model.load_state_dict(self.shadow.state_dict())


# ================================================================
# Checkpoint helpers
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
# Training with PatchGAN + LPIPS + EMA
# ================================================================
def train_stage1_gan(model, dataset, n_steps=10000, lr_g=1e-4, lr_d=1e-4,
                     batch_size=64, lpips_weight=1.0, adv_weight=0.1,
                     ema_decay=0.9999, device='cpu'):
    """
    Train Stage 1 (denoising AE) with:
      L_total = MSE + lpips_weight * LPIPS + adv_weight * G_hinge
    Plus a PatchGAN discriminator and EMA.
    """
    img_size = model.image_shape[1]
    sigma_t0 = model.t0

    # LPIPS
    loss_fn_lpips = None
    if LPIPS_AVAILABLE:
        loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
        loss_fn_lpips.eval()
        for p in loss_fn_lpips.parameters():
            p.requires_grad = False

    # Discriminator
    disc = PatchDiscriminator(in_channels=3, base_ch=64, n_layers=3).to(device)
    n_disc_params = sum(p.numel() for p in disc.parameters())
    print(f"  PatchGAN discriminator: {n_disc_params:,} params")

    # Generator = encoder + decoder
    gen_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    opt_g = torch.optim.Adam(gen_params, lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # EMA
    ema = EMA(model, decay=ema_decay)

    print(f"  Stage 1: t0={model.t0}, sigma_t0={sigma_t0:.4f}, "
          f"latent_dim={model.latent_dim}")
    print(f"  Training {n_steps} steps, lr_g={lr_g}, lr_d={lr_d}, "
          f"batch_size={batch_size}")
    print(f"  LPIPS weight={lpips_weight}, adv_weight={adv_weight}, "
          f"EMA decay={ema_decay}")

    # Warmup: skip discriminator for first N steps so AE learns basic reconstruction
    disc_start = min(1000, n_steps // 10)
    print(f"  Discriminator starts at step {disc_start}")

    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)
        x_t = x0 + sigma_t0 * torch.randn_like(x0)

        # ---- Generator step ----
        z, x_hat = model.project(x_t)
        x0_img = x0.view(-1, 3, img_size, img_size)
        xh_img = x_hat.view(-1, 3, img_size, img_size)

        # MSE
        mse_loss = F.mse_loss(x_hat, x0)

        # LPIPS
        if loss_fn_lpips is not None:
            lpips_loss = loss_fn_lpips(xh_img * 2 - 1, x0_img * 2 - 1).mean()
        else:
            lpips_loss = torch.tensor(0.0, device=device)

        # Adversarial (generator wants disc to say "real")
        if step >= disc_start:
            fake_pred = disc(xh_img)
            g_adv_loss = hinge_loss_g(fake_pred)
        else:
            g_adv_loss = torch.tensor(0.0, device=device)

        g_loss = mse_loss + lpips_weight * lpips_loss + adv_weight * g_adv_loss

        opt_g.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
        opt_g.step()

        # ---- Discriminator step ----
        d_loss = torch.tensor(0.0, device=device)
        if step >= disc_start:
            with torch.no_grad():
                _, x_hat_d = model.project(x_t)
                xh_img_d = x_hat_d.view(-1, 3, img_size, img_size)

            real_pred = disc(x0_img)
            fake_pred = disc(xh_img_d)
            d_loss = hinge_loss_d(real_pred, fake_pred)

            opt_d.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            opt_d.step()

        # EMA update
        ema.update(model)

        if (step + 1) % 200 == 0:
            print(f"  Stage 1 [{step+1}/{n_steps}] "
                  f"g_total={g_loss.item():.6f} mse={mse_loss.item():.6f} "
                  f"lpips={lpips_loss.item():.6f} g_adv={g_adv_loss.item():.6f} "
                  f"d_loss={d_loss.item():.6f}")

    # Apply EMA weights
    ema.apply(model)
    print("  Applied EMA weights.")

    # Compute latent normalization
    with torch.no_grad():
        zs = []
        for _ in range(10):
            x0, _ = dataset.sample_clean(min(batch_size, 2000))
            x0 = x0.to(device)
            zs.append(model.encoder(x0))
        zs = torch.cat(zs, dim=0)
        model.z_mean = zs.mean(dim=0)
        model.z_std = zs.std(dim=0).clamp(min=1e-6)

    return disc


def train_vae_gan(model, dataset, n_steps=10000, lr_g=1e-4, lr_d=1e-4,
                  batch_size=64, lpips_weight=1.0, adv_weight=0.1,
                  kl_weight=1e-4, ema_decay=0.9999, device='cpu'):
    """
    Train VAE (LDM encoder) with MSE + LPIPS + KL + PatchGAN + EMA.
    """
    img_size = int(np.sqrt(model.d // 3))

    # LPIPS
    loss_fn_lpips = None
    if LPIPS_AVAILABLE:
        loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
        loss_fn_lpips.eval()
        for p in loss_fn_lpips.parameters():
            p.requires_grad = False

    # Discriminator
    disc = PatchDiscriminator(in_channels=3, base_ch=64, n_layers=3).to(device)
    n_disc_params = sum(p.numel() for p in disc.parameters())
    print(f"  PatchGAN discriminator: {n_disc_params:,} params")

    gen_params = (list(model.vae_encoder.parameters()) +
                  list(model.decoder.parameters()))
    opt_g = torch.optim.Adam(gen_params, lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # EMA
    ema = EMA(model, decay=ema_decay)

    print(f"  VAE-GAN training: {n_steps} steps, lr_g={lr_g}, lr_d={lr_d}, "
          f"batch_size={batch_size}")
    print(f"  LPIPS={lpips_weight}, adv={adv_weight}, kl={kl_weight}, "
          f"EMA={ema_decay}")

    disc_start = min(1000, n_steps // 10)
    print(f"  Discriminator starts at step {disc_start}")

    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)

        # ---- Generator step ----
        mu, logvar = model.vae_encoder(x0)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        x_recon = model.decoder(z)

        x0_img = x0.view(-1, 3, img_size, img_size)
        xr_img = x_recon.view(-1, 3, img_size, img_size)

        mse_loss = F.mse_loss(x_recon, x0)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        if loss_fn_lpips is not None:
            lpips_loss = loss_fn_lpips(xr_img * 2 - 1, x0_img * 2 - 1).mean()
        else:
            lpips_loss = torch.tensor(0.0, device=device)

        if step >= disc_start:
            fake_pred = disc(xr_img)
            g_adv_loss = hinge_loss_g(fake_pred)
        else:
            g_adv_loss = torch.tensor(0.0, device=device)

        g_loss = (mse_loss + lpips_weight * lpips_loss +
                  kl_weight * kl_loss + adv_weight * g_adv_loss)

        opt_g.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
        opt_g.step()

        # ---- Discriminator step ----
        d_loss = torch.tensor(0.0, device=device)
        if step >= disc_start:
            with torch.no_grad():
                mu_d, _ = model.vae_encoder(x0)
                x_recon_d = model.decoder(mu_d)
                xr_img_d = x_recon_d.view(-1, 3, img_size, img_size)

            real_pred = disc(x0_img)
            fake_pred = disc(xr_img_d)
            d_loss = hinge_loss_d(real_pred, fake_pred)

            opt_d.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            opt_d.step()

        # EMA update
        ema.update(model)

        if (step + 1) % 200 == 0:
            print(f"  VAE-GAN [{step+1}/{n_steps}] "
                  f"g_total={g_loss.item():.6f} mse={mse_loss.item():.6f} "
                  f"lpips={lpips_loss.item():.6f} kl={kl_loss.item():.4f} "
                  f"g_adv={g_adv_loss.item():.6f} d_loss={d_loss.item():.6f}")

    # Apply EMA weights
    ema.apply(model)
    print("  Applied EMA weights.")

    # Compute latent normalization
    with torch.no_grad():
        zs = []
        for _ in range(10):
            x0, _ = dataset.sample_clean(min(batch_size, 2000))
            x0 = x0.to(device)
            zs.append(model.vae_encoder.encode_deterministic(x0))
        zs = torch.cat(zs, dim=0)
        model.z_mean = zs.mean(dim=0)
        model.z_std = zs.std(dim=0).clamp(min=1e-6)

    return disc


def train_diffusion_with_ema(model, dataset, n_steps, lr, batch_size,
                             ema_decay=0.9999, is_twostage=True):
    """Train latent diffusion (Stage 2) with EMA."""
    device = next(model.parameters()).device
    ema = EMA(model, decay=ema_decay)
    t_min = float(model.diffusion_t_min)
    t_max = float(model.T)
    print(f"  Stage 2 time range: [{t_min}, {t_max}]")

    if is_twostage:
        # Use the model's own train_stage2 but manually with EMA
        score_params = list(model.score_net.parameters())
        optimizer = torch.optim.Adam(score_params, lr=lr)

        for step in range(n_steps):
            x0, _ = dataset.sample_clean(batch_size)
            x0 = x0.to(device)

            with torch.no_grad():
                z0 = model.encode(x0)

            t = torch.rand(z0.shape[0], device=device) * (t_max - t_min) + t_min
            z_t, eps_z = model.forward_diffusion_latent(z0, t)
            eps_z_pred = model.predict_eps_latent(z_t, t)
            loss = F.mse_loss(eps_z_pred, eps_z)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(score_params, 1.0)
            optimizer.step()
            ema.update(model)

            if (step + 1) % 200 == 0:
                print(f"  Stage 2 [{step+1}/{n_steps}] "
                      f"latent_eps_loss={loss.item():.6f}")
    else:
        # LDM diffusion training with EMA
        score_params = list(model.score_net.parameters())
        optimizer = torch.optim.Adam(score_params, lr=lr)

        for step in range(n_steps):
            x0, _ = dataset.sample_clean(batch_size)
            x0 = x0.to(device)

            with torch.no_grad():
                z0 = model.vae_encoder.encode_deterministic(x0)
                z0 = (z0 - model.z_mean) / model.z_std.clamp(min=1e-6)

            t = torch.rand(z0.shape[0], device=device) * (t_max - t_min) + t_min
            alpha_bar, sigma = model.noise_schedule(t)
            eps_z = torch.randn_like(z0)
            z_t = torch.sqrt(alpha_bar.unsqueeze(-1)) * z0 + sigma.unsqueeze(-1) * eps_z
            eps_z_pred = model.score_net(z_t, t)
            loss = F.mse_loss(eps_z_pred, eps_z)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(score_params, 1.0)
            optimizer.step()
            ema.update(model)

            if (step + 1) % 200 == 0:
                print(f"  Diffusion [{step+1}/{n_steps}] "
                      f"latent_eps_loss={loss.item():.6f}")

    # Apply EMA
    ema.apply(model)
    print("  Applied EMA weights to score network.")


# ================================================================
# Main experiment
# ================================================================
def run_experiment(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Experiment variant: {experiment_variant()}")

    img_size = 64
    image_shape = (3, img_size, img_size)
    d = 3 * img_size * img_size  # 12288
    latent_dim = 256
    t0 = 0.005
    stage2_t_min = paired_stage2_t_min(t0)
    print(f"Paired Stage-2 time range: [{stage2_t_min}, 0.999]")

    # Network params
    base_ch = 128
    hidden_dim_score = 1024
    n_score_layers = 8

    # Training params
    ae_steps = 100000
    diff_steps = 300000
    batch_size = 64
    lpips_weight = 1.0       # increased from 0.5
    adv_weight = 0.1         # PatchGAN weight
    ema_decay = 0.9999

    print("=" * 70)
    print(f"CelebA EXPERIMENT — GAN + EMA (d={d}, {img_size}x{img_size}x3)")
    print(f"  latent_dim={latent_dim}")
    print(f"  base_ch={base_ch}, score_hidden={hidden_dim_score}, "
          f"score_layers={n_score_layers}")
    print(f"  AE steps={ae_steps}, Diffusion steps={diff_steps}")
    print(f"  LPIPS={lpips_weight}, adv={adv_weight}, EMA={ema_decay}")
    print("=" * 70)

    dataset = CelebADataset(img_size=img_size, max_samples=20000)

    # ================================================================
    # PCA ANALYSIS
    # ================================================================
    print("\n--- PCA Analysis ---")
    explained_var = dataset.pca_explained_variance
    cumvar = torch.cumsum(explained_var, dim=0) / explained_var.sum()

    for threshold in [0.90, 0.95, 0.99]:
        k_est = (cumvar < threshold).sum().item() + 1
        print(f"  {threshold*100:.0f}% variance at k={k_est}")

    k_95 = (cumvar < 0.95).sum().item() + 1

    # ================================================================
    # MODEL 1: Two-Stage Manifold Diffusion
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODEL 1: Two-Stage Manifold Diffusion")
    print("=" * 70)

    model_ts = TwoStageManifoldDiffusion(
        d=d, latent_dim=latent_dim, hidden_dim_enc=512,
        hidden_dim_score=hidden_dim_score,
        n_layers=n_score_layers, t0=t0, image_shape=image_shape,
        base_ch=base_ch, diffusion_t_min=stage2_t_min,
    ).to(device)

    ckpt_ts_full = load_checkpoint(model_ts, 'celeba_gan_twostage_full',
                                   device=device)
    if ckpt_ts_full is not None:
        print("  Resumed Two-Stage (full) from checkpoint.")
        model_ts.freeze_stage1()
        recon_mse_ts = ckpt_ts_full.get('recon_mse', 0.0)
        avg_loss_ts = ckpt_ts_full.get('avg_loss', 0.0)
    else:
        ckpt_ts_s1 = load_checkpoint(model_ts, 'celeba_gan_twostage_stage1',
                                     device=device)
        if ckpt_ts_s1 is not None:
            print("  Resumed Two-Stage Stage 1 from checkpoint.")
            model_ts.freeze_stage1()
            recon_mse_ts = ckpt_ts_s1.get('recon_mse', 0.0)
        else:
            print("\n--- Stage 1: Manifold Learning with GAN + LPIPS + EMA ---")
            train_stage1_gan(
                model_ts, dataset, n_steps=ae_steps, lr_g=1e-4, lr_d=1e-4,
                batch_size=batch_size, lpips_weight=lpips_weight,
                adv_weight=adv_weight, ema_decay=ema_decay, device=device,
            )

            with torch.no_grad():
                x0_test, _ = dataset.sample_clean(2000)
                x0_test = x0_test.to(device)
                z_test, x_hat_test = model_ts.project(x0_test)
                recon_mse_ts = F.mse_loss(x_hat_test, x0_test).item()
                z_norm = model_ts.encode(x0_test)
                z_var = z_norm.var(dim=0).sort(descending=True).values
                active_dims = (z_var > z_var[0] * 0.01).sum().item()
                print(f"\n  Reconstruction MSE: {recon_mse_ts:.6f}")
                print(f"  Active latent dims: {active_dims}/{latent_dim}")

            model_ts.freeze_stage1()
            print("  Stage 1 frozen.")
            save_checkpoint(model_ts, 'celeba_gan_twostage_stage1',
                            {'recon_mse': recon_mse_ts})

        # Stage 2: latent diffusion with EMA
        print("\n--- Stage 2: Latent Diffusion with EMA ---")
        train_diffusion_with_ema(
            model_ts, dataset, n_steps=diff_steps, lr=3e-4,
            batch_size=batch_size, ema_decay=ema_decay, is_twostage=True,
        )

        # Evaluate
        eval_times = [0.01, 0.1, 0.3, 0.5, 0.8]
        total_loss_ts = 0.0
        for t_eval in eval_times:
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                z0 = model_ts.encode(x0)
                t_batch = torch.full((2000,), t_eval, device=device)
                z_t, eps_z = model_ts.forward_diffusion_latent(z0, t_batch)
                eps_z_pred = model_ts.predict_eps_latent(z_t, t_batch)
                mse = F.mse_loss(eps_z_pred, eps_z).item()
                total_loss_ts += mse
                print(f"  t={t_eval:.2f}: latent_eps_MSE={mse:.6f}")
        avg_loss_ts = total_loss_ts / len(eval_times)

        save_checkpoint(model_ts, 'celeba_gan_twostage_full',
                        {'recon_mse': recon_mse_ts, 'avg_loss': avg_loss_ts})

    # ================================================================
    # MODEL 2: LDM-CNN (fair comparison)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODEL 2: LDM-CNN (fair comparison)")
    print("=" * 70)

    model_ldm = LatentDiffusionCNN(
        d=d, latent_dim=latent_dim, hidden_dim_enc=512,
        hidden_dim_score=hidden_dim_score,
        n_layers=n_score_layers, kl_weight=1e-4, image_shape=image_shape,
        base_ch=base_ch, diffusion_t_min=stage2_t_min,
    ).to(device)

    ckpt_ldm_full = load_checkpoint(model_ldm, 'celeba_gan_ldm_full',
                                    device=device)
    if ckpt_ldm_full is not None:
        print("  Resumed LDM (full) from checkpoint.")
        model_ldm.freeze_ae()
        recon_mse_ldm = ckpt_ldm_full.get('recon_mse', 0.0)
        avg_loss_ldm = ckpt_ldm_full.get('avg_loss', 0.0)
    else:
        ckpt_ldm_ae = load_checkpoint(model_ldm, 'celeba_gan_ldm_ae',
                                      device=device)
        if ckpt_ldm_ae is not None:
            print("  Resumed LDM AE from checkpoint.")
            model_ldm.freeze_ae()
            recon_mse_ldm = ckpt_ldm_ae.get('recon_mse', 0.0)
        else:
            print("\n--- VAE-GAN Training with LPIPS + EMA ---")
            train_vae_gan(
                model_ldm, dataset, n_steps=ae_steps, lr_g=1e-4, lr_d=1e-4,
                batch_size=batch_size, lpips_weight=lpips_weight,
                adv_weight=adv_weight, kl_weight=1e-4, ema_decay=ema_decay,
                device=device,
            )

            with torch.no_grad():
                x0_test, _ = dataset.sample_clean(2000)
                x0_test = x0_test.to(device)
                z_ldm = model_ldm.vae_encoder.encode_deterministic(x0_test)
                x_recon = model_ldm.decoder(z_ldm)
                recon_mse_ldm = F.mse_loss(x_recon, x0_test).item()
                print(f"\n  VAE Reconstruction MSE: {recon_mse_ldm:.6f}")

            model_ldm.freeze_ae()
            save_checkpoint(model_ldm, 'celeba_gan_ldm_ae',
                            {'recon_mse': recon_mse_ldm})

        # Diffusion with EMA
        print("\n--- Diffusion Training with EMA ---")
        train_diffusion_with_ema(
            model_ldm, dataset, n_steps=diff_steps, lr=3e-4,
            batch_size=batch_size, ema_decay=ema_decay, is_twostage=False,
        )

        # Evaluate LDM
        eval_times = [0.01, 0.1, 0.3, 0.5, 0.8]
        total_loss_ldm = 0.0
        for t_eval in eval_times:
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                z0 = (model_ldm.vae_encoder.encode_deterministic(x0) -
                      model_ldm.z_mean) / model_ldm.z_std.clamp(min=1e-6)
                t_batch = torch.full((2000,), t_eval, device=device)
                alpha_bar, sigma = model_ldm.noise_schedule(t_batch)
                eps_z = torch.randn_like(z0)
                z_t = (torch.sqrt(alpha_bar.unsqueeze(-1)) * z0 +
                       sigma.unsqueeze(-1) * eps_z)
                eps_z_pred = model_ldm.score_net(z_t, t_batch)
                mse = F.mse_loss(eps_z_pred, eps_z).item()
                total_loss_ldm += mse
                print(f"  t={t_eval:.2f}: latent_eps_MSE={mse:.6f}")
        avg_loss_ldm = total_loss_ldm / len(eval_times)

        save_checkpoint(model_ldm, 'celeba_gan_ldm_full',
                        {'recon_mse': recon_mse_ldm, 'avg_loss': avg_loss_ldm})

    results_ts = {'total_loss': avg_loss_ts, 'recon_error': recon_mse_ts}
    results_ldm = {'total_loss': avg_loss_ldm, 'recon_error': recon_mse_ldm}

    # ================================================================
    # GENERATE SAMPLES
    # ================================================================
    print("\n--- Generating Samples ---")

    with torch.no_grad():
        print("  Two-Stage: sampling...")
        x_gen_ts = model_ts.sample(64, n_steps=500, device=device,
                                   clamp=(0.0, 1.0)).cpu()
        plot_image_grid(x_gen_ts, nrow=8, title='Two-Stage (GAN+EMA)',
                        save_name='celeba_gan_gen_twostage.png',
                        img_size=img_size)

        print("  LDM-CNN: sampling...")
        x_gen_ldm = model_ldm.sample(64, n_steps=500, device=device,
                                     clamp=(0.0, 1.0)).cpu()
        plot_image_grid(x_gen_ldm, nrow=8, title='LDM-CNN (GAN+EMA)',
                        save_name='celeba_gan_gen_ldm_cnn.png',
                        img_size=img_size)

        x_real, _ = dataset.sample_clean(64)
        plot_image_grid(x_real, nrow=8, title='Real CelebA',
                        save_name='celeba_gan_real.png', img_size=img_size)

    # ================================================================
    # FID EVALUATION
    # ================================================================
    if FID_AVAILABLE:
        print("\n--- FID Evaluation ---")
        n_fid = 2048
        fid_batch = 64

        # Collect real images
        real_imgs = []
        for _ in range(n_fid // fid_batch):
            x0, _ = dataset.sample_clean(fid_batch)
            imgs = x0.view(-1, 3, img_size, img_size).clamp(0, 1)
            real_imgs.append((imgs * 255).to(torch.uint8))
        real_imgs = torch.cat(real_imgs, dim=0)[:n_fid]

        # Generate for FID
        with torch.no_grad():
            print(f"  Generating {n_fid} samples for FID...")
            gen_ts_imgs = []
            gen_ldm_imgs = []
            for i in range(n_fid // fid_batch):
                x_ts = model_ts.sample(fid_batch, n_steps=500, device=device,
                                       clamp=(0.0, 1.0)).cpu()
                gen_ts_imgs.append(
                    (x_ts.view(-1, 3, img_size, img_size).clamp(0, 1) * 255
                     ).to(torch.uint8))
                x_ldm = model_ldm.sample(fid_batch, n_steps=500, device=device,
                                         clamp=(0.0, 1.0)).cpu()
                gen_ldm_imgs.append(
                    (x_ldm.view(-1, 3, img_size, img_size).clamp(0, 1) * 255
                     ).to(torch.uint8))
                if (i + 1) % 8 == 0:
                    print(f"    {(i+1)*fid_batch}/{n_fid} samples generated")
            gen_ts_imgs = torch.cat(gen_ts_imgs, dim=0)[:n_fid]
            gen_ldm_imgs = torch.cat(gen_ldm_imgs, dim=0)[:n_fid]

        # FID Two-Stage
        fid_metric = FrechetInceptionDistance(feature=2048,
                                             normalize=False).to(device)
        fid_metric.update(real_imgs.to(device), real=True)
        fid_metric.update(gen_ts_imgs.to(device), real=False)
        fid_ts = fid_metric.compute().item()
        print(f"  Two-Stage FID: {fid_ts:.2f}")

        # FID LDM
        fid_metric.reset()
        fid_metric.update(real_imgs.to(device), real=True)
        fid_metric.update(gen_ldm_imgs.to(device), real=False)
        fid_ldm = fid_metric.compute().item()
        print(f"  LDM-CNN FID: {fid_ldm:.2f}")

        del fid_metric, real_imgs, gen_ts_imgs, gen_ldm_imgs
        torch.cuda.empty_cache()
    else:
        fid_ts = float('nan')
        fid_ldm = float('nan')

    # ================================================================
    # RECONSTRUCTION COMPARISON
    # ================================================================
    print("\n--- Reconstruction Comparison ---")
    with torch.no_grad():
        x_test, _ = dataset.sample_clean(8)
        x_test_d = x_test.to(device)
        _, x_recon_ts = model_ts.project(x_test_d)
        z_ldm = model_ldm.vae_encoder.encode_deterministic(x_test_d)
        x_recon_ldm = model_ldm.decoder(z_ldm)
        x_recon_ts = x_recon_ts.cpu()
        x_recon_ldm = x_recon_ldm.cpu()

        fig, axes = plt.subplots(3, 8, figsize=(20, 7.5))
        for j in range(8):
            for row, img in enumerate([x_test[j], x_recon_ts[j],
                                       x_recon_ldm[j]]):
                img_np = (img.view(3, img_size, img_size)
                          .permute(1, 2, 0).clamp(0, 1).numpy())
                axes[row, j].imshow(img_np)
                axes[row, j].axis('off')

        axes[0, 0].set_ylabel('Original', fontsize=11, rotation=0,
                              labelpad=60, va='center')
        axes[1, 0].set_ylabel('Two-Stage', fontsize=11, rotation=0,
                              labelpad=60, va='center')
        axes[2, 0].set_ylabel('LDM-CNN', fontsize=11, rotation=0,
                              labelpad=60, va='center')
        plt.suptitle('CelebA Reconstruction (GAN+EMA)', fontsize=14)
        plt.tight_layout()
        path = os.path.join(FIG_DIR, 'celeba_gan_reconstruction.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # ================================================================
    # SIDE-BY-SIDE COMPARISON
    # ================================================================
    print("\n--- Side-by-side Comparison ---")
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, (title, imgs) in zip(axes, [
        ('Real', x_real[:16]),
        ('Two-Stage', x_gen_ts[:16]),
        ('LDM-CNN', x_gen_ldm[:16]),
    ]):
        imgs_np = (imgs.view(-1, 3, img_size, img_size)
                   .permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy())
        grid = np.zeros((4 * img_size, 4 * img_size, 3))
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                if idx < len(imgs_np):
                    grid[i*img_size:(i+1)*img_size,
                         j*img_size:(j+1)*img_size] = imgs_np[idx]
        ax.imshow(grid)
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    plt.suptitle('CelebA (GAN+EMA): Real vs Two-Stage vs LDM-CNN',
                 fontsize=16)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'celeba_gan_generation_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'=' * 70}")
    print("CelebA SUMMARY (GAN + EMA)")
    print("=" * 70)
    print(f"  Intrinsic dim (95% PCA): k={k_95}")
    print(f"\n  {'Model':<30} {'Latent':<8} {'Eps Loss':<12} "
          f"{'Recon MSE':<12} {'FID':<10}")
    print("  " + "-" * 72)
    print(f"  {'Two-Stage (ours)':<30} {latent_dim:<8} "
          f"{results_ts['total_loss']:<12.6f} "
          f"{results_ts['recon_error']:<12.6f} {fid_ts:<10.2f}")
    print(f"  {'LDM-CNN (fair)':<30} {latent_dim:<8} "
          f"{results_ldm['total_loss']:<12.6f} "
          f"{results_ldm['recon_error']:<12.6f} {fid_ldm:<10.2f}")

    out_path = os.path.join(LOG_DIR, 'celeba_gan_results.json')
    with open(out_path, 'w') as f:
        json.dump({
            'experiment_variant': experiment_variant(),
            'stage2_time_range': [stage2_t_min, 0.999],
            'config': {
                't0': t0, 'ae_steps': ae_steps, 'diff_steps': diff_steps,
                'batch_size': batch_size, 'ema_decay': ema_decay,
                'lpips_weight': lpips_weight, 'adv_weight': adv_weight,
            },
            'sild': {
                'eps_loss': avg_loss_ts, 'recon_mse': recon_mse_ts,
                'fid': fid_ts,
            },
            'ldm': {
                'eps_loss': avg_loss_ldm, 'recon_mse': recon_mse_ldm,
                'fid': fid_ldm,
            },
        }, f, indent=2)
    print(f"  Saved results: {out_path}")


if __name__ == '__main__':
    run_experiment()
