"""
Generate CelebA-HQ samples from best checkpoints (GAN+EMA, FID 31/31).
Loads celebahq_gan_twostage_full.pt and celebahq_gan_ldm_full.pt.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.two_stage_latent import TwoStageManifoldDiffusion
from models.latent_diffusion_cnn import LatentDiffusionCNN
from data.image_data import CelebAHQDataset

FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')


def load_model(model, name, device):
    path = os.path.join(CKPT_DIR, f'{name}.pt')
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model'])
    print(f"  Loaded: {path}")
    return state


def save_grid(images, nrow, ncol, title, fname, img_size=64):
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.8, nrow * 1.8))
    axes = np.atleast_2d(axes)
    idx = 0
    for i in range(nrow):
        for j in range(ncol):
            if idx < len(images):
                img = images[idx].detach().cpu().clamp(0, 1)
                axes[i, j].imshow(img.view(3, img_size, img_size).permute(1, 2, 0).numpy())
            axes[i, j].axis('off')
            idx += 1
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    img_size = 64
    image_shape = (3, img_size, img_size)
    d = 3 * img_size * img_size
    latent_dim = 256
    base_ch = 128
    hidden_dim_score = 1024
    n_score_layers = 8
    t0 = 0.005

    n_samples = 16  # paired comparison
    n_steps = 500
    n_grid = 64  # for individual grids

    # Load real images for comparison
    dataset = CelebAHQDataset(img_size=img_size, max_samples=5000)

    # Fix random seed and generate shared initial noise
    torch.manual_seed(2026)
    z_init = torch.randn(n_samples, latent_dim, device=device)

    # ================================================================
    # Shared sampling function with fixed init noise
    # ================================================================
    def sample_from_noise(model, z_start, n_steps=500):
        """DDIM sampling from fixed initial noise."""
        model.eval()
        T = 0.999
        ts = torch.linspace(T, 1e-3, n_steps + 1, device=z_start.device)
        z = z_start.clone()
        n = z.shape[0]

        for i in range(n_steps):
            t_cur, t_next = ts[i], ts[i + 1]
            t_batch = torch.full((n,), t_cur.item(), device=z.device)

            beta_min, beta_max = 0.1, 20.0
            log_mc_cur = -0.25 * t_cur**2 * (beta_max - beta_min) - 0.5 * t_cur * beta_min
            alpha_cur = torch.exp(2.0 * log_mc_cur)
            sigma_cur = torch.sqrt((1 - alpha_cur).clamp(min=1e-8))

            log_mc_next = -0.25 * t_next**2 * (beta_max - beta_min) - 0.5 * t_next * beta_min
            alpha_next = torch.exp(2.0 * log_mc_next)
            sigma_next = torch.sqrt((1 - alpha_next).clamp(min=1e-8))

            eps_pred = model.predict_eps_latent(z, t_batch)
            z0_pred = (z - sigma_cur * eps_pred) / torch.sqrt(alpha_cur).clamp(min=1e-8)
            z = torch.sqrt(alpha_next) * z0_pred + sigma_next * torch.randn_like(z)

        # Decode
        x = model.decode(z)
        return x.clamp(0, 1)

    def sample_from_noise_ldm(model, z_start, n_steps=500):
        """DDIM sampling for LDM from fixed initial noise."""
        model.eval()
        T = 0.999
        ts = torch.linspace(T, 1e-3, n_steps + 1, device=z_start.device)
        z = z_start.clone()
        n = z.shape[0]

        for i in range(n_steps):
            t_cur, t_next = ts[i], ts[i + 1]
            t_batch = torch.full((n,), t_cur.item(), device=z.device)

            alpha_cur, sigma_cur = model.noise_schedule(t_cur)
            alpha_next, sigma_next = model.noise_schedule(t_next)

            eps_pred = model.score_net(z, t_batch)
            z0_pred = (z - sigma_cur * eps_pred) / torch.sqrt(alpha_cur).clamp(min=1e-8)
            z = torch.sqrt(alpha_next) * z0_pred + sigma_next * torch.randn_like(z)

        # Decode (un-normalize)
        z_raw = z * model.z_std.clamp(min=1e-6) + model.z_mean
        x = model.decoder(z_raw)
        return x.clamp(0, 1)

    # ================================================================
    # ULD
    # ================================================================
    print("\n--- Loading ULD ---")
    model_ts = TwoStageManifoldDiffusion(
        d=d, latent_dim=latent_dim, hidden_dim_enc=512,
        hidden_dim_score=hidden_dim_score, n_layers=n_score_layers,
        t0=t0, image_shape=image_shape, base_ch=base_ch,
    ).to(device)
    load_model(model_ts, 'celebahq_gan_twostage_full', device)

    print(f"  Generating {n_samples} paired samples...")
    with torch.no_grad():
        x_gen_ts = sample_from_noise(model_ts, z_init, n_steps).cpu()

    # Also generate 64 random for grid
    print(f"  Generating {n_grid} random samples...")
    with torch.no_grad():
        x_grid_ts = model_ts.sample(n_grid, n_steps=n_steps, device=device,
                                    clamp=(0.0, 1.0)).cpu()
    save_grid(x_grid_ts, 8, 8, 'ULD (Ours) — CelebA-HQ',
              'paper_celebahq_gen_uld.png', img_size)

    del model_ts
    torch.cuda.empty_cache()

    # ================================================================
    # LDM
    # ================================================================
    print("\n--- Loading LDM ---")
    model_ldm = LatentDiffusionCNN(
        d=d, latent_dim=latent_dim, hidden_dim_enc=512,
        hidden_dim_score=hidden_dim_score, n_layers=n_score_layers,
        kl_weight=1e-4, image_shape=image_shape, base_ch=base_ch,
    ).to(device)
    load_model(model_ldm, 'celebahq_gan_ldm_full', device)

    print(f"  Generating {n_samples} paired samples (same noise)...")
    with torch.no_grad():
        x_gen_ldm = sample_from_noise_ldm(model_ldm, z_init, n_steps).cpu()

    print(f"  Generating {n_grid} random samples...")
    with torch.no_grad():
        x_grid_ldm = model_ldm.sample(n_grid, n_steps=n_steps, device=device,
                                      clamp=(0.0, 1.0)).cpu()
    save_grid(x_grid_ldm, 8, 8, 'VAE-LDM — CelebA-HQ',
              'paper_celebahq_gen_ldm.png', img_size)

    del model_ldm
    torch.cuda.empty_cache()

    # ================================================================
    # Real
    # ================================================================
    x_real, _ = dataset.sample_clean(n_grid)
    save_grid(x_real, 8, 8, 'Real — CelebA-HQ',
              'paper_celebahq_real.png', img_size)

    # ================================================================
    # Paired comparison: same noise → ULD vs LDM (2 rows x 16 cols)
    # ================================================================
    print("\n--- Paired comparison (same initial noise) ---")
    n_show = min(n_samples, 10)
    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2, 4))
    for j in range(n_show):
        axes[0, j].imshow(x_gen_ts[j].view(3, 64, 64).permute(1, 2, 0).clamp(0, 1).numpy())
        axes[0, j].axis('off')
        axes[1, j].imshow(x_gen_ldm[j].view(3, 64, 64).permute(1, 2, 0).clamp(0, 1).numpy())
        axes[1, j].axis('off')
    axes[0, 0].set_ylabel('ULD\n(Ours)', fontsize=13, fontweight='bold',
                          rotation=0, labelpad=50, va='center')
    axes[1, 0].set_ylabel('VAE-\nLDM', fontsize=13, fontweight='bold',
                          rotation=0, labelpad=50, va='center')
    plt.suptitle('Same initial noise $\\mathbf{z}_T \\sim \\mathcal{N}(0, I)$',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'paper_celebahq_paired.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ================================================================
    # Side-by-side: Real | ULD | LDM (4x4 each)
    # ================================================================
    print("\n--- Side-by-side comparison ---")
    fig, axes = plt.subplots(4, 12, figsize=(21.6, 7.2))
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            axes[i, j].imshow(x_real[idx].view(3, 64, 64).permute(1, 2, 0).clamp(0, 1).numpy())
            axes[i, j].axis('off')
            axes[i, j+4].imshow(x_grid_ts[idx].view(3, 64, 64).permute(1, 2, 0).clamp(0, 1).numpy())
            axes[i, j+4].axis('off')
            axes[i, j+8].imshow(x_grid_ldm[idx].view(3, 64, 64).permute(1, 2, 0).clamp(0, 1).numpy())
            axes[i, j+8].axis('off')

    for j, title in [(1.5, 'Real'), (5.5, 'ULD (Ours)'), (9.5, 'VAE-LDM')]:
        axes[0, int(j)].set_title(title, fontsize=14, fontweight='bold', pad=10)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'paper_celebahq_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
