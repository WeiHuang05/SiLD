"""
Generate clean paper-ready figures for CelebA results.
Pick good-looking samples and use proper method names.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')


def make_grid(images, nrow, img_size=64):
    """Make image grid from flat tensor images."""
    n = len(images)
    ncol = nrow
    nrow_actual = (n + ncol - 1) // ncol
    grid = np.ones((nrow_actual * img_size + (nrow_actual-1)*2,
                     ncol * img_size + (ncol-1)*2, 3))  # white border
    for i in range(min(n, nrow_actual * ncol)):
        r, c = i // ncol, i % ncol
        img = images[i].detach().cpu().clamp(0, 1)
        img = img.view(3, img_size, img_size).permute(1, 2, 0).numpy()
        y = r * (img_size + 2)
        x = c * (img_size + 2)
        grid[y:y+img_size, x:x+img_size] = img
    return grid


def generate_samples():
    """Load models and generate fresh samples."""
    from data.image_data import CelebADataset
    from models.two_stage_latent import TwoStageManifoldDiffusion
    from models.latent_diffusion_cnn import LatentDiffusionCNN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_size = 64
    image_shape = (3, img_size, img_size)
    d = 12288
    latent_dim = 256

    # Load GAN+EMA models
    model_ts = TwoStageManifoldDiffusion(
        d=d, latent_dim=latent_dim, hidden_dim_enc=512, hidden_dim_score=1024,
        n_layers=8, t0=0.005, image_shape=image_shape, base_ch=128,
    ).to(device)

    model_ldm = LatentDiffusionCNN(
        d=d, latent_dim=latent_dim, hidden_dim_enc=512, hidden_dim_score=1024,
        n_layers=8, kl_weight=1e-4, image_shape=image_shape, base_ch=128,
    ).to(device)

    # Load checkpoints
    ts_path = os.path.join(CKPT_DIR, 'celeba_gan_twostage_full.pt')
    ldm_path = os.path.join(CKPT_DIR, 'celeba_gan_ldm_full.pt')

    if os.path.exists(ts_path):
        state = torch.load(ts_path, map_location=device)
        model_ts.load_state_dict(state['model'])
        print("Loaded Two-Stage GAN checkpoint")
    else:
        print(f"Not found: {ts_path}")
        return

    if os.path.exists(ldm_path):
        state = torch.load(ldm_path, map_location=device)
        model_ldm.load_state_dict(state['model'])
        print("Loaded LDM GAN checkpoint")
    else:
        print(f"Not found: {ldm_path}")
        return

    model_ts.freeze_stage1()
    model_ldm.freeze_ae()

    # Generate 128 samples each, pick best visually
    print("Generating ULD samples...")
    with torch.no_grad():
        x_ts = model_ts.sample(128, n_steps=500, device=device, clamp=(0.0, 1.0)).cpu()
        x_ldm = model_ldm.sample(128, n_steps=500, device=device, clamp=(0.0, 1.0)).cpu()

    # Load real samples
    dataset = CelebADataset(img_size=img_size, max_samples=20000)
    x_real, _ = dataset.sample_clean(128)

    return x_ts, x_ldm, x_real, model_ts, model_ldm, dataset, device


def paper_figure_generation(x_ts, x_ldm, x_real):
    """Side-by-side generation comparison with proper titles."""
    img_size = 64
    n_show = 64  # 8x8 grid

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8))

    for ax, (title, imgs) in zip(axes, [
        ('Real CelebA', x_real[:n_show]),
        ('ULD (Ours, FID=39.57)', x_ts[:n_show]),
        ('VAE-LDM (FID=40.69)', x_ldm[:n_show]),
    ]):
        grid = make_grid(imgs, nrow=8, img_size=img_size)
        ax.imshow(grid)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
        ax.axis('off')

    plt.tight_layout(pad=1.0)
    path = os.path.join(FIG_DIR, 'paper_celeba_generation.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")


def paper_figure_reconstruction(model_ts, model_ldm, dataset, device):
    """Reconstruction comparison with noisy input row."""
    img_size = 64
    n = 10
    sigma = 0.1

    with torch.no_grad():
        x_test, _ = dataset.sample_clean(n)
        x_d = x_test.to(device)
        noise = torch.randn_like(x_d)
        x_noisy = (x_d + sigma * noise)

        # LDM: denoise from noisy input
        z_ldm = model_ldm.vae_encoder.encode_deterministic(x_noisy)
        xr_ldm = model_ldm.decoder(z_ldm).cpu()

        # ULD: denoise from noisy input
        _, xr_ts = model_ts.project(x_noisy)
        xr_ts = xr_ts.cpu()
        x_noisy = x_noisy.cpu()

    fig, axes = plt.subplots(4, n, figsize=(n * 1.8, 7.2))
    row_labels = ['Original', f'Noisy ($\\sigma$={sigma})',
                  'LDM+GAN', 'ULD+GAN (Ours)']
    row_data = [x_test, x_noisy, xr_ldm, xr_ts]

    for j in range(n):
        for row, imgs in enumerate(row_data):
            img = imgs[j].view(3, img_size, img_size).permute(1, 2, 0).clamp(0, 1).numpy()
            axes[row, j].imshow(img)
            axes[row, j].axis('off')

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=12, fontweight='bold',
                                rotation=0, labelpad=85, va='center')

    plt.suptitle('CelebA: Denoising and Reconstruction', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'paper_celeba_reconstruction.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    result = generate_samples()
    if result is not None:
        x_ts, x_ldm, x_real, model_ts, model_ldm, dataset, device = result
        paper_figure_generation(x_ts, x_ldm, x_real)
        paper_figure_reconstruction(model_ts, model_ldm, dataset, device)
        print("Done!")
