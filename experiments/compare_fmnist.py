"""
Experiment 6: FashionMNIST (d=784, 28x28)

Fair comparison: Two-Stage vs LDM-CNN with identical architecture.
FashionMNIST is more complex than MNIST (clothing items with varied textures).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.image_data import FashionMNISTDataset
from models.two_stage_latent import TwoStageManifoldDiffusion
from models.latent_diffusion_cnn import LatentDiffusionCNN

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def plot_image_grid(images, nrow=10, title='', save_name=None):
    n = min(len(images), nrow * nrow)
    images = images[:n].view(n, 28, 28).detach().cpu().numpy()
    images = np.clip(images, 0, 1)

    fig, axes = plt.subplots(nrow, nrow, figsize=(nrow, nrow))
    for i in range(nrow):
        for j in range(nrow):
            idx = i * nrow + j
            if idx < n:
                axes[i, j].imshow(images[idx], cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_name:
        path = os.path.join(FIG_DIR, save_name)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")
    else:
        plt.close()


def run_experiment(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = 0.005
    latent_dim = 64
    d = 784

    print("=" * 70)
    print(f"FashionMNIST EXPERIMENT (d={d})")
    print(f"  latent_dim={latent_dim}")
    print("=" * 70)

    dataset = FashionMNISTDataset(max_samples=20000)

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
        d=d, latent_dim=latent_dim, hidden_dim_enc=512, hidden_dim_score=512,
        n_layers=6, t0=t0, image_shape=(1, 28, 28),
    )

    results_ts = model_ts.full_training(
        dataset,
        stage1_steps=2000, stage2_steps=10000,
        lr1=1e-3, lr2=3e-4, batch_size=2048,
    )

    # ================================================================
    # MODEL 2: LDM-CNN (fair comparison)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODEL 2: LDM-CNN (fair comparison)")
    print("=" * 70)

    model_ldm = LatentDiffusionCNN(
        d=d, latent_dim=latent_dim, hidden_dim_enc=512, hidden_dim_score=512,
        n_layers=6, kl_weight=1e-4, image_shape=(1, 28, 28),
    )

    results_ldm = model_ldm.full_training(
        dataset,
        ae_steps=2000, diffusion_steps=10000,
        lr_ae=1e-3, lr_diff=3e-4, batch_size=2048,
    )

    # ================================================================
    # GENERATE SAMPLES
    # ================================================================
    print("\n--- Generating Samples ---")

    with torch.no_grad():
        print("  Two-Stage: sampling...")
        x_gen_ts = model_ts.sample(100, n_steps=500, clamp=(0.0, 1.0))
        plot_image_grid(x_gen_ts, nrow=10, title='Two-Stage Generated',
                        save_name='fmnist_gen_twostage.png')

        print("  LDM-CNN: sampling...")
        x_gen_ldm = model_ldm.sample(100, n_steps=500, clamp=(0.0, 1.0))
        plot_image_grid(x_gen_ldm, nrow=10, title='LDM-CNN Generated',
                        save_name='fmnist_gen_ldm_cnn.png')

        # Real samples
        x_real, _ = dataset.sample_clean(100)
        plot_image_grid(x_real, nrow=10, title='Real FashionMNIST',
                        save_name='fmnist_real.png')

    # ================================================================
    # RECONSTRUCTION COMPARISON
    # ================================================================
    print("\n--- Reconstruction Comparison ---")
    with torch.no_grad():
        x_test, _ = dataset.sample_clean(8)
        _, x_recon_ts = model_ts.project(x_test)
        z_ldm = model_ldm.vae_encoder.encode_deterministic(x_test)
        x_recon_ldm = model_ldm.decoder(z_ldm)

        fig, axes = plt.subplots(3, 8, figsize=(16, 6))
        for j in range(8):
            for row, imgs in enumerate([x_test, x_recon_ts, x_recon_ldm]):
                img = imgs[j].view(28, 28).clamp(0, 1).numpy()
                axes[row, j].imshow(img, cmap='gray', vmin=0, vmax=1)
                axes[row, j].axis('off')

        axes[0, 0].set_ylabel('Original', fontsize=10, rotation=0, labelpad=60, va='center')
        axes[1, 0].set_ylabel('Two-Stage', fontsize=10, rotation=0, labelpad=60, va='center')
        axes[2, 0].set_ylabel('LDM-CNN', fontsize=10, rotation=0, labelpad=60, va='center')
        plt.suptitle('FashionMNIST Reconstruction Comparison', fontsize=14)
        plt.tight_layout()
        path = os.path.join(FIG_DIR, 'fmnist_reconstruction.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # ================================================================
    # SIDE-BY-SIDE GENERATION COMPARISON
    # ================================================================
    print("\n--- Side-by-side Comparison ---")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, imgs) in zip(axes, [
        ('Real', x_real[:25]),
        ('Two-Stage', x_gen_ts[:25]),
        ('LDM-CNN', x_gen_ldm[:25]),
    ]):
        imgs_np = imgs[:25].view(25, 28, 28).detach().cpu().numpy()
        imgs_np = np.clip(imgs_np, 0, 1)
        grid = np.zeros((5 * 28, 5 * 28))
        for i in range(5):
            for j in range(5):
                grid[i*28:(i+1)*28, j*28:(j+1)*28] = imgs_np[i*5+j]
        ax.imshow(grid, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.suptitle('FashionMNIST: Real vs Two-Stage vs LDM-CNN', fontsize=14)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fmnist_generation_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'=' * 70}")
    print("FashionMNIST SUMMARY")
    print("=" * 70)
    print(f"  Intrinsic dim (95% PCA): k={k_95}")
    print(f"\n  {'Model':<30} {'Latent':<8} {'Eps Loss':<12} {'Recon MSE':<12}")
    print("  " + "-" * 62)
    print(f"  {'Two-Stage (ours)':<30} {latent_dim:<8} "
          f"{results_ts['total_loss']:<12.6f} {results_ts['recon_error']:<12.6f}")
    print(f"  {'LDM-CNN (fair)':<30} {latent_dim:<8} "
          f"{results_ldm['total_loss']:<12.6f} {results_ldm['recon_error']:<12.6f}")


if __name__ == '__main__':
    run_experiment()
