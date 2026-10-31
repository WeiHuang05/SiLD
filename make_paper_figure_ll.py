"""
Paper figure: CelebA-HQ Large Latent reconstruction with noise.
4 rows: Original, Noisy(σ=0.5), LDM recon, ULD recon
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import the models from the large latent experiment
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))
from compare_celebahq_large_latent import (
    TwoStageLargeLatent, LDMLargeLatent,
)
from data.image_data import CelebAHQDataset

FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_size = 64
    image_shape = (3, img_size, img_size)
    d = 3 * img_size * img_size
    latent_ch = 8
    base_ch = 128
    unet_ch = 128

    # Load models
    m_ts = TwoStageLargeLatent(d, image_shape, latent_ch, base_ch, unet_ch).to(device)
    m_ldm = LDMLargeLatent(d, image_shape, latent_ch, base_ch, unet_ch).to(device)

    ts_path = os.path.join(CKPT_DIR, 'celebahq_ll_ts_s1.pt')
    ldm_path = os.path.join(CKPT_DIR, 'celebahq_ll_ldm_ae.pt')

    if os.path.exists(ts_path):
        state = torch.load(ts_path, map_location=device)
        m_ts.load_state_dict(state['model'])
        print("Loaded Two-Stage LL checkpoint")
    else:
        print(f"Not found: {ts_path}")
        return

    if os.path.exists(ldm_path):
        state = torch.load(ldm_path, map_location=device)
        m_ldm.load_state_dict(state['model'])
        print("Loaded LDM LL checkpoint")
    else:
        print(f"Not found: {ldm_path}")
        return

    # Load data
    dataset = CelebAHQDataset(img_size=img_size, max_samples=20000)

    sigma = 0.1
    n = 10

    with torch.no_grad():
        x_test, _ = dataset.sample_clean(n)
        x_d = x_test.to(device)
        noise = torch.randn_like(x_d)
        x_noisy = x_d + sigma * noise

        # LDM: encode noisy → decode
        z_ldm = m_ldm.vae_encoder.encode_deterministic(x_noisy)
        xr_ldm = m_ldm.decoder(z_ldm).cpu()

        # ULD: encode noisy → decode
        _, xr_ts = m_ts.project(x_noisy)
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

    plt.suptitle('CelebA-HQ: Denoising (32×32×8 latent, rFID: ULD=0.82, LDM=0.31)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'paper_celebahq_ll_denoise.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")

    # Also make a clean reconstruction (no noise) version
    with torch.no_grad():
        x_test2, _ = dataset.sample_clean(n)
        x_d2 = x_test2.to(device)
        _, xr_ts2 = m_ts.project(x_d2)
        z_ldm2 = m_ldm.vae_encoder.encode_deterministic(x_d2)
        xr_ldm2 = m_ldm.decoder(z_ldm2).cpu()
        xr_ts2 = xr_ts2.cpu()

    fig, axes = plt.subplots(3, n, figsize=(n * 1.8, 5.4))
    row_labels2 = ['Original', 'ULD (rFID=0.82)', 'LDM (rFID=0.31)']
    row_data2 = [x_test2, xr_ts2, xr_ldm2]

    for j in range(n):
        for row, imgs in enumerate(row_data2):
            img = imgs[j].view(3, img_size, img_size).permute(1, 2, 0).clamp(0, 1).numpy()
            axes[row, j].imshow(img)
            axes[row, j].axis('off')

    for row, label in enumerate(row_labels2):
        axes[row, 0].set_ylabel(label, fontsize=12, fontweight='bold',
                                rotation=0, labelpad=85, va='center')

    plt.suptitle('CelebA-HQ: Clean Reconstruction (32×32×8 latent)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'paper_celebahq_ll_recon.png')
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    run()
