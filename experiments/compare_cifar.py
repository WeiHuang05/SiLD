"""
Experiment 4: CIFAR-10 (d=3072, 32x32x3)

Fair comparison: Two-Stage vs LDM-CNN with identical architecture.
Only difference: how autoencoder is trained.
Uses LPIPS perceptual loss for better visual quality.
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

from data.image_data import CIFAR10Dataset
from models.two_stage_latent import TwoStageManifoldDiffusion
from models.latent_diffusion_cnn import LatentDiffusionCNN

# LPIPS for perceptual loss
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available, using MSE only")

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


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


def plot_image_grid(images, nrow=8, title='', save_name=None, img_size=32):
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


def train_stage1_with_lpips(model, dataset, n_steps=10000, lr=1e-4,
                            batch_size=64, lpips_weight=0.5, device='cpu'):
    """Train Stage 1 with MSE + LPIPS perceptual loss."""
    img_size = model.image_shape[1]
    sigma_t0 = model.t0

    loss_fn_lpips = None
    if LPIPS_AVAILABLE:
        loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
        loss_fn_lpips.eval()
        for p in loss_fn_lpips.parameters():
            p.requires_grad = False
        print(f"  Using LPIPS perceptual loss (weight={lpips_weight})")

    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    print(f"  Stage 1: t0={model.t0}, sigma_t0={sigma_t0:.4f}, "
          f"latent_dim={model.latent_dim}")
    print(f"  Training {n_steps} steps, lr={lr}, batch_size={batch_size}")

    losses = []
    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)
        x_t = x0 + sigma_t0 * torch.randn_like(x0)
        z, x_hat = model.project(x_t)

        mse_loss = F.mse_loss(x_hat, x0)

        if loss_fn_lpips is not None:
            x0_img = x0.view(-1, 3, img_size, img_size) * 2 - 1
            xh_img = x_hat.view(-1, 3, img_size, img_size) * 2 - 1
            lpips_loss = loss_fn_lpips(xh_img, x0_img).mean()
            loss = mse_loss + lpips_weight * lpips_loss
        else:
            lpips_loss = torch.tensor(0.0)
            loss = mse_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if (step + 1) % 200 == 0:
            print(f"  Stage 1 [{step+1}/{n_steps}] "
                  f"total={loss.item():.6f} mse={mse_loss.item():.6f} "
                  f"lpips={lpips_loss.item():.6f}")

    # Compute latent normalization statistics
    with torch.no_grad():
        zs = []
        for _ in range(10):
            x0, _ = dataset.sample_clean(min(batch_size, 2000))
            x0 = x0.to(device)
            zs.append(model.encoder(x0))
        zs = torch.cat(zs, dim=0)
        model.z_mean = zs.mean(dim=0)
        model.z_std = zs.std(dim=0).clamp(min=1e-6)

    return losses


def train_vae_with_lpips(model, dataset, n_steps=10000, lr=1e-4,
                         batch_size=64, lpips_weight=0.5, kl_weight=1e-4,
                         device='cpu'):
    """Train VAE (LDM encoder) with MSE + LPIPS + KL."""
    img_size = int(np.sqrt(model.d // 3))

    loss_fn_lpips = None
    if LPIPS_AVAILABLE:
        loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
        loss_fn_lpips.eval()
        for p in loss_fn_lpips.parameters():
            p.requires_grad = False
        print(f"  Using LPIPS perceptual loss (weight={lpips_weight})")

    params = (list(model.vae_encoder.parameters()) +
              list(model.decoder.parameters()))
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    print(f"  VAE training: {n_steps} steps, lr={lr}, batch_size={batch_size}")

    losses = []
    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)
        mu, logvar = model.vae_encoder(x0)

        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        x_recon = model.decoder(z)

        mse_loss = F.mse_loss(x_recon, x0)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        if loss_fn_lpips is not None:
            x0_img = x0.view(-1, 3, img_size, img_size) * 2 - 1
            xr_img = x_recon.view(-1, 3, img_size, img_size) * 2 - 1
            lpips_loss = loss_fn_lpips(xr_img, x0_img).mean()
            loss = mse_loss + lpips_weight * lpips_loss + kl_weight * kl_loss
        else:
            lpips_loss = torch.tensor(0.0)
            loss = mse_loss + kl_weight * kl_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if (step + 1) % 200 == 0:
            print(f"  VAE [{step+1}/{n_steps}] "
                  f"total={loss.item():.6f} mse={mse_loss.item():.6f} "
                  f"lpips={lpips_loss.item():.6f} kl={kl_loss.item():.4f}")

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

    return losses


def run_experiment(class_idx=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    img_size = 32
    image_shape = (3, img_size, img_size)
    d = 3 * img_size * img_size  # 3072
    latent_dim = 128
    t0 = 0.005

    # Network params — larger networks
    base_ch = 128                # was 64
    hidden_dim_score = 1024      # was 512
    n_score_layers = 8           # was 6

    # Training params — 10x longer with LPIPS
    ae_steps = 30000             # was 3000
    diff_steps = 150000          # was 15000
    batch_size = 64
    lpips_weight = 0.5

    print("=" * 70)
    print(f"CIFAR-10 EXPERIMENT (d={d}, {img_size}x{img_size}x3)")
    print(f"  class={class_idx if class_idx is not None else 'all'}")
    print(f"  latent_dim={latent_dim}")
    print(f"  base_ch={base_ch}, score_hidden={hidden_dim_score}, score_layers={n_score_layers}")
    print(f"  AE steps={ae_steps}, Diffusion steps={diff_steps}")
    print(f"  LPIPS weight={lpips_weight}")
    print("=" * 70)

    dataset = CIFAR10Dataset(class_idx=class_idx, max_samples=20000)

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
        d=d, latent_dim=latent_dim, hidden_dim_enc=512, hidden_dim_score=hidden_dim_score,
        n_layers=n_score_layers, t0=t0, image_shape=image_shape, base_ch=base_ch,
    ).to(device)

    # Try to resume from checkpoint
    ckpt_ts_full = load_checkpoint(model_ts, 'cifar_twostage_full', device=device)
    if ckpt_ts_full is not None:
        print("  Resumed Two-Stage (full) from checkpoint.")
        model_ts.freeze_stage1()
        recon_mse_ts = ckpt_ts_full.get('recon_mse', 0.0)
        avg_loss_ts = ckpt_ts_full.get('avg_loss', 0.0)
    else:
        ckpt_ts_s1 = load_checkpoint(model_ts, 'cifar_twostage_stage1', device=device)
        if ckpt_ts_s1 is not None:
            print("  Resumed Two-Stage Stage 1 from checkpoint.")
            model_ts.freeze_stage1()
            recon_mse_ts = ckpt_ts_s1.get('recon_mse', 0.0)
        else:
            # Stage 1 with LPIPS
            print("\n--- Stage 1: Manifold Learning with LPIPS ---")
            stage1_losses = train_stage1_with_lpips(
                model_ts, dataset, n_steps=ae_steps, lr=1e-4,
                batch_size=batch_size, lpips_weight=lpips_weight, device=device,
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
            save_checkpoint(model_ts, 'cifar_twostage_stage1',
                            {'recon_mse': recon_mse_ts})

        # Stage 2: latent diffusion
        print("\n--- Stage 2: Latent Diffusion ---")
        stage2_losses = model_ts.train_stage2(
            dataset, n_steps=diff_steps, lr=3e-4, batch_size=batch_size,
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

        save_checkpoint(model_ts, 'cifar_twostage_full',
                        {'recon_mse': recon_mse_ts, 'avg_loss': avg_loss_ts})

    # ================================================================
    # MODEL 2: LDM-CNN (fair comparison)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODEL 2: LDM-CNN (fair comparison)")
    print("=" * 70)

    model_ldm = LatentDiffusionCNN(
        d=d, latent_dim=latent_dim, hidden_dim_enc=512, hidden_dim_score=hidden_dim_score,
        n_layers=n_score_layers, kl_weight=1e-4, image_shape=image_shape, base_ch=base_ch,
    ).to(device)

    # Try to resume from checkpoint
    ckpt_ldm_full = load_checkpoint(model_ldm, 'cifar_ldm_full', device=device)
    if ckpt_ldm_full is not None:
        print("  Resumed LDM (full) from checkpoint.")
        model_ldm.freeze_ae()
        recon_mse_ldm = ckpt_ldm_full.get('recon_mse', 0.0)
        avg_loss_ldm = ckpt_ldm_full.get('avg_loss', 0.0)
    else:
        ckpt_ldm_ae = load_checkpoint(model_ldm, 'cifar_ldm_ae', device=device)
        if ckpt_ldm_ae is not None:
            print("  Resumed LDM AE from checkpoint.")
            model_ldm.freeze_ae()
            recon_mse_ldm = ckpt_ldm_ae.get('recon_mse', 0.0)
        else:
            # VAE training with LPIPS
            print("\n--- VAE Training with LPIPS ---")
            vae_losses = train_vae_with_lpips(
                model_ldm, dataset, n_steps=ae_steps, lr=1e-4,
                batch_size=batch_size, lpips_weight=lpips_weight,
                kl_weight=1e-4, device=device,
            )

            with torch.no_grad():
                x0_test, _ = dataset.sample_clean(2000)
                x0_test = x0_test.to(device)
                z_ldm = model_ldm.vae_encoder.encode_deterministic(x0_test)
                x_recon = model_ldm.decoder(z_ldm)
                recon_mse_ldm = F.mse_loss(x_recon, x0_test).item()
                print(f"\n  VAE Reconstruction MSE: {recon_mse_ldm:.6f}")

            model_ldm.freeze_ae()
            save_checkpoint(model_ldm, 'cifar_ldm_ae',
                            {'recon_mse': recon_mse_ldm})

        # Diffusion training
        print("\n--- Diffusion Training ---")
        diff_losses = model_ldm.train_diffusion(
            dataset, n_steps=diff_steps, lr=3e-4, batch_size=batch_size,
        )

        # Evaluate LDM
        eval_times = [0.01, 0.1, 0.3, 0.5, 0.8]
        total_loss_ldm = 0.0
        for t_eval in eval_times:
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                z0 = (model_ldm.vae_encoder.encode_deterministic(x0) - model_ldm.z_mean) / model_ldm.z_std.clamp(min=1e-6)
                t_batch = torch.full((2000,), t_eval, device=device)
                alpha_bar, sigma = model_ldm.noise_schedule(t_batch)
                eps_z = torch.randn_like(z0)
                z_t = torch.sqrt(alpha_bar.unsqueeze(-1)) * z0 + sigma.unsqueeze(-1) * eps_z
                eps_z_pred = model_ldm.score_net(z_t, t_batch)
                mse = F.mse_loss(eps_z_pred, eps_z).item()
                total_loss_ldm += mse
                print(f"  t={t_eval:.2f}: latent_eps_MSE={mse:.6f}")
        avg_loss_ldm = total_loss_ldm / len(eval_times)

        save_checkpoint(model_ldm, 'cifar_ldm_full',
                        {'recon_mse': recon_mse_ldm, 'avg_loss': avg_loss_ldm})

    results_ts = {'total_loss': avg_loss_ts, 'recon_error': recon_mse_ts}
    results_ldm = {'total_loss': avg_loss_ldm, 'recon_error': recon_mse_ldm}

    # ================================================================
    # GENERATE SAMPLES
    # ================================================================
    print("\n--- Generating Samples ---")

    with torch.no_grad():
        print("  Two-Stage: sampling...")
        x_gen_ts = model_ts.sample(64, n_steps=500, device=device, clamp=(0.0, 1.0)).cpu()
        plot_image_grid(x_gen_ts, nrow=8, title='Two-Stage Generated',
                        save_name='cifar_gen_twostage.png', img_size=img_size)

        print("  LDM-CNN: sampling...")
        x_gen_ldm = model_ldm.sample(64, n_steps=500, device=device, clamp=(0.0, 1.0)).cpu()
        plot_image_grid(x_gen_ldm, nrow=8, title='LDM-CNN Generated',
                        save_name='cifar_gen_ldm_cnn.png', img_size=img_size)

        x_real, _ = dataset.sample_clean(64)
        plot_image_grid(x_real, nrow=8, title='Real CIFAR-10',
                        save_name='cifar_real.png', img_size=img_size)

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

        fig, axes = plt.subplots(3, 8, figsize=(16, 6))
        for j in range(8):
            for row, img in enumerate([x_test[j], x_recon_ts[j], x_recon_ldm[j]]):
                img_np = img.view(3, img_size, img_size).permute(1, 2, 0).clamp(0, 1).numpy()
                axes[row, j].imshow(img_np)
                axes[row, j].axis('off')

        axes[0, 0].set_ylabel('Original', fontsize=10, rotation=0, labelpad=60, va='center')
        axes[1, 0].set_ylabel('Two-Stage', fontsize=10, rotation=0, labelpad=60, va='center')
        axes[2, 0].set_ylabel('LDM-CNN', fontsize=10, rotation=0, labelpad=60, va='center')
        plt.suptitle('CIFAR-10 Reconstruction Comparison', fontsize=14)
        plt.tight_layout()
        path = os.path.join(FIG_DIR, 'cifar_reconstruction.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # ================================================================
    # SIDE-BY-SIDE GENERATION COMPARISON
    # ================================================================
    print("\n--- Side-by-side Comparison ---")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (title, imgs) in zip(axes, [
        ('Real', x_real[:16]),
        ('Two-Stage', x_gen_ts[:16]),
        ('LDM-CNN', x_gen_ldm[:16]),
    ]):
        imgs_np = imgs.view(-1, 3, img_size, img_size).permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()
        grid = np.zeros((4 * img_size, 4 * img_size, 3))
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                if idx < len(imgs_np):
                    grid[i*img_size:(i+1)*img_size, j*img_size:(j+1)*img_size] = imgs_np[idx]
        ax.imshow(grid)
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    plt.suptitle('CIFAR-10: Real vs Two-Stage vs LDM-CNN', fontsize=16)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'cifar_generation_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'=' * 70}")
    print("CIFAR-10 SUMMARY")
    print("=" * 70)
    print(f"  Intrinsic dim (95% PCA): k={k_95}")
    print(f"\n  {'Model':<30} {'Latent':<8} {'Eps Loss':<12} {'Recon MSE':<12}")
    print("  " + "-" * 62)
    print(f"  {'Two-Stage (ours)':<30} {latent_dim:<8} "
          f"{results_ts['total_loss']:<12.6f} {results_ts['recon_error']:<12.6f}")
    print(f"  {'LDM-CNN (fair)':<30} {latent_dim:<8} "
          f"{results_ldm['total_loss']:<12.6f} {results_ldm['recon_error']:<12.6f}")


if __name__ == '__main__':
    run_experiment(class_idx=None)
