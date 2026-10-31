"""
Experiment 3: Image Dataset (MNIST)

Compares Two-Stage Manifold Diffusion vs Latent Diffusion on real images.
MNIST: d=784 ambient, estimated intrinsic dim ~10-20.

Evaluates:
  1. Dimensional collapse: does W W^T spectrum reveal intrinsic dim?
  2. Score estimation quality at different noise levels
  3. LDM with varying latent dims vs Two-Stage
  4. Generated sample quality (visual + MSE)
  5. Optimal stopping for LDM
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

from data.image_data import MNISTDataset
from models.two_stage_latent import TwoStageManifoldDiffusion
from models.latent_diffusion import LatentDiffusionModel
from models.latent_diffusion_cnn import LatentDiffusionCNN

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def plot_image_grid(images, nrow=10, title='', save_name='samples.png'):
    """Plot a grid of 28x28 images."""
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
    path = os.path.join(FIG_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_denoising_comparison(originals, noisy, denoised_ts, denoised_ldm,
                               sigma_t, save_name='denoising_comparison.png'):
    """Compare denoising from Two-Stage vs LDM."""
    n_show = 8
    fig, axes = plt.subplots(4, n_show, figsize=(n_show * 1.5, 6))

    rows = [
        ('Original', originals),
        (f'Noisy (σ={sigma_t:.2f})', noisy),
        ('Two-Stage', denoised_ts),
        ('LDM', denoised_ldm),
    ]

    for row_idx, (label, imgs) in enumerate(rows):
        for col_idx in range(n_show):
            img = imgs[col_idx].view(28, 28).detach().cpu().numpy()
            img = np.clip(img, 0, 1)
            axes[row_idx, col_idx].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[row_idx, col_idx].axis('off')
        axes[row_idx, 0].set_ylabel(label, fontsize=10, rotation=0,
                                     labelpad=60, va='center')

    plt.suptitle('Denoising Comparison: Two-Stage vs LDM', fontsize=14)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def run_experiment(digit=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = 0.005      # Stage 1: small noise for geometric collapse

    print("=" * 70)
    print(f"IMAGE EXPERIMENT: MNIST (d=784)")
    print(f"  digit={digit if digit is not None else 'all'}")
    print(f"  t0={t0}, Stage 2: latent diffusion in R^k for ALL t")
    print("=" * 70)

    dataset = MNISTDataset(digit=digit, max_samples=20000)
    d = dataset.d  # 784

    # ================================================================
    # PCA ANALYSIS: estimate intrinsic dimension
    # ================================================================
    print("\n--- PCA Analysis ---")
    explained_var = dataset.pca_explained_variance
    cumvar = torch.cumsum(explained_var, dim=0) / explained_var.sum()

    # Find k for 90%, 95%, 99% variance explained
    for threshold in [0.90, 0.95, 0.99]:
        k_est = (cumvar < threshold).sum().item() + 1
        print(f"  {threshold*100:.0f}% variance explained at k={k_est}")

    k_95 = (cumvar < 0.95).sum().item() + 1

    # Plot PCA spectrum
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    n_show_eig = min(100, len(explained_var))
    ax.semilogy(range(n_show_eig), explained_var[:n_show_eig].numpy(),
                'o-', markersize=2, color='steelblue')
    ax.axvline(x=k_95, color='red', linestyle='--', label=f'95% var (k={k_95})')
    ax.set_xlabel('Component Index')
    ax.set_ylabel('Explained Variance (log)')
    ax.set_title('MNIST PCA Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(range(min(100, len(cumvar))), cumvar[:100].numpy(), '-', color='steelblue')
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='95%')
    ax.axvline(x=k_95, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance Ratio')
    ax.set_title('Cumulative Explained Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'mnist_pca_spectrum.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ================================================================
    # TWO-STAGE MANIFOLD DIFFUSION
    # ================================================================
    print("\n" + "=" * 70)
    print("MODEL 1: Two-Stage Manifold Diffusion")
    print("=" * 70)

    model_ts = TwoStageManifoldDiffusion(
        d=d, latent_dim=64, hidden_dim_enc=512, hidden_dim_score=512,
        n_layers=6, t0=t0, image_shape=(1, 28, 28)
    )

    results_ts = model_ts.full_training(
        dataset,
        stage1_steps=2000, stage2_steps=10000,
        lr1=1e-3, lr2=3e-4, batch_size=2048
    )

    # Plot WW^T eigenvalue spectrum
    fig, ax = plt.subplots(figsize=(8, 5))
    eigs = results_ts['eigenvalues']
    n_plot = min(100, len(eigs))
    ax.semilogy(range(n_plot), np.maximum(eigs[:n_plot], 1e-10),
                'o-', markersize=3, color='steelblue', label='Latent variance')

    # Overlay PCA spectrum (normalized to same scale)
    pca_eigs = explained_var[:n_plot].numpy()
    pca_eigs_norm = pca_eigs / pca_eigs[0] * eigs[0]
    ax.semilogy(range(n_plot), pca_eigs_norm, '--', color='coral',
                alpha=0.7, label='PCA spectrum (normalized)')

    ax.axvline(x=k_95, color='red', linestyle=':', alpha=0.5,
               label=f'PCA 95% dim (k={k_95})')
    ax.set_xlabel('Index')
    ax.set_ylabel('Eigenvalue (log)')
    ax.set_title('MNIST: Dimensional Collapse vs PCA Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'mnist_dimensional_collapse.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ================================================================
    # LATENT DIFFUSION MODELS
    # ================================================================
    latent_dims = [10, 20, 50, 100]
    results_ldm_list = []

    for d_latent in latent_dims:
        print(f"\n{'=' * 70}")
        print(f"MODEL 2: LDM (d_latent={d_latent})")
        print("=" * 70)

        proj_matrix = dataset.get_pca_projection(d_latent)

        model_ldm = LatentDiffusionModel(
            d=d, d_latent=d_latent, hidden_dim=256, n_layers=3,
            projection_matrix=proj_matrix
        )

        losses = model_ldm.train_model(
            dataset, n_steps=3000, lr=1e-3, batch_size=2048
        )

        # Evaluate reconstruction and generation
        with torch.no_grad():
            x_true, _ = dataset.sample_clean(1000)
            x_recon = model_ldm.autoencoder.decode(model_ldm.autoencoder.encode(x_true))
            recon_mse = F.mse_loss(x_recon, x_true).item()

            x_gen = model_ldm.generate(1000, n_steps=200, t_stop=0.01)
            gen_mse = F.mse_loss(x_gen, x_true).item()

        results_ldm_list.append((f'd_lat={d_latent}', {
            'losses': losses,
            'final_score_mse': losses[-1],
            'recon_mse': recon_mse,
            'generation_mse': gen_mse,
            'model': model_ldm,
        }))
        print(f"  Final loss: {losses[-1]:.6f}, Recon MSE: {recon_mse:.6f}, "
              f"Gen MSE: {gen_mse:.6f}")

    # ================================================================
    # FAIR COMPARISON: LDM-CNN (same architecture as Two-Stage)
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"MODEL 3: LDM-CNN (fair comparison — same arch, VAE training)")
    print("=" * 70)

    model_ldm_cnn = LatentDiffusionCNN(
        d=d, latent_dim=64, hidden_dim_enc=512, hidden_dim_score=512,
        n_layers=6, kl_weight=1e-4, image_shape=(1, 28, 28)
    )

    results_ldm_cnn = model_ldm_cnn.full_training(
        dataset,
        ae_steps=2000, diffusion_steps=10000,
        lr_ae=1e-3, lr_diff=3e-4, batch_size=2048
    )

    # Generate samples
    with torch.no_grad():
        x_gen_ldm_cnn = model_ldm_cnn.sample(100, n_steps=500, clamp=(0.0, 1.0))
    plot_image_grid(x_gen_ldm_cnn, nrow=10,
                    title='LDM-CNN Generated (fair comparison)',
                    save_name='mnist_gen_ldm_cnn.png')

    # ================================================================
    # DENOISING COMPARISON
    # ================================================================
    print("\n--- Denoising Comparison ---")
    sigma_denoise = 0.15
    with torch.no_grad():
        x_orig, _ = dataset.sample_clean(8)
        eps = torch.randn_like(x_orig)
        x_noisy = x_orig + sigma_denoise * eps

        # Two-Stage denoising via Tweedie's formula
        x_denoised_ts, info_ts = model_ts.denoise(x_noisy, sigma_denoise)

        # LDM denoising (use best model)
        best_ldm = results_ldm_list[-1][1]['model']  # highest latent dim
        z_noisy = best_ldm.autoencoder.encode(x_noisy)
        x_denoised_ldm = best_ldm.autoencoder.decode(z_noisy)  # just projection

    plot_denoising_comparison(
        x_orig, x_noisy, x_denoised_ts, x_denoised_ldm,
        sigma_denoise, save_name='mnist_denoising_comparison.png'
    )

    # ================================================================
    # GENERATED SAMPLES
    # ================================================================
    print("\n--- Generating Samples ---")

    # Two-Stage: generate via reverse SDE (start from t_max)
    print("  Two-Stage: reverse SDE sampling...")
    x_gen_ts_sde = model_ts.sample(
        100, n_steps=500, clamp=(0.0, 1.0)
    )
    plot_image_grid(x_gen_ts_sde, nrow=10,
                   title='Two-Stage Generated (Reverse SDE)',
                   save_name='mnist_gen_twostage_sde.png')

    # Two-Stage: generate via Predictor-Corrector
    print("  Two-Stage: predictor-corrector sampling...")
    x_gen_ts_pc = model_ts.sample_predictor_corrector(
        100, n_steps=300, n_corrector=3, snr=0.05, clamp=(0.0, 1.0)
    )
    plot_image_grid(x_gen_ts_pc, nrow=10,
                   title='Two-Stage Generated (Predictor-Corrector)',
                   save_name='mnist_gen_twostage_pc.png')

    # Two-Stage: progressive denoising visualization
    # Shows the two-stage decomposition: manifold projection + density refinement
    print("  Two-Stage: progressive denoising visualization...")
    with torch.no_grad():
        x_clean, _ = dataset.sample_clean(8)
        noise_levels = [0.5, 0.3, 0.2, 0.1, 0.05]
        fig, axes = plt.subplots(len(noise_levels) + 2, 8, figsize=(16, (len(noise_levels)+2)*2))

        # Row 0: original
        for j in range(8):
            axes[0, j].imshow(x_clean[j].view(28, 28).numpy(), cmap='gray', vmin=0, vmax=1)
            axes[0, j].axis('off')
        axes[0, 0].set_ylabel('Original', fontsize=9, rotation=0, labelpad=50, va='center')

        # Rows 1..N: noisy -> denoised at each sigma
        for row_idx, sigma in enumerate(noise_levels):
            eps = torch.randn_like(x_clean)
            x_noisy = x_clean + sigma * eps
            x_denoised, info = model_ts.denoise(x_noisy, sigma)

            for j in range(8):
                img = torch.clamp(x_denoised[j], 0, 1).view(28, 28).numpy()
                axes[row_idx + 1, j].imshow(img, cmap='gray', vmin=0, vmax=1)
                axes[row_idx + 1, j].axis('off')
            axes[row_idx + 1, 0].set_ylabel(f'σ={sigma}', fontsize=9,
                                             rotation=0, labelpad=50, va='center')

        # Last row: Stage 1 projection only (geometry without density)
        eps = torch.randn_like(x_clean)
        x_noisy = x_clean + 0.1 * eps
        _, info = model_ts.forward(x_noisy, torch.tensor(0.1))
        for j in range(8):
            img = torch.clamp(info['x_hat'][j], 0, 1).view(28, 28).numpy()
            axes[-1, j].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[-1, j].axis('off')
        axes[-1, 0].set_ylabel('Stage1\nProjection', fontsize=9,
                                rotation=0, labelpad=50, va='center')

        plt.suptitle('Two-Stage Progressive Denoising\n'
                     '(Top: original, Middle: denoised at decreasing σ, '
                     'Bottom: Stage 1 manifold projection only)', fontsize=12)
        plt.tight_layout()
        path = os.path.join(FIG_DIR, 'mnist_progressive_denoising.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # LDM: generate for each latent dim
    for label, res in results_ldm_list:
        with torch.no_grad():
            x_gen = res['model'].generate(100, n_steps=200, t_stop=0.01)
        plot_image_grid(x_gen, nrow=10,
                       title=f'LDM Generated ({label})',
                       save_name=f'mnist_gen_ldm_{label}.png')

    # Real samples for reference
    x_real, _ = dataset.sample_clean(100)
    plot_image_grid(x_real, nrow=10, title='Real MNIST',
                   save_name='mnist_real.png')

    # Side-by-side comparison: Real vs Two-Stage denoising vs LDM generation
    print("  Creating side-by-side comparison...")
    with torch.no_grad():
        # Two-Stage: denoise from moderate noise (shows the model works)
        x_base, _ = dataset.sample_clean(25)
        x_noisy_vis = x_base + 0.3 * torch.randn_like(x_base)
        x_ts_denoised, _ = model_ts.denoise(x_noisy_vis, 0.3)
        x_ts_denoised = torch.clamp(x_ts_denoised, 0, 1)

        # LDM generation
        x_gen_ldm_best = results_ldm_list[1][1]['model'].generate(
            25, n_steps=200, t_stop=0.01
        )

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    panel_data = [
        ('Real', x_base),
        ('Noisy (σ=0.3)', torch.clamp(x_noisy_vis, 0, 1)),
        ('Two-Stage\nDenoised', x_ts_denoised),
        ('Two-Stage\nGenerated (SDE)', torch.clamp(x_gen_ts_sde[:25], 0, 1)),
        (f'LDM Generated\n(d_lat={latent_dims[1]})', torch.clamp(x_gen_ldm_best, 0, 1)),
    ]

    for ax, (title, imgs) in zip(axes, panel_data):
        imgs_np = imgs[:25].view(25, 28, 28).detach().cpu().numpy()
        imgs_np = np.clip(imgs_np, 0, 1)
        grid = np.zeros((5 * 28, 5 * 28))
        for i in range(5):
            for j in range(5):
                grid[i*28:(i+1)*28, j*28:(j+1)*28] = imgs_np[i*5+j]
        ax.imshow(grid, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.suptitle('MNIST: Two-Stage Manifold Diffusion vs LDM', fontsize=14)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'mnist_generation_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ================================================================
    # OPTIMAL STOPPING
    # ================================================================
    print("\n--- Optimal Stopping Analysis ---")
    stopping_results = []
    for d_latent in [10, 20, 50]:
        P = dataset.get_pca_projection(d_latent)
        model_stop = LatentDiffusionModel(d=d, d_latent=d_latent, projection_matrix=P)
        stop_res = model_stop.optimal_stopping_analysis(dataset, n_eval=2000)
        stopping_results.append((f'd_lat={d_latent}', stop_res))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, res in stopping_results:
        line, = ax.plot(res['t_stops'], res['mse_scores'], '-o', markersize=3,
                       linewidth=1.5, label=label)
        ax.plot(res['optimal_t_stop'], res['optimal_mse'], '*',
                color=line.get_color(), markersize=12)
    ax.set_xlabel('Stopping Time $t_{stop}$')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('MNIST: Optimal Stopping Analysis (Wu et al.)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'mnist_optimal_stopping.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ================================================================
    # SCORE MSE AT MULTIPLE NOISE LEVELS
    # ================================================================
    print("\n--- Score MSE vs Noise Level ---")
    test_sigmas = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    ts_mses = []
    ldm_recon_mses = {}

    for sigma in test_sigmas:
        with torch.no_grad():
            xt, x0, eps, _ = dataset.sample_noisy(2000, sigma)
            true_score = dataset.true_score(xt, x0, sigma)
            t_eval = torch.tensor(sigma)
            pred_score, _ = model_ts(xt, t_eval)
            mse = F.mse_loss(pred_score, true_score).item()
            ts_mses.append(mse)

        for label, res in results_ldm_list:
            if label not in ldm_recon_mses:
                ldm_recon_mses[label] = []
            with torch.no_grad():
                x_recon = res['model'].autoencoder.decode(
                    res['model'].autoencoder.encode(x0))
                ldm_recon_mses[label].append(F.mse_loss(x_recon, x0).item())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.semilogy(test_sigmas, ts_mses, 'o-', color='#2ecc71', linewidth=2,
                markersize=6, label='Two-Stage (score MSE)')
    ax.set_xlabel('Noise Level σ')
    ax.set_ylabel('Score MSE (log)')
    ax.set_title('Two-Stage: Score MSE vs Noise Level')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    for label, mses in ldm_recon_mses.items():
        ax.plot(test_sigmas, mses, 'o-', markersize=4, label=label)
    ax.set_xlabel('Noise Level σ')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('LDM: Autoencoder Reconstruction Error')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'mnist_score_vs_noise.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ================================================================
    # COMPREHENSIVE SUMMARY FIGURE
    # ================================================================
    fig = plt.figure(figsize=(18, 10))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # (a) PCA spectrum
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(range(n_plot), explained_var[:n_plot].numpy(), 'o-', markersize=2)
    ax.axvline(x=k_95, color='red', linestyle='--', label=f'k={k_95} (95%)')
    ax.set_xlabel('Component')
    ax.set_ylabel('Variance')
    ax.set_title('(a) MNIST PCA Spectrum')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Dimensional collapse
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(range(n_plot), np.maximum(eigs[:n_plot], 1e-10), 'o-', markersize=2)
    ax.axvline(x=k_95, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('(b) Learned $WW^T$ Spectrum')
    ax.grid(True, alpha=0.3)

    # (c) Training curves
    ax = fig.add_subplot(gs[0, 2])
    for label, res in results_ldm_list:
        losses = res['losses']
        w = min(50, len(losses)//5)
        sm = np.convolve(losses, np.ones(w)/w, mode='valid') if w > 1 else losses
        ax.semilogy(sm, label=label, linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('(c) LDM Training Curves')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (d) Score MSE vs noise
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(test_sigmas, ts_mses, 'o-', color='#2ecc71', linewidth=2, label='Two-Stage')
    ax.set_xlabel('σ')
    ax.set_ylabel('Score MSE')
    ax.set_title('(d) Score MSE vs Noise Level')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (e) Optimal stopping
    ax = fig.add_subplot(gs[1, 1])
    for label, res in stopping_results:
        ax.plot(res['t_stops'], res['mse_scores'], '-o', markersize=2, label=label)
        ax.plot(res['optimal_t_stop'], res['optimal_mse'], '*', markersize=10)
    ax.set_xlabel('$t_{stop}$')
    ax.set_ylabel('MSE')
    ax.set_title('(e) Optimal Stopping')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (f) Comparison bar
    ax = fig.add_subplot(gs[1, 2])
    names = []
    gen_mses = []
    recon_mses = []
    for label, res in results_ldm_list:
        names.append(label)
        gen_mses.append(res['generation_mse'])
        recon_mses.append(res['recon_mse'])

    x_pos = np.arange(len(names))
    width = 0.35
    ax.bar(x_pos - width/2, recon_mses, width, label='Recon MSE', color='#3498db')
    ax.bar(x_pos + width/2, gen_mses, width, label='Gen MSE', color='#e74c3c')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=8, rotation=20)
    ax.set_ylabel('MSE')
    ax.set_title('(f) LDM: Reconstruction vs Generation')
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    path = os.path.join(FIG_DIR, 'mnist_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  MNIST intrinsic dim (95% PCA): k={k_95}")
    print(f"\n  {'Model':<30} {'Latent':<8} {'Eps Loss':<12} {'Recon MSE':<12}")
    print("  " + "-" * 70)
    print(f"  {'Two-Stage (ours)':<30} {64:<8} "
          f"{results_ts['total_loss']:<12.6f} {results_ts['recon_error']:<12.6f}")
    print(f"  {'LDM-CNN (fair baseline)':<30} {64:<8} "
          f"{results_ldm_cnn['total_loss']:<12.6f} {results_ldm_cnn['recon_error']:<12.6f}")
    for label, res in results_ldm_list:
        print(f"  {'LDM-Linear ' + label:<30} {label.split('=')[1]:<8} "
              f"{res['final_score_mse']:<12.6f} {res['recon_mse']:<12.6f}")

    return {
        'two_stage': results_ts,
        'ldm_cnn': results_ldm_cnn,
        'ldm_list': results_ldm_list,
        'k_95': k_95,
        'stopping': stopping_results,
    }


if __name__ == '__main__':
    run_experiment(digit=None)
