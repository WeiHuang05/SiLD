"""
Experiment 2: Nonlinear Circle Manifold Comparison

Compares Two-Stage Manifold Diffusion vs Latent Diffusion Models
on S^1 embedded in R^d (the "Tokyo model" from the paper).
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

from data.manifold_data import CircleManifoldDataset
from models.two_stage import TwoStageManifoldDiffusion
from models.latent_diffusion import LatentDiffusionModel
from utils.visualization import plot_optimal_stopping

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def run_experiment(d=20, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = 0.001      # Stage 1: very small noise
    t_min = 0.01     # Stage 2: lower bound
    t_max = 0.5      # Stage 2: upper bound
    sigma_eval = 0.1 # for visualization/LDM comparison

    print("=" * 70)
    print(f"NONLINEAR CIRCLE EXPERIMENT: d={d}")
    print(f"  t0={t0} (Stage 1), t_range=[{t_min}, {t_max}] (Stage 2)")
    print("=" * 70)

    dataset = CircleManifoldDataset(d=d, seed=seed)

    # ================================================================
    # TWO-STAGE MANIFOLD DIFFUSION
    # ================================================================
    print("\n--- Two-Stage Manifold Diffusion ---")
    model_ts = TwoStageManifoldDiffusion(
        d=d, latent_dim=2, hidden_dim_enc=512, hidden_dim_score=256,
        n_layers=3, t0=t0, t_min=t_min, t_max=t_max
    )

    results_ts = model_ts.full_training(
        dataset,
        stage1_steps=500, stage2_steps=2000,
        lr1=1e-3, lr2=1e-3, batch_size=2048
    )

    # ================================================================
    # LATENT DIFFUSION MODELS
    # ================================================================
    latent_dims = [1, 2, 5, d]
    results_ldm = []

    for d_latent in latent_dims:
        print(f"\n--- LDM (d_latent={d_latent}) ---")

        x_samples, _ = dataset.sample_clean(5000)
        U, S, Vt = torch.linalg.svd(x_samples, full_matrices=False)
        proj_matrix = Vt[:d_latent, :]

        model_ldm = LatentDiffusionModel(
            d=d, d_latent=d_latent, hidden_dim=128, n_layers=2,
            projection_matrix=proj_matrix
        )

        losses = model_ldm.train_model(
            dataset, n_steps=2000, lr=1e-3, batch_size=2048
        )

        with torch.no_grad():
            x_true, _ = dataset.sample_clean(2000)
            x_gen = model_ldm.generate(2000, n_steps=200, t_stop=0.01)
            gen_mse = F.mse_loss(x_gen, x_true).item()
            x_recon = model_ldm.autoencoder.decode(model_ldm.autoencoder.encode(x_true))
            recon_mse = F.mse_loss(x_recon, x_true).item()

        results_ldm.append((f'd_lat={d_latent}', {
            'losses': losses,
            'final_score_mse': losses[-1],
            'generation_mse': gen_mse,
            'recon_mse': recon_mse,
        }))
        print(f"  Final loss: {losses[-1]:.6f}, Recon MSE: {recon_mse:.6f}, "
              f"Gen MSE: {gen_mse:.6f}")

    # ================================================================
    # SCORE FIELD VISUALIZATION (Fig 2 equivalent)
    # ================================================================
    print("\n--- Score Field Visualization ---")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ['Step 100\nOnly Manifold', 'Step 500\nManifold + Weak Density',
              'Final\nFull Score (Density Learned)']
    stage_steps = [100, 500, None]

    grid_range = 1.5
    n_grid = 20
    xx, yy = np.meshgrid(
        np.linspace(-grid_range, grid_range, n_grid),
        np.linspace(-grid_range, grid_range, n_grid)
    )
    grid_points = torch.zeros(n_grid * n_grid, d)
    grid_points[:, 0] = torch.tensor(xx.ravel(), dtype=torch.float32)
    grid_points[:, 1] = torch.tensor(yy.ravel(), dtype=torch.float32)

    t_vis = torch.tensor(sigma_eval)  # visualize at t=0.1

    for ax_idx, (ax, title, steps) in enumerate(zip(axes, titles, stage_steps)):
        if steps is not None:
            model_snap = TwoStageManifoldDiffusion(
                d=d, latent_dim=2, hidden_dim_enc=512, hidden_dim_score=256,
                n_layers=3, t0=t0, t_min=t_min, t_max=t_max
            )
            model_snap.train_stage1(dataset, n_steps=steps,
                                    lr=1e-3, batch_size=2048)
        else:
            model_snap = model_ts

        with torch.no_grad():
            pred_score, _ = model_snap(grid_points, t_vis)
            u = pred_score[:, 0].numpy().reshape(n_grid, n_grid)
            v = pred_score[:, 1].numpy().reshape(n_grid, n_grid)

        speed = np.sqrt(u**2 + v**2)
        ax.streamplot(xx, yy, u, v, density=1.5, linewidth=0.8,
                      color=speed, cmap='viridis')

        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'r-', linewidth=2, alpha=0.6)
        ax.plot([1, -1], [0, 0], 'r*', markersize=15)

        ax.set_xlim(-grid_range, grid_range)
        ax.set_ylim(-grid_range, grid_range)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.2)

    plt.suptitle('Evolution of Score Field: From Geometry to Density', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'nonlinear_score_evolution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ================================================================
    # COMPARISON BAR CHART
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    names = ['Two-Stage\n(ours)']
    vals = [results_ts['total_loss']]
    for label, res in results_ldm:
        names.append(f'LDM\n{label}')
        vals.append(res['final_score_mse'])
    colors = ['#2ecc71'] + ['#3498db'] * len(results_ldm)
    ax.bar(range(len(names)), vals, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel('Score Matching Loss')
    ax.set_title(f'Score Estimation on Circle Manifold (d={d}, k=1)')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    eigs = results_ts['eigenvalues']
    ax.semilogy(range(min(20, len(eigs))), np.maximum(eigs[:20], 1e-10),
                'o-', color='steelblue', markersize=4)
    ax.axvline(x=1.5, color='red', linestyle='--', alpha=0.7, label='k=1 (circle)')
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue (log scale)')
    ax.set_title('Dimensional Collapse for Nonlinear Manifold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'nonlinear_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ================================================================
    # OPTIMAL STOPPING
    # ================================================================
    print("\n--- Optimal Stopping Analysis ---")
    stopping_results = []
    for d_latent in [1, 2, 5]:
        x_samples, _ = dataset.sample_clean(3000)
        U, S, Vt = torch.linalg.svd(x_samples, full_matrices=False)
        P = Vt[:d_latent, :]
        model_stop = LatentDiffusionModel(d=d, d_latent=d_latent, projection_matrix=P)
        stop_res = model_stop.optimal_stopping_analysis(dataset, n_eval=2000)
        stopping_results.append((f'd_lat={d_latent}', stop_res))

    plot_optimal_stopping(stopping_results, save_name='nonlinear_optimal_stopping.png')

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30} {'Loss':<15} {'Recon MSE':<15} {'Gen MSE'}")
    print("-" * 75)
    print(f"{'Two-Stage (ours)':<30} {results_ts['total_loss']:<15.6f}")
    for label, res in results_ldm:
        print(f"{'LDM ' + label:<30} {res['final_score_mse']:<15.6f} "
              f"{res['recon_mse']:<15.6f} {res['generation_mse']:.6f}")

    return {
        'two_stage': results_ts,
        'ldm_list': results_ldm,
        'stopping': stopping_results,
    }


if __name__ == '__main__':
    run_experiment(d=20)
