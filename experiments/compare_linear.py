"""
Experiment 1: Linear Manifold Comparison

Compares Two-Stage Manifold Diffusion vs Latent Diffusion Models
on a linear subspace manifold (MoG in k-dim subspace of R^d).
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

from data.manifold_data import LinearManifoldDataset
from models.two_stage import TwoStageManifoldDiffusion
from models.latent_diffusion import LatentDiffusionModel
from utils.metrics import compute_decomposed_errors
from utils.visualization import (
    plot_training_curves,
    plot_eigenvalue_spectrum,
    plot_optimal_stopping,
    plot_comparison_summary,
)

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def run_experiment(d=100, k=5, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Noise scale parameters
    t0 = 0.001     # Stage 1: very small noise for geometric collapse
    t_min = 0.01   # Stage 2: lower bound of multi-time training
    t_max = 0.5    # Stage 2: upper bound
    sigma_eval = 0.1  # evaluation noise level (for LDM comparison)

    print("=" * 70)
    print(f"LINEAR MANIFOLD EXPERIMENT: d={d}, k={k}")
    print(f"  t0={t0} (Stage 1), t_range=[{t_min}, {t_max}] (Stage 2)")
    print(f"  h(t0)={t0**2:.8f} << h(t_min)={t_min**2:.6f} (scale separation)")
    print("=" * 70)

    dataset = LinearManifoldDataset(d=d, k=k, seed=seed)

    # ================================================================
    # 1) TWO-STAGE MANIFOLD DIFFUSION
    # ================================================================
    print("\n" + "=" * 70)
    print("MODEL 1: Two-Stage Manifold Diffusion (Ours)")
    print("=" * 70)

    model_ts = TwoStageManifoldDiffusion(
        d=d, latent_dim=k, hidden_dim_enc=512, hidden_dim_score=256,
        n_layers=3, t0=t0, t_min=t_min, t_max=t_max
    )

    results_ts = model_ts.full_training(
        dataset,
        stage1_steps=500, stage2_steps=3000,
        lr1=3e-3, lr2=1e-3, batch_size=4096
    )

    # Compute decomposed errors at sigma_eval
    sigma_eval_val, h_eval = model_ts.noise_schedule(sigma_eval)
    decomposed_ts = compute_decomposed_errors(model_ts, dataset, sigma_eval,
                                               model_type='two_stage_multitime')
    print(f"  Decomposed at t={sigma_eval}: tangent={decomposed_ts.get('tangent_mse', 'N/A')}, "
          f"normal={decomposed_ts.get('normal_mse', 'N/A')}")

    # ================================================================
    # 2) LATENT DIFFUSION MODELS (varying latent dimension)
    # ================================================================
    latent_dims = [k, 2 * k, d // 2]
    results_ldm_list = []

    for d_latent in latent_dims:
        print(f"\n{'=' * 70}")
        print(f"MODEL 2: Latent Diffusion Model (d_latent={d_latent})")
        print("=" * 70)

        x_samples, _ = dataset.sample_clean(5000)
        U, S, Vt = torch.linalg.svd(x_samples, full_matrices=False)
        proj_matrix = Vt[:d_latent, :]

        model_ldm = LatentDiffusionModel(
            d=d, d_latent=d_latent, hidden_dim=256, n_layers=3,
            projection_matrix=proj_matrix
        )

        losses_ldm = model_ldm.train_model(
            dataset, n_steps=3000, lr=1e-3, batch_size=4096
        )

        with torch.no_grad():
            x_true, _ = dataset.sample_clean(2000)
            x_gen = model_ldm.generate(2000, n_steps=200, t_stop=0.01)
            gen_mse = F.mse_loss(x_gen, x_true).item()
            x_recon = model_ldm.autoencoder.decode(model_ldm.autoencoder.encode(x_true))
            recon_mse = F.mse_loss(x_recon, x_true).item()

        ldm_res = {
            'losses': losses_ldm,
            'final_score_mse': losses_ldm[-1],
            'generation_mse': gen_mse,
            'recon_mse': recon_mse,
        }
        results_ldm_list.append((f'd_lat={d_latent}', ldm_res))
        print(f"  Final noise pred loss: {losses_ldm[-1]:.6f}")
        print(f"  Reconstruction MSE:    {recon_mse:.6f}")
        print(f"  Generation MSE:        {gen_mse:.6f}")

    # ================================================================
    # 3) OPTIMAL STOPPING ANALYSIS
    # ================================================================
    print(f"\n{'=' * 70}")
    print("OPTIMAL STOPPING ANALYSIS (Wu et al.)")
    print("=" * 70)

    stopping_results = []
    for d_latent in latent_dims:
        x_samples, _ = dataset.sample_clean(5000)
        U, S, Vt = torch.linalg.svd(x_samples, full_matrices=False)
        P = Vt[:d_latent, :]
        model_stop = LatentDiffusionModel(d=d, d_latent=d_latent, projection_matrix=P)
        stop_res = model_stop.optimal_stopping_analysis(dataset, n_eval=3000)
        stopping_results.append((f'd_lat={d_latent}', stop_res))

    # ================================================================
    # 4) GENERATE PLOTS
    # ================================================================
    print(f"\n{'=' * 70}")
    print("GENERATING FIGURES")
    print("=" * 70)

    plot_training_curves(results_ts, results_ldm_list,
                        save_name='linear_training_curves.png')
    plot_eigenvalue_spectrum(results_ts['eigenvalues'], k,
                           save_name='linear_eigenvalue_spectrum.png')
    plot_optimal_stopping(stopping_results,
                         save_name='linear_optimal_stopping.png')

    all_results = {
        'two_stage': results_ts,
        'ldm_list': results_ldm_list,
        'stopping': stopping_results,
        'k_true': k,
        'decomposed': {'Two-Stage': decomposed_ts},
    }
    plot_comparison_summary(all_results, save_name='linear_comparison_summary.png')

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30} {'Score/Noise MSE':<18} {'Gen MSE':<15} {'Recon MSE'}")
    print("-" * 80)
    print(f"{'Two-Stage (ours)':<30} {results_ts['total_loss']:<18.6f}")
    for label, res in results_ldm_list:
        print(f"{'LDM ' + label:<30} {res['final_score_mse']:<18.6f} "
              f"{res['generation_mse']:<15.6f} {res['recon_mse']:.6f}")
    print("-" * 80)

    return all_results


if __name__ == '__main__':
    run_experiment(d=100, k=5)
