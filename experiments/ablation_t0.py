"""
t0 ablation on the synthetic multi-manifold dataset.

For each t0 in {0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2}, train a
fresh ULD on MultiManifoldLiftedDataset and report:
  - mean / p95 distance to true manifold M (lower = on manifold)
  - mode coverage and KL(gen || true)
  - Stage-1 reconstruction MSE (a sanity check)
  - Stage-2 average eps loss

The synthetic dataset is the right testbed because it has ground-truth
manifold-distance metrics that show t0's effect directly. CIFAR FID would
require 8x full retraining which is too expensive — the t0 dependence is
the same theoretical phenomenon either way.

Theoretical expectation: the curve should be U-shaped.
  - very small t0 → Stage 1 sees almost no noise, recon target is the
    identity, manifold is recovered exactly, but the score net is trained
    on a thin shell near M and may be poorly conditioned at slightly
    larger t.
  - very large t0 → the projection target is no longer onto M but onto a
    "blurred" version of M. Manifold-distance increases, mode separation
    erodes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.manifold_data import MultiManifoldLiftedDataset
from models.two_stage_latent import TwoStageManifoldDiffusion
import torch.nn.functional as F


FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def run_one_t0(dataset, t0, latent_dim=8, stage1_steps=4000,
               stage2_steps=8000, batch_size=2048, n_eval=4000,
               seed=42, device='cpu'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n{'#' * 70}")
    print(f"# t0 = {t0}")
    print(f"{'#' * 70}")

    model = TwoStageManifoldDiffusion(
        d=dataset.D, latent_dim=latent_dim,
        hidden_dim_enc=512, hidden_dim_score=256, n_layers=4,
        t0=t0,
    ).to(device)

    print(f"  Stage 1: t0={t0}, {stage1_steps} steps")
    model.train_stage1(dataset, n_steps=stage1_steps, lr=1e-3,
                       batch_size=batch_size)
    model.freeze_stage1()

    # Recon MSE on a fresh batch
    with torch.no_grad():
        x0_t, _ = dataset.sample_clean(2000)
        x0_t = x0_t.to(device)
        _, x_hat = model.project(x0_t)
        recon_mse = F.mse_loss(x_hat, x0_t).item()
        print(f"  Stage 1 recon MSE: {recon_mse:.6e}")

    print(f"  Stage 2: {stage2_steps} steps, t in [{t0}, 0.999]")
    model.train_stage2(dataset, n_steps=stage2_steps, lr=3e-4,
                       batch_size=batch_size)

    # Stage 2 eps eval
    eval_times = [max(t0, 0.01), 0.1, 0.3, 0.5, 0.8]
    total = 0.0
    for t_eval in eval_times:
        with torch.no_grad():
            x0, _ = dataset.sample_clean(2000)
            x0 = x0.to(device)
            z0 = model.encode(x0)
            t_b = torch.full((2000,), t_eval, device=device)
            z_t, eps_z = model.forward_diffusion_latent(z0, t_b)
            eps_z_pred = model.predict_eps_latent(z_t, t_b)
            mse = F.mse_loss(eps_z_pred, eps_z).item()
            total += mse
    avg_eps_loss = total / len(eval_times)
    print(f"  Stage 2 avg eps loss: {avg_eps_loss:.6f}")

    # Sample + eval on dataset metrics
    with torch.no_grad():
        gen = model.sample(n_eval, n_steps=200, device=device).cpu()
    metrics = dataset.evaluate_samples(gen)
    print(f"  Generated mean dM: {metrics['mean_dist_to_M']:.4f}  "
          f"p95: {metrics['p95_dist_to_M']:.4f}")
    print(f"  Coverage: {[round(c,3) for c in metrics['mode_coverage']]}  "
          f"KL_to_true: {metrics['mode_kl_to_true']:.4f}")

    return {
        't0': t0,
        'recon_mse': recon_mse,
        'avg_eps_loss': avg_eps_loss,
        **metrics,
    }


def run_ablation(t0_values=None, D=128, K=3, latent_dim=8,
                 stage1_steps=4000, stage2_steps=8000,
                 batch_size=2048, n_eval=4000, seed=42):
    if t0_values is None:
        t0_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = MultiManifoldLiftedDataset(D=D, K=K, n_pool=20000, seed=seed)
    print(f"\nReal-data baseline:")
    real_x, _ = dataset.sample_clean(n_eval)
    real_eval = dataset.evaluate_samples(real_x)
    print(f"  mean_dist_to_M={real_eval['mean_dist_to_M']:.4f}  "
          f"p95={real_eval['p95_dist_to_M']:.4f}")
    print(f"  Ground-truth coverage: "
          f"{[round(c,3) for c in dataset.true_coverage.tolist()]}")

    results = []
    for t0 in t0_values:
        r = run_one_t0(dataset, t0, latent_dim=latent_dim,
                       stage1_steps=stage1_steps, stage2_steps=stage2_steps,
                       batch_size=batch_size, n_eval=n_eval,
                       seed=seed, device=device)
        results.append(r)

    # Save JSON
    out = {
        'config': {
            'D': D, 'K': K, 'latent_dim': latent_dim,
            'stage1_steps': stage1_steps, 'stage2_steps': stage2_steps,
            'batch_size': batch_size, 'n_eval': n_eval, 'seed': seed,
        },
        'real_baseline': real_eval,
        'sweep': results,
    }
    out_path = os.path.join(LOG_DIR, 'ablation_t0_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Plot the sweep
    t0s = np.array([r['t0'] for r in results])
    mean_dM = np.array([r['mean_dist_to_M'] for r in results])
    p95_dM = np.array([r['p95_dist_to_M'] for r in results])
    kl = np.array([r['mode_kl_to_true'] for r in results])
    recon = np.array([r['recon_mse'] for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].axhline(real_eval['mean_dist_to_M'], color='gray', ls='--',
                    label='real baseline')
    axes[0].plot(t0s, mean_dM, 'o-', color='C0', label='mean')
    axes[0].plot(t0s, p95_dM, 's--', color='C1', label='p95')
    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlabel('t0'); axes[0].set_ylabel('distance to manifold')
    axes[0].set_title('Distance to true manifold M')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(t0s, kl, 'o-', color='C2')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('t0'); axes[1].set_ylabel('KL(gen || true)')
    axes[1].set_title('Mode-coverage KL to ground truth')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t0s, recon, 'o-', color='C3')
    axes[2].set_xscale('log'); axes[2].set_yscale('log')
    axes[2].set_xlabel('t0'); axes[2].set_ylabel('Stage 1 recon MSE')
    axes[2].set_title('Stage 1 reconstruction quality')
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('t0 ablation on MultiManifoldLifted (D=%d, K=%d)' % (D, K),
                 fontsize=14)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, 'ablation_t0.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")

    print("\n" + "=" * 70)
    print("t0 ABLATION SUMMARY")
    print("=" * 70)
    print(f"  {'t0':<10} {'recon MSE':<14} {'mean dM':<12} {'p95 dM':<12} "
          f"{'KL_true':<10}")
    print("  " + "-" * 60)
    for r in results:
        print(f"  {r['t0']:<10.4f} {r['recon_mse']:<14.6e} "
              f"{r['mean_dist_to_M']:<12.4f} {r['p95_dist_to_M']:<12.4f} "
              f"{r['mode_kl_to_true']:<10.4f}")


if __name__ == '__main__':
    run_ablation()
