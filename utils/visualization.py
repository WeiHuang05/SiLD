"""
Visualization utilities for diffusion model experiments.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(SAVE_DIR, exist_ok=True)


def plot_training_curves(results_two_stage, results_ldm_list, save_name='training_curves.png'):
    """
    Plot training loss curves for Two-Stage vs LDM at different latent dims.

    Fig 1 from the paper: monitoring loss decomposition.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Two-Stage training dynamics
    ax = axes[0]
    s1_losses = results_two_stage['stage1_losses']
    s2_losses = results_two_stage.get('stage2_losses', [])
    n1 = len(s1_losses)

    all_losses = s1_losses + s2_losses
    ax.semilogy(range(len(all_losses)), all_losses, 'k-', linewidth=1.5, label='Total Loss')
    ax.axvline(x=n1, color='blue', linestyle='--', alpha=0.7, label='Stage Switch')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('MSE Loss (log scale)')
    ax.set_title('Two-Stage Manifold Diffusion')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: LDM training curves at different latent dims
    ax = axes[1]
    for label, res in results_ldm_list:
        losses = res['losses']
        # Smooth for readability
        window = min(50, len(losses) // 5)
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        else:
            smoothed = losses
        ax.semilogy(smoothed, label=label, linewidth=1.2)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Score Matching Loss (log scale)')
    ax.set_title('Latent Diffusion Models')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Final comparison bar chart
    ax = axes[2]
    names = ['Two-Stage']
    values = [results_two_stage['total_loss']]
    for label, res in results_ldm_list:
        names.append(label)
        values.append(res.get('final_score_mse', res['losses'][-1]))

    colors = ['#2ecc71'] + ['#3498db'] * len(results_ldm_list)
    bars = ax.bar(range(len(names)), values, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Score MSE')
    ax.set_title('Final Score Estimation Error')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_eigenvalue_spectrum(eigenvalues, k_true, save_name='eigenvalue_spectrum.png'):
    """
    Plot eigenvalue spectrum of W W^T to visualize dimensional collapse.

    Should show k large eigenvalues and (d-k) near-zero eigenvalues.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Full spectrum
    ax = axes[0]
    ax.bar(range(len(eigenvalues)), eigenvalues, color='steelblue', alpha=0.8)
    ax.axvline(x=k_true - 0.5, color='red', linestyle='--', label=f'True dim k={k_true}')
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue Magnitude')
    ax.set_title('Eigenvalue Spectrum of $WW^T$ (Dimensional Collapse)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Log scale zoom
    ax = axes[1]
    ax.semilogy(range(len(eigenvalues)), np.maximum(eigenvalues, 1e-10),
                'o-', color='steelblue', markersize=3)
    ax.axvline(x=k_true - 0.5, color='red', linestyle='--', label=f'True dim k={k_true}')
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue (log scale)')
    ax.set_title('Log-Scale Eigenvalue Decay')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_optimal_stopping(stopping_results, save_name='optimal_stopping.png'):
    """
    Plot the U-shaped Frechet/MSE curve as a function of stopping time.
    Demonstrates the optimal stopping phenomenon from Wu et al.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for label, res in stopping_results:
        t_stops = res['t_stops']
        mse_scores = res['mse_scores']
        opt_t = res['optimal_t_stop']
        opt_mse = res['optimal_mse']

        line, = ax.plot(t_stops, mse_scores, '-o', markersize=3, linewidth=1.5, label=label)
        ax.plot(opt_t, opt_mse, '*', color=line.get_color(), markersize=12)

    ax.set_xlabel('Stopping Time $t_{stop}$')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Optimal Stopping Analysis (Wu et al.)\n'
                 'Stars mark optimal stopping times')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_comparison_summary(results, save_name='comparison_summary.png'):
    """
    Comprehensive comparison figure with multiple panels.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: Training curves
    ax = fig.add_subplot(gs[0, 0])
    ts = results['two_stage']
    s1 = ts['stage1_losses']
    s2 = ts.get('stage2_losses', [])
    all_l = s1 + s2
    ax.semilogy(all_l, 'k-', linewidth=1, label='Two-Stage Total')
    ax.axvline(x=len(s1), color='blue', linestyle='--', alpha=0.5, label='Stage Switch')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('(a) Two-Stage Training')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: LDM training curves
    ax = fig.add_subplot(gs[0, 1])
    for label, res in results['ldm_list']:
        losses = res['losses']
        w = min(50, len(losses) // 5)
        smoothed = np.convolve(losses, np.ones(w)/w, mode='valid') if w > 1 else losses
        ax.semilogy(smoothed, label=label, linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('(b) LDM Training (varying latent dim)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Eigenvalue spectrum
    ax = fig.add_subplot(gs[0, 2])
    eigs = ts['eigenvalues']
    k_true = results.get('k_true', 5)
    ax.semilogy(range(len(eigs)), np.maximum(eigs, 1e-10), 'o-',
                color='steelblue', markersize=2)
    ax.axvline(x=k_true - 0.5, color='red', linestyle='--', label=f'k={k_true}')
    ax.set_xlabel('Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('(c) Dimensional Collapse ($WW^T$ spectrum)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Optimal stopping curves
    ax = fig.add_subplot(gs[1, 0])
    if 'stopping' in results:
        for label, res in results['stopping']:
            ax.plot(res['t_stops'], res['mse_scores'], '-o', markersize=2,
                    label=label, linewidth=1)
            ax.plot(res['optimal_t_stop'], res['optimal_mse'], '*',
                    markersize=10)
    ax.set_xlabel('$t_{stop}$')
    ax.set_ylabel('MSE')
    ax.set_title('(d) Optimal Stopping (LDM)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 5: Score MSE comparison
    ax = fig.add_subplot(gs[1, 1])
    names = ['Two-Stage\n(ours)']
    mses = [ts['total_loss']]
    for label, res in results['ldm_list']:
        names.append(f'LDM\n{label}')
        mses.append(res.get('final_score_mse', res['losses'][-1]))

    colors = ['#2ecc71'] + ['#3498db'] * len(results['ldm_list'])
    ax.bar(range(len(names)), mses, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('Score MSE')
    ax.set_title('(e) Final Score Estimation Error')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 6: Decomposed errors (if available)
    ax = fig.add_subplot(gs[1, 2])
    if 'decomposed' in results:
        dec = results['decomposed']
        categories = list(dec.keys())
        tangent = [dec[c].get('tangent_mse', 0) for c in categories]
        normal = [dec[c].get('normal_mse', 0) for c in categories]

        x_pos = np.arange(len(categories))
        width = 0.35
        ax.bar(x_pos - width/2, tangent, width, label='Tangential', color='#e74c3c')
        ax.bar(x_pos + width/2, normal, width, label='Normal', color='#3498db')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, fontsize=8, rotation=20)
        ax.set_ylabel('MSE')
        ax.set_title('(f) Tangent vs Normal Error Decomposition')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', fontsize=14)
        ax.set_title('(f) Error Decomposition')

    path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_score_field_2d(model, dataset, sigma_t, steps_list=[100, 1000, 10000],
                        save_name='score_field_evolution.png'):
    """
    Visualize 2D score field (for circle manifold).
    Shows evolution from geometry-only to full density.
    """
    import torch

    fig, axes = plt.subplots(1, len(steps_list), figsize=(6*len(steps_list), 5))
    if len(steps_list) == 1:
        axes = [axes]

    # Create grid
    grid_range = 1.5
    n_grid = 20
    xx, yy = np.meshgrid(
        np.linspace(-grid_range, grid_range, n_grid),
        np.linspace(-grid_range, grid_range, n_grid)
    )
    grid_points = np.zeros((n_grid * n_grid, dataset.d))
    grid_points[:, 0] = xx.ravel()
    grid_points[:, 1] = yy.ravel()
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

    h_t = sigma_t ** 2

    for ax_idx, (ax, step) in enumerate(zip(axes, steps_list)):
        with torch.no_grad():
            pred_score, _ = model(grid_tensor, h_t)
            u = pred_score[:, 0].numpy().reshape(n_grid, n_grid)
            v = pred_score[:, 1].numpy().reshape(n_grid, n_grid)

        # Plot streamlines
        speed = np.sqrt(u**2 + v**2)
        ax.streamplot(xx, yy, u, v, density=1.5, linewidth=0.8,
                      color=speed, cmap='viridis')

        # Plot manifold (circle)
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'r-', linewidth=2, alpha=0.5)

        # Plot density peaks
        ax.plot([1, -1], [0, 0], 'r*', markersize=12)

        ax.set_xlim(-grid_range, grid_range)
        ax.set_ylim(-grid_range, grid_range)
        ax.set_aspect('equal')
        ax.set_title(f'Step {step}')
        ax.grid(True, alpha=0.2)

    plt.suptitle('Score Field Evolution: Geometry → Density', fontsize=14)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path
