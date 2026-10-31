"""
Toy diagnostic for the decomposed score network.

Setup: data lives on a k-dim linear subspace of R^d, with an in-manifold
mixture-of-Gaussians distribution. Observations x = x0 + sigma_t * eps with
small sigma_t (so h_t = sigma_t**2 is small). The denoiser score has a
clean closed-form decomposition into three parts:
  (i)  a nonlinear in-manifold MoG score,
  (ii) a linear in-manifold correction (active once the basis W is found),
  (iii) a global -x/h contraction along directions orthogonal to the
       manifold.

The script trains a two-stage model that exposes these three branches and
tracks per-branch error against the analytical ground truth, producing
three diagnostic plots.
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.experiment_variant import output_dir, experiment_variant


# ------------------------------------------------------------
# 0. Hyperparameters and setup
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--projection-reg-weight', type=float, default=10.0)
parser.add_argument(
    '--regularizer', choices=['none', 'l2', 'projection'], default='projection',
)
parser.add_argument('--reg-weight', type=float, default=None)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

d, k_true, m = 100, 5, 200
sigma_t = 0.1
h_t = sigma_t ** 2  # 0.01

batch_size = 4096
lr_stage1 = 0.001
lr_stage2 = 0.005
steps_stage1 = 1000
steps_stage2 = 1000
regularizer = args.regularizer
reg_weight = (
    args.projection_reg_weight if args.reg_weight is None else args.reg_weight
)
if reg_weight < 0:
    raise ValueError("reg-weight must be non-negative")

torch.manual_seed(args.seed)

# Ground truth: a random k-dim subspace + 3-component MoG inside it.
TRUE_A, _ = torch.linalg.qr(torch.randn(d, k_true))
TRUE_P = TRUE_A @ TRUE_A.T
TRUE_MEANS = torch.tensor([
    [2.5, 0.0, 0.0, 0.0, 0.0],
    [-2.5, 2.0, 0.0, 0.0, 0.0],
    [0.0, -2.5, 0.0, 0.0, 0.0],
])
TRUE_WTS = torch.tensor([0.3, 0.3, 0.4])


def get_batch(n):
    idx = torch.multinomial(TRUE_WTS, n, replacement=True)
    z = TRUE_MEANS[idx] + torch.randn(n, k_true) * 0.5
    x0 = z @ TRUE_A.T
    noise = torch.randn(n, d)
    xt = x0 + sigma_t * noise
    target = -noise / sigma_t
    return xt, target, z


def get_analytical_scores(x, z):
    """Closed-form scores for the in-manifold MoG and the orthogonal contraction.
    Used only for monitoring; never seen by the optimizer.
    """
    # 1. In-manifold MoG score, lifted back to ambient space via TRUE_A.
    z_q = z.unsqueeze(1)
    mu = TRUE_MEANS.unsqueeze(0)
    var = 0.5 ** 2 + h_t
    diff = z_q - mu
    exponent = -0.5 * torch.sum(diff ** 2, dim=2) / var
    probs = TRUE_WTS.unsqueeze(0) * torch.exp(exponent)
    p_z = torch.sum(probs, dim=1, keepdim=True) + 1e-10
    score_z = torch.sum(probs.unsqueeze(2) * (-diff / var), dim=1) / p_z
    true_nonlin = score_z @ TRUE_A.T

    # 2. Orthogonal linear contraction: -(I - AA^T) x / h.
    true_lin_ortho = -(x @ (torch.eye(d) - TRUE_P)) / h_t

    return true_nonlin, true_lin_ortho


# ------------------------------------------------------------
# 1. Architecture
# ------------------------------------------------------------
class DecomposedScoreNet(nn.Module):
    def __init__(self, d, m, h_t):
        super().__init__()
        self.h_t = h_t
        self.W = nn.Parameter(torch.randn(d, m) * 0.02)
        self.V = nn.Parameter(torch.eye(m))
        self.bias = nn.Parameter(torch.zeros(1, m))

    def forward(self, x):
        u = x @ self.W + self.bias
        # Branch 1: nonlinear in-manifold score (learned in Stage 2).
        s_nonlin = (F.relu(u) @ self.V.T) @ self.W.T

        # Branch 2: linear in-manifold correction (frozen once W is set in Stage 1).
        s_lin_manifold = (((x @ self.W) / self.h_t)) @ self.W.T

        # Branch 3: global -x/h contraction (a fixed physical constant).
        s_contraction = -x / self.h_t

        return s_nonlin + s_lin_manifold + s_contraction

    @torch.no_grad()
    def extract_learned_components(self, x):
        u = x @ self.W + self.bias
        s_nonlin = (F.relu(u) @ self.V.T) @ self.W.T

        # Combined linear part = in-manifold correction + global contraction.
        # Should converge to -(I - AA^T) x / h once W spans the true subspace.
        s_lin_total = (((x @ self.W) / self.h_t)) @ self.W.T - (x / self.h_t)
        return s_nonlin, s_lin_total


model = DecomposedScoreNet(d, m, h_t)

history = {
    'total_loss': [],
    'nonlin_mse': [],   # Oracle metric, monitoring only.
    'lin_mse': [],
}


# ------------------------------------------------------------
# 2. Training loop with monitoring
# ------------------------------------------------------------

# --- Stage 1: discover the manifold basis W. ---
model.V.requires_grad = False
model.bias.requires_grad = False
opt1 = torch.optim.Adam([model.W], lr=lr_stage1)

print(f"--- Stage 1: Manifold Discovery ({steps_stage1} steps) ---")
print(f"Regularizer: {regularizer}; weight: {reg_weight}")
for s in range(steps_stage1):
    xt, target, z = get_batch(batch_size)
    pred = model(xt)
    dsm_loss = F.mse_loss(pred, target)
    if regularizer == 'projection':
        projector = model.W @ model.W.T
        projection_residual = projector @ projector - projector
        reg_loss = projection_residual.pow(2).sum()
    elif regularizer == 'l2':
        # Algorithm 1 uses standard weight decay; unlike the projection
        # penalty, this does not encode idempotence or the target rank.
        reg_loss = model.W.pow(2).sum()
    else:
        reg_loss = torch.zeros((), device=model.W.device)
    loss = dsm_loss + reg_weight * reg_loss

    opt1.zero_grad(); loss.backward(); opt1.step()

    if s % 50 == 0:
        with torch.no_grad():
            true_nl, true_lin = get_analytical_scores(xt, z)
            pred_nl, pred_lin = model.extract_learned_components(xt)
            err_nl = F.mse_loss(pred_nl, true_nl).item()
            err_lin = F.mse_loss(pred_lin, true_lin).item()
            history['total_loss'].append(loss.item())
            history['nonlin_mse'].append(err_nl)
            history['lin_mse'].append(err_lin)
            print(f"Step {s:03d} | Total: {loss.item():.2f} | "
                  f"DSM: {dsm_loss.item():.2f} | "
                  f"Reg: {reg_loss.item():.4f} | "
                  f"Linear Err: {err_lin:.2f} | NonLin Err: {err_nl:.2f}")

# Do not post-process W with an SVD projection. This keeps the Stage-1
# solution exactly as learned by gradient descent and isolates whether the
# two-stage mechanism works without an oracle-like orthogonalization step.
print("\n[Skipping SVD projection; initializing bias only...]")
with torch.no_grad():
    # Bias init matters: if pre-activations sit far from 0, all ReLUs stay
    # silent and the nonlinear branch effectively dies.
    proj_x = (get_batch(2000)[0] @ model.W)
    model.bias.copy_(-proj_x.mean(0) + torch.randn(1, m) * proj_x.std(0))

# --- Stage 2: learn the in-manifold distribution (V, bias). ---
model.W.requires_grad = False
model.V.requires_grad = True
model.bias.requires_grad = True
opt2 = torch.optim.Adam([model.V, model.bias], lr=lr_stage2)

print(f"\n--- Stage 2: Distribution Learning ({steps_stage2} steps) ---")
for s in range(steps_stage2):
    xt, target, z = get_batch(batch_size)
    pred = model(xt)
    loss = F.mse_loss(pred, target)

    opt2.zero_grad(); loss.backward(); opt2.step()

    if s % 50 == 0:
        with torch.no_grad():
            true_nl, true_lin = get_analytical_scores(xt, z)
            pred_nl, pred_lin = model.extract_learned_components(xt)

            err_nl = F.mse_loss(pred_nl, true_nl).item()
            err_lin = F.mse_loss(pred_lin, true_lin).item()

            history['total_loss'].append(loss.item())
            history['nonlin_mse'].append(err_nl)
            history['lin_mse'].append(err_lin)
            print(f"Step {s:03d} | Total: {loss.item():.2f} | "
                  f"Linear Err: {err_lin:.2f} | NonLin Err: {err_nl:.2f}")


# ------------------------------------------------------------
# 3. Visualization
# ------------------------------------------------------------
plt.figure(figsize=(18, 5))

# Plot 1: per-branch error decomposition during training.
plt.subplot(1, 3, 1)
x_axis = np.arange(len(history['total_loss'])) * 50
plt.plot(x_axis, history['total_loss'], 'k-', label='Total MSE Loss (observed)')
plt.plot(x_axis, history['nonlin_mse'], 'r-', label='Non-linear Component Error (hidden)')
plt.plot(x_axis, history['lin_mse'], 'g--', label='Linear Component Error (hidden)')
plt.axvline(x=steps_stage1, color='gray', linestyle=':', label='Stage Switch')
plt.yscale('log')
plt.title("Monitoring Loss Decomposition")
plt.xlabel("Steps")
plt.ylabel("MSE (log scale)")
plt.legend()

# Plot 2: learned vs analytical nonlinear profile along the first manifold axis.
plt.subplot(1, 3, 2)
with torch.no_grad():
    z_scan = torch.zeros(500, k_true)
    z_scan[:, 0] = torch.linspace(-6, 6, 500)
    x_scan = z_scan @ TRUE_A.T
    t_nonlin, _ = get_analytical_scores(x_scan, z_scan)
    p_nonlin, _ = model.extract_learned_components(x_scan)
    t_val = (t_nonlin @ TRUE_A)[:, 0]
    p_val = (p_nonlin @ TRUE_A)[:, 0]

    plt.plot(z_scan[:, 0], t_val, 'r--', lw=3, label='Ground Truth (MoG)')
    plt.plot(z_scan[:, 0], p_val, 'b-', label='Model Learned')
    plt.title("Learned Density Profile (projected)")
    plt.legend()

# Plot 3: response along directions orthogonal to the manifold.
plt.subplot(1, 3, 3)
with torch.no_grad():
    xt_test, _, _ = get_batch(1000)
    ortho_proj_true = torch.eye(d) - TRUE_P
    _, p_lin_total = model.extract_learned_components(xt_test)
    x_ortho = (xt_test @ ortho_proj_true).numpy().flatten()
    p_lin_val = (p_lin_total @ ortho_proj_true).numpy().flatten()

    plt.scatter(x_ortho, p_lin_val, alpha=0.1, color='blue')
    plt.plot(x_ortho, -x_ortho / h_t, 'r--', label='Theoretical -1/h')
    plt.title("Orthogonal Linear Response")
    plt.legend()

plt.tight_layout()
figure_path = os.path.join(
    output_dir(__file__, 'figures'),
    f'toy_decomposed_score_no_svd_{regularizer}_'
    f'{reg_weight:g}_seed{args.seed}.png',
)
plt.savefig(figure_path, dpi=200, bbox_inches='tight')
plt.close()

with torch.no_grad():
    xt_eval, target_eval, z_eval = get_batch(10000)
    true_nonlin, true_lin = get_analytical_scores(xt_eval, z_eval)
    pred_nonlin, pred_lin = model.extract_learned_components(xt_eval)
    final_metrics = {
        'total_mse': F.mse_loss(model(xt_eval), target_eval).item(),
        'nonlinear_mse': F.mse_loss(pred_nonlin, true_nonlin).item(),
        'linear_mse': F.mse_loss(pred_lin, true_lin).item(),
        'wwt_to_true_projector_fro': (
            (model.W @ model.W.T - TRUE_P).pow(2).sum().sqrt().item()
        ),
    }

import json
result_path = os.path.join(
    output_dir(__file__, 'logs'),
    f'toy_decomposed_score_no_svd_{regularizer}_'
    f'{reg_weight:g}_seed{args.seed}.json',
)
with open(result_path, 'w') as handle:
    json.dump({
        'experiment_variant': experiment_variant(),
        'config': {
            'd': d, 'k_true_evaluation_only': k_true, 'width': m,
            'sigma_t': sigma_t, 'seed': args.seed,
            'regularizer': regularizer, 'reg_weight': reg_weight,
            'post_training_svd': False,
        },
        'metrics': final_metrics,
    }, handle, indent=2)
print(f"Experiment variant: {experiment_variant()}")
print(f"Saved: {figure_path}")
print(f"Saved: {result_path}")
