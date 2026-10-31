"""
Evaluation metrics for comparing diffusion models.
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_score_mse(model, dataset, sigma_t, n_samples=5000, model_type='two_stage'):
    """Compute MSE between predicted and true score."""
    with torch.no_grad():
        xt, x0, eps, _ = dataset.sample_noisy(n_samples, sigma_t)
        true_score = dataset.true_score(xt, x0, sigma_t)

        if model_type == 'two_stage':
            h_t = sigma_t ** 2
            pred_score, _ = model(xt, h_t)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        mse = F.mse_loss(pred_score, true_score).item()

        # Also compute per-component errors
        # Normal direction error (orthogonal to manifold)
        # Tangential direction error (on manifold)
        return {
            'total_mse': mse,
            'score_norm_pred': pred_score.norm(dim=-1).mean().item(),
            'score_norm_true': true_score.norm(dim=-1).mean().item(),
        }


def compute_projection_error(model, dataset, sigma_t, n_samples=5000):
    """Compute how well Stage 1 approximates the manifold projection."""
    with torch.no_grad():
        xt, x0, eps, _ = dataset.sample_noisy(n_samples, sigma_t)
        h_t = sigma_t ** 2
        _, x_hat = model.stage1(xt, h_t)
        proj_mse = F.mse_loss(x_hat, x0).item()

        # Subspace alignment: compare W W^T to true projector
        if hasattr(dataset, 'get_projector'):
            true_P = dataset.get_projector()
            learned_G = model.stage1.get_gram_matrix()
            # Normalize both
            learned_P = learned_G / (learned_G.diagonal().mean() + 1e-8)
            alignment = F.mse_loss(learned_P, true_P).item()
        else:
            alignment = None

        return {
            'projection_mse': proj_mse,
            'subspace_alignment': alignment,
        }


def compute_frechet_distance(samples_true, samples_gen):
    """
    Compute Frechet distance between two sets of samples.
    FD = ||mu1 - mu2||^2 + Tr(S1 + S2 - 2*(S1 S2)^{1/2})
    """
    mu1 = samples_true.mean(dim=0)
    mu2 = samples_gen.mean(dim=0)

    # Covariance matrices
    diff1 = samples_true - mu1
    diff2 = samples_gen - mu2
    S1 = (diff1.T @ diff1) / (samples_true.shape[0] - 1)
    S2 = (diff2.T @ diff2) / (samples_gen.shape[0] - 1)

    # Mean term
    mean_diff = torch.sum((mu1 - mu2) ** 2).item()

    # Trace term via eigendecomposition
    product = S1 @ S2
    eigvals = torch.linalg.eigvalsh(product)
    eigvals = torch.clamp(eigvals, min=0)  # numerical stability
    sqrt_product_trace = torch.sum(torch.sqrt(eigvals)).item()

    trace_term = torch.trace(S1).item() + torch.trace(S2).item() - 2 * sqrt_product_trace

    return mean_diff + trace_term


def compute_decomposed_errors(model, dataset, sigma_t, n_samples=5000,
                              model_type='two_stage'):
    """
    Decompose score error into normal and tangential components.
    Only works for LinearManifoldDataset where we know the true projector.
    """
    with torch.no_grad():
        xt, x0, eps, _ = dataset.sample_noisy(n_samples, sigma_t)
        true_score = dataset.true_score(xt, x0, sigma_t)

        # Model always takes (x, t) now
        t = torch.tensor(sigma_t)
        pred_score, info = model(xt, t)

        error = pred_score - true_score

        if hasattr(dataset, 'get_projector'):
            P = dataset.get_projector()  # (d, d)
            P_perp = torch.eye(dataset.d) - P

            error_tangent = error @ P
            error_normal = error @ P_perp

            return {
                'total_mse': (error ** 2).mean().item(),
                'tangent_mse': (error_tangent ** 2).mean().item(),
                'normal_mse': (error_normal ** 2).mean().item(),
            }

        return {'total_mse': (error ** 2).mean().item()}
