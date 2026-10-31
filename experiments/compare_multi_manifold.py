"""
Experiment: Multi-Manifold Lifted Synthetic — ULD vs nonlinear-VAE LDM.

The dataset (data.manifold_data.MultiManifoldLiftedDataset) is K disjoint
2D Swiss-roll components lifted to R^D via a *fixed* random Tanh-MLP.
Because the lift map is known, we can measure ground-truth metrics that
FID can't give us:

  - distance from each generated sample to the true manifold M
  - per-component coverage (which Swiss roll did the sample land on?)
  - KL(generated coverage || true coverage)

Theoretical hypothesis:
  ULD's denoising AE at small t0 should map noisy points back to the
  nearest manifold point (Tweedie's formula), so generated samples sit
  on M. The bottleneck preserves the disconnected components.

  The VAE-LDM, with KL pulling its posterior toward N(0,I), has to bridge
  the gaps between disjoint components — its mean estimator smears mass
  off-manifold and into the gaps.

This experiment tests whether that hypothesis holds in practice on a
controlled dataset where we KNOW the manifold.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.manifold_data import MultiManifoldLiftedDataset
from models.two_stage_latent import (
    TwoStageManifoldDiffusion,
    LatentScoreNet,
    ManifoldEncoder,
    ManifoldDecoder,
)

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ================================================================
# Nonlinear VAE-LDM baseline (matched MLP capacity to ULD)
# ================================================================

class VAEEncoder(nn.Module):
    """MLP encoder producing (mu, logvar) — same width as ULD's encoder."""
    def __init__(self, d, latent_dim, hidden_dim=512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def encode_deterministic(self, x):
        mu, _ = self.forward(x)
        return mu


class NonlinearVAELDM(nn.Module):
    """
    Fair nonlinear LDM baseline:
      - MLP VAE encoder (mu, logvar) of same width as ULD
      - MLP decoder of same width as ULD
      - Same LatentScoreNet diffusion head
      - VAE training with KL regularization (Stage 1)
      - DSM in latent space (Stage 2)
    """
    def __init__(self, d, latent_dim, hidden_dim_enc=512,
                 hidden_dim_score=256, n_layers=4, kl_weight=1e-4):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.T = 0.999

        self.vae_encoder = VAEEncoder(d, latent_dim, hidden_dim_enc)
        self.decoder = ManifoldDecoder(latent_dim, d, hidden_dim_enc)
        self.score_net = LatentScoreNet(latent_dim,
                                        hidden_dim=hidden_dim_score,
                                        n_layers=n_layers)

        self.beta_min = 0.1
        self.beta_max = 20.0

        self.register_buffer('z_mean', torch.zeros(latent_dim))
        self.register_buffer('z_std', torch.ones(latent_dim))

    def noise_schedule(self, t):
        t_t = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
        log_mean = -0.25 * t_t**2 * (self.beta_max - self.beta_min) \
                   - 0.5 * t_t * self.beta_min
        alpha_bar = torch.exp(2.0 * log_mean)
        sigma = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-8))
        return alpha_bar, sigma

    def freeze_ae(self):
        for p in self.vae_encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    def train_vae(self, dataset, n_steps, lr=1e-3, batch_size=2048, device='cpu'):
        params = list(self.vae_encoder.parameters()) + list(self.decoder.parameters())
        opt = torch.optim.Adam(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)
        for step in range(n_steps):
            x0, _ = dataset.sample_clean(batch_size)
            x0 = x0.to(device)
            mu, logvar = self.vae_encoder(x0)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            x_recon = self.decoder(z)
            mse = F.mse_loss(x_recon, x0)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse + self.kl_weight * kl
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            if (step + 1) % 200 == 0:
                print(f"  VAE [{step+1}/{n_steps}] mse={mse.item():.6f} "
                      f"kl={kl.item():.4f}")
        # Latent stats
        with torch.no_grad():
            zs = []
            for _ in range(5):
                x0, _ = dataset.sample_clean(min(batch_size, 2000))
                x0 = x0.to(device)
                zs.append(self.vae_encoder.encode_deterministic(x0))
            zs = torch.cat(zs, dim=0)
            self.z_mean = zs.mean(0)
            self.z_std = zs.std(0).clamp(min=1e-6)

    def train_diffusion(self, dataset, n_steps, lr=3e-4, batch_size=2048,
                        device='cpu'):
        opt = torch.optim.Adam(self.score_net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)
        t_min = 1e-3
        for step in range(n_steps):
            x0, _ = dataset.sample_clean(batch_size)
            x0 = x0.to(device)
            with torch.no_grad():
                z0 = (self.vae_encoder.encode_deterministic(x0) - self.z_mean) \
                     / self.z_std.clamp(min=1e-6)
            t = torch.rand(z0.shape[0], device=device) * (self.T - t_min) + t_min
            alpha_bar, sigma = self.noise_schedule(t)
            eps = torch.randn_like(z0)
            z_t = torch.sqrt(alpha_bar).unsqueeze(-1) * z0 \
                  + sigma.unsqueeze(-1) * eps
            eps_pred = self.score_net(z_t, t)
            loss = F.mse_loss(eps_pred, eps)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.score_net.parameters(), 1.0)
            opt.step(); sched.step()
            if (step + 1) % 200 == 0:
                print(f"  Diff [{step+1}/{n_steps}] eps_loss={loss.item():.6f}")

    @torch.no_grad()
    def sample(self, n, n_steps=200, device='cpu'):
        ts = torch.linspace(self.T, 1e-3, n_steps + 1, device=device)
        z = torch.randn(n, self.latent_dim, device=device)
        for i in range(n_steps):
            t_cur, t_next = ts[i], ts[i + 1]
            t_b = torch.full((n,), t_cur.item(), device=device)
            alpha_cur, sigma_cur = self.noise_schedule(t_cur)
            alpha_next, sigma_next = self.noise_schedule(t_next)
            eps_pred = self.score_net(z, t_b)
            z0_pred = (z - sigma_cur * eps_pred) / torch.sqrt(alpha_cur).clamp(min=1e-6)
            if i == n_steps - 1:
                z = z0_pred
            else:
                z = torch.sqrt(alpha_next) * z0_pred + sigma_next * eps_pred
        z_raw = z * self.z_std + self.z_mean
        return self.decoder(z_raw)


# ================================================================
# Plotting helpers
# ================================================================

def plot_pca_comparison(real, gen_uld, gen_ldm, save_path, title_extra=''):
    """Project everything to 2D via PCA on the real data and plot side-by-side."""
    real = real.detach().cpu().numpy()
    gen_uld = gen_uld.detach().cpu().numpy()
    gen_ldm = gen_ldm.detach().cpu().numpy()
    mu = real.mean(0, keepdims=True)
    Xc = real - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:2]  # (2, D)
    proj = lambda X: (X - mu) @ P.T
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (label, X) in zip(axes, [('Real', real),
                                     ('ULD (ours)', gen_uld),
                                     ('VAE-LDM', gen_ldm)]):
        Xp = proj(X)
        ax.scatter(Xp[:, 0], Xp[:, 1], s=4, alpha=0.5)
        ax.set_title(label, fontsize=13)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        ax.set_aspect('equal', adjustable='datalim')
    fig.suptitle(f'Multi-manifold (PCA on real){title_extra}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ================================================================
# Main experiment
# ================================================================

def run_experiment(D=128, K=3, latent_dim=8, t0=0.05,
                   stage1_steps=4000, stage2_steps=8000,
                   batch_size=2048, n_eval=4000, seed=42, tag=''):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("=" * 70)
    print(f"Multi-Manifold Lifted (D={D}, K={K} components, intrinsic=2)")
    print(f"  latent_dim={latent_dim}, t0={t0}")
    print(f"  Stage1={stage1_steps}, Stage2={stage2_steps}, batch={batch_size}")
    print("=" * 70)

    dataset = MultiManifoldLiftedDataset(D=D, K=K, n_pool=20000, seed=seed)
    print(f"  Ground-truth coverage: {[round(c,3) for c in dataset.true_coverage.tolist()]}")

    # PCA k95 baseline
    ev = dataset.pca_explained_variance
    cumvar = torch.cumsum(ev, 0) / ev.sum()
    k95 = int((cumvar < 0.95).sum().item()) + 1
    print(f"  Intrinsic (95% PCA): k={k95}")

    # Real-data baseline metrics
    real_x, _ = dataset.sample_clean(n_eval)
    base = dataset.evaluate_samples(real_x)
    print(f"\n  Real-data baseline:")
    print(f"    mean_dist_to_M={base['mean_dist_to_M']:.4f}  "
          f"p95={base['p95_dist_to_M']:.4f}")
    print(f"    coverage={[round(c,3) for c in base['mode_coverage']]}")

    # =================================================================
    # ULD
    # =================================================================
    print("\n" + "=" * 70)
    print("MODEL 1: ULD (Two-Stage Manifold Diffusion)")
    print("=" * 70)
    uld = TwoStageManifoldDiffusion(
        d=D, latent_dim=latent_dim,
        hidden_dim_enc=512, hidden_dim_score=256,
        n_layers=4, t0=t0,
    ).to(device)

    print("\n--- ULD Stage 1 (denoising AE at small t0) ---")
    uld.train_stage1(dataset, n_steps=stage1_steps, lr=1e-3, batch_size=batch_size)
    uld.freeze_stage1()

    print("\n--- ULD Stage 2 (latent diffusion on [t0, T]) ---")
    uld.train_stage2(dataset, n_steps=stage2_steps, lr=3e-4, batch_size=batch_size)

    print("\n--- ULD Sampling + Eval ---")
    with torch.no_grad():
        gen_uld = uld.sample(n_eval, n_steps=200, device=device).cpu()
    eval_uld = dataset.evaluate_samples(gen_uld)
    print(f"  ULD: mean_dist_to_M={eval_uld['mean_dist_to_M']:.4f}  "
          f"p95={eval_uld['p95_dist_to_M']:.4f}")
    print(f"       coverage={[round(c,3) for c in eval_uld['mode_coverage']]}")
    print(f"       KL(gen||true)={eval_uld['mode_kl_to_true']:.4f}")

    # =================================================================
    # VAE-LDM
    # =================================================================
    print("\n" + "=" * 70)
    print("MODEL 2: Nonlinear VAE-LDM (matched MLP capacity)")
    print("=" * 70)
    ldm = NonlinearVAELDM(
        d=D, latent_dim=latent_dim,
        hidden_dim_enc=512, hidden_dim_score=256,
        n_layers=4, kl_weight=1e-4,
    ).to(device)

    print("\n--- LDM Stage 1 (VAE) ---")
    ldm.train_vae(dataset, n_steps=stage1_steps, lr=1e-3,
                  batch_size=batch_size, device=device)
    ldm.freeze_ae()

    print("\n--- LDM Stage 2 (latent diffusion) ---")
    ldm.train_diffusion(dataset, n_steps=stage2_steps, lr=3e-4,
                        batch_size=batch_size, device=device)

    print("\n--- LDM Sampling + Eval ---")
    with torch.no_grad():
        gen_ldm = ldm.sample(n_eval, n_steps=200, device=device).cpu()
    eval_ldm = dataset.evaluate_samples(gen_ldm)
    print(f"  LDM: mean_dist_to_M={eval_ldm['mean_dist_to_M']:.4f}  "
          f"p95={eval_ldm['p95_dist_to_M']:.4f}")
    print(f"       coverage={[round(c,3) for c in eval_ldm['mode_coverage']]}")
    print(f"       KL(gen||true)={eval_ldm['mode_kl_to_true']:.4f}")

    # =================================================================
    # Plots and summary
    # =================================================================
    suffix = f'_{tag}' if tag else ''
    plot_pca_comparison(
        real_x, gen_uld, gen_ldm,
        os.path.join(FIG_DIR, f'multi_manifold_pca{suffix}.png'),
        title_extra=f'  D={D}, K={K}, latent={latent_dim}, t0={t0}',
    )

    summary = {
        'config': {
            'D': D, 'K': K, 'latent_dim': latent_dim, 't0': t0,
            'stage1_steps': stage1_steps, 'stage2_steps': stage2_steps,
            'seed': seed,
        },
        'real_baseline': base,
        'uld': eval_uld,
        'ldm': eval_ldm,
    }
    out_path = os.path.join(LOG_DIR, f'multi_manifold_results{suffix}.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved results JSON: {out_path}")

    print("\n" + "=" * 70)
    print("MULTI-MANIFOLD SUMMARY")
    print("=" * 70)
    print(f"  {'Method':<14} {'mean dM':<10} {'p95 dM':<10} {'KL(gen||true)':<14} coverage")
    print("  " + "-" * 70)
    print(f"  {'Real':<14} {base['mean_dist_to_M']:<10.4f} "
          f"{base['p95_dist_to_M']:<10.4f} "
          f"{0.0:<14.4f} {[round(c,3) for c in base['mode_coverage']]}")
    print(f"  {'ULD (ours)':<14} {eval_uld['mean_dist_to_M']:<10.4f} "
          f"{eval_uld['p95_dist_to_M']:<10.4f} "
          f"{eval_uld['mode_kl_to_true']:<14.4f} "
          f"{[round(c,3) for c in eval_uld['mode_coverage']]}")
    print(f"  {'VAE-LDM':<14} {eval_ldm['mean_dist_to_M']:<10.4f} "
          f"{eval_ldm['p95_dist_to_M']:<10.4f} "
          f"{eval_ldm['mode_kl_to_true']:<14.4f} "
          f"{[round(c,3) for c in eval_ldm['mode_coverage']]}")
    return summary


if __name__ == '__main__':
    run_experiment()
