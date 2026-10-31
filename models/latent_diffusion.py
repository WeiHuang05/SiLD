"""
Latent Diffusion Model (LDM) baseline with optimal stopping analysis.

Following Wu et al. (2025) "Optimal Stopping in Latent Diffusion Models":
  1. Linear autoencoder: encoder E = P (projection), decoder D = P^T
  2. Diffusion in the latent space of dimension d_latent
  3. Noise prediction (eps-parameterization) for stability
  4. Optimal stopping: early termination of backward process

The key insight from Wu et al. is that the Frechet distance between generated
and true distributions is non-monotonic in diffusion time -- there exists an
optimal stopping time that minimizes generation error.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearAutoencoder(nn.Module):
    """
    Linear autoencoder: E(x) = P x, D(z) = P^T z
    where P in R^{d_latent x d} is a projection matrix.

    Following Wu et al., the encoder projects onto the top-d_latent
    principal components of the data.
    """

    def __init__(self, d, d_latent, projection_matrix=None):
        super().__init__()
        self.d = d
        self.d_latent = d_latent

        if projection_matrix is not None:
            self.P = nn.Parameter(projection_matrix, requires_grad=False)
        else:
            M = torch.randn(d, d_latent)
            Q, _ = torch.linalg.qr(M)
            self.P = nn.Parameter(Q[:, :d_latent].T, requires_grad=False)  # d_latent x d

    def encode(self, x):
        """x: (batch, d) -> z: (batch, d_latent)"""
        return x @ self.P.T

    def decode(self, z):
        """z: (batch, d_latent) -> x_hat: (batch, d)"""
        return z @ self.P

    def reconstruction_error(self, x):
        return F.mse_loss(self.decode(self.encode(x)), x)


class LatentScoreNetwork(nn.Module):
    """
    Noise prediction network operating in latent space.
    Predicts eps (noise) given z_t and t.
    """

    def __init__(self, d_latent, hidden_dim=256, n_layers=3, time_embed_dim=64):
        super().__init__()
        self.d_latent = d_latent
        self.time_embed_dim = time_embed_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(d_latent, hidden_dim)

        # Hidden layers with residual connections
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))

        # Output
        self.output = nn.Linear(hidden_dim, d_latent)

    def sinusoidal_embedding(self, t):
        half_dim = self.time_embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    def forward(self, z_t, t):
        """
        Predict noise eps given noisy latent z_t and time t.

        Args:
            z_t: (batch, d_latent) noisy latent
            t: (batch,) diffusion time
        Returns:
            eps_pred: (batch, d_latent) predicted noise
        """
        t_emb = self.sinusoidal_embedding(t)
        t_feat = self.time_embed(t_emb)

        h = self.input_proj(z_t) + t_feat
        for layer in self.layers:
            h = h + layer(h)  # residual
        return self.output(h)


class LatentDiffusionModel(nn.Module):
    """
    Full Latent Diffusion Model with optimal stopping.

    Uses eps-parameterization for numerical stability:
      - Forward: z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1-alpha_bar_t) * eps
      - Network predicts eps
      - Score: s(z_t, t) = -eps_pred / sqrt(1-alpha_bar_t)
    """

    def __init__(self, d, d_latent, hidden_dim=256, n_layers=3,
                 projection_matrix=None):
        super().__init__()
        self.d = d
        self.d_latent = d_latent

        self.autoencoder = LinearAutoencoder(d, d_latent, projection_matrix)
        self.noise_net = LatentScoreNetwork(
            d_latent, hidden_dim=hidden_dim, n_layers=n_layers
        )

        # Noise schedule parameters
        self.beta_min = 0.1
        self.beta_max = 20.0

    def noise_schedule(self, t):
        """
        Variance-preserving schedule.
        Returns (alpha_bar_t, sigma_t) such that
          z_t = sqrt(alpha_bar_t) * z_0 + sigma_t * eps
        """
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        alpha_bar_t = torch.exp(2.0 * log_mean_coeff)
        sigma_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-8))
        return alpha_bar_t, sigma_t

    def forward_diffusion(self, z0, t):
        """Add noise to latent codes."""
        alpha_bar_t, sigma_t = self.noise_schedule(t)
        eps = torch.randn_like(z0)

        if t.dim() > 0:
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)

        z_t = torch.sqrt(alpha_bar_t) * z0 + sigma_t * eps
        return z_t, eps, alpha_bar_t.squeeze(-1), sigma_t.squeeze(-1)

    def score_matching_loss(self, x, t=None):
        """
        Denoising score matching loss (eps-parameterization).
        Predicts the noise eps directly — numerically stable.
        """
        batch_size = x.shape[0]
        z0 = self.autoencoder.encode(x)

        if t is None:
            t = torch.rand(batch_size, device=x.device) * 0.98 + 0.01

        z_t, eps, _, _ = self.forward_diffusion(z0, t)
        eps_pred = self.noise_net(z_t, t)

        loss = F.mse_loss(eps_pred, eps)
        return loss

    @torch.no_grad()
    def reverse_ddim(self, batch_size, n_steps=200, t_stop=0.0, device='cpu'):
        """
        DDIM deterministic sampling (Song et al. 2020).

        At each step:
          1. Predict z0 from z_t: z0_pred = (z_t - sigma_t * eps_pred) / sqrt(alpha_bar_t)
          2. Step to t_next: z_{t-1} = sqrt(alpha_bar_{t-1}) * z0_pred + sigma_{t-1} * eps_pred
        """
        T = 0.999
        ts = torch.linspace(T, max(t_stop, 1e-3), n_steps + 1, device=device)

        z = torch.randn(batch_size, self.d_latent, device=device)

        for i in range(n_steps):
            t_cur = ts[i]
            t_next = ts[i + 1]
            t_batch = torch.full((batch_size,), t_cur.item(), device=device)

            alpha_bar_cur, sigma_cur = self.noise_schedule(t_cur)
            alpha_bar_next, sigma_next = self.noise_schedule(t_next)

            # Predict noise
            eps_pred = self.noise_net(z, t_batch)

            # Predict z0
            z0_pred = (z - sigma_cur * eps_pred) / torch.sqrt(alpha_bar_cur).clamp(min=1e-6)

            # Clip z0 prediction for stability
            z0_pred = torch.clamp(z0_pred, -10.0, 10.0)

            # DDIM step: deterministic
            z = torch.sqrt(alpha_bar_next) * z0_pred + sigma_next * eps_pred

        return z

    @torch.no_grad()
    def generate(self, batch_size, n_steps=200, t_stop=0.0, device='cpu'):
        """Generate samples: reverse process in latent space, then decode."""
        z = self.reverse_ddim(batch_size, n_steps, t_stop, device)
        x_gen = self.autoencoder.decode(z)
        return x_gen

    @torch.no_grad()
    def noisy_autoencoder_proxy(self, x, sigma_noise):
        """
        Wu et al. noisy autoencoder proxy for optimal stopping analysis.
        x_gen = D(b_t * E(x) + a_t * Z)

        This captures both:
        - Reconstruction loss from the projection (d_latent < d)
        - Residual noise from incomplete denoising
        """
        z0 = self.autoencoder.encode(x)
        sigma_noise = min(sigma_noise, 0.9999)  # numerical safety
        b_t = (1 - sigma_noise ** 2) ** 0.5
        noise = torch.randn_like(z0) * sigma_noise
        z_noisy = b_t * z0 + noise
        return self.autoencoder.decode(z_noisy)

    def train_model(self, dataset, n_steps=2000, lr=1e-3, batch_size=4096):
        """Train the latent diffusion model via denoising score matching."""
        optimizer = torch.optim.Adam(self.noise_net.parameters(), lr=lr)

        losses = []
        for step in range(n_steps):
            x0, _ = dataset.sample_clean(batch_size)
            loss = self.score_matching_loss(x0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.noise_net.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())

            if (step + 1) % 200 == 0:
                print(f"  LDM [{step+1}/{n_steps}] loss={loss.item():.6f}")

        return losses

    @torch.no_grad()
    def evaluate_score_mse(self, dataset, sigma_t, n_samples=5000):
        """
        Evaluate score prediction quality in the ambient space.
        Compares the LDM's implied ambient score with the true denoising score.
        """
        xt, x0, eps, _ = dataset.sample_noisy(n_samples, sigma_t)
        true_score = dataset.true_score(xt, x0, sigma_t)

        # Encode noisy data, predict noise in latent space, map back
        z_t = self.autoencoder.encode(xt)
        t_val = 0.5  # mid-range time
        t = torch.full((n_samples,), t_val)
        eps_pred = self.noise_net(z_t, t)

        # Implied ambient score: P^T * (-eps_pred / sigma_t) in latent space
        _, sigma_schedule = self.noise_schedule(torch.tensor(t_val))
        latent_score = -eps_pred / sigma_schedule
        ambient_score = latent_score @ self.autoencoder.P  # back to ambient

        mse = F.mse_loss(ambient_score, true_score).item()
        return mse

    @torch.no_grad()
    def optimal_stopping_analysis(self, dataset, n_eval=2000, t_stops=None):
        """
        Analyze generation quality at different stopping times using
        the noisy autoencoder proxy from Wu et al.

        The proxy captures the U-shaped behavior:
        - Small t_stop: residual noise is low, but projection loses information
        - Large t_stop: lots of residual noise remains
        - Optimal: balance between the two
        """
        if t_stops is None:
            t_stops = np.linspace(0.01, 0.99, 30)

        x_true, _ = dataset.sample_clean(n_eval)

        # Also compute reconstruction error (baseline: projection only, no noise)
        x_recon = self.autoencoder.decode(self.autoencoder.encode(x_true))
        recon_mse = F.mse_loss(x_recon, x_true).item()

        mse_scores = []
        for t_stop in t_stops:
            t_tensor = torch.tensor(t_stop, dtype=torch.float32)
            _, sigma_t = self.noise_schedule(t_tensor)

            x_gen = self.noisy_autoencoder_proxy(x_true, sigma_t.item())
            mse = F.mse_loss(x_gen, x_true).item()
            mse_scores.append(mse)

        mse_scores = np.array(mse_scores)
        opt_idx = np.argmin(mse_scores)

        print(f"\n  Optimal stopping analysis (d_latent={self.d_latent}):")
        print(f"    Reconstruction MSE (no noise) = {recon_mse:.6f}")
        print(f"    Best t_stop = {t_stops[opt_idx]:.3f}")
        print(f"    Best proxy MSE = {mse_scores[opt_idx]:.6f}")
        print(f"    t_stop=0.99 MSE = {mse_scores[-1]:.6f}")

        return {
            't_stops': t_stops,
            'mse_scores': mse_scores,
            'optimal_t_stop': t_stops[opt_idx],
            'optimal_mse': mse_scores[opt_idx],
            'recon_mse': recon_mse,
        }
