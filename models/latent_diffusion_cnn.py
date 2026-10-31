"""
Latent Diffusion Model (LDM) with CNN autoencoder — fair comparison baseline.

Uses the EXACT same components as our Two-Stage method:
  - Same CNN encoder/decoder architecture
  - Same LatentScoreNet (AdaGN, LayerNorm, residual blocks)
  - Same latent_dim, hidden_dim, n_layers
  - Same noise schedule, same DDIM sampler

ONLY difference: autoencoder is trained with reconstruction + KL (standard VAE),
NOT via score matching at small t.

This isolates the effect of autoencoder training method on generation quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Import shared components from two_stage_latent
from models.two_stage_latent import (
    ConvEncoder, ConvDecoder,
    ManifoldEncoder, ManifoldDecoder,
    LatentScoreNet,
)


class VAEEncoder(nn.Module):
    """
    Wraps a deterministic encoder to output (mu, log_var) for VAE training.
    """
    def __init__(self, base_encoder, latent_dim):
        super().__init__()
        self.base = base_encoder
        # Replace the last linear layer with mu + logvar heads
        # Find the output dim of the base encoder's second-to-last layer
        self.mu_head = nn.Linear(latent_dim, latent_dim)
        self.logvar_head = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        h = self.base(x)
        return self.mu_head(h), self.logvar_head(h)

    def encode_deterministic(self, x):
        """Encode without sampling (use mu only)."""
        h = self.base(x)
        return self.mu_head(h)

    def encode_stochastic(self, x):
        """Encode with reparameterization trick."""
        mu, logvar = self.forward(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar


class LatentDiffusionCNN(nn.Module):
    """
    LDM with CNN autoencoder (fair comparison to Two-Stage).

    Training:
      1. Train autoencoder with reconstruction + KL (standard VAE)
      2. Freeze autoencoder, train latent diffusion

    Same architecture as Two-Stage. Only AE training differs.
    """

    def __init__(self, d, latent_dim=32, hidden_dim_enc=512, hidden_dim_score=256,
                 n_layers=4, kl_weight=1e-4, image_shape=None, base_ch=64,
                 diffusion_t_min=1e-4):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.T = 0.999
        if not 0.0 < diffusion_t_min < self.T:
            raise ValueError(
                f"diffusion_t_min must be in (0, {self.T}), got {diffusion_t_min}"
            )
        self.diffusion_t_min = float(diffusion_t_min)
        self.image_shape = image_shape

        # --- Autoencoder (same architecture as Two-Stage) ---
        if image_shape is not None:
            base_enc = ConvEncoder(image_shape, latent_dim, base_ch=base_ch)
            self.decoder = ConvDecoder(latent_dim, image_shape, base_ch=base_ch)
        else:
            base_enc = ManifoldEncoder(d, latent_dim, hidden_dim=hidden_dim_enc)
            self.decoder = ManifoldDecoder(latent_dim, d, hidden_dim=hidden_dim_enc)

        self.vae_encoder = VAEEncoder(base_enc, latent_dim)
        self.ae_frozen = False

        # --- Score network (SAME as Two-Stage) ---
        self.score_net = LatentScoreNet(
            latent_dim, hidden_dim=hidden_dim_score, n_layers=n_layers,
        )

        # VP-SDE noise schedule
        self.beta_min = 0.1
        self.beta_max = 20.0

        # Latent normalization
        self.register_buffer('z_mean', torch.zeros(latent_dim))
        self.register_buffer('z_std', torch.ones(latent_dim))

    def noise_schedule(self, t):
        t_tensor = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
        log_mean_coeff = -0.25 * t_tensor ** 2 * (self.beta_max - self.beta_min) \
                         - 0.5 * t_tensor * self.beta_min
        alpha_bar = torch.exp(2.0 * log_mean_coeff)
        sigma = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-8))
        return alpha_bar, sigma

    def encode(self, x):
        """Encode to normalized latent (deterministic, use mu)."""
        z = self.vae_encoder.encode_deterministic(x)
        return (z - self.z_mean) / self.z_std.clamp(min=1e-6)

    def decode(self, z_norm):
        """Decode from normalized latent."""
        z = z_norm * self.z_std + self.z_mean
        return self.decoder(z)

    def forward_diffusion_latent(self, z0, t):
        alpha_bar, sigma = self.noise_schedule(t)
        eps_z = torch.randn_like(z0)
        if alpha_bar.dim() > 0:
            alpha_bar = alpha_bar.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        z_t = torch.sqrt(alpha_bar) * z0 + sigma * eps_z
        return z_t, eps_z

    def freeze_ae(self):
        self.ae_frozen = True
        for p in self.vae_encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_autoencoder(self, dataset, n_steps=1000, lr=1e-3, batch_size=4096):
        """
        Train autoencoder with reconstruction + KL divergence.
        Standard VAE objective: L = recon + beta * KL(q(z|x) || N(0,I))
        """
        params = list(self.vae_encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
        device = next(self.parameters()).device

        losses = []
        for step in range(n_steps):
            x0, _ = dataset.sample_clean(batch_size)
            x0 = x0.to(device)

            # VAE forward
            z, mu, logvar = self.vae_encoder.encode_stochastic(x0)
            x_hat = self.decoder(z)

            # Reconstruction loss
            recon_loss = F.mse_loss(x_hat, x0)

            # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            kl_loss = -0.5 * torch.mean(
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            )

            loss = recon_loss + self.kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            if (step + 1) % 50 == 0:
                print(f"  AE [{step+1}/{n_steps}] recon={recon_loss.item():.6f} "
                      f"kl={kl_loss.item():.4f} total={loss.item():.6f}")

        # Compute latent normalization
        with torch.no_grad():
            zs = []
            for _ in range(10):
                x0, _ = dataset.sample_clean(min(batch_size, 5000))
                x0 = x0.to(device)
                z = self.vae_encoder.encode_deterministic(x0)
                zs.append(z)
            zs = torch.cat(zs, dim=0)
            self.z_mean = zs.mean(dim=0)
            self.z_std = zs.std(dim=0).clamp(min=1e-6)
            print(f"  Latent stats: mean_norm={self.z_mean.norm():.4f}, "
                  f"std=[{self.z_std.min():.4f}, {self.z_std.max():.4f}]")

        return losses

    def train_diffusion(self, dataset, n_steps=3000, lr=1e-3, batch_size=4096,
                        min_snr_gamma=None):
        """
        Train latent diffusion (same as Two-Stage Stage 2).

        If ``min_snr_gamma`` is not None, apply Min-SNR-gamma loss
        weighting. Default None preserves the original unweighted
        behavior so existing call sites remain reproducible.
        """
        optimizer = torch.optim.Adam(self.score_net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

        device = next(self.parameters()).device
        print(
            f"  LDM diffusion: t in [{self.diffusion_t_min}, {self.T}], "
            f"min_snr_gamma={min_snr_gamma}"
        )
        losses = []
        for step in range(n_steps):
            x0, _ = dataset.sample_clean(batch_size)
            x0 = x0.to(device)

            with torch.no_grad():
                z0 = self.encode(x0)

            t = (
                torch.rand(z0.shape[0], device=device)
                * (self.T - self.diffusion_t_min)
                + self.diffusion_t_min
            )
            z_t, eps_z = self.forward_diffusion_latent(z0, t)

            eps_z_pred = self.score_net(z_t, t)

            if min_snr_gamma is None:
                loss = F.mse_loss(eps_z_pred, eps_z)
            else:
                alpha_bar, sigma = self.noise_schedule(t)
                snr = (alpha_bar / sigma.pow(2).clamp(min=1e-8)).clamp(min=1e-8)
                gamma = torch.full_like(snr, float(min_snr_gamma))
                w = torch.minimum(snr, gamma) / snr
                w = w.unsqueeze(-1)
                loss = (w * (eps_z_pred - eps_z).pow(2)).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.score_net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            if (step + 1) % 200 == 0:
                tag = 'weighted_loss' if min_snr_gamma is not None else 'latent_eps_loss'
                print(f"  Diffusion [{step+1}/{n_steps}] {tag}={loss.item():.6f}")

        return losses

    # ------------------------------------------------------------------
    # Sampling (identical to Two-Stage)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, n_samples, n_steps=200, device='cpu', clamp=None):
        """DDIM on the same interval used for Stage-2 training, then decode."""
        self.eval()
        ts = torch.linspace(
            self.T, self.diffusion_t_min, n_steps + 1, device=device
        )
        z = torch.randn(n_samples, self.latent_dim, device=device)

        for i in range(n_steps):
            t_cur, t_next = ts[i], ts[i + 1]
            t_batch = torch.full((n_samples,), t_cur.item(), device=device)

            alpha_cur, sigma_cur = self.noise_schedule(t_cur)
            alpha_next, sigma_next = self.noise_schedule(t_next)

            eps_z_pred = self.score_net(z, t_batch)
            z0_pred = (z - sigma_cur * eps_z_pred) / torch.sqrt(alpha_cur).clamp(min=1e-6)
            if i == n_steps - 1:
                # Match SiLD: stop at t_min and use the final Tweedie clean
                # prediction instead of handing the decoder a re-noised latent.
                z = z0_pred
            else:
                z = torch.sqrt(alpha_next) * z0_pred + sigma_next * eps_z_pred

        x = self.decode(z)
        if clamp is not None:
            x = torch.clamp(x, clamp[0], clamp[1])
        self.train()
        return x

    @torch.no_grad()
    def generate(self, n_samples, n_steps=200, t_stop=0.0, device='cpu'):
        """Alias for sample() — compatibility with old LDM API."""
        return self.sample(n_samples, n_steps=n_steps, device=device, clamp=(0.0, 1.0))

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def full_training(self, dataset, ae_steps=1000, diffusion_steps=3000, **kwargs):
        print("=" * 60)
        print(f"LDM-CNN (fair baseline)")
        print(f"  d={self.d}, latent_dim={self.latent_dim}")
        print(f"  AE: reconstruction + KL (beta={self.kl_weight})")
        print(
            f"  Diffusion: latent R^{self.latent_dim}, "
            f"t in [{self.diffusion_t_min}, {self.T}]"
        )
        print(f"  Same encoder/decoder/score_net as Two-Stage")
        print("=" * 60)

        # Phase 1: Train autoencoder
        print("\n--- Phase 1: VAE Training (recon + KL) ---")
        ae_losses = self.train_autoencoder(
            dataset, n_steps=ae_steps,
            lr=kwargs.get('lr_ae', 1e-3),
            batch_size=kwargs.get('batch_size', 4096),
        )

        with torch.no_grad():
            x0_test, _ = dataset.sample_clean(5000)
            z_test = self.vae_encoder.encode_deterministic(x0_test)
            x_hat = self.decoder(z_test)
            recon_mse = F.mse_loss(x_hat, x0_test).item()
            z_norm = self.encode(x0_test)
            z_var = z_norm.var(dim=0).sort(descending=True).values
            active_dims = (z_var > z_var[0] * 0.01).sum().item()
            print(f"\n  Reconstruction MSE: {recon_mse:.6f}")
            print(f"  Active latent dims: {active_dims}/{self.latent_dim}")
            eigenvalues = z_var.numpy()

        self.freeze_ae()
        print("  AE frozen.\n")

        # Phase 2: Train diffusion
        print("--- Phase 2: Latent Diffusion ---")
        diff_losses = self.train_diffusion(
            dataset, n_steps=diffusion_steps,
            lr=kwargs.get('lr_diff', 1e-3),
            batch_size=kwargs.get('batch_size', 4096),
        )

        # Evaluation
        print("\n--- Final Evaluation ---")
        eval_times = [0.01, 0.1, 0.3, 0.5, 0.8]
        total_loss = 0.0
        for t_eval in eval_times:
            with torch.no_grad():
                x0, _ = dataset.sample_clean(5000)
                z0 = self.encode(x0)
                t_batch = torch.full((5000,), t_eval)
                z_t, eps_z = self.forward_diffusion_latent(z0, t_batch)
                eps_z_pred = self.score_net(z_t, t_batch)
                mse = F.mse_loss(eps_z_pred, eps_z).item()
                total_loss += mse
                print(f"  t={t_eval:.2f}: latent_eps_MSE={mse:.6f}")

        avg_loss = total_loss / len(eval_times)
        print(f"  Average latent eps MSE: {avg_loss:.6f}")

        return {
            'ae_losses': ae_losses,
            'diff_losses': diff_losses,
            'total_loss': avg_loss,
            'recon_error': recon_mse,
            'eigenvalues': eigenvalues,
        }
