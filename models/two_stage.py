"""
Two-Stage Manifold Diffusion Training (Algorithm 1 from the paper).

Key insight from Theorem 3.18 (End-to-End Sampling Guarantee):
  Phase I  [t_max, T]:  Large noise — vanilla score network on raw x_t.
                         The SDE trivially contracts the ambient space.
  Phase II [t_min, t_max]: Manifold regime — two-stage decomposition.
                         Stage 1 projection + latent refinement.

Stage 1: Learn manifold projection at SMALL t0 (Theorem 3.4).
  - The 1/h(t0)^2 singularity drives encoder-decoder to learn Pi_M.
  - Bottleneck (latent_dim << d) IS the "dimensional collapse".

Stage 2: Two-regime score network.
  - t > t_max: vanilla eps-prediction on raw x_t (standard diffusion)
  - t <= t_max: x0-prediction via latent refinement on projected data
    x0_pred = Decoder(z + delta_z) where z = Encoder(x_t), delta_z = ScoreNet(z, t)

During sampling:
  1. Start from noise at t=T
  2. t in [t_max, T]: vanilla network brings x_t toward manifold
  3. t in [t_min, t_max]: two-stage refines on the manifold
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ========================================================================
# MLP encoder/decoder for vector data
# ========================================================================

class ManifoldEncoder(nn.Module):
    def __init__(self, d, latent_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    def forward(self, x):
        return self.net(x)

class ManifoldDecoder(nn.Module):
    def __init__(self, latent_dim, d, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, d),
        )
    def forward(self, z):
        return self.net(z)


# ========================================================================
# CNN encoder/decoder for image data
# ========================================================================

class ConvEncoder(nn.Module):
    def __init__(self, image_shape, latent_dim, base_ch=64):
        super().__init__()
        C, H, W = image_shape
        self.image_shape = image_shape
        self.conv = nn.Sequential(
            nn.Conv2d(C, base_ch, 3, stride=2, padding=1), nn.BatchNorm2d(base_ch), nn.SiLU(),
            nn.Conv2d(base_ch, base_ch*2, 3, stride=2, padding=1), nn.BatchNorm2d(base_ch*2), nn.SiLU(),
            nn.Conv2d(base_ch*2, base_ch*4, 3, stride=2, padding=1), nn.BatchNorm2d(base_ch*4), nn.SiLU(),
        )
        with torch.no_grad():
            self._flat_size = self.conv(torch.zeros(1, C, H, W)).view(1, -1).shape[1]
        self.fc = nn.Sequential(nn.Linear(self._flat_size, 256), nn.SiLU(), nn.Linear(256, latent_dim))

    def forward(self, x):
        x_img = x.view(-1, *self.image_shape)
        return self.fc(self.conv(x_img).view(x.size(0), -1))

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, image_shape, base_ch=64):
        super().__init__()
        C, H, W = image_shape
        self.image_shape = image_shape
        self.d = C * H * W
        self.start_h, self.start_w = max(H // 7, 4), max(W // 7, 4)
        self.start_ch = base_ch * 4
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.SiLU(),
            nn.Linear(256, self.start_ch * self.start_h * self.start_w), nn.SiLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_ch*4, base_ch*2, 3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(base_ch*2), nn.SiLU(),
            nn.ConvTranspose2d(base_ch*2, base_ch, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(base_ch), nn.SiLU(),
            nn.ConvTranspose2d(base_ch, C, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
    def forward(self, z):
        h = self.fc(z).view(-1, self.start_ch, self.start_h, self.start_w)
        img = self.deconv(h)[:, :, :self.image_shape[1], :self.image_shape[2]]
        return img.reshape(-1, self.d)


# ========================================================================
# Score networks
# ========================================================================

class LatentScoreNet(nn.Module):
    """MLP score network for vector data."""
    def __init__(self, in_dim, out_dim, hidden_dim=256, n_layers=3, time_embed_dim=64):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_embed = nn.Sequential(nn.Linear(time_embed_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(n_layers)])
        self.output = nn.Linear(hidden_dim, out_dim)

    def sinusoidal_embedding(self, t):
        half_dim = self.time_embed_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim)
        if t.dim() == 0: t = t.unsqueeze(0)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, x, t):
        h = self.input_proj(x) + self.time_embed(self.sinusoidal_embedding(t))
        for layer in self.layers:
            h = h + layer(h)
        return self.output(h)


class LatentScoreNetWide(nn.Module):
    """Wider score network with AdaGN-style time conditioning."""
    def __init__(self, in_dim, out_dim, hidden_dim=512, n_layers=4, time_embed_dim=128):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_embed = nn.Sequential(nn.Linear(time_embed_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList()
        self.ada_lin = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)))
            self.ada_lin.append(nn.Linear(hidden_dim, hidden_dim * 2))
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim))

    def sinusoidal_embedding(self, t):
        half_dim = self.time_embed_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim)
        if t.dim() == 0: t = t.unsqueeze(0)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, x, t):
        t_feat = self.time_embed(self.sinusoidal_embedding(t))
        h = self.input_proj(x)
        for block, ada in zip(self.blocks, self.ada_lin):
            scale, shift = ada(t_feat).chunk(2, dim=-1)
            h = h + block(h) * (1 + scale) + shift
        return self.output(h)


# ========================================================================
# Main model
# ========================================================================

class TwoStageManifoldDiffusion(nn.Module):
    """
    Two-Stage Manifold Diffusion with two-regime sampling (Theorem 3.18).

    Phase I  (t > t_max): Vanilla eps-prediction on raw x_t.
    Phase II (t <= t_max): Two-stage x0-prediction via manifold projection.

    Stage 1 is trained at small t0 (geometric alignment via singularity).
    Stage 2 covers both phases.
    """

    def __init__(self, d, latent_dim=32, hidden_dim_enc=512, hidden_dim_score=256,
                 n_layers=3, t0=0.005, t_min=0.02, t_max=0.5,
                 image_shape=None):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim
        self.t0 = t0
        self.t_min = t_min
        self.t_max = t_max
        self.T = 0.999  # max diffusion time
        self.image_shape = image_shape

        # --- Stage 1: Encoder-Decoder (manifold projection) ---
        # Trained at small t0 where 1/h(t) singularity drives alignment
        if image_shape is not None:
            self.encoder = ConvEncoder(image_shape, latent_dim)
            self.decoder = ConvDecoder(latent_dim, image_shape)
        else:
            self.encoder = ManifoldEncoder(d, latent_dim, hidden_dim=hidden_dim_enc)
            self.decoder = ManifoldDecoder(latent_dim, d, hidden_dim=hidden_dim_enc)
        self.stage1_frozen = False

        # --- Stage 2a: Latent refinement (manifold regime, t <= t_max) ---
        # Predicts delta_z in R^k. x0_pred = Decoder(z + delta_z).
        ScoreNetCls = LatentScoreNetWide if image_shape else LatentScoreNet
        self.manifold_net = ScoreNetCls(
            latent_dim, latent_dim, hidden_dim=hidden_dim_score, n_layers=n_layers
        )

        # --- Stage 2b: Vanilla eps network (large noise, t > t_max) ---
        # Standard eps-prediction on raw x_t, no geometric decomposition
        self.vanilla_net = ScoreNetCls(
            d, d, hidden_dim=hidden_dim_score, n_layers=n_layers
        )

        self.use_x0_prediction = (image_shape is not None)

        # VP-SDE noise schedule
        self.beta_min = 0.1
        self.beta_max = 20.0

    def noise_schedule(self, t):
        t_tensor = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
        log_mean_coeff = -0.25 * t_tensor ** 2 * (self.beta_max - self.beta_min) - 0.5 * t_tensor * self.beta_min
        alpha_bar_t = torch.exp(2.0 * log_mean_coeff)
        sigma_t = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-8))
        return alpha_bar_t, sigma_t

    def forward_diffusion(self, x0, t):
        alpha_bar_t, sigma_t = self.noise_schedule(t)
        eps = torch.randn_like(x0)
        if alpha_bar_t.dim() > 0:
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)
        x_t = torch.sqrt(alpha_bar_t) * x0 + sigma_t * eps
        return x_t, eps

    def project(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def predict_x0(self, x_t, t):
        """
        Predict clean x0 — dispatches by noise level.

        t > t_max: vanilla eps-prediction → Tweedie x0
        t <= t_max: manifold x0-prediction via latent refinement
        """
        alpha_bar_t, sigma_t = self.noise_schedule(t)
        if isinstance(sigma_t, torch.Tensor) and sigma_t.dim() > 0:
            sigma_t_e = sigma_t.unsqueeze(-1)
            alpha_bar_e = alpha_bar_t.unsqueeze(-1)
        else:
            sigma_t_e = sigma_t
            alpha_bar_e = alpha_bar_t

        t_tensor = t if isinstance(t, torch.Tensor) else torch.tensor(t)
        if t_tensor.dim() == 0:
            t_tensor = t_tensor.expand(x_t.shape[0])

        # Determine which regime each sample belongs to
        t_val = t_tensor if t_tensor.dim() > 0 else t_tensor.unsqueeze(0)
        is_manifold = (t_val <= self.t_max)

        if is_manifold.all():
            # All in manifold regime
            return self._predict_x0_manifold(x_t, t_tensor, alpha_bar_e, sigma_t_e)
        elif (~is_manifold).all():
            # All in vanilla regime
            return self._predict_x0_vanilla(x_t, t_tensor, alpha_bar_e, sigma_t_e)
        else:
            # Mixed batch — handle both
            x0_pred = torch.zeros_like(x_t)
            info = {'x_hat': torch.zeros_like(x_t), 'z': None, 'delta_z': None}

            idx_m = is_manifold.nonzero(as_tuple=True)[0]
            idx_v = (~is_manifold).nonzero(as_tuple=True)[0]

            if len(idx_m) > 0:
                x0_m, info_m = self._predict_x0_manifold(
                    x_t[idx_m], t_tensor[idx_m],
                    alpha_bar_e[idx_m] if alpha_bar_e.dim() > 0 else alpha_bar_e,
                    sigma_t_e[idx_m] if sigma_t_e.dim() > 0 else sigma_t_e
                )
                x0_pred[idx_m] = x0_m
                info['x_hat'][idx_m] = info_m['x_hat']

            if len(idx_v) > 0:
                x0_v, info_v = self._predict_x0_vanilla(
                    x_t[idx_v], t_tensor[idx_v],
                    alpha_bar_e[idx_v] if alpha_bar_e.dim() > 0 else alpha_bar_e,
                    sigma_t_e[idx_v] if sigma_t_e.dim() > 0 else sigma_t_e
                )
                x0_pred[idx_v] = x0_v
                info['x_hat'][idx_v] = info_v['x_hat']

            return x0_pred, info

    def _predict_x0_manifold(self, x_t, t, alpha_bar_e, sigma_t_e):
        """Manifold regime: x0 = Decoder(Encoder(x_t) + delta_z)."""
        if self.stage1_frozen:
            with torch.no_grad():
                z, x_hat = self.project(x_t)
            z = z.detach()
            x_hat = x_hat.detach()
        else:
            z, x_hat = self.project(x_t)

        delta_z = self.manifold_net(z, t)
        x0_pred = self.decoder(z + delta_z)
        return x0_pred, {'x_hat': x_hat, 'z': z, 'delta_z': delta_z}

    def _predict_x0_vanilla(self, x_t, t, alpha_bar_e, sigma_t_e):
        """Vanilla regime: eps-prediction → Tweedie x0."""
        eps_pred = self.vanilla_net(x_t, t)
        x0_pred = (x_t - sigma_t_e * eps_pred) / torch.sqrt(alpha_bar_e).clamp(min=1e-6)
        return x0_pred, {'x_hat': x0_pred, 'z': None, 'delta_z': None}

    def predict_eps(self, x_t, t):
        """Convert x0 prediction to eps prediction."""
        alpha_bar_t, sigma_t = self.noise_schedule(t)
        if isinstance(sigma_t, torch.Tensor) and sigma_t.dim() > 0:
            sigma_t_e = sigma_t.unsqueeze(-1)
            alpha_bar_e = alpha_bar_t.unsqueeze(-1)
        else:
            sigma_t_e = sigma_t
            alpha_bar_e = alpha_bar_t

        x0_pred, info = self.predict_x0(x_t, t)
        eps_pred = (x_t - torch.sqrt(alpha_bar_e) * x0_pred) / sigma_t_e.clamp(min=1e-6)
        info['eps_pred'] = eps_pred
        info['eps_normal'] = eps_pred  # for compatibility
        info['eps_residual'] = torch.zeros_like(eps_pred)
        return eps_pred, info

    def forward(self, x, t):
        """Full score prediction. Returns (score, info_dict)."""
        eps_pred, info = self.predict_eps(x, t)
        _, sigma_t = self.noise_schedule(t)
        if isinstance(sigma_t, torch.Tensor) and sigma_t.dim() > 0:
            sigma_t = sigma_t.unsqueeze(-1)
        score = -eps_pred / sigma_t.clamp(min=1e-6)
        return score, info

    def freeze_stage1(self):
        self.stage1_frozen = True
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    # ==================================================================
    # Training
    # ==================================================================

    def train_stage1(self, dataset, n_steps=500, lr=1e-3, batch_size=4096):
        """
        Stage 1: learn manifold projection at SMALL t0.

        The 1/h(t0) singularity drives reconstruction: x_hat -> x0.
        The bottleneck forces dimensional collapse.
        """
        sigma_t0 = self.t0
        print(f"  Stage 1: t0={self.t0}, sigma_t0={sigma_t0:.4f}, latent_dim={self.latent_dim}")

        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

        losses = []
        for step in range(n_steps):
            x0, _ = dataset.sample_clean(batch_size)
            eps = torch.randn_like(x0)
            x_t = x0 + sigma_t0 * eps

            z, x_hat = self.project(x_t)
            loss = F.mse_loss(x_hat, x0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()), 1.0
            )
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            if (step + 1) % 50 == 0:
                print(f"  Stage 1 [{step+1}/{n_steps}] recon_loss={loss.item():.6f}")
        return losses

    def train_stage2(self, dataset, n_steps=3000, lr=1e-3, batch_size=4096):
        """
        Stage 2: train both vanilla and manifold networks.

        Vanilla network: t ~ U[t_max, T], predicts eps from raw x_t.
        Manifold network: t ~ U[t_min, t_max], predicts delta_z for refinement.
        """
        params = list(self.manifold_net.parameters()) + list(self.vanilla_net.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

        losses_manifold = []
        losses_vanilla = []

        for step in range(n_steps):
            x0, _ = dataset.sample_clean(batch_size)
            half = batch_size // 2

            # --- Manifold regime: t in [t_min, t_max] ---
            x0_m = x0[:half]
            t_m = torch.rand(half) * (self.t_max - self.t_min) + self.t_min
            x_t_m, eps_m = self.forward_diffusion(x0_m, t_m)

            with torch.no_grad():
                z_m, x_hat_m = self.project(x_t_m)

            delta_z = self.manifold_net(z_m, t_m)
            x0_pred_m = self.decoder(z_m + delta_z)
            loss_m = F.mse_loss(x0_pred_m, x0_m)

            # --- Vanilla regime: t in [t_max, T] ---
            x0_v = x0[half:]
            t_v = torch.rand(batch_size - half) * (self.T - self.t_max) + self.t_max
            x_t_v, eps_v = self.forward_diffusion(x0_v, t_v)

            eps_pred_v = self.vanilla_net(x_t_v, t_v)
            loss_v = F.mse_loss(eps_pred_v, eps_v)

            # Combined loss
            loss = loss_m + loss_v

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()

            losses_manifold.append(loss_m.item())
            losses_vanilla.append(loss_v.item())

            if (step + 1) % 200 == 0:
                print(f"  Stage 2 [{step+1}/{n_steps}] "
                      f"manifold_loss={loss_m.item():.6f}  "
                      f"vanilla_loss={loss_v.item():.6f}")

        return losses_manifold

    # ==================================================================
    # Sampling
    # ==================================================================

    @torch.no_grad()
    def sample(self, n_samples, n_steps=200, device='cpu', clamp=None):
        """
        DDIM sampling with two-regime score.
        Phase I  (t > t_max): vanilla eps-prediction
        Phase II (t <= t_max): manifold x0-prediction
        """
        self.eval()
        ts = torch.linspace(self.T, max(self.t_min, 1e-3), n_steps + 1, device=device)

        _, sigma_T = self.noise_schedule(torch.tensor(self.T))
        x = torch.randn(n_samples, self.d, device=device) * sigma_T

        for i in range(n_steps):
            t_cur = ts[i]
            t_next = ts[i + 1]
            t_batch = torch.full((n_samples,), t_cur.item(), device=device)

            alpha_bar_cur, sigma_cur = self.noise_schedule(t_cur)
            alpha_bar_next, sigma_next = self.noise_schedule(t_next)

            # Predict x0 (dispatches to vanilla or manifold automatically)
            x0_pred, _ = self.predict_x0(x, t_batch)

            if clamp is not None:
                x0_pred = torch.clamp(x0_pred, clamp[0], clamp[1])

            # DDIM step
            eps_pred = (x - torch.sqrt(alpha_bar_cur) * x0_pred) / sigma_cur.clamp(min=1e-6)
            x = torch.sqrt(alpha_bar_next) * x0_pred + sigma_next * eps_pred

        self.train()
        return x

    @torch.no_grad()
    def sample_predictor_corrector(self, n_samples, n_steps=200,
                                    n_corrector=2, snr=0.1,
                                    device='cpu', clamp=None):
        """Predictor-Corrector sampler."""
        self.eval()
        ts = torch.linspace(self.T, max(self.t_min, 1e-3), n_steps + 1, device=device)
        dt = (self.T - max(self.t_min, 1e-3)) / n_steps

        _, sigma_T = self.noise_schedule(torch.tensor(self.T))
        x = torch.randn(n_samples, self.d, device=device) * sigma_T

        for i in range(n_steps):
            t_cur = ts[i]
            score, _ = self.forward(x, t_cur)
            beta_t = self.beta_min + (self.beta_max - self.beta_min) * t_cur.item()
            drift = -0.5 * beta_t * x - beta_t * score
            diffusion = math.sqrt(beta_t * dt)
            x = x - drift * dt
            if i < n_steps - 1:
                x = x + diffusion * torch.randn_like(x)
            for _ in range(n_corrector):
                score, _ = self.forward(x, t_cur)
                grad_norm = torch.mean(torch.norm(score, dim=-1))
                noise_norm = math.sqrt(self.d)
                step_size = (snr * noise_norm / (grad_norm + 1e-8)) ** 2 * 2
                x = x + step_size * score + math.sqrt(2 * step_size) * torch.randn_like(x)
            if clamp is not None:
                x = torch.clamp(x, clamp[0], clamp[1])

        self.train()
        return x

    @torch.no_grad()
    def denoise(self, x_noisy, sigma_t):
        t = torch.tensor(sigma_t)
        score, info = self.forward(x_noisy, t)
        x_denoised = x_noisy + (sigma_t ** 2) * score
        return x_denoised, info

    def get_gram_matrix(self):
        W = list(self.encoder.parameters())[0]
        return W @ W.T

    # ==================================================================
    # Full pipeline
    # ==================================================================

    def full_training(self, dataset, stage1_steps=500,
                      stage2_steps=3000, **kwargs):
        print("=" * 60)
        print("Two-Stage Manifold Diffusion Training")
        print(f"  d={self.d}, latent_dim={self.latent_dim}")
        print(f"  Stage 1: reconstruction at t0={self.t0} (small noise)")
        print(f"  Stage 2: manifold t∈[{self.t_min},{self.t_max}] + vanilla t∈[{self.t_max},{self.T}]")
        print(f"  Stage 2 input dim: {self.latent_dim} (manifold) / {self.d} (vanilla)")
        print("=" * 60)

        # Stage 1
        print("\n--- Stage 1: Manifold Learning (small t0) ---")
        stage1_losses = self.train_stage1(
            dataset, n_steps=stage1_steps,
            lr=kwargs.get('lr1', 1e-3), batch_size=kwargs.get('batch_size', 4096)
        )

        with torch.no_grad():
            x0_test, _ = dataset.sample_clean(5000)
            z_test, x_hat_test = self.project(x0_test)
            recon_mse = F.mse_loss(x_hat_test, x0_test).item()
            z_var = z_test.var(dim=0).sort(descending=True).values
            active_dims = (z_var > z_var[0] * 0.01).sum().item()
            print(f"\n  Reconstruction MSE: {recon_mse:.6f}")
            print(f"  Active latent dims: {active_dims}/{self.latent_dim}")
            print(f"  Compression: {self.d}d → {self.latent_dim}d ({self.d/self.latent_dim:.1f}x)")
            eigenvalues = z_var.numpy()

        self.freeze_stage1()
        print("  Stage 1 frozen.\n")

        # Stage 2
        print("--- Stage 2: Two-Regime Score Learning ---")
        stage2_losses = self.train_stage2(
            dataset, n_steps=stage2_steps,
            lr=kwargs.get('lr2', 1e-3), batch_size=kwargs.get('batch_size', 4096)
        )

        # Evaluation
        print("\n--- Final Evaluation ---")
        eval_times = [self.t_min, (self.t_min + self.t_max) / 2, self.t_max,
                      (self.t_max + self.T) / 2]
        total_loss = 0.0
        for t_eval in eval_times:
            with torch.no_grad():
                x0, _ = dataset.sample_clean(5000)
                t_batch = torch.full((5000,), t_eval)
                x_t, eps = self.forward_diffusion(x0, t_batch)
                eps_pred, info = self.predict_eps(x_t, t_batch)
                eps_mse = F.mse_loss(eps_pred, eps).item()
                regime = "manifold" if t_eval <= self.t_max else "vanilla"
                total_loss += eps_mse
                print(f"  t={t_eval:.3f} [{regime}]: eps_MSE={eps_mse:.6f}")

        avg_loss = total_loss / len(eval_times)
        print(f"  Average eps MSE: {avg_loss:.6f}")

        return {
            'stage1_losses': stage1_losses,
            'stage2_losses': stage2_losses,
            'total_loss': avg_loss,
            'recon_error': recon_mse,
            'eigenvalues': eigenvalues,
        }
