"""
Two-Stage Manifold Diffusion Model (Latent Diffusion Variant).

Same architecture as LDM — diffusion runs entirely in latent space z ∈ R^k.
The ONLY difference: how the autoencoder is trained.

  LDM:  autoencoder trained with reconstruction + KL (heuristic)
  Ours: autoencoder trained via denoising at small t0 (theory-driven)
        The 1/h(t0)^2 singularity drives dimensional collapse (Theorem 3.4).

Stage 1: Encoder-Decoder at small t0.
  - Encoder: x ∈ R^d → z ∈ R^k  (only sees clean/near-clean data)
  - Decoder: z ∈ R^k → x_hat ∈ R^d
  - Bottleneck k << d IS the dimensional collapse.

Stage 2: Standard latent diffusion with ONE score network in R^k.
  - z_t = sqrt(alpha_bar) * z_0 + sigma_t * eps_z
  - Score network: (z_t, t) → eps_z  (predicts noise in latent space)
  - ONE network for ALL times t ∈ [0, T].

Sampling:
  - Start from z_T ~ N(0, I) in R^k
  - Reverse diffuse entirely in R^k (encoder NEVER sees noisy data)
  - Decode once at the end: x = Decoder(z_final)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# ========================================================================
# Encoder / Decoder
# ========================================================================

class ManifoldEncoder(nn.Module):
    """MLP encoder for vector data."""
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
    """MLP decoder for vector data."""
    def __init__(self, latent_dim, d, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, d),
        )
    def forward(self, z):
        return self.net(z)


class ResBlock2d(nn.Module):
    """Residual block with optional channel change."""
    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        out_ch = out_ch or in_ch
        self.net = nn.Sequential(
            nn.GroupNorm(min(32, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.skip(x) + self.net(x)


class ConvEncoder(nn.Module):
    """Deep CNN encoder with multiple ResBlocks per level."""
    def __init__(self, image_shape, latent_dim, base_ch=64):
        super().__init__()
        C, H, W = image_shape
        self.image_shape = image_shape

        ch = base_ch
        layers = [nn.Conv2d(C, ch, 3, padding=1)]

        # Level 1: ch -> ch, then downsample
        layers += [ResBlock2d(ch), ResBlock2d(ch)]
        layers += [nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)]
        ch *= 2

        # Level 2: 2*ch -> 2*ch, then downsample
        layers += [ResBlock2d(ch), ResBlock2d(ch)]
        layers += [nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)]
        ch *= 2

        # Level 3: 4*ch -> 4*ch, then downsample
        layers += [ResBlock2d(ch), ResBlock2d(ch)]
        layers += [nn.Conv2d(ch, ch, 3, stride=2, padding=1)]

        # Final
        layers += [ResBlock2d(ch), nn.GroupNorm(min(32, ch), ch), nn.SiLU()]

        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            self._flat_size = self.conv(torch.zeros(1, C, H, W)).view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(self._flat_size, 512), nn.SiLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        x_img = x.view(-1, *self.image_shape)
        return self.fc(self.conv(x_img).view(x.size(0), -1))


class ConvDecoder(nn.Module):
    """Deep CNN decoder mirroring ConvEncoder."""
    def __init__(self, latent_dim, image_shape, base_ch=64):
        super().__init__()
        C, H, W = image_shape
        self.image_shape = image_shape
        self.d = C * H * W
        ch = base_ch * 4

        # Compute start spatial size (3 stride-2 convs)
        h, w = H, W
        for _ in range(3):
            h = (h + 1) // 2
            w = (w + 1) // 2
        self.start_h, self.start_w, self.start_ch = h, w, ch

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.SiLU(),
            nn.Linear(512, ch * h * w), nn.SiLU(),
        )

        # Level 3: upsample
        self.up3 = nn.Sequential(
            ResBlock2d(ch), ResBlock2d(ch),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch, ch // 2, 3, padding=1),
        )
        ch //= 2

        # Level 2: upsample
        self.up2 = nn.Sequential(
            ResBlock2d(ch), ResBlock2d(ch),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch, ch // 2, 3, padding=1),
        )
        ch //= 2

        # Level 1: upsample
        self.up1 = nn.Sequential(
            ResBlock2d(ch), ResBlock2d(ch),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

        # Final output
        self.out = nn.Sequential(
            ResBlock2d(ch),
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, C, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, self.start_ch, self.start_h, self.start_w)
        h = self.up3(h)
        h = self.up2(h)
        h = self.up1(h)
        h = self.out(h)
        # Crop to exact target size
        img = h[:, :, :self.image_shape[1], :self.image_shape[2]]
        return img.reshape(-1, self.d)


# ========================================================================
# Latent score network (single network for ALL t)
# ========================================================================

class LatentScoreNet(nn.Module):
    """
    Score network in latent space R^k.
    Input:  (z_t, t) where z_t ∈ R^k
    Output: eps_z ∈ R^k (predicted noise)
    Uses AdaGN-style time conditioning with residual blocks.
    """
    def __init__(self, latent_dim, hidden_dim=256, n_layers=4, time_embed_dim=128):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList()
        self.ada_lin = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
            self.ada_lin.append(nn.Linear(hidden_dim, hidden_dim * 2))
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def sinusoidal_embedding(self, t):
        half_dim = self.time_embed_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=t.device, dtype=torch.float32)
            / half_dim
        )
        if t.dim() == 0:
            t = t.unsqueeze(0)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, z_t, t):
        t_feat = self.time_embed(self.sinusoidal_embedding(t))
        h = self.input_proj(z_t)
        for block, ada in zip(self.blocks, self.ada_lin):
            scale, shift = ada(t_feat).chunk(2, dim=-1)
            h = h + block(h) * (1 + scale) + shift
        return self.output(h)


# ========================================================================
# Main model
# ========================================================================

class TwoStageManifoldDiffusion(nn.Module):
    """
    Two-Stage Manifold Diffusion = Latent Diffusion with
    score-matching-trained autoencoder.

    Identical architecture to LDM. Only difference is Stage 1 training.
    """

    def __init__(self, d, latent_dim=32, hidden_dim_enc=512, hidden_dim_score=256,
                 n_layers=4, t0=0.005, image_shape=None, base_ch=64,
                 diffusion_t_min=None):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim
        self.t0 = t0
        self.T = 0.999
        if diffusion_t_min is None:
            diffusion_t_min = t0
        if not 0.0 < diffusion_t_min < self.T:
            raise ValueError(
                f"diffusion_t_min must be in (0, {self.T}), got {diffusion_t_min}"
            )
        self.diffusion_t_min = float(diffusion_t_min)
        self.image_shape = image_shape

        # --- Stage 1: Encoder-Decoder ---
        if image_shape is not None:
            self.encoder = ConvEncoder(image_shape, latent_dim, base_ch=base_ch)
            self.decoder = ConvDecoder(latent_dim, image_shape, base_ch=base_ch)
        else:
            self.encoder = ManifoldEncoder(d, latent_dim, hidden_dim=hidden_dim_enc)
            self.decoder = ManifoldDecoder(latent_dim, d, hidden_dim=hidden_dim_enc)
        self.stage1_frozen = False

        # --- Stage 2: ONE score network in R^k ---
        self.score_net = LatentScoreNet(
            latent_dim, hidden_dim=hidden_dim_score, n_layers=n_layers,
        )

        # VP-SDE noise schedule
        self.beta_min = 0.1
        self.beta_max = 20.0

        # Latent normalization (computed after Stage 1)
        self.register_buffer('z_mean', torch.zeros(latent_dim))
        self.register_buffer('z_std', torch.ones(latent_dim))

    # ------------------------------------------------------------------
    # Noise schedule
    # ------------------------------------------------------------------

    def noise_schedule(self, t):
        t_tensor = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
        log_mean_coeff = -0.25 * t_tensor ** 2 * (self.beta_max - self.beta_min) \
                         - 0.5 * t_tensor * self.beta_min
        alpha_bar = torch.exp(2.0 * log_mean_coeff)
        sigma = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-8))
        return alpha_bar, sigma

    # ------------------------------------------------------------------
    # Encode / decode with normalization
    # ------------------------------------------------------------------

    def encode(self, x):
        """Encode clean x to normalized latent z."""
        z = self.encoder(x)
        return (z - self.z_mean) / self.z_std.clamp(min=1e-6)

    def decode(self, z_norm):
        """Decode normalized latent z to ambient x."""
        z = z_norm * self.z_std + self.z_mean
        return self.decoder(z)

    def project(self, x):
        """Encode then decode (for evaluation)."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    # ------------------------------------------------------------------
    # Forward diffusion (in latent space)
    # ------------------------------------------------------------------

    def forward_diffusion_latent(self, z0, t):
        """Add noise in latent space: z_t = sqrt(a) * z0 + sigma * eps."""
        alpha_bar, sigma = self.noise_schedule(t)
        eps_z = torch.randn_like(z0)
        if alpha_bar.dim() > 0:
            alpha_bar = alpha_bar.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        z_t = torch.sqrt(alpha_bar) * z0 + sigma * eps_z
        return z_t, eps_z

    def forward_diffusion(self, x0, t):
        """Add noise in ambient space (for compatibility / evaluation)."""
        alpha_bar, sigma = self.noise_schedule(t)
        eps = torch.randn_like(x0)
        if alpha_bar.dim() > 0:
            alpha_bar = alpha_bar.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        x_t = torch.sqrt(alpha_bar) * x0 + sigma * eps
        return x_t, eps

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_eps_latent(self, z_t, t):
        """Predict latent noise: eps_z_pred = score_net(z_t, t)."""
        return self.score_net(z_t, t)

    def predict_eps(self, x_t, t):
        """
        Predict ambient noise (for evaluation compatibility).
        Encodes x_t, predicts latent eps, converts to ambient eps.
        """
        z_t = self.encode(x_t)
        t_tensor = t if isinstance(t, torch.Tensor) else torch.tensor(t)
        if t_tensor.dim() == 0:
            t_tensor = t_tensor.expand(x_t.shape[0])

        eps_z = self.predict_eps_latent(z_t, t_tensor)

        alpha_bar, sigma = self.noise_schedule(t)
        if isinstance(sigma, torch.Tensor) and sigma.dim() > 0:
            sigma_e = sigma.unsqueeze(-1)
            alpha_e = alpha_bar.unsqueeze(-1)
        else:
            sigma_e = sigma
            alpha_e = alpha_bar

        # z0_pred from latent eps
        z0_pred = (z_t - sigma_e * eps_z) / torch.sqrt(alpha_e).clamp(min=1e-6)
        x0_pred = self.decode(z0_pred)

        # Ambient eps from x0_pred
        eps_pred = (x_t - torch.sqrt(alpha_e) * x0_pred) / sigma_e.clamp(min=1e-6)

        info = {'x_hat': x0_pred, 'z': z_t, 'eps_z': eps_z}
        return eps_pred, info

    def forward(self, x, t):
        """Score prediction (for compatibility)."""
        eps_pred, info = self.predict_eps(x, t)
        _, sigma = self.noise_schedule(t)
        if isinstance(sigma, torch.Tensor) and sigma.dim() > 0:
            sigma = sigma.unsqueeze(-1)
        score = -eps_pred / sigma.clamp(min=1e-6)
        return score, info

    # ------------------------------------------------------------------
    # Stage 1 & 2 training
    # ------------------------------------------------------------------

    def freeze_stage1(self):
        self.stage1_frozen = True
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    def train_stage1(self, dataset, n_steps=500, lr=1e-3, batch_size=4096):
        """
        Stage 1: denoising autoencoder at small t0.

        At small noise, score matching reduces to reconstruction:
          ||x_hat - x0||^2 / sigma_t0^2
        The bottleneck forces dimensional collapse (Theorem 3.4).
        """
        sigma_t0 = self.t0
        device = next(self.parameters()).device
        print(f"  Stage 1: t0={self.t0}, sigma_t0={sigma_t0:.4f}, "
              f"latent_dim={self.latent_dim}, device={device}")

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

        losses = []
        for step in range(n_steps):
            x0, _ = dataset.sample_clean(batch_size)
            x0 = x0.to(device)
            # Small noise perturbation
            x_t = x0 + sigma_t0 * torch.randn_like(x0)
            z, x_hat = self.project(x_t)
            loss = F.mse_loss(x_hat, x0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            if (step + 1) % 50 == 0:
                print(f"  Stage 1 [{step+1}/{n_steps}] recon_loss={loss.item():.6f}")

        # Compute latent normalization statistics
        with torch.no_grad():
            zs = []
            for _ in range(10):
                x0, _ = dataset.sample_clean(min(batch_size, 5000))
                x0 = x0.to(device)
                zs.append(self.encoder(x0))
            zs = torch.cat(zs, dim=0)
            self.z_mean = zs.mean(dim=0)
            self.z_std = zs.std(dim=0).clamp(min=1e-6)
            print(f"  Latent stats: mean_norm={self.z_mean.norm():.4f}, "
                  f"std=[{self.z_std.min():.4f}, {self.z_std.max():.4f}]")

        return losses

    def train_stage2(self, dataset, n_steps=3000, lr=1e-3, batch_size=4096,
                     min_snr_gamma=None):
        """
        Stage 2: standard latent diffusion.

        Encode clean data → z0 (frozen encoder).
        Diffuse in latent space: z_t = sqrt(a)*z0 + sigma*eps_z.
        Train score_net to predict eps_z from (z_t, t).
        ONE network, ALL times.

        If ``min_snr_gamma`` is not None, apply Min-SNR-gamma loss
        weighting (Hang et al. 2023): each per-sample loss is weighted
        by min(SNR(t), gamma) / SNR(t). Downweights very-high-noise t
        (where unweighted MSE saturates at ~1) and gives the score net
        a fair gradient signal at small t. Pass gamma=5 as a sensible
        default if you want it. Default None preserves the original
        uniform-weighted behavior.
        """
        optimizer = torch.optim.Adam(self.score_net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

        device = next(self.parameters()).device
        print(
            f"  Stage 2: t in [{self.diffusion_t_min}, {self.T}], "
            f"min_snr_gamma={min_snr_gamma}"
        )
        losses = []
        for step in range(n_steps):
            x0, _ = dataset.sample_clean(batch_size)
            x0 = x0.to(device)

            # Encode clean data (frozen)
            with torch.no_grad():
                z0 = self.encode(x0)

            # Stage-2 lower bound is configurable independently of the
            # Stage-1 geometric noise level t0 for controlled comparisons.
            t = (
                torch.rand(z0.shape[0], device=device)
                * (self.T - self.diffusion_t_min)
                + self.diffusion_t_min
            )
            z_t, eps_z = self.forward_diffusion_latent(z0, t)

            # Predict noise
            eps_z_pred = self.predict_eps_latent(z_t, t)

            if min_snr_gamma is None:
                loss = F.mse_loss(eps_z_pred, eps_z)
            else:
                alpha_bar, sigma = self.noise_schedule(t)
                snr = (alpha_bar / sigma.pow(2).clamp(min=1e-8)).clamp(min=1e-8)
                gamma = torch.full_like(snr, float(min_snr_gamma))
                w = torch.minimum(snr, gamma) / snr   # (B,)
                w = w.unsqueeze(-1)                   # (B, 1) — broadcasts over latent dim
                loss = (w * (eps_z_pred - eps_z).pow(2)).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.score_net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            if (step + 1) % 200 == 0:
                tag = 'weighted_loss' if min_snr_gamma is not None else 'latent_eps_loss'
                print(f"  Stage 2 [{step+1}/{n_steps}] {tag}={loss.item():.6f}")

        return losses

    # ------------------------------------------------------------------
    # Sampling (all in latent space)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, n_samples, n_steps=200, device='cpu', clamp=None):
        """
        DDIM sampling in latent space, then decode once.
        Encoder is NEVER applied to noisy data.

        Reverse diffusion is integrated on the Stage-2 training interval.
        The final step takes z0_pred directly (Tweedie) so we hand the
        decoder a clean latent rather than a re-noised one.
        """
        self.eval()
        ts = torch.linspace(
            self.T, self.diffusion_t_min, n_steps + 1, device=device
        )

        # Start from noise in latent space R^k
        z = torch.randn(n_samples, self.latent_dim, device=device)

        for i in range(n_steps):
            t_cur, t_next = ts[i], ts[i + 1]
            t_batch = torch.full((n_samples,), t_cur.item(), device=device)

            alpha_cur, sigma_cur = self.noise_schedule(t_cur)
            alpha_next, sigma_next = self.noise_schedule(t_next)

            eps_z_pred = self.predict_eps_latent(z, t_batch)

            # DDIM step in latent space
            z0_pred = (z - sigma_cur * eps_z_pred) / torch.sqrt(alpha_cur).clamp(min=1e-6)
            if i == n_steps - 1:
                # Final step: hand the decoder z0_pred directly so we don't
                # re-noise back up to the Stage-2 lower bound.
                z = z0_pred
            else:
                z = torch.sqrt(alpha_next) * z0_pred + sigma_next * eps_z_pred

        # Decode final latent → ambient
        x = self.decode(z)
        if clamp is not None:
            x = torch.clamp(x, clamp[0], clamp[1])

        self.train()
        return x

    @torch.no_grad()
    def sample_predictor_corrector(self, n_samples, n_steps=200,
                                    n_corrector=2, snr=0.1,
                                    device='cpu', clamp=None):
        """Predictor-Corrector sampling on the Stage-2 training interval."""
        self.eval()
        ts = torch.linspace(
            self.T, self.diffusion_t_min, n_steps + 1, device=device
        )
        dt = (self.T - self.diffusion_t_min) / n_steps

        z = torch.randn(n_samples, self.latent_dim, device=device)

        for i in range(n_steps):
            t_cur = ts[i]
            t_batch = torch.full((n_samples,), t_cur.item(), device=device)

            # Predictor (Euler-Maruyama on reverse SDE)
            eps_z_pred = self.predict_eps_latent(z, t_batch)
            _, sigma_cur = self.noise_schedule(t_cur)
            score_z = -eps_z_pred / sigma_cur.clamp(min=1e-6)

            beta_t = self.beta_min + (self.beta_max - self.beta_min) * t_cur.item()
            drift = -0.5 * beta_t * z - beta_t * score_z
            z = z - drift * dt
            if i < n_steps - 1:
                z = z + math.sqrt(beta_t * dt) * torch.randn_like(z)

            # Corrector (Langevin MCMC)
            for _ in range(n_corrector):
                eps_z_pred = self.predict_eps_latent(z, t_batch)
                score_z = -eps_z_pred / sigma_cur.clamp(min=1e-6)
                grad_norm = torch.mean(torch.norm(score_z, dim=-1))
                noise_norm = math.sqrt(self.latent_dim)
                step_size = (snr * noise_norm / (grad_norm + 1e-8)) ** 2 * 2
                z = z + step_size * score_z + math.sqrt(2 * step_size) * torch.randn_like(z)

        # Final Tweedie denoise to z0 from z_{t_min}
        t_final = torch.full(
            (n_samples,), self.diffusion_t_min, device=device
        )
        alpha_t0, sigma_t0 = self.noise_schedule(t_final)
        eps_z_pred = self.predict_eps_latent(z, t_final)
        z = (z - sigma_t0.unsqueeze(-1) * eps_z_pred) / torch.sqrt(alpha_t0).unsqueeze(-1).clamp(min=1e-6)

        x = self.decode(z)
        if clamp is not None:
            x = torch.clamp(x, clamp[0], clamp[1])

        self.train()
        return x

    @torch.no_grad()
    def denoise(self, x_noisy, sigma_t):
        """Denoise via encoder-decoder projection."""
        _, x_hat = self.project(x_noisy)
        return x_hat, {'x_hat': x_hat}

    def get_gram_matrix(self):
        W = list(self.encoder.parameters())[0]
        return W @ W.T

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def full_training(self, dataset, stage1_steps=500,
                      stage2_steps=3000, **kwargs):
        print("=" * 60)
        print("Two-Stage Manifold Diffusion (Latent)")
        print(f"  d={self.d}, latent_dim={self.latent_dim}")
        print(f"  Stage 1: denoising AE at t0={self.t0}")
        print(
            f"  Stage 2: latent diffusion in R^{self.latent_dim}, "
            f"t in [{self.diffusion_t_min}, {self.T}]"
        )
        print("=" * 60)

        # Stage 1
        print("\n--- Stage 1: Manifold Learning (small t0) ---")
        stage1_losses = self.train_stage1(
            dataset, n_steps=stage1_steps,
            lr=kwargs.get('lr1', 1e-3),
            batch_size=kwargs.get('batch_size', 4096),
        )

        with torch.no_grad():
            x0_test, _ = dataset.sample_clean(5000)
            z_test, x_hat_test = self.project(x0_test)
            recon_mse = F.mse_loss(x_hat_test, x0_test).item()
            z_norm = self.encode(x0_test)
            z_var = z_norm.var(dim=0).sort(descending=True).values
            active_dims = (z_var > z_var[0] * 0.01).sum().item()
            print(f"\n  Reconstruction MSE: {recon_mse:.6f}")
            print(f"  Active latent dims: {active_dims}/{self.latent_dim}")
            print(f"  Compression: {self.d}d -> {self.latent_dim}d "
                  f"({self.d / self.latent_dim:.1f}x)")
            eigenvalues = z_var.numpy()

        self.freeze_stage1()
        print("  Stage 1 frozen.\n")

        # Stage 2
        print("--- Stage 2: Latent Diffusion ---")
        stage2_losses = self.train_stage2(
            dataset, n_steps=stage2_steps,
            lr=kwargs.get('lr2', 1e-3),
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
                eps_z_pred = self.predict_eps_latent(z_t, t_batch)
                mse = F.mse_loss(eps_z_pred, eps_z).item()
                total_loss += mse
                print(f"  t={t_eval:.2f}: latent_eps_MSE={mse:.6f}")

        avg_loss = total_loss / len(eval_times)
        print(f"  Average latent eps MSE: {avg_loss:.6f}")

        return {
            'stage1_losses': stage1_losses,
            'stage2_losses': stage2_losses,
            'total_loss': avg_loss,
            'recon_error': recon_mse,
            'eigenvalues': eigenvalues,
        }
