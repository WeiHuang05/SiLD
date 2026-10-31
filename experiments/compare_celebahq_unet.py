"""
Experiment: CelebA-HQ (64x64x3) — Spatial Latent + U-Net Score Network

Key difference from previous experiments:
  - Encoder outputs a SPATIAL latent (8x8xC) instead of flat vector
  - Score network is a U-Net operating on spatial latent, not an MLP
  - This preserves spatial structure through the diffusion process

Architecture:
  64x64x3 → Encoder → 8x8x4 (spatial latent) → U-Net denoiser → Decoder → 64x64x3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.image_data import CelebAHQDataset

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available")

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: torchmetrics not available")

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# ================================================================
# Building blocks
# ================================================================

class ResBlock2d(nn.Module):
    def __init__(self, ch, out_ch=None):
        super().__init__()
        out_ch = out_ch or ch
        self.net = nn.Sequential(
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(ch, out_ch, 1) if ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.skip(x) + self.net(x)


class TimeCondResBlock(nn.Module):
    """ResBlock with time embedding injection via AdaGN."""
    def __init__(self, ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, ch), ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, ch), ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, ch * 2)  # scale + shift
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        # Time conditioning via scale and shift
        scale, shift = self.t_proj(t_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)
        return x + h


# ================================================================
# Spatial Encoder / Decoder (no FC bottleneck)
# ================================================================

class SpatialEncoder(nn.Module):
    """
    Encodes 64x64x3 → 8x8 x latent_ch spatial latent.
    3 downsample levels, then 1x1 conv to latent channels.
    """
    def __init__(self, image_shape, latent_ch=4, base_ch=128):
        super().__init__()
        C, H, W = image_shape
        self.image_shape = image_shape

        ch = base_ch
        layers = [nn.Conv2d(C, ch, 3, padding=1)]

        # Level 1: 64x64 -> 32x32
        layers += [ResBlock2d(ch), ResBlock2d(ch)]
        layers += [nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)]
        ch *= 2

        # Level 2: 32x32 -> 16x16
        layers += [ResBlock2d(ch), ResBlock2d(ch)]
        layers += [nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)]
        ch *= 2

        # Level 3: 16x16 -> 8x8
        layers += [ResBlock2d(ch), ResBlock2d(ch)]
        layers += [nn.Conv2d(ch, ch, 3, stride=2, padding=1)]

        # Final processing + project to latent channels
        layers += [
            ResBlock2d(ch),
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, latent_ch, 1),  # 1x1 conv to latent_ch
        ]

        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            out = self.conv(torch.zeros(1, C, H, W))
            self.spatial_shape = out.shape[1:]  # (latent_ch, h, w)
            print(f"    SpatialEncoder: output shape = {list(self.spatial_shape)}, "
                  f"total latent dims = {out.numel() // 1}")

    def forward(self, x):
        """Returns spatial latent (B, latent_ch, h, w)."""
        x_img = x.view(-1, *self.image_shape)
        return self.conv(x_img)


class SpatialDecoder(nn.Module):
    """
    Decodes 8x8 x latent_ch → 64x64x3.
    """
    def __init__(self, latent_ch, image_shape, base_ch=128):
        super().__init__()
        C, H, W = image_shape
        self.image_shape = image_shape
        self.d = C * H * W

        ch = base_ch * 4  # match encoder's deepest level

        # Project from latent channels
        self.proj_in = nn.Sequential(
            nn.Conv2d(latent_ch, ch, 1),
            nn.SiLU(),
        )

        # Level 3: 8x8 -> 16x16
        self.up3 = nn.Sequential(
            ResBlock2d(ch), ResBlock2d(ch),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch, ch // 2, 3, padding=1),
        )
        ch //= 2

        # Level 2: 16x16 -> 32x32
        self.up2 = nn.Sequential(
            ResBlock2d(ch), ResBlock2d(ch),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch, ch // 2, 3, padding=1),
        )
        ch //= 2

        # Level 1: 32x32 -> 64x64
        self.up1 = nn.Sequential(
            ResBlock2d(ch), ResBlock2d(ch),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

        self.out = nn.Sequential(
            ResBlock2d(ch),
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, C, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z_spatial):
        """Takes spatial latent (B, latent_ch, h, w), returns flat (B, d)."""
        h = self.proj_in(z_spatial)
        h = self.up3(h)
        h = self.up2(h)
        h = self.up1(h)
        h = self.out(h)
        img = h[:, :, :self.image_shape[1], :self.image_shape[2]]
        return img.reshape(-1, self.d)


class SpatialVAEEncoder(nn.Module):
    """Spatial encoder that outputs mu and logvar maps."""
    def __init__(self, image_shape, latent_ch=4, base_ch=128):
        super().__init__()
        C, H, W = image_shape
        self.image_shape = image_shape

        ch = base_ch
        layers = [nn.Conv2d(C, ch, 3, padding=1)]

        layers += [ResBlock2d(ch), ResBlock2d(ch)]
        layers += [nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)]
        ch *= 2

        layers += [ResBlock2d(ch), ResBlock2d(ch)]
        layers += [nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)]
        ch *= 2

        layers += [ResBlock2d(ch), ResBlock2d(ch)]
        layers += [nn.Conv2d(ch, ch, 3, stride=2, padding=1)]

        layers += [
            ResBlock2d(ch),
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
        ]
        self.conv = nn.Sequential(*layers)
        self.conv_mu = nn.Conv2d(ch, latent_ch, 1)
        self.conv_logvar = nn.Conv2d(ch, latent_ch, 1)

    def forward(self, x):
        x_img = x.view(-1, *self.image_shape)
        h = self.conv(x_img)
        return self.conv_mu(h), self.conv_logvar(h)

    def encode_deterministic(self, x):
        mu, _ = self.forward(x)
        return mu


# ================================================================
# U-Net Score Network for spatial latents
# ================================================================

class UNetScoreNet(nn.Module):
    """
    Small U-Net for denoising spatial latents.
    Input: (z_t, t) where z_t is (B, latent_ch, H, W)
    Output: predicted noise eps of same shape.

    For 8x8 latent: 8->4->2 (2 downsample levels).
    """
    def __init__(self, latent_ch=4, base_ch=128, time_embed_dim=256):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, base_ch * 4),
            nn.SiLU(),
            nn.Linear(base_ch * 4, base_ch * 4),
        )
        t_dim = base_ch * 4

        # Input conv
        self.conv_in = nn.Conv2d(latent_ch, base_ch, 3, padding=1)

        # Down path: 8x8 -> 4x4 -> 2x2
        self.down1_res1 = TimeCondResBlock(base_ch, t_dim)
        self.down1_res2 = TimeCondResBlock(base_ch, t_dim)
        self.down1_conv = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)

        self.down2_res1 = TimeCondResBlock(base_ch * 2, t_dim)
        self.down2_res2 = TimeCondResBlock(base_ch * 2, t_dim)
        self.down2_conv = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)

        # Middle (2x2)
        self.mid_res1 = TimeCondResBlock(base_ch * 4, t_dim)
        self.mid_res2 = TimeCondResBlock(base_ch * 4, t_dim)

        # Up path: 2x2 -> 4x4 -> 8x8
        self.up2_conv = nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1)
        self.up2_res1 = TimeCondResBlock(base_ch * 4, t_dim)  # concat skip
        self.up2_res2 = TimeCondResBlock(base_ch * 4, t_dim)
        self.up2_out = nn.Conv2d(base_ch * 4, base_ch * 2, 1)

        self.up1_conv = nn.Conv2d(base_ch * 2, base_ch, 3, padding=1)
        self.up1_res1 = TimeCondResBlock(base_ch * 2, t_dim)  # concat skip
        self.up1_res2 = TimeCondResBlock(base_ch * 2, t_dim)
        self.up1_out = nn.Conv2d(base_ch * 2, base_ch, 1)

        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(min(32, base_ch), base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, latent_ch, 3, padding=1),
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
        t_emb = self.time_mlp(self.sinusoidal_embedding(t))

        # Input
        h = self.conv_in(z_t)

        # Down 1: 8x8
        h1 = self.down1_res1(h, t_emb)
        h1 = self.down1_res2(h1, t_emb)
        h = self.down1_conv(h1)  # -> 4x4

        # Down 2: 4x4
        h2 = self.down2_res1(h, t_emb)
        h2 = self.down2_res2(h2, t_emb)
        h = self.down2_conv(h2)  # -> 2x2

        # Middle: 2x2
        h = self.mid_res1(h, t_emb)
        h = self.mid_res2(h, t_emb)

        # Up 2: 2x2 -> 4x4
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.up2_conv(h)
        h = torch.cat([h, h2], dim=1)  # skip connection
        h = self.up2_res1(h, t_emb)
        h = self.up2_res2(h, t_emb)
        h = self.up2_out(h)

        # Up 1: 4x4 -> 8x8
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.up1_conv(h)
        h = torch.cat([h, h1], dim=1)  # skip connection
        h = self.up1_res1(h, t_emb)
        h = self.up1_res2(h, t_emb)
        h = self.up1_out(h)

        return self.conv_out(h)


# ================================================================
# PatchGAN + EMA + helpers
# ================================================================

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_ch=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(in_channels, base_ch, 4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, inplace=True)]
        ch = base_ch
        for i in range(1, n_layers):
            ch_next = min(ch * 2, 512)
            layers += [nn.Conv2d(ch, ch_next, 4, stride=2, padding=1, bias=False),
                       nn.InstanceNorm2d(ch_next), nn.LeakyReLU(0.2, inplace=True)]
            ch = ch_next
        layers += [nn.Conv2d(ch, ch, 4, stride=1, padding=1, bias=False),
                   nn.InstanceNorm2d(ch), nn.LeakyReLU(0.2, inplace=True),
                   nn.Conv2d(ch, 1, 4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def hinge_loss_d(real_pred, fake_pred):
    return (F.relu(1.0 - real_pred).mean() + F.relu(1.0 + fake_pred).mean()) * 0.5

def hinge_loss_g(fake_pred):
    return -fake_pred.mean()


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1.0 - self.decay)
        for s, m in zip(self.shadow.buffers(), model.buffers()):
            s.data.copy_(m.data)

    def apply(self, model):
        model.load_state_dict(self.shadow.state_dict())


def save_checkpoint(model, name, extra=None):
    path = os.path.join(CKPT_DIR, f'{name}.pt')
    state = {'model': model.state_dict()}
    if extra:
        state.update(extra)
    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(model, name, device='cpu'):
    path = os.path.join(CKPT_DIR, f'{name}.pt')
    if os.path.exists(path):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state['model'])
        print(f"  Checkpoint loaded: {path}")
        return state
    return None


def plot_image_grid(images, nrow=8, title='', save_name=None, img_size=64):
    n = min(len(images), nrow * nrow)
    ncol = nrow
    nrow_actual = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow_actual, ncol,
                             figsize=(ncol * 1.5, nrow_actual * 1.5))
    if nrow_actual == 1:
        axes = axes.reshape(1, -1)
    for i in range(nrow_actual):
        for j in range(ncol):
            idx = i * ncol + j
            ax = axes[i, j]
            if idx < n:
                img = images[idx].detach().cpu().clamp(0, 1)
                img = img.view(3, img_size, img_size).permute(1, 2, 0).numpy()
                ax.imshow(img)
            ax.axis('off')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_name:
        path = os.path.join(FIG_DIR, save_name)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")
    else:
        plt.close()


# ================================================================
# Two-Stage model with spatial latent + U-Net score
# ================================================================

class TwoStageSpatial(nn.Module):
    def __init__(self, d, image_shape, latent_ch=4, base_ch=128,
                 unet_base_ch=128, t0=0.005):
        super().__init__()
        self.d = d
        self.t0 = t0
        self.T = 0.999
        self.image_shape = image_shape

        self.encoder = SpatialEncoder(image_shape, latent_ch, base_ch)
        self.decoder = SpatialDecoder(latent_ch, image_shape, base_ch)
        self.score_net = UNetScoreNet(latent_ch, base_ch=unet_base_ch)

        self.latent_ch = latent_ch
        self.spatial_shape = self.encoder.spatial_shape  # (latent_ch, h, w)
        self.latent_dim = latent_ch * self.spatial_shape[1] * self.spatial_shape[2]

        self.beta_min = 0.1
        self.beta_max = 20.0

        # Per-channel normalization stats
        self.register_buffer('z_mean', torch.zeros(latent_ch, 1, 1))
        self.register_buffer('z_std', torch.ones(latent_ch, 1, 1))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"    TwoStageSpatial: {n_params:,} params, "
              f"latent={list(self.spatial_shape)}")

    def noise_schedule(self, t):
        t_tensor = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
        log_mean_coeff = -0.25 * t_tensor**2 * (self.beta_max - self.beta_min) \
                         - 0.5 * t_tensor * self.beta_min
        alpha_bar = torch.exp(2.0 * log_mean_coeff)
        sigma = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-8))
        return alpha_bar, sigma

    def encode(self, x):
        z = self.encoder(x)  # (B, C, H, W)
        return (z - self.z_mean) / self.z_std.clamp(min=1e-6)

    def decode(self, z_norm):
        z = z_norm * self.z_std + self.z_mean
        return self.decoder(z)

    def project(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def freeze_stage1(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    def forward_diffusion(self, z0, t):
        alpha_bar, sigma = self.noise_schedule(t)
        eps = torch.randn_like(z0)
        ab = alpha_bar.view(-1, 1, 1, 1)
        sig = sigma.view(-1, 1, 1, 1)
        z_t = torch.sqrt(ab) * z0 + sig * eps
        return z_t, eps

    @torch.no_grad()
    def sample(self, n, n_steps=500, device='cpu', clamp=None):
        _, h, w = self.spatial_shape
        z = torch.randn(n, self.latent_ch, h, w, device=device)
        dt = self.T / n_steps
        for i in range(n_steps, 0, -1):
            t = torch.full((n,), i * dt, device=device)
            alpha_bar, sigma = self.noise_schedule(t)
            eps_pred = self.score_net(z, t)
            ab = alpha_bar.view(-1, 1, 1, 1)
            sig = sigma.view(-1, 1, 1, 1)
            x0_pred = (z - sig * eps_pred) / torch.sqrt(ab).clamp(min=1e-8)
            alpha_bar_prev, sigma_prev = self.noise_schedule(t - dt)
            ab_p = alpha_bar_prev.view(-1, 1, 1, 1)
            sig_p = sigma_prev.view(-1, 1, 1, 1)
            z = torch.sqrt(ab_p) * x0_pred + sig_p * torch.randn_like(z)
        x = self.decode(z)
        if clamp:
            x = x.clamp(*clamp)
        return x


class LDMSpatial(nn.Module):
    def __init__(self, d, image_shape, latent_ch=4, base_ch=128,
                 unet_base_ch=128, kl_weight=1e-4):
        super().__init__()
        self.d = d
        self.image_shape = image_shape
        self.kl_weight = kl_weight

        self.vae_encoder = SpatialVAEEncoder(image_shape, latent_ch, base_ch)
        self.decoder = SpatialDecoder(latent_ch, image_shape, base_ch)
        self.score_net = UNetScoreNet(latent_ch, base_ch=unet_base_ch)

        self.latent_ch = latent_ch

        self.beta_min = 0.1
        self.beta_max = 20.0
        self.register_buffer('z_mean', torch.zeros(latent_ch, 1, 1))
        self.register_buffer('z_std', torch.ones(latent_ch, 1, 1))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"    LDMSpatial: {n_params:,} params")

    def noise_schedule(self, t):
        t_tensor = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
        log_mean_coeff = -0.25 * t_tensor**2 * (self.beta_max - self.beta_min) \
                         - 0.5 * t_tensor * self.beta_min
        alpha_bar = torch.exp(2.0 * log_mean_coeff)
        sigma = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-8))
        return alpha_bar, sigma

    def freeze_ae(self):
        for p in self.vae_encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def sample(self, n, n_steps=500, device='cpu', clamp=None):
        # Infer spatial shape from encoder
        with torch.no_grad():
            dummy = torch.zeros(1, *self.image_shape, device=device)
            dummy_flat = dummy.view(1, -1)
            mu, _ = self.vae_encoder(dummy_flat)
            _, c, h, w = mu.shape

        z = torch.randn(n, c, h, w, device=device)
        dt = 0.999 / n_steps
        for i in range(n_steps, 0, -1):
            t = torch.full((n,), i * dt, device=device)
            alpha_bar, sigma = self.noise_schedule(t)
            eps_pred = self.score_net(z, t)
            ab = alpha_bar.view(-1, 1, 1, 1)
            sig = sigma.view(-1, 1, 1, 1)
            x0_pred = (z - sig * eps_pred) / torch.sqrt(ab).clamp(min=1e-8)
            alpha_bar_prev, sigma_prev = self.noise_schedule(t - dt)
            ab_p = alpha_bar_prev.view(-1, 1, 1, 1)
            sig_p = sigma_prev.view(-1, 1, 1, 1)
            z = torch.sqrt(ab_p) * x0_pred + sig_p * torch.randn_like(z)
        z_raw = z * self.z_std + self.z_mean
        x = self.decoder(z_raw)
        if clamp:
            x = x.clamp(*clamp)
        return x


# ================================================================
# Training functions
# ================================================================

def train_stage1_gan(model, dataset, n_steps, lr_g=1e-4, lr_d=1e-4,
                     batch_size=64, lpips_weight=1.0, adv_weight=0.1,
                     ema_decay=0.9999, device='cpu'):
    img_size = model.image_shape[1]
    sigma_t0 = model.t0

    loss_fn_lpips = None
    if LPIPS_AVAILABLE:
        loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
        loss_fn_lpips.eval()
        for p in loss_fn_lpips.parameters():
            p.requires_grad = False

    disc = PatchDiscriminator(in_channels=3, base_ch=64, n_layers=3).to(device)
    gen_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    opt_g = torch.optim.Adam(gen_params, lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.999))
    ema = EMA(model, decay=ema_decay)

    disc_start = min(1000, n_steps // 10)
    print(f"  Stage 1: {n_steps} steps, disc at {disc_start}")

    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)
        x_t = x0 + sigma_t0 * torch.randn_like(x0)

        z, x_hat = model.project(x_t)
        x0_img = x0.view(-1, 3, img_size, img_size)
        xh_img = x_hat.view(-1, 3, img_size, img_size)

        mse_loss = F.mse_loss(x_hat, x0)
        lpips_loss = torch.tensor(0.0, device=device)
        if loss_fn_lpips is not None:
            lpips_loss = loss_fn_lpips(xh_img * 2 - 1, x0_img * 2 - 1).mean()

        g_adv = torch.tensor(0.0, device=device)
        if step >= disc_start:
            g_adv = hinge_loss_g(disc(xh_img))

        g_loss = mse_loss + lpips_weight * lpips_loss + adv_weight * g_adv
        opt_g.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
        opt_g.step()

        d_loss = torch.tensor(0.0, device=device)
        if step >= disc_start:
            with torch.no_grad():
                _, xh_d = model.project(x_t)
                xh_d_img = xh_d.view(-1, 3, img_size, img_size)
            d_loss = hinge_loss_d(disc(x0_img), disc(xh_d_img))
            opt_d.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            opt_d.step()

        ema.update(model)

        if (step + 1) % 200 == 0:
            print(f"  S1 [{step+1}/{n_steps}] g={g_loss.item():.6f} "
                  f"mse={mse_loss.item():.6f} lpips={lpips_loss.item():.6f} "
                  f"adv={g_adv.item():.6f} d={d_loss.item():.6f}")

    ema.apply(model)

    # Compute spatial normalization stats (per-channel)
    with torch.no_grad():
        zs = []
        for _ in range(20):
            x0, _ = dataset.sample_clean(min(batch_size, 500))
            x0 = x0.to(device)
            zs.append(model.encoder(x0))
        zs = torch.cat(zs, dim=0)
        model.z_mean = zs.mean(dim=(0, 2, 3), keepdim=True).squeeze(0)
        model.z_std = zs.std(dim=(0, 2, 3), keepdim=True).squeeze(0).clamp(min=1e-6)
    print("  Applied EMA + computed z stats.")


def train_vae_gan(model, dataset, n_steps, lr_g=1e-4, lr_d=1e-4,
                  batch_size=64, lpips_weight=1.0, adv_weight=0.1,
                  kl_weight=1e-4, ema_decay=0.9999, device='cpu'):
    img_size = int(np.sqrt(model.d // 3))

    loss_fn_lpips = None
    if LPIPS_AVAILABLE:
        loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
        loss_fn_lpips.eval()
        for p in loss_fn_lpips.parameters():
            p.requires_grad = False

    disc = PatchDiscriminator(in_channels=3, base_ch=64, n_layers=3).to(device)
    gen_params = list(model.vae_encoder.parameters()) + list(model.decoder.parameters())
    opt_g = torch.optim.Adam(gen_params, lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.999))
    ema = EMA(model, decay=ema_decay)

    disc_start = min(1000, n_steps // 10)
    print(f"  VAE-GAN: {n_steps} steps, disc at {disc_start}")

    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)

        mu, logvar = model.vae_encoder(x0)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        x_recon = model.decoder(z)

        x0_img = x0.view(-1, 3, img_size, img_size)
        xr_img = x_recon.view(-1, 3, img_size, img_size)

        mse_loss = F.mse_loss(x_recon, x0)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        lpips_loss = torch.tensor(0.0, device=device)
        if loss_fn_lpips is not None:
            lpips_loss = loss_fn_lpips(xr_img * 2 - 1, x0_img * 2 - 1).mean()

        g_adv = torch.tensor(0.0, device=device)
        if step >= disc_start:
            g_adv = hinge_loss_g(disc(xr_img))

        g_loss = mse_loss + lpips_weight * lpips_loss + kl_weight * kl_loss + adv_weight * g_adv
        opt_g.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
        opt_g.step()

        d_loss = torch.tensor(0.0, device=device)
        if step >= disc_start:
            with torch.no_grad():
                mu_d, _ = model.vae_encoder(x0)
                xr_d = model.decoder(mu_d).view(-1, 3, img_size, img_size)
            d_loss = hinge_loss_d(disc(x0_img), disc(xr_d))
            opt_d.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            opt_d.step()

        ema.update(model)

        if (step + 1) % 200 == 0:
            print(f"  VAE [{step+1}/{n_steps}] g={g_loss.item():.6f} "
                  f"mse={mse_loss.item():.6f} lpips={lpips_loss.item():.6f} "
                  f"kl={kl_loss.item():.4f} adv={g_adv.item():.6f} "
                  f"d={d_loss.item():.6f}")

    ema.apply(model)

    with torch.no_grad():
        zs = []
        for _ in range(20):
            x0, _ = dataset.sample_clean(min(batch_size, 500))
            x0 = x0.to(device)
            zs.append(model.vae_encoder.encode_deterministic(x0))
        zs = torch.cat(zs, dim=0)
        model.z_mean = zs.mean(dim=(0, 2, 3), keepdim=True).squeeze(0)
        model.z_std = zs.std(dim=(0, 2, 3), keepdim=True).squeeze(0).clamp(min=1e-6)
    print("  Applied EMA + computed z stats.")


def train_diffusion_ema(model, dataset, n_steps, lr, batch_size,
                        ema_decay=0.9999, is_twostage=True, device='cpu'):
    ema = EMA(model, decay=ema_decay)
    score_params = list(model.score_net.parameters())
    optimizer = torch.optim.Adam(score_params, lr=lr)

    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)

        with torch.no_grad():
            if is_twostage:
                z0 = model.encode(x0)
            else:
                z0 = (model.vae_encoder.encode_deterministic(x0) -
                      model.z_mean) / model.z_std.clamp(min=1e-6)

        t = torch.rand(z0.shape[0], device=device) * 0.998 + 0.001
        alpha_bar, sigma = model.noise_schedule(t)
        eps = torch.randn_like(z0)
        ab = alpha_bar.view(-1, 1, 1, 1)
        sig = sigma.view(-1, 1, 1, 1)
        z_t = torch.sqrt(ab) * z0 + sig * eps
        eps_pred = model.score_net(z_t, t)
        loss = F.mse_loss(eps_pred, eps)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(score_params, 1.0)
        optimizer.step()
        ema.update(model)

        if (step + 1) % 200 == 0:
            print(f"  Diff [{step+1}/{n_steps}] eps_loss={loss.item():.6f}")

    ema.apply(model)
    print("  Applied EMA to score net.")


# ================================================================
# Main
# ================================================================

def run_experiment(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    img_size = 64
    image_shape = (3, img_size, img_size)
    d = 3 * img_size * img_size
    latent_ch = 4          # spatial latent channels (8x8x4 = 256 total)
    base_ch = 128          # encoder/decoder base channels
    unet_base_ch = 128     # U-Net score net base channels
    t0 = 0.005

    ae_steps = 100000
    diff_steps = 300000
    batch_size = 64
    lpips_weight = 1.0
    adv_weight = 0.1
    ema_decay = 0.9999

    print("=" * 70)
    print(f"CelebA-HQ — Spatial Latent + U-Net Score (d={d})")
    print(f"  latent: {latent_ch}ch x 8x8 = {latent_ch*64} dims")
    print(f"  encoder/decoder base_ch={base_ch}")
    print(f"  U-Net score base_ch={unet_base_ch}")
    print(f"  AE={ae_steps}, Diff={diff_steps}, batch={batch_size}")
    print(f"  LPIPS={lpips_weight}, adv={adv_weight}, EMA={ema_decay}")
    print("=" * 70)

    dataset = CelebAHQDataset(img_size=img_size, max_samples=20000)

    # PCA
    print("\n--- PCA Analysis ---")
    explained_var = dataset.pca_explained_variance
    cumvar = torch.cumsum(explained_var, dim=0) / explained_var.sum()
    for threshold in [0.90, 0.95, 0.99]:
        k_est = (cumvar < threshold).sum().item() + 1
        print(f"  {threshold*100:.0f}% variance at k={k_est}")
    k_95 = (cumvar < 0.95).sum().item() + 1

    # ================================================================
    # MODEL 1: Two-Stage Spatial
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODEL 1: Two-Stage (Spatial Latent + U-Net)")
    print("=" * 70)

    model_ts = TwoStageSpatial(
        d=d, image_shape=image_shape, latent_ch=latent_ch,
        base_ch=base_ch, unet_base_ch=unet_base_ch, t0=t0,
    ).to(device)

    ckpt = load_checkpoint(model_ts, 'celebahq_unet_twostage_full', device=device)
    if ckpt is not None:
        print("  Resumed full from checkpoint.")
        model_ts.freeze_stage1()
        recon_mse_ts = ckpt.get('recon_mse', 0.0)
        avg_loss_ts = ckpt.get('avg_loss', 0.0)
    else:
        ckpt_s1 = load_checkpoint(model_ts, 'celebahq_unet_twostage_stage1',
                                  device=device)
        if ckpt_s1 is not None:
            print("  Resumed Stage 1.")
            model_ts.freeze_stage1()
            recon_mse_ts = ckpt_s1.get('recon_mse', 0.0)
        else:
            print("\n--- Stage 1: GAN + LPIPS + EMA ---")
            train_stage1_gan(model_ts, dataset, n_steps=ae_steps,
                             batch_size=batch_size, lpips_weight=lpips_weight,
                             adv_weight=adv_weight, ema_decay=ema_decay,
                             device=device)

            with torch.no_grad():
                x0_t, _ = dataset.sample_clean(2000)
                x0_t = x0_t.to(device)
                _, x_hat = model_ts.project(x0_t)
                recon_mse_ts = F.mse_loss(x_hat, x0_t).item()
                print(f"\n  Reconstruction MSE: {recon_mse_ts:.6f}")

            model_ts.freeze_stage1()
            save_checkpoint(model_ts, 'celebahq_unet_twostage_stage1',
                            {'recon_mse': recon_mse_ts})

        print("\n--- Stage 2: U-Net Diffusion + EMA ---")
        train_diffusion_ema(model_ts, dataset, n_steps=diff_steps, lr=3e-4,
                            batch_size=batch_size, ema_decay=ema_decay,
                            is_twostage=True, device=device)

        eval_times = [0.01, 0.1, 0.3, 0.5, 0.8]
        total = 0.0
        for t_eval in eval_times:
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                z0 = model_ts.encode(x0)
                t_batch = torch.full((2000,), t_eval, device=device)
                z_t, eps = model_ts.forward_diffusion(z0, t_batch)
                eps_pred = model_ts.score_net(z_t, t_batch)
                mse = F.mse_loss(eps_pred, eps).item()
                total += mse
                print(f"  t={t_eval:.2f}: eps_MSE={mse:.6f}")
        avg_loss_ts = total / len(eval_times)
        save_checkpoint(model_ts, 'celebahq_unet_twostage_full',
                        {'recon_mse': recon_mse_ts, 'avg_loss': avg_loss_ts})

    # ================================================================
    # MODEL 2: LDM Spatial
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODEL 2: LDM-CNN (Spatial Latent + U-Net)")
    print("=" * 70)

    model_ldm = LDMSpatial(
        d=d, image_shape=image_shape, latent_ch=latent_ch,
        base_ch=base_ch, unet_base_ch=unet_base_ch, kl_weight=1e-4,
    ).to(device)

    ckpt = load_checkpoint(model_ldm, 'celebahq_unet_ldm_full', device=device)
    if ckpt is not None:
        print("  Resumed full from checkpoint.")
        model_ldm.freeze_ae()
        recon_mse_ldm = ckpt.get('recon_mse', 0.0)
        avg_loss_ldm = ckpt.get('avg_loss', 0.0)
    else:
        ckpt_ae = load_checkpoint(model_ldm, 'celebahq_unet_ldm_ae',
                                  device=device)
        if ckpt_ae is not None:
            print("  Resumed AE.")
            model_ldm.freeze_ae()
            recon_mse_ldm = ckpt_ae.get('recon_mse', 0.0)
        else:
            print("\n--- VAE-GAN + LPIPS + EMA ---")
            train_vae_gan(model_ldm, dataset, n_steps=ae_steps,
                          batch_size=batch_size, lpips_weight=lpips_weight,
                          adv_weight=adv_weight, kl_weight=1e-4,
                          ema_decay=ema_decay, device=device)

            with torch.no_grad():
                x0_t, _ = dataset.sample_clean(2000)
                x0_t = x0_t.to(device)
                z = model_ldm.vae_encoder.encode_deterministic(x0_t)
                x_recon = model_ldm.decoder(z)
                recon_mse_ldm = F.mse_loss(x_recon, x0_t).item()
                print(f"\n  VAE Recon MSE: {recon_mse_ldm:.6f}")

            model_ldm.freeze_ae()
            save_checkpoint(model_ldm, 'celebahq_unet_ldm_ae',
                            {'recon_mse': recon_mse_ldm})

        print("\n--- U-Net Diffusion + EMA ---")
        train_diffusion_ema(model_ldm, dataset, n_steps=diff_steps, lr=3e-4,
                            batch_size=batch_size, ema_decay=ema_decay,
                            is_twostage=False, device=device)

        eval_times = [0.01, 0.1, 0.3, 0.5, 0.8]
        total = 0.0
        for t_eval in eval_times:
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                z0 = (model_ldm.vae_encoder.encode_deterministic(x0) -
                      model_ldm.z_mean) / model_ldm.z_std.clamp(min=1e-6)
                t_batch = torch.full((2000,), t_eval, device=device)
                alpha_bar, sigma = model_ldm.noise_schedule(t_batch)
                eps = torch.randn_like(z0)
                ab = alpha_bar.view(-1, 1, 1, 1)
                sig = sigma.view(-1, 1, 1, 1)
                z_t = torch.sqrt(ab) * z0 + sig * eps
                eps_pred = model_ldm.score_net(z_t, t_batch)
                mse = F.mse_loss(eps_pred, eps).item()
                total += mse
                print(f"  t={t_eval:.2f}: eps_MSE={mse:.6f}")
        avg_loss_ldm = total / len(eval_times)
        save_checkpoint(model_ldm, 'celebahq_unet_ldm_full',
                        {'recon_mse': recon_mse_ldm, 'avg_loss': avg_loss_ldm})

    results_ts = {'total_loss': avg_loss_ts, 'recon_error': recon_mse_ts}
    results_ldm = {'total_loss': avg_loss_ldm, 'recon_error': recon_mse_ldm}

    # ================================================================
    # GENERATE + FID
    # ================================================================
    print("\n--- Generating Samples ---")
    with torch.no_grad():
        print("  Two-Stage: sampling...")
        x_gen_ts = model_ts.sample(64, n_steps=500, device=device,
                                   clamp=(0.0, 1.0)).cpu()
        plot_image_grid(x_gen_ts, nrow=8, title='Two-Stage (U-Net)',
                        save_name='celebahq_unet_gen_twostage.png')

        print("  LDM: sampling...")
        x_gen_ldm = model_ldm.sample(64, n_steps=500, device=device,
                                     clamp=(0.0, 1.0)).cpu()
        plot_image_grid(x_gen_ldm, nrow=8, title='LDM (U-Net)',
                        save_name='celebahq_unet_gen_ldm.png')

        x_real, _ = dataset.sample_clean(64)
        plot_image_grid(x_real, nrow=8, title='Real CelebA-HQ',
                        save_name='celebahq_unet_real.png')

    if FID_AVAILABLE:
        print("\n--- FID ---")
        n_fid = 2048
        fb = 64
        real_imgs = []
        for _ in range(n_fid // fb):
            x0, _ = dataset.sample_clean(fb)
            real_imgs.append((x0.view(-1, 3, img_size, img_size).clamp(0, 1)
                              * 255).to(torch.uint8))
        real_imgs = torch.cat(real_imgs)[:n_fid]

        with torch.no_grad():
            print(f"  Generating {n_fid} for FID...")
            g_ts, g_ldm = [], []
            for i in range(n_fid // fb):
                xt = model_ts.sample(fb, n_steps=500, device=device,
                                     clamp=(0.0, 1.0)).cpu()
                g_ts.append((xt.view(-1, 3, img_size, img_size).clamp(0, 1)
                             * 255).to(torch.uint8))
                xl = model_ldm.sample(fb, n_steps=500, device=device,
                                      clamp=(0.0, 1.0)).cpu()
                g_ldm.append((xl.view(-1, 3, img_size, img_size).clamp(0, 1)
                              * 255).to(torch.uint8))
                if (i + 1) % 8 == 0:
                    print(f"    {(i+1)*fb}/{n_fid}")
            g_ts = torch.cat(g_ts)[:n_fid]
            g_ldm = torch.cat(g_ldm)[:n_fid]

        fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        fid.update(real_imgs.to(device), real=True)
        fid.update(g_ts.to(device), real=False)
        fid_ts = fid.compute().item()
        print(f"  Two-Stage FID: {fid_ts:.2f}")

        fid.reset()
        fid.update(real_imgs.to(device), real=True)
        fid.update(g_ldm.to(device), real=False)
        fid_ldm = fid.compute().item()
        print(f"  LDM FID: {fid_ldm:.2f}")

        del fid, real_imgs, g_ts, g_ldm
        torch.cuda.empty_cache()
    else:
        fid_ts = float('nan')
        fid_ldm = float('nan')

    # Reconstruction
    print("\n--- Reconstruction ---")
    with torch.no_grad():
        x_test, _ = dataset.sample_clean(8)
        x_d = x_test.to(device)
        _, xr_ts = model_ts.project(x_d)
        z_l = model_ldm.vae_encoder.encode_deterministic(x_d)
        xr_ldm = model_ldm.decoder(z_l)
        xr_ts, xr_ldm = xr_ts.cpu(), xr_ldm.cpu()

        fig, axes = plt.subplots(3, 8, figsize=(20, 7.5))
        for j in range(8):
            for row, img in enumerate([x_test[j], xr_ts[j], xr_ldm[j]]):
                axes[row, j].imshow(img.view(3, img_size, img_size)
                                    .permute(1, 2, 0).clamp(0, 1).numpy())
                axes[row, j].axis('off')
        axes[0, 0].set_ylabel('Original', fontsize=11, rotation=0,
                              labelpad=60, va='center')
        axes[1, 0].set_ylabel('Two-Stage', fontsize=11, rotation=0,
                              labelpad=60, va='center')
        axes[2, 0].set_ylabel('LDM', fontsize=11, rotation=0,
                              labelpad=60, va='center')
        plt.suptitle('CelebA-HQ Recon (Spatial+U-Net)', fontsize=14)
        plt.tight_layout()
        path = os.path.join(FIG_DIR, 'celebahq_unet_reconstruction.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("CelebA-HQ SUMMARY (Spatial Latent + U-Net Score)")
    print("=" * 70)
    print(f"  Latent: {latent_ch}ch x 8x8 = {latent_ch*64} dims")
    print(f"  Intrinsic dim (95% PCA): k={k_95}")
    print(f"\n  {'Model':<30} {'Eps Loss':<12} {'Recon MSE':<12} {'FID':<10}")
    print("  " + "-" * 64)
    print(f"  {'Two-Stage (ours)':<30} "
          f"{results_ts['total_loss']:<12.6f} "
          f"{results_ts['recon_error']:<12.6f} {fid_ts:<10.2f}")
    print(f"  {'LDM (fair)':<30} "
          f"{results_ldm['total_loss']:<12.6f} "
          f"{results_ldm['recon_error']:<12.6f} {fid_ldm:<10.2f}")


if __name__ == '__main__':
    run_experiment()
