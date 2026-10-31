"""
Experiment: CIFAR-10 (32x32x3) — Spatial Latent + U-Net Score (v2 for CIFAR).

Ports the CelebA-HQ U-Net v2 recipe to CIFAR-10:
  - 8x8x8 spatial latent (512 dims, 6:1 compression vs 3072-dim ambient)
  - U-Net score net (8 -> 4 -> 2 -> 4 -> 8 with skips)
  - Encoder/decoder via 2 stride-2 conv levels (32 -> 16 -> 8)
  - Stage 1: GAN + LPIPS + EMA (matching CelebA-HQ v2)
  - Stage 2: latent DDPM with EMA + restricted to [t0, T] (post bug-fix)
  - FID via torch-fidelity (user-requested) over 10k generated samples vs
    10k real test samples (saved to disk for the FID library).

The point is to give BOTH ULD and the VAE-LDM baseline a real chance at
good CIFAR FID, then check whether ULD still wins on the same axis it
won at CelebA-HQ.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import copy
import math
import json
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.image_data import CIFAR10Dataset

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available — Stage 1 will use MSE-only.")

try:
    import torch_fidelity
    TORCH_FIDELITY_AVAILABLE = True
except ImportError:
    TORCH_FIDELITY_AVAILABLE = False
    print("Warning: torch-fidelity not available — FID will be skipped.")


FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ================================================================
# Building blocks (CIFAR-sized; mirrors v2 but with smaller dims)
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
    def __init__(self, ch, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, ch), ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, ch), ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, ch * 2)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.t_proj(t_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)
        return x + h


# ================================================================
# Spatial Encoder / Decoder for 32x32 → 8x8
# ================================================================

class CIFARSpatialEncoder(nn.Module):
    """32x32x3 → 8x8 x latent_ch via 2 stride-2 levels."""
    def __init__(self, image_shape, latent_ch=8, base_ch=128):
        super().__init__()
        C, H, W = image_shape
        self.image_shape = image_shape

        ch = base_ch
        layers = [nn.Conv2d(C, ch, 3, padding=1)]
        # Level 1: 32 -> 16
        layers += [ResBlock2d(ch), ResBlock2d(ch)]
        layers += [nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1)]
        ch *= 2
        # Level 2: 16 -> 8
        layers += [ResBlock2d(ch), ResBlock2d(ch)]
        layers += [nn.Conv2d(ch, ch, 3, stride=2, padding=1)]
        # Final + project
        layers += [
            ResBlock2d(ch),
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, latent_ch, 1),
        ]
        self.conv = nn.Sequential(*layers)
        with torch.no_grad():
            out = self.conv(torch.zeros(1, C, H, W))
            self.spatial_shape = tuple(out.shape[1:])
            total = int(np.prod(self.spatial_shape))
            print(f"    CIFARSpatialEncoder: out={list(self.spatial_shape)}, dims={total}")

    def forward(self, x):
        x_img = x.view(-1, *self.image_shape)
        return self.conv(x_img)


class CIFARSpatialDecoder(nn.Module):
    """8x8 x latent_ch → 32x32x3 via 2 nearest-upsample levels."""
    def __init__(self, latent_ch, image_shape, base_ch=128):
        super().__init__()
        C, H, W = image_shape
        self.image_shape = image_shape
        self.d = C * H * W

        ch = base_ch * 2  # match encoder bottom level
        self.proj_in = nn.Sequential(
            nn.Conv2d(latent_ch, ch, 1),
            nn.SiLU(),
        )
        # Level 2: 8 -> 16
        self.up2 = nn.Sequential(
            ResBlock2d(ch), ResBlock2d(ch),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch, ch // 2, 3, padding=1),
        )
        ch //= 2
        # Level 1: 16 -> 32
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
        h = self.proj_in(z_spatial)
        h = self.up2(h)
        h = self.up1(h)
        h = self.out(h)
        img = h[:, :, :self.image_shape[1], :self.image_shape[2]]
        return img.reshape(-1, self.d)


class CIFARSpatialVAEEncoder(nn.Module):
    """Same backbone as CIFARSpatialEncoder but emits (mu, logvar)."""
    def __init__(self, image_shape, latent_ch=8, base_ch=128):
        super().__init__()
        C, H, W = image_shape
        self.image_shape = image_shape

        ch = base_ch
        layers = [nn.Conv2d(C, ch, 3, padding=1)]
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
# U-Net Score Net for 8x8 latent
# ================================================================

class CIFARUNetScoreNet(nn.Module):
    """U-Net for 8x8 spatial latents: 8 -> 4 -> 2 -> 4 -> 8 with skips."""
    def __init__(self, latent_ch=8, base_ch=192, time_embed_dim=192):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, base_ch * 4),
            nn.SiLU(),
            nn.Linear(base_ch * 4, base_ch * 4),
        )
        t_dim = base_ch * 4

        self.conv_in = nn.Conv2d(latent_ch, base_ch, 3, padding=1)

        # Down 1: 8 -> 4
        self.down1_res1 = TimeCondResBlock(base_ch, t_dim)
        self.down1_res2 = TimeCondResBlock(base_ch, t_dim)
        self.down1_conv = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)

        # Down 2: 4 -> 2
        self.down2_res1 = TimeCondResBlock(base_ch * 2, t_dim)
        self.down2_res2 = TimeCondResBlock(base_ch * 2, t_dim)
        self.down2_conv = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)

        # Middle (2x2)
        self.mid_res1 = TimeCondResBlock(base_ch * 4, t_dim)
        self.mid_res2 = TimeCondResBlock(base_ch * 4, t_dim)

        # Up 2: 2 -> 4 (concat with down2 skip = base_ch*2)
        self.up2_upsample = nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1)
        self.up2_res1 = TimeCondResBlock(base_ch * 4, t_dim)  # concat
        self.up2_res2 = TimeCondResBlock(base_ch * 4, t_dim)
        self.up2_out = nn.Conv2d(base_ch * 4, base_ch * 2, 1)

        # Up 1: 4 -> 8 (concat with down1 skip = base_ch)
        self.up1_upsample = nn.Conv2d(base_ch * 2, base_ch, 3, padding=1)
        self.up1_res1 = TimeCondResBlock(base_ch * 2, t_dim)  # concat
        self.up1_res2 = TimeCondResBlock(base_ch * 2, t_dim)
        self.up1_out = nn.Conv2d(base_ch * 2, base_ch, 1)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(min(32, base_ch), base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, latent_ch, 3, padding=1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"    CIFARUNetScoreNet: {n_params:,} params")

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

        h = self.conv_in(z_t)

        h1 = self.down1_res1(h, t_emb)
        h1 = self.down1_res2(h1, t_emb)
        h = self.down1_conv(h1)  # 8 -> 4

        h2 = self.down2_res1(h, t_emb)
        h2 = self.down2_res2(h2, t_emb)
        h = self.down2_conv(h2)  # 4 -> 2

        h = self.mid_res1(h, t_emb)
        h = self.mid_res2(h, t_emb)

        # Up 2: 2 -> 4
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.up2_upsample(h)
        h = torch.cat([h, h2], dim=1)
        h = self.up2_res1(h, t_emb)
        h = self.up2_res2(h, t_emb)
        h = self.up2_out(h)

        # Up 1: 4 -> 8
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.up1_upsample(h)
        h = torch.cat([h, h1], dim=1)
        h = self.up1_res1(h, t_emb)
        h = self.up1_res2(h, t_emb)
        h = self.up1_out(h)

        return self.conv_out(h)


# ================================================================
# PatchGAN + EMA + helpers (same as v2)
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
                       nn.InstanceNorm2d(ch_next),
                       nn.LeakyReLU(0.2, inplace=True)]
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


def plot_image_grid(images, nrow=8, title='', save_name=None, img_size=32):
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
# Two-Stage and LDM models for CIFAR (8x8x8 latent)
# ================================================================

class CIFARTwoStageSpatial(nn.Module):
    def __init__(self, d, image_shape, latent_ch=8, base_ch=128,
                 unet_base_ch=192, t0=0.005):
        super().__init__()
        self.d = d
        self.t0 = t0
        self.T = 0.999
        self.image_shape = image_shape

        self.encoder = CIFARSpatialEncoder(image_shape, latent_ch, base_ch)
        self.decoder = CIFARSpatialDecoder(latent_ch, image_shape, base_ch)
        self.score_net = CIFARUNetScoreNet(latent_ch, base_ch=unet_base_ch)

        self.latent_ch = latent_ch
        self.spatial_shape = self.encoder.spatial_shape
        self.latent_dim = int(np.prod(self.spatial_shape))

        self.beta_min = 0.1
        self.beta_max = 20.0

        self.register_buffer('z_mean', torch.zeros(latent_ch, 1, 1))
        self.register_buffer('z_std', torch.ones(latent_ch, 1, 1))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"    CIFARTwoStageSpatial: {n_params:,} params, "
              f"latent={list(self.spatial_shape)} = {self.latent_dim} dims")

    def noise_schedule(self, t):
        t_t = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
        log_mean = -0.25 * t_t**2 * (self.beta_max - self.beta_min) \
                   - 0.5 * t_t * self.beta_min
        alpha_bar = torch.exp(2.0 * log_mean)
        sigma = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-8))
        return alpha_bar, sigma

    def encode(self, x):
        z = self.encoder(x)
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
        return torch.sqrt(ab) * z0 + sig * eps, eps

    @torch.no_grad()
    def sample(self, n, n_steps=500, device='cpu', clamp=None):
        """DDIM on [t0, T], final step takes x0_pred (no re-noising at t0)."""
        _, h, w = self.spatial_shape
        z = torch.randn(n, self.latent_ch, h, w, device=device)
        ts = torch.linspace(self.T, self.t0, n_steps + 1, device=device)
        for i in range(n_steps):
            t_cur = ts[i]
            t_next = ts[i + 1]
            t = torch.full((n,), t_cur.item(), device=device)
            alpha_bar, sigma = self.noise_schedule(t)
            eps_pred = self.score_net(z, t)
            ab = alpha_bar.view(-1, 1, 1, 1)
            sig = sigma.view(-1, 1, 1, 1)
            x0_pred = (z - sig * eps_pred) / torch.sqrt(ab).clamp(min=1e-8)
            if i == n_steps - 1:
                z = x0_pred
            else:
                t_next_b = torch.full((n,), t_next.item(), device=device)
                alpha_bar_prev, sigma_prev = self.noise_schedule(t_next_b)
                ab_p = alpha_bar_prev.view(-1, 1, 1, 1)
                sig_p = sigma_prev.view(-1, 1, 1, 1)
                z = torch.sqrt(ab_p) * x0_pred + sig_p * torch.randn_like(z)
        x = self.decode(z)
        if clamp:
            x = x.clamp(*clamp)
        return x


class CIFARLDMSpatial(nn.Module):
    def __init__(self, d, image_shape, latent_ch=8, base_ch=128,
                 unet_base_ch=192, kl_weight=1e-4):
        super().__init__()
        self.d = d
        self.image_shape = image_shape
        self.kl_weight = kl_weight

        self.vae_encoder = CIFARSpatialVAEEncoder(image_shape, latent_ch, base_ch)
        self.decoder = CIFARSpatialDecoder(latent_ch, image_shape, base_ch)
        self.score_net = CIFARUNetScoreNet(latent_ch, base_ch=unet_base_ch)

        self.latent_ch = latent_ch
        self.beta_min = 0.1
        self.beta_max = 20.0
        self.register_buffer('z_mean', torch.zeros(latent_ch, 1, 1))
        self.register_buffer('z_std', torch.ones(latent_ch, 1, 1))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"    CIFARLDMSpatial: {n_params:,} params")

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

    @torch.no_grad()
    def sample(self, n, n_steps=500, device='cpu', clamp=None):
        with torch.no_grad():
            dummy = torch.zeros(1, *self.image_shape, device=device)
            mu, _ = self.vae_encoder(dummy.view(1, -1))
            _, c, h, w = mu.shape
        z = torch.randn(n, c, h, w, device=device)
        T = 0.999
        t_min = 1e-3
        ts = torch.linspace(T, t_min, n_steps + 1, device=device)
        for i in range(n_steps):
            t_cur = ts[i]
            t_next = ts[i + 1]
            t = torch.full((n,), t_cur.item(), device=device)
            alpha_bar, sigma = self.noise_schedule(t)
            eps_pred = self.score_net(z, t)
            ab = alpha_bar.view(-1, 1, 1, 1)
            sig = sigma.view(-1, 1, 1, 1)
            x0_pred = (z - sig * eps_pred) / torch.sqrt(ab).clamp(min=1e-8)
            if i == n_steps - 1:
                z = x0_pred
            else:
                t_next_b = torch.full((n,), t_next.item(), device=device)
                alpha_bar_prev, sigma_prev = self.noise_schedule(t_next_b)
                ab_p = alpha_bar_prev.view(-1, 1, 1, 1)
                sig_p = sigma_prev.view(-1, 1, 1, 1)
                z = torch.sqrt(ab_p) * x0_pred + sig_p * torch.randn_like(z)
        z_raw = z * self.z_std + self.z_mean
        x = self.decoder(z_raw)
        if clamp:
            x = x.clamp(*clamp)
        return x


# ================================================================
# Training functions (mirror v2 with bug-fixed t_min)
# ================================================================

def train_stage1_gan(model, dataset, n_steps, lr_g=1e-4, lr_d=1e-4,
                     batch_size=128, lpips_weight=1.0, adv_weight=0.01,
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

    disc_start = min(2000, n_steps // 10)
    print(f"  Stage 1: {n_steps} steps, adv={adv_weight}, disc at {disc_start}")

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

        if (step + 1) % 500 == 0:
            print(f"  S1 [{step+1}/{n_steps}] g={g_loss.item():.6f} "
                  f"mse={mse_loss.item():.6f} lpips={lpips_loss.item():.6f} "
                  f"adv={g_adv.item():.6f} d={d_loss.item():.6f}")

    ema.apply(model)

    with torch.no_grad():
        zs = []
        for _ in range(20):
            x0, _ = dataset.sample_clean(min(batch_size, 500))
            x0 = x0.to(device)
            zs.append(model.encoder(x0))
        zs = torch.cat(zs, dim=0)
        model.z_mean = zs.mean(dim=(0, 2, 3), keepdim=True).squeeze(0)
        model.z_std = zs.std(dim=(0, 2, 3), keepdim=True).squeeze(0).clamp(min=1e-6)


def train_vae_gan(model, dataset, n_steps, lr_g=1e-4, lr_d=1e-4,
                  batch_size=128, lpips_weight=1.0, adv_weight=0.01,
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

    disc_start = min(2000, n_steps // 10)
    print(f"  VAE-GAN: {n_steps} steps, adv={adv_weight}, disc at {disc_start}")

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

        if (step + 1) % 500 == 0:
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


def train_diffusion_ema(model, dataset, n_steps, lr, batch_size,
                        ema_decay=0.9999, is_twostage=True, device='cpu',
                        t_min=None, min_snr_gamma=5.0):
    """
    ULD restricted to [t0, 0.999]; LDM uses [0.001, 0.999].

    Min-SNR-gamma loss weighting (Hang et al. 2023): each per-sample loss
    is weighted by min(SNR(t), gamma) / SNR(t). This downweights very-high-
    noise t (where the loss saturates at ~1) and gives the score net a
    fair gradient signal at small t. Without it, the v2 spatial-latent
    diffusion failed to fit small-t and gave catastrophic FID. With
    gamma=5 the small-t regime gets equal attention.
    """
    if t_min is None:
        t_min = float(model.t0) if is_twostage else 0.001
    t_max = 0.999

    ema = EMA(model, decay=ema_decay)
    score_params = list(model.score_net.parameters())
    optimizer = torch.optim.Adam(score_params, lr=lr)
    print(f"  train_diffusion_ema: t in [{t_min:.4f}, {t_max:.4f}], "
          f"is_twostage={is_twostage}, min_snr_gamma={min_snr_gamma}")

    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)
        with torch.no_grad():
            if is_twostage:
                z0 = (model.encoder(x0) - model.z_mean) / model.z_std.clamp(min=1e-6)
            else:
                z0 = (model.vae_encoder.encode_deterministic(x0) -
                      model.z_mean) / model.z_std.clamp(min=1e-6)

        t = torch.rand(z0.shape[0], device=device) * (t_max - t_min) + t_min
        alpha_bar, sigma = model.noise_schedule(t)
        eps = torch.randn_like(z0)
        ab = alpha_bar.view(-1, 1, 1, 1)
        sig = sigma.view(-1, 1, 1, 1)
        z_t = torch.sqrt(ab) * z0 + sig * eps
        eps_pred = model.score_net(z_t, t)

        # Min-SNR-gamma weighting per sample
        snr = (alpha_bar / sigma.pow(2).clamp(min=1e-8)).clamp(min=1e-8)
        snr_gamma = torch.full_like(snr, float(min_snr_gamma))
        weight = torch.minimum(snr, snr_gamma) / snr  # in (0, 1]
        weight = weight.view(-1, 1, 1, 1)
        loss = (weight * (eps_pred - eps).pow(2)).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(score_params, 1.0)
        optimizer.step()
        ema.update(model)

        if (step + 1) % 1000 == 0:
            print(f"  Diff [{step+1}/{n_steps}] weighted_loss={loss.item():.6f}")

    ema.apply(model)


# ================================================================
# FID via torch-fidelity
# ================================================================

def _save_uint8_pngs(images, out_dir, prefix):
    """Save (N, 3, H, W) tensors in [0,1] as PNGs for torch-fidelity."""
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images):
        arr = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        arr = np.transpose(arr, (1, 2, 0))
        Image.fromarray(arr).save(os.path.join(out_dir, f'{prefix}_{i:05d}.png'))


def compute_fid_torch_fidelity(real_dir, gen_dir):
    """Run torch-fidelity to get FID + IS. Returns dict."""
    if not TORCH_FIDELITY_AVAILABLE:
        return {'fid': float('nan'), 'is_mean': float('nan'), 'is_std': float('nan')}
    metrics = torch_fidelity.calculate_metrics(
        input1=gen_dir,
        input2=real_dir,
        cuda=torch.cuda.is_available(),
        isc=True, fid=True, kid=False, verbose=False,
        samples_find_deep=False,
    )
    return {
        'fid': float(metrics.get('frechet_inception_distance', float('nan'))),
        'is_mean': float(metrics.get('inception_score_mean', float('nan'))),
        'is_std': float(metrics.get('inception_score_std', float('nan'))),
    }


# ================================================================
# Main
# ================================================================

def run_experiment(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    img_size = 32
    image_shape = (3, img_size, img_size)
    d = 3 * img_size * img_size  # 3072
    latent_ch = 8           # 8x8x8 = 512 dims (6:1 compression vs 3072)
    base_ch = 128
    unet_base_ch = 256      # bumped from 192 (the 192 model failed to converge)
    t0 = 0.05               # bumped from 0.005 — see logs/ablation_t0_results.json

    ae_steps = 60000
    diff_steps = 300000     # bumped from 250000
    batch_size = 128
    lpips_weight = 1.0
    adv_weight = 0.01
    ema_decay = 0.9999
    n_fid = 10000  # generated samples for FID

    print("=" * 70)
    print(f"CIFAR-10 — Spatial Latent + U-Net v2 (d={d})")
    print(f"  latent: {latent_ch}ch x 8x8 = {latent_ch*64} dims")
    print(f"  encoder/decoder base_ch={base_ch}, U-Net base_ch={unet_base_ch}")
    print(f"  AE={ae_steps}, Diff={diff_steps}, batch={batch_size}")
    print(f"  LPIPS={lpips_weight}, adv={adv_weight}, EMA={ema_decay}")
    print(f"  FID samples: {n_fid}")
    print("=" * 70)

    dataset = CIFAR10Dataset(class_idx=None, max_samples=50000)

    print("\n--- PCA Analysis ---")
    explained_var = dataset.pca_explained_variance
    cumvar = torch.cumsum(explained_var, dim=0) / explained_var.sum()
    for thr in [0.90, 0.95, 0.99]:
        k_est = (cumvar < thr).sum().item() + 1
        print(f"  {int(thr*100)}% variance at k={k_est}")
    k_95 = int((cumvar < 0.95).sum().item()) + 1

    # =================================================================
    # MODEL 1: ULD
    # =================================================================
    print("\n" + "=" * 70)
    print("MODEL 1: ULD — Two-Stage Spatial (8x8x8) + U-Net")
    print("=" * 70)
    model_ts = CIFARTwoStageSpatial(
        d=d, image_shape=image_shape, latent_ch=latent_ch,
        base_ch=base_ch, unet_base_ch=unet_base_ch, t0=t0,
    ).to(device)

    ckpt = load_checkpoint(model_ts, 'cifar_v2_twostage_full', device=device)
    if ckpt is not None:
        print("  Resumed full from checkpoint.")
        model_ts.freeze_stage1()
        recon_mse_ts = ckpt.get('recon_mse', 0.0)
        avg_loss_ts = ckpt.get('avg_loss', 0.0)
    else:
        ckpt_s1 = load_checkpoint(model_ts, 'cifar_v2_twostage_stage1', device=device)
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
            save_checkpoint(model_ts, 'cifar_v2_twostage_stage1',
                            {'recon_mse': recon_mse_ts})

        print("\n--- Stage 2: U-Net Diffusion + EMA (t in [t0, T]) ---")
        train_diffusion_ema(model_ts, dataset, n_steps=diff_steps, lr=2e-4,
                            batch_size=batch_size, ema_decay=ema_decay,
                            is_twostage=True, device=device)

        eval_times = [0.01, 0.1, 0.3, 0.5, 0.8]
        total = 0.0
        for t_eval in eval_times:
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                z0 = (model_ts.encoder(x0) - model_ts.z_mean) / model_ts.z_std.clamp(min=1e-6)
                t_batch = torch.full((2000,), t_eval, device=device)
                z_t, eps = model_ts.forward_diffusion(z0, t_batch)
                eps_pred = model_ts.score_net(z_t, t_batch)
                mse = F.mse_loss(eps_pred, eps).item()
                total += mse
                print(f"  t={t_eval:.2f}: eps_MSE={mse:.6f}")
        avg_loss_ts = total / len(eval_times)
        save_checkpoint(model_ts, 'cifar_v2_twostage_full',
                        {'recon_mse': recon_mse_ts, 'avg_loss': avg_loss_ts})

    # =================================================================
    # MODEL 2: LDM
    # =================================================================
    print("\n" + "=" * 70)
    print("MODEL 2: LDM — Spatial VAE (8x8x8) + U-Net")
    print("=" * 70)
    model_ldm = CIFARLDMSpatial(
        d=d, image_shape=image_shape, latent_ch=latent_ch,
        base_ch=base_ch, unet_base_ch=unet_base_ch, kl_weight=1e-4,
    ).to(device)

    ckpt = load_checkpoint(model_ldm, 'cifar_v2_ldm_full', device=device)
    if ckpt is not None:
        print("  Resumed full from checkpoint.")
        model_ldm.freeze_ae()
        recon_mse_ldm = ckpt.get('recon_mse', 0.0)
        avg_loss_ldm = ckpt.get('avg_loss', 0.0)
    else:
        ckpt_ae = load_checkpoint(model_ldm, 'cifar_v2_ldm_ae', device=device)
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
            save_checkpoint(model_ldm, 'cifar_v2_ldm_ae',
                            {'recon_mse': recon_mse_ldm})

        print("\n--- U-Net Diffusion + EMA (t in [0.001, T]) ---")
        train_diffusion_ema(model_ldm, dataset, n_steps=diff_steps, lr=2e-4,
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
        save_checkpoint(model_ldm, 'cifar_v2_ldm_full',
                        {'recon_mse': recon_mse_ldm, 'avg_loss': avg_loss_ldm})

    # =================================================================
    # Sample grids + FID via torch-fidelity
    # =================================================================
    print("\n--- Generating sample grids ---")
    with torch.no_grad():
        x_gen_ts = model_ts.sample(64, n_steps=500, device=device, clamp=(0, 1)).cpu()
        plot_image_grid(x_gen_ts, nrow=8, title='ULD (CIFAR v2)',
                        save_name='cifar_v2_gen_uld.png')
        x_gen_ldm = model_ldm.sample(64, n_steps=500, device=device, clamp=(0, 1)).cpu()
        plot_image_grid(x_gen_ldm, nrow=8, title='LDM (CIFAR v2)',
                        save_name='cifar_v2_gen_ldm.png')
        x_real, _ = dataset.sample_clean(64)
        plot_image_grid(x_real, nrow=8, title='Real CIFAR-10',
                        save_name='cifar_v2_real.png')

    fid_results = {}
    if TORCH_FIDELITY_AVAILABLE:
        print(f"\n--- FID via torch-fidelity ({n_fid} samples) ---")
        scratch = os.path.join('/tmp', f'cifar_v2_fid_{os.getpid()}')
        real_dir = os.path.join(scratch, 'real')
        uld_dir = os.path.join(scratch, 'uld')
        ldm_dir = os.path.join(scratch, 'ldm')
        try:
            # Save n_fid real CIFAR images
            print("  Saving real images...")
            real_imgs_all = []
            for _ in range(n_fid // 256):
                xr, _ = dataset.sample_clean(256)
                real_imgs_all.append(xr.view(-1, 3, img_size, img_size))
            real_imgs_all = torch.cat(real_imgs_all)[:n_fid]
            _save_uint8_pngs(real_imgs_all, real_dir, 'r')

            # Generate ULD samples
            print("  Sampling ULD...")
            uld_imgs = []
            with torch.no_grad():
                for i in range(n_fid // 64):
                    x = model_ts.sample(64, n_steps=500, device=device,
                                        clamp=(0, 1)).cpu().view(-1, 3, img_size, img_size)
                    uld_imgs.append(x)
                    if (i + 1) % 10 == 0:
                        print(f"    ULD {(i+1)*64}/{n_fid}")
            uld_imgs = torch.cat(uld_imgs)[:n_fid]
            _save_uint8_pngs(uld_imgs, uld_dir, 'u')

            # Generate LDM samples
            print("  Sampling LDM...")
            ldm_imgs = []
            with torch.no_grad():
                for i in range(n_fid // 64):
                    x = model_ldm.sample(64, n_steps=500, device=device,
                                         clamp=(0, 1)).cpu().view(-1, 3, img_size, img_size)
                    ldm_imgs.append(x)
                    if (i + 1) % 10 == 0:
                        print(f"    LDM {(i+1)*64}/{n_fid}")
            ldm_imgs = torch.cat(ldm_imgs)[:n_fid]
            _save_uint8_pngs(ldm_imgs, ldm_dir, 'l')

            print("  Computing FID...")
            uld_fid = compute_fid_torch_fidelity(real_dir, uld_dir)
            ldm_fid = compute_fid_torch_fidelity(real_dir, ldm_dir)
            fid_results = {'uld': uld_fid, 'ldm': ldm_fid}
            print(f"  ULD FID: {uld_fid['fid']:.2f}  IS: "
                  f"{uld_fid['is_mean']:.2f}+/-{uld_fid['is_std']:.2f}")
            print(f"  LDM FID: {ldm_fid['fid']:.2f}  IS: "
                  f"{ldm_fid['is_mean']:.2f}+/-{ldm_fid['is_std']:.2f}")
        finally:
            if os.path.isdir(scratch):
                shutil.rmtree(scratch, ignore_errors=True)

    # =================================================================
    # Save summary
    # =================================================================
    summary = {
        'config': {
            'img_size': img_size, 'd': d, 'latent_ch': latent_ch,
            'latent_dim': latent_ch * 64, 'base_ch': base_ch,
            'unet_base_ch': unet_base_ch, 't0': t0,
            'ae_steps': ae_steps, 'diff_steps': diff_steps,
            'batch_size': batch_size, 'n_fid': n_fid,
        },
        'k95_intrinsic': k_95,
        'uld': {
            'recon_mse': recon_mse_ts,
            'avg_eps_loss': avg_loss_ts,
            **fid_results.get('uld', {}),
        },
        'ldm': {
            'recon_mse': recon_mse_ldm,
            'avg_eps_loss': avg_loss_ldm,
            **fid_results.get('ldm', {}),
        },
    }
    out_path = os.path.join(LOG_DIR, 'cifar_v2_results.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved results JSON: {out_path}")

    print("\n" + "=" * 70)
    print("CIFAR-10 v2 SUMMARY")
    print("=" * 70)
    print(f"  Latent: {latent_ch}ch x 8x8 = {latent_ch*64} dims")
    print(f"  Intrinsic dim (95% PCA): k={k_95}")
    print(f"\n  {'Model':<20} {'Eps Loss':<12} {'Recon MSE':<12} {'FID':<10} {'IS':<10}")
    print("  " + "-" * 70)
    uld_fid = fid_results.get('uld', {}).get('fid', float('nan'))
    uld_is = fid_results.get('uld', {}).get('is_mean', float('nan'))
    ldm_fid = fid_results.get('ldm', {}).get('fid', float('nan'))
    ldm_is = fid_results.get('ldm', {}).get('is_mean', float('nan'))
    print(f"  {'ULD (ours)':<20} {avg_loss_ts:<12.6f} "
          f"{recon_mse_ts:<12.6f} {uld_fid:<10.2f} {uld_is:<10.2f}")
    print(f"  {'LDM (fair)':<20} {avg_loss_ldm:<12.6f} "
          f"{recon_mse_ldm:<12.6f} {ldm_fid:<10.2f} {ldm_is:<10.2f}")


if __name__ == '__main__':
    run_experiment()
