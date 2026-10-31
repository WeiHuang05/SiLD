"""
Experiment: CelebA-HQ — Large spatial latent (32x32x8 = 8192 dims)

Motivated by VQ-GAN paper which uses 64x64x3 latent (no compression).
Here we use mild 1.5:1 compression (12288 → 8192) to retain most spatial info.

Architecture:
  64x64x3 → Encoder (1 downsample) → 32x32x8 → U-Net score → Decoder → 64x64x3

Fair comparison: Two-Stage vs LDM with identical architecture.
Both use GAN + LPIPS + EMA.
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

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False

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
# Encoder / Decoder — shallow (1 downsample: 64→32)
# ================================================================

class ShallowEncoder(nn.Module):
    """64x64x3 → 32x32 x latent_ch. Only 1 downsample level."""
    def __init__(self, image_shape, latent_ch=8, base_ch=128):
        super().__init__()
        C, H, W = image_shape
        self.image_shape = image_shape
        ch = base_ch

        self.net = nn.Sequential(
            nn.Conv2d(C, ch, 3, padding=1),
            ResBlock2d(ch), ResBlock2d(ch), ResBlock2d(ch),
            # Downsample 64→32
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
            ResBlock2d(ch * 2), ResBlock2d(ch * 2), ResBlock2d(ch * 2),
            # Project to latent channels
            nn.GroupNorm(min(32, ch * 2), ch * 2),
            nn.SiLU(),
            nn.Conv2d(ch * 2, latent_ch, 1),
        )
        with torch.no_grad():
            out = self.net(torch.zeros(1, C, H, W))
            self.spatial_shape = out.shape[1:]
        n_params = sum(p.numel() for p in self.parameters())
        print(f"    Encoder: {n_params:,} params, output {list(self.spatial_shape)}")

    def forward(self, x):
        return self.net(x.view(-1, *self.image_shape))


class ShallowDecoder(nn.Module):
    """32x32 x latent_ch → 64x64x3."""
    def __init__(self, latent_ch, image_shape, base_ch=128):
        super().__init__()
        C, H, W = image_shape
        self.image_shape = image_shape
        self.d = C * H * W
        ch = base_ch * 2

        self.net = nn.Sequential(
            nn.Conv2d(latent_ch, ch, 1),
            nn.SiLU(),
            ResBlock2d(ch), ResBlock2d(ch), ResBlock2d(ch),
            # Upsample 32→64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch, ch // 2, 3, padding=1),
            ResBlock2d(ch // 2), ResBlock2d(ch // 2), ResBlock2d(ch // 2),
            # Output
            nn.GroupNorm(min(32, ch // 2), ch // 2),
            nn.SiLU(),
            nn.Conv2d(ch // 2, C, 3, padding=1),
            nn.Sigmoid(),
        )
        n_params = sum(p.numel() for p in self.parameters())
        print(f"    Decoder: {n_params:,} params")

    def forward(self, z):
        out = self.net(z)
        return out[:, :, :self.image_shape[1], :self.image_shape[2]].reshape(-1, self.d)


class ShallowVAEEncoder(nn.Module):
    """Like ShallowEncoder but outputs mu, logvar."""
    def __init__(self, image_shape, latent_ch=8, base_ch=128):
        super().__init__()
        C, H, W = image_shape
        self.image_shape = image_shape
        ch = base_ch

        self.backbone = nn.Sequential(
            nn.Conv2d(C, ch, 3, padding=1),
            ResBlock2d(ch), ResBlock2d(ch), ResBlock2d(ch),
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
            ResBlock2d(ch * 2), ResBlock2d(ch * 2), ResBlock2d(ch * 2),
            nn.GroupNorm(min(32, ch * 2), ch * 2),
            nn.SiLU(),
        )
        self.conv_mu = nn.Conv2d(ch * 2, latent_ch, 1)
        self.conv_logvar = nn.Conv2d(ch * 2, latent_ch, 1)

    def forward(self, x):
        h = self.backbone(x.view(-1, *self.image_shape))
        return self.conv_mu(h), self.conv_logvar(h)

    def encode_deterministic(self, x):
        mu, _ = self.forward(x)
        return mu


# ================================================================
# U-Net Score Network for 32x32 spatial latent
# ================================================================

class UNetScore32(nn.Module):
    """
    U-Net for 32x32 spatial latent. 3 levels: 32→16→8→4.
    """
    def __init__(self, latent_ch=8, base_ch=128, time_embed_dim=256):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, base_ch * 4),
            nn.SiLU(),
            nn.Linear(base_ch * 4, base_ch * 4),
        )
        t_dim = base_ch * 4

        self.conv_in = nn.Conv2d(latent_ch, base_ch, 3, padding=1)

        # Down 1: 32→16
        self.d1_r1 = TimeCondResBlock(base_ch, t_dim)
        self.d1_r2 = TimeCondResBlock(base_ch, t_dim)
        self.d1_down = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)

        # Down 2: 16→8
        self.d2_r1 = TimeCondResBlock(base_ch * 2, t_dim)
        self.d2_r2 = TimeCondResBlock(base_ch * 2, t_dim)
        self.d2_down = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)

        # Down 3: 8→4
        self.d3_r1 = TimeCondResBlock(base_ch * 4, t_dim)
        self.d3_r2 = TimeCondResBlock(base_ch * 4, t_dim)
        self.d3_down = nn.Conv2d(base_ch * 4, base_ch * 4, 3, stride=2, padding=1)

        # Mid: 4x4
        self.mid_r1 = TimeCondResBlock(base_ch * 4, t_dim)
        self.mid_r2 = TimeCondResBlock(base_ch * 4, t_dim)

        # Up 3: 4→8
        self.u3_up = nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1)
        self.u3_r1 = TimeCondResBlock(base_ch * 8, t_dim)  # concat skip
        self.u3_r2 = TimeCondResBlock(base_ch * 8, t_dim)
        self.u3_out = nn.Conv2d(base_ch * 8, base_ch * 4, 1)

        # Up 2: 8→16
        self.u2_up = nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1)
        self.u2_r1 = TimeCondResBlock(base_ch * 4, t_dim)
        self.u2_r2 = TimeCondResBlock(base_ch * 4, t_dim)
        self.u2_out = nn.Conv2d(base_ch * 4, base_ch * 2, 1)

        # Up 1: 16→32
        self.u1_up = nn.Conv2d(base_ch * 2, base_ch, 3, padding=1)
        self.u1_r1 = TimeCondResBlock(base_ch * 2, t_dim)
        self.u1_r2 = TimeCondResBlock(base_ch * 2, t_dim)
        self.u1_out = nn.Conv2d(base_ch * 2, base_ch, 1)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(min(32, base_ch), base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, latent_ch, 3, padding=1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        print(f"    UNetScore32: {n_params:,} params")

    def sinusoidal_embedding(self, t):
        half = self.time_embed_dim // 2
        freqs = torch.exp(-math.log(10000.0) *
                          torch.arange(half, device=t.device) / half)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, z_t, t):
        te = self.time_mlp(self.sinusoidal_embedding(t))

        h = self.conv_in(z_t)

        h1 = self.d1_r1(h, te)
        h1 = self.d1_r2(h1, te)
        h = self.d1_down(h1)

        h2 = self.d2_r1(h, te)
        h2 = self.d2_r2(h2, te)
        h = self.d2_down(h2)

        h3 = self.d3_r1(h, te)
        h3 = self.d3_r2(h3, te)
        h = self.d3_down(h3)

        h = self.mid_r1(h, te)
        h = self.mid_r2(h, te)

        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.u3_up(h)
        h = torch.cat([h, h3], dim=1)
        h = self.u3_r1(h, te)
        h = self.u3_r2(h, te)
        h = self.u3_out(h)

        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.u2_up(h)
        h = torch.cat([h, h2], dim=1)
        h = self.u2_r1(h, te)
        h = self.u2_r2(h, te)
        h = self.u2_out(h)

        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = self.u1_up(h)
        h = torch.cat([h, h1], dim=1)
        h = self.u1_r1(h, te)
        h = self.u1_r2(h, te)
        h = self.u1_out(h)

        return self.conv_out(h)


# ================================================================
# PatchGAN + EMA + helpers
# ================================================================

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, True)]
        ch = base_ch
        for _ in range(1, n_layers):
            ch_next = min(ch * 2, 512)
            layers += [nn.Conv2d(ch, ch_next, 4, stride=2, padding=1, bias=False),
                       nn.InstanceNorm2d(ch_next), nn.LeakyReLU(0.2, True)]
            ch = ch_next
        layers += [nn.Conv2d(ch, ch, 4, stride=1, padding=1, bias=False),
                   nn.InstanceNorm2d(ch), nn.LeakyReLU(0.2, True),
                   nn.Conv2d(ch, 1, 4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def hinge_d(real, fake):
    return (F.relu(1 - real).mean() + F.relu(1 + fake).mean()) * 0.5

def hinge_g(fake):
    return -fake.mean()


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
            s.data.mul_(self.decay).add_(m.data, alpha=1 - self.decay)
        for s, m in zip(self.shadow.buffers(), model.buffers()):
            s.data.copy_(m.data)

    def apply(self, model):
        model.load_state_dict(self.shadow.state_dict())


def save_ckpt(model, name, extra=None):
    path = os.path.join(CKPT_DIR, f'{name}.pt')
    state = {'model': model.state_dict()}
    if extra:
        state.update(extra)
    torch.save(state, path)
    print(f"  Saved: {path}")


def load_ckpt(model, name, device='cpu'):
    path = os.path.join(CKPT_DIR, f'{name}.pt')
    if os.path.exists(path):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state['model'])
        print(f"  Loaded: {path}")
        return state
    return None


def plot_grid(images, nrow=8, title='', save_name=None, img_size=64):
    n = min(len(images), nrow * nrow)
    fig, axes = plt.subplots((n + nrow - 1) // nrow, nrow,
                             figsize=(nrow * 1.5, ((n + nrow - 1) // nrow) * 1.5))
    axes = np.atleast_2d(axes)
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            idx = i * nrow + j
            if idx < n:
                img = images[idx].detach().cpu().clamp(0, 1)
                axes[i, j].imshow(img.view(3, img_size, img_size)
                                  .permute(1, 2, 0).numpy())
            axes[i, j].axis('off')
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
# Two-Stage and LDM wrappers
# ================================================================

class TwoStageLargeLatent(nn.Module):
    def __init__(self, d, image_shape, latent_ch=8, base_ch=128,
                 unet_base_ch=128, t0=0.005):
        super().__init__()
        self.d = d
        self.t0 = t0
        self.T = 0.999
        self.image_shape = image_shape
        self.latent_ch = latent_ch
        self.beta_min, self.beta_max = 0.1, 20.0

        self.encoder = ShallowEncoder(image_shape, latent_ch, base_ch)
        self.decoder = ShallowDecoder(latent_ch, image_shape, base_ch)
        self.score_net = UNetScore32(latent_ch, unet_base_ch)
        self.spatial_shape = self.encoder.spatial_shape

        self.register_buffer('z_mean', torch.zeros(latent_ch, 1, 1))
        self.register_buffer('z_std', torch.ones(latent_ch, 1, 1))

    def noise_schedule(self, t):
        t = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
        log_mc = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        ab = torch.exp(2 * log_mc)
        sig = torch.sqrt((1 - ab).clamp(min=1e-8))
        return ab, sig

    def encode(self, x):
        return (self.encoder(x) - self.z_mean) / self.z_std.clamp(min=1e-6)

    def decode(self, z):
        return self.decoder(z * self.z_std + self.z_mean)

    def project(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)

    def freeze_stage1(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    def forward_diffusion(self, z0, t):
        ab, sig = self.noise_schedule(t)
        eps = torch.randn_like(z0)
        z_t = torch.sqrt(ab).view(-1, 1, 1, 1) * z0 + sig.view(-1, 1, 1, 1) * eps
        return z_t, eps

    @torch.no_grad()
    def sample(self, n, n_steps=500, device='cpu', clamp=None):
        _, c, h, w = 1, *self.spatial_shape
        z = torch.randn(n, c, h, w, device=device)
        dt = self.T / n_steps
        for i in range(n_steps, 0, -1):
            t = torch.full((n,), i * dt, device=device)
            ab, sig = self.noise_schedule(t)
            eps_pred = self.score_net(z, t)
            x0p = (z - sig.view(-1, 1, 1, 1) * eps_pred) / torch.sqrt(ab).view(-1, 1, 1, 1).clamp(min=1e-8)
            ab_p, sig_p = self.noise_schedule(t - dt)
            z = torch.sqrt(ab_p).view(-1, 1, 1, 1) * x0p + sig_p.view(-1, 1, 1, 1) * torch.randn_like(z)
        x = self.decode(z)
        return x.clamp(*clamp) if clamp else x


class LDMLargeLatent(nn.Module):
    def __init__(self, d, image_shape, latent_ch=8, base_ch=128,
                 unet_base_ch=128, kl_weight=1e-4):
        super().__init__()
        self.d = d
        self.image_shape = image_shape
        self.kl_weight = kl_weight
        self.latent_ch = latent_ch
        self.beta_min, self.beta_max = 0.1, 20.0

        self.vae_encoder = ShallowVAEEncoder(image_shape, latent_ch, base_ch)
        self.decoder = ShallowDecoder(latent_ch, image_shape, base_ch)
        self.score_net = UNetScore32(latent_ch, unet_base_ch)

        self.register_buffer('z_mean', torch.zeros(latent_ch, 1, 1))
        self.register_buffer('z_std', torch.ones(latent_ch, 1, 1))

    def noise_schedule(self, t):
        t = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
        log_mc = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        ab = torch.exp(2 * log_mc)
        sig = torch.sqrt((1 - ab).clamp(min=1e-8))
        return ab, sig

    def freeze_ae(self):
        for p in self.vae_encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def sample(self, n, n_steps=500, device='cpu', clamp=None):
        dummy = torch.zeros(1, *self.image_shape, device=device)
        mu, _ = self.vae_encoder(dummy.view(1, -1))
        _, c, h, w = mu.shape
        z = torch.randn(n, c, h, w, device=device)
        dt = 0.999 / n_steps
        for i in range(n_steps, 0, -1):
            t = torch.full((n,), i * dt, device=device)
            ab, sig = self.noise_schedule(t)
            eps_pred = self.score_net(z, t)
            x0p = (z - sig.view(-1, 1, 1, 1) * eps_pred) / torch.sqrt(ab).view(-1, 1, 1, 1).clamp(min=1e-8)
            ab_p, sig_p = self.noise_schedule(t - dt)
            z = torch.sqrt(ab_p).view(-1, 1, 1, 1) * x0p + sig_p.view(-1, 1, 1, 1) * torch.randn_like(z)
        z_raw = z * self.z_std + self.z_mean
        x = self.decoder(z_raw)
        return x.clamp(*clamp) if clamp else x


# ================================================================
# Training
# ================================================================

def train_ae_gan(model, dataset, n_steps, batch_size, lpips_weight, adv_weight,
                 ema_decay, is_twostage, device):
    img_size = model.image_shape[1]
    sigma_t0 = model.t0 if is_twostage else 0.0

    loss_fn = None
    if LPIPS_AVAILABLE:
        loss_fn = lpips.LPIPS(net='vgg').to(device)
        loss_fn.eval()
        for p in loss_fn.parameters():
            p.requires_grad = False

    disc = PatchDiscriminator(3, 64, 3).to(device)
    if is_twostage:
        gen_p = list(model.encoder.parameters()) + list(model.decoder.parameters())
    else:
        gen_p = list(model.vae_encoder.parameters()) + list(model.decoder.parameters())
    opt_g = torch.optim.Adam(gen_p, lr=1e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))
    ema = EMA(model, ema_decay)
    disc_start = min(1000, n_steps // 10)

    mode = "Stage 1 (denoise)" if is_twostage else "VAE-GAN"
    print(f"  {mode}: {n_steps} steps, adv={adv_weight}, disc@{disc_start}")

    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)

        if is_twostage:
            x_in = x0 + sigma_t0 * torch.randn_like(x0)
            z, x_hat = model.project(x_in)
            kl_loss = torch.tensor(0.0, device=device)
        else:
            mu, logvar = model.vae_encoder(x0)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            x_hat = model.decoder(z)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        x0_img = x0.view(-1, 3, img_size, img_size)
        xh_img = x_hat.view(-1, 3, img_size, img_size)

        mse = F.mse_loss(x_hat, x0)
        lp = loss_fn(xh_img * 2 - 1, x0_img * 2 - 1).mean() if loss_fn else torch.tensor(0.0, device=device)
        g_adv = hinge_g(disc(xh_img)) if step >= disc_start else torch.tensor(0.0, device=device)

        g_loss = mse + lpips_weight * lp + adv_weight * g_adv
        if not is_twostage:
            g_loss = g_loss + model.kl_weight * kl_loss

        opt_g.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_p, 1.0)
        opt_g.step()

        if step >= disc_start:
            with torch.no_grad():
                if is_twostage:
                    _, xh_d = model.project(x_in)
                else:
                    mu_d, _ = model.vae_encoder(x0)
                    xh_d = model.decoder(mu_d)
                xh_d_img = xh_d.view(-1, 3, img_size, img_size)
            d_loss = hinge_d(disc(x0_img), disc(xh_d_img))
            opt_d.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            opt_d.step()
        else:
            d_loss = torch.tensor(0.0)

        ema.update(model)

        if (step + 1) % 500 == 0:
            extra = f" kl={kl_loss.item():.4f}" if not is_twostage else ""
            print(f"  [{step+1}/{n_steps}] g={g_loss.item():.6f} "
                  f"mse={mse.item():.6f} lp={lp.item():.6f} "
                  f"adv={g_adv.item():.6f} d={d_loss.item():.6f}{extra}")

    ema.apply(model)

    # Compute z stats
    with torch.no_grad():
        zs = []
        for _ in range(20):
            x0, _ = dataset.sample_clean(min(batch_size, 500))
            x0 = x0.to(device)
            if is_twostage:
                zs.append(model.encoder(x0))
            else:
                zs.append(model.vae_encoder.encode_deterministic(x0))
        zs = torch.cat(zs, dim=0)
        model.z_mean = zs.mean(dim=(0, 2, 3), keepdim=True).squeeze(0)
        model.z_std = zs.std(dim=(0, 2, 3), keepdim=True).squeeze(0).clamp(min=1e-6)
    print("  EMA applied, z stats computed.")


def train_diffusion_ema(model, dataset, n_steps, batch_size, ema_decay,
                        is_twostage, device):
    ema = EMA(model, ema_decay)
    opt = torch.optim.Adam(model.score_net.parameters(), lr=3e-4)

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
        ab, sig = model.noise_schedule(t)
        eps = torch.randn_like(z0)
        z_t = torch.sqrt(ab).view(-1, 1, 1, 1) * z0 + sig.view(-1, 1, 1, 1) * eps
        loss = F.mse_loss(model.score_net(z_t, t), eps)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.score_net.parameters(), 1.0)
        opt.step()
        ema.update(model)

        if (step + 1) % 500 == 0:
            print(f"  Diff [{step+1}/{n_steps}] eps={loss.item():.6f}")

    ema.apply(model)
    print("  EMA applied to score net.")


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

    latent_ch = 8          # 32x32x8 = 8192 dims (1.5:1 compression)
    base_ch = 128          # encoder/decoder
    unet_base_ch = 128     # U-Net score
    t0 = 0.005

    ae_steps = 100000
    diff_steps = 300000
    batch_size = 32        # smaller due to large latent
    lpips_weight = 1.0
    adv_weight = 0.1
    ema_decay = 0.9999

    print("=" * 70)
    print(f"CelebA-HQ — Large Latent (32x32x{latent_ch} = {32*32*latent_ch} dims)")
    print(f"  Compression: {d} → {32*32*latent_ch} ({d/(32*32*latent_ch):.1f}:1)")
    print(f"  enc/dec base_ch={base_ch}, U-Net base_ch={unet_base_ch}")
    print(f"  AE={ae_steps}, Diff={diff_steps}, batch={batch_size}")
    print(f"  LPIPS={lpips_weight}, adv={adv_weight}, EMA={ema_decay}")
    print("=" * 70)

    dataset = CelebAHQDataset(img_size=img_size, max_samples=20000)

    print("\n--- PCA ---")
    ev = dataset.pca_explained_variance
    cv = torch.cumsum(ev, dim=0) / ev.sum()
    for th in [0.90, 0.95, 0.99]:
        print(f"  {th*100:.0f}% at k={(cv < th).sum().item() + 1}")
    k95 = (cv < 0.95).sum().item() + 1

    # ================================================================
    # MODEL 1: Two-Stage
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODEL 1: Two-Stage (large latent)")
    print("=" * 70)

    m_ts = TwoStageLargeLatent(d, image_shape, latent_ch, base_ch,
                                unet_base_ch, t0).to(device)

    ck = load_ckpt(m_ts, 'celebahq_ll_ts_full', device)
    if ck:
        m_ts.freeze_stage1()
        r_ts = ck.get('recon', 0.0)
        e_ts = ck.get('eps', 0.0)
    else:
        ck1 = load_ckpt(m_ts, 'celebahq_ll_ts_s1', device)
        if ck1:
            m_ts.freeze_stage1()
            r_ts = ck1.get('recon', 0.0)
        else:
            print("\n--- Stage 1: GAN+LPIPS+EMA ---")
            train_ae_gan(m_ts, dataset, ae_steps, batch_size, lpips_weight,
                         adv_weight, ema_decay, True, device)
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                _, xh = m_ts.project(x0)
                r_ts = F.mse_loss(xh, x0).item()
                print(f"\n  Recon MSE: {r_ts:.6f}")
            m_ts.freeze_stage1()
            save_ckpt(m_ts, 'celebahq_ll_ts_s1', {'recon': r_ts})

        print("\n--- Stage 2: U-Net Diffusion + EMA ---")
        train_diffusion_ema(m_ts, dataset, diff_steps, batch_size,
                            ema_decay, True, device)
        # Eval
        times = [0.01, 0.1, 0.3, 0.5, 0.8]
        tot = 0
        for tv in times:
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                z0 = m_ts.encode(x0)
                tb = torch.full((2000,), tv, device=device)
                zt, ep = m_ts.forward_diffusion(z0, tb)
                mse = F.mse_loss(m_ts.score_net(zt, tb), ep).item()
                tot += mse
                print(f"  t={tv:.2f}: eps={mse:.6f}")
        e_ts = tot / len(times)
        save_ckpt(m_ts, 'celebahq_ll_ts_full', {'recon': r_ts, 'eps': e_ts})

    # ================================================================
    # MODEL 2: LDM
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODEL 2: LDM (large latent)")
    print("=" * 70)

    m_ldm = LDMLargeLatent(d, image_shape, latent_ch, base_ch,
                            unet_base_ch, kl_weight=1e-4).to(device)

    ck = load_ckpt(m_ldm, 'celebahq_ll_ldm_full', device)
    if ck:
        m_ldm.freeze_ae()
        r_ldm = ck.get('recon', 0.0)
        e_ldm = ck.get('eps', 0.0)
    else:
        ck1 = load_ckpt(m_ldm, 'celebahq_ll_ldm_ae', device)
        if ck1:
            m_ldm.freeze_ae()
            r_ldm = ck1.get('recon', 0.0)
        else:
            print("\n--- VAE-GAN+LPIPS+EMA ---")
            train_ae_gan(m_ldm, dataset, ae_steps, batch_size, lpips_weight,
                         adv_weight, ema_decay, False, device)
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                z = m_ldm.vae_encoder.encode_deterministic(x0)
                xr = m_ldm.decoder(z)
                r_ldm = F.mse_loss(xr, x0).item()
                print(f"\n  Recon MSE: {r_ldm:.6f}")
            m_ldm.freeze_ae()
            save_ckpt(m_ldm, 'celebahq_ll_ldm_ae', {'recon': r_ldm})

        print("\n--- U-Net Diffusion + EMA ---")
        train_diffusion_ema(m_ldm, dataset, diff_steps, batch_size,
                            ema_decay, False, device)
        times = [0.01, 0.1, 0.3, 0.5, 0.8]
        tot = 0
        for tv in times:
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                z0 = (m_ldm.vae_encoder.encode_deterministic(x0) -
                      m_ldm.z_mean) / m_ldm.z_std.clamp(min=1e-6)
                tb = torch.full((2000,), tv, device=device)
                ab, sig = m_ldm.noise_schedule(tb)
                ep = torch.randn_like(z0)
                zt = torch.sqrt(ab).view(-1, 1, 1, 1) * z0 + sig.view(-1, 1, 1, 1) * ep
                mse = F.mse_loss(m_ldm.score_net(zt, tb), ep).item()
                tot += mse
                print(f"  t={tv:.2f}: eps={mse:.6f}")
        e_ldm = tot / len(times)
        save_ckpt(m_ldm, 'celebahq_ll_ldm_full', {'recon': r_ldm, 'eps': e_ldm})

    # ================================================================
    # rFID (reconstruction FID) — measures AE quality before diffusion
    # ================================================================
    print("\n--- rFID (Reconstruction FID) ---")
    if FID_AVAILABLE:
        nf, fb = 2048, 32
        reals_r = []
        recons_ts_r = []
        recons_ldm_r = []
        with torch.no_grad():
            for i in range(nf // fb):
                x0, _ = dataset.sample_clean(fb)
                x0 = x0.to(device)
                reals_r.append((x0.view(-1, 3, img_size, img_size).clamp(0, 1) * 255).to(torch.uint8).cpu())
                _, xr_ts = m_ts.project(x0)
                recons_ts_r.append((xr_ts.view(-1, 3, img_size, img_size).clamp(0, 1) * 255).to(torch.uint8).cpu())
                zl = m_ldm.vae_encoder.encode_deterministic(x0)
                xr_ldm = m_ldm.decoder(zl)
                recons_ldm_r.append((xr_ldm.view(-1, 3, img_size, img_size).clamp(0, 1) * 255).to(torch.uint8).cpu())
        reals_r = torch.cat(reals_r)[:nf]
        recons_ts_r = torch.cat(recons_ts_r)[:nf]
        recons_ldm_r = torch.cat(recons_ldm_r)[:nf]

        fid_r = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        fid_r.update(reals_r.to(device), real=True)
        fid_r.update(recons_ts_r.to(device), real=False)
        rfid_ts = fid_r.compute().item()
        print(f"  Two-Stage rFID: {rfid_ts:.2f}")

        fid_r.reset()
        fid_r.update(reals_r.to(device), real=True)
        fid_r.update(recons_ldm_r.to(device), real=False)
        rfid_ldm = fid_r.compute().item()
        print(f"  LDM rFID: {rfid_ldm:.2f}")

        del fid_r, reals_r, recons_ts_r, recons_ldm_r
        torch.cuda.empty_cache()
    else:
        rfid_ts = rfid_ldm = float('nan')

    # ================================================================
    # Generate + FID
    # ================================================================
    print("\n--- Generating ---")
    with torch.no_grad():
        xg_ts = m_ts.sample(64, 500, device, (0, 1)).cpu()
        plot_grid(xg_ts, 8, 'Two-Stage (32x32x8)', 'celebahq_ll_gen_ts.png')
        xg_ldm = m_ldm.sample(64, 500, device, (0, 1)).cpu()
        plot_grid(xg_ldm, 8, 'LDM (32x32x8)', 'celebahq_ll_gen_ldm.png')
        xr, _ = dataset.sample_clean(64)
        plot_grid(xr, 8, 'Real', 'celebahq_ll_real.png')

    fid_ts = fid_ldm = float('nan')
    if FID_AVAILABLE:
        print("\n--- FID ---")
        nf, fb = 2048, 32
        reals = []
        for _ in range(nf // fb):
            x0, _ = dataset.sample_clean(fb)
            reals.append((x0.view(-1, 3, img_size, img_size).clamp(0, 1) * 255).to(torch.uint8))
        reals = torch.cat(reals)[:nf]

        for name, model in [('Two-Stage', m_ts), ('LDM', m_ldm)]:
            print(f"  {name}: generating {nf}...")
            gens = []
            with torch.no_grad():
                for i in range(nf // fb):
                    x = model.sample(fb, 500, device, (0, 1)).cpu()
                    gens.append((x.view(-1, 3, img_size, img_size).clamp(0, 1) * 255).to(torch.uint8))
                    if (i + 1) % 16 == 0:
                        print(f"    {(i+1)*fb}/{nf}")
            gens = torch.cat(gens)[:nf]
            fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
            fid.update(reals.to(device), real=True)
            fid.update(gens.to(device), real=False)
            val = fid.compute().item()
            if name == 'Two-Stage':
                fid_ts = val
            else:
                fid_ldm = val
            print(f"  {name} FID: {val:.2f}")
            del fid, gens
            torch.cuda.empty_cache()
        del reals

    # Reconstruction
    print("\n--- Reconstruction ---")
    with torch.no_grad():
        xt, _ = dataset.sample_clean(8)
        xd = xt.to(device)
        _, xrt = m_ts.project(xd)
        zl = m_ldm.vae_encoder.encode_deterministic(xd)
        xrl = m_ldm.decoder(zl)
        fig, axes = plt.subplots(3, 8, figsize=(20, 7.5))
        for j in range(8):
            for r, im in enumerate([xt[j], xrt.cpu()[j], xrl.cpu()[j]]):
                axes[r, j].imshow(im.view(3, 64, 64).permute(1, 2, 0).clamp(0, 1).numpy())
                axes[r, j].axis('off')
        for r, lb in enumerate(['Original', 'Two-Stage', 'LDM']):
            axes[r, 0].set_ylabel(lb, fontsize=11, rotation=0, labelpad=60, va='center')
        plt.suptitle(f'CelebA-HQ Recon (32x32x{latent_ch})', fontsize=14)
        plt.tight_layout()
        p = os.path.join(FIG_DIR, 'celebahq_ll_reconstruction.png')
        plt.savefig(p, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {p}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"CelebA-HQ SUMMARY (Large Latent: 32x32x{latent_ch} = {32*32*latent_ch})")
    print("=" * 70)
    print(f"  Compression: {d} → {32*32*latent_ch} ({d/(32*32*latent_ch):.1f}:1)")
    print(f"  k_95={k95}")
    print(f"\n  {'Model':<25} {'Eps Loss':<12} {'Recon MSE':<12} {'rFID':<10} {'FID':<10}")
    print("  " + "-" * 70)
    print(f"  {'Two-Stage (ours)':<25} {e_ts:<12.6f} {r_ts:<12.6f} {rfid_ts:<10.2f} {fid_ts:<10.2f}")
    print(f"  {'LDM (fair)':<25} {e_ldm:<12.6f} {r_ldm:<12.6f} {rfid_ldm:<10.2f} {fid_ldm:<10.2f}")


if __name__ == '__main__':
    run_experiment()
