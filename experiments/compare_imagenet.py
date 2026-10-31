"""
Experiment: ImageNet-256 (256x256x3, d=196608)

Two-Stage vs LDM-CNN with:
  - Deep ConvEncoder/ConvDecoder (5 downsample levels for 256x256)
  - PatchGAN discriminator + LPIPS + EMA
  - FID evaluation
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

from data.image_data import ImageNet256Dataset

# LPIPS
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available, using MSE only")

# FID
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    print("Warning: torchmetrics not available, skipping FID")

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


# ================================================================
# Deep Encoder/Decoder for 256x256 (5 downsample levels)
# 256 -> 128 -> 64 -> 32 -> 16 -> 8
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


class DeepConvEncoder(nn.Module):
    """
    5-level CNN encoder for 256x256 images.
    256->128->64->32->16->8, channels: base->2x->4x->8x->8x
    Bottleneck: 8x8 spatial with 8*base_ch channels.
    """
    def __init__(self, image_shape, latent_dim, base_ch=64):
        super().__init__()
        C, H, W = image_shape
        self.image_shape = image_shape

        # Channel progression: base, 2x, 4x, 8x, 8x
        chs = [base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*8]

        layers = [nn.Conv2d(C, chs[0], 3, padding=1)]

        in_ch = chs[0]
        for i, out_ch in enumerate(chs):
            layers += [ResBlock2d(in_ch, out_ch), ResBlock2d(out_ch)]
            if i < len(chs) - 1:  # downsample except last level
                layers += [nn.Conv2d(out_ch, chs[i+1], 3, stride=2, padding=1)]
                in_ch = chs[i+1]
            else:
                # Last level: one more downsample to get to 8x8
                layers += [nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)]
                in_ch = out_ch

        layers += [ResBlock2d(in_ch), nn.GroupNorm(min(32, in_ch), in_ch), nn.SiLU()]
        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            self._flat_size = self.conv(torch.zeros(1, C, H, W)).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self._flat_size, 1024), nn.SiLU(),
            nn.Linear(1024, latent_dim),
        )
        print(f"    DeepConvEncoder: {sum(p.numel() for p in self.parameters()):,} params, "
              f"flat_size={self._flat_size}")

    def forward(self, x):
        x_img = x.view(-1, *self.image_shape)
        return self.fc(self.conv(x_img).view(x.size(0), -1))


class DeepConvDecoder(nn.Module):
    """
    5-level CNN decoder for 256x256 images.
    Mirrors DeepConvEncoder.
    """
    def __init__(self, latent_dim, image_shape, base_ch=64):
        super().__init__()
        C, H, W = image_shape
        self.image_shape = image_shape
        self.d = C * H * W

        # Same channel progression as encoder (reversed)
        chs = [base_ch*8, base_ch*8, base_ch*4, base_ch*2, base_ch]

        # Compute start spatial size (5 stride-2 downsamples)
        h, w = H, W
        for _ in range(5):
            h = (h + 1) // 2
            w = (w + 1) // 2
        self.start_h, self.start_w = h, w
        self.start_ch = chs[0]

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.SiLU(),
            nn.Linear(1024, chs[0] * h * w), nn.SiLU(),
        )

        # Build upsample levels
        self.ups = nn.ModuleList()
        for i in range(5):
            ch_in = chs[i]
            ch_out = chs[i+1] if i < 4 else chs[4]
            self.ups.append(nn.Sequential(
                ResBlock2d(ch_in), ResBlock2d(ch_in),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(ch_in, ch_out, 3, padding=1),
            ))

        self.out = nn.Sequential(
            ResBlock2d(chs[4]),
            nn.GroupNorm(min(32, chs[4]), chs[4]),
            nn.SiLU(),
            nn.Conv2d(chs[4], C, 3, padding=1),
            nn.Sigmoid(),
        )
        print(f"    DeepConvDecoder: {sum(p.numel() for p in self.parameters()):,} params, "
              f"start={self.start_h}x{self.start_w}")

    def forward(self, z):
        h = self.fc(z).view(-1, self.start_ch, self.start_h, self.start_w)
        for up in self.ups:
            h = up(h)
        h = self.out(h)
        img = h[:, :, :self.image_shape[1], :self.image_shape[2]]
        return img.reshape(-1, self.d)


class VAEEncoder(nn.Module):
    """Wraps DeepConvEncoder to output mu and logvar."""
    def __init__(self, image_shape, latent_dim, base_ch=64):
        super().__init__()
        C, H, W = image_shape
        self.image_shape = image_shape

        chs = [base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*8]
        layers = [nn.Conv2d(C, chs[0], 3, padding=1)]
        in_ch = chs[0]
        for i, out_ch in enumerate(chs):
            layers += [ResBlock2d(in_ch, out_ch), ResBlock2d(out_ch)]
            if i < len(chs) - 1:
                layers += [nn.Conv2d(out_ch, chs[i+1], 3, stride=2, padding=1)]
                in_ch = chs[i+1]
            else:
                layers += [nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)]
                in_ch = out_ch
        layers += [ResBlock2d(in_ch), nn.GroupNorm(min(32, in_ch), in_ch), nn.SiLU()]
        self.conv = nn.Sequential(*layers)

        with torch.no_grad():
            self._flat_size = self.conv(torch.zeros(1, C, H, W)).view(1, -1).shape[1]

        self.fc_shared = nn.Sequential(nn.Linear(self._flat_size, 1024), nn.SiLU())
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

    def forward(self, x):
        x_img = x.view(-1, *self.image_shape)
        h = self.fc_shared(self.conv(x_img).view(x.size(0), -1))
        return self.fc_mu(h), self.fc_logvar(h)

    def encode_deterministic(self, x):
        mu, _ = self.forward(x)
        return mu


# ================================================================
# Latent Score Network
# ================================================================

class LatentScoreNet(nn.Module):
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


# ================================================================
# PatchGAN Discriminator
# ================================================================

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_ch=64, n_layers=4):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_ch
        for i in range(1, n_layers):
            ch_next = min(ch * 2, 512)
            layers += [
                nn.Conv2d(ch, ch_next, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ch_next),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = ch_next
        layers += [
            nn.Conv2d(ch, ch, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, 1, 4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def hinge_loss_d(real_pred, fake_pred):
    return (F.relu(1.0 - real_pred).mean() + F.relu(1.0 + fake_pred).mean()) * 0.5

def hinge_loss_g(fake_pred):
    return -fake_pred.mean()


# ================================================================
# EMA
# ================================================================

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.data.copy_(m_buf.data)

    def apply(self, model):
        model.load_state_dict(self.shadow.state_dict())


# ================================================================
# Two-Stage Model (assembled here for ImageNet-256)
# ================================================================

class TwoStageImageNet(nn.Module):
    def __init__(self, d, latent_dim, image_shape, base_ch=64,
                 hidden_dim_score=1024, n_score_layers=8, t0=0.005):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim
        self.t0 = t0
        self.T = 0.999
        self.image_shape = image_shape

        self.encoder = DeepConvEncoder(image_shape, latent_dim, base_ch=base_ch)
        self.decoder = DeepConvDecoder(latent_dim, image_shape, base_ch=base_ch)
        self.score_net = LatentScoreNet(
            latent_dim, hidden_dim=hidden_dim_score, n_layers=n_score_layers,
        )
        self.stage1_frozen = False

        self.beta_min = 0.1
        self.beta_max = 20.0
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
        self.stage1_frozen = True

    def forward_diffusion_latent(self, z0, t):
        alpha_bar, sigma = self.noise_schedule(t)
        eps_z = torch.randn_like(z0)
        if alpha_bar.dim() > 0:
            alpha_bar = alpha_bar.unsqueeze(-1)
            sigma = sigma.unsqueeze(-1)
        z_t = torch.sqrt(alpha_bar) * z0 + sigma * eps_z
        return z_t, eps_z

    def predict_eps_latent(self, z_t, t):
        return self.score_net(z_t, t)

    @torch.no_grad()
    def sample(self, n, n_steps=500, device='cpu', clamp=None):
        z = torch.randn(n, self.latent_dim, device=device)
        dt = self.T / n_steps
        for i in range(n_steps, 0, -1):
            t = torch.full((n,), i * dt, device=device)
            alpha_bar, sigma = self.noise_schedule(t)
            eps_pred = self.predict_eps_latent(z, t)
            alpha_bar_prev, sigma_prev = self.noise_schedule(t - dt)
            x0_pred = (z - sigma.unsqueeze(-1) * eps_pred) / torch.sqrt(alpha_bar.unsqueeze(-1)).clamp(min=1e-8)
            z = torch.sqrt(alpha_bar_prev.unsqueeze(-1)) * x0_pred + sigma_prev.unsqueeze(-1) * torch.randn_like(z)
        x = self.decode(z)
        if clamp:
            x = x.clamp(*clamp)
        return x


class LDMImageNet(nn.Module):
    def __init__(self, d, latent_dim, image_shape, base_ch=64,
                 hidden_dim_score=1024, n_score_layers=8, kl_weight=1e-4):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.kl_weight = kl_weight

        self.vae_encoder = VAEEncoder(image_shape, latent_dim, base_ch=base_ch)
        self.decoder = DeepConvDecoder(latent_dim, image_shape, base_ch=base_ch)
        self.score_net = LatentScoreNet(
            latent_dim, hidden_dim=hidden_dim_score, n_layers=n_score_layers,
        )
        self.ae_frozen = False

        self.beta_min = 0.1
        self.beta_max = 20.0
        self.register_buffer('z_mean', torch.zeros(latent_dim))
        self.register_buffer('z_std', torch.ones(latent_dim))

    def noise_schedule(self, t):
        t_tensor = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
        log_mean_coeff = -0.25 * t_tensor ** 2 * (self.beta_max - self.beta_min) \
                         - 0.5 * t_tensor * self.beta_min
        alpha_bar = torch.exp(2.0 * log_mean_coeff)
        sigma = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-8))
        return alpha_bar, sigma

    def freeze_ae(self):
        for p in self.vae_encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.ae_frozen = True

    @torch.no_grad()
    def sample(self, n, n_steps=500, device='cpu', clamp=None):
        z = torch.randn(n, self.latent_dim, device=device)
        dt = 0.999 / n_steps
        for i in range(n_steps, 0, -1):
            t = torch.full((n,), i * dt, device=device)
            alpha_bar, sigma = self.noise_schedule(t)
            eps_pred = self.score_net(z, t)
            alpha_bar_prev, sigma_prev = self.noise_schedule(t - dt)
            x0_pred = (z - sigma.unsqueeze(-1) * eps_pred) / torch.sqrt(alpha_bar.unsqueeze(-1)).clamp(min=1e-8)
            z = torch.sqrt(alpha_bar_prev.unsqueeze(-1)) * x0_pred + sigma_prev.unsqueeze(-1) * torch.randn_like(z)
        # Decode
        z_raw = z * self.z_std + self.z_mean
        x = self.decoder(z_raw)
        if clamp:
            x = x.clamp(*clamp)
        return x


# ================================================================
# Checkpoint helpers
# ================================================================

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


def plot_image_grid(images, nrow=8, title='', save_name=None, img_size=256):
    n = min(len(images), nrow * nrow)
    ncol = nrow
    nrow_actual = (n + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow_actual, ncol,
                             figsize=(ncol * 2, nrow_actual * 2))
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
# Training functions
# ================================================================

def train_stage1_gan(model, dataset, n_steps, lr_g=1e-4, lr_d=1e-4,
                     batch_size=32, lpips_weight=1.0, adv_weight=0.1,
                     ema_decay=0.9999, device='cpu'):
    img_size = model.image_shape[1]
    sigma_t0 = model.t0

    loss_fn_lpips = None
    if LPIPS_AVAILABLE:
        loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
        loss_fn_lpips.eval()
        for p in loss_fn_lpips.parameters():
            p.requires_grad = False

    disc = PatchDiscriminator(in_channels=3, base_ch=64, n_layers=4).to(device)
    print(f"  PatchGAN: {sum(p.numel() for p in disc.parameters()):,} params")

    gen_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    opt_g = torch.optim.Adam(gen_params, lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.999))
    ema = EMA(model, decay=ema_decay)

    disc_start = min(2000, n_steps // 10)
    print(f"  Stage 1: {n_steps} steps, disc starts at {disc_start}")

    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)
        x_t = x0 + sigma_t0 * torch.randn_like(x0)

        z, x_hat = model.project(x_t)
        x0_img = x0.view(-1, 3, img_size, img_size)
        xh_img = x_hat.view(-1, 3, img_size, img_size)

        mse_loss = F.mse_loss(x_hat, x0)

        if loss_fn_lpips is not None:
            lpips_loss = loss_fn_lpips(xh_img * 2 - 1, x0_img * 2 - 1).mean()
        else:
            lpips_loss = torch.tensor(0.0, device=device)

        if step >= disc_start:
            fake_pred = disc(xh_img)
            g_adv_loss = hinge_loss_g(fake_pred)
        else:
            g_adv_loss = torch.tensor(0.0, device=device)

        g_loss = mse_loss + lpips_weight * lpips_loss + adv_weight * g_adv_loss

        opt_g.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
        opt_g.step()

        d_loss = torch.tensor(0.0, device=device)
        if step >= disc_start:
            with torch.no_grad():
                _, x_hat_d = model.project(x_t)
                xh_img_d = x_hat_d.view(-1, 3, img_size, img_size)
            real_pred = disc(x0_img)
            fake_pred = disc(xh_img_d)
            d_loss = hinge_loss_d(real_pred, fake_pred)
            opt_d.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            opt_d.step()

        ema.update(model)

        if (step + 1) % 500 == 0:
            print(f"  Stage 1 [{step+1}/{n_steps}] "
                  f"g={g_loss.item():.6f} mse={mse_loss.item():.6f} "
                  f"lpips={lpips_loss.item():.6f} adv={g_adv_loss.item():.6f} "
                  f"d={d_loss.item():.6f}")

    ema.apply(model)
    print("  Applied EMA weights.")

    with torch.no_grad():
        zs = []
        for _ in range(10):
            x0, _ = dataset.sample_clean(min(batch_size, 500))
            x0 = x0.to(device)
            zs.append(model.encoder(x0))
        zs = torch.cat(zs, dim=0)
        model.z_mean = zs.mean(dim=0)
        model.z_std = zs.std(dim=0).clamp(min=1e-6)


def train_vae_gan(model, dataset, n_steps, lr_g=1e-4, lr_d=1e-4,
                  batch_size=32, lpips_weight=1.0, adv_weight=0.1,
                  kl_weight=1e-4, ema_decay=0.9999, device='cpu'):
    img_size = model.image_shape[1]

    loss_fn_lpips = None
    if LPIPS_AVAILABLE:
        loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
        loss_fn_lpips.eval()
        for p in loss_fn_lpips.parameters():
            p.requires_grad = False

    disc = PatchDiscriminator(in_channels=3, base_ch=64, n_layers=4).to(device)
    print(f"  PatchGAN: {sum(p.numel() for p in disc.parameters()):,} params")

    gen_params = list(model.vae_encoder.parameters()) + list(model.decoder.parameters())
    opt_g = torch.optim.Adam(gen_params, lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.999))
    ema = EMA(model, decay=ema_decay)

    disc_start = min(2000, n_steps // 10)
    print(f"  VAE-GAN: {n_steps} steps, disc starts at {disc_start}")

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

        if loss_fn_lpips is not None:
            lpips_loss = loss_fn_lpips(xr_img * 2 - 1, x0_img * 2 - 1).mean()
        else:
            lpips_loss = torch.tensor(0.0, device=device)

        if step >= disc_start:
            fake_pred = disc(xr_img)
            g_adv_loss = hinge_loss_g(fake_pred)
        else:
            g_adv_loss = torch.tensor(0.0, device=device)

        g_loss = mse_loss + lpips_weight * lpips_loss + kl_weight * kl_loss + adv_weight * g_adv_loss

        opt_g.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
        opt_g.step()

        d_loss = torch.tensor(0.0, device=device)
        if step >= disc_start:
            with torch.no_grad():
                mu_d, _ = model.vae_encoder(x0)
                x_recon_d = model.decoder(mu_d)
                xr_img_d = x_recon_d.view(-1, 3, img_size, img_size)
            real_pred = disc(x0_img)
            fake_pred = disc(xr_img_d)
            d_loss = hinge_loss_d(real_pred, fake_pred)
            opt_d.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            opt_d.step()

        ema.update(model)

        if (step + 1) % 500 == 0:
            print(f"  VAE-GAN [{step+1}/{n_steps}] "
                  f"g={g_loss.item():.6f} mse={mse_loss.item():.6f} "
                  f"lpips={lpips_loss.item():.6f} kl={kl_loss.item():.4f} "
                  f"adv={g_adv_loss.item():.6f} d={d_loss.item():.6f}")

    ema.apply(model)
    print("  Applied EMA weights.")

    with torch.no_grad():
        zs = []
        for _ in range(10):
            x0, _ = dataset.sample_clean(min(batch_size, 500))
            x0 = x0.to(device)
            zs.append(model.vae_encoder.encode_deterministic(x0))
        zs = torch.cat(zs, dim=0)
        model.z_mean = zs.mean(dim=0)
        model.z_std = zs.std(dim=0).clamp(min=1e-6)


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

        if is_twostage:
            z_t, eps_z = model.forward_diffusion_latent(z0, t)
            eps_pred = model.predict_eps_latent(z_t, t)
        else:
            alpha_bar, sigma = model.noise_schedule(t)
            eps_z = torch.randn_like(z0)
            z_t = torch.sqrt(alpha_bar.unsqueeze(-1)) * z0 + sigma.unsqueeze(-1) * eps_z
            eps_pred = model.score_net(z_t, t)

        loss = F.mse_loss(eps_pred, eps_z)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(score_params, 1.0)
        optimizer.step()
        ema.update(model)

        if (step + 1) % 500 == 0:
            print(f"  Diffusion [{step+1}/{n_steps}] eps_loss={loss.item():.6f}")

    ema.apply(model)
    print("  Applied EMA weights to score network.")


# ================================================================
# Main experiment
# ================================================================

def run_experiment(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    img_size = 256
    image_shape = (3, img_size, img_size)
    d = 3 * img_size * img_size  # 196608
    latent_dim = 512
    t0 = 0.005

    # Network params
    base_ch = 64           # channels: 64->128->256->512->512
    hidden_dim_score = 1024
    n_score_layers = 8

    # Training params
    ae_steps = 100000
    diff_steps = 300000
    batch_size = 16        # smaller batch for 256x256
    lpips_weight = 1.0
    adv_weight = 0.1
    ema_decay = 0.9999

    print("=" * 70)
    print(f"ImageNet-256 EXPERIMENT (d={d}, {img_size}x{img_size}x3)")
    print(f"  latent_dim={latent_dim}")
    print(f"  base_ch={base_ch}, score_hidden={hidden_dim_score}, "
          f"score_layers={n_score_layers}")
    print(f"  AE steps={ae_steps}, Diffusion steps={diff_steps}")
    print(f"  LPIPS={lpips_weight}, adv={adv_weight}, EMA={ema_decay}")
    print(f"  batch_size={batch_size}")
    print("=" * 70)

    dataset = ImageNet256Dataset(max_samples=50000)

    # PCA analysis
    print("\n--- PCA Analysis ---")
    explained_var = dataset.pca_explained_variance
    cumvar = torch.cumsum(explained_var, dim=0) / explained_var.sum()
    for threshold in [0.90, 0.95, 0.99]:
        k_est = (cumvar < threshold).sum().item() + 1
        print(f"  {threshold*100:.0f}% variance at k={k_est}")
    k_95 = (cumvar < 0.95).sum().item() + 1

    # ================================================================
    # MODEL 1: Two-Stage
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODEL 1: Two-Stage Manifold Diffusion")
    print("=" * 70)

    model_ts = TwoStageImageNet(
        d=d, latent_dim=latent_dim, image_shape=image_shape,
        base_ch=base_ch, hidden_dim_score=hidden_dim_score,
        n_score_layers=n_score_layers, t0=t0,
    ).to(device)
    n_params = sum(p.numel() for p in model_ts.parameters())
    print(f"  Total params: {n_params:,}")

    ckpt = load_checkpoint(model_ts, 'imagenet_twostage_full', device=device)
    if ckpt is not None:
        print("  Resumed Two-Stage (full) from checkpoint.")
        model_ts.freeze_stage1()
        recon_mse_ts = ckpt.get('recon_mse', 0.0)
        avg_loss_ts = ckpt.get('avg_loss', 0.0)
    else:
        ckpt_s1 = load_checkpoint(model_ts, 'imagenet_twostage_stage1',
                                  device=device)
        if ckpt_s1 is not None:
            print("  Resumed Stage 1 from checkpoint.")
            model_ts.freeze_stage1()
            recon_mse_ts = ckpt_s1.get('recon_mse', 0.0)
        else:
            print("\n--- Stage 1: GAN + LPIPS + EMA ---")
            train_stage1_gan(
                model_ts, dataset, n_steps=ae_steps,
                lr_g=1e-4, lr_d=1e-4, batch_size=batch_size,
                lpips_weight=lpips_weight, adv_weight=adv_weight,
                ema_decay=ema_decay, device=device,
            )

            with torch.no_grad():
                x0_test, _ = dataset.sample_clean(500)
                x0_test = x0_test.to(device)
                _, x_hat = model_ts.project(x0_test)
                recon_mse_ts = F.mse_loss(x_hat, x0_test).item()
                z_norm = model_ts.encode(x0_test)
                z_var = z_norm.var(dim=0).sort(descending=True).values
                active = (z_var > z_var[0] * 0.01).sum().item()
                print(f"\n  Reconstruction MSE: {recon_mse_ts:.6f}")
                print(f"  Active latent dims: {active}/{latent_dim}")

            model_ts.freeze_stage1()
            save_checkpoint(model_ts, 'imagenet_twostage_stage1',
                            {'recon_mse': recon_mse_ts})

        print("\n--- Stage 2: Latent Diffusion + EMA ---")
        train_diffusion_ema(
            model_ts, dataset, n_steps=diff_steps, lr=3e-4,
            batch_size=batch_size * 2, ema_decay=ema_decay,
            is_twostage=True, device=device,
        )

        eval_times = [0.01, 0.1, 0.3, 0.5, 0.8]
        total_loss = 0.0
        for t_eval in eval_times:
            with torch.no_grad():
                x0, _ = dataset.sample_clean(500)
                x0 = x0.to(device)
                z0 = model_ts.encode(x0)
                t_batch = torch.full((500,), t_eval, device=device)
                z_t, eps_z = model_ts.forward_diffusion_latent(z0, t_batch)
                eps_pred = model_ts.predict_eps_latent(z_t, t_batch)
                mse = F.mse_loss(eps_pred, eps_z).item()
                total_loss += mse
                print(f"  t={t_eval:.2f}: latent_eps_MSE={mse:.6f}")
        avg_loss_ts = total_loss / len(eval_times)

        save_checkpoint(model_ts, 'imagenet_twostage_full',
                        {'recon_mse': recon_mse_ts, 'avg_loss': avg_loss_ts})

    # ================================================================
    # MODEL 2: LDM-CNN
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODEL 2: LDM-CNN (fair comparison)")
    print("=" * 70)

    model_ldm = LDMImageNet(
        d=d, latent_dim=latent_dim, image_shape=image_shape,
        base_ch=base_ch, hidden_dim_score=hidden_dim_score,
        n_score_layers=n_score_layers, kl_weight=1e-4,
    ).to(device)

    ckpt = load_checkpoint(model_ldm, 'imagenet_ldm_full', device=device)
    if ckpt is not None:
        print("  Resumed LDM (full) from checkpoint.")
        model_ldm.freeze_ae()
        recon_mse_ldm = ckpt.get('recon_mse', 0.0)
        avg_loss_ldm = ckpt.get('avg_loss', 0.0)
    else:
        ckpt_ae = load_checkpoint(model_ldm, 'imagenet_ldm_ae', device=device)
        if ckpt_ae is not None:
            print("  Resumed LDM AE from checkpoint.")
            model_ldm.freeze_ae()
            recon_mse_ldm = ckpt_ae.get('recon_mse', 0.0)
        else:
            print("\n--- VAE-GAN + LPIPS + EMA ---")
            train_vae_gan(
                model_ldm, dataset, n_steps=ae_steps,
                lr_g=1e-4, lr_d=1e-4, batch_size=batch_size,
                lpips_weight=lpips_weight, adv_weight=adv_weight,
                kl_weight=1e-4, ema_decay=ema_decay, device=device,
            )

            with torch.no_grad():
                x0_test, _ = dataset.sample_clean(500)
                x0_test = x0_test.to(device)
                z_ldm = model_ldm.vae_encoder.encode_deterministic(x0_test)
                x_recon = model_ldm.decoder(z_ldm)
                recon_mse_ldm = F.mse_loss(x_recon, x0_test).item()
                print(f"\n  VAE Reconstruction MSE: {recon_mse_ldm:.6f}")

            model_ldm.freeze_ae()
            save_checkpoint(model_ldm, 'imagenet_ldm_ae',
                            {'recon_mse': recon_mse_ldm})

        print("\n--- Diffusion + EMA ---")
        train_diffusion_ema(
            model_ldm, dataset, n_steps=diff_steps, lr=3e-4,
            batch_size=batch_size * 2, ema_decay=ema_decay,
            is_twostage=False, device=device,
        )

        eval_times = [0.01, 0.1, 0.3, 0.5, 0.8]
        total_loss = 0.0
        for t_eval in eval_times:
            with torch.no_grad():
                x0, _ = dataset.sample_clean(500)
                x0 = x0.to(device)
                z0 = (model_ldm.vae_encoder.encode_deterministic(x0) -
                      model_ldm.z_mean) / model_ldm.z_std.clamp(min=1e-6)
                t_batch = torch.full((500,), t_eval, device=device)
                alpha_bar, sigma = model_ldm.noise_schedule(t_batch)
                eps_z = torch.randn_like(z0)
                z_t = (torch.sqrt(alpha_bar.unsqueeze(-1)) * z0 +
                       sigma.unsqueeze(-1) * eps_z)
                eps_pred = model_ldm.score_net(z_t, t_batch)
                mse = F.mse_loss(eps_pred, eps_z).item()
                total_loss += mse
                print(f"  t={t_eval:.2f}: latent_eps_MSE={mse:.6f}")
        avg_loss_ldm = total_loss / len(eval_times)

        save_checkpoint(model_ldm, 'imagenet_ldm_full',
                        {'recon_mse': recon_mse_ldm, 'avg_loss': avg_loss_ldm})

    results_ts = {'total_loss': avg_loss_ts, 'recon_error': recon_mse_ts}
    results_ldm = {'total_loss': avg_loss_ldm, 'recon_error': recon_mse_ldm}

    # ================================================================
    # GENERATE SAMPLES
    # ================================================================
    print("\n--- Generating Samples ---")
    with torch.no_grad():
        print("  Two-Stage: sampling...")
        x_gen_ts = model_ts.sample(32, n_steps=500, device=device,
                                   clamp=(0.0, 1.0)).cpu()
        plot_image_grid(x_gen_ts, nrow=8, title='Two-Stage Generated',
                        save_name='imagenet_gen_twostage.png',
                        img_size=img_size)

        print("  LDM-CNN: sampling...")
        x_gen_ldm = model_ldm.sample(32, n_steps=500, device=device,
                                     clamp=(0.0, 1.0)).cpu()
        plot_image_grid(x_gen_ldm, nrow=8, title='LDM-CNN Generated',
                        save_name='imagenet_gen_ldm_cnn.png',
                        img_size=img_size)

        x_real, _ = dataset.sample_clean(32)
        plot_image_grid(x_real, nrow=8, title='Real ImageNet-256',
                        save_name='imagenet_real.png', img_size=img_size)

    # ================================================================
    # FID
    # ================================================================
    if FID_AVAILABLE:
        print("\n--- FID Evaluation ---")
        n_fid = 2048
        fid_batch = 32

        real_imgs = []
        for _ in range(n_fid // fid_batch):
            x0, _ = dataset.sample_clean(fid_batch)
            imgs = x0.view(-1, 3, img_size, img_size).clamp(0, 1)
            real_imgs.append((imgs * 255).to(torch.uint8))
        real_imgs = torch.cat(real_imgs, dim=0)[:n_fid]

        with torch.no_grad():
            print(f"  Generating {n_fid} samples for FID...")
            gen_ts, gen_ldm = [], []
            for i in range(n_fid // fid_batch):
                x_ts = model_ts.sample(fid_batch, n_steps=500, device=device,
                                       clamp=(0.0, 1.0)).cpu()
                gen_ts.append((x_ts.view(-1, 3, img_size, img_size).clamp(0, 1)
                               * 255).to(torch.uint8))
                x_ldm = model_ldm.sample(fid_batch, n_steps=500, device=device,
                                         clamp=(0.0, 1.0)).cpu()
                gen_ldm.append((x_ldm.view(-1, 3, img_size, img_size).clamp(0, 1)
                                * 255).to(torch.uint8))
                if (i + 1) % 16 == 0:
                    print(f"    {(i+1)*fid_batch}/{n_fid}")
            gen_ts = torch.cat(gen_ts, dim=0)[:n_fid]
            gen_ldm = torch.cat(gen_ldm, dim=0)[:n_fid]

        fid_metric = FrechetInceptionDistance(feature=2048,
                                             normalize=False).to(device)
        fid_metric.update(real_imgs.to(device), real=True)
        fid_metric.update(gen_ts.to(device), real=False)
        fid_ts = fid_metric.compute().item()
        print(f"  Two-Stage FID: {fid_ts:.2f}")

        fid_metric.reset()
        fid_metric.update(real_imgs.to(device), real=True)
        fid_metric.update(gen_ldm.to(device), real=False)
        fid_ldm = fid_metric.compute().item()
        print(f"  LDM-CNN FID: {fid_ldm:.2f}")

        del fid_metric, real_imgs, gen_ts, gen_ldm
        torch.cuda.empty_cache()
    else:
        fid_ts = float('nan')
        fid_ldm = float('nan')

    # ================================================================
    # RECONSTRUCTION
    # ================================================================
    print("\n--- Reconstruction Comparison ---")
    with torch.no_grad():
        x_test, _ = dataset.sample_clean(8)
        x_test_d = x_test.to(device)
        _, x_recon_ts = model_ts.project(x_test_d)
        z_ldm = model_ldm.vae_encoder.encode_deterministic(x_test_d)
        x_recon_ldm = model_ldm.decoder(z_ldm)
        x_recon_ts = x_recon_ts.cpu()
        x_recon_ldm = x_recon_ldm.cpu()

        fig, axes = plt.subplots(3, 8, figsize=(24, 9))
        for j in range(8):
            for row, img in enumerate([x_test[j], x_recon_ts[j],
                                       x_recon_ldm[j]]):
                img_np = (img.view(3, img_size, img_size)
                          .permute(1, 2, 0).clamp(0, 1).numpy())
                axes[row, j].imshow(img_np)
                axes[row, j].axis('off')
        axes[0, 0].set_ylabel('Original', fontsize=11, rotation=0,
                              labelpad=60, va='center')
        axes[1, 0].set_ylabel('Two-Stage', fontsize=11, rotation=0,
                              labelpad=60, va='center')
        axes[2, 0].set_ylabel('LDM-CNN', fontsize=11, rotation=0,
                              labelpad=60, va='center')
        plt.suptitle('ImageNet-256 Reconstruction', fontsize=14)
        plt.tight_layout()
        path = os.path.join(FIG_DIR, 'imagenet_reconstruction.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'=' * 70}")
    print("ImageNet-256 SUMMARY")
    print("=" * 70)
    print(f"  Intrinsic dim (95% PCA): k={k_95}")
    print(f"\n  {'Model':<30} {'Latent':<8} {'Eps Loss':<12} "
          f"{'Recon MSE':<12} {'FID':<10}")
    print("  " + "-" * 72)
    print(f"  {'Two-Stage (ours)':<30} {latent_dim:<8} "
          f"{results_ts['total_loss']:<12.6f} "
          f"{results_ts['recon_error']:<12.6f} {fid_ts:<10.2f}")
    print(f"  {'LDM-CNN (fair)':<30} {latent_dim:<8} "
          f"{results_ldm['total_loss']:<12.6f} "
          f"{results_ldm['recon_error']:<12.6f} {fid_ldm:<10.2f}")


if __name__ == '__main__':
    run_experiment()
