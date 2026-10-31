"""
ImageNet-256 — Large spatial latent (32x32x8 = 8192 dims)

Stage 1 only: Train AE for both Two-Stage and LDM, measure rFID.
If rFID is good, proceed to diffusion (Stage 2).

Architecture:
  256x256x3 → Encoder (3 downsamples) → 32x32x8 → Decoder → 256x256x3
  Compression: 196608 → 8192 (24:1)
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
        s, sh = self.t_proj(t_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        h = self.norm2(h) * (1 + s) + sh
        h = self.act(h)
        h = self.conv2(h)
        return x + h


# ================================================================
# Encoder/Decoder for 256x256 → 32x32 (3 downsamples)
# ================================================================

class Encoder256(nn.Module):
    """256x256x3 → 32x32 x latent_ch."""
    def __init__(self, image_shape, latent_ch=8, base_ch=64):
        super().__init__()
        C = image_shape[0]
        self.image_shape = image_shape
        ch = base_ch

        self.net = nn.Sequential(
            nn.Conv2d(C, ch, 3, padding=1),
            # Level 1: 256→128
            ResBlock2d(ch), ResBlock2d(ch),
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
            # Level 2: 128→64
            ResBlock2d(ch * 2), ResBlock2d(ch * 2),
            nn.Conv2d(ch * 2, ch * 4, 3, stride=2, padding=1),
            # Level 3: 64→32
            ResBlock2d(ch * 4), ResBlock2d(ch * 4),
            nn.Conv2d(ch * 4, ch * 4, 3, stride=2, padding=1),
            # Final
            ResBlock2d(ch * 4),
            nn.GroupNorm(min(32, ch * 4), ch * 4),
            nn.SiLU(),
            nn.Conv2d(ch * 4, latent_ch, 1),
        )
        with torch.no_grad():
            out = self.net(torch.zeros(1, *image_shape))
            self.spatial_shape = out.shape[1:]
        n_p = sum(p.numel() for p in self.parameters())
        print(f"    Encoder: {n_p:,} params, output {list(self.spatial_shape)}")

    def forward(self, x):
        return self.net(x.view(-1, *self.image_shape))


class Decoder256(nn.Module):
    """32x32 x latent_ch → 256x256x3."""
    def __init__(self, latent_ch, image_shape, base_ch=64):
        super().__init__()
        C = image_shape[0]
        self.image_shape = image_shape
        self.d = C * image_shape[1] * image_shape[2]
        ch = base_ch * 4

        self.net = nn.Sequential(
            nn.Conv2d(latent_ch, ch, 1), nn.SiLU(),
            ResBlock2d(ch),
            # Level 3: 32→64
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch, ch, 3, padding=1),
            ResBlock2d(ch), ResBlock2d(ch),
            # Level 2: 64→128
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch, ch // 2, 3, padding=1),
            ResBlock2d(ch // 2), ResBlock2d(ch // 2),
            # Level 1: 128→256
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch // 2, ch // 4, 3, padding=1),
            ResBlock2d(ch // 4), ResBlock2d(ch // 4),
            # Output
            nn.GroupNorm(min(32, ch // 4), ch // 4),
            nn.SiLU(),
            nn.Conv2d(ch // 4, C, 3, padding=1),
            nn.Sigmoid(),
        )
        n_p = sum(p.numel() for p in self.parameters())
        print(f"    Decoder: {n_p:,} params")

    def forward(self, z):
        out = self.net(z)
        return out[:, :, :self.image_shape[1], :self.image_shape[2]].reshape(-1, self.d)


class VAEEncoder256(nn.Module):
    def __init__(self, image_shape, latent_ch=8, base_ch=64):
        super().__init__()
        C = image_shape[0]
        self.image_shape = image_shape
        ch = base_ch

        self.backbone = nn.Sequential(
            nn.Conv2d(C, ch, 3, padding=1),
            ResBlock2d(ch), ResBlock2d(ch),
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
            ResBlock2d(ch * 2), ResBlock2d(ch * 2),
            nn.Conv2d(ch * 2, ch * 4, 3, stride=2, padding=1),
            ResBlock2d(ch * 4), ResBlock2d(ch * 4),
            nn.Conv2d(ch * 4, ch * 4, 3, stride=2, padding=1),
            ResBlock2d(ch * 4),
            nn.GroupNorm(min(32, ch * 4), ch * 4),
            nn.SiLU(),
        )
        self.conv_mu = nn.Conv2d(ch * 4, latent_ch, 1)
        self.conv_logvar = nn.Conv2d(ch * 4, latent_ch, 1)

    def forward(self, x):
        h = self.backbone(x.view(-1, *self.image_shape))
        return self.conv_mu(h), self.conv_logvar(h)

    def encode_deterministic(self, x):
        mu, _ = self.forward(x)
        return mu


# ================================================================
# U-Net for 32x32 latent
# ================================================================

class UNetScore32(nn.Module):
    def __init__(self, latent_ch=8, base_ch=128, time_embed_dim=256):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, base_ch * 4), nn.SiLU(),
            nn.Linear(base_ch * 4, base_ch * 4),
        )
        t_dim = base_ch * 4

        self.conv_in = nn.Conv2d(latent_ch, base_ch, 3, padding=1)

        # 32→16→8→4
        self.d1_r1 = TimeCondResBlock(base_ch, t_dim)
        self.d1_r2 = TimeCondResBlock(base_ch, t_dim)
        self.d1_down = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)

        self.d2_r1 = TimeCondResBlock(base_ch * 2, t_dim)
        self.d2_r2 = TimeCondResBlock(base_ch * 2, t_dim)
        self.d2_down = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)

        self.d3_r1 = TimeCondResBlock(base_ch * 4, t_dim)
        self.d3_r2 = TimeCondResBlock(base_ch * 4, t_dim)
        self.d3_down = nn.Conv2d(base_ch * 4, base_ch * 4, 3, stride=2, padding=1)

        self.mid_r1 = TimeCondResBlock(base_ch * 4, t_dim)
        self.mid_r2 = TimeCondResBlock(base_ch * 4, t_dim)

        self.u3_up = nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1)
        self.u3_r1 = TimeCondResBlock(base_ch * 8, t_dim)
        self.u3_r2 = TimeCondResBlock(base_ch * 8, t_dim)
        self.u3_out = nn.Conv2d(base_ch * 8, base_ch * 4, 1)

        self.u2_up = nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1)
        self.u2_r1 = TimeCondResBlock(base_ch * 4, t_dim)
        self.u2_r2 = TimeCondResBlock(base_ch * 4, t_dim)
        self.u2_out = nn.Conv2d(base_ch * 4, base_ch * 2, 1)

        self.u1_up = nn.Conv2d(base_ch * 2, base_ch, 3, padding=1)
        self.u1_r1 = TimeCondResBlock(base_ch * 2, t_dim)
        self.u1_r2 = TimeCondResBlock(base_ch * 2, t_dim)
        self.u1_out = nn.Conv2d(base_ch * 2, base_ch, 1)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(min(32, base_ch), base_ch), nn.SiLU(),
            nn.Conv2d(base_ch, latent_ch, 3, padding=1),
        )
        n_p = sum(p.numel() for p in self.parameters())
        print(f"    UNetScore32: {n_p:,} params")

    def sinusoidal_embedding(self, t):
        half = self.time_embed_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        if t.dim() == 0: t = t.unsqueeze(0)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, z_t, t):
        te = self.time_mlp(self.sinusoidal_embedding(t))
        h = self.conv_in(z_t)

        h1 = self.d1_r2(self.d1_r1(h, te), te)
        h = self.d1_down(h1)
        h2 = self.d2_r2(self.d2_r1(h, te), te)
        h = self.d2_down(h2)
        h3 = self.d3_r2(self.d3_r1(h, te), te)
        h = self.d3_down(h3)

        h = self.mid_r2(self.mid_r1(h, te), te)

        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = torch.cat([self.u3_up(h), h3], 1)
        h = self.u3_out(self.u3_r2(self.u3_r1(h, te), te))

        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = torch.cat([self.u2_up(h), h2], 1)
        h = self.u2_out(self.u2_r2(self.u2_r1(h, te), te))

        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = torch.cat([self.u1_up(h), h1], 1)
        h = self.u1_out(self.u1_r2(self.u1_r1(h, te), te))

        return self.conv_out(h)


# ================================================================
# Helpers
# ================================================================

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, n_layers=4):
        super().__init__()
        layers = [nn.Conv2d(in_ch, base_ch, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        ch = base_ch
        for _ in range(1, n_layers):
            ch_n = min(ch * 2, 512)
            layers += [nn.Conv2d(ch, ch_n, 4, 2, 1, bias=False),
                       nn.InstanceNorm2d(ch_n), nn.LeakyReLU(0.2, True)]
            ch = ch_n
        layers += [nn.Conv2d(ch, ch, 4, 1, 1, bias=False),
                   nn.InstanceNorm2d(ch), nn.LeakyReLU(0.2, True),
                   nn.Conv2d(ch, 1, 4, 1, 1)]
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

def hinge_d(r, f): return (F.relu(1-r).mean() + F.relu(1+f).mean()) * 0.5
def hinge_g(f): return -f.mean()

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters(): p.requires_grad = False
    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1-self.decay)
        for s, m in zip(self.shadow.buffers(), model.buffers()):
            s.data.copy_(m.data)
    def apply(self, model):
        model.load_state_dict(self.shadow.state_dict())

def save_ckpt(model, name, extra=None):
    path = os.path.join(CKPT_DIR, f'{name}.pt')
    state = {'model': model.state_dict()}
    if extra: state.update(extra)
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

def plot_grid(images, nrow=8, title='', save_name=None, img_size=256):
    n = min(len(images), nrow * nrow)
    nr = (n + nrow - 1) // nrow
    fig, axes = plt.subplots(nr, nrow, figsize=(nrow * 2, nr * 2))
    axes = np.atleast_2d(axes)
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            idx = i * nrow + j
            if idx < n:
                img = images[idx].detach().cpu().clamp(0, 1)
                axes[i, j].imshow(img.view(3, img_size, img_size).permute(1, 2, 0).numpy())
            axes[i, j].axis('off')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_name:
        path = os.path.join(FIG_DIR, save_name)
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")
    else:
        plt.close()


# ================================================================
# Model wrappers
# ================================================================

class TwoStageModel(nn.Module):
    def __init__(self, d, image_shape, latent_ch, base_ch, unet_ch, t0=0.005):
        super().__init__()
        self.d, self.t0, self.T = d, t0, 0.999
        self.image_shape = image_shape
        self.latent_ch = latent_ch
        self.beta_min, self.beta_max = 0.1, 20.0
        self.encoder = Encoder256(image_shape, latent_ch, base_ch)
        self.decoder = Decoder256(latent_ch, image_shape, base_ch)
        self.score_net = UNetScore32(latent_ch, unet_ch)
        self.spatial_shape = self.encoder.spatial_shape
        self.register_buffer('z_mean', torch.zeros(latent_ch, 1, 1))
        self.register_buffer('z_std', torch.ones(latent_ch, 1, 1))

    def noise_schedule(self, t):
        t = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
        lmc = -0.25*t**2*(self.beta_max-self.beta_min) - 0.5*t*self.beta_min
        ab = torch.exp(2*lmc)
        return ab, torch.sqrt((1-ab).clamp(min=1e-8))

    def encode(self, x): return (self.encoder(x) - self.z_mean) / self.z_std.clamp(min=1e-6)
    def decode(self, z): return self.decoder(z * self.z_std + self.z_mean)
    def project(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)

    def freeze_stage1(self):
        for p in self.encoder.parameters(): p.requires_grad = False
        for p in self.decoder.parameters(): p.requires_grad = False

    def forward_diffusion(self, z0, t):
        ab, sig = self.noise_schedule(t)
        eps = torch.randn_like(z0)
        return torch.sqrt(ab).view(-1,1,1,1)*z0 + sig.view(-1,1,1,1)*eps, eps

    @torch.no_grad()
    def sample(self, n, n_steps=500, device='cpu', clamp=None):
        _, c, h, w = 1, *self.spatial_shape
        z = torch.randn(n, c, h, w, device=device)
        dt = self.T / n_steps
        for i in range(n_steps, 0, -1):
            t = torch.full((n,), i*dt, device=device)
            ab, sig = self.noise_schedule(t)
            ep = self.score_net(z, t)
            x0p = (z - sig.view(-1,1,1,1)*ep) / torch.sqrt(ab).view(-1,1,1,1).clamp(min=1e-8)
            abp, sigp = self.noise_schedule(t - dt)
            z = torch.sqrt(abp).view(-1,1,1,1)*x0p + sigp.view(-1,1,1,1)*torch.randn_like(z)
        x = self.decode(z)
        return x.clamp(*clamp) if clamp else x


class LDMModel(nn.Module):
    def __init__(self, d, image_shape, latent_ch, base_ch, unet_ch, kl_weight=1e-4):
        super().__init__()
        self.d = d
        self.image_shape = image_shape
        self.latent_ch = latent_ch
        self.kl_weight = kl_weight
        self.beta_min, self.beta_max = 0.1, 20.0
        self.vae_encoder = VAEEncoder256(image_shape, latent_ch, base_ch)
        self.decoder = Decoder256(latent_ch, image_shape, base_ch)
        self.score_net = UNetScore32(latent_ch, unet_ch)
        self.register_buffer('z_mean', torch.zeros(latent_ch, 1, 1))
        self.register_buffer('z_std', torch.ones(latent_ch, 1, 1))

    def noise_schedule(self, t):
        t = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
        lmc = -0.25*t**2*(self.beta_max-self.beta_min) - 0.5*t*self.beta_min
        ab = torch.exp(2*lmc)
        return ab, torch.sqrt((1-ab).clamp(min=1e-8))

    def freeze_ae(self):
        for p in self.vae_encoder.parameters(): p.requires_grad = False
        for p in self.decoder.parameters(): p.requires_grad = False

    @torch.no_grad()
    def sample(self, n, n_steps=500, device='cpu', clamp=None):
        dummy = torch.zeros(1, *self.image_shape, device=device)
        mu, _ = self.vae_encoder(dummy.view(1, -1))
        _, c, h, w = mu.shape
        z = torch.randn(n, c, h, w, device=device)
        dt = 0.999 / n_steps
        for i in range(n_steps, 0, -1):
            t = torch.full((n,), i*dt, device=device)
            ab, sig = self.noise_schedule(t)
            ep = self.score_net(z, t)
            x0p = (z - sig.view(-1,1,1,1)*ep) / torch.sqrt(ab).view(-1,1,1,1).clamp(min=1e-8)
            abp, sigp = self.noise_schedule(t - dt)
            z = torch.sqrt(abp).view(-1,1,1,1)*x0p + sigp.view(-1,1,1,1)*torch.randn_like(z)
        x = self.decoder(z * self.z_std + self.z_mean)
        return x.clamp(*clamp) if clamp else x


# ================================================================
# Training
# ================================================================

def train_ae(model, dataset, n_steps, batch_size, lpips_weight, adv_weight,
             ema_decay, is_twostage, device):
    img_size = model.image_shape[1]
    sigma_t0 = model.t0 if is_twostage else 0.0

    loss_fn = None
    if LPIPS_AVAILABLE:
        loss_fn = lpips.LPIPS(net='vgg').to(device)
        loss_fn.eval()
        for p in loss_fn.parameters(): p.requires_grad = False

    disc = PatchDiscriminator(3, 64, 4).to(device)
    if is_twostage:
        gp = list(model.encoder.parameters()) + list(model.decoder.parameters())
    else:
        gp = list(model.vae_encoder.parameters()) + list(model.decoder.parameters())
    opt_g = torch.optim.Adam(gp, lr=1e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))
    ema = EMA(model, ema_decay)
    ds = min(1000, n_steps // 10)

    tag = "Stage1" if is_twostage else "VAE"
    print(f"  {tag}: {n_steps} steps, adv={adv_weight}, disc@{ds}")

    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)

        if is_twostage:
            x_in = x0 + sigma_t0 * torch.randn_like(x0)
            z, xh = model.project(x_in)
            kl = torch.tensor(0.0, device=device)
        else:
            mu, lv = model.vae_encoder(x0)
            z = mu + torch.exp(0.5*lv) * torch.randn_like(mu)
            xh = model.decoder(z)
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())

        x0i = x0.view(-1, 3, img_size, img_size)
        xhi = xh.view(-1, 3, img_size, img_size)
        mse = F.mse_loss(xh, x0)
        lp = loss_fn(xhi*2-1, x0i*2-1).mean() if loss_fn else torch.tensor(0.0, device=device)
        ga = hinge_g(disc(xhi)) if step >= ds else torch.tensor(0.0, device=device)

        gl = mse + lpips_weight * lp + adv_weight * ga
        if not is_twostage:
            gl = gl + model.kl_weight * kl

        opt_g.zero_grad()
        gl.backward()
        torch.nn.utils.clip_grad_norm_(gp, 1.0)
        opt_g.step()

        if step >= ds:
            with torch.no_grad():
                if is_twostage:
                    _, xhd = model.project(x_in)
                else:
                    mud, _ = model.vae_encoder(x0)
                    xhd = model.decoder(mud)
                xhdi = xhd.view(-1, 3, img_size, img_size)
            dl = hinge_d(disc(x0i), disc(xhdi))
            opt_d.zero_grad()
            dl.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            opt_d.step()
        else:
            dl = torch.tensor(0.0)

        ema.update(model)

        if (step + 1) % 500 == 0:
            ex = f" kl={kl.item():.4f}" if not is_twostage else ""
            print(f"  [{step+1}/{n_steps}] g={gl.item():.5f} "
                  f"mse={mse.item():.5f} lp={lp.item():.5f} "
                  f"adv={ga.item():.5f} d={dl.item():.5f}{ex}")

    ema.apply(model)

    with torch.no_grad():
        zs = []
        for _ in range(20):
            x0, _ = dataset.sample_clean(min(batch_size, 200))
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
        zt = torch.sqrt(ab).view(-1,1,1,1)*z0 + sig.view(-1,1,1,1)*eps
        loss = F.mse_loss(model.score_net(zt, t), eps)

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

    img_size = 256
    image_shape = (3, img_size, img_size)
    d = 3 * img_size * img_size  # 196608

    latent_ch = 8          # 32x32x8 = 8192 dims
    base_ch = 64           # encoder/decoder
    unet_ch = 128          # U-Net score
    t0 = 0.005

    ae_steps = 100000
    diff_steps = 300000
    batch_size = 8         # small for 256x256
    lpips_weight = 1.0
    adv_weight = 0.1
    ema_decay = 0.9999

    print("=" * 70)
    print(f"ImageNet-256 — Large Latent (32x32x{latent_ch} = {32*32*latent_ch})")
    print(f"  Compression: {d} → {32*32*latent_ch} ({d/(32*32*latent_ch):.0f}:1)")
    print(f"  enc/dec base_ch={base_ch}, U-Net base_ch={unet_ch}")
    print(f"  AE={ae_steps}, Diff={diff_steps}, batch={batch_size}")
    print(f"  LPIPS={lpips_weight}, adv={adv_weight}, EMA={ema_decay}")
    print("=" * 70)

    dataset = ImageNet256Dataset(max_samples=50000)

    print("\n--- PCA ---")
    ev = dataset.pca_explained_variance
    cv = torch.cumsum(ev, dim=0) / ev.sum()
    for th in [0.90, 0.95, 0.99]:
        print(f"  {th*100:.0f}% at k={(cv < th).sum().item() + 1}")
    k95 = (cv < 0.95).sum().item() + 1

    # ================================================================
    # Two-Stage
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODEL 1: Two-Stage")
    print("=" * 70)

    m_ts = TwoStageModel(d, image_shape, latent_ch, base_ch, unet_ch, t0).to(device)

    ck = load_ckpt(m_ts, 'imagenet_ll_ts_s1', device)
    if ck:
        m_ts.freeze_stage1()
        r_ts = ck.get('recon', 0.0)
    else:
        print("\n--- Stage 1: GAN+LPIPS+EMA ---")
        train_ae(m_ts, dataset, ae_steps, batch_size, lpips_weight,
                 adv_weight, ema_decay, True, device)
        with torch.no_grad():
            x0, _ = dataset.sample_clean(1000)
            x0 = x0.to(device)
            _, xh = m_ts.project(x0)
            r_ts = F.mse_loss(xh, x0).item()
            print(f"\n  Recon MSE: {r_ts:.6f}")
        m_ts.freeze_stage1()
        save_ckpt(m_ts, 'imagenet_ll_ts_s1', {'recon': r_ts})

    # ================================================================
    # LDM
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODEL 2: LDM")
    print("=" * 70)

    m_ldm = LDMModel(d, image_shape, latent_ch, base_ch, unet_ch).to(device)

    ck = load_ckpt(m_ldm, 'imagenet_ll_ldm_ae', device)
    if ck:
        m_ldm.freeze_ae()
        r_ldm = ck.get('recon', 0.0)
    else:
        print("\n--- VAE-GAN+LPIPS+EMA ---")
        train_ae(m_ldm, dataset, ae_steps, batch_size, lpips_weight,
                 adv_weight, ema_decay, False, device)
        with torch.no_grad():
            x0, _ = dataset.sample_clean(1000)
            x0 = x0.to(device)
            z = m_ldm.vae_encoder.encode_deterministic(x0)
            xr = m_ldm.decoder(z)
            r_ldm = F.mse_loss(xr, x0).item()
            print(f"\n  Recon MSE: {r_ldm:.6f}")
        m_ldm.freeze_ae()
        save_ckpt(m_ldm, 'imagenet_ll_ldm_ae', {'recon': r_ldm})

    # ================================================================
    # rFID
    # ================================================================
    print("\n--- rFID (Reconstruction FID) ---")
    if FID_AVAILABLE:
        nf, fb = 2048, 8
        reals, recs_ts, recs_ldm = [], [], []
        with torch.no_grad():
            for i in range(nf // fb):
                x0, _ = dataset.sample_clean(fb)
                x0 = x0.to(device)
                reals.append((x0.view(-1, 3, img_size, img_size).clamp(0,1)*255).to(torch.uint8).cpu())
                _, xrt = m_ts.project(x0)
                recs_ts.append((xrt.view(-1, 3, img_size, img_size).clamp(0,1)*255).to(torch.uint8).cpu())
                zl = m_ldm.vae_encoder.encode_deterministic(x0)
                xrl = m_ldm.decoder(zl)
                recs_ldm.append((xrl.view(-1, 3, img_size, img_size).clamp(0,1)*255).to(torch.uint8).cpu())
                if (i+1) % 32 == 0:
                    print(f"    {(i+1)*fb}/{nf}")

        reals = torch.cat(reals)[:nf]
        recs_ts = torch.cat(recs_ts)[:nf]
        recs_ldm = torch.cat(recs_ldm)[:nf]

        fid_m = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        fid_m.update(reals.to(device), real=True)
        fid_m.update(recs_ts.to(device), real=False)
        rfid_ts = fid_m.compute().item()
        print(f"  Two-Stage rFID: {rfid_ts:.2f}")

        fid_m.reset()
        fid_m.update(reals.to(device), real=True)
        fid_m.update(recs_ldm.to(device), real=False)
        rfid_ldm = fid_m.compute().item()
        print(f"  LDM rFID: {rfid_ldm:.2f}")

        del fid_m, reals, recs_ts, recs_ldm
        torch.cuda.empty_cache()
    else:
        rfid_ts = rfid_ldm = float('nan')

    # Reconstruction visualization
    print("\n--- Reconstruction Samples ---")
    with torch.no_grad():
        xt, _ = dataset.sample_clean(8)
        xd = xt.to(device)
        _, xrt = m_ts.project(xd)
        zl = m_ldm.vae_encoder.encode_deterministic(xd)
        xrl = m_ldm.decoder(zl)

        fig, axes = plt.subplots(3, 8, figsize=(24, 9))
        for j in range(8):
            for r, im in enumerate([xt[j], xrt.cpu()[j], xrl.cpu()[j]]):
                axes[r, j].imshow(im.view(3, img_size, img_size).permute(1,2,0).clamp(0,1).numpy())
                axes[r, j].axis('off')
        for r, lb in enumerate(['Original', 'Two-Stage', 'LDM']):
            axes[r, 0].set_ylabel(lb, fontsize=11, rotation=0, labelpad=70, va='center')
        plt.suptitle(f'ImageNet-256 Reconstruction (32x32x{latent_ch})', fontsize=14)
        plt.tight_layout()
        p = os.path.join(FIG_DIR, 'imagenet_ll_reconstruction.png')
        plt.savefig(p, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {p}")

    # ================================================================
    # Stage 2: Diffusion (if AE checkpoints exist, continue to diffusion)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Stage 2: Diffusion Training")
    print("=" * 70)

    ck_ts_full = load_ckpt(m_ts, 'imagenet_ll_ts_full', device)
    if ck_ts_full:
        e_ts = ck_ts_full.get('eps', 0.0)
    else:
        print("\n--- Two-Stage Diffusion ---")
        train_diffusion_ema(m_ts, dataset, diff_steps, batch_size * 2,
                            ema_decay, True, device)
        times = [0.01, 0.1, 0.3, 0.5, 0.8]
        tot = 0
        for tv in times:
            with torch.no_grad():
                x0, _ = dataset.sample_clean(1000)
                x0 = x0.to(device)
                z0 = m_ts.encode(x0)
                tb = torch.full((1000,), tv, device=device)
                zt, ep = m_ts.forward_diffusion(z0, tb)
                mse = F.mse_loss(m_ts.score_net(zt, tb), ep).item()
                tot += mse
                print(f"  t={tv:.2f}: eps={mse:.6f}")
        e_ts = tot / len(times)
        save_ckpt(m_ts, 'imagenet_ll_ts_full', {'recon': r_ts, 'eps': e_ts})

    ck_ldm_full = load_ckpt(m_ldm, 'imagenet_ll_ldm_full', device)
    if ck_ldm_full:
        e_ldm = ck_ldm_full.get('eps', 0.0)
    else:
        print("\n--- LDM Diffusion ---")
        train_diffusion_ema(m_ldm, dataset, diff_steps, batch_size * 2,
                            ema_decay, False, device)
        times = [0.01, 0.1, 0.3, 0.5, 0.8]
        tot = 0
        for tv in times:
            with torch.no_grad():
                x0, _ = dataset.sample_clean(1000)
                x0 = x0.to(device)
                z0 = (m_ldm.vae_encoder.encode_deterministic(x0) -
                      m_ldm.z_mean) / m_ldm.z_std.clamp(min=1e-6)
                tb = torch.full((1000,), tv, device=device)
                ab, sig = m_ldm.noise_schedule(tb)
                ep = torch.randn_like(z0)
                zt = torch.sqrt(ab).view(-1,1,1,1)*z0 + sig.view(-1,1,1,1)*ep
                mse = F.mse_loss(m_ldm.score_net(zt, tb), ep).item()
                tot += mse
                print(f"  t={tv:.2f}: eps={mse:.6f}")
        e_ldm = tot / len(times)
        save_ckpt(m_ldm, 'imagenet_ll_ldm_full', {'recon': r_ldm, 'eps': e_ldm})

    # FID
    fid_ts = fid_ldm = float('nan')
    if FID_AVAILABLE:
        print("\n--- Generation FID ---")
        nf, fb = 2048, 8
        reals = []
        for _ in range(nf // fb):
            x0, _ = dataset.sample_clean(fb)
            reals.append((x0.view(-1, 3, img_size, img_size).clamp(0,1)*255).to(torch.uint8))
        reals = torch.cat(reals)[:nf]

        for name, model in [('Two-Stage', m_ts), ('LDM', m_ldm)]:
            print(f"  {name}: generating {nf}...")
            gens = []
            with torch.no_grad():
                for i in range(nf // fb):
                    x = model.sample(fb, 250, device, (0,1)).cpu()
                    gens.append((x.view(-1, 3, img_size, img_size).clamp(0,1)*255).to(torch.uint8))
                    if (i+1) % 32 == 0: print(f"    {(i+1)*fb}/{nf}")
            gens = torch.cat(gens)[:nf]
            fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
            fid.update(reals.to(device), real=True)
            fid.update(gens.to(device), real=False)
            v = fid.compute().item()
            if name == 'Two-Stage': fid_ts = v
            else: fid_ldm = v
            print(f"  {name} FID: {v:.2f}")
            del fid, gens
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 70}")
    print(f"ImageNet-256 SUMMARY (32x32x{latent_ch} = {32*32*latent_ch}, "
          f"{d/(32*32*latent_ch):.0f}:1)")
    print("=" * 70)
    print(f"  k_95={k95}")
    print(f"\n  {'Model':<20} {'Recon MSE':<12} {'rFID':<10} {'Eps Loss':<12} {'FID':<10}")
    print("  " + "-" * 65)
    print(f"  {'Two-Stage':<20} {r_ts:<12.6f} {rfid_ts:<10.2f} {e_ts:<12.6f} {fid_ts:<10.2f}")
    print(f"  {'LDM':<20} {r_ldm:<12.6f} {rfid_ldm:<10.2f} {e_ldm:<12.6f} {fid_ldm:<10.2f}")


if __name__ == '__main__':
    run_experiment()
