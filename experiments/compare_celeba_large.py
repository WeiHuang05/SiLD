"""
CelebA Large-Scale Experiment — single GPU, lazy loading

Config:
  - CelebA full: ~200k samples (64x64), lazy loaded from disk
  - Spatial latent: 32x32x4 = 4096 dims (3:1 compression)
  - Encoder/Decoder: base_ch=128, 1 downsample
  - Score U-Net: base_ch=256, 3 levels (32→16→8→4)
  - GAN + LPIPS(1.0) + EMA(0.9999)
  - AE: 200k steps, Diffusion: 500k steps
  - Checkpoints every 50k steps
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

from data.image_data import CelebALazyDataset

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
            nn.GroupNorm(min(32, ch), ch), nn.SiLU(),
            nn.Conv2d(ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch), nn.SiLU(),
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
        return x + self.conv2(h)


# ================================================================
# Encoder / Decoder (64→32, 1 downsample)
# ================================================================

class Encoder(nn.Module):
    def __init__(self, image_shape, latent_ch=4, base_ch=128):
        super().__init__()
        C = image_shape[0]
        self.image_shape = image_shape
        ch = base_ch
        self.net = nn.Sequential(
            nn.Conv2d(C, ch, 3, padding=1),
            ResBlock2d(ch), ResBlock2d(ch), ResBlock2d(ch),
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
            ResBlock2d(ch * 2), ResBlock2d(ch * 2), ResBlock2d(ch * 2),
            nn.GroupNorm(min(32, ch * 2), ch * 2), nn.SiLU(),
            nn.Conv2d(ch * 2, latent_ch, 1),
        )
        with torch.no_grad():
            self.spatial_shape = self.net(torch.zeros(1, *image_shape)).shape[1:]
        print(f"    Encoder: {sum(p.numel() for p in self.parameters()):,} params, "
              f"output {list(self.spatial_shape)}")

    def forward(self, x):
        return self.net(x.view(-1, *self.image_shape))


class Decoder(nn.Module):
    def __init__(self, latent_ch, image_shape, base_ch=128):
        super().__init__()
        C = image_shape[0]
        self.image_shape = image_shape
        self.d = C * image_shape[1] * image_shape[2]
        ch = base_ch * 2
        self.net = nn.Sequential(
            nn.Conv2d(latent_ch, ch, 1), nn.SiLU(),
            ResBlock2d(ch), ResBlock2d(ch), ResBlock2d(ch),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch, ch // 2, 3, padding=1),
            ResBlock2d(ch // 2), ResBlock2d(ch // 2), ResBlock2d(ch // 2),
            nn.GroupNorm(min(32, ch // 2), ch // 2), nn.SiLU(),
            nn.Conv2d(ch // 2, C, 3, padding=1),
            nn.Sigmoid(),
        )
        print(f"    Decoder: {sum(p.numel() for p in self.parameters()):,} params")

    def forward(self, z):
        out = self.net(z)
        return out[:, :, :self.image_shape[1], :self.image_shape[2]].reshape(-1, self.d)


class VAEEncoder(nn.Module):
    def __init__(self, image_shape, latent_ch=4, base_ch=128):
        super().__init__()
        C = image_shape[0]
        self.image_shape = image_shape
        ch = base_ch
        self.backbone = nn.Sequential(
            nn.Conv2d(C, ch, 3, padding=1),
            ResBlock2d(ch), ResBlock2d(ch), ResBlock2d(ch),
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
            ResBlock2d(ch * 2), ResBlock2d(ch * 2), ResBlock2d(ch * 2),
            nn.GroupNorm(min(32, ch * 2), ch * 2), nn.SiLU(),
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
# U-Net Score for 32x32 latent (larger: base_ch=256)
# ================================================================

class SelfAttention2d(nn.Module):
    """Multi-head self-attention for spatial feature maps."""
    def __init__(self, ch, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.norm = nn.GroupNorm(min(32, ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.n_heads, C // self.n_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        # q,k,v: (B, heads, C//heads, H*W)
        scale = (C // self.n_heads) ** -0.5
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class UNetScore32(nn.Module):
    """U-Net with self-attention at 8x8 and 4x4 (bottleneck)."""
    def __init__(self, latent_ch=4, base_ch=256, time_embed_dim=256):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, base_ch * 4), nn.SiLU(),
            nn.Linear(base_ch * 4, base_ch * 4),
        )
        t_dim = base_ch * 4
        self.conv_in = nn.Conv2d(latent_ch, base_ch, 3, padding=1)

        # 32x32
        self.d1_r1 = TimeCondResBlock(base_ch, t_dim)
        self.d1_r2 = TimeCondResBlock(base_ch, t_dim)
        self.d1_down = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)

        # 16x16
        self.d2_r1 = TimeCondResBlock(base_ch * 2, t_dim)
        self.d2_r2 = TimeCondResBlock(base_ch * 2, t_dim)
        self.d2_down = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)

        # 8x8 + attention
        self.d3_r1 = TimeCondResBlock(base_ch * 4, t_dim)
        self.d3_attn = SelfAttention2d(base_ch * 4)
        self.d3_r2 = TimeCondResBlock(base_ch * 4, t_dim)
        self.d3_down = nn.Conv2d(base_ch * 4, base_ch * 4, 3, stride=2, padding=1)

        # 4x4 bottleneck + attention
        self.mid_r1 = TimeCondResBlock(base_ch * 4, t_dim)
        self.mid_attn = SelfAttention2d(base_ch * 4)
        self.mid_r2 = TimeCondResBlock(base_ch * 4, t_dim)

        # Up 8x8 + attention
        self.u3_up = nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1)
        self.u3_r1 = TimeCondResBlock(base_ch * 8, t_dim)
        self.u3_attn = SelfAttention2d(base_ch * 8)
        self.u3_r2 = TimeCondResBlock(base_ch * 8, t_dim)
        self.u3_out = nn.Conv2d(base_ch * 8, base_ch * 4, 1)

        # Up 16x16
        self.u2_up = nn.Conv2d(base_ch * 4, base_ch * 2, 3, padding=1)
        self.u2_r1 = TimeCondResBlock(base_ch * 4, t_dim)
        self.u2_r2 = TimeCondResBlock(base_ch * 4, t_dim)
        self.u2_out = nn.Conv2d(base_ch * 4, base_ch * 2, 1)

        # Up 32x32
        self.u1_up = nn.Conv2d(base_ch * 2, base_ch, 3, padding=1)
        self.u1_r1 = TimeCondResBlock(base_ch * 2, t_dim)
        self.u1_r2 = TimeCondResBlock(base_ch * 2, t_dim)
        self.u1_out = nn.Conv2d(base_ch * 2, base_ch, 1)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(min(32, base_ch), base_ch), nn.SiLU(),
            nn.Conv2d(base_ch, latent_ch, 3, padding=1),
        )
        print(f"    UNet+Attn: {sum(p.numel() for p in self.parameters()):,} params")

    def sinusoidal_embedding(self, t):
        half = self.time_embed_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        if t.dim() == 0: t = t.unsqueeze(0)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, z_t, t):
        te = self.time_mlp(self.sinusoidal_embedding(t))
        h = self.conv_in(z_t)

        # Down
        h1 = self.d1_r2(self.d1_r1(h, te), te)
        h = self.d1_down(h1)

        h2 = self.d2_r2(self.d2_r1(h, te), te)
        h = self.d2_down(h2)

        h3 = self.d3_r2(self.d3_attn(self.d3_r1(h, te)), te)
        h = self.d3_down(h3)

        # Mid
        h = self.mid_r2(self.mid_attn(self.mid_r1(h, te)), te)

        # Up
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = torch.cat([self.u3_up(h), h3], 1)
        h = self.u3_r2(self.u3_attn(self.u3_r1(h, te)), te)
        h = self.u3_out(h)

        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = torch.cat([self.u2_up(h), h2], 1)
        h = self.u2_r2(self.u2_r1(h, te), te)
        h = self.u2_out(h)

        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = torch.cat([self.u1_up(h), h1], 1)
        h = self.u1_r2(self.u1_r1(h, te), te)
        h = self.u1_out(h)

        return self.conv_out(h)


# ================================================================
# Helpers
# ================================================================

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(in_ch, base_ch, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        ch = base_ch
        for _ in range(1, n_layers):
            cn = min(ch * 2, 512)
            layers += [nn.Conv2d(ch, cn, 4, 2, 1, bias=False),
                       nn.InstanceNorm2d(cn), nn.LeakyReLU(0.2, True)]
            ch = cn
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

def save_ckpt(obj, name, extra=None):
    path = os.path.join(CKPT_DIR, f'{name}.pt')
    state = {'model': obj.state_dict()} if hasattr(obj, 'state_dict') else {}
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

def plot_grid(images, nrow=8, title='', save_name=None, img_size=64):
    n = min(len(images), nrow * nrow)
    nr = (n + nrow - 1) // nrow
    fig, axes = plt.subplots(nr, nrow, figsize=(nrow * 1.5, nr * 1.5))
    axes = np.atleast_2d(axes)
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            idx = i * nrow + j
            if idx < n:
                axes[i, j].imshow(images[idx].detach().cpu().clamp(0, 1)
                                  .view(3, img_size, img_size).permute(1, 2, 0).numpy())
            axes[i, j].axis('off')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_name:
        p = os.path.join(FIG_DIR, save_name)
        plt.savefig(p, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {p}")
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
        self.encoder = Encoder(image_shape, latent_ch, base_ch)
        self.decoder = Decoder(latent_ch, image_shape, base_ch)
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
        c, h, w = self.spatial_shape
        z = torch.randn(n, c, h, w, device=device)
        dt = self.T / n_steps
        # Use the unwrapped model for sampling (handles DataParallel)
        score = self.score_net.module if hasattr(self.score_net, 'module') else self.score_net
        for i in range(n_steps, 0, -1):
            t = torch.full((n,), i*dt, device=device)
            ab, sig = self.noise_schedule(t)
            ep = score(z, t)
            x0p = (z - sig.view(-1,1,1,1)*ep) / torch.sqrt(ab).view(-1,1,1,1).clamp(min=1e-8)
            abp, sigp = self.noise_schedule(t - dt)
            z = torch.sqrt(abp).view(-1,1,1,1)*x0p + sigp.view(-1,1,1,1)*torch.randn_like(z)
        # Use unwrapped decoder
        dec = self.decoder.module if hasattr(self.decoder, 'module') else self.decoder
        x = dec(z * self.z_std + self.z_mean)
        return x.clamp(*clamp) if clamp else x


class LDMModel(nn.Module):
    def __init__(self, d, image_shape, latent_ch, base_ch, unet_ch, kl_weight=1e-4):
        super().__init__()
        self.d, self.kl_weight = d, kl_weight
        self.image_shape = image_shape
        self.latent_ch = latent_ch
        self.beta_min, self.beta_max = 0.1, 20.0
        self.vae_encoder = VAEEncoder(image_shape, latent_ch, base_ch)
        self.decoder = Decoder(latent_ch, image_shape, base_ch)
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
        vae = self.vae_encoder.module if hasattr(self.vae_encoder, 'module') else self.vae_encoder
        dummy = torch.zeros(1, *self.image_shape, device=device)
        mu, _ = vae(dummy.view(1, -1))
        _, c, h, w = mu.shape
        z = torch.randn(n, c, h, w, device=device)
        dt = 0.999 / n_steps
        score = self.score_net.module if hasattr(self.score_net, 'module') else self.score_net
        for i in range(n_steps, 0, -1):
            t = torch.full((n,), i*dt, device=device)
            ab, sig = self.noise_schedule(t)
            ep = score(z, t)
            x0p = (z - sig.view(-1,1,1,1)*ep) / torch.sqrt(ab).view(-1,1,1,1).clamp(min=1e-8)
            abp, sigp = self.noise_schedule(t - dt)
            z = torch.sqrt(abp).view(-1,1,1,1)*x0p + sigp.view(-1,1,1,1)*torch.randn_like(z)
        dec = self.decoder.module if hasattr(self.decoder, 'module') else self.decoder
        x = dec(z * self.z_std + self.z_mean)
        return x.clamp(*clamp) if clamp else x


# ================================================================
# Training with multi-GPU
# ================================================================

def train_ae(model, dataset, n_steps, batch_size, lpips_weight, adv_weight,
             ema_decay, is_twostage, device, ckpt_name=None):
    img_size = model.image_shape[1]
    sigma_t0 = model.t0 if is_twostage else 0.0

    loss_fn = None
    if LPIPS_AVAILABLE:
        loss_fn = lpips.LPIPS(net='vgg').to(device)
        loss_fn.eval()
        for p in loss_fn.parameters(): p.requires_grad = False

    disc = PatchDiscriminator(3, 64, 3).to(device)

    if is_twostage:
        gp = list(model.encoder.parameters()) + list(model.decoder.parameters())
    else:
        gp = list(model.vae_encoder.parameters()) + list(model.decoder.parameters())
    opt_g = torch.optim.Adam(gp, lr=1e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))
    ema = EMA(model, ema_decay)
    ds = min(2000, n_steps // 10)

    tag = "Stage1" if is_twostage else "VAE"
    print(f"  {tag}: {n_steps} steps, batch={batch_size}, "
          f"adv={adv_weight}, disc@{ds}")

    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)

        if is_twostage:
            x_in = x0 + sigma_t0 * torch.randn_like(x0)
            z = model.encoder(x_in)
            xh = model.decoder(z)
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
                    z_d = model.encoder(x_in)
                    xhd = model.decoder(z_d)
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

        # Periodic checkpoint every 50k steps
        if (step + 1) % 50000 == 0 and ckpt_name:
            ema.apply(model)
            save_ckpt(model, f'{ckpt_name}_step{step+1}')
            print(f"  Periodic checkpoint at step {step+1}")

    ema.apply(model)
    del ema  # free EMA shadow copy

    # Compute z stats
    with torch.no_grad():
        zs = []
        for _ in range(40):
            x0, _ = dataset.sample_clean(min(batch_size, 500))
            x0 = x0.to(device)
            if is_twostage:
                zs.append(model.encoder(x0))
            else:
                zs.append(model.vae_encoder.encode_deterministic(x0))
        zs = torch.cat(zs, dim=0)
        model.z_mean = zs.mean(dim=(0, 2, 3), keepdim=True).squeeze(0)
        model.z_std = zs.std(dim=(0, 2, 3), keepdim=True).squeeze(0).clamp(min=1e-6)

    # Cleanup training objects
    del disc, opt_g, opt_d
    if loss_fn is not None:
        del loss_fn
    torch.cuda.empty_cache()
    print("  EMA applied, z stats computed, memory cleaned.")


def train_diffusion(model, dataset, n_steps, batch_size, ema_decay,
                    is_twostage, device, ckpt_name=None, start_step=0):
    ema = EMA(model, ema_decay)
    opt = torch.optim.Adam(model.score_net.parameters(), lr=3e-4)

    # Pre-encode all data to avoid lazy loading bottleneck in diffusion
    print("  Pre-encoding latents...")
    all_z = []
    with torch.no_grad():
        for i in range(0, min(dataset.n_total, 30000), 500):
            x0, _ = dataset.sample_clean(500)
            x0 = x0.to(device)
            if is_twostage:
                z = (model.encoder(x0) - model.z_mean) / model.z_std.clamp(min=1e-6)
            else:
                z = (model.vae_encoder.encode_deterministic(x0) -
                     model.z_mean) / model.z_std.clamp(min=1e-6)
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0)
    n_z = len(all_z)
    print(f"  Pre-encoded {n_z} latents, shape={list(all_z.shape[1:])}")

    for step in range(start_step, n_steps):
        idx = torch.randint(0, n_z, (batch_size,))
        z0 = all_z[idx].to(device)

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

        if (step + 1) % 1000 == 0:
            print(f"  Diff [{step+1}/{n_steps}] eps={loss.item():.6f}")

        # Periodic checkpoint
        if (step + 1) % 100000 == 0 and ckpt_name:
            ema.apply(model)
            save_ckpt(model, f'{ckpt_name}_diff{step+1}')
            print(f"  Checkpoint at diff step {step+1}")

    ema.apply(model)
    del ema, opt, all_z
    torch.cuda.empty_cache()
    print("  EMA applied, memory cleaned.")


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

    latent_ch = 4          # 32x32x4 = 4096 dims (3:1)
    base_ch = 128          # encoder/decoder
    unet_ch = 256          # U-Net score (larger)
    t0 = 0.005

    ae_steps = 200000
    diff_steps = 500000
    batch_size = 64
    lpips_weight = 1.0
    adv_weight = 0.1
    ema_decay = 0.9999

    # Use full CelebA
    max_samples = 200000

    print("=" * 70)
    print(f"CelebA LARGE-SCALE (d={d}, {img_size}x{img_size}x3)")
    print(f"  latent: 32x32x{latent_ch} = {32*32*latent_ch} dims "
          f"({d/(32*32*latent_ch):.1f}:1)")
    print(f"  enc/dec base_ch={base_ch}, U-Net base_ch={unet_ch}")
    print(f"  AE={ae_steps}, Diff={diff_steps}")
    print(f"  batch={batch_size}")
    print(f"  LPIPS={lpips_weight}, adv={adv_weight}, EMA={ema_decay}")
    print(f"  data: {max_samples} samples (lazy loading)")
    print("=" * 70)

    dataset = CelebALazyDataset(img_size=img_size, max_samples=max_samples)

    # ================================================================
    # Two-Stage
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODEL 1: ULD (Two-Stage)")
    print("=" * 70)

    m_ts = TwoStageModel(d, image_shape, latent_ch, base_ch, unet_ch, t0).to(device)
    total_p = sum(p.numel() for p in m_ts.parameters())
    print(f"  Total params: {total_p:,}")

    ck = load_ckpt(m_ts, 'celeba_large_ts_full', device)
    if ck:
        m_ts.freeze_stage1()
        r_ts, e_ts = ck.get('recon', 0), ck.get('eps', 0)
    else:
        ck1 = load_ckpt(m_ts, 'celeba_large_ts_s1', device)
        if ck1:
            m_ts.freeze_stage1()
            r_ts = ck1.get('recon', 0)
        else:
            print("\n--- Stage 1: GAN+LPIPS+EMA ---")
            train_ae(m_ts, dataset, ae_steps, batch_size, lpips_weight,
                     adv_weight, ema_decay, True, device,
                     ckpt_name='celeba_large_ts_s1')
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                z = m_ts.encoder(x0)
                xh = m_ts.decoder(z)
                r_ts = F.mse_loss(xh, x0).item()
                print(f"\n  Recon MSE: {r_ts:.6f}")
            m_ts.freeze_stage1()
            save_ckpt(m_ts, 'celeba_large_ts_s1', {'recon': r_ts})

        print("\n--- Stage 2: U-Net Diffusion ---")
        train_diffusion(m_ts, dataset, diff_steps, batch_size, ema_decay,
                        True, device, ckpt_name='celeba_large_ts')

        times = [0.01, 0.1, 0.3, 0.5, 0.8]
        tot = 0
        for tv in times:
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                z0 = (m_ts.encoder(x0) - m_ts.z_mean) / m_ts.z_std.clamp(min=1e-6)
                tb = torch.full((2000,), tv, device=device)
                zt, ep = m_ts.forward_diffusion(z0, tb)
                mse = F.mse_loss(m_ts.score_net(zt, tb), ep).item()
                tot += mse
                print(f"  t={tv:.2f}: eps={mse:.6f}")
        e_ts = tot / len(times)
        save_ckpt(m_ts, 'celeba_large_ts_full', {'recon': r_ts, 'eps': e_ts})

    # Generate Two-Stage samples and save before creating LDM
    print("\n--- Two-Stage: Generate samples ---")
    with torch.no_grad():
        x_gen_ts = m_ts.sample(64, n_steps=250, device=device, clamp=(0,1)).cpu()

    # Free Two-Stage model from GPU to make room for LDM
    m_ts_cpu = copy.deepcopy(m_ts).cpu()
    del m_ts
    torch.cuda.empty_cache()
    print("  Two-Stage moved to CPU, GPU freed.")

    # ================================================================
    # LDM
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODEL 2: VAE-LDM")
    print("=" * 70)

    m_ldm = LDMModel(d, image_shape, latent_ch, base_ch, unet_ch).to(device)

    ck = load_ckpt(m_ldm, 'celeba_large_ldm_full', device)
    if ck:
        m_ldm.freeze_ae()
        r_ldm, e_ldm = ck.get('recon', 0), ck.get('eps', 0)
    else:
        ck1 = load_ckpt(m_ldm, 'celeba_large_ldm_ae', device)
        if ck1:
            m_ldm.freeze_ae()
            r_ldm = ck1.get('recon', 0)
        else:
            print("\n--- VAE-GAN+LPIPS+EMA ---")
            train_ae(m_ldm, dataset, ae_steps, batch_size, lpips_weight,
                     adv_weight, ema_decay, False, device,
                     ckpt_name='celeba_large_ldm_ae')
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                z = m_ldm.vae_encoder.encode_deterministic(x0)
                xr = m_ldm.decoder(z)
                r_ldm = F.mse_loss(xr, x0).item()
                print(f"\n  Recon MSE: {r_ldm:.6f}")
            m_ldm.freeze_ae()
            save_ckpt(m_ldm, 'celeba_large_ldm_ae', {'recon': r_ldm})

        print("\n--- U-Net Diffusion ---")
        train_diffusion(m_ldm, dataset, diff_steps, batch_size, ema_decay,
                        False, device, ckpt_name='celeba_large_ldm')

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
                zt = torch.sqrt(ab).view(-1,1,1,1)*z0 + sig.view(-1,1,1,1)*ep
                mse = F.mse_loss(m_ldm.score_net(zt, tb), ep).item()
                tot += mse
                print(f"  t={tv:.2f}: eps={mse:.6f}")
        e_ldm = tot / len(times)
        save_ckpt(m_ldm, 'celeba_large_ldm_full', {'recon': r_ldm, 'eps': e_ldm})

    # Generate LDM samples
    print("\n--- Generating ---")
    with torch.no_grad():
        plot_grid(x_gen_ts, 8, 'ULD (Ours)', 'celeba_large_gen_ts.png')
        xg_ldm = m_ldm.sample(64, 250, device, (0, 1)).cpu()
        plot_grid(xg_ldm, 8, 'VAE-LDM', 'celeba_large_gen_ldm.png')
        xr, _ = dataset.sample_clean(64)
        plot_grid(xr, 8, 'Real', 'celeba_large_real.png')

    fid_ts = fid_ldm = float('nan')
    if FID_AVAILABLE:
        print("\n--- FID (2k samples) ---")
        nf, fb = 2048, 32
        reals = []
        for _ in range(nf // fb):
            x0, _ = dataset.sample_clean(fb)
            reals.append((x0.view(-1, 3, img_size, img_size).clamp(0,1)*255).to(torch.uint8))
        reals = torch.cat(reals)[:nf]

        # Two-Stage FID: use pre-generated samples + generate more
        print("  ULD: generating for FID...")
        gens_ts = []
        m_ts_gpu = m_ts_cpu.to(device)
        with torch.no_grad():
            for i in range(nf // fb):
                x = m_ts_gpu.sample(fb, 250, device, (0, 1)).cpu()
                gens_ts.append((x.view(-1, 3, img_size, img_size).clamp(0,1)*255).to(torch.uint8))
                if (i+1) % 16 == 0: print(f"    {(i+1)*fb}/{nf}")
        gens_ts = torch.cat(gens_ts)[:nf]
        del m_ts_gpu
        torch.cuda.empty_cache()

        fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        fid.update(reals.to(device), real=True)
        fid.update(gens_ts.to(device), real=False)
        fid_ts = fid.compute().item()
        print(f"  ULD FID: {fid_ts:.2f}")
        del fid, gens_ts
        torch.cuda.empty_cache()

        # LDM FID
        print("  LDM: generating for FID...")
        gens_ldm = []
        with torch.no_grad():
            for i in range(nf // fb):
                x = m_ldm.sample(fb, 250, device, (0, 1)).cpu()
                gens_ldm.append((x.view(-1, 3, img_size, img_size).clamp(0,1)*255).to(torch.uint8))
                if (i+1) % 16 == 0: print(f"    {(i+1)*fb}/{nf}")
        gens_ldm = torch.cat(gens_ldm)[:nf]

        fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        fid.update(reals.to(device), real=True)
        fid.update(gens_ldm.to(device), real=False)
        fid_ldm = fid.compute().item()
        print(f"  LDM FID: {fid_ldm:.2f}")
        del fid, gens_ldm, reals
        torch.cuda.empty_cache()

    # Reconstruction
    print("\n--- Reconstruction ---")
    m_ts_gpu = m_ts_cpu.to(device)
    with torch.no_grad():
        xt, _ = dataset.sample_clean(10)
        xd = xt.to(device)
        z_ts = m_ts_gpu.encoder(xd)
        xrt = m_ts_gpu.decoder(z_ts).cpu()
        z_ldm = m_ldm.vae_encoder.encode_deterministic(xd)
        xrl = m_ldm.decoder(z_ldm).cpu()
    del m_ts_gpu
    torch.cuda.empty_cache()

    fig, axes = plt.subplots(3, 10, figsize=(18, 5.4))
    for j in range(10):
        for row, imgs in enumerate([xt, xrt, xrl]):
            axes[row, j].imshow(imgs[j].view(3, 64, 64).permute(1, 2, 0).clamp(0, 1).numpy())
            axes[row, j].axis('off')
    for r, lb in enumerate(['Original', 'ULD (Ours)', 'VAE-LDM']):
        axes[r, 0].set_ylabel(lb, fontsize=12, fontweight='bold',
                              rotation=0, labelpad=70, va='center')
    plt.tight_layout()
    p = os.path.join(FIG_DIR, 'celeba_large_reconstruction.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {p}")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"CelebA LARGE-SCALE SUMMARY")
    print(f"  32x32x{latent_ch} = {32*32*latent_ch} dims, "
          f"{d/(32*32*latent_ch):.1f}:1 compression")
    print(f"  {max_samples} samples, {ae_steps} AE + {diff_steps} diff steps")
    print(f"  batch={batch_size}")
    print("=" * 70)
    print(f"\n  {'Model':<20} {'Recon MSE':<12} {'Eps Loss':<12} {'FID':<10}")
    print("  " + "-" * 55)
    print(f"  {'ULD (ours)':<20} {r_ts:<12.6f} {e_ts:<12.6f} {fid_ts:<10.2f}")
    print(f"  {'VAE-LDM':<20} {r_ldm:<12.6f} {e_ldm:<12.6f} {fid_ldm:<10.2f}")


if __name__ == '__main__':
    run_experiment()
