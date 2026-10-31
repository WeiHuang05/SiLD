"""
ImageNet-256 Stage 1 Only — Focus on AE quality (rFID).

Goal: Get rFID < 2 before proceeding to diffusion.

Architecture:
  256x256x3 → Encoder (2 downsamples) → 64x64x4 = 16384 dims (12:1)
  base_ch=128, GAN + LPIPS + EMA
  200k samples, 200k steps
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.image_data import ImageNet256LazyDataset

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


# ================================================================
# Encoder / Decoder: 256→128→64 (2 downsamples)
# ================================================================

class Encoder(nn.Module):
    """256x256x3 → 64x64 x latent_ch."""
    def __init__(self, image_shape, latent_ch=4, base_ch=128):
        super().__init__()
        C = image_shape[0]
        self.image_shape = image_shape
        ch = base_ch

        self.net = nn.Sequential(
            nn.Conv2d(C, ch, 3, padding=1),
            # Level 1: 256→128
            ResBlock2d(ch), ResBlock2d(ch), ResBlock2d(ch),
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
            # Level 2: 128→64
            ResBlock2d(ch * 2), ResBlock2d(ch * 2), ResBlock2d(ch * 2),
            nn.Conv2d(ch * 2, ch * 2, 3, stride=2, padding=1),
            # Final
            ResBlock2d(ch * 2), ResBlock2d(ch * 2),
            nn.GroupNorm(min(32, ch * 2), ch * 2), nn.SiLU(),
            nn.Conv2d(ch * 2, latent_ch, 1),
        )
        with torch.no_grad():
            out = self.net(torch.zeros(1, *image_shape))
            self.spatial_shape = out.shape[1:]
        n_p = sum(p.numel() for p in self.parameters())
        print(f"    Encoder: {n_p:,} params, output {list(self.spatial_shape)}")

    def forward(self, x):
        return self.net(x.view(-1, *self.image_shape))


class Decoder(nn.Module):
    """64x64 x latent_ch → 256x256x3."""
    def __init__(self, latent_ch, image_shape, base_ch=128):
        super().__init__()
        C = image_shape[0]
        self.image_shape = image_shape
        self.d = C * image_shape[1] * image_shape[2]
        ch = base_ch * 2

        self.net = nn.Sequential(
            nn.Conv2d(latent_ch, ch, 1), nn.SiLU(),
            ResBlock2d(ch), ResBlock2d(ch), ResBlock2d(ch),
            # Level 2: 64→128
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch, ch, 3, padding=1),
            ResBlock2d(ch), ResBlock2d(ch), ResBlock2d(ch),
            # Level 1: 128→256
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch, ch // 2, 3, padding=1),
            ResBlock2d(ch // 2), ResBlock2d(ch // 2), ResBlock2d(ch // 2),
            # Output
            nn.GroupNorm(min(32, ch // 2), ch // 2), nn.SiLU(),
            nn.Conv2d(ch // 2, C, 3, padding=1),
            nn.Sigmoid(),
        )
        n_p = sum(p.numel() for p in self.parameters())
        print(f"    Decoder: {n_p:,} params")

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
            nn.Conv2d(ch * 2, ch * 2, 3, stride=2, padding=1),
            ResBlock2d(ch * 2), ResBlock2d(ch * 2),
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
# Helpers
# ================================================================

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, n_layers=4):
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


# ================================================================
# Training
# ================================================================

def train_ae(encoder, decoder, dataset, n_steps, batch_size, lpips_weight,
             adv_weight, ema_encoder, ema_decoder, is_twostage, t0,
             vae_encoder=None, kl_weight=1e-4, device='cpu'):
    img_size = 256

    loss_fn = None
    if LPIPS_AVAILABLE:
        loss_fn = lpips.LPIPS(net='vgg').to(device)
        loss_fn.eval()
        for p in loss_fn.parameters(): p.requires_grad = False

    disc = PatchDiscriminator(3, 64, 4).to(device)

    if is_twostage:
        gp = list(encoder.parameters()) + list(decoder.parameters())
    else:
        gp = list(vae_encoder.parameters()) + list(decoder.parameters())
    opt_g = torch.optim.Adam(gp, lr=1e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.5, 0.999))
    ds = min(2000, n_steps // 10)

    tag = "Stage1 (denoise)" if is_twostage else "VAE"
    print(f"  {tag}: {n_steps} steps, batch={batch_size}, adv={adv_weight}")

    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)

        if is_twostage:
            x_in = x0 + t0 * torch.randn_like(x0)
            z = encoder(x_in)
            xh = decoder(z)
            kl = torch.tensor(0.0, device=device)
        else:
            mu, lv = vae_encoder(x0)
            z = mu + torch.exp(0.5*lv) * torch.randn_like(mu)
            xh = decoder(z)
            kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())

        x0i = x0.view(-1, 3, img_size, img_size)
        xhi = xh.view(-1, 3, img_size, img_size)
        mse = F.mse_loss(xh, x0)
        lp = loss_fn(xhi*2-1, x0i*2-1).mean() if loss_fn else torch.tensor(0.0, device=device)
        ga = hinge_g(disc(xhi)) if step >= ds else torch.tensor(0.0, device=device)

        gl = mse + lpips_weight * lp + adv_weight * ga
        if not is_twostage:
            gl = gl + kl_weight * kl

        opt_g.zero_grad()
        gl.backward()
        torch.nn.utils.clip_grad_norm_(gp, 1.0)
        opt_g.step()

        if step >= ds:
            with torch.no_grad():
                if is_twostage:
                    xhd = decoder(encoder(x_in))
                else:
                    mud, _ = vae_encoder(x0)
                    xhd = decoder(mud)
                xhdi = xhd.view(-1, 3, img_size, img_size)
            dl = hinge_d(disc(x0i), disc(xhdi))
            opt_d.zero_grad()
            dl.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            opt_d.step()
        else:
            dl = torch.tensor(0.0)

        ema_encoder.update(encoder if is_twostage else vae_encoder)
        ema_decoder.update(decoder)

        if (step + 1) % 500 == 0:
            ex = f" kl={kl.item():.4f}" if not is_twostage else ""
            print(f"  [{step+1}/{n_steps}] g={gl.item():.5f} "
                  f"mse={mse.item():.5f} lp={lp.item():.5f} "
                  f"adv={ga.item():.5f} d={dl.item():.5f}{ex}")

        # Periodic rFID checkpoint
        if (step + 1) % 50000 == 0 and FID_AVAILABLE:
            ema_encoder.apply(encoder if is_twostage else vae_encoder)
            ema_decoder.apply(decoder)
            rfid = compute_rfid(encoder if is_twostage else vae_encoder,
                                decoder, dataset, is_twostage, device)
            print(f"  [Step {step+1}] rFID = {rfid:.2f}")

    ema_encoder.apply(encoder if is_twostage else vae_encoder)
    ema_decoder.apply(decoder)


def compute_rfid(encoder, decoder, dataset, is_twostage, device,
                 n_samples=2048, batch=8):
    """Compute reconstruction FID."""
    if not FID_AVAILABLE:
        return float('nan')

    reals, recs = [], []
    with torch.no_grad():
        for _ in range(n_samples // batch):
            x0, _ = dataset.sample_clean(batch)
            x0 = x0.to(device)
            reals.append((x0.view(-1, 3, 256, 256).clamp(0,1)*255).to(torch.uint8).cpu())
            if is_twostage:
                z = encoder(x0)
            else:
                z = encoder.encode_deterministic(x0)
            xr = decoder(z)
            recs.append((xr.view(-1, 3, 256, 256).clamp(0,1)*255).to(torch.uint8).cpu())

    reals = torch.cat(reals)[:n_samples]
    recs = torch.cat(recs)[:n_samples]

    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid.update(reals.to(device), real=True)
    fid.update(recs.to(device), real=False)
    val = fid.compute().item()
    del fid
    torch.cuda.empty_cache()
    return val


def plot_recon(encoder, decoder, dataset, is_twostage, name, device, n=8):
    img_size = 256
    with torch.no_grad():
        x0, _ = dataset.sample_clean(n)
        x0 = x0.to(device)
        if is_twostage:
            xr = decoder(encoder(x0))
        else:
            xr = decoder(encoder.encode_deterministic(x0))
        x0, xr = x0.cpu(), xr.cpu()

    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))
    for j in range(n):
        axes[0, j].imshow(x0[j].view(3, img_size, img_size).permute(1,2,0).clamp(0,1).numpy())
        axes[0, j].axis('off')
        axes[1, j].imshow(xr[j].view(3, img_size, img_size).permute(1,2,0).clamp(0,1).numpy())
        axes[1, j].axis('off')
    axes[0, 0].set_ylabel('Original', fontsize=12, rotation=0, labelpad=60, va='center')
    axes[1, 0].set_ylabel(name, fontsize=12, rotation=0, labelpad=60, va='center')
    plt.tight_layout()
    fname = f'imagenet_s1_{name.lower().replace(" ", "_")}.png'
    path = os.path.join(FIG_DIR, fname)
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


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

    latent_ch = 4          # 64x64x4 = 16384 dims
    base_ch = 128          # larger encoder/decoder
    t0 = 0.005

    ae_steps = 200000
    batch_size = 8
    lpips_weight = 1.0
    adv_weight = 0.1
    ema_decay = 0.9999
    max_samples = 200000

    latent_dims = 64 * 64 * latent_ch
    compression = d / latent_dims

    print("=" * 70)
    print(f"ImageNet-256 Stage 1 Only")
    print(f"  Latent: 64x64x{latent_ch} = {latent_dims} dims ({compression:.0f}:1)")
    print(f"  Encoder/Decoder base_ch={base_ch}")
    print(f"  AE={ae_steps} steps, batch={batch_size}")
    print(f"  LPIPS={lpips_weight}, adv={adv_weight}, EMA={ema_decay}")
    print(f"  Data: {max_samples} samples")
    print("=" * 70)

    dataset = ImageNet256LazyDataset(max_samples=max_samples)

    # ================================================================
    # Two-Stage AE
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODEL 1: ULD (Two-Stage AE)")
    print("=" * 70)

    enc_ts = Encoder(image_shape, latent_ch, base_ch).to(device)
    dec_ts = Decoder(latent_ch, image_shape, base_ch).to(device)

    ck = load_ckpt(enc_ts, 'imagenet_s1_ts_enc', device)
    if ck:
        load_ckpt(dec_ts, 'imagenet_s1_ts_dec', device)
        r_ts = ck.get('recon', 0)
        rfid_ts = ck.get('rfid', 0)
        print(f"  Resumed: recon={r_ts:.6f}, rFID={rfid_ts:.2f}")
    else:
        ema_enc = EMA(enc_ts, ema_decay)
        ema_dec = EMA(dec_ts, ema_decay)

        train_ae(enc_ts, dec_ts, dataset, ae_steps, batch_size,
                 lpips_weight, adv_weight, ema_enc, ema_dec,
                 is_twostage=True, t0=t0, device=device)

        with torch.no_grad():
            x0, _ = dataset.sample_clean(500)
            x0 = x0.to(device)
            xr = dec_ts(enc_ts(x0))
            r_ts = F.mse_loss(xr, x0).item()
            print(f"\n  Recon MSE: {r_ts:.6f}")

        rfid_ts = compute_rfid(enc_ts, dec_ts, dataset, True, device)
        print(f"  rFID: {rfid_ts:.2f}")

        save_ckpt(enc_ts, 'imagenet_s1_ts_enc', {'recon': r_ts, 'rfid': rfid_ts})
        save_ckpt(dec_ts, 'imagenet_s1_ts_dec')

    plot_recon(enc_ts, dec_ts, dataset, True, 'ULD', device)

    # ================================================================
    # LDM VAE
    # ================================================================
    print(f"\n{'=' * 70}")
    print("MODEL 2: VAE-LDM")
    print("=" * 70)

    vae_enc = VAEEncoder(image_shape, latent_ch, base_ch).to(device)
    dec_ldm = Decoder(latent_ch, image_shape, base_ch).to(device)

    ck = load_ckpt(vae_enc, 'imagenet_s1_ldm_enc', device)
    if ck:
        load_ckpt(dec_ldm, 'imagenet_s1_ldm_dec', device)
        r_ldm = ck.get('recon', 0)
        rfid_ldm = ck.get('rfid', 0)
        print(f"  Resumed: recon={r_ldm:.6f}, rFID={rfid_ldm:.2f}")
    else:
        ema_enc = EMA(vae_enc, ema_decay)
        ema_dec = EMA(dec_ldm, ema_decay)

        train_ae(None, dec_ldm, dataset, ae_steps, batch_size,
                 lpips_weight, adv_weight, ema_enc, ema_dec,
                 is_twostage=False, t0=0, vae_encoder=vae_enc,
                 device=device)

        with torch.no_grad():
            x0, _ = dataset.sample_clean(500)
            x0 = x0.to(device)
            z = vae_enc.encode_deterministic(x0)
            xr = dec_ldm(z)
            r_ldm = F.mse_loss(xr, x0).item()
            print(f"\n  Recon MSE: {r_ldm:.6f}")

        rfid_ldm = compute_rfid(vae_enc, dec_ldm, dataset, False, device)
        print(f"  rFID: {rfid_ldm:.2f}")

        save_ckpt(vae_enc, 'imagenet_s1_ldm_enc', {'recon': r_ldm, 'rfid': rfid_ldm})
        save_ckpt(dec_ldm, 'imagenet_s1_ldm_dec')

    plot_recon(vae_enc, dec_ldm, dataset, False, 'LDM', device)

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"ImageNet-256 Stage 1 SUMMARY")
    print(f"  Latent: 64x64x{latent_ch} = {latent_dims} ({compression:.0f}:1)")
    print(f"  {max_samples} samples, {ae_steps} steps")
    print("=" * 70)
    print(f"\n  {'Model':<20} {'Recon MSE':<12} {'rFID':<10}")
    print("  " + "-" * 42)
    print(f"  {'ULD (ours)':<20} {r_ts:<12.6f} {rfid_ts:<10.2f}")
    print(f"  {'VAE-LDM':<20} {r_ldm:<12.6f} {rfid_ldm:<10.2f}")


if __name__ == '__main__':
    run_experiment()
