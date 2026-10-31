"""
QM9 3D with E(3)-equivariant EGNN encoder/decoder.

Direct comparison with EDM/GeoLDM table using the SAME metrics AND
E(3) equivariance. ULD vs LDM at identical EGNN architecture — only
the Stage-1 AE loss differs.

Architecture:
  - Encoder: EGNN (4 layers, h=128) → global mean pool → z ∈ R^64
  - Decoder: z → broadcast → EGNN (4 layers) → (types, positions)
  - Score net: standard MLP LatentScoreNet on z (invariant latent)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json, torch, torch.nn as nn, torch.nn.functional as F, numpy as np, math
from data.molecule_3d_data import QM93DDataset, MAX_ATOMS, N_ATOM_TYPES, DIMS_PER_ATOM
from models.egnn import NodeLevelEncoder, NodeLevelDecoder, NodeLevelVAEEncoder
from models.two_stage_latent import LatentScoreNet

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def flat_to_structured(x_flat):
    """(B, 232) → types (B, 29, 5), positions (B, 29, 3), mask (B, 29)."""
    B = x_flat.shape[0]
    x = x_flat.view(B, MAX_ATOMS, DIMS_PER_ATOM)
    types = x[:, :, :N_ATOM_TYPES]
    positions = x[:, :, N_ATOM_TYPES:]
    mask = (types.sum(-1) > 0.5).float()
    return types, positions, mask


def structured_to_flat(types, positions):
    """types (B, N, 5), positions (B, N, 3) → (B, 232)."""
    return torch.cat([types, positions], dim=-1).view(types.shape[0], -1)


# ===================================================================
# Noise schedule (VP-SDE, same as two_stage_latent)
# ===================================================================
def noise_schedule(t, beta_min=0.1, beta_max=20.0):
    t_t = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
    log_mean = -0.25 * t_t**2 * (beta_max - beta_min) - 0.5 * t_t * beta_min
    alpha_bar = torch.exp(2.0 * log_mean)
    sigma = torch.sqrt(torch.clamp(1.0 - alpha_bar, min=1e-8))
    return alpha_bar, sigma


# ===================================================================
# Training
# ===================================================================

def train_uld_stage1(encoder, decoder, dataset, n_steps, lr, batch_size,
                     t0, device):
    """Denoising AE at small t0 — add noise to positions, reconstruct."""
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)
    sigma_t0 = t0

    for step in range(n_steps):
        x_flat, _ = dataset.sample_clean(batch_size)
        x_flat = x_flat.to(device)
        types, positions, mask = flat_to_structured(x_flat)

        # Add small noise to positions (and optionally types)
        pos_noisy = positions + sigma_t0 * torch.randn_like(positions)
        types_noisy = types + sigma_t0 * 0.1 * torch.randn_like(types)

        z = encoder(types_noisy, pos_noisy, mask)
        types_hat, pos_hat = decoder(z, mask)

        loss_types = F.mse_loss(types_hat * mask.unsqueeze(-1),
                                types * mask.unsqueeze(-1))
        loss_pos = F.mse_loss(pos_hat * mask.unsqueeze(-1),
                              positions * mask.unsqueeze(-1))
        loss = loss_types + loss_pos

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()

        if (step + 1) % 500 == 0:
            print(f'  S1 [{step+1}/{n_steps}] loss={loss.item():.6f} '
                  f'types={loss_types.item():.6f} pos={loss_pos.item():.6f}')

    return loss.item()


def train_vae_stage1(vae_encoder, decoder, dataset, n_steps, lr, batch_size,
                     kl_weight, device):
    """VAE encoder training — same EGNN architecture, KL regularization."""
    params = list(vae_encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)

    for step in range(n_steps):
        x_flat, _ = dataset.sample_clean(batch_size)
        x_flat = x_flat.to(device)
        types, positions, mask = flat_to_structured(x_flat)

        mu, logvar = vae_encoder(types, positions, mask)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        types_hat, pos_hat = decoder(z, mask)

        loss_types = F.mse_loss(types_hat * mask.unsqueeze(-1),
                                types * mask.unsqueeze(-1))
        loss_pos = F.mse_loss(pos_hat * mask.unsqueeze(-1),
                              positions * mask.unsqueeze(-1))
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = loss_types + loss_pos + kl_weight * kl

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()

        if (step + 1) % 500 == 0:
            print(f'  VAE [{step+1}/{n_steps}] loss={loss.item():.6f} '
                  f'types={loss_types.item():.6f} pos={loss_pos.item():.6f} '
                  f'kl={kl.item():.4f}')

    return loss.item()


def compute_latent_stats(encoder, dataset, batch_size, device, is_vae=False):
    """Compute z mean/std for latent normalization."""
    zs = []
    with torch.no_grad():
        for _ in range(10):
            x, _ = dataset.sample_clean(min(batch_size, 2000))
            x = x.to(device)
            types, positions, mask = flat_to_structured(x)
            if is_vae:
                z = encoder.encode_deterministic(types, positions, mask)
            else:
                z = encoder(types, positions, mask)
            zs.append(z)
    zs = torch.cat(zs, dim=0)
    return zs.mean(0), zs.std(0).clamp(min=1e-6)


def train_diffusion(score_net, encoder, dataset, n_steps, lr, batch_size,
                    z_mean, z_std, t0, device, is_vae=False, min_snr_gamma=5.0):
    """Latent diffusion on invariant z."""
    opt = torch.optim.Adam(score_net.parameters(), lr=lr)
    T = 0.999
    t_min = t0 if not is_vae else 1e-3

    for step in range(n_steps):
        x, _ = dataset.sample_clean(batch_size)
        x = x.to(device)
        types, positions, mask = flat_to_structured(x)

        with torch.no_grad():
            if is_vae:
                z0 = encoder.encode_deterministic(types, positions, mask)
            else:
                z0 = encoder(types, positions, mask)
            z0 = (z0 - z_mean) / z_std

        t = torch.rand(z0.shape[0], device=device) * (T - t_min) + t_min
        alpha_bar, sigma = noise_schedule(t)
        eps = torch.randn_like(z0)
        z_t = torch.sqrt(alpha_bar).unsqueeze(-1) * z0 + sigma.unsqueeze(-1) * eps
        eps_pred = score_net(z_t, t)

        # Min-SNR-γ
        snr = (alpha_bar / sigma.pow(2).clamp(min=1e-8)).clamp(min=1e-8)
        w = torch.minimum(snr, torch.full_like(snr, min_snr_gamma)) / snr
        loss = (w.unsqueeze(-1) * (eps_pred - eps).pow(2)).mean()

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(score_net.parameters(), 1.0)
        opt.step()

        if (step + 1) % 1000 == 0:
            print(f'  Diff [{step+1}/{n_steps}] loss={loss.item():.6f}')


@torch.no_grad()
def sample_3d(score_net, decoder, z_mean, z_std, n, n_steps, t0, device):
    """DDIM in invariant latent → decode to 3D."""
    latent_dim = z_mean.shape[0]
    T = 0.999
    ts = torch.linspace(T, t0, n_steps + 1, device=device)
    z = torch.randn(n, latent_dim, device=device)

    for i in range(n_steps):
        t_cur, t_next = ts[i], ts[i+1]
        t_b = torch.full((n,), t_cur.item(), device=device)
        alpha_cur, sigma_cur = noise_schedule(t_cur)
        alpha_next, sigma_next = noise_schedule(t_next)
        eps_pred = score_net(z, t_b)
        z0_pred = (z - sigma_cur * eps_pred) / torch.sqrt(alpha_cur).clamp(min=1e-6)
        if i == n_steps - 1:
            z = z0_pred
        else:
            z = torch.sqrt(alpha_next) * z0_pred + sigma_next * eps_pred

    # Denormalize and decode
    z_raw = z * z_std + z_mean
    mask_full = torch.ones(n, MAX_ATOMS, device=device)
    types_logits, positions = decoder(z_raw, mask_full)
    return structured_to_flat(types_logits, positions).cpu()


# ===================================================================
# Main
# ===================================================================

def run(seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    h_dim = 64           # EGNN hidden dim per node
    node_dim = 16        # per-node latent dim
    n_egnn_layers = 4
    n_score_layers = 6
    # Total latent = MAX_ATOMS * node_dim = 29 * 16 = 464
    latent_dim = MAX_ATOMS * node_dim
    t0 = 0.05
    ae_steps = 30000
    diff_steps = 300000
    batch = 256
    n_eval = 10000

    print('=' * 70)
    print(f'QM9 3D — E(3)-Equivariant ULD vs LDM')
    print(f'EGNN h={h_dim}, node_dim={node_dim}, total_latent={latent_dim} '
          f'(={MAX_ATOMS}x{node_dim}), egnn_layers={n_egnn_layers}')
    print(f'AE={ae_steps}, Diff={diff_steps}, t0={t0}')
    print('=' * 70)

    ds = QM93DDataset(augment_rotation=False, seed=seed)  # no augmentation needed — EGNN is equivariant

    # Real baseline
    real, _ = ds.sample_clean(n_eval)
    real_m = ds.evaluate_3d(real)
    print(f'\nReal: atom_sta={real_m["atom_stability"]*100:.1f}%  '
          f'mol_sta={real_m["molecule_stability"]*100:.1f}%  '
          f'valid={real_m["validity"]*100:.1f}%')

    results = {'real_baseline': real_m}

    # === ULD ===
    print('\n' + '=' * 70)
    print('MODEL 1: ULD (EGNN equivariant)')
    print('=' * 70)
    enc_uld = NodeLevelEncoder(N_ATOM_TYPES, h_dim, node_dim, n_egnn_layers, MAX_ATOMS).to(device)
    dec_uld = NodeLevelDecoder(N_ATOM_TYPES, h_dim, node_dim, n_egnn_layers, MAX_ATOMS).to(device)
    score_uld = LatentScoreNet(latent_dim, hidden_dim=256, n_layers=n_score_layers).to(device)

    n_params = sum(p.numel() for p in enc_uld.parameters()) + \
               sum(p.numel() for p in dec_uld.parameters()) + \
               sum(p.numel() for p in score_uld.parameters())
    print(f'  Total params: {n_params:,}')

    print('\n--- ULD Stage 1 (denoising AE at t0) ---')
    train_uld_stage1(enc_uld, dec_uld, ds, ae_steps, lr=3e-4, batch_size=batch,
                     t0=t0, device=device)
    for p in enc_uld.parameters(): p.requires_grad = False
    for p in dec_uld.parameters(): p.requires_grad = False

    z_mean_uld, z_std_uld = compute_latent_stats(enc_uld, ds, batch, device)

    # Recon MSE
    with torch.no_grad():
        x0, _ = ds.sample_clean(2000); x0 = x0.to(device)
        types, pos, mask = flat_to_structured(x0)
        z = enc_uld(types, pos, mask)
        t_hat, p_hat = dec_uld(z, mask)
        x_hat = structured_to_flat(t_hat, p_hat)
        recon_uld = F.mse_loss(x_hat, x0).item()
    print(f'  Recon MSE: {recon_uld:.6f}')

    print('\n--- ULD Stage 2 (latent diffusion, Min-SNR-γ) ---')
    train_diffusion(score_uld, enc_uld, ds, diff_steps, lr=3e-4, batch_size=batch,
                    z_mean=z_mean_uld, z_std=z_std_uld, t0=t0, device=device,
                    is_vae=False)

    print(f'\n  Sampling {n_eval}...')
    gen_uld = sample_3d(score_uld, dec_uld, z_mean_uld, z_std_uld,
                        n_eval, 200, t0, device)
    uld_m = ds.evaluate_3d(gen_uld)
    print(f'  ULD: atom_sta={uld_m["atom_stability"]*100:.1f}%  '
          f'mol_sta={uld_m["molecule_stability"]*100:.1f}%  '
          f'valid={uld_m["validity"]*100:.1f}%  '
          f'v&u={uld_m["valid_unique"]*100:.1f}%')
    results['uld'] = {**uld_m, 'recon_mse': recon_uld}

    # === LDM ===
    print('\n' + '=' * 70)
    print('MODEL 2: LDM (EGNN equivariant VAE)')
    print('=' * 70)
    enc_ldm = NodeLevelVAEEncoder(N_ATOM_TYPES, h_dim, node_dim, n_egnn_layers, MAX_ATOMS).to(device)
    dec_ldm = NodeLevelDecoder(N_ATOM_TYPES, h_dim, node_dim, n_egnn_layers, MAX_ATOMS).to(device)
    score_ldm = LatentScoreNet(latent_dim, hidden_dim=256, n_layers=n_score_layers).to(device)

    print('\n--- LDM Stage 1 (VAE) ---')
    train_vae_stage1(enc_ldm, dec_ldm, ds, ae_steps, lr=3e-4, batch_size=batch,
                     kl_weight=1e-4, device=device)
    for p in enc_ldm.parameters(): p.requires_grad = False
    for p in dec_ldm.parameters(): p.requires_grad = False

    z_mean_ldm, z_std_ldm = compute_latent_stats(enc_ldm, ds, batch, device, is_vae=True)

    with torch.no_grad():
        x0, _ = ds.sample_clean(2000); x0 = x0.to(device)
        types, pos, mask = flat_to_structured(x0)
        z = enc_ldm.encode_deterministic(types, pos, mask)
        t_hat, p_hat = dec_ldm(z, mask)
        x_hat = structured_to_flat(t_hat, p_hat)
        recon_ldm = F.mse_loss(x_hat, x0).item()
    print(f'  VAE Recon MSE: {recon_ldm:.6f}')

    print('\n--- LDM Stage 2 (latent diffusion, Min-SNR-γ) ---')
    train_diffusion(score_ldm, enc_ldm, ds, diff_steps, lr=3e-4, batch_size=batch,
                    z_mean=z_mean_ldm, z_std=z_std_ldm, t0=t0, device=device,
                    is_vae=True)

    print(f'\n  Sampling {n_eval}...')
    gen_ldm = sample_3d(score_ldm, dec_ldm, z_mean_ldm, z_std_ldm,
                        n_eval, 200, t0, device)
    ldm_m = ds.evaluate_3d(gen_ldm)
    print(f'  LDM: atom_sta={ldm_m["atom_stability"]*100:.1f}%  '
          f'mol_sta={ldm_m["molecule_stability"]*100:.1f}%  '
          f'valid={ldm_m["validity"]*100:.1f}%  '
          f'v&u={ldm_m["valid_unique"]*100:.1f}%')
    results['ldm'] = {**ldm_m, 'recon_mse': recon_ldm}

    # Save
    out = os.path.join(LOG_DIR, 'qm9_3d_equivariant_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out}')

    # Summary
    print(f'\n{"="*70}')
    print('QM9 3D EQUIVARIANT SUMMARY')
    print('='*70)
    print(f'  {"Method":<22} {"Atom Sta%":<12} {"Mol Sta%":<12} '
          f'{"Valid%":<12} {"V&U%":<12} {"Recon":<12}')
    print('  '+'-'*72)
    for name, m, rec in [('Data', real_m, '—'),
                          ('ULD (EGNN)', uld_m, f'{recon_uld:.6f}'),
                          ('LDM (EGNN VAE)', ldm_m, f'{recon_ldm:.6f}')]:
        print(f'  {name:<22} {m["atom_stability"]*100:<12.1f} '
              f'{m["molecule_stability"]*100:<12.1f} '
              f'{m["validity"]*100:<12.1f} '
              f'{m["valid_unique"]*100:<12.1f} {rec:<12}')
    print('  '+'-'*72)
    for name, asta, msta, val, vu in [
        ('ENF (pub)', 85.0, 4.9, 40.2, 39.4),
        ('G-SchNet (pub)', 95.7, 68.1, 85.5, 80.3),
        ('EDM (pub)', 98.7, 82.0, 91.9, 90.7),
        ('GeoLDM (pub)', 98.9, 89.4, 93.8, 92.7),
    ]:
        print(f'  {name:<22} {asta:<12.1f} {msta:<12.1f} '
              f'{val:<12.1f} {vu:<12.1f}')


if __name__ == '__main__':
    run()
