"""
QM9 3D: GeoLDM-ULD vs GeoLDM-VAE (ES-reg).

Full GeoLDM architecture with point-structured equivariant latent space.
Compares ULD's denoising AE vs VAE (ES-reg) for Stage 1, both using
equivariant EGNN score network in Stage 2.

Should achieve results comparable to GeoLDM paper:
  - Atom stability ~98.9%
  - Molecule stability ~89%
  - Validity ~93.8%

Usage: python experiments/compare_qm9_3d_geoldm.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from data.molecule_3d_data import (QM93DDataset, MAX_ATOMS, N_ATOM_TYPES,
                                    DIMS_PER_ATOM, ATOM_TYPES, ATOM_VALENCE)
from models.geoldm_uld import GeoLDMULD, GeoEncoder, GeoDecoder, \
    EquivariantScoreNet, remove_mean

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def save_ckpt(state, name):
    path = os.path.join(CKPT_DIR, f'{name}.pt')
    torch.save(state, path)
    print(f'  Saved: {path}')

def load_ckpt(name, device):
    path = os.path.join(CKPT_DIR, f'{name}.pt')
    if os.path.exists(path):
        print(f'  Loading: {path}')
        return torch.load(path, map_location=device)
    return None


def get_n_atoms_distribution(dataset, n_sample=50000):
    """Count number of atoms per molecule in the training set."""
    from collections import Counter
    counts = Counter()
    x, _ = dataset.sample_clean(min(n_sample, len(dataset.data)))
    B = x.shape[0]
    x = x.view(B, MAX_ATOMS, DIMS_PER_ATOM)
    types = x[:, :, :N_ATOM_TYPES]
    mask = (types.sum(-1) > 0.5)
    for i in range(B):
        na = mask[i].sum().item()
        counts[int(na)] = counts.get(int(na), 0) + 1
    return dict(counts)


def evaluate_generated(types_logits, positions, mask, atoms_list, dataset):
    """Convert generated samples to flat format and evaluate."""
    B = types_logits.shape[0]
    # Argmax types → one-hot
    types_idx = types_logits.argmax(dim=-1)  # (B, N)
    types_oh = F.one_hot(types_idx, N_ATOM_TYPES).float()
    # Zero out padding
    types_oh = types_oh * mask.unsqueeze(-1)
    positions = positions * mask.unsqueeze(-1)
    # Flatten
    x_flat = torch.cat([types_oh, positions], dim=-1).view(B, -1).cpu()
    return dataset.evaluate_3d(x_flat)


def run(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Hyperparameters (following GeoLDM paper more closely)
    h_dim = 256
    latent_h_dim = 1      # k=1 for QM9 (GeoLDM Appendix D)
    n_enc_layers = 1      # shallow encoder (GeoLDM uses 1 layer)
    n_dec_layers = 4
    n_score_layers = 9    # 9 EGCL layers for score net
    t0 = 0.05
    ae_steps = 50000      # more AE steps for better latent
    diff_steps = 1000000  # 1M steps (was 300K) — closer to GeoLDM's ~2000 epochs
    batch_size = 64
    n_eval = 10000
    lr = 1e-4

    print('=' * 70)
    print('QM9 3D — GeoLDM Architecture: ULD vs VAE (ES-reg)')
    print(f'h_dim={h_dim}, latent_h_dim={latent_h_dim} (point-structured)')
    print(f'enc_layers={n_enc_layers}, dec_layers={n_dec_layers}, '
          f'score_layers={n_score_layers}')
    print(f'AE={ae_steps}, Diff={diff_steps}, t0={t0}, batch={batch_size}')
    print('=' * 70)

    ds = QM93DDataset(augment_rotation=False, seed=seed)
    n_atoms_dist = get_n_atoms_distribution(ds)
    print(f'N_atoms distribution (top 5): '
          f'{sorted(n_atoms_dist.items(), key=lambda x: -x[1])[:5]}')

    # Real baseline
    real, _ = ds.sample_clean(n_eval)
    real_m = ds.evaluate_3d(real)
    print(f'\nReal: atom_sta={real_m["atom_stability"]*100:.1f}%  '
          f'mol_sta={real_m["molecule_stability"]*100:.1f}%  '
          f'valid={real_m["validity"]*100:.1f}%')

    results = {'real_baseline': real_m}

    # =================================================================
    # MODEL 1: GeoLDM-ULD (our denoising AE)
    # =================================================================
    print('\n' + '=' * 70)
    print('MODEL 1: GeoLDM-ULD (denoising AE at t0)')
    print('=' * 70)

    model_uld = GeoLDMULD(
        n_atom_types=N_ATOM_TYPES, max_atoms=MAX_ATOMS,
        h_dim=h_dim, latent_h_dim=latent_h_dim,
        n_enc_layers=n_enc_layers, n_dec_layers=n_dec_layers,
        n_score_layers=n_score_layers, t0=t0,
        noise_schedule_type='polynomial',
    ).to(device)

    n_params = sum(p.numel() for p in model_uld.parameters())
    print(f'  Total params: {n_params:,}')

    tag = 'geoldm_uld_v2_full'
    ckpt = load_ckpt(tag, device)
    if ckpt is not None:
        model_uld.load_state_dict(ckpt['model'])
        model_uld.freeze_stage1()
    else:
        s1_tag = 'geoldm_uld_v2_s1'
        s1_ckpt = load_ckpt(s1_tag, device)
        if s1_ckpt is not None:
            model_uld.load_state_dict(s1_ckpt['model'], strict=False)
            model_uld.freeze_stage1()
        else:
            print('\n--- Stage 1: Denoising AE ---')
            model_uld.train_stage1(ds, n_steps=ae_steps, lr=lr,
                                   batch_size=batch_size)
            model_uld.freeze_stage1()
            save_ckpt({'model': model_uld.state_dict()}, s1_tag)

        print('\n--- Stage 2: Equivariant Latent Diffusion ---')
        model_uld.train_stage2(ds, n_steps=diff_steps, lr=lr,
                                batch_size=batch_size)
        save_ckpt({'model': model_uld.state_dict()}, tag)

    # Evaluate
    print(f'\n  Sampling {n_eval} molecules...')
    all_types, all_pos, all_mask = [], [], []
    all_na = []
    bs = 256
    with torch.no_grad():
        remaining = n_eval
        while remaining > 0:
            k = min(bs, remaining)
            t, p, m, na = model_uld.sample(k, n_atoms_dist, n_steps=200,
                                            device=device)
            all_types.append(t.cpu())
            all_pos.append(p.cpu())
            all_mask.append(m.cpu())
            all_na.extend(na)
            remaining -= k
            if len(all_types) % 5 == 0:
                print(f'    {n_eval - remaining}/{n_eval}')

    types_all = torch.cat(all_types)[:n_eval]
    pos_all = torch.cat(all_pos)[:n_eval]
    mask_all = torch.cat(all_mask)[:n_eval]

    uld_m = evaluate_generated(types_all, pos_all, mask_all, all_na, ds)
    print(f'  GeoLDM-ULD: atom_sta={uld_m["atom_stability"]*100:.1f}%  '
          f'mol_sta={uld_m["molecule_stability"]*100:.1f}%  '
          f'valid={uld_m["validity"]*100:.1f}%  '
          f'v&u={uld_m["valid_unique"]*100:.1f}%')
    results['geoldm_uld'] = uld_m

    # =================================================================
    # MODEL 2: GeoLDM-VAE (ES-reg, as in the original paper)
    # =================================================================
    print('\n' + '=' * 70)
    print('MODEL 2: GeoLDM-VAE (ES-reg, original GeoLDM recipe)')
    print('=' * 70)

    model_vae = GeoLDMULD(
        n_atom_types=N_ATOM_TYPES, max_atoms=MAX_ATOMS,
        h_dim=h_dim, latent_h_dim=latent_h_dim,
        n_enc_layers=n_enc_layers, n_dec_layers=n_dec_layers,
        n_score_layers=n_score_layers, t0=0.001,
        noise_schedule_type='polynomial',
    ).to(device)

    tag_vae = 'geoldm_vae_v2_full'
    ckpt = load_ckpt(tag_vae, device)
    if ckpt is not None:
        model_vae.load_state_dict(ckpt['model'])
        model_vae.freeze_stage1()
    else:
        # ES-reg: train encoder for only 1000 steps (early stop)
        s1_tag_vae = 'geoldm_vae_v2_s1'
        s1_ckpt = load_ckpt(s1_tag_vae, device)
        if s1_ckpt is not None:
            model_vae.load_state_dict(s1_ckpt['model'], strict=False)
            model_vae.freeze_stage1()
        else:
            print('\n--- Stage 1: VAE with ES-reg (1000 warmup steps) ---')
            # Train AE but early-stop encoder at 1000 steps
            _params = list(model_vae.encoder.parameters()) + \
                      list(model_vae.decoder.parameters())
            _opt = torch.optim.Adam(_params, lr=lr)
            es_steps = 1000  # early stop encoder

            for step in range(ae_steps):
                x_flat, _ = ds.sample_clean(batch_size)
                x_flat = x_flat.to(device)
                types, positions, mask = model_vae._flat_to_struct(x_flat)
                z_x, z_h = model_vae.encoder(types, positions, mask)
                types_hat, pos_hat = model_vae.decoder(z_x, z_h, mask)

                loss_pos = F.mse_loss(pos_hat * mask.unsqueeze(-1),
                                      positions * mask.unsqueeze(-1))
                loss_types = F.cross_entropy(
                    types_hat[mask > 0.5], types[mask > 0.5].argmax(-1)
                )
                loss = loss_pos + loss_types

                _opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(_params, 1.0)

                if step < es_steps:
                    _opt.step()  # update encoder + decoder
                else:
                    # Only update decoder after early stop
                    for p in model_vae.encoder.parameters():
                        p.grad = None
                    _opt.step()

                if (step + 1) % 500 == 0:
                    enc_status = 'training' if step < es_steps else 'FROZEN'
                    print(f'  VAE-ES [{step+1}/{ae_steps}] loss={loss.item():.4f} '
                          f'enc={enc_status}')

            model_vae.freeze_stage1()
            model_vae._compute_latent_stats(ds, batch_size, device)
            save_ckpt({'model': model_vae.state_dict()}, s1_tag_vae)

        print('\n--- Stage 2: Equivariant Latent Diffusion ---')
        model_vae.train_stage2(ds, n_steps=diff_steps, lr=lr,
                                batch_size=batch_size)
        save_ckpt({'model': model_vae.state_dict()}, tag_vae)

    # Evaluate
    print(f'\n  Sampling {n_eval} molecules...')
    all_types, all_pos, all_mask = [], [], []
    all_na = []
    with torch.no_grad():
        remaining = n_eval
        while remaining > 0:
            k = min(bs, remaining)
            t, p, m, na = model_vae.sample(k, n_atoms_dist, n_steps=200,
                                            device=device)
            all_types.append(t.cpu())
            all_pos.append(p.cpu())
            all_mask.append(m.cpu())
            all_na.extend(na)
            remaining -= k

    types_all = torch.cat(all_types)[:n_eval]
    pos_all = torch.cat(all_pos)[:n_eval]
    mask_all = torch.cat(all_mask)[:n_eval]

    vae_m = evaluate_generated(types_all, pos_all, mask_all, all_na, ds)
    print(f'  GeoLDM-VAE: atom_sta={vae_m["atom_stability"]*100:.1f}%  '
          f'mol_sta={vae_m["molecule_stability"]*100:.1f}%  '
          f'valid={vae_m["validity"]*100:.1f}%  '
          f'v&u={vae_m["valid_unique"]*100:.1f}%')
    results['geoldm_vae'] = vae_m

    # Save results
    out = os.path.join(LOG_DIR, 'qm9_3d_geoldm_v2_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out}')

    # Summary
    print(f'\n{"="*70}')
    print('QM9 3D GEOLDM SUMMARY')
    print('='*70)
    fmt = '  {:<22} {:<12} {:<12} {:<12} {:<12}'
    print(fmt.format('Method', 'Atom Sta%', 'Mol Sta%', 'Valid%', 'V&U%'))
    print('  '+'-'*60)
    for name, m in [('Data', real_m),
                    ('GeoLDM-ULD (ours)', uld_m),
                    ('GeoLDM-VAE (ES-reg)', vae_m)]:
        print(fmt.format(name,
              f'{m["atom_stability"]*100:.1f}',
              f'{m["molecule_stability"]*100:.1f}',
              f'{m["validity"]*100:.1f}',
              f'{m["valid_unique"]*100:.1f}'))
    print('  '+'-'*60)
    print('  Published baselines:')
    for name, asta, msta, val, vu in [
        ('ENF', 85.0, 4.9, 40.2, 39.4),
        ('G-SchNet', 95.7, 68.1, 85.5, 80.3),
        ('EDM', 98.7, 82.0, 91.9, 90.7),
        ('EDM-Bridge', 98.8, 84.6, 92.0, 90.7),
        ('GeoLDM (pub)', 98.9, 89.4, 93.8, 92.7),
    ]:
        print(fmt.format(name, f'{asta}', f'{msta}', f'{val}', f'{vu}'))


if __name__ == '__main__':
    run()
