"""
GeoLDM-VAE only (ES-reg) — runs in parallel with the full ULD-vs-VAE script.
Uses the existing geoldm_vae_s1.pt checkpoint to skip Stage 1 and only train Stage 2.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json, torch, torch.nn.functional as F, numpy as np
from data.molecule_3d_data import (QM93DDataset, MAX_ATOMS, N_ATOM_TYPES)
from models.geoldm_uld import GeoLDMULD
from experiments.compare_qm9_3d_geoldm import (
    save_ckpt, load_ckpt, get_n_atoms_distribution, evaluate_generated,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)


def run(seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    h_dim = 256; latent_h_dim = 1
    n_enc_layers = 1; n_dec_layers = 4; n_score_layers = 9
    diff_steps = 1000000; batch_size = 64; n_eval = 10000; lr = 1e-4

    print('=' * 70)
    print('QM9 3D — GeoLDM-VAE ONLY (parallel run)')
    print('=' * 70)

    ds = QM93DDataset(augment_rotation=False, seed=seed)
    n_atoms_dist = get_n_atoms_distribution(ds)

    real, _ = ds.sample_clean(n_eval)
    real_m = ds.evaluate_3d(real)
    print(f'Real: atom_sta={real_m["atom_stability"]*100:.1f}%  '
          f'mol_sta={real_m["molecule_stability"]*100:.1f}%  '
          f'valid={real_m["validity"]*100:.1f}%')

    model_vae = GeoLDMULD(
        n_atom_types=N_ATOM_TYPES, max_atoms=MAX_ATOMS,
        h_dim=h_dim, latent_h_dim=latent_h_dim,
        n_enc_layers=n_enc_layers, n_dec_layers=n_dec_layers,
        n_score_layers=n_score_layers, t0=0.001,
        noise_schedule_type='polynomial',
    ).to(device)

    tag_full = 'geoldm_vae_v2_full'
    tag_s1 = 'geoldm_vae_v2_s1'

    ckpt = load_ckpt(tag_full, device)
    if ckpt is not None:
        model_vae.load_state_dict(ckpt['model'])
        model_vae.freeze_stage1()
    else:
        s1 = load_ckpt(tag_s1, device)
        assert s1 is not None, "Need geoldm_vae_s1.pt — run main job first"
        model_vae.load_state_dict(s1['model'], strict=False)
        model_vae.freeze_stage1()

        print('\n--- Stage 2: Equivariant Latent Diffusion ---')
        model_vae.train_stage2(ds, n_steps=diff_steps, lr=lr,
                                batch_size=batch_size)
        save_ckpt({'model': model_vae.state_dict()}, tag_full)

    # Sample
    print(f'\n  Sampling {n_eval} molecules...')
    all_t, all_p, all_m, all_na = [], [], [], []
    bs = 256
    with torch.no_grad():
        remaining = n_eval
        while remaining > 0:
            k = min(bs, remaining)
            t, p, m, na = model_vae.sample(k, n_atoms_dist, n_steps=200, device=device)
            all_t.append(t.cpu()); all_p.append(p.cpu()); all_m.append(m.cpu())
            all_na.extend(na)
            remaining -= k

    types_all = torch.cat(all_t)[:n_eval]
    pos_all = torch.cat(all_p)[:n_eval]
    mask_all = torch.cat(all_m)[:n_eval]

    vae_m = evaluate_generated(types_all, pos_all, mask_all, all_na, ds)
    print(f'\nGeoLDM-VAE: atom_sta={vae_m["atom_stability"]*100:.1f}%  '
          f'mol_sta={vae_m["molecule_stability"]*100:.1f}%  '
          f'valid={vae_m["validity"]*100:.1f}%  '
          f'v&u={vae_m["valid_unique"]*100:.1f}%')

    out = os.path.join(LOG_DIR, 'qm9_3d_geoldm_vae_only.json')
    with open(out, 'w') as f:
        json.dump({'real_baseline': real_m, 'geoldm_vae': vae_m}, f, indent=2)
    print(f'Saved: {out}')


if __name__ == '__main__':
    run()
