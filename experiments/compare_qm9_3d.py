"""
QM9 3D: ULD vs LDM-CNN on atom coordinates + atom types.

Direct comparison with EDM/GeoLDM table using the SAME metrics
(atom stability, molecule stability, validity, valid & unique).

We use a simple MLP (no E(3) equivariance) with random rotation
augmentation. d=232 (29 atoms × 8 dims), latent=64, trivially small.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json, torch, torch.nn.functional as F, numpy as np
from data.molecule_3d_data import QM93DDataset
from models.two_stage_latent import TwoStageManifoldDiffusion
from models.latent_diffusion_cnn import LatentDiffusionCNN

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def save_ckpt(model, name, extra=None):
    state = {'model': model.state_dict()}
    if extra: state.update(extra)
    torch.save(state, os.path.join(CKPT_DIR, f'{name}.pt'))

def load_ckpt(model, name, device):
    path = os.path.join(CKPT_DIR, f'{name}.pt')
    if os.path.exists(path):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state['model'])
        return state
    return None


def run(seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    d = 232
    latent_dim = 64
    hidden = 512
    n_layers = 6
    t0 = 0.05
    ae_steps = 30000
    diff_steps = 300000
    batch = 256
    n_eval = 10000

    print('='*70)
    print(f'QM9 3D — ULD vs LDM-CNN (d={d}, latent={latent_dim})')
    print(f'Same metrics as EDM/GeoLDM: atom_sta, mol_sta, validity')
    print(f'AE={ae_steps}, Diff={diff_steps}, batch={batch}, t0={t0}')
    print('='*70)

    ds = QM93DDataset(augment_rotation=True, seed=seed)

    # Real-data baseline
    real, _ = ds.sample_clean(n_eval)
    real_m = ds.evaluate_3d(real)
    print(f'\nReal data baseline:')
    print(f'  atom_sta={real_m["atom_stability"]:.4f}  '
          f'mol_sta={real_m["molecule_stability"]:.4f}  '
          f'valid={real_m["validity"]:.4f}  '
          f'valid_uniq={real_m["valid_unique"]:.4f}')

    results = {'real_baseline': real_m}

    # === ULD ===
    print('\n' + '='*70)
    print('MODEL 1: ULD (MLP, rotation augmented)')
    print('='*70)
    uld = TwoStageManifoldDiffusion(
        d=d, latent_dim=latent_dim, hidden_dim_enc=hidden,
        hidden_dim_score=hidden, n_layers=n_layers, t0=t0,
        image_shape=None,
    ).to(device)

    ck = load_ckpt(uld, 'qm9_3d_uld_full', device)
    if ck:
        uld.freeze_stage1()
        recon_ts = ck.get('recon_mse', 0)
    else:
        ck1 = load_ckpt(uld, 'qm9_3d_uld_s1', device)
        if ck1:
            uld.freeze_stage1()
            recon_ts = ck1.get('recon_mse', 0)
        else:
            uld.train_stage1(ds, n_steps=ae_steps, lr=1e-3, batch_size=batch)
            with torch.no_grad():
                x0, _ = ds.sample_clean(2000); x0 = x0.to(device)
                _, xh = uld.project(x0)
                recon_ts = F.mse_loss(xh, x0).item()
            uld.freeze_stage1()
            save_ckpt(uld, 'qm9_3d_uld_s1', {'recon_mse': recon_ts})
            print(f'  Recon MSE: {recon_ts:.6f}')

        uld.train_stage2(ds, n_steps=diff_steps, lr=3e-4,
                         batch_size=batch, min_snr_gamma=5.0)
        save_ckpt(uld, 'qm9_3d_uld_full', {'recon_mse': recon_ts})

    print(f'\n  Sampling {n_eval} from ULD...')
    with torch.no_grad():
        gen_uld = []
        for i in range(0, n_eval, 512):
            k = min(512, n_eval - i)
            gen_uld.append(uld.sample(k, n_steps=200, device=device).cpu())
        gen_uld = torch.cat(gen_uld)[:n_eval]
    uld_m = ds.evaluate_3d(gen_uld)
    print(f'  ULD: atom_sta={uld_m["atom_stability"]:.4f}  '
          f'mol_sta={uld_m["molecule_stability"]:.4f}  '
          f'valid={uld_m["validity"]:.4f}  '
          f'valid_uniq={uld_m["valid_unique"]:.4f}')
    results['uld'] = {**uld_m, 'recon_mse': recon_ts}

    # === LDM ===
    print('\n' + '='*70)
    print('MODEL 2: LDM-CNN (MLP, rotation augmented)')
    print('='*70)
    ldm = LatentDiffusionCNN(
        d=d, latent_dim=latent_dim, hidden_dim_enc=hidden,
        hidden_dim_score=hidden, n_layers=n_layers, kl_weight=1e-4,
        image_shape=None,
    ).to(device)

    ck = load_ckpt(ldm, 'qm9_3d_ldm_full', device)
    if ck:
        ldm.freeze_ae()
        recon_ldm = ck.get('recon_mse', 0)
    else:
        ck1 = load_ckpt(ldm, 'qm9_3d_ldm_ae', device)
        if ck1:
            ldm.freeze_ae()
            recon_ldm = ck1.get('recon_mse', 0)
        else:
            ldm.train_autoencoder(ds, n_steps=ae_steps, lr=1e-3,
                                  batch_size=batch)
            with torch.no_grad():
                x0, _ = ds.sample_clean(2000); x0 = x0.to(device)
                z = ldm.vae_encoder.encode_deterministic(x0)
                xh = ldm.decoder(z)
                recon_ldm = F.mse_loss(xh, x0).item()
            ldm.freeze_ae()
            save_ckpt(ldm, 'qm9_3d_ldm_ae', {'recon_mse': recon_ldm})
            print(f'  VAE Recon MSE: {recon_ldm:.6f}')

        ldm.train_diffusion(ds, n_steps=diff_steps, lr=3e-4,
                            batch_size=batch, min_snr_gamma=5.0)
        save_ckpt(ldm, 'qm9_3d_ldm_full', {'recon_mse': recon_ldm})

    print(f'\n  Sampling {n_eval} from LDM...')
    with torch.no_grad():
        gen_ldm = []
        for i in range(0, n_eval, 512):
            k = min(512, n_eval - i)
            gen_ldm.append(ldm.sample(k, n_steps=200, device=device).cpu())
        gen_ldm = torch.cat(gen_ldm)[:n_eval]
    ldm_m = ds.evaluate_3d(gen_ldm)
    print(f'  LDM: atom_sta={ldm_m["atom_stability"]:.4f}  '
          f'mol_sta={ldm_m["molecule_stability"]:.4f}  '
          f'valid={ldm_m["validity"]:.4f}  '
          f'valid_uniq={ldm_m["valid_unique"]:.4f}')
    results['ldm'] = {**ldm_m, 'recon_mse': recon_ldm}

    # Save
    out = os.path.join(LOG_DIR, 'qm9_3d_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out}')

    # Summary table matching EDM/GeoLDM format
    print(f'\n{"="*70}')
    print('QM9 3D SUMMARY (compare to EDM/GeoLDM table)')
    print('='*70)
    print(f'  {"Method":<20} {"Atom Sta%":<12} {"Mol Sta%":<12} '
          f'{"Valid%":<12} {"V&U%":<12} {"Recon MSE":<12}')
    print('  '+'-'*68)
    print(f'  {"Data":<20} {real_m["atom_stability"]*100:<12.1f} '
          f'{real_m["molecule_stability"]*100:<12.1f} '
          f'{real_m["validity"]*100:<12.1f} '
          f'{real_m["valid_unique"]*100:<12.1f} {"—":<12}')
    print(f'  {"ULD (ours)":<20} {uld_m["atom_stability"]*100:<12.1f} '
          f'{uld_m["molecule_stability"]*100:<12.1f} '
          f'{uld_m["validity"]*100:<12.1f} '
          f'{uld_m["valid_unique"]*100:<12.1f} {recon_ts:<12.6f}')
    print(f'  {"LDM-CNN":<20} {ldm_m["atom_stability"]*100:<12.1f} '
          f'{ldm_m["molecule_stability"]*100:<12.1f} '
          f'{ldm_m["validity"]*100:<12.1f} '
          f'{ldm_m["valid_unique"]*100:<12.1f} {recon_ldm:<12.6f}')
    # Published baselines for context
    print('  '+'-'*68)
    print(f'  {"ENF (published)":<20} {"85.0":<12} {"4.9":<12} '
          f'{"40.2":<12} {"39.4":<12}')
    print(f'  {"G-SchNet (pub)":<20} {"95.7":<12} {"68.1":<12} '
          f'{"85.5":<12} {"80.3":<12}')
    print(f'  {"EDM (published)":<20} {"98.7":<12} {"82.0":<12} '
          f'{"91.9":<12} {"90.7":<12}')
    print(f'  {"GeoLDM (pub)":<20} {"98.9":<12} {"89.4":<12} '
          f'{"93.8":<12} {"92.7":<12}')


if __name__ == '__main__':
    run()
