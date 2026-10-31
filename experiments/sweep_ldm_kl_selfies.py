"""Baseline sanity check for the VAE-LDM arm on SELFIES benchmarks.

Trains ONLY the VAE-LDM arm (SiLD and CharRNN are untouched), so one run is
roughly half the cost of compare_selfies.py. Two uses:

  * reproduce the collapse across seeds          --seed 43 / 44
  * check it is not an over-regularized latent   --kl-weight 1e-6 ... 1e-3

Every run also emits the same localization diagnostics as
diagnose_selfies_collapse.py (round-trip, prior decode, latent effective
rank), so each sweep point says both *what* happened and *where*.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from data.molecule_data import SelfiesDataset
from models.latent_diffusion_cnn import LatentDiffusionCNN
from experiments.diagnose_selfies_collapse import (
    SELFIES_CONFIGS, latent_stats, decode_in_batches,
    sample_capturing_latent, evaluate,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, 'logs', 'ldm_kl_sweep')
CKPT_DIR = os.path.join(ROOT, 'checkpoints', 'ldm_kl_sweep')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


def kl_tag(kl):
    return f'{kl:.0e}'.replace('-', 'm').replace('+', 'p')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(SELFIES_CONFIGS))
    ap.add_argument('--kl-weight', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ae-steps', type=int, default=40000)
    ap.add_argument('--diff-steps', type=int, default=200000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--min-snr-gamma', type=float, default=5.0)
    ap.add_argument('--n-eval', type=int, default=10000)
    ap.add_argument('--sample-steps', type=int, default=200)
    ap.add_argument('--stage2-t-min', type=float, default=0.001)
    ap.add_argument('--force-train', action='store_true')
    a = ap.parse_args()

    cfg = SELFIES_CONFIGS[a.dataset]
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tag = f'{a.dataset}_kl{kl_tag(a.kl_weight)}_seed{a.seed}'

    print('=' * 78)
    print(f'VAE-LDM BASELINE SANITY — {cfg["label"]}')
    print(f'  kl_weight={a.kl_weight:g}  seed={a.seed}  '
          f'stage2=[{a.stage2_t_min}, 0.999]')
    print(f'  AE={a.ae_steps}  Diff={a.diff_steps}  batch={a.batch_size}  '
          f'min_snr_gamma={a.min_snr_gamma}')
    print(f'  n_eval={a.n_eval}  sample_steps={a.sample_steps}')
    print('=' * 78, flush=True)

    dataset = SelfiesDataset(
        csv_path=cfg['csv'], smiles_col=cfg['smiles_col'],
        max_len=cfg['max_len'], seed=a.seed, name=cfg['label'],
    )
    d = dataset.d
    latent_dim, hidden_dim = cfg['latent_dim'], cfg['hidden_dim']

    x_real, _ = dataset.sample_clean(a.n_eval)
    real_eval = evaluate(dataset, x_real, 'real data')

    model = LatentDiffusionCNN(
        d=d, latent_dim=latent_dim, hidden_dim_enc=hidden_dim,
        hidden_dim_score=hidden_dim, n_layers=8,
        kl_weight=a.kl_weight, image_shape=None,
        diffusion_t_min=a.stage2_t_min,
    ).to(device)

    ckpt_path = os.path.join(CKPT_DIR, f'{tag}_full.pt')
    ae_path = os.path.join(CKPT_DIR, f'{tag}_ae.pt')
    recon_mse = None
    if not a.force_train and os.path.exists(ckpt_path):
        st = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(st['model'])
        recon_mse = st.get('recon_mse')
        print(f'  resumed full: {ckpt_path}')
    else:
        if not a.force_train and os.path.exists(ae_path):
            st = torch.load(ae_path, map_location=device)
            model.load_state_dict(st['model'])
            recon_mse = st.get('recon_mse')
            print(f'  resumed AE: {ae_path}')
        else:
            print('\n--- VAE AE training ---', flush=True)
            model.train_autoencoder(dataset, n_steps=a.ae_steps, lr=1e-4,
                                    batch_size=a.batch_size)
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                z = model.vae_encoder.encode_deterministic(x0)
                recon_mse = F.mse_loss(model.decoder(z), x0).item()
            print(f'  VAE Recon MSE: {recon_mse:.6f}')
            torch.save({'model': model.state_dict(),
                        'recon_mse': recon_mse}, ae_path)
        model.freeze_ae()
        print(f'\n--- LDM diffusion (Min-SNR-g={a.min_snr_gamma}) ---', flush=True)
        model.train_diffusion(dataset, n_steps=a.diff_steps, lr=3e-4,
                              batch_size=a.batch_size,
                              min_snr_gamma=a.min_snr_gamma)
        torch.save({'model': model.state_dict(),
                    'recon_mse': recon_mse}, ckpt_path)
    model.freeze_ae()
    model.eval()

    # eps MSE across time -- is the score net fitting at small t?
    eps_by_t = {}
    for t_eval in (max(a.stage2_t_min, 0.01), 0.1, 0.3, 0.5, 0.8):
        with torch.no_grad():
            x0, _ = dataset.sample_clean(2000)
            x0 = x0.to(device)
            z0 = model.encode(x0)
            tb = torch.full((2000,), t_eval, device=device)
            z_t, eps_z = model.forward_diffusion_latent(z0, tb)
            eps_by_t[f'{t_eval:.2f}'] = F.mse_loss(model.score_net(z_t, tb),
                                                   eps_z).item()
    print('  eps_MSE by t: ' + '  '.join(f't={k}:{v:.4f}'
                                         for k, v in eps_by_t.items()))

    zs_buf = model.z_std.detach().float().cpu()
    norm_info = {
        'raw_z_std_mean': float(zs_buf.mean()),
        'raw_z_std_min': float(zs_buf.min()),
        'raw_z_std_max': float(zs_buf.max()),
        'frac_raw_dims_std_lt_1e-3': float((zs_buf < 1e-3).float().mean()),
    }
    print(f'  raw z_std in [{norm_info["raw_z_std_min"]:.3e}, '
          f'{norm_info["raw_z_std_max"]:.3e}]', flush=True)

    with torch.no_grad():
        z_real = torch.cat([model.encode(x_real[i:i + 512].to(device)).cpu()
                            for i in range(0, a.n_eval, 512)])
    x_rt = decode_in_batches(model, z_real.to(device), d)
    rt_eval = evaluate(dataset, x_rt, 'A round-trip')

    x_gen, z_gen = sample_capturing_latent(model, a.n_eval, d, device,
                                           n_steps=a.sample_steps)
    gen_eval = evaluate(dataset, x_gen, 'B generation')

    z_prior = torch.randn(a.n_eval, latent_dim)
    pr_eval = evaluate(dataset, decode_in_batches(model, z_prior.to(device), d),
                       'C prior decode')

    geo = {'z_real': latent_stats(z_real), 'z_gen': latent_stats(z_gen),
           'z_prior': latent_stats(z_prior)}
    print(f'  effective rank (PR)/{latent_dim}:  '
          f'z_real={geo["z_real"]["effective_rank_PR"]:.1f}  '
          f'z_gen={geo["z_gen"]["effective_rank_PR"]:.1f}')

    out = {
        'dataset': cfg['label'], 'dataset_key': a.dataset,
        'config': {**vars(a), 'latent_dim': latent_dim, 'd': d},
        'real_baseline': real_eval,
        'recon_mse': recon_mse,
        'eps_mse_by_t': eps_by_t,
        'latent_normalization': norm_info,
        'A_roundtrip': rt_eval,
        'B_generation': gen_eval,
        'C_prior_decode': pr_eval,
        'latent_geometry': geo,
    }
    path = os.path.join(LOG_DIR, f'{tag}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved: {path}')


if __name__ == '__main__':
    main()
