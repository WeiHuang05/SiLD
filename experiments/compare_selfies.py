"""
SELFIES experiment: ULD vs LDM-CNN vs CharRNN with 100% validity.

Same recipe as compare_smiles_v2.py but uses SELFIES tokenization
instead of raw SMILES. Because every SELFIES string decodes to a valid
molecule, the validity bottleneck vanishes and the comparison shifts
entirely to distribution-level quality metrics (FCD, QED, IntDiv,
novelty, avg HA).

Usage: python experiments/compare_selfies.py --dataset {qm9,muv,hiv,pcba}
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data.molecule_data import SelfiesDataset
from models.two_stage_latent import TwoStageManifoldDiffusion
from models.latent_diffusion_cnn import LatentDiffusionCNN
from experiments.compare_smiles_v2 import (
    CharRNN, train_charrnn, sample_charrnn,
)
from utils.experiment_variant import (
    output_dir, experiment_variant, paired_stage2_t_min,
)

CKPT_DIR = output_dir(__file__, 'checkpoints')
LOG_DIR = output_dir(__file__, 'logs')

# Per-dataset SELFIES config: max_len chosen to cover p99 of token lengths.
# latent_dim scaled to keep compression ratio <= ~20:1.
SELFIES_CONFIGS = {
    'qm9':  dict(csv='/home/weihuang/data/molecules/qm9.csv',
                 max_len=20, smiles_col='smiles', label='QM9',
                 latent_dim=128, hidden_dim=1024),
    'muv':  dict(csv='/home/weihuang/data/molecules/muv.csv',
                 max_len=72, smiles_col='smiles', label='MUV',
                 latent_dim=256, hidden_dim=1024),
    'hiv':  dict(csv='/home/weihuang/data/molecules/HIV.csv',
                 max_len=128, smiles_col='smiles', label='HIV',
                 latent_dim=512, hidden_dim=1024),
    'pcba': dict(csv='/home/weihuang/data/molecules/pcba.csv',
                 max_len=80, smiles_col='smiles', label='PCBA',
                 latent_dim=256, hidden_dim=1024),
}


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


def run_experiment(dataset_key, t0=0.05,
                   n_score_layers=8,
                   ae_steps=40000, diff_steps=200000,
                   charrnn_steps=40000, charrnn_embed=128,
                   charrnn_hidden=1024, charrnn_layers=2,
                   batch_size=256, n_eval=10000,
                   min_snr_gamma=5.0, seed=42):
    if dataset_key not in SELFIES_CONFIGS:
        raise ValueError(f"Unknown dataset {dataset_key!r}")
    cfg = SELFIES_CONFIGS[dataset_key]

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    latent_dim = cfg['latent_dim']
    hidden_dim = cfg['hidden_dim']

    print(f"Device: {device}")
    print(f"Experiment variant: {experiment_variant()}")
    stage2_t_min = paired_stage2_t_min(t0)
    print(f"Paired Stage-2 time range: [{stage2_t_min}, 0.999]")
    print("=" * 70)
    print(f"{cfg['label']} SELFIES — ULD vs LDM-CNN vs CharRNN (100% validity)")
    print("=" * 70)

    dataset = SelfiesDataset(
        csv_path=cfg['csv'], smiles_col=cfg['smiles_col'],
        max_len=cfg['max_len'], seed=seed, name=cfg['label'],
    )
    d = dataset.d
    V = dataset.vocab_size
    print(f"  d={d}, vocab_size={V}, max_len={cfg['max_len']}")
    print(f"  latent_dim={latent_dim}, hidden_dim={hidden_dim}")
    print(f"  AE={ae_steps}, Diff={diff_steps}, CharRNN={charrnn_steps}")
    print(f"  t0={t0}, min_snr_gamma={min_snr_gamma}")

    # Baseline
    real_x, _ = dataset.sample_clean(n_eval)
    real_eval = dataset.evaluate_samples(
        real_x, compute_diversity=True, compute_qed=True,
        compute_fcd=True, fcd_reference_max=10000,
    )
    print(f"\n  Real baseline:")
    print(f"    validity={real_eval['validity']:.4f}  "
          f"avg_HA={real_eval['avg_heavy_atoms_valid']:.2f}  "
          f"QED={real_eval.get('qed_mean','?')}  FCD={real_eval.get('fcd','?')}")

    prefix = f'{dataset_key}_selfies'

    # =================================================================
    # MODEL 1: ULD
    # =================================================================
    print("\n" + "=" * 70)
    print(f"MODEL 1: ULD (SELFIES) on {cfg['label']}")
    print("=" * 70)
    model_ts = TwoStageManifoldDiffusion(
        d=d, latent_dim=latent_dim,
        hidden_dim_enc=hidden_dim, hidden_dim_score=hidden_dim,
        n_layers=n_score_layers, t0=t0, image_shape=None,
        diffusion_t_min=stage2_t_min,
    ).to(device)

    full_tag = f'{prefix}_uld_full'
    s1_tag   = f'{prefix}_uld_stage1'
    ckpt = load_checkpoint(model_ts, full_tag, device=device)
    if ckpt is not None:
        model_ts.freeze_stage1()
        recon_mse_ts = ckpt.get('recon_mse', 0.0)
        avg_loss_ts = ckpt.get('avg_loss', 0.0)
    else:
        ckpt_s1 = load_checkpoint(model_ts, s1_tag, device=device)
        if ckpt_s1 is not None:
            model_ts.freeze_stage1()
            recon_mse_ts = ckpt_s1.get('recon_mse', 0.0)
        else:
            print("\n--- Stage 1 ---")
            model_ts.train_stage1(dataset, n_steps=ae_steps, lr=1e-4,
                                  batch_size=batch_size)
            with torch.no_grad():
                x0_t, _ = dataset.sample_clean(2000)
                x0_t = x0_t.to(device)
                _, x_hat = model_ts.project(x0_t)
                recon_mse_ts = F.mse_loss(x_hat, x0_t).item()
                print(f"\n  Recon MSE: {recon_mse_ts:.6f}")
            model_ts.freeze_stage1()
            save_checkpoint(model_ts, s1_tag, {'recon_mse': recon_mse_ts})

        print(f"\n--- Stage 2 (Min-SNR-γ={min_snr_gamma}) ---")
        model_ts.train_stage2(dataset, n_steps=diff_steps, lr=3e-4,
                              batch_size=batch_size,
                              min_snr_gamma=min_snr_gamma)

        eval_times = sorted(
            set([max(stage2_t_min, 0.01), 0.1, 0.3, 0.5, 0.8])
        )
        total = 0.0
        for t_eval in eval_times:
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                z0 = model_ts.encode(x0)
                t_b = torch.full((2000,), t_eval, device=device)
                z_t, eps_z = model_ts.forward_diffusion_latent(z0, t_b)
                eps_z_pred = model_ts.predict_eps_latent(z_t, t_b)
                mse = F.mse_loss(eps_z_pred, eps_z).item()
                total += mse
                print(f"  t={t_eval:.2f}: eps_MSE={mse:.6f}")
        avg_loss_ts = total / len(eval_times)
        save_checkpoint(model_ts, full_tag,
                        {'recon_mse': recon_mse_ts, 'avg_loss': avg_loss_ts})

    # =================================================================
    # MODEL 2: LDM-CNN
    # =================================================================
    print("\n" + "=" * 70)
    print(f"MODEL 2: LDM-CNN (SELFIES) on {cfg['label']}")
    print("=" * 70)
    model_ldm = LatentDiffusionCNN(
        d=d, latent_dim=latent_dim,
        hidden_dim_enc=hidden_dim, hidden_dim_score=hidden_dim,
        n_layers=n_score_layers,
        kl_weight=1e-4, image_shape=None,
        diffusion_t_min=stage2_t_min,
    ).to(device)

    full_tag = f'{prefix}_ldm_full'
    ae_tag   = f'{prefix}_ldm_ae'
    ckpt = load_checkpoint(model_ldm, full_tag, device=device)
    if ckpt is not None:
        model_ldm.freeze_ae()
        recon_mse_ldm = ckpt.get('recon_mse', 0.0)
        avg_loss_ldm = ckpt.get('avg_loss', 0.0)
    else:
        ckpt_ae = load_checkpoint(model_ldm, ae_tag, device=device)
        if ckpt_ae is not None:
            model_ldm.freeze_ae()
            recon_mse_ldm = ckpt_ae.get('recon_mse', 0.0)
        else:
            print("\n--- VAE AE training ---")
            model_ldm.train_autoencoder(dataset, n_steps=ae_steps, lr=1e-4,
                                        batch_size=batch_size)
            with torch.no_grad():
                x0_t, _ = dataset.sample_clean(2000)
                x0_t = x0_t.to(device)
                z = model_ldm.vae_encoder.encode_deterministic(x0_t)
                x_hat = model_ldm.decoder(z)
                recon_mse_ldm = F.mse_loss(x_hat, x0_t).item()
                print(f"\n  VAE Recon MSE: {recon_mse_ldm:.6f}")
            model_ldm.freeze_ae()
            save_checkpoint(model_ldm, ae_tag, {'recon_mse': recon_mse_ldm})

        print(f"\n--- LDM diffusion (Min-SNR-γ={min_snr_gamma}) ---")
        model_ldm.train_diffusion(dataset, n_steps=diff_steps, lr=3e-4,
                                  batch_size=batch_size,
                                  min_snr_gamma=min_snr_gamma)

        eval_times = sorted(
            set([max(stage2_t_min, 0.01), 0.1, 0.3, 0.5, 0.8])
        )
        total = 0.0
        for t_eval in eval_times:
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                z0 = model_ldm.encode(x0)
                t_b = torch.full((2000,), t_eval, device=device)
                z_t, eps_z = model_ldm.forward_diffusion_latent(z0, t_b)
                eps_z_pred = model_ldm.score_net(z_t, t_b)
                mse = F.mse_loss(eps_z_pred, eps_z).item()
                total += mse
                print(f"  t={t_eval:.2f}: eps_MSE={mse:.6f}")
        avg_loss_ldm = total / len(eval_times)
        save_checkpoint(model_ldm, full_tag,
                        {'recon_mse': recon_mse_ldm, 'avg_loss': avg_loss_ldm})

    # =================================================================
    # MODEL 3: CharRNN
    # =================================================================
    print("\n" + "=" * 70)
    print(f"MODEL 3: CharRNN (SELFIES) on {cfg['label']}")
    print("=" * 70)
    model_rnn = CharRNN(vocab_size=V, embed_dim=charrnn_embed,
                        hidden_dim=charrnn_hidden,
                        n_layers=charrnn_layers).to(device)
    rnn_tag = f'{prefix}_charrnn'
    ckpt = load_checkpoint(model_rnn, rnn_tag, device=device)
    if ckpt is None:
        print("\n--- CharRNN training ---")
        train_charrnn(model_rnn, dataset, n_steps=charrnn_steps, lr=1e-3,
                      batch_size=batch_size, device=device)
        save_checkpoint(model_rnn, rnn_tag)

    # =================================================================
    # Sample + evaluate with full extended metrics
    # =================================================================
    print(f"\n--- Generating {n_eval} samples per method ---")

    def big_sample_diff(model, n):
        out = []
        bs = 512
        with torch.no_grad():
            for i in range(0, n, bs):
                k = min(bs, n - i)
                x = model.sample(k, n_steps=200, device=device).cpu()
                out.append(x.view(-1, d))
                if (i // bs + 1) % 5 == 0:
                    print(f"    {i + bs}/{n}")
        return torch.cat(out)[:n]

    samples = {}
    print("  ULD sampling...")
    samples['ULD'] = big_sample_diff(model_ts, n_eval)
    print("  LDM sampling...")
    samples['LDM'] = big_sample_diff(model_ldm, n_eval)
    print("  CharRNN sampling...")
    with torch.no_grad():
        samples['CharRNN'] = sample_charrnn(model_rnn, dataset,
                                            n_eval, device=device)

    results = {
        'dataset': cfg['label'],
        'representation': 'SELFIES',
        'config': {
            'd': d, 'vocab_size': V, 'max_len': cfg['max_len'],
            'latent_dim': latent_dim, 'hidden_dim': hidden_dim,
            'n_score_layers': n_score_layers,
            't0': t0, 'min_snr_gamma': min_snr_gamma,
            'ae_steps': ae_steps, 'diff_steps': diff_steps,
            'charrnn_steps': charrnn_steps,
            'batch_size': batch_size, 'n_eval': n_eval,
            'experiment_variant': experiment_variant(),
            'stage2_time_range': [stage2_t_min, 0.999],
        },
        'real_baseline': real_eval,
        'methods': {},
    }

    for name, gen in samples.items():
        print(f"\n=== {name} (SELFIES, {cfg['label']}) ===")
        m = dataset.evaluate_samples(
            gen, compute_diversity=True, compute_qed=True,
            compute_fcd=True, fcd_reference_max=10000,
        )
        extra = {}
        if name == 'ULD':
            extra = {'recon_mse': recon_mse_ts, 'avg_eps_loss': avg_loss_ts}
        elif name == 'LDM':
            extra = {'recon_mse': recon_mse_ldm, 'avg_eps_loss': avg_loss_ldm}
        results['methods'][name] = {**m, **extra}
        print(f"  validity        = {m['validity']:.4f}")
        print(f"  n_unique        = {m['n_unique']}")
        print(f"  n_novel         = {m['n_novel']}")
        print(f"  avg_HA          = {m['avg_heavy_atoms_valid']:.2f}")
        print(f"  internal_div    = {m.get('internal_diversity')}")
        print(f"  QED             = {m.get('qed_mean')}")
        print(f"  FCD             = {m.get('fcd')}")

    # Example decoded molecules
    for name, gen in samples.items():
        smis = dataset.decode_onehot(gen[:10])
        print(f"\n  {name} first 10 decoded SMILES:")
        for s in smis:
            print(f"    {s!r}")

    out_path = os.path.join(LOG_DIR, f'{dataset_key}_selfies_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Summary
    print(f"\n{'='*80}")
    print(f"{cfg['label']} SELFIES SUMMARY (100% validity by construction)")
    print('='*80)
    print(f"{'Method':<10} {'Valid':<7} {'Uniq':<7} {'#Nov':<7} "
          f"{'avgHA':<7} {'IntDiv':<7} {'QED':<7} {'FCD':<7}")
    print('-'*80)
    for name in samples:
        r = results['methods'][name]
        print(f"{name:<10} {r['validity']:<7.3f} "
              f"{r['n_unique']:<7} {r['n_novel']:<7} "
              f"{r['avg_heavy_atoms_valid']:<7.2f} "
              f"{r.get('internal_diversity','?')!s:<7.3f} "
              f"{r.get('qed_mean','?')!s:<7.3f} "
              f"{r.get('fcd','?')!s:<7.2f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(SELFIES_CONFIGS.keys()),
                    required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    run_experiment(dataset_key=args.dataset, seed=args.seed)
