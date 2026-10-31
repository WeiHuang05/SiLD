"""
Parameterized SMILES experiment: ULD vs VAE-LDM at matched MLP capacity.

Same recipe as compare_qm9.py but parameterized so we can run it on any
CSV with a SMILES column. Usage:

    python experiments/compare_smiles.py --dataset muv
    python experiments/compare_smiles.py --dataset hiv
    python experiments/compare_smiles.py --dataset pcba

Each dataset gets its own ``{dataset}_uld_full.pt`` and ``{dataset}_ldm_full.pt``
checkpoints plus a ``{dataset}_results.json`` under logs/.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np

from data.molecule_data import SmilesDataset, QM9SmilesDataset
from models.two_stage_latent import TwoStageManifoldDiffusion
from models.latent_diffusion_cnn import LatentDiffusionCNN


CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# Per-dataset defaults. max_len is chosen to cover ~p99 of the
# length distribution while keeping d manageable. smiles_col defaults
# to 'smiles' (all current datasets use that).
DATASET_CONFIGS = {
    'qm9':  dict(csv='/home/weihuang/data/molecules/qm9.csv',
                 max_len=32, smiles_col='smiles', label='QM9'),
    'muv':  dict(csv='/home/weihuang/data/molecules/muv.csv',
                 max_len=64, smiles_col='smiles', label='MUV'),
    'hiv':  dict(csv='/home/weihuang/data/molecules/HIV.csv',
                 max_len=96, smiles_col='smiles', label='HIV'),
    'pcba': dict(csv='/home/weihuang/data/molecules/pcba.csv',
                 max_len=96, smiles_col='smiles', label='PCBA'),
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


def run_experiment(dataset_key, latent_dim=64, hidden_dim_enc=512,
                   hidden_dim_score=512, n_score_layers=6, t0=0.05,
                   ae_steps=20000, diff_steps=80000,
                   batch_size=256, n_eval=10000, seed=42):
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset {dataset_key!r}; "
                         f"known: {list(DATASET_CONFIGS)}")
    cfg = DATASET_CONFIGS[dataset_key]

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 70)
    print(f"{cfg['label']} SMILES — ULD vs LDM-CNN (matched MLP capacity)")
    print("=" * 70)

    dataset = SmilesDataset(csv_path=cfg['csv'], smiles_col=cfg['smiles_col'],
                            max_len=cfg['max_len'], seed=seed,
                            name=cfg['label'])
    d = dataset.d
    print(f"  d={d}, vocab_size={dataset.vocab_size}, max_len={cfg['max_len']}")
    print(f"  latent_dim={latent_dim}, t0={t0}")
    print(f"  AE={ae_steps}, Diff={diff_steps}, batch={batch_size}, n_eval={n_eval}")

    real_x, _ = dataset.sample_clean(n_eval)
    real_eval = dataset.evaluate_samples(real_x)
    print(f"\n  Training-set baseline:")
    print(f"    validity={real_eval['validity']:.4f} "
          f"uniqueness={real_eval['uniqueness']:.4f}")

    # =================================================================
    # MODEL 1: ULD
    # =================================================================
    print("\n" + "=" * 70)
    print(f"MODEL 1: ULD on {cfg['label']}")
    print("=" * 70)
    model_ts = TwoStageManifoldDiffusion(
        d=d, latent_dim=latent_dim,
        hidden_dim_enc=hidden_dim_enc, hidden_dim_score=hidden_dim_score,
        n_layers=n_score_layers, t0=t0, image_shape=None,
    ).to(device)

    full_tag = f'{dataset_key}_uld_full'
    s1_tag = f'{dataset_key}_uld_stage1'
    ckpt = load_checkpoint(model_ts, full_tag, device=device)
    if ckpt is not None:
        print("  Resumed full from checkpoint.")
        model_ts.freeze_stage1()
        recon_mse_ts = ckpt.get('recon_mse', 0.0)
        avg_loss_ts = ckpt.get('avg_loss', 0.0)
    else:
        ckpt_s1 = load_checkpoint(model_ts, s1_tag, device=device)
        if ckpt_s1 is not None:
            print("  Resumed Stage 1.")
            model_ts.freeze_stage1()
            recon_mse_ts = ckpt_s1.get('recon_mse', 0.0)
        else:
            print("\n--- Stage 1: denoising AE at small t0 ---")
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

        print("\n--- Stage 2: latent diffusion on [t0, T] ---")
        model_ts.train_stage2(dataset, n_steps=diff_steps, lr=3e-4,
                              batch_size=batch_size)

        eval_times = [max(t0, 0.01), 0.1, 0.3, 0.5, 0.8]
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
                print(f"  t={t_eval:.2f}: latent_eps_MSE={mse:.6f}")
        avg_loss_ts = total / len(eval_times)
        save_checkpoint(model_ts, full_tag,
                        {'recon_mse': recon_mse_ts, 'avg_loss': avg_loss_ts})

    # =================================================================
    # MODEL 2: LDM-CNN (MLP mode)
    # =================================================================
    print("\n" + "=" * 70)
    print(f"MODEL 2: LDM-CNN on {cfg['label']}")
    print("=" * 70)
    model_ldm = LatentDiffusionCNN(
        d=d, latent_dim=latent_dim,
        hidden_dim_enc=hidden_dim_enc, hidden_dim_score=hidden_dim_score,
        n_layers=n_score_layers,
        kl_weight=1e-4, image_shape=None,
    ).to(device)

    full_tag = f'{dataset_key}_ldm_full'
    ae_tag = f'{dataset_key}_ldm_ae'
    ckpt = load_checkpoint(model_ldm, full_tag, device=device)
    if ckpt is not None:
        print("  Resumed full from checkpoint.")
        model_ldm.freeze_ae()
        recon_mse_ldm = ckpt.get('recon_mse', 0.0)
        avg_loss_ldm = ckpt.get('avg_loss', 0.0)
    else:
        ckpt_ae = load_checkpoint(model_ldm, ae_tag, device=device)
        if ckpt_ae is not None:
            print("  Resumed AE.")
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

        print("\n--- LDM diffusion training ---")
        model_ldm.train_diffusion(dataset, n_steps=diff_steps, lr=3e-4,
                                  batch_size=batch_size)

        eval_times = [0.01, 0.1, 0.3, 0.5, 0.8]
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
                print(f"  t={t_eval:.2f}: latent_eps_MSE={mse:.6f}")
        avg_loss_ldm = total / len(eval_times)
        save_checkpoint(model_ldm, full_tag,
                        {'recon_mse': recon_mse_ldm, 'avg_loss': avg_loss_ldm})

    # =================================================================
    # Generate + evaluate
    # =================================================================
    print(f"\n--- Generating {n_eval} samples per method ---")

    def big_sample(model, n):
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

    print("  ULD sampling...")
    uld_gen = big_sample(model_ts, n_eval)
    print("  LDM sampling...")
    ldm_gen = big_sample(model_ldm, n_eval)

    uld_metrics = dataset.evaluate_samples(uld_gen)
    ldm_metrics = dataset.evaluate_samples(ldm_gen)

    print(f"\n  ULD: {uld_metrics}")
    print(f"  LDM: {ldm_metrics}")

    # Example SMILES
    uld_smis = dataset.decode_onehot(uld_gen[:20])
    ldm_smis = dataset.decode_onehot(ldm_gen[:20])
    print(f"\n  ULD first 20 generated SMILES:")
    for s in uld_smis:
        print(f"    {s!r}")
    print(f"\n  LDM first 20 generated SMILES:")
    for s in ldm_smis:
        print(f"    {s!r}")

    summary = {
        'dataset': cfg['label'],
        'config': {
            'd': d, 'vocab_size': dataset.vocab_size,
            'max_len': cfg['max_len'],
            'latent_dim': latent_dim, 't0': t0,
            'ae_steps': ae_steps, 'diff_steps': diff_steps,
            'batch_size': batch_size, 'n_eval': n_eval,
        },
        'real_baseline': real_eval,
        'uld': {'recon_mse': recon_mse_ts, 'avg_eps_loss': avg_loss_ts,
                **uld_metrics},
        'ldm': {'recon_mse': recon_mse_ldm, 'avg_eps_loss': avg_loss_ldm,
                **ldm_metrics},
    }
    out_path = os.path.join(LOG_DIR, f'{dataset_key}_results.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved results JSON: {out_path}")

    print("\n" + "=" * 70)
    print(f"{cfg['label']} SMILES SUMMARY")
    print("=" * 70)
    print(f"  {'Method':<14} {'Validity':<10} {'Uniqueness':<12} "
          f"{'Novelty':<10} {'Recon MSE':<12} {'Eps Loss':<10}")
    print("  " + "-" * 78)
    print(f"  {'Real (train)':<14} {real_eval['validity']:<10.4f} "
          f"{real_eval['uniqueness']:<12.4f} {real_eval['novelty']:<10.4f} "
          f"{0.0:<12.6f} {0.0:<10.6f}")
    print(f"  {'ULD (ours)':<14} {uld_metrics['validity']:<10.4f} "
          f"{uld_metrics['uniqueness']:<12.4f} {uld_metrics['novelty']:<10.4f} "
          f"{recon_mse_ts:<12.6f} {avg_loss_ts:<10.6f}")
    print(f"  {'LDM-CNN':<14} {ldm_metrics['validity']:<10.4f} "
          f"{ldm_metrics['uniqueness']:<12.4f} {ldm_metrics['novelty']:<10.4f} "
          f"{recon_mse_ldm:<12.6f} {avg_loss_ldm:<10.6f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(DATASET_CONFIGS.keys()),
                    required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    run_experiment(dataset_key=args.dataset, seed=args.seed)
