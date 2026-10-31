"""
Experiment: QM9 SMILES — ULD vs VAE-LDM on molecule generation.

Each QM9 molecule is encoded as a 32-position one-hot SMILES tensor over
a 21-character vocabulary, flattened to a 672-dim continuous vector.
Both methods train on the same continuous relaxation and decode via
argmax at inference. The fair-comparison story from Stacked MNIST
carries over: identical MLP encoder, decoder, and latent score network;
only the Stage-1 AE training loss differs.

Metrics (via RDKit):
  - Validity:   fraction of decoded SMILES that RDKit successfully parses
  - Uniqueness: |unique canonical| / |valid|
  - Novelty:    |unique canonical not in training set| / |unique canonical|
  - Reconstruction MSE (on the continuous relaxation)
  - Avg diffusion eps loss

This is a controlled experiment, not a state-of-the-art molecule
generator. A character-level MLP on one-hot SMILES is weaker than an
E(3)-equivariant 3D generator, but it exposes the AE training target's
effect cleanly because every other component is identical.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.molecule_data import QM9SmilesDataset
from models.two_stage_latent import TwoStageManifoldDiffusion
from models.latent_diffusion_cnn import LatentDiffusionCNN


FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


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


def run_experiment(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    max_len = 32
    latent_dim = 64
    hidden_dim_enc = 512
    hidden_dim_score = 512
    n_score_layers = 6
    t0 = 0.05  # per ablation_t0

    ae_steps = 20000
    diff_steps = 80000
    batch_size = 256
    n_eval = 10000   # samples for validity/uniqueness/novelty

    print("=" * 70)
    print("QM9 SMILES — ULD vs LDM-CNN (fair controlled comparison)")
    print("=" * 70)

    dataset = QM9SmilesDataset(max_len=max_len, seed=seed)
    d = dataset.d
    print(f"  d = {d}, vocab_size = {dataset.vocab_size}")
    print(f"  latent_dim={latent_dim}, t0={t0}")
    print(f"  AE={ae_steps}, Diff={diff_steps}, batch={batch_size}, n_eval={n_eval}")

    # Training-set baseline
    real_x, _ = dataset.sample_clean(n_eval)
    real_eval = dataset.evaluate_samples(real_x)
    print(f"\n  Training-set baseline:")
    print(f"    validity={real_eval['validity']:.4f} "
          f"uniqueness={real_eval['uniqueness']:.4f} "
          f"novelty={real_eval['novelty']:.4f}")

    # =================================================================
    # MODEL 1: ULD (TwoStageManifoldDiffusion, MLP mode)
    # =================================================================
    print("\n" + "=" * 70)
    print("MODEL 1: ULD (MLP AE + latent diffusion)")
    print("=" * 70)
    model_ts = TwoStageManifoldDiffusion(
        d=d, latent_dim=latent_dim,
        hidden_dim_enc=hidden_dim_enc, hidden_dim_score=hidden_dim_score,
        n_layers=n_score_layers, t0=t0,
        image_shape=None,  # MLP mode
    ).to(device)

    ckpt = load_checkpoint(model_ts, 'qm9_uld_full', device=device)
    if ckpt is not None:
        print("  Resumed full from checkpoint.")
        model_ts.freeze_stage1()
        recon_mse_ts = ckpt.get('recon_mse', 0.0)
        avg_loss_ts = ckpt.get('avg_loss', 0.0)
    else:
        ckpt_s1 = load_checkpoint(model_ts, 'qm9_uld_stage1', device=device)
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
            save_checkpoint(model_ts, 'qm9_uld_stage1',
                            {'recon_mse': recon_mse_ts})

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
        save_checkpoint(model_ts, 'qm9_uld_full',
                        {'recon_mse': recon_mse_ts, 'avg_loss': avg_loss_ts})

    # =================================================================
    # MODEL 2: LDM-CNN (same architecture in MLP mode)
    # =================================================================
    print("\n" + "=" * 70)
    print("MODEL 2: LDM-CNN (matched MLP AE + same score net)")
    print("=" * 70)
    model_ldm = LatentDiffusionCNN(
        d=d, latent_dim=latent_dim,
        hidden_dim_enc=hidden_dim_enc, hidden_dim_score=hidden_dim_score,
        n_layers=n_score_layers,
        kl_weight=1e-4, image_shape=None,
    ).to(device)

    ckpt = load_checkpoint(model_ldm, 'qm9_ldm_full', device=device)
    if ckpt is not None:
        print("  Resumed full from checkpoint.")
        model_ldm.freeze_ae()
        recon_mse_ldm = ckpt.get('recon_mse', 0.0)
        avg_loss_ldm = ckpt.get('avg_loss', 0.0)
    else:
        ckpt_ae = load_checkpoint(model_ldm, 'qm9_ldm_ae', device=device)
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
            save_checkpoint(model_ldm, 'qm9_ldm_ae',
                            {'recon_mse': recon_mse_ldm})

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
        save_checkpoint(model_ldm, 'qm9_ldm_full',
                        {'recon_mse': recon_mse_ldm, 'avg_loss': avg_loss_ldm})

    # =================================================================
    # Generate + evaluate validity / uniqueness / novelty
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

    # Show a few example SMILES from each
    uld_smis = dataset.decode_onehot(uld_gen[:20])
    ldm_smis = dataset.decode_onehot(ldm_gen[:20])
    print(f"\n  ULD first 20 generated SMILES:")
    for s in uld_smis:
        print(f"    {s!r}")
    print(f"\n  LDM first 20 generated SMILES:")
    for s in ldm_smis:
        print(f"    {s!r}")

    # =================================================================
    # Save summary
    # =================================================================
    summary = {
        'config': {
            'd': d, 'vocab_size': dataset.vocab_size, 'max_len': max_len,
            'latent_dim': latent_dim, 't0': t0,
            'ae_steps': ae_steps, 'diff_steps': diff_steps,
            'batch_size': batch_size, 'n_eval': n_eval,
        },
        'real_baseline': real_eval,
        'uld': {
            'recon_mse': recon_mse_ts,
            'avg_eps_loss': avg_loss_ts,
            **uld_metrics,
        },
        'ldm': {
            'recon_mse': recon_mse_ldm,
            'avg_eps_loss': avg_loss_ldm,
            **ldm_metrics,
        },
    }
    out_path = os.path.join(LOG_DIR, 'qm9_results.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved results JSON: {out_path}")

    print("\n" + "=" * 70)
    print("QM9 SMILES SUMMARY")
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
    run_experiment()
