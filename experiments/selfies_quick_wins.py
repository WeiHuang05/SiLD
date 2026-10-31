"""
Quick-win experiments for improving SELFIES molecule generation.

Re-uses existing trained ULD checkpoints and evaluates:
  1. Sampling steps sweep: {50, 100, 200, 500}
  2. Predictor-Corrector sampler vs DDIM
  3. t0 sweep: retrain Stage 2 only with different t0 values

Usage:
  python experiments/selfies_quick_wins.py --dataset qm9 --exp steps
  python experiments/selfies_quick_wins.py --dataset qm9 --exp sampler
  python experiments/selfies_quick_wins.py --dataset qm9 --exp t0
  python experiments/selfies_quick_wins.py --dataset qm9 --exp all
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import time
import torch
import torch.nn.functional as F
import numpy as np

from data.molecule_data import SelfiesDataset
from models.two_stage_latent import TwoStageManifoldDiffusion

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

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


def load_uld_model(dataset_key, device, t0=0.05):
    """Load trained ULD model from checkpoint."""
    cfg = SELFIES_CONFIGS[dataset_key]
    dataset = SelfiesDataset(
        csv_path=cfg['csv'], smiles_col=cfg['smiles_col'],
        max_len=cfg['max_len'], seed=42, name=cfg['label'],
    )
    model = TwoStageManifoldDiffusion(
        d=dataset.d, latent_dim=cfg['latent_dim'],
        hidden_dim_enc=cfg['hidden_dim'], hidden_dim_score=cfg['hidden_dim'],
        n_layers=8, t0=t0, image_shape=None,
    ).to(device)

    prefix = f'{dataset_key}_selfies'
    ckpt_path = os.path.join(CKPT_DIR, f'{prefix}_uld_full.pt')
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model'])
        model.freeze_stage1()
        print(f"  Loaded: {ckpt_path}")
    else:
        raise FileNotFoundError(f"No checkpoint: {ckpt_path}")
    return model, dataset


def sample_and_evaluate(model, dataset, n_eval, device, n_steps=200,
                        sampler='ddim', pc_corrector=2, pc_snr=0.1):
    """Sample molecules and evaluate."""
    d = dataset.d
    out = []
    bs = 512
    t_start = time.time()

    with torch.no_grad():
        for i in range(0, n_eval, bs):
            k = min(bs, n_eval - i)
            if sampler == 'ddim':
                x = model.sample(k, n_steps=n_steps, device=device).cpu()
            elif sampler == 'pc':
                x = model.sample_predictor_corrector(
                    k, n_steps=n_steps, n_corrector=pc_corrector,
                    snr=pc_snr, device=device,
                ).cpu()
            out.append(x.view(-1, d))

    samples = torch.cat(out)[:n_eval]
    t_elapsed = time.time() - t_start

    metrics = dataset.evaluate_samples(
        samples, compute_diversity=True, compute_qed=True,
        compute_fcd=True, fcd_reference_max=10000,
    )
    metrics['wall_time_sec'] = round(t_elapsed, 2)
    metrics['n_steps'] = n_steps
    metrics['sampler'] = sampler
    return metrics


def print_metrics(name, m):
    print(f"  {name}: valid={m['validity']:.4f}  uniq={m['uniqueness']:.4f}  "
          f"novel={m['novelty']:.4f}  intdiv={m.get('internal_diversity', '?'):.4f}  "
          f"QED={m.get('qed_mean', '?'):.4f}  FCD={m.get('fcd', '?'):.3f}  "
          f"time={m['wall_time_sec']:.1f}s")


# ======================================================================
# Experiment 1: Sampling steps sweep
# ======================================================================

def exp_steps(dataset_key, device, n_eval=10000):
    print("\n" + "=" * 70)
    print(f"EXP 1: Sampling Steps Sweep ({dataset_key.upper()} SELFIES)")
    print("=" * 70)

    model, dataset = load_uld_model(dataset_key, device)

    results = {}
    for n_steps in [25, 50, 100, 200, 500]:
        print(f"\n  --- n_steps={n_steps} ---")
        m = sample_and_evaluate(model, dataset, n_eval, device, n_steps=n_steps)
        print_metrics(f"steps={n_steps}", m)
        results[n_steps] = m

    out_path = os.path.join(LOG_DIR, f'{dataset_key}_selfies_steps_sweep.json')
    with open(out_path, 'w') as f:
        json.dump({'dataset': dataset_key, 'experiment': 'steps_sweep',
                   'results': {str(k): v for k, v in results.items()}}, f, indent=2)
    print(f"\nSaved: {out_path}")
    return results


# ======================================================================
# Experiment 2: Predictor-Corrector sampler
# ======================================================================

def exp_sampler(dataset_key, device, n_eval=10000):
    print("\n" + "=" * 70)
    print(f"EXP 2: Sampler Comparison ({dataset_key.upper()} SELFIES)")
    print("=" * 70)

    model, dataset = load_uld_model(dataset_key, device)

    results = {}

    # Baseline DDIM
    print("\n  --- DDIM (n_steps=200) ---")
    m = sample_and_evaluate(model, dataset, n_eval, device, n_steps=200, sampler='ddim')
    print_metrics("DDIM-200", m)
    results['ddim_200'] = m

    # PC sampler with different settings
    for n_steps in [100, 200]:
        for n_corr in [1, 2, 4]:
            for snr in [0.05, 0.1, 0.2]:
                tag = f'pc_s{n_steps}_c{n_corr}_snr{snr}'
                print(f"\n  --- PC: steps={n_steps}, corrector={n_corr}, snr={snr} ---")
                m = sample_and_evaluate(model, dataset, n_eval, device,
                                        n_steps=n_steps, sampler='pc',
                                        pc_corrector=n_corr, pc_snr=snr)
                print_metrics(tag, m)
                results[tag] = m

    out_path = os.path.join(LOG_DIR, f'{dataset_key}_selfies_sampler_sweep.json')
    with open(out_path, 'w') as f:
        json.dump({'dataset': dataset_key, 'experiment': 'sampler_sweep',
                   'results': results}, f, indent=2)
    print(f"\nSaved: {out_path}")
    return results


# ======================================================================
# Experiment 3: t0 sweep (retrain Stage 2 only)
# ======================================================================

def exp_t0(dataset_key, device, n_eval=10000,
           diff_steps=200000, batch_size=256):
    print("\n" + "=" * 70)
    print(f"EXP 3: t0 Sweep ({dataset_key.upper()} SELFIES)")
    print("=" * 70)

    cfg = SELFIES_CONFIGS[dataset_key]
    dataset = SelfiesDataset(
        csv_path=cfg['csv'], smiles_col=cfg['smiles_col'],
        max_len=cfg['max_len'], seed=42, name=cfg['label'],
    )
    d = dataset.d

    results = {}
    for t0 in [0.01, 0.02, 0.05, 0.1, 0.2]:
        print(f"\n  --- t0={t0} ---")

        # Create model with this t0
        model = TwoStageManifoldDiffusion(
            d=d, latent_dim=cfg['latent_dim'],
            hidden_dim_enc=cfg['hidden_dim'], hidden_dim_score=cfg['hidden_dim'],
            n_layers=8, t0=t0, image_shape=None,
        ).to(device)

        # Load Stage 1 from existing checkpoint (same encoder/decoder)
        prefix = f'{dataset_key}_selfies'
        s1_path = os.path.join(CKPT_DIR, f'{prefix}_uld_stage1.pt')
        full_path = os.path.join(CKPT_DIR, f'{prefix}_uld_full.pt')

        # Try to load Stage 1 weights
        if os.path.exists(s1_path):
            state = torch.load(s1_path, map_location=device)
            model.load_state_dict(state['model'], strict=False)
            print(f"    Loaded Stage 1 from: {s1_path}")
        elif os.path.exists(full_path):
            state = torch.load(full_path, map_location=device)
            # Load only encoder/decoder weights
            full_sd = state['model']
            partial_sd = {k: v for k, v in full_sd.items()
                          if k.startswith('encoder.') or k.startswith('decoder.')
                          or k.startswith('z_mean') or k.startswith('z_std')}
            model.load_state_dict(partial_sd, strict=False)
            print(f"    Loaded Stage 1 from full checkpoint: {full_path}")
        else:
            print(f"    ERROR: No Stage 1 checkpoint found, skipping t0={t0}")
            continue

        model.freeze_stage1()

        # Check if we already have this t0 variant trained
        t0_tag = f'{prefix}_uld_t0_{t0}'
        t0_ckpt = os.path.join(CKPT_DIR, f'{t0_tag}.pt')
        if os.path.exists(t0_ckpt):
            state = torch.load(t0_ckpt, map_location=device)
            model.load_state_dict(state['model'])
            model.freeze_stage1()
            print(f"    Loaded trained t0={t0} model: {t0_ckpt}")
        else:
            # Retrain Stage 2 with new t0
            print(f"    Training Stage 2 with t0={t0} ...")
            model.train_stage2(dataset, n_steps=diff_steps, lr=3e-4,
                               batch_size=batch_size, min_snr_gamma=5.0)
            torch.save({'model': model.state_dict()}, t0_ckpt)
            print(f"    Saved: {t0_ckpt}")

        # Evaluate
        m = sample_and_evaluate(model, dataset, n_eval, device, n_steps=200)
        print_metrics(f"t0={t0}", m)
        results[str(t0)] = m

    out_path = os.path.join(LOG_DIR, f'{dataset_key}_selfies_t0_sweep.json')
    with open(out_path, 'w') as f:
        json.dump({'dataset': dataset_key, 'experiment': 't0_sweep',
                   'results': results}, f, indent=2)
    print(f"\nSaved: {out_path}")
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(SELFIES_CONFIGS.keys()),
                    required=True)
    ap.add_argument('--exp', choices=['steps', 'sampler', 't0', 'all'],
                    default='all')
    ap.add_argument('--n_eval', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if args.exp in ('steps', 'all'):
        exp_steps(args.dataset, device, n_eval=args.n_eval)

    if args.exp in ('sampler', 'all'):
        exp_sampler(args.dataset, device, n_eval=args.n_eval)

    if args.exp in ('t0', 'all'):
        exp_t0(args.dataset, device, n_eval=args.n_eval)

    print("\nDone!")
