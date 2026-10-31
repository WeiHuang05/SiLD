"""
Post-hoc evaluation of v2 checkpoints under multiple decoding strategies.

Loads the v2 checkpoints for a SMILES dataset (ULD, LDM-CNN, CharRNN),
regenerates ``n_eval`` samples per method, and reports validity /
uniqueness / novelty / #novel / novel-over-valid ratio / avg heavy-atom
count under FOUR decoding strategies:

  - raw            : plain argmax
  - repair         : longest bracket+ring balanced prefix (syntactic only)
  - constrained    : parser-state-masked argmax (grammar-aware decoding)
  - rdkit_longest  : greedy longest RDKit-parseable prefix (uses RDKit as oracle)

The same decoding strategy is applied to all three methods for
fairness. Output written to ``logs/{dataset}_v2_constrained.json``
alongside (not replacing) the original ``{dataset}_v2_results.json``.

Usage: python experiments/evaluate_constrained.py --dataset {qm9,muv,hiv,pcba}
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import torch

from data.molecule_data import SmilesDataset
from models.two_stage_latent import TwoStageManifoldDiffusion
from models.latent_diffusion_cnn import LatentDiffusionCNN
from experiments.compare_smiles_v2 import CharRNN, sample_charrnn, DATASET_CONFIGS

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')

# Match compare_smiles_v2.py hyperparameters
V2_CONFIG = dict(
    latent_dim=128, hidden_dim_enc=1024, hidden_dim_score=1024,
    n_score_layers=8, t0=0.05,
    charrnn_embed=128, charrnn_hidden=1024, charrnn_layers=2,
)


def load_uld(d, device):
    m = TwoStageManifoldDiffusion(
        d=d, latent_dim=V2_CONFIG['latent_dim'],
        hidden_dim_enc=V2_CONFIG['hidden_dim_enc'],
        hidden_dim_score=V2_CONFIG['hidden_dim_score'],
        n_layers=V2_CONFIG['n_score_layers'],
        t0=V2_CONFIG['t0'], image_shape=None,
    ).to(device)
    return m


def load_ldm(d, device):
    m = LatentDiffusionCNN(
        d=d, latent_dim=V2_CONFIG['latent_dim'],
        hidden_dim_enc=V2_CONFIG['hidden_dim_enc'],
        hidden_dim_score=V2_CONFIG['hidden_dim_score'],
        n_layers=V2_CONFIG['n_score_layers'],
        kl_weight=1e-4, image_shape=None,
    ).to(device)
    return m


def load_charrnn(ds, device):
    m = CharRNN(vocab_size=ds.vocab_size,
                embed_dim=V2_CONFIG['charrnn_embed'],
                hidden_dim=V2_CONFIG['charrnn_hidden'],
                n_layers=V2_CONFIG['charrnn_layers']).to(device)
    return m


def sample_diffusion(model, n, d, device, batch=512):
    out = []
    with torch.no_grad():
        for i in range(0, n, batch):
            k = min(batch, n - i)
            x = model.sample(k, n_steps=200, device=device).cpu()
            out.append(x.view(-1, d))
            if (i // batch + 1) % 5 == 0:
                print(f"    {i + batch}/{n}")
    return torch.cat(out)[:n]


def run(dataset_key, n_eval=10000, seed=42):
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset {dataset_key!r}")
    cfg = DATASET_CONFIGS[dataset_key]

    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Dataset: {cfg['label']} ({dataset_key})  max_len={cfg['max_len']}")

    ds = SmilesDataset(csv_path=cfg['csv'], smiles_col=cfg['smiles_col'],
                       max_len=cfg['max_len'], name=cfg['label'], seed=seed)
    d = ds.d

    # --- Load each v2 checkpoint ---
    required = [
        ('ULD',     f'{dataset_key}_v2_uld_full',
                    lambda: load_uld(d, device),
                    lambda m: m.freeze_stage1()),
        ('LDM',     f'{dataset_key}_v2_ldm_full',
                    lambda: load_ldm(d, device),
                    lambda m: m.freeze_ae()),
        ('CharRNN', f'{dataset_key}_v2_charrnn',
                    lambda: load_charrnn(ds, device),
                    lambda m: None),
    ]
    models = {}
    for name, tag, ctor, post in required:
        path = os.path.join(CKPT_DIR, f'{tag}.pt')
        if not os.path.exists(path):
            print(f"  [WARN] {tag}.pt missing — skipping {name}")
            continue
        m = ctor()
        state = torch.load(path, map_location=device)
        m.load_state_dict(state['model'])
        if post is not None:
            post(m)
        if hasattr(m, 'eval'):
            m.eval()
        models[name] = m
        print(f"  Loaded {name} from {path}")

    # --- Sample n_eval from each ---
    samples = {}
    if 'ULD' in models:
        print(f"Sampling {n_eval} from ULD...")
        samples['ULD'] = sample_diffusion(models['ULD'], n_eval, d, device)
    if 'LDM' in models:
        print(f"Sampling {n_eval} from LDM...")
        samples['LDM'] = sample_diffusion(models['LDM'], n_eval, d, device)
    if 'CharRNN' in models:
        print(f"Sampling {n_eval} from CharRNN (autoregressive)...")
        with torch.no_grad():
            samples['CharRNN'] = sample_charrnn(models['CharRNN'], ds,
                                                n_eval, device=device)

    # --- Evaluate each under all 4 decode modes ---
    decode_modes = ['raw', 'repair', 'constrained', 'rdkit_longest']
    results = {'dataset': cfg['label'], 'n_eval': n_eval, 'methods': {}}
    for name, gen in samples.items():
        print(f"\n=== {name} ({cfg['label']}) ===")
        results['methods'][name] = {}
        for mode in decode_modes:
            m = ds.evaluate_samples(gen, decode_mode=mode)
            results['methods'][name][mode] = m
            print(f"  {mode:14}  validity={m['validity']:.4f}  "
                  f"unique={m['n_unique']:<5}  novel={m['n_novel']:<5}  "
                  f"novel/valid={m['novel_over_valid']:.4f}  "
                  f"avg_HA={m['avg_heavy_atoms_valid']:.2f}")

    # Training-set reference
    real, _ = ds.sample_clean(min(n_eval, ds.n_total))
    real_raw = ds.evaluate_samples(real, decode_mode='raw')
    results['real_baseline'] = real_raw

    out_path = os.path.join(LOG_DIR, f'{dataset_key}_v2_constrained.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Compact summary table
    print(f"\n{'='*76}")
    print(f"{cfg['label']} v2 — decoding-strategy sweep")
    print('='*76)
    print(f"{'Method':<10} {'Mode':<14} {'Valid':<8} {'Unique':<8} "
          f"{'Novel':<8} {'N/V':<8} {'avg HA':<8}")
    print('-'*76)
    for name in samples:
        for mode in decode_modes:
            r = results['methods'][name][mode]
            print(f"{name:<10} {mode:<14} "
                  f"{r['validity']:<8.4f} {r['n_unique']:<8} "
                  f"{r['n_novel']:<8} {r['novel_over_valid']:<8.4f} "
                  f"{r['avg_heavy_atoms_valid']:<8.2f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(DATASET_CONFIGS.keys()),
                    required=True)
    ap.add_argument('--n_eval', type=int, default=10000)
    ap.add_argument('--seed',   type=int, default=42)
    args = ap.parse_args()
    run(args.dataset, n_eval=args.n_eval, seed=args.seed)
