"""
Extended evaluation: for each SMILES v2 checkpoint, resample 10k and
compute the standard (validity / uniqueness / novelty / #novel /
novel-over-valid / avg heavy atoms) AND the extended metrics reviewers
expect on molecule generation papers:

  - internal_diversity : 1 - mean pairwise Tanimoto over Morgan FP-2048
  - qed_mean / qed_std : RDKit QED drug-likeness distribution over valids
  - fcd                : Fréchet ChemNet Distance to training set

Writes ``logs/{dataset}_v2_diversity.json`` alongside the earlier
``{dataset}_v2_results.json`` and ``{dataset}_v2_constrained.json``.

Usage: python experiments/evaluate_diversity.py --dataset {qm9,muv,hiv,pcba}
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

LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')

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

    # Per-dataset "main decoding mode" — the honest one that preserves
    # molecule size (see discussion in paper_notes_molecules.md).
    main_mode = 'rdkit_longest' if dataset_key == 'qm9' else 'constrained'

    # --- Load checkpoints ---
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

    # --- Sample ---
    samples = {}
    if 'ULD' in models:
        print(f"Sampling {n_eval} from ULD...")
        samples['ULD'] = sample_diffusion(models['ULD'], n_eval, d, device)
    if 'LDM' in models:
        print(f"Sampling {n_eval} from LDM...")
        samples['LDM'] = sample_diffusion(models['LDM'], n_eval, d, device)
    if 'CharRNN' in models:
        print(f"Sampling {n_eval} from CharRNN...")
        with torch.no_grad():
            samples['CharRNN'] = sample_charrnn(models['CharRNN'], ds,
                                                n_eval, device=device)

    # --- Evaluate under the main decode mode with extended metrics ---
    print(f"\nMain decode mode for {cfg['label']}: {main_mode}")
    results = {'dataset': cfg['label'], 'n_eval': n_eval,
               'main_decode_mode': main_mode,
               'methods': {}}
    for name, gen in samples.items():
        print(f"\n=== {name} / {main_mode} (with diversity + QED + FCD) ===")
        m = ds.evaluate_samples(
            gen, decode_mode=main_mode,
            compute_diversity=True,
            compute_qed=True,
            compute_fcd=True,
            fcd_reference_max=10000,
            diversity_max_n=2000,
        )
        results['methods'][name] = m
        print(f"  validity           = {m['validity']:.4f}")
        print(f"  n_unique           = {m['n_unique']}")
        print(f"  n_novel            = {m['n_novel']}")
        print(f"  novel_over_valid   = {m['novel_over_valid']:.4f}")
        print(f"  avg_heavy_atoms    = {m['avg_heavy_atoms_valid']:.2f}")
        print(f"  internal_diversity = {m['internal_diversity']}")
        print(f"  qed_mean ± std     = {m['qed_mean']} ± {m['qed_std']}")
        print(f"  fcd                = {m['fcd']}")

    # --- Training-set reference with matching extended metrics ---
    print(f"\n=== Training-set reference (n={n_eval}) ===")
    real, _ = ds.sample_clean(min(n_eval, ds.n_total))
    real_m = ds.evaluate_samples(
        real, decode_mode='raw',
        compute_diversity=True,
        compute_qed=True,
        compute_fcd=True,
        fcd_reference_max=10000,
    )
    results['real_baseline'] = real_m
    print(f"  validity           = {real_m['validity']:.4f}")
    print(f"  avg_heavy_atoms    = {real_m['avg_heavy_atoms_valid']:.2f}")
    print(f"  internal_diversity = {real_m['internal_diversity']}")
    print(f"  qed_mean ± std     = {real_m['qed_mean']} ± {real_m['qed_std']}")
    print(f"  fcd (real vs ref)  = {real_m['fcd']}")

    out_path = os.path.join(LOG_DIR, f'{dataset_key}_v2_diversity.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print(f"\n{'='*80}")
    print(f"{cfg['label']} v2 — extended metrics summary ({main_mode} decoding)")
    print('='*80)
    hdr = f"{'Method':<10} {'Val':<7} {'#Uni':<7} {'#Nov':<7} {'avgHA':<7} {'IntDiv':<7} {'QED':<14} {'FCD':<7}"
    print(hdr)
    print('-'*80)
    for name in samples:
        r = results['methods'][name]
        idiv = f"{r['internal_diversity']:.3f}" if r['internal_diversity'] is not None else '—'
        qed  = (f"{r['qed_mean']:.2f}\u00b1{r['qed_std']:.2f}"
                if r['qed_mean'] is not None else '—')
        fcd  = f"{r['fcd']:.2f}" if r['fcd'] is not None else '—'
        print(f"{name:<10} {r['validity']:<7.3f} {r['n_unique']:<7} "
              f"{r['n_novel']:<7} {r['avg_heavy_atoms_valid']:<7.2f} "
              f"{idiv:<7} {qed:<14} {fcd:<7}")
    r = real_m
    idiv = f"{r['internal_diversity']:.3f}"
    qed  = f"{r['qed_mean']:.2f}\u00b1{r['qed_std']:.2f}"
    fcd  = f"{r['fcd']:.2f}"
    print(f"{'Real data':<10} {r['validity']:<7.3f} {r['n_unique']:<7} "
          f"{0:<7} {r['avg_heavy_atoms_valid']:<7.2f} "
          f"{idiv:<7} {qed:<14} {fcd:<7}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(DATASET_CONFIGS.keys()),
                    required=True)
    ap.add_argument('--n_eval', type=int, default=10000)
    ap.add_argument('--seed',   type=int, default=42)
    args = ap.parse_args()
    run(args.dataset, n_eval=args.n_eval, seed=args.seed)
