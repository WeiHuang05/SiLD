"""
Run all baselines needed for the full SOTA comparison table.

For each dataset (QM9, HIV, MUV, PCBA), produces:
  - VAE-only (ULD decoder): sample z ~ N(0,I), decode
  - VAE-only (LDM decoder): sample z ~ N(0,I), decode
  - Latent GAN (ULD): WGAN-GP in ULD latent space
  - Latent GAN (LDM): WGAN-GP in LDM latent space
  - ULD + Diffusion: existing trained model
  - LDM + Diffusion: existing trained model
  - CharRNN: existing trained model

All using SELFIES representation.
Also runs SMILES representation for cross-representation comparison.

Usage:
  python experiments/run_all_baselines_for_table.py --dataset hiv
  python experiments/run_all_baselines_for_table.py --dataset all
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse

DATASETS = ['qm9', 'hiv', 'muv', 'pcba']


def run_one(dataset_key, representation):
    """Delegate to the existing ablation script."""
    cmd = (f"python -u experiments/evaluate_ablation_baselines.py "
           f"--dataset {dataset_key} --representation {representation}")
    print(f"\n{'='*70}")
    print(f"Running: {cmd}")
    print(f"{'='*70}")
    os.system(cmd)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=DATASETS + ['all'], default='all')
    ap.add_argument('--representation', choices=['selfies', 'smiles', 'both'],
                    default='selfies')
    args = ap.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    reprs = ['selfies', 'smiles'] if args.representation == 'both' else [args.representation]

    for ds in datasets:
        for rep in reprs:
            run_one(ds, rep)

    print("\n=== All baselines complete ===")
