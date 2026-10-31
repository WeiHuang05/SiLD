"""
Train MOSES reference implementations (CharRNN, VAE, AAE, LatentGAN) on
our four benchmarks (QM9, HIV, MUV, PCBA). Evaluate with OUR unified
metrics protocol (validity, uniqueness, novelty, internal diversity,
QED, FCD over 10K samples) for direct comparison with ULD/LDM.

Usage:
  python experiments/moses_train_all.py --dataset qm9 --model charrnn
  python experiments/moses_train_all.py --dataset qm9 --model all
  python experiments/moses_train_all.py --dataset all --model all
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse, json, pickle, time
import numpy as np
import pandas as pd
import torch

# MOSES
from moses.char_rnn import CharRNN, CharRNNTrainer
from moses.char_rnn.config import get_parser as charrnn_parser
from moses.vae import VAE, VAETrainer
from moses.vae.config import get_parser as vae_parser
from moses.aae import AAE, AAETrainer
from moses.aae.config import get_parser as aae_parser
from moses.latentgan import LatentGAN, LatentGANTrainer
from moses.latentgan.config import get_parser as latentgan_parser
from moses.utils import CharVocab, OneHotVocab
from moses.script_utils import set_seed

# Our evaluator (unified across all methods)
from data.molecule_data import SelfiesDataset, compute_internal_diversity, \
    compute_qed_stats, compute_fcd_score

import selfies as sf
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DATASET_CSVS = {
    'qm9':  '/home/weihuang/data/molecules/qm9.csv',
    'hiv':  '/home/weihuang/data/molecules/HIV.csv',
    'muv':  '/home/weihuang/data/molecules/muv.csv',
    'pcba': '/home/weihuang/data/molecules/pcba.csv',
}

DATASET_SMILES_COLS = {
    'qm9':  'smiles',
    'hiv':  'smiles',
    'muv':  'smiles',
    'pcba': 'smiles',
}


def load_smiles(dataset_key):
    """Read SMILES from CSV, return canonical SMILES list and train/test split."""
    csv = DATASET_CSVS[dataset_key]
    col = DATASET_SMILES_COLS[dataset_key]
    df = pd.read_csv(csv, low_memory=False)
    smiles_list = df[col].astype(str).dropna().tolist()
    # Canonicalize + drop invalid
    canonical = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            canonical.append(Chem.MolToSmiles(mol, canonical=True))
    # Remove duplicates
    canonical = list(dict.fromkeys(canonical))
    # Split: 90% train, 10% test (fixed seed)
    rng = np.random.default_rng(42)
    idx = np.arange(len(canonical))
    rng.shuffle(idx)
    n_test = max(1000, len(canonical) // 10)
    test_idx = set(idx[:n_test].tolist())
    train = [canonical[i] for i in range(len(canonical)) if i not in test_idx]
    test  = [canonical[i] for i in range(len(canonical)) if i     in test_idx]
    return train, test


def smiles_to_mol(smi):
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def evaluate_smiles(gen_smiles, train_smiles, test_smiles, n_eval=10000):
    """Compute validity/uniqueness/novelty/IntDiv/QED/FCD. Uses our evaluators."""
    gen_smiles = list(gen_smiles)[:n_eval]
    n_total = len(gen_smiles)

    # Validity via RDKit
    mols = [smiles_to_mol(s) for s in gen_smiles]
    valid_mask = [m is not None for m in mols]
    valid_mols = [m for m in mols if m is not None]
    valid_smis_canon = [Chem.MolToSmiles(m, canonical=True) for m in valid_mols]

    n_valid = len(valid_mols)
    validity = n_valid / max(n_total, 1)

    # Uniqueness (among valid)
    unique_set = set(valid_smis_canon)
    n_unique = len(unique_set)
    uniqueness = n_unique / max(n_valid, 1)

    # Novelty (among unique, vs train)
    train_set = set(train_smiles)
    novel_set = [s for s in unique_set if s not in train_set]
    n_novel = len(novel_set)
    novelty = n_novel / max(n_unique, 1)

    # Internal diversity (on up to 2000 unique valid mols)
    int_div = compute_internal_diversity(valid_mols, max_n=2000)

    # QED (on valid mols)
    qed_stats = compute_qed_stats(valid_mols)

    # FCD (valid gen vs test) — use up to 10000 test refs
    fcd = compute_fcd_score(valid_smis_canon[:10000], test_smiles[:10000])

    return dict(
        n_total=n_total, n_valid=n_valid, n_unique=n_unique, n_novel=n_novel,
        validity=validity, uniqueness=uniqueness, novelty=novelty,
        internal_diversity=int_div, fcd=fcd, **qed_stats,
    )


# =====================================================================
# MOSES training configs (reduced from defaults for faster iteration)
# =====================================================================

def get_moses_config(model_name, dataset_key):
    """Model-specific config namespace for MOSES Trainer."""
    import argparse as _ap
    if model_name == 'charrnn':
        parser = charrnn_parser()
    elif model_name == 'vae':
        parser = vae_parser()
    elif model_name == 'aae':
        parser = aae_parser()
    elif model_name == 'latentgan':
        parser = latentgan_parser()
    else:
        raise ValueError(model_name)
    # Parse with empty args → defaults
    cfg = parser.parse_known_args([])[0]
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.seed = 42
    cfg.n_workers = 2
    cfg.n_jobs = 2
    # Attributes MOSES trainers expect but aren't in model-specific config
    cfg.log_file = None
    cfg.model_save = None
    cfg.save_frequency = 999999  # disable periodic save
    # Reduce epochs for smaller datasets to keep wall time manageable.
    epochs_map = {'qm9': 40, 'hiv': 40, 'muv': 40, 'pcba': 20}
    if hasattr(cfg, 'train_epochs'):
        cfg.train_epochs = epochs_map[dataset_key]
    if hasattr(cfg, 'n_epochs'):
        cfg.n_epochs = epochs_map[dataset_key]
    if hasattr(cfg, 'n_batch'):
        cfg.n_batch = 256
    return cfg


def train_model(model_name, dataset_key, train_smiles, n_samples=10000):
    print(f"\n=== Training {model_name} on {dataset_key} ===")
    set_seed(42)
    cfg = get_moses_config(model_name, dataset_key)
    device = torch.device(cfg.device)

    # VAE needs OneHotVocab (which has .vectors); others use CharVocab
    if model_name == 'vae':
        vocab = OneHotVocab.from_data(train_smiles)
    else:
        vocab = CharVocab.from_data(train_smiles)

    if model_name == 'charrnn':
        model = CharRNN(vocab, cfg).to(device)
        trainer = CharRNNTrainer(cfg)
    elif model_name == 'vae':
        model = VAE(vocab, cfg).to(device)
        trainer = VAETrainer(cfg)
    elif model_name == 'aae':
        model = AAE(vocab, cfg).to(device)
        trainer = AAETrainer(cfg)
    elif model_name == 'latentgan':
        # LatentGAN requires a pretrained heteroencoder from DeepChem.
        # Skip if not available; caller will handle.
        raise NotImplementedError("LatentGAN needs pretrained heteroencoder.")
    else:
        raise ValueError(model_name)

    t0 = time.time()
    trainer.fit(model, train_smiles)
    train_time = time.time() - t0
    print(f"  Train time: {train_time:.1f}s")

    # Save checkpoint
    tag = f'{model_name}_{dataset_key}'
    ckpt_path = os.path.join(CKPT_DIR, f'moses_{tag}.pt')
    torch.save({'model': model.state_dict(), 'vocab': vocab}, ckpt_path)
    print(f"  Saved: {ckpt_path}")

    # Sample — different model APIs for max length kwarg
    print(f"  Sampling {n_samples} molecules...")
    model.eval()
    with torch.no_grad():
        if model_name == 'charrnn':
            samples = model.sample(n_samples, max_length=100)
        else:
            samples = model.sample(n_samples, max_len=100)
    return samples, train_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all',
                    choices=['all', 'qm9', 'hiv', 'muv', 'pcba'])
    ap.add_argument('--model', default='all',
                    choices=['all', 'charrnn', 'vae', 'aae'])
    ap.add_argument('--n_eval', type=int, default=10000)
    args = ap.parse_args()

    datasets = ['qm9', 'hiv', 'muv', 'pcba'] if args.dataset == 'all' else [args.dataset]
    models = ['charrnn', 'vae', 'aae'] if args.model == 'all' else [args.model]

    all_results = {}
    for ds in datasets:
        print(f"\n{'=' * 70}\nDataset: {ds.upper()}\n{'=' * 70}")
        train_sm, test_sm = load_smiles(ds)
        print(f"  Train: {len(train_sm)} | Test: {len(test_sm)}")
        ds_results = {}
        for mdl in models:
            try:
                samples, tt = train_model(mdl, ds, train_sm, n_samples=args.n_eval)
                m = evaluate_smiles(samples, train_sm, test_sm, n_eval=args.n_eval)
                m['train_time_sec'] = tt
                ds_results[mdl] = m
                print(f"  {mdl}: valid={m['validity']:.4f}  "
                      f"uniq={m['uniqueness']:.4f}  "
                      f"nov={m['novelty']:.4f}  "
                      f"intdiv={m['internal_diversity']:.4f}  "
                      f"qed={m.get('qed_mean', 0):.4f}  "
                      f"fcd={m.get('fcd', 0):.3f}")
            except Exception as e:
                print(f"  ERROR {mdl} on {ds}: {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()
                ds_results[mdl] = {'error': str(e)}
        all_results[ds] = ds_results
        # Save partial as we go
        out = os.path.join(LOG_DIR, 'moses_baselines_all.json')
        with open(out, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved partial: {out}")

    print("\n\n=== FINAL SUMMARY ===")
    print(f"{'Dataset':<8} {'Model':<10} {'Valid':<7} {'Uniq':<7} {'Novel':<7} "
          f"{'IntDiv':<7} {'QED':<7} {'FCD':<7}")
    for ds, ds_res in all_results.items():
        for mdl, m in ds_res.items():
            if 'error' in m:
                print(f"{ds:<8} {mdl:<10}  ERROR: {m['error'][:50]}")
            else:
                print(f"{ds:<8} {mdl:<10} "
                      f"{m['validity']:<7.3f} {m['uniqueness']:<7.3f} "
                      f"{m['novelty']:<7.3f} {m['internal_diversity']:<7.3f} "
                      f"{m.get('qed_mean', 0):<7.3f} {m.get('fcd', 0):<7.2f}")


if __name__ == '__main__':
    main()
