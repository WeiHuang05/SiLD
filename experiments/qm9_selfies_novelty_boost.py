"""
Novelty-boosting sampling experiments for QM9 SELFIES ULD.
Uses existing trained model, only modifies sampling.

Methods tested:
  1. Temperature-scaled initial noise: z_T = N(0, sigma^2 I), sigma > 1
  2. Temperature-scaled score: eps_pred *= T during denoising
  3. Score temperature with SDE sampling (predictor-corrector)

No retraining — just different sampling strategies.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse, json, time, torch
import torch.nn.functional as F
import numpy as np
import math

from data.molecule_data import SelfiesDataset
from models.two_stage_latent import TwoStageManifoldDiffusion

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)


def load_uld():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = dict(csv='/home/weihuang/data/molecules/qm9.csv', max_len=20,
               smiles_col='smiles', label='QM9')
    dataset = SelfiesDataset(csv_path=cfg['csv'], smiles_col=cfg['smiles_col'],
                             max_len=cfg['max_len'], seed=42, name=cfg['label'])
    model = TwoStageManifoldDiffusion(
        d=dataset.d, latent_dim=128, hidden_dim_enc=1024, hidden_dim_score=1024,
        n_layers=8, t0=0.05, image_shape=None,
    ).to(device)
    state = torch.load(os.path.join(CKPT_DIR, 'qm9_selfies_uld_full.pt'),
                       map_location=device)
    model.load_state_dict(state['model'])
    model.freeze_stage1()
    model.eval()
    return model, dataset, device


@torch.no_grad()
def sample_temperature(model, n_samples, device, n_steps=50,
                       init_sigma=1.0, score_temp=1.0, clamp=None):
    """DDIM sampling with adjustable initial noise scale and score temperature.

    init_sigma > 1: sample z_T from a wider Gaussian → more diverse starting points.
    score_temp > 1: downweight the score → more noise retained → more diverse samples.
    """
    model.eval()
    ts = torch.linspace(model.T, model.t0, n_steps + 1, device=device)
    z = init_sigma * torch.randn(n_samples, model.latent_dim, device=device)

    for i in range(n_steps):
        t_cur, t_next = ts[i], ts[i+1]
        t_batch = torch.full((n_samples,), t_cur.item(), device=device)
        alpha_cur, sigma_cur = model.noise_schedule(t_cur)
        alpha_next, sigma_next = model.noise_schedule(t_next)
        eps_pred = model.predict_eps_latent(z, t_batch) / score_temp

        z0_pred = (z - sigma_cur * eps_pred) / torch.sqrt(alpha_cur).clamp(min=1e-6)
        if i == n_steps - 1:
            z = z0_pred
        else:
            z = torch.sqrt(alpha_next) * z0_pred + sigma_next * eps_pred

    x = model.decode(z)
    if clamp is not None:
        x = torch.clamp(x, clamp[0], clamp[1])
    return x


def sample_batched(model, dataset, n, device, **kw):
    out = []
    bs = 512
    with torch.no_grad():
        for i in range(0, n, bs):
            k = min(bs, n - i)
            x = sample_temperature(model, k, device, **kw).cpu()
            out.append(x.view(-1, dataset.d))
    return torch.cat(out)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_eval', type=int, default=10000)
    args = ap.parse_args()

    print("Loading QM9 SELFIES ULD model...")
    model, dataset, device = load_uld()
    d = dataset.d
    print(f"d={d}, latent_dim=128")

    configs = []
    # Baseline
    configs.append(('baseline_n25',            dict(n_steps=25, init_sigma=1.0, score_temp=1.0)))
    configs.append(('baseline_n50',            dict(n_steps=50, init_sigma=1.0, score_temp=1.0)))
    # Initial noise scaling
    for sig in [1.1, 1.2, 1.3, 1.5, 2.0]:
        configs.append((f'init_sigma_{sig}_n50', dict(n_steps=50, init_sigma=sig, score_temp=1.0)))
    # Score temperature
    for T in [1.1, 1.2, 1.3, 1.5]:
        configs.append((f'score_T_{T}_n50',      dict(n_steps=50, init_sigma=1.0, score_temp=T)))
    # Combined
    for sig, T in [(1.2, 1.2), (1.3, 1.2), (1.5, 1.3)]:
        configs.append((f'combo_sig{sig}_T{T}_n50', dict(n_steps=50, init_sigma=sig, score_temp=T)))

    results = {}
    for tag, kw in configs:
        print(f"\n=== {tag}: {kw} ===")
        t_start = time.time()
        samples = sample_batched(model, dataset, args.n_eval, device, **kw)
        t_el = time.time() - t_start
        m = dataset.evaluate_samples(samples, compute_diversity=True,
                                     compute_qed=True, compute_fcd=True,
                                     fcd_reference_max=10000)
        m['wall_time_sec'] = round(t_el, 1)
        m['config'] = kw
        results[tag] = m
        print(f"  valid={m['validity']:.3f}  uniq={m['uniqueness']:.3f}  "
              f"novelty={m['novelty']:.3f}  "
              f"intdiv={m.get('internal_diversity', 0):.3f}  "
              f"QED={m.get('qed_mean', 0):.3f}  FCD={m.get('fcd', 0):.3f}  "
              f"time={t_el:.1f}s")

    out = os.path.join(LOG_DIR, 'qm9_selfies_novelty_boost.json')
    with open(out, 'w') as f:
        json.dump({'results': results}, f, indent=2)
    print(f"\nSaved: {out}")

    # Summary
    print("\n" + "=" * 90)
    print(f"{'Config':<30} {'Valid':<7} {'Uniq':<7} {'Novel':<7} {'IntDiv':<8} {'QED':<7} {'FCD':<7}")
    print("-" * 90)
    for tag, m in results.items():
        print(f"{tag:<30} {m['validity']:<7.3f} {m['uniqueness']:<7.3f} "
              f"{m['novelty']:<7.3f} {m.get('internal_diversity', 0):<8.3f} "
              f"{m.get('qed_mean', 0):<7.3f} {m.get('fcd', 0):<7.2f}")


if __name__ == '__main__':
    main()
