"""Localize where the VAE-LDM collapse happens on SELFIES benchmarks.

Runs three decode paths through the SAME frozen Stage-1 decoder and compares
the diversity of what comes out:

  A. round-trip   real x -> encode -> decode            (is the AE/decoder itself collapsed?)
  B. generation   diffusion latent -> decode            (the full pipeline, i.e. the Table-2 number)
  C. prior decode z ~ N(0, I) -> decode                 (does the decoder work off the data manifold?)

plus latent-space geometry for each of the three latent sets, so we can say
whether the diffusion produced diverse latents that the decoder then threw
away, or whether the diffusion itself collapsed.

Reading the result:
  A collapsed                      -> representation learning / decoder
  A fine, z_gen collapsed          -> latent diffusion or sampling
  A fine, z_gen diverse, B collapsed -> decoder has no mass off the encoded manifold

Loads the frozen checkpoints from the matched Stage-2 rerun; trains nothing.
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
from models.two_stage_latent import TwoStageManifoldDiffusion
from models.latent_diffusion_cnn import LatentDiffusionCNN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, 'checkpoints', 'sild_matches_ldm')
LOG_DIR = os.path.join(ROOT, 'logs', 'collapse_diagnosis')
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


def latent_stats(z, max_n=4000):
    """Geometry of a latent cloud: spread, effective rank, pairwise distance."""
    z = z.detach().float().cpu()
    n = z.shape[0]
    sd = z.std(dim=0)
    zc = z - z.mean(dim=0, keepdim=True)
    sub = zc[:max_n]
    # participation ratio of the covariance spectrum = effective dimensionality
    cov = (sub.T @ sub) / max(sub.shape[0] - 1, 1)
    ev = torch.linalg.eigvalsh(cov.double()).clamp(min=0)
    pr = (ev.sum() ** 2 / (ev.pow(2).sum() + 1e-30)).item() if ev.sum() > 0 else 0.0
    d = torch.cdist(sub[:2000], sub[:2000])
    iu = torch.triu_indices(d.shape[0], d.shape[0], offset=1)
    return {
        'n': int(n),
        'dim': int(z.shape[1]),
        'per_dim_std_mean': float(sd.mean()),
        'per_dim_std_min': float(sd.min()),
        'per_dim_std_max': float(sd.max()),
        'frac_dims_std_lt_0.01': float((sd < 0.01).float().mean()),
        'effective_rank_PR': pr,
        'effective_rank_PR_frac': pr / z.shape[1],
        'mean_pairwise_dist': float(d[iu[0], iu[1]].mean()),
        'norm_mean': float(z.norm(dim=1).mean()),
    }


def decode_in_batches(model, z, d, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, z.shape[0], bs):
            out.append(model.decode(z[i:i + bs]).view(-1, d).cpu())
    return torch.cat(out)


def sample_capturing_latent(model, n, d, device, n_steps=200, bs=512):
    """Run the real sampler, intercepting the latent handed to decode()."""
    xs, zs = [], []
    real_decode = model.decode

    def spy(z_norm):
        zs.append(z_norm.detach().float().cpu())
        return real_decode(z_norm)

    model.decode = spy
    try:
        with torch.no_grad():
            for i in range(0, n, bs):
                k = min(bs, n - i)
                x = model.sample(k, n_steps=n_steps, device=device).cpu()
                xs.append(x.view(-1, d))
    finally:
        del model.decode          # restore the bound class method
    model.eval()                  # sample() leaves the model in train mode
    return torch.cat(xs)[:n], torch.cat(zs)[:n]


def evaluate(dataset, x, tag):
    print(f"    evaluating {tag} ...", flush=True)
    r = dataset.evaluate_samples(
        x, compute_diversity=True, compute_qed=True,
        compute_fcd=True, fcd_reference_max=10000,
    )
    print(f"      {tag:14s} valid={r['validity']*100:6.2f}% uniq={r['uniqueness']*100:6.2f}% "
          f"IntDiv={r.get('internal_diversity', float('nan')):.4f} "
          f"QED={r.get('qed_mean', float('nan')):.3f} FCD={r.get('fcd', float('nan')):.2f}",
          flush=True)
    return r


def run(dataset_key, n_eval, sample_steps, seed):
    cfg = SELFIES_CONFIGS[dataset_key]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = SelfiesDataset(
        csv_path=cfg['csv'], smiles_col=cfg['smiles_col'],
        max_len=cfg['max_len'], seed=seed, name=cfg['label'],
    )
    d, V = dataset.d, dataset.vocab_size
    latent_dim, hidden_dim = cfg['latent_dim'], cfg['hidden_dim']
    print('=' * 78)
    print(f"COLLAPSE DIAGNOSIS — {cfg['label']} (d={d}, V={V}, latent={latent_dim})")
    print(f"  n_eval={n_eval}, sample_steps={sample_steps}, seed={seed}")
    print(f"  checkpoints: {CKPT_DIR}")
    print('=' * 78)

    x_real, _ = dataset.sample_clean(n_eval)
    real_eval = evaluate(dataset, x_real, 'real data')

    out = {'dataset': cfg['label'], 'dataset_key': dataset_key,
           'config': {'n_eval': n_eval, 'sample_steps': sample_steps,
                      'seed': seed, 'latent_dim': latent_dim,
                      'ckpt_dir': CKPT_DIR, 'stage2_t_min': 0.001},
           'real_baseline': real_eval, 'arms': {}}

    arms = {
        'SiLD': dict(cls=TwoStageManifoldDiffusion, tag=f'{dataset_key}_selfies_uld_full',
                     kw=dict(t0=0.05)),
        'VAE-LDM': dict(cls=LatentDiffusionCNN, tag=f'{dataset_key}_selfies_ldm_full',
                        kw=dict(kl_weight=1e-4)),
    }

    for name, spec in arms.items():
        print(f"\n{'-' * 78}\n### {name}\n{'-' * 78}", flush=True)
        model = spec['cls'](
            d=d, latent_dim=latent_dim, hidden_dim_enc=hidden_dim,
            hidden_dim_score=hidden_dim, n_layers=8, image_shape=None,
            diffusion_t_min=0.001, **spec['kw'],
        ).to(device)
        path = os.path.join(CKPT_DIR, f"{spec['tag']}.pt")
        if not os.path.exists(path):
            print(f"  MISSING checkpoint {path} -- skipping {name}")
            continue
        state = torch.load(path, map_location=device)
        model.load_state_dict(state['model'])
        model.eval()
        print(f"  loaded {path}  (recon_mse={state.get('recon_mse')})")

        # raw latent normalization buffers -- posterior-collapse check
        zs_buf = model.z_std.detach().float().cpu()
        zm_buf = model.z_mean.detach().float().cpu()
        norm_info = {
            'raw_z_std_mean': float(zs_buf.mean()),
            'raw_z_std_min': float(zs_buf.min()),
            'raw_z_std_max': float(zs_buf.max()),
            'raw_z_std_ratio_max_min': float(zs_buf.max() / zs_buf.min().clamp(min=1e-12)),
            'frac_raw_dims_std_lt_1e-3': float((zs_buf < 1e-3).float().mean()),
            'raw_z_mean_norm': float(zm_buf.norm()),
        }
        print(f"  latent-norm buffers: raw z_std in [{norm_info['raw_z_std_min']:.3e}, "
              f"{norm_info['raw_z_std_max']:.3e}], "
              f"frac(<1e-3)={norm_info['frac_raw_dims_std_lt_1e-3']:.3f}")

        # ---- A. round-trip -------------------------------------------------
        with torch.no_grad():
            zr = []
            for i in range(0, n_eval, 512):
                zr.append(model.encode(x_real[i:i + 512].to(device)).cpu())
            z_real = torch.cat(zr)
        x_rt = decode_in_batches(model, z_real.to(device), d)
        rt_eval = evaluate(dataset, x_rt, 'A round-trip')
        rt_mse = F.mse_loss(x_rt, x_real.view(-1, d)).item()

        # ---- B. generation (captures the latent the sampler hands decode) ---
        x_gen, z_gen = sample_capturing_latent(model, n_eval, d, device,
                                               n_steps=sample_steps)
        gen_eval = evaluate(dataset, x_gen, 'B generation')

        # ---- C. prior decode -----------------------------------------------
        z_prior = torch.randn(n_eval, latent_dim)
        x_pr = decode_in_batches(model, z_prior.to(device), d)
        pr_eval = evaluate(dataset, x_pr, 'C prior decode')

        geo = {'z_real': latent_stats(z_real), 'z_gen': latent_stats(z_gen),
               'z_prior': latent_stats(z_prior)}
        print(f"  latent geometry  effective-rank (PR) / {latent_dim}:  "
              f"z_real={geo['z_real']['effective_rank_PR']:.1f}  "
              f"z_gen={geo['z_gen']['effective_rank_PR']:.1f}  "
              f"z_prior={geo['z_prior']['effective_rank_PR']:.1f}")
        print(f"                   mean pairwise dist:  "
              f"z_real={geo['z_real']['mean_pairwise_dist']:.3f}  "
              f"z_gen={geo['z_gen']['mean_pairwise_dist']:.3f}  "
              f"z_prior={geo['z_prior']['mean_pairwise_dist']:.3f}")

        out['arms'][name] = {
            'checkpoint_recon_mse': state.get('recon_mse'),
            'latent_normalization': norm_info,
            'A_roundtrip': rt_eval, 'A_roundtrip_mse': rt_mse,
            'B_generation': gen_eval,
            'C_prior_decode': pr_eval,
            'latent_geometry': geo,
        }

        # verdict
        u_rt, u_gen = rt_eval['uniqueness'], gen_eval['uniqueness']
        pr_real = geo['z_real']['effective_rank_PR']
        pr_gen = geo['z_gen']['effective_rank_PR']
        if u_rt < 0.5 * real_eval['uniqueness']:
            v = 'COLLAPSE IN STAGE 1 (AE/decoder): round-trip already loses diversity'
        elif u_gen > 0.5 * real_eval['uniqueness']:
            v = 'NO COLLAPSE'
        elif pr_gen < 0.5 * pr_real:
            v = 'COLLAPSE IN STAGE 2 (latent diffusion/sampling): generated latents are low-rank'
        else:
            v = ('COLLAPSE IN DECODER OFF-MANIFOLD: latents stay diverse, '
                 'decoder maps them to few molecules')
        out['arms'][name]['verdict'] = v
        print(f"  VERDICT: {v}", flush=True)

    path = os.path.join(LOG_DIR, f'{dataset_key}_collapse_diagnosis.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(SELFIES_CONFIGS))
    ap.add_argument('--n-eval', type=int, default=10000)
    ap.add_argument('--sample-steps', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    run(a.dataset, a.n_eval, a.sample_steps, a.seed)
