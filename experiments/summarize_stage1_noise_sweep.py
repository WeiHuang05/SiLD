"""Stacked MNIST Stage-1 noise sweep, all arms on Stage-2 = [0.001, 0.999].

Merges results across EXPERIMENT_VARIANT directories into one table:

  sigma=0      CleanAE + LDM        logs/clean_ae_ldm_stacked/
  sigma=0.001  SiLD t0=0.001        logs/sild_t001/
  sigma=0.005  SiLD t0=0.005        logs/sild_matches_ldm/
  sigma=0.0274 VP schedule @t0=.005 logs/stage1_vp_noise/
  (VAE)        LDM-CNN baseline     logs/sild_matches_ldm/

Reports mean +/- both population and sample std (the paper table mixes the
two), plus paired per-seed differences against the CleanAE arm.
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, 'logs')
SEEDS = (42, 43, 44)


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def collect():
    """Return {arm_name: {'sigma': float, 'by_seed': {seed: {metric: val}}}}."""
    arms = {}

    def put(name, sigma, seed, rec):
        if rec is None:
            return
        slot = arms.setdefault(name, {'sigma': sigma, 'by_seed': {}})
        slot['by_seed'][seed] = {
            'recon_mse': rec['recon_mse'],
            'fid': rec.get('fid'),
            'n_modes_covered': rec.get('n_modes_covered'),
            'kl_to_uniform': rec.get('kl_to_uniform'),
        }

    for seed in SEEDS:
        # SiLD t0=0.005 + the LDM-CNN baseline (both already on [0.001, 0.999])
        d = _load(os.path.join(
            LOGS, 'sild_matches_ldm', f'stacked_mnist_results_t005_seed{seed}.json'))
        if d:
            put('SiLD t0=0.005', 0.005, seed, d.get('uld'))
            put('LDM-CNN (beta-VAE)', float('nan'), seed, d.get('ldm'))

        # SiLD t0=0.001 (this sweep's new point)
        d = _load(os.path.join(
            LOGS, 'sild_t001', f'stacked_mnist_results_t001_seed{seed}.json'))
        if d:
            put('SiLD t0=0.001', 0.001, seed, d.get('uld'))

        # CleanAE + LDM (sigma = 0)
        d = _load(os.path.join(
            LOGS, 'clean_ae_ldm_stacked',
            f'stacked_mnist_clean_ae_ldm_results_tmin001_seed{seed}.json'))
        if d:
            put('CleanAE + LDM', 0.0, seed, d.get('clean_ae_ldm'))

        # VP-schedule Stage-1 noise at t0=0.005 -> sigma = 0.02736
        d = _load(os.path.join(
            LOGS, 'stage1_vp_noise',
            f'stacked_mnist_clean_ae_ldm_results_tmin001_vp_schedule_seed{seed}.json'))
        if d:
            sigma = d.get('config', {}).get('stage1_noise_std', float('nan'))
            put('SiLD VP sigma', sigma, seed, d.get('clean_ae_ldm'))

    return arms


def stats(values):
    n = len(values)
    if n == 0:
        return None
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, 0.0
    var = sum((v - mean) ** 2 for v in values) / n
    var_s = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, var ** 0.5, var_s ** 0.5


def main():
    arms = collect()
    order = ['LDM-CNN (beta-VAE)', 'CleanAE + LDM', 'SiLD t0=0.001',
             'SiLD t0=0.005', 'SiLD VP sigma']

    print('=' * 94)
    print('Stacked MNIST -- Stage-1 noise sweep, Stage-2 = [0.001, 0.999] for every arm')
    print('FID: 10k samples; modes: 25600 samples. std shown as pop / sample(ddof=1).')
    print('=' * 94)
    header = (f"{'arm':22s} {'sigma':>8s} {'n':>2s} {'modes':>6s} "
              f"{'recon x1e-3':>20s} {'FID':>20s}")
    print(header)
    print('-' * 94)

    for name in order:
        arm = arms.get(name)
        if not arm:
            print(f"{name:22s} {'':>8s} {'--':>2s}   (no results yet)")
            continue
        seeds = sorted(arm['by_seed'])
        recon = [arm['by_seed'][s]['recon_mse'] for s in seeds]
        fid = [arm['by_seed'][s]['fid'] for s in seeds
               if arm['by_seed'][s]['fid'] is not None]
        modes = [arm['by_seed'][s]['n_modes_covered'] for s in seeds
                 if arm['by_seed'][s]['n_modes_covered'] is not None]
        r = stats(recon)
        f = stats(fid)
        sigma = arm['sigma']
        sig_s = 'VAE' if sigma != sigma else f'{sigma:.5f}'
        mode_s = f'{min(modes)}' if modes and min(modes) == max(modes) else (
            f'{min(modes)}-{max(modes)}' if modes else '--')
        rs = f'{r[0]*1e3:6.3f}+-{r[1]*1e3:.3f}/{r[2]*1e3:.3f}' if r else '--'
        fs = f'{f[0]:6.3f}+-{f[1]:.3f}/{f[2]:.3f}' if f else '--'
        print(f"{name:22s} {sig_s:>8s} {len(seeds):2d} {mode_s:>6s} {rs:>20s} {fs:>20s}")

    # Paired per-seed FID differences vs CleanAE (sigma = 0)
    base = arms.get('CleanAE + LDM')
    if base:
        print()
        print('Paired per-seed FID difference vs CleanAE (negative = better than CleanAE)')
        print('-' * 94)
        for name in order:
            if name in ('CleanAE + LDM', 'LDM-CNN (beta-VAE)'):
                continue
            arm = arms.get(name)
            if not arm:
                continue
            shared = sorted(set(arm['by_seed']) & set(base['by_seed']))
            diffs = [arm['by_seed'][s]['fid'] - base['by_seed'][s]['fid']
                     for s in shared
                     if arm['by_seed'][s]['fid'] is not None
                     and base['by_seed'][s]['fid'] is not None]
            if not diffs:
                continue
            d = stats(diffs)
            per_seed = '  '.join(
                f'{s}:{v:+.3f}' for s, v in zip(shared, diffs))
            t = (d[0] / (d[2] / len(diffs) ** 0.5)) if d[2] > 0 else float('inf')
            print(f"{name:22s} {per_seed}   mean={d[0]:+.3f} sd={d[2]:.3f} t={t:+.2f}")

    out = os.path.join(LOGS, 'stage1_noise_sweep_summary.json')
    with open(out, 'w') as handle:
        json.dump(arms, handle, indent=2)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
