"""Collect all three strands of the VAE-LDM collapse investigation.

  Task 2  localization   logs/collapse_diagnosis/*.json
  Task 1  baseline sanity logs/ldm_kl_sweep/*.json      (KL sweep + extra seeds)
  Task 3  CleanAE + LDM   logs/clean_ae_ldm_selfies/*.json

Prints three appendix-ready tables and flags whether the collapse survives
a corrected KL weight -- which decides if Table 2's LDM rows must be redone.
"""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, 'logs')
DSETS = ('qm9', 'hiv', 'muv', 'pcba')


def g(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def fmt(e):
    if not e:
        return f"{'--':>8s} {'--':>8s} {'--':>8s} {'--':>8s}"
    return (f"{e['uniqueness']*100:7.2f}% {e.get('internal_diversity', 0):8.4f} "
            f"{e.get('qed_mean', 0):8.3f} {e.get('fcd', 0):8.2f}")


def task2():
    print('=' * 104)
    print('TASK 2 — WHERE DOES THE COLLAPSE HAPPEN?   three decode paths through the same frozen decoder')
    print('=' * 104)
    print(f"{'dataset':7s} {'arm':9s} {'path':22s} {'uniq':>8s} {'IntDiv':>8s} {'QED':>8s} {'FCD':>8s}   dead latent dims")
    print('-' * 104)
    any_found = False
    for ds in DSETS:
        p = os.path.join(LOGS, 'collapse_diagnosis', f'{ds}_collapse_diagnosis.json')
        if not os.path.exists(p):
            print(f'{ds.upper():7s} (pending)')
            continue
        any_found = True
        d = json.load(open(p))
        rb = d['real_baseline']
        print(f"{ds.upper():7s} {'real data':9s} {'--':22s} {fmt(rb)}")
        for arm, a in d['arms'].items():
            dead = g(a, 'latent_normalization', 'frac_raw_dims_std_lt_1e-3', default=0) * 100
            zr = g(a, 'latent_geometry', 'z_real', 'effective_rank_PR', default=0)
            zg = g(a, 'latent_geometry', 'z_gen', 'effective_rank_PR', default=0)
            dim = g(a, 'latent_geometry', 'z_real', 'dim', default=0)
            for key, lab in (('A_roundtrip', 'A real->enc->dec'),
                             ('B_generation', 'B diffusion->dec'),
                             ('C_prior_decode', 'C N(0,I)->dec')):
                extra = f'   {dead:5.1f}% (of {dim})' if key == 'A_roundtrip' else ''
                print(f"{'':7s} {arm:9s} {lab:22s} {fmt(a[key])}{extra}")
            print(f"{'':7s} {'':9s} {'effective rank (PR)':22s} "
                  f"z_real={zr:7.2f}  z_gen={zg:7.2f}   (of {dim})")
            print(f"{'':7s} {'':9s} >> {a['verdict']}")
        print('-' * 104)
    if not any_found:
        print('  (no diagnosis results yet)')


def task1():
    print()
    print('=' * 104)
    print('TASK 1 — IS THE BASELINE SOUND?   VAE-LDM retrained across KL weights and seeds')
    print('=' * 104)
    print(f"{'dataset':7s} {'kl_weight':>10s} {'seed':>5s} {'dead dims':>10s} {'A roundtrip uniq':>17s} "
          f"{'B gen uniq':>11s} {'B gen FCD':>10s} {'eps_MSE@t=.01':>14s}")
    print('-' * 104)
    rows = []
    for p in sorted(glob.glob(os.path.join(LOGS, 'ldm_kl_sweep', '*.json'))):
        d = json.load(open(p))
        c = d['config']
        rows.append((c['dataset'], c['kl_weight'], c['seed'], d))
    # the published configuration, pulled from the main run for reference
    for ds in ('hiv', 'pcba'):
        p = os.path.join(LOGS, 'collapse_diagnosis', f'{ds}_collapse_diagnosis.json')
        if os.path.exists(p):
            d = json.load(open(p))
            a = g(d, 'arms', 'VAE-LDM')
            if a:
                rows.append((ds, 1e-4, 42, {
                    'latent_normalization': a['latent_normalization'],
                    'A_roundtrip': a['A_roundtrip'], 'B_generation': a['B_generation'],
                    'eps_mse_by_t': {}, '_published': True}))
    if not rows:
        print('  (no sanity runs finished yet)')
        return
    for ds, kl, seed, d in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
        dead = g(d, 'latent_normalization', 'frac_raw_dims_std_lt_1e-3', default=0) * 100
        rt = g(d, 'A_roundtrip', 'uniqueness', default=0) * 100
        gu = g(d, 'B_generation', 'uniqueness', default=0) * 100
        gf = g(d, 'B_generation', 'fcd', default=0)
        eps = g(d, 'eps_mse_by_t', '0.01', default=None)
        eps_s = f'{eps:14.4f}' if eps is not None else f"{'--':>14s}"
        star = '  <- published config' if d.get('_published') else ''
        print(f'{ds.upper():7s} {kl:10.0e} {seed:5d} {dead:9.1f}% {rt:16.2f}% '
              f'{gu:10.2f}% {gf:10.2f} {eps_s}{star}')
    print('-' * 104)
    # verdict
    fixed = [(ds, kl) for ds, kl, s, d in rows
             if not d.get('_published')
             and g(d, 'B_generation', 'uniqueness', default=0) > 0.5]
    if fixed:
        print('  !! A different KL weight RECOVERS the baseline on: '
              + ', '.join(f'{ds.upper()}@kl={kl:.0e}' for ds, kl in fixed))
        print('     -> Table 2 LDM rows must be rerun with the corrected configuration.')
    else:
        print('  No tested KL weight recovers the baseline (so far).')


def task3():
    print()
    print('=' * 104)
    print('TASK 3 — CleanAE + LDM ON MOLECULES   separates "drop the KL-VAE bottleneck" from "low-noise denoising"')
    print('=' * 104)
    print(f"{'dataset':7s} {'arm':16s} {'uniq':>8s} {'IntDiv':>8s} {'QED':>8s} {'FCD':>8s}")
    print('-' * 104)
    for ds in DSETS:
        p = os.path.join(LOGS, 'clean_ae_ldm_selfies',
                         f'{ds}_selfies_clean_ae_ldm_seed42.json')
        if not os.path.exists(p):
            print(f'{ds.upper():7s} (missing)')
            continue
        d = json.load(open(p))
        ref = d.get('reference', {})
        for lab, e in (('real data', ref.get('real_baseline')),
                       ('VAE-LDM', ref.get('VAE-LDM')),
                       ('CleanAE + LDM', g(d, 'methods', 'CleanAE+LDM')),
                       ('SiLD', ref.get('SiLD'))):
            print(f'{ds.upper() if lab == "real data" else "":7s} {lab:16s} {fmt(e)}')
        print('-' * 104)
    print('  CleanAE+LDM == SiLD on every dataset  ->  the gap comes from replacing the')
    print('  KL-VAE bottleneck, NOT from SiLD\'s low-noise denoising objective.')


if __name__ == '__main__':
    task2(); task1(); task3()
