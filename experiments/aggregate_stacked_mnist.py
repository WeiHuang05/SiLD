"""
Aggregate Stacked MNIST results across seeds, grouped by t0.

Reads logs/stacked_mnist_results_t{NNN}_seed{N}.json (where NNN = round(t0*1000)),
prints mean +/- std per t0 group for the metrics we care about
(Recon MSE, n_modes_covered, KL_unif, FID, IS), and writes a combined
JSON to logs/stacked_mnist_results_agg.json.
"""

import glob
import json
import os
import re
import math
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.experiment_variant import output_dir

LOG_DIR = output_dir(__file__, 'logs')


def mean_std(values):
    n = len(values)
    if n == 0:
        return float('nan'), float('nan')
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)  # sample std (ddof=1)
    return m, math.sqrt(var)


def parse_tag(path):
    """Parse t and seed from a filename like stacked_mnist_results_t050_seed42.json."""
    m = re.search(r't(\d+)_seed(\d+)\.json$', path)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def load_runs():
    files = sorted(glob.glob(os.path.join(LOG_DIR, 'stacked_mnist_results_t*_seed*.json')))
    runs = []
    for path in files:
        with open(path) as f:
            data = json.load(f)
        t_int, seed = parse_tag(path)
        cfg = data.get('config', {})
        t0 = cfg.get('t0')
        if t0 is None and t_int is not None:
            t0 = t_int / 1000.0
        if seed is None:
            seed = cfg.get('seed')
        data['_path'] = path
        data['_seed'] = seed
        data['_t0'] = t0
        runs.append(data)
    return runs


def report_group(runs, t0):
    methods = ['uld', 'ldm']
    metrics = [
        ('recon_mse',        'Recon MSE',     '.6f'),
        ('avg_eps_loss',     'eps loss',      '.6f'),
        ('n_modes_covered',  '#modes/1000',   '.2f'),
        ('kl_to_uniform',    'KL_unif',       '.4f'),
        ('entropy_nats',     'Entropy(nats)', '.4f'),
        ('fid',              'FID',           '.3f'),
        ('is_mean',          'IS',            '.4f'),
    ]
    seeds = [r['_seed'] for r in runs]
    print()
    print("=" * 86)
    print(f"STACKED MNIST  t0={t0}  (n_seeds={len(runs)},  seeds={seeds})")
    print("=" * 86)
    header = f"  {'Metric':<16}" + "".join(f"  {m.upper():>26}" for m in methods)
    print(header)
    print("  " + "-" * (len(header) - 2))

    group_summary = {'t0': t0, 'n_seeds': len(runs), 'seeds': seeds}
    for key, label, fmt in metrics:
        line = f"  {label:<16}"
        group_summary[key] = {}
        for method in methods:
            vals = [r[method].get(key) for r in runs if key in r.get(method, {})]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
            if not vals:
                line += f"  {'N/A':>26}"
                continue
            m, s = mean_std(vals)
            group_summary[key][method] = {'mean': m, 'std': s, 'values': vals, 'n': len(vals)}
            cell = f"{m:{fmt}} +/- {s:{fmt}}"
            line += f"  {cell:>26}"
        print(line)

    real_modes = [r['real']['n_modes_covered'] for r in runs if 'real' in r]
    real_kl = [r['real']['kl_to_uniform'] for r in runs if 'real' in r]
    if real_modes:
        m, s = mean_std(real_modes)
        print(f"  {'Real #modes':<16}  {m:>23.2f} +/- {s:.2f}")
        m, s = mean_std(real_kl)
        print(f"  {'Real KL_unif':<16}  {m:>23.4f} +/- {s:.4f}")
    print("=" * 86)
    return group_summary


def main():
    runs = load_runs()
    if not runs:
        print(f"No files matched stacked_mnist_results_t*_seed*.json in {LOG_DIR}")
        return

    by_t0 = defaultdict(list)
    for r in runs:
        by_t0[r['_t0']].append(r)

    print(f"Found {len(runs)} runs:")
    for t0, group in sorted(by_t0.items(), key=lambda kv: kv[0] or 0):
        for r in sorted(group, key=lambda x: x['_seed']):
            print(f"  t0={t0:<6}  seed={r['_seed']}  {os.path.basename(r['_path'])}")

    agg = {'groups': []}
    for t0, group in sorted(by_t0.items(), key=lambda kv: kv[0] or 0):
        group_sorted = sorted(group, key=lambda x: x['_seed'])
        agg['groups'].append(report_group(group_sorted, t0))

    out_path = os.path.join(LOG_DIR, 'stacked_mnist_results_agg.json')
    with open(out_path, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f"\n  Saved aggregate JSON: {out_path}")


if __name__ == '__main__':
    main()
