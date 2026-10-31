"""Aggregate structured outputs from the matched Stage-2 paper rerun."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.experiment_variant import output_dir, experiment_variant


LOG_DIR = output_dir(__file__, 'logs')


def read_json(name):
    path = os.path.join(LOG_DIR, name)
    with open(path) as f:
        return json.load(f)


def main():
    variant = experiment_variant()
    if variant == 'sild_matches_ldm':
        matching_rule = (
            'Keep Stage 1 t0 unchanged; train and sample both SiLD and LDM '
            'Stage 2 on the LDM interval [0.001, 0.999].'
        )
    else:
        matching_rule = (
            'For each paired comparison, SiLD and LDM Stage 2 both train '
            'and sample on [t0, 0.999].'
        )
    summary = {
        'experiment_variant': variant,
        'matching_rule': matching_rule,
        'stacked_mnist': read_json('stacked_mnist_results_agg.json'),
        'celeba_mmd': read_json('celeba_mmd_sweep_results.json'),
        'celeba_gan': read_json('celeba_gan_results.json'),
        'molecules': {
            dataset: read_json(f'{dataset}_selfies_results.json')
            for dataset in ('qm9', 'hiv', 'muv', 'pcba')
        },
        'celebahq': {
            config: read_json(f'celebahq_{config}_results.json')
            for config in ('small', 'large', 'large_10x')
        },
    }
    out_path = os.path.join(LOG_DIR, 'paper_matched_stage2_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
