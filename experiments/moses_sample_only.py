"""
Load trained MOSES checkpoints and only sample + evaluate.
(moses_v3 trained successfully but crashed at sampling due to bugs.)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse, json, torch
from experiments.moses_train_all import (
    load_smiles, evaluate_smiles, get_moses_config, DATASET_CSVS)

from moses.char_rnn import CharRNN
from moses.vae import VAE
from moses.aae import AAE
from moses.utils import CharVocab, OneHotVocab

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_eval', type=int, default=10000)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_results = {}

    for ds in ['qm9', 'hiv', 'muv', 'pcba']:
        print(f"\n==== {ds.upper()} ====")
        train_sm, test_sm = load_smiles(ds)
        ds_results = {}
        for mdl in ['charrnn', 'vae', 'aae']:
            ckpt_path = os.path.join(CKPT_DIR, f'moses_{mdl}_{ds}.pt')
            if not os.path.exists(ckpt_path):
                print(f"  {mdl}: NO CHECKPOINT")
                ds_results[mdl] = {'error': 'no_checkpoint'}
                continue
            try:
                ck = torch.load(ckpt_path, map_location=device, weights_only=False)
                vocab = ck['vocab']
                cfg = get_moses_config(mdl, ds)

                if mdl == 'charrnn':
                    model = CharRNN(vocab, cfg).to(device)
                elif mdl == 'vae':
                    model = VAE(vocab, cfg).to(device)
                elif mdl == 'aae':
                    model = AAE(vocab, cfg).to(device)

                model.load_state_dict(ck['model'])
                model.eval()
                print(f"  Sampling {args.n_eval} from {mdl} on {ds}...")
                with torch.no_grad():
                    if mdl == 'charrnn':
                        samples = model.sample(args.n_eval, max_length=100)
                    else:
                        samples = model.sample(args.n_eval, max_len=100)

                m = evaluate_smiles(samples, train_sm, test_sm, n_eval=args.n_eval)
                ds_results[mdl] = m
                print(f"  {mdl}: valid={m['validity']:.4f}  uniq={m['uniqueness']:.4f}  "
                      f"nov={m['novelty']:.4f}  intdiv={m['internal_diversity']:.4f}  "
                      f"qed={m.get('qed_mean', 0):.4f}  fcd={m.get('fcd', 0):.3f}")
            except Exception as e:
                print(f"  {mdl}: ERROR {type(e).__name__}: {e}")
                import traceback; traceback.print_exc()
                ds_results[mdl] = {'error': str(e)}

        all_results[ds] = ds_results
        out = os.path.join(LOG_DIR, 'moses_sample_results.json')
        with open(out, 'w') as f:
            json.dump(all_results, f, indent=2)

    print("\n===== FINAL =====")
    for ds, ds_res in all_results.items():
        for mdl, m in ds_res.items():
            if 'error' in m:
                print(f"  {ds:<6} {mdl:<10} ERROR: {m['error'][:60]}")
            else:
                print(f"  {ds:<6} {mdl:<10} "
                      f"v={m['validity']:.3f} u={m['uniqueness']:.3f} "
                      f"n={m['novelty']:.3f} id={m['internal_diversity']:.3f} "
                      f"qed={m.get('qed_mean', 0):.3f} fcd={m.get('fcd', 0):.2f}")


if __name__ == '__main__':
    main()
