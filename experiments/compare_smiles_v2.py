"""
SMILES experiment v2: ULD vs LDM-CNN vs CharRNN.

Changes from v1 (compare_smiles.py):
  - Latent dim 64 -> 128, hidden 512 -> 1024, 6 score layers -> 8
  - AE steps 20k -> 40k, diff steps 80k -> 200k
  - Min-SNR-gamma=5 weighting in both ULD Stage 2 and LDM diffusion
  - Third baseline: **CharRNN** (2-layer LSTM, ~matched parameter count)
    that consumes the SAME one-hot representation as ULD/LDM. Trained
    with teacher-forced next-char cross-entropy; sampled autoregressively
    and converted back to the one-hot format for the unified evaluator.
  - Validity-aware decoding: ``evaluate_samples(..., repair=True)``
    applies repair_smiles() to every decoded string before RDKit
    parsing. Same repair is applied to all three methods for fairness.

Writes a second set of checkpoints under ``*_v2_*`` so the v1 results
aren't clobbered.

Usage: python experiments/compare_smiles_v2.py --dataset {qm9,muv,hiv,pcba}
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data.molecule_data import SmilesDataset
from models.two_stage_latent import TwoStageManifoldDiffusion
from models.latent_diffusion_cnn import LatentDiffusionCNN


CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


DATASET_CONFIGS = {
    'qm9':  dict(csv='/home/weihuang/data/molecules/qm9.csv',
                 max_len=32, smiles_col='smiles', label='QM9'),
    'muv':  dict(csv='/home/weihuang/data/molecules/muv.csv',
                 max_len=64, smiles_col='smiles', label='MUV'),
    'hiv':  dict(csv='/home/weihuang/data/molecules/HIV.csv',
                 max_len=96, smiles_col='smiles', label='HIV'),
    'pcba': dict(csv='/home/weihuang/data/molecules/pcba.csv',
                 max_len=96, smiles_col='smiles', label='PCBA'),
}


# =====================================================================
# CharRNN baseline — 2-layer LSTM, matched parameter scale
# =====================================================================

class CharRNN(nn.Module):
    """
    Character-level LSTM language model over SMILES tokens.

    Input:  integer token ids (B, L)
    Output: logits over vocab (B, L, V)

    Parameter count is scaled to roughly match the ULD/LDM score net
    at the same configuration (embed=128, hidden=512 or 1024, 2 layers).
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=512, n_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=0.1 if n_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        n = sum(p.numel() for p in self.parameters())
        print(f"    CharRNN: vocab={vocab_size}, embed={embed_dim}, "
              f"hidden={hidden_dim}, n_layers={n_layers}, params={n:,}")

    def forward(self, x, hidden=None):
        h = self.embed(x)
        out, hidden = self.lstm(h, hidden)
        return self.fc(out), hidden


def train_charrnn(model, dataset, n_steps, lr, batch_size, device):
    """Teacher-forced next-token cross-entropy training on SMILES."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)
    V = dataset.vocab_size
    L = dataset.max_len
    for step in range(n_steps):
        x_flat, _ = dataset.sample_clean(batch_size)
        # (B, L, V) one-hot → (B, L) int ids
        tokens = x_flat.view(-1, L, V).argmax(-1).to(device)
        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]
        logits, _ = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); scheduler.step()
        if (step + 1) % 500 == 0:
            print(f"  CharRNN [{step+1}/{n_steps}] xent={loss.item():.4f}")


@torch.no_grad()
def sample_charrnn(model, dataset, n, device, temperature=1.0):
    """
    Autoregressive sampling: seed with a random starting token from the
    training marginal, then unroll until PAD or max_len. Returns a
    (n, d) flat one-hot tensor in the same format the diffusion models
    produce, so the same evaluator can consume it.
    """
    model.eval()
    L = dataset.max_len
    V = dataset.vocab_size
    # Marginal distribution over the first non-PAD token (from training data)
    all_tokens = dataset.data_onehot.argmax(-1)    # (N, L)
    first = all_tokens[:, 0]
    first_hist = torch.bincount(first, minlength=V).float()
    first_hist[0] = 0  # never start with PAD
    first_dist = first_hist / first_hist.sum().clamp(min=1)

    samples = torch.zeros(n, L, dtype=torch.long)
    # Batch the autoregressive loop to go faster
    bs = 256
    for start in range(0, n, bs):
        end = min(start + bs, n)
        b = end - start
        # Sample initial tokens
        init = torch.multinomial(first_dist, b, replacement=True).to(device)
        seqs = torch.zeros(b, L, dtype=torch.long, device=device)
        seqs[:, 0] = init
        hidden = None
        x_in = init.unsqueeze(1)  # (b, 1)
        for t in range(1, L):
            logits, hidden = model(x_in, hidden)
            logits = logits[:, -1] / max(temperature, 1e-4)  # (b, V)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).squeeze(-1)    # (b,)
            seqs[:, t] = nxt
            x_in = nxt.unsqueeze(1)
        samples[start:end] = seqs.cpu()

    # Zero out everything after the first PAD on each row (clean padding)
    for i in range(n):
        pad_positions = (samples[i] == 0).nonzero(as_tuple=True)[0]
        if len(pad_positions) > 0:
            first_pad = pad_positions[0].item()
            samples[i, first_pad:] = 0  # already 0, but explicit

    # Convert to one-hot (n, L, V) → flatten to (n, d)
    one_hot = torch.zeros(n, L, V, dtype=torch.float32)
    one_hot.scatter_(2, samples.unsqueeze(-1), 1.0)
    return one_hot.view(n, -1)


# =====================================================================
# Checkpoint helpers
# =====================================================================

def save_checkpoint(model, name, extra=None):
    path = os.path.join(CKPT_DIR, f'{name}.pt')
    state = {'model': model.state_dict()}
    if extra:
        state.update(extra)
    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(model, name, device='cpu'):
    path = os.path.join(CKPT_DIR, f'{name}.pt')
    if os.path.exists(path):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state['model'])
        print(f"  Checkpoint loaded: {path}")
        return state
    return None


# =====================================================================
# Main
# =====================================================================

def run_experiment(dataset_key,
                   latent_dim=128, hidden_dim_enc=1024, hidden_dim_score=1024,
                   n_score_layers=8, t0=0.05,
                   ae_steps=40000, diff_steps=200000, charrnn_steps=40000,
                   batch_size=256, n_eval=10000,
                   charrnn_embed=128, charrnn_hidden=1024, charrnn_layers=2,
                   min_snr_gamma=5.0, seed=42):
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset {dataset_key!r}")
    cfg = DATASET_CONFIGS[dataset_key]

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 70)
    print(f"{cfg['label']} SMILES v2 — ULD vs LDM-CNN vs CharRNN")
    print("=" * 70)

    dataset = SmilesDataset(csv_path=cfg['csv'], smiles_col=cfg['smiles_col'],
                            max_len=cfg['max_len'], seed=seed,
                            name=cfg['label'])
    d = dataset.d
    V = dataset.vocab_size
    print(f"  d={d}, vocab_size={V}, max_len={cfg['max_len']}")
    print(f"  latent_dim={latent_dim}, hidden_enc={hidden_dim_enc}, "
          f"hidden_score={hidden_dim_score}, n_score_layers={n_score_layers}")
    print(f"  AE={ae_steps}, Diff={diff_steps}, CharRNN={charrnn_steps}, "
          f"batch={batch_size}, n_eval={n_eval}")
    print(f"  t0={t0}, min_snr_gamma={min_snr_gamma}")

    real_x, _ = dataset.sample_clean(n_eval)
    real_eval_raw    = dataset.evaluate_samples(real_x, repair=False)
    real_eval_repair = dataset.evaluate_samples(real_x, repair=True)
    print(f"\n  Training-set baseline (raw):      "
          f"validity={real_eval_raw['validity']:.4f}")
    print(f"  Training-set baseline (repaired): "
          f"validity={real_eval_repair['validity']:.4f}")

    # =================================================================
    # MODEL 1: ULD
    # =================================================================
    print("\n" + "=" * 70)
    print(f"MODEL 1: ULD (v2 + Min-SNR-γ={min_snr_gamma}) on {cfg['label']}")
    print("=" * 70)
    model_ts = TwoStageManifoldDiffusion(
        d=d, latent_dim=latent_dim,
        hidden_dim_enc=hidden_dim_enc, hidden_dim_score=hidden_dim_score,
        n_layers=n_score_layers, t0=t0, image_shape=None,
    ).to(device)

    full_tag = f'{dataset_key}_v2_uld_full'
    s1_tag   = f'{dataset_key}_v2_uld_stage1'
    ckpt = load_checkpoint(model_ts, full_tag, device=device)
    if ckpt is not None:
        print("  Resumed full from checkpoint.")
        model_ts.freeze_stage1()
        recon_mse_ts = ckpt.get('recon_mse', 0.0)
        avg_loss_ts = ckpt.get('avg_loss', 0.0)
    else:
        ckpt_s1 = load_checkpoint(model_ts, s1_tag, device=device)
        if ckpt_s1 is not None:
            print("  Resumed Stage 1.")
            model_ts.freeze_stage1()
            recon_mse_ts = ckpt_s1.get('recon_mse', 0.0)
        else:
            print("\n--- Stage 1: denoising AE at small t0 ---")
            model_ts.train_stage1(dataset, n_steps=ae_steps, lr=1e-4,
                                  batch_size=batch_size)
            with torch.no_grad():
                x0_t, _ = dataset.sample_clean(2000)
                x0_t = x0_t.to(device)
                _, x_hat = model_ts.project(x0_t)
                recon_mse_ts = F.mse_loss(x_hat, x0_t).item()
                print(f"\n  Recon MSE: {recon_mse_ts:.6f}")
            model_ts.freeze_stage1()
            save_checkpoint(model_ts, s1_tag, {'recon_mse': recon_mse_ts})

        print(f"\n--- Stage 2 (Min-SNR-γ={min_snr_gamma}) ---")
        model_ts.train_stage2(dataset, n_steps=diff_steps, lr=3e-4,
                              batch_size=batch_size,
                              min_snr_gamma=min_snr_gamma)

        eval_times = [max(t0, 0.01), 0.1, 0.3, 0.5, 0.8]
        total = 0.0
        for t_eval in eval_times:
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                z0 = model_ts.encode(x0)
                t_b = torch.full((2000,), t_eval, device=device)
                z_t, eps_z = model_ts.forward_diffusion_latent(z0, t_b)
                eps_z_pred = model_ts.predict_eps_latent(z_t, t_b)
                mse = F.mse_loss(eps_z_pred, eps_z).item()
                total += mse
                print(f"  t={t_eval:.2f}: latent_eps_MSE={mse:.6f}")
        avg_loss_ts = total / len(eval_times)
        save_checkpoint(model_ts, full_tag,
                        {'recon_mse': recon_mse_ts, 'avg_loss': avg_loss_ts})

    # =================================================================
    # MODEL 2: LDM-CNN
    # =================================================================
    print("\n" + "=" * 70)
    print(f"MODEL 2: LDM-CNN (v2 + Min-SNR-γ={min_snr_gamma}) on {cfg['label']}")
    print("=" * 70)
    model_ldm = LatentDiffusionCNN(
        d=d, latent_dim=latent_dim,
        hidden_dim_enc=hidden_dim_enc, hidden_dim_score=hidden_dim_score,
        n_layers=n_score_layers,
        kl_weight=1e-4, image_shape=None,
    ).to(device)

    full_tag = f'{dataset_key}_v2_ldm_full'
    ae_tag   = f'{dataset_key}_v2_ldm_ae'
    ckpt = load_checkpoint(model_ldm, full_tag, device=device)
    if ckpt is not None:
        print("  Resumed full from checkpoint.")
        model_ldm.freeze_ae()
        recon_mse_ldm = ckpt.get('recon_mse', 0.0)
        avg_loss_ldm = ckpt.get('avg_loss', 0.0)
    else:
        ckpt_ae = load_checkpoint(model_ldm, ae_tag, device=device)
        if ckpt_ae is not None:
            print("  Resumed AE.")
            model_ldm.freeze_ae()
            recon_mse_ldm = ckpt_ae.get('recon_mse', 0.0)
        else:
            print("\n--- VAE AE training ---")
            model_ldm.train_autoencoder(dataset, n_steps=ae_steps, lr=1e-4,
                                        batch_size=batch_size)
            with torch.no_grad():
                x0_t, _ = dataset.sample_clean(2000)
                x0_t = x0_t.to(device)
                z = model_ldm.vae_encoder.encode_deterministic(x0_t)
                x_hat = model_ldm.decoder(z)
                recon_mse_ldm = F.mse_loss(x_hat, x0_t).item()
                print(f"\n  VAE Recon MSE: {recon_mse_ldm:.6f}")
            model_ldm.freeze_ae()
            save_checkpoint(model_ldm, ae_tag, {'recon_mse': recon_mse_ldm})

        print(f"\n--- LDM diffusion (Min-SNR-γ={min_snr_gamma}) ---")
        model_ldm.train_diffusion(dataset, n_steps=diff_steps, lr=3e-4,
                                  batch_size=batch_size,
                                  min_snr_gamma=min_snr_gamma)

        eval_times = [0.01, 0.1, 0.3, 0.5, 0.8]
        total = 0.0
        for t_eval in eval_times:
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                z0 = model_ldm.encode(x0)
                t_b = torch.full((2000,), t_eval, device=device)
                z_t, eps_z = model_ldm.forward_diffusion_latent(z0, t_b)
                eps_z_pred = model_ldm.score_net(z_t, t_b)
                mse = F.mse_loss(eps_z_pred, eps_z).item()
                total += mse
                print(f"  t={t_eval:.2f}: latent_eps_MSE={mse:.6f}")
        avg_loss_ldm = total / len(eval_times)
        save_checkpoint(model_ldm, full_tag,
                        {'recon_mse': recon_mse_ldm, 'avg_loss': avg_loss_ldm})

    # =================================================================
    # MODEL 3: CharRNN (second controlled baseline)
    # =================================================================
    print("\n" + "=" * 70)
    print(f"MODEL 3: CharRNN autoregressive baseline on {cfg['label']}")
    print("=" * 70)
    model_rnn = CharRNN(vocab_size=V, embed_dim=charrnn_embed,
                        hidden_dim=charrnn_hidden, n_layers=charrnn_layers).to(device)

    rnn_tag = f'{dataset_key}_v2_charrnn'
    ckpt = load_checkpoint(model_rnn, rnn_tag, device=device)
    if ckpt is None:
        print("\n--- CharRNN training (teacher-forced CE) ---")
        train_charrnn(model_rnn, dataset, n_steps=charrnn_steps, lr=1e-3,
                      batch_size=batch_size, device=device)
        save_checkpoint(model_rnn, rnn_tag)

    # =================================================================
    # Sample + evaluate with BOTH raw and repaired modes
    # =================================================================
    print(f"\n--- Generating {n_eval} samples per method ---")

    def big_sample_diff(model, n):
        out = []
        bs = 512
        with torch.no_grad():
            for i in range(0, n, bs):
                k = min(bs, n - i)
                x = model.sample(k, n_steps=200, device=device).cpu()
                out.append(x.view(-1, d))
                if (i // bs + 1) % 5 == 0:
                    print(f"    {i + bs}/{n}")
        return torch.cat(out)[:n]

    print("  ULD sampling...")
    uld_gen = big_sample_diff(model_ts, n_eval)
    print("  LDM sampling...")
    ldm_gen = big_sample_diff(model_ldm, n_eval)
    print("  CharRNN sampling...")
    rnn_gen = sample_charrnn(model_rnn, dataset, n_eval, device=device)

    def eval_both(name, gen):
        raw    = dataset.evaluate_samples(gen, repair=False)
        repair = dataset.evaluate_samples(gen, repair=True)
        return {'raw': raw, 'repair': repair}

    metrics = {
        'uld':     eval_both('ULD', uld_gen),
        'ldm':     eval_both('LDM', ldm_gen),
        'charrnn': eval_both('CharRNN', rnn_gen),
    }

    # Print example decoded SMILES for each
    for name, gen in [('ULD', uld_gen), ('LDM', ldm_gen), ('CharRNN', rnn_gen)]:
        smis = dataset.decode_onehot(gen[:15])
        print(f"\n  {name} first 15 generated SMILES:")
        for s in smis:
            print(f"    {s!r}")

    # =================================================================
    # Save summary
    # =================================================================
    summary = {
        'dataset': cfg['label'],
        'config': {
            'd': d, 'vocab_size': V, 'max_len': cfg['max_len'],
            'latent_dim': latent_dim, 'hidden_dim_enc': hidden_dim_enc,
            'hidden_dim_score': hidden_dim_score, 'n_score_layers': n_score_layers,
            't0': t0, 'min_snr_gamma': min_snr_gamma,
            'ae_steps': ae_steps, 'diff_steps': diff_steps,
            'charrnn_steps': charrnn_steps, 'charrnn_hidden': charrnn_hidden,
            'batch_size': batch_size, 'n_eval': n_eval,
        },
        'real_baseline_raw':    real_eval_raw,
        'real_baseline_repair': real_eval_repair,
        'uld':     {'recon_mse': recon_mse_ts, 'avg_eps_loss': avg_loss_ts,
                    **metrics['uld']},
        'ldm':     {'recon_mse': recon_mse_ldm, 'avg_eps_loss': avg_loss_ldm,
                    **metrics['ldm']},
        'charrnn': metrics['charrnn'],
    }
    out_path = os.path.join(LOG_DIR, f'{dataset_key}_v2_results.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved results JSON: {out_path}")

    print("\n" + "=" * 70)
    print(f"{cfg['label']} SMILES v2 SUMMARY")
    print("=" * 70)
    print(f"  {'Method':<14} {'Validity':<20} {'Uniqueness':<14} {'Novelty':<10}")
    print(f"  {'':<14} {'raw / repair':<20}")
    print("  " + "-" * 72)
    for name, key in [('Real', '_'), ('ULD (ours)', 'uld'),
                      ('LDM-CNN', 'ldm'), ('CharRNN', 'charrnn')]:
        if key == '_':
            r = real_eval_raw; rp = real_eval_repair
        else:
            r  = metrics[key]['raw']
            rp = metrics[key]['repair']
        vstr = f"{r['validity']:.4f} / {rp['validity']:.4f}"
        ustr = f"{rp['uniqueness']:.4f}"
        nstr = f"{rp['novelty']:.4f}"
        print(f"  {name:<14} {vstr:<20} {ustr:<14} {nstr:<10}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(DATASET_CONFIGS.keys()),
                    required=True)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    run_experiment(dataset_key=args.dataset, seed=args.seed)
