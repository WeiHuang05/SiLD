"""
Transformer autoregressive baseline on QM9 (SMILES and SELFIES).
Serves as a modern substitute for cMolGPT / SAFE-GPT in the SOTA table.

Small transformer (6 layers, 256 dim), trained with teacher-forced
next-token cross-entropy. Sampled autoregressively; evaluated with the
same unified evaluator used for ULD/LDM/CharRNN.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse, json, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
import math

from data.molecule_data import SmilesDataset, SelfiesDataset

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class TransformerAR(nn.Module):
    """Small decoder-only transformer (like GPT) for molecular strings."""
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6,
                 max_len=32, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        n = sum(p.numel() for p in self.parameters())
        print(f"    TransformerAR: vocab={vocab_size}, d_model={d_model}, "
              f"layers={n_layers}, params={n:,}")

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.token_embed(x) + self.pos_embed(pos)
        # Causal mask
        causal = torch.triu(
            torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
        h = self.transformer(h, mask=causal, is_causal=True)
        h = self.ln_f(h)
        return self.head(h)


def train(model, dataset, n_steps, lr, batch_size, device):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)
    V = dataset.vocab_size
    L = dataset.max_len
    for step in range(n_steps):
        x_flat, _ = dataset.sample_clean(batch_size)
        tokens = x_flat.view(-1, L, V).argmax(-1).to(device)  # (B, L)
        inp = tokens[:, :-1]
        tgt = tokens[:, 1:]
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 500 == 0:
            print(f"  Trans [{step+1}/{n_steps}] xent={loss.item():.4f}")


@torch.no_grad()
def sample(model, dataset, n, device, temperature=1.0):
    model.eval()
    L = dataset.max_len
    V = dataset.vocab_size
    # Marginal of first token
    first = dataset.data_onehot.argmax(-1)[:, 0]
    first_hist = torch.bincount(first, minlength=V).float()
    first_hist[0] = 0
    first_dist = first_hist / first_hist.sum().clamp(min=1)

    samples = torch.zeros(n, L, dtype=torch.long)
    bs = 256
    for start in range(0, n, bs):
        end = min(start + bs, n)
        b = end - start
        seqs = torch.zeros(b, L, dtype=torch.long, device=device)
        seqs[:, 0] = torch.multinomial(first_dist, b, replacement=True).to(device)
        for t in range(1, L):
            logits = model(seqs[:, :t])  # (b, t, V)
            logits = logits[:, -1] / max(temperature, 1e-4)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).squeeze(-1)
            seqs[:, t] = nxt
        samples[start:end] = seqs.cpu()

    one_hot = torch.zeros(n, L, V, dtype=torch.float32)
    one_hot.scatter_(2, samples.unsqueeze(-1), 1.0)
    return one_hot.view(n, -1)


DATASET_CONFIGS = {
    'qm9':  dict(csv='/home/weihuang/data/molecules/qm9.csv',
                 smiles_col='smiles', label='QM9',
                 selfies_max_len=20, smiles_max_len=32),
    'hiv':  dict(csv='/home/weihuang/data/molecules/HIV.csv',
                 smiles_col='smiles', label='HIV',
                 selfies_max_len=128, smiles_max_len=96),
    'muv':  dict(csv='/home/weihuang/data/molecules/muv.csv',
                 smiles_col='smiles', label='MUV',
                 selfies_max_len=72, smiles_max_len=64),
    'pcba': dict(csv='/home/weihuang/data/molecules/pcba.csv',
                 smiles_col='smiles', label='PCBA',
                 selfies_max_len=80, smiles_max_len=96),
}


def run(dataset_key='qm9', representation='selfies', n_steps=40000,
        batch_size=256, n_eval=10000, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = DATASET_CONFIGS[dataset_key]
    print(f'Device: {device}, dataset={cfg["label"]}, representation={representation}')

    if representation == 'selfies':
        dataset = SelfiesDataset(
            csv_path=cfg['csv'],
            smiles_col=cfg['smiles_col'], max_len=cfg['selfies_max_len'],
            seed=seed, name=cfg['label'])
    else:
        dataset = SmilesDataset(
            csv_path=cfg['csv'],
            smiles_col=cfg['smiles_col'], max_len=cfg['smiles_max_len'],
            seed=seed, name=cfg['label'])

    print(f'Dataset loaded: vocab={dataset.vocab_size}, max_len={dataset.max_len}')

    model = TransformerAR(
        vocab_size=dataset.vocab_size, d_model=256, n_heads=8,
        n_layers=6, max_len=dataset.max_len, dropout=0.1,
    ).to(device)

    ckpt_tag = f'{dataset_key}_{representation}_transformer'
    ckpt_path = os.path.join(CKPT_DIR, f'{ckpt_tag}.pt')
    if os.path.exists(ckpt_path):
        print(f'Loading {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, map_location=device)['model'])
    else:
        print(f'Training {n_steps} steps...')
        train(model, dataset, n_steps, lr=3e-4, batch_size=batch_size, device=device)
        torch.save({'model': model.state_dict()}, ckpt_path)

    print(f'Sampling {n_eval} molecules...')
    samples = sample(model, dataset, n_eval, device)

    print('Evaluating...')
    eval_kwargs = dict(compute_diversity=True, compute_qed=True, compute_fcd=True,
                       fcd_reference_max=10000)
    if representation == 'smiles':
        eval_kwargs['decode_mode'] = 'rdkit_longest' if dataset_key == 'qm9' else 'constrained'
    m = dataset.evaluate_samples(samples, **eval_kwargs)
    print(f'  Transformer AR ({cfg["label"]}, {representation}): '
          f'valid={m["validity"]:.4f}  uniq={m["uniqueness"]:.4f}  '
          f'nov={m["novelty"]:.4f}  '
          f'intdiv={m.get("internal_diversity",0):.4f}  '
          f'QED={m.get("qed_mean",0):.4f}  FCD={m.get("fcd",0):.3f}')

    out = os.path.join(LOG_DIR, f'{dataset_key}_{representation}_transformer.json')
    with open(out, 'w') as f:
        json.dump(m, f, indent=2)
    print(f'Saved: {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=list(DATASET_CONFIGS.keys()), default='qm9')
    ap.add_argument('--representation', choices=['smiles', 'selfies'], default='selfies')
    ap.add_argument('--n_steps', type=int, default=40000)
    ap.add_argument('--n_eval', type=int, default=10000)
    args = ap.parse_args()
    run(dataset_key=args.dataset, representation=args.representation,
        n_steps=args.n_steps, n_eval=args.n_eval)
