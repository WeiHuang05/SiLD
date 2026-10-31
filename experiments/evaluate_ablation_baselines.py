"""
Ablation baselines: VAE-only (no diffusion) + Latent GAN.

These answer two key reviewer questions:
  1. "Why not just sample from the VAE latent without diffusion?"
     → VAE-only baseline shows it produces garbage.
  2. "Is ULD's advantage from diffusion specifically, or any generative
     model in the latent space?"
     → Latent GAN shows diffusion is better than GAN in the same latent.

Both baselines reuse the EXISTING v2 checkpoints — no retraining of the
encoder/decoder. We only train a small GAN generator/discriminator on
the already-frozen latent space.

Usage: python experiments/evaluate_ablation_baselines.py --dataset {qm9,muv,hiv,pcba}
       [--representation {smiles,selfies}]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse, json, torch, torch.nn as nn, torch.nn.functional as F, numpy as np

from data.molecule_data import SmilesDataset, SelfiesDataset
from models.two_stage_latent import TwoStageManifoldDiffusion
from models.latent_diffusion_cnn import LatentDiffusionCNN
from experiments.compare_smiles_v2 import DATASET_CONFIGS

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
LOG_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
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


# =====================================================================
# Latent GAN
# =====================================================================

class LatentGenerator(nn.Module):
    """Simple MLP generator: noise z_rand → z_fake in latent space."""
    def __init__(self, noise_dim, latent_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_noise):
        return self.net(z_noise)


class LatentDiscriminator(nn.Module):
    """Simple MLP discriminator: z → real/fake scalar."""
    def __init__(self, latent_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(z)


def train_latent_gan(generator, discriminator, encoder, dataset,
                     n_steps, batch_size, noise_dim, z_mean, z_std,
                     device, is_vae=False, lr=1e-4):
    """
    Train a GAN in the frozen latent space of the encoder.
    Uses WGAN-GP for stability.
    """
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    for step in range(n_steps):
        # --- Discriminator step ---
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)
        with torch.no_grad():
            if is_vae:
                z_real = encoder.encode_deterministic(x0)
            else:
                z_real = encoder(x0)
            z_real = (z_real - z_mean) / z_std

        z_noise = torch.randn(batch_size, noise_dim, device=device)
        z_fake = generator(z_noise).detach()

        d_real = discriminator(z_real).mean()
        d_fake = discriminator(z_fake).mean()

        # Gradient penalty (WGAN-GP)
        alpha = torch.rand(batch_size, 1, device=device)
        z_interp = (alpha * z_real + (1 - alpha) * z_fake).requires_grad_(True)
        d_interp = discriminator(z_interp)
        grad = torch.autograd.grad(
            outputs=d_interp, inputs=z_interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True, retain_graph=True,
        )[0]
        gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()

        d_loss = d_fake - d_real + 10.0 * gp
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()

        # --- Generator step (every 5 disc steps) ---
        if (step + 1) % 5 == 0:
            z_noise = torch.randn(batch_size, noise_dim, device=device)
            z_fake = generator(z_noise)
            g_loss = -discriminator(z_fake).mean()
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()

        if (step + 1) % 2000 == 0:
            print(f'  GAN [{step+1}/{n_steps}] d_loss={d_loss.item():.4f} '
                  f'd_real={d_real.item():.4f} d_fake={d_fake.item():.4f}')


@torch.no_grad()
def sample_latent_gan(generator, decoder, z_mean, z_std, n, noise_dim, device):
    """Sample from the GAN generator, denormalize, decode."""
    z_noise = torch.randn(n, noise_dim, device=device)
    z_fake = generator(z_noise)
    z_raw = z_fake * z_std + z_mean
    return decoder(z_raw).cpu()


@torch.no_grad()
def sample_vae_only(decoder, z_mean, z_std, latent_dim, n, device):
    """Sample z ~ N(0,I) in normalized space, denormalize, decode.
    No diffusion at all — just the VAE prior."""
    z = torch.randn(n, latent_dim, device=device)
    z_raw = z * z_std + z_mean
    return decoder(z_raw).cpu()


# =====================================================================
# Main
# =====================================================================

def run(dataset_key, representation='smiles', n_eval=10000,
        gan_steps=20000, noise_dim=64, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # --- Load dataset ---
    if representation == 'selfies':
        cfg = SELFIES_CONFIGS[dataset_key]
        dataset = SelfiesDataset(
            csv_path=cfg['csv'], smiles_col=cfg['smiles_col'],
            max_len=cfg['max_len'], name=cfg['label'], seed=seed,
        )
        latent_dim = cfg['latent_dim']
        hidden_dim = cfg['hidden_dim']
        prefix = f'{dataset_key}_selfies'
    else:
        cfg = DATASET_CONFIGS[dataset_key]
        dataset = SmilesDataset(
            csv_path=cfg['csv'], smiles_col=cfg['smiles_col'],
            max_len=cfg['max_len'], name=cfg['label'], seed=seed,
        )
        latent_dim = 128
        hidden_dim = 1024
        prefix = f'{dataset_key}_v2'

    d = dataset.d
    print(f'Dataset: {cfg["label"]} ({representation}), d={d}, latent={latent_dim}')

    # --- Load existing ULD encoder/decoder from v2 checkpoints ---
    model_uld = TwoStageManifoldDiffusion(
        d=d, latent_dim=latent_dim,
        hidden_dim_enc=hidden_dim, hidden_dim_score=hidden_dim,
        n_layers=8, t0=0.05, image_shape=None,
    ).to(device)
    uld_path = os.path.join(CKPT_DIR, f'{prefix}_uld_full.pt')
    if not os.path.exists(uld_path):
        print(f'ERROR: {uld_path} not found. Run the v2 experiment first.')
        return
    state = torch.load(uld_path, map_location=device)
    model_uld.load_state_dict(state['model'])
    model_uld.freeze_stage1()
    model_uld.eval()
    print(f'  Loaded ULD from {uld_path}')

    # --- Load existing LDM encoder/decoder ---
    model_ldm = LatentDiffusionCNN(
        d=d, latent_dim=latent_dim,
        hidden_dim_enc=hidden_dim, hidden_dim_score=hidden_dim,
        n_layers=8, kl_weight=1e-4, image_shape=None,
    ).to(device)
    ldm_path = os.path.join(CKPT_DIR, f'{prefix}_ldm_full.pt')
    if not os.path.exists(ldm_path):
        print(f'ERROR: {ldm_path} not found.')
        return
    state = torch.load(ldm_path, map_location=device)
    model_ldm.load_state_dict(state['model'])
    model_ldm.freeze_ae()
    model_ldm.eval()
    print(f'  Loaded LDM from {ldm_path}')

    # Latent stats
    z_mean_uld = model_uld.z_mean.to(device)
    z_std_uld = model_uld.z_std.to(device)
    z_mean_ldm = model_ldm.z_mean.to(device)
    z_std_ldm = model_ldm.z_std.to(device)

    results = {'dataset': cfg['label'], 'representation': representation,
               'n_eval': n_eval, 'methods': {}}

    eval_kwargs = {}
    if representation == 'smiles':
        eval_kwargs = dict(decode_mode='rdkit_longest') if dataset_key == 'qm9' \
                      else dict(decode_mode='constrained')
    eval_kwargs.update(compute_diversity=True, compute_qed=True, compute_fcd=True)

    # --- 1. VAE-only (no diffusion): sample z ~ N(0,I), decode ---
    print(f'\n=== VAE-only (ULD encoder, no diffusion) ===')
    gen_vae_uld = sample_vae_only(
        model_uld.decoder if hasattr(model_uld, 'decoder') else model_uld.decode,
        z_mean_uld, z_std_uld, latent_dim, n_eval, device)
    m = dataset.evaluate_samples(gen_vae_uld, **eval_kwargs)
    results['methods']['VAE-only (ULD dec)'] = m
    print(f'  validity={m["validity"]:.4f}  #novel={m.get("n_novel",0)}  '
          f'FCD={m.get("fcd","?")}')

    print(f'\n=== VAE-only (LDM encoder, no diffusion) ===')
    gen_vae_ldm = sample_vae_only(
        model_ldm.decoder, z_mean_ldm, z_std_ldm, latent_dim, n_eval, device)
    m = dataset.evaluate_samples(gen_vae_ldm, **eval_kwargs)
    results['methods']['VAE-only (LDM dec)'] = m
    print(f'  validity={m["validity"]:.4f}  #novel={m.get("n_novel",0)}  '
          f'FCD={m.get("fcd","?")}')

    # --- 2. Latent GAN (ULD encoder) ---
    print(f'\n=== Latent GAN (ULD latent space) ===')
    gen_uld = LatentGenerator(noise_dim, latent_dim).to(device)
    disc_uld = LatentDiscriminator(latent_dim).to(device)
    train_latent_gan(gen_uld, disc_uld, model_uld.encoder, dataset,
                     n_steps=gan_steps, batch_size=256, noise_dim=noise_dim,
                     z_mean=z_mean_uld, z_std=z_std_uld, device=device,
                     is_vae=False)
    gen_gan_uld = sample_latent_gan(
        gen_uld, model_uld.decoder if hasattr(model_uld, 'decoder')
        else lambda z: model_uld.decode(z),
        z_mean_uld, z_std_uld, n_eval, noise_dim, device)
    m = dataset.evaluate_samples(gen_gan_uld, **eval_kwargs)
    results['methods']['Latent GAN (ULD)'] = m
    print(f'  validity={m["validity"]:.4f}  #novel={m.get("n_novel",0)}  '
          f'FCD={m.get("fcd","?")}')

    # --- 3. Latent GAN (LDM encoder) ---
    print(f'\n=== Latent GAN (LDM latent space) ===')
    gen_ldm_g = LatentGenerator(noise_dim, latent_dim).to(device)
    disc_ldm_g = LatentDiscriminator(latent_dim).to(device)
    train_latent_gan(gen_ldm_g, disc_ldm_g, model_ldm.vae_encoder, dataset,
                     n_steps=gan_steps, batch_size=256, noise_dim=noise_dim,
                     z_mean=z_mean_ldm, z_std=z_std_ldm, device=device,
                     is_vae=True)
    gen_gan_ldm = sample_latent_gan(
        gen_ldm_g, model_ldm.decoder,
        z_mean_ldm, z_std_ldm, n_eval, noise_dim, device)
    m = dataset.evaluate_samples(gen_gan_ldm, **eval_kwargs)
    results['methods']['Latent GAN (LDM)'] = m
    print(f'  validity={m["validity"]:.4f}  #novel={m.get("n_novel",0)}  '
          f'FCD={m.get("fcd","?")}')

    # --- For comparison: reload the diffusion results ---
    for label, tag in [('ULD + Diffusion', f'{prefix}_uld'),
                       ('LDM + Diffusion', f'{prefix}_ldm')]:
        print(f'\n=== {label} (from existing v2 results) ===')
        # Re-sample from the full model
        model = model_uld if 'ULD' in label else model_ldm
        out = []
        with torch.no_grad():
            for i in range(0, n_eval, 512):
                k = min(512, n_eval - i)
                x = model.sample(k, n_steps=200, device=device).cpu()
                out.append(x.view(-1, d))
        gen = torch.cat(out)[:n_eval]
        m = dataset.evaluate_samples(gen, **eval_kwargs)
        results['methods'][label] = m
        print(f'  validity={m["validity"]:.4f}  #novel={m.get("n_novel",0)}  '
              f'FCD={m.get("fcd","?")}')

    # --- Save ---
    suffix = f'_{representation}' if representation != 'smiles' else ''
    out_path = os.path.join(LOG_DIR, f'{dataset_key}_ablation_baselines{suffix}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_path}')

    # Summary
    print(f'\n{"="*80}')
    print(f'{cfg["label"]} Ablation Baselines ({representation})')
    print('='*80)
    print(f'  {"Method":<25} {"Valid":<8} {"#Novel":<8} {"FCD":<8} {"IntDiv":<8} {"QED":<8}')
    print('  ' + '-'*70)
    for name, m in results['methods'].items():
        fcd = m.get('fcd', None)
        fcd_s = f'{fcd:.2f}' if isinstance(fcd, (int,float)) and fcd is not None else '—'
        idiv = m.get('internal_diversity', None)
        idiv_s = f'{idiv:.3f}' if isinstance(idiv, (int,float)) and idiv is not None else '—'
        qed = m.get('qed_mean', None)
        qed_s = f'{qed:.3f}' if isinstance(qed, (int,float)) and qed is not None else '—'
        print(f'  {name:<25} {m["validity"]:<8.4f} '
              f'{m.get("n_novel",0):<8} {fcd_s:<8} {idiv_s:<8} {qed_s:<8}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['qm9', 'muv', 'hiv', 'pcba'],
                    required=True)
    ap.add_argument('--representation', choices=['smiles', 'selfies'],
                    default='smiles')
    ap.add_argument('--n_eval', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    run(args.dataset, args.representation, args.n_eval, seed=args.seed)
