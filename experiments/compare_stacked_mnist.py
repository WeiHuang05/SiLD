"""
Experiment: Stacked MNIST — ULD vs VAE-LDM mode-coverage benchmark.

Stacked MNIST is the classic 1000-mode test: each image is a 3-channel
28x28 RGB picture made by stacking three random MNIST digits. The set of
modes is {0..9}^3 = 1000. Mode-collapsing methods (VAEs from KL pull,
GANs from nonconvex losses) typically cover far fewer than 1000 modes.

Comparison: ULD (Two-Stage Manifold Diffusion) vs LatentDiffusionCNN
(VAE-style baseline) at IDENTICAL CNN encoder/decoder/score-net capacity.
The only difference is how the autoencoder is trained.

Metrics:
  - n_modes_covered (out of 1000)
  - KL(empirical mode distribution || uniform 1/1000)
  - FID via torch-fidelity (10k generated vs 10k real)

A pre-trained MNIST classifier (LeNet-style, cached to checkpoints/) is
used to read off per-channel digit labels from generated samples.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.image_data import StackedMNISTDataset
from models.two_stage_latent import TwoStageManifoldDiffusion
from models.latent_diffusion_cnn import LatentDiffusionCNN
from utils.experiment_variant import (
    output_dir, experiment_variant, paired_stage2_t_min,
)

try:
    import torch_fidelity
    TORCH_FIDELITY_AVAILABLE = True
except ImportError:
    TORCH_FIDELITY_AVAILABLE = False
    print("Warning: torch-fidelity not available — FID will be skipped.")


FIG_DIR = output_dir(__file__, 'figures')
CKPT_DIR = output_dir(__file__, 'checkpoints')
LOG_DIR = output_dir(__file__, 'logs')


# ================================================================
# MNIST classifier (LeNet-style) — cached
# ================================================================

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def train_or_load_mnist_classifier(device='cpu', n_epochs=3, batch_size=256):
    """Train (or load cached) MNIST classifier. Returns the model in eval mode."""
    ckpt_path = os.path.join(CKPT_DIR, 'mnist_classifier.pt')
    model = MNISTClassifier().to(device)
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model'])
        print(f"  MNIST classifier loaded from {ckpt_path} "
              f"(test_acc={state.get('test_acc', float('nan')):.4f})")
        model.eval()
        return model

    print(f"  Training MNIST classifier for {n_epochs} epochs...")
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(
        root='/home/weihuang/data', train=True, download=True, transform=transform,
    )
    test_set = torchvision.datasets.MNIST(
        root='/home/weihuang/data', train=False, download=True, transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=512,
                                              shuffle=False, num_workers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(n_epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device); y = y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.shape[0]
        acc = correct / total
        print(f"    epoch {epoch+1}: test_acc={acc:.4f}")
    torch.save({'model': model.state_dict(), 'test_acc': acc}, ckpt_path)
    print(f"  MNIST classifier saved to {ckpt_path}")
    model.eval()
    return model


@torch.no_grad()
def classify_stacked(classifier, images_flat, device='cpu', batch=512):
    """
    Take stacked-MNIST images flattened to (n, 2352), split into 3
    grayscale 28x28 channels, run the classifier on each, and return
    a (n, 3) long tensor of predicted digits + the (n,) mode index.
    """
    n = images_flat.shape[0]
    preds = torch.empty(n, 3, dtype=torch.long)
    for start in range(0, n, batch):
        chunk = images_flat[start:start + batch].to(device)
        # (b, 3, 28, 28)
        ch = chunk.view(-1, 3, 28, 28).clamp(0, 1)
        # Classify each channel separately
        for k in range(3):
            ch_k = ch[:, k:k+1]  # (b, 1, 28, 28)
            preds[start:start + ch_k.shape[0], k] = \
                classifier(ch_k).argmax(dim=1).cpu()
    mode_idx = preds[:, 0] * 100 + preds[:, 1] * 10 + preds[:, 2]
    return preds, mode_idx


def evaluate_mode_coverage(mode_idx, n_modes=1000):
    """Return n_modes_covered, KL(emp || uniform 1/1000), entropy."""
    counts = torch.bincount(mode_idx, minlength=n_modes).float()
    n_covered = int((counts > 0).sum().item())
    p = counts / counts.sum().clamp(min=1)
    # KL(p || uniform) = sum p log(p * 1000)
    eps = 1e-12
    kl = (p * (p.clamp(min=eps).log() - np.log(1.0 / n_modes))).sum()
    # Entropy in nats
    H = -(p * p.clamp(min=eps).log()).sum()
    return {
        'n_modes_covered': n_covered,
        'kl_to_uniform': float(kl),
        'entropy_nats': float(H),
    }


# ================================================================
# Sample-grid + FID helpers
# ================================================================

def plot_image_grid(images, nrow=8, title='', save_name=None):
    n = min(len(images), nrow * nrow)
    ncol = nrow
    nrow_actual = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow_actual, ncol,
                             figsize=(ncol * 1.2, nrow_actual * 1.2))
    if nrow_actual == 1:
        axes = axes.reshape(1, -1)
    for i in range(nrow_actual):
        for j in range(ncol):
            idx = i * ncol + j
            ax = axes[i, j]
            if idx < n:
                img = images[idx].detach().cpu().clamp(0, 1)
                img = img.view(3, 28, 28).permute(1, 2, 0).numpy()
                ax.imshow(img)
            ax.axis('off')
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_name:
        path = os.path.join(FIG_DIR, save_name)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")
    else:
        plt.close()


def _save_uint8_pngs(images, out_dir, prefix):
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images):
        arr = (img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        arr = np.transpose(arr, (1, 2, 0))
        Image.fromarray(arr).save(os.path.join(out_dir, f'{prefix}_{i:05d}.png'))


def compute_fid_torch_fidelity(real_dir, gen_dir):
    if not TORCH_FIDELITY_AVAILABLE:
        return {'fid': float('nan'), 'is_mean': float('nan'), 'is_std': float('nan')}
    metrics = torch_fidelity.calculate_metrics(
        input1=gen_dir, input2=real_dir,
        cuda=torch.cuda.is_available(),
        isc=True, fid=True, kid=False, verbose=False,
        samples_find_deep=False,
    )
    return {
        'fid': float(metrics.get('frechet_inception_distance', float('nan'))),
        'is_mean': float(metrics.get('inception_score_mean', float('nan'))),
        'is_std': float(metrics.get('inception_score_std', float('nan'))),
    }


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


# ================================================================
# Main
# ================================================================

def run_experiment(seed=42, t0=0.05):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  Seed: {seed}  t0: {t0}")
    print(f"Experiment variant: {experiment_variant()}")
    stage2_t_min = paired_stage2_t_min(t0)
    print(f"Paired Stage-2 time range: [{stage2_t_min}, 0.999]")
    tag = f"t{int(round(t0 * 1000)):03d}_seed{seed}"

    image_shape = (3, 28, 28)
    d = 3 * 28 * 28  # 2352
    latent_dim = 64
    base_ch = 64
    hidden_dim_score = 512
    n_score_layers = 6
    # t0 is now an argument (default 0.05); 0.005 is the original setting.

    ae_steps = 20000
    diff_steps = 80000
    batch_size = 128
    n_eval = 25600       # samples used for mode-coverage
    n_fid = 10000        # samples used for FID

    print("=" * 70)
    print(f"STACKED MNIST — ULD vs LDM-CNN (d={d})")
    print(f"  latent_dim={latent_dim}, t0={t0}")
    print(f"  AE={ae_steps}, Diff={diff_steps}, batch={batch_size}")
    print(f"  Mode-coverage samples={n_eval}, FID samples={n_fid}")
    print("=" * 70)

    dataset = StackedMNISTDataset(n_pool=60000, seed=seed)
    print(f"  Pool unique modes: {len(dataset.mode_idx.unique())}/1000")

    # MNIST classifier
    classifier = train_or_load_mnist_classifier(device=device)

    # Sanity-check classifier on real stacked samples
    real_x, real_lbls = dataset.sample_clean(2000)
    pred, mode_pred = classify_stacked(classifier, real_x, device=device)
    real_mode = real_lbls[:, 0] * 100 + real_lbls[:, 1] * 10 + real_lbls[:, 2]
    cls_acc = (mode_pred == real_mode).float().mean().item()
    print(f"  Classifier mode accuracy on real stacked MNIST: {cls_acc:.4f}")

    # =================================================================
    # MODEL 1: ULD
    # =================================================================
    print("\n" + "=" * 70)
    print("MODEL 1: ULD (Two-Stage Manifold Diffusion, CNN AE)")
    print("=" * 70)
    model_ts = TwoStageManifoldDiffusion(
        d=d, latent_dim=latent_dim, hidden_dim_enc=512,
        hidden_dim_score=hidden_dim_score, n_layers=n_score_layers,
        t0=t0, image_shape=image_shape, base_ch=base_ch,
        diffusion_t_min=stage2_t_min,
    ).to(device)

    ckpt = load_checkpoint(model_ts, f'stacked_mnist_uld_full_{tag}', device=device)
    if ckpt is not None:
        print("  Resumed full from checkpoint.")
        model_ts.freeze_stage1()
        recon_mse_ts = ckpt.get('recon_mse', 0.0)
        avg_loss_ts = ckpt.get('avg_loss', 0.0)
    else:
        ckpt_s1 = load_checkpoint(model_ts, f'stacked_mnist_uld_stage1_{tag}', device=device)
        if ckpt_s1 is not None:
            print("  Resumed Stage 1.")
            model_ts.freeze_stage1()
            recon_mse_ts = ckpt_s1.get('recon_mse', 0.0)
        else:
            print("\n--- Stage 1 (denoising AE at small t0) ---")
            model_ts.train_stage1(dataset, n_steps=ae_steps, lr=1e-4,
                                  batch_size=batch_size)
            with torch.no_grad():
                x0_t, _ = dataset.sample_clean(2000)
                x0_t = x0_t.to(device)
                _, x_hat = model_ts.project(x0_t)
                recon_mse_ts = F.mse_loss(x_hat, x0_t).item()
                print(f"\n  Recon MSE: {recon_mse_ts:.6f}")
            model_ts.freeze_stage1()
            save_checkpoint(model_ts, f'stacked_mnist_uld_stage1_{tag}',
                            {'recon_mse': recon_mse_ts})

        print("\n--- Stage 2 (latent diffusion on [t0, T]) ---")
        model_ts.train_stage2(dataset, n_steps=diff_steps, lr=3e-4,
                              batch_size=batch_size)

        eval_times = sorted(
            set([max(stage2_t_min, 0.01), 0.1, 0.3, 0.5, 0.8])
        )
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
        save_checkpoint(model_ts, f'stacked_mnist_uld_full_{tag}',
                        {'recon_mse': recon_mse_ts, 'avg_loss': avg_loss_ts})

    # =================================================================
    # MODEL 2: LDM-CNN
    # =================================================================
    print("\n" + "=" * 70)
    print("MODEL 2: LDM-CNN (matched CNN AE + same score net)")
    print("=" * 70)
    model_ldm = LatentDiffusionCNN(
        d=d, latent_dim=latent_dim, hidden_dim_enc=512,
        hidden_dim_score=hidden_dim_score, n_layers=n_score_layers,
        kl_weight=1e-4, image_shape=image_shape, base_ch=base_ch,
        diffusion_t_min=stage2_t_min,
    ).to(device)

    ckpt = load_checkpoint(model_ldm, f'stacked_mnist_ldm_full_{tag}', device=device)
    if ckpt is not None:
        print("  Resumed full from checkpoint.")
        model_ldm.freeze_ae()
        recon_mse_ldm = ckpt.get('recon_mse', 0.0)
        avg_loss_ldm = ckpt.get('avg_loss', 0.0)
    else:
        ckpt_ae = load_checkpoint(model_ldm, f'stacked_mnist_ldm_ae_{tag}', device=device)
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
            save_checkpoint(model_ldm, f'stacked_mnist_ldm_ae_{tag}',
                            {'recon_mse': recon_mse_ldm})

        print("\n--- LDM diffusion training ---")
        model_ldm.train_diffusion(dataset, n_steps=diff_steps, lr=3e-4,
                                  batch_size=batch_size)

        eval_times = sorted(
            set([max(stage2_t_min, 0.01), 0.1, 0.3, 0.5, 0.8])
        )
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
        save_checkpoint(model_ldm, f'stacked_mnist_ldm_full_{tag}',
                        {'recon_mse': recon_mse_ldm, 'avg_loss': avg_loss_ldm})

    # =================================================================
    # Generate samples + sample grids
    # =================================================================
    print("\n--- Generating sample grids ---")
    with torch.no_grad():
        x_gen_uld_grid = model_ts.sample(64, n_steps=500, device=device,
                                         clamp=(0, 1)).cpu()
        plot_image_grid(x_gen_uld_grid, nrow=8, title=f'ULD ({tag})',
                        save_name=f'stacked_mnist_gen_uld_{tag}.png')
        x_gen_ldm_grid = model_ldm.sample(64, n_steps=500, device=device,
                                          clamp=(0, 1)).cpu()
        plot_image_grid(x_gen_ldm_grid, nrow=8, title=f'LDM-CNN ({tag})',
                        save_name=f'stacked_mnist_gen_ldm_{tag}.png')
        x_real_grid, _ = dataset.sample_clean(64)
        plot_image_grid(x_real_grid, nrow=8, title='Real',
                        save_name=f'stacked_mnist_real_{tag}.png')

    # =================================================================
    # Mode coverage (n_eval samples)
    # =================================================================
    print(f"\n--- Mode coverage ({n_eval} samples) ---")
    def big_sample(model, n):
        out = []
        bs = 256
        with torch.no_grad():
            for i in range(0, n, bs):
                k = min(bs, n - i)
                x = model.sample(k, n_steps=500, device=device,
                                 clamp=(0, 1)).cpu()
                out.append(x.view(-1, d))
                if (i // bs + 1) % 8 == 0:
                    print(f"    {i + bs}/{n}")
        return torch.cat(out)[:n]

    print("  ULD sampling for mode coverage...")
    uld_eval = big_sample(model_ts, n_eval)
    print("  LDM sampling for mode coverage...")
    ldm_eval = big_sample(model_ldm, n_eval)

    _, uld_modes = classify_stacked(classifier, uld_eval, device=device)
    _, ldm_modes = classify_stacked(classifier, ldm_eval, device=device)

    uld_cov = evaluate_mode_coverage(uld_modes)
    ldm_cov = evaluate_mode_coverage(ldm_modes)

    real_eval, real_lbls_eval = dataset.sample_clean(n_eval)
    real_modes = real_lbls_eval[:, 0] * 100 + real_lbls_eval[:, 1] * 10 + real_lbls_eval[:, 2]
    real_cov = evaluate_mode_coverage(real_modes)

    print(f"  Real:  n_modes={real_cov['n_modes_covered']}/1000  "
          f"KL_unif={real_cov['kl_to_uniform']:.4f}")
    print(f"  ULD :  n_modes={uld_cov['n_modes_covered']}/1000  "
          f"KL_unif={uld_cov['kl_to_uniform']:.4f}")
    print(f"  LDM :  n_modes={ldm_cov['n_modes_covered']}/1000  "
          f"KL_unif={ldm_cov['kl_to_uniform']:.4f}")

    # =================================================================
    # FID via torch-fidelity
    # =================================================================
    fid_results = {}
    if TORCH_FIDELITY_AVAILABLE:
        print(f"\n--- FID via torch-fidelity ({n_fid} samples) ---")
        scratch = os.path.join('/tmp', f'stacked_mnist_fid_{os.getpid()}')
        try:
            real_for_fid = real_eval[:n_fid].view(-1, 3, 28, 28)
            uld_for_fid = uld_eval[:n_fid].view(-1, 3, 28, 28)
            ldm_for_fid = ldm_eval[:n_fid].view(-1, 3, 28, 28)
            print("  Saving real PNGs...")
            _save_uint8_pngs(real_for_fid, os.path.join(scratch, 'real'), 'r')
            print("  Saving ULD PNGs...")
            _save_uint8_pngs(uld_for_fid, os.path.join(scratch, 'uld'), 'u')
            print("  Saving LDM PNGs...")
            _save_uint8_pngs(ldm_for_fid, os.path.join(scratch, 'ldm'), 'l')
            print("  Computing FID...")
            uld_fid = compute_fid_torch_fidelity(
                os.path.join(scratch, 'real'), os.path.join(scratch, 'uld'))
            ldm_fid = compute_fid_torch_fidelity(
                os.path.join(scratch, 'real'), os.path.join(scratch, 'ldm'))
            fid_results = {'uld': uld_fid, 'ldm': ldm_fid}
            print(f"  ULD FID: {uld_fid['fid']:.2f}  IS: "
                  f"{uld_fid['is_mean']:.2f}+/-{uld_fid['is_std']:.2f}")
            print(f"  LDM FID: {ldm_fid['fid']:.2f}  IS: "
                  f"{ldm_fid['is_mean']:.2f}+/-{ldm_fid['is_std']:.2f}")
        finally:
            if os.path.isdir(scratch):
                shutil.rmtree(scratch, ignore_errors=True)

    # =================================================================
    # Save summary
    # =================================================================
    summary = {
        'config': {
            'd': d, 'latent_dim': latent_dim, 't0': t0,
            'ae_steps': ae_steps, 'diff_steps': diff_steps,
            'batch_size': batch_size, 'n_eval': n_eval, 'n_fid': n_fid,
            'seed': seed, 'experiment_variant': experiment_variant(),
            'stage2_time_range': [stage2_t_min, 0.999],
        },
        'classifier_acc_on_real_stacked': cls_acc,
        'real': real_cov,
        'uld': {
            'recon_mse': recon_mse_ts,
            'avg_eps_loss': avg_loss_ts,
            **uld_cov,
            **fid_results.get('uld', {}),
        },
        'ldm': {
            'recon_mse': recon_mse_ldm,
            'avg_eps_loss': avg_loss_ldm,
            **ldm_cov,
            **fid_results.get('ldm', {}),
        },
    }
    out_path = os.path.join(LOG_DIR, f'stacked_mnist_results_{tag}.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved results JSON: {out_path}")

    print("\n" + "=" * 70)
    print("STACKED MNIST SUMMARY (1000-mode benchmark)")
    print("=" * 70)
    print(f"  {'Method':<14} {'#modes/1000':<14} {'KL_unif':<10} "
          f"{'recon MSE':<12} {'eps loss':<12} {'FID':<8}")
    print("  " + "-" * 80)
    print(f"  {'Real':<14} {real_cov['n_modes_covered']:<14} "
          f"{real_cov['kl_to_uniform']:<10.4f} "
          f"{0.0:<12.6f} {0.0:<12.6f} {0.0:<8.2f}")
    print(f"  {'ULD (ours)':<14} {uld_cov['n_modes_covered']:<14} "
          f"{uld_cov['kl_to_uniform']:<10.4f} "
          f"{recon_mse_ts:<12.6f} {avg_loss_ts:<12.6f} "
          f"{fid_results.get('uld', {}).get('fid', float('nan')):<8.2f}")
    print(f"  {'LDM-CNN':<14} {ldm_cov['n_modes_covered']:<14} "
          f"{ldm_cov['kl_to_uniform']:<10.4f} "
          f"{recon_mse_ldm:<12.6f} {avg_loss_ldm:<12.6f} "
          f"{fid_results.get('ldm', {}).get('fid', float('nan')):<8.2f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--t0', type=float, default=0.05)
    args = parser.parse_args()
    run_experiment(seed=args.seed, t0=args.t0)
