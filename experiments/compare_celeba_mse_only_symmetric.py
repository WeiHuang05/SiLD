"""
CelebA true no-auxiliary-loss matrix.

Methods:
  sild:  deterministic denoising AE, MSE only
  clean: deterministic clean AE, MSE only
  vae:   stochastic VAE, MSE + method-defining KL only

No method uses LPIPS, MMD, GAN, or another auxiliary perceptual/prior loss.
All methods use the same latent diffusion interval and training budget.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from data.image_data import CelebADataset
from experiments.compare_celeba_mmd_sweep import (
    compute_fid,
    compute_latent_stats,
    evaluate_eps,
    plot_image_grid,
)
from models.latent_diffusion_cnn import LatentDiffusionCNN
from models.two_stage_latent import TwoStageManifoldDiffusion
from utils.experiment_variant import experiment_variant, output_dir


FIG_DIR = output_dir(__file__, "figures")
CKPT_DIR = output_dir(__file__, "checkpoints")
LOG_DIR = output_dir(__file__, "logs")


def save_checkpoint(model, path, **extra):
    torch.save({"model": model.state_dict(), **extra}, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(model, path, device):
    if not os.path.exists(path):
        return None
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    print(f"  Checkpoint loaded: {path}")
    return state


def train_deterministic_ae(
    model, dataset, n_steps, lr, batch_size, input_noise_std,
):
    device = next(model.parameters()).device
    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    print(
        f"  Deterministic MSE-only AE: steps={n_steps}, "
        f"input_noise_std={input_noise_std}"
    )
    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)
        x_in = x0 + input_noise_std * torch.randn_like(x0)
        _, reconstruction = model.project(x_in)
        loss = F.mse_loss(reconstruction, x0)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        if (step + 1) % 200 == 0:
            print(f"  AE [{step + 1}/{n_steps}] mse={loss.item():.6f}")

    with torch.no_grad():
        latents = []
        for _ in range(10):
            x0, _ = dataset.sample_clean(min(batch_size, 2000))
            latents.append(model.encoder(x0.to(device)))
        latents = torch.cat(latents)
        model.z_mean.copy_(latents.mean(0))
        model.z_std.copy_(latents.std(0).clamp(min=1e-6))


def train_vae_mse_kl(
    model, dataset, n_steps, lr, batch_size, kl_weight,
):
    device = next(model.parameters()).device
    params = list(model.vae_encoder.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    print(f"  VAE MSE+KL only: steps={n_steps}, KL={kl_weight}")
    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)
        mu, logvar = model.vae_encoder(x0)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        reconstruction = model.decoder(z)
        mse = F.mse_loss(reconstruction, x0)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = mse + kl_weight * kl
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        if (step + 1) % 200 == 0:
            print(
                f"  VAE [{step + 1}/{n_steps}] total={loss.item():.6f} "
                f"mse={mse.item():.6f} kl={kl.item():.4f}"
            )

    with torch.no_grad():
        latents = []
        for _ in range(10):
            x0, _ = dataset.sample_clean(min(batch_size, 2000))
            latents.append(
                model.vae_encoder.encode_deterministic(x0.to(device))
            )
        latents = torch.cat(latents)
        model.z_mean.copy_(latents.mean(0))
        model.z_std.copy_(latents.std(0).clamp(min=1e-6))


@torch.no_grad()
def reconstruction_mse(model, dataset, device, is_sild):
    x0, _ = dataset.sample_clean(2000)
    x0 = x0.to(device)
    if is_sild:
        _, reconstruction = model.project(x0)
    else:
        z = model.vae_encoder.encode_deterministic(x0)
        reconstruction = model.decoder(z)
    return F.mse_loss(reconstruction, x0).item()


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_shape = (3, 64, 64)
    d = int(np.prod(image_shape))
    is_sild = args.method in {"sild", "clean"}
    input_noise_std = 0.005 if args.method == "sild" else 0.0

    print("=" * 76)
    print(f"CelebA TRUE NO-AUX CONTROL — {args.method.upper()}")
    print(f"  device={device}, seed={args.seed}, variant={experiment_variant()}")
    print("  LPIPS=0, MMD=0, GAN=0")
    print(f"  Stage-2=[{args.stage2_t_min}, 0.999]")
    print("=" * 76)

    dataset = CelebADataset(img_size=64, max_samples=20000)
    common = dict(
        d=d,
        latent_dim=256,
        hidden_dim_enc=512,
        hidden_dim_score=1024,
        n_layers=8,
        image_shape=image_shape,
        base_ch=128,
        diffusion_t_min=args.stage2_t_min,
    )
    if is_sild:
        model = TwoStageManifoldDiffusion(t0=0.005, **common).to(device)
    else:
        model = LatentDiffusionCNN(
            kl_weight=args.kl_weight, **common
        ).to(device)

    tag = f"celeba_mse_only_{args.method}_seed{args.seed}"
    ae_path = os.path.join(CKPT_DIR, f"{tag}_ae.pt")
    full_path = os.path.join(CKPT_DIR, f"{tag}_full.pt")
    state = load_checkpoint(model, full_path, device)
    if state is None:
        ae_state = load_checkpoint(model, ae_path, device)
        if ae_state is None:
            if is_sild:
                train_deterministic_ae(
                    model,
                    dataset,
                    args.ae_steps,
                    args.ae_lr,
                    args.batch_size,
                    input_noise_std,
                )
            else:
                train_vae_mse_kl(
                    model,
                    dataset,
                    args.ae_steps,
                    args.ae_lr,
                    args.batch_size,
                    args.kl_weight,
                )
            recon = reconstruction_mse(model, dataset, device, is_sild)
            save_checkpoint(model, ae_path, recon_mse=recon)
        else:
            recon = ae_state["recon_mse"]

        if is_sild:
            model.freeze_stage1()
            model.train_stage2(
                dataset,
                n_steps=args.diff_steps,
                lr=args.diff_lr,
                batch_size=args.batch_size,
            )
        else:
            model.freeze_ae()
            model.train_diffusion(
                dataset,
                n_steps=args.diff_steps,
                lr=args.diff_lr,
                batch_size=args.batch_size,
            )
        eps = evaluate_eps(model, dataset, is_sild, device)
        save_checkpoint(model, full_path, recon_mse=recon, avg_loss=eps)
    else:
        recon = state["recon_mse"]
        eps = state["avg_loss"]
        model.freeze_stage1() if is_sild else model.freeze_ae()

    latent_stats = compute_latent_stats(model, dataset, is_sild, device)
    fid = compute_fid(model, dataset, args.n_fid, 64, 64, device)
    with torch.no_grad():
        samples = model.sample(
            64, n_steps=args.sample_steps, device=device, clamp=(0.0, 1.0)
        )
    plot_image_grid(
        samples.cpu(),
        nrow=8,
        title=f"CelebA MSE-only {args.method}",
        save_name=f"{tag}.png",
        img_size=64,
    )

    result = {
        "method": args.method,
        "seed": args.seed,
        "experiment_variant": experiment_variant(),
        "config": {
            **vars(args),
            "input_noise_std": input_noise_std,
            "lpips_weight": 0.0,
            "mmd_weight": 0.0,
            "gan_weight": 0.0,
            "stage2_time_range": [args.stage2_t_min, 0.999],
        },
        "results": {
            "recon_mse": recon,
            "eps_loss": eps,
            "fid": fid,
            "latent_stats": latent_stats,
        },
    }
    output_path = os.path.join(LOG_DIR, f"{tag}_results.json")
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"  Saved results: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["sild", "clean", "vae"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage2-t-min", type=float, default=0.001)
    parser.add_argument("--ae-steps", type=int, default=20000)
    parser.add_argument("--diff-steps", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ae-lr", type=float, default=1e-4)
    parser.add_argument("--diff-lr", type=float, default=3e-4)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--n-fid", type=int, default=2048)
    parser.add_argument("--sample-steps", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
