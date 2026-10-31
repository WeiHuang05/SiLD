"""
CelebA symmetric-regularization control: VAE-LDM + MMD.

This fills the missing cells in the existing CelebA MMD sweep.  The SiLD
models use

    MSE + 0.5 * LPIPS + alpha * MMD

for Stage 1.  Here the VAE-LDM models use the same reconstruction and MMD
terms, retaining only the method-defining KL term:

    MSE + 0.5 * LPIPS + 1e-4 * KL + alpha * MMD.

MMD is evaluated on the deterministic encoder mean after the same
per-coordinate minibatch standardization used by the SiLD sweep.
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
from models.latent_diffusion_cnn import LatentDiffusionCNN
from utils.experiment_variant import (
    experiment_variant,
    output_dir,
    paired_stage2_t_min,
)
from experiments.compare_celeba_mmd_sweep import (
    LPIPS_AVAILABLE,
    compute_fid,
    compute_latent_stats,
    evaluate_eps,
    mmd_loss,
    plot_image_grid,
)

if LPIPS_AVAILABLE:
    import lpips


FIG_DIR = output_dir(__file__, "figures")
CKPT_DIR = output_dir(__file__, "checkpoints")
LOG_DIR = output_dir(__file__, "logs")


def save_checkpoint(model, name, extra=None):
    path = os.path.join(CKPT_DIR, f"{name}.pt")
    state = {"model": model.state_dict()}
    if extra:
        state.update(extra)
    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(model, name, device):
    path = os.path.join(CKPT_DIR, f"{name}.pt")
    if not os.path.exists(path):
        return None
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    print(f"  Checkpoint loaded: {path}")
    return state


def train_vae_lpips_mmd(
    model,
    dataset,
    n_steps,
    lr,
    batch_size,
    lpips_weight,
    kl_weight,
    mmd_weight,
    device,
):
    img_size = int(np.sqrt(model.d // 3))

    loss_fn_lpips = None
    if LPIPS_AVAILABLE:
        loss_fn_lpips = lpips.LPIPS(net="vgg").to(device)
        loss_fn_lpips.eval()
        for parameter in loss_fn_lpips.parameters():
            parameter.requires_grad = False

    params = list(model.vae_encoder.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)

    print(
        f"  VAE symmetric MMD: {n_steps} steps, "
        f"LPIPS={lpips_weight}, KL={kl_weight}, MMD={mmd_weight}"
    )

    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)
        mu, logvar = model.vae_encoder(x0)
        std = torch.exp(0.5 * logvar)
        z_sample = mu + std * torch.randn_like(std)
        x_recon = model.decoder(z_sample)

        mse = F.mse_loss(x_recon, x0)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        perceptual = torch.tensor(0.0, device=device)
        if loss_fn_lpips is not None:
            x0_img = x0.view(-1, 3, img_size, img_size) * 2 - 1
            xr_img = x_recon.view(-1, 3, img_size, img_size) * 2 - 1
            perceptual = loss_fn_lpips(xr_img, x0_img).mean()

        # Match the SiLD MMD implementation exactly: deterministic code,
        # standardized per coordinate within the current minibatch.
        z_norm = (mu - mu.mean(dim=0)) / mu.std(dim=0).clamp(min=1e-6)
        mmd = mmd_loss(z_norm)
        loss = (
            mse
            + lpips_weight * perceptual
            + kl_weight * kl
            + mmd_weight * mmd
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 200 == 0:
            print(
                f"  AE [{step + 1}/{n_steps}] total={loss.item():.6f} "
                f"mse={mse.item():.6f} lpips={perceptual.item():.6f} "
                f"kl={kl.item():.4f} mmd={mmd.item():.6f}"
            )

    with torch.no_grad():
        latent_batches = []
        for _ in range(10):
            x0, _ = dataset.sample_clean(min(batch_size, 2000))
            latent_batches.append(
                model.vae_encoder.encode_deterministic(x0.to(device))
            )
        latents = torch.cat(latent_batches, dim=0)
        model.z_mean.copy_(latents.mean(dim=0))
        model.z_std.copy_(latents.std(dim=0).clamp(min=1e-6))


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_shape = (3, 64, 64)
    d = int(np.prod(image_shape))
    t0 = 0.005
    stage2_t_min = paired_stage2_t_min(t0)
    tag = f"mmd{args.mmd_weight:g}_seed{args.seed}"

    print("=" * 72)
    print("CelebA SYMMETRIC CONTROL — VAE-LDM + MMD")
    print(f"  variant={experiment_variant()}, device={device}, tag={tag}")
    print(f"  Stage-2 time range=[{stage2_t_min}, 0.999]")
    print(
        f"  AE={args.ae_steps}, diffusion={args.diff_steps}, "
        f"batch={args.batch_size}"
    )
    print("=" * 72)

    dataset = CelebADataset(img_size=64, max_samples=20000)
    model = LatentDiffusionCNN(
        d=d,
        latent_dim=256,
        hidden_dim_enc=512,
        hidden_dim_score=1024,
        n_layers=8,
        kl_weight=args.kl_weight,
        image_shape=image_shape,
        base_ch=128,
        diffusion_t_min=stage2_t_min,
    ).to(device)

    ae_name = f"celeba_ldm_symmetric_{tag}_ae"
    full_name = f"celeba_ldm_symmetric_{tag}_full"
    checkpoint = load_checkpoint(model, full_name, device)
    if checkpoint is None:
        ae_checkpoint = load_checkpoint(model, ae_name, device)
        if ae_checkpoint is None:
            train_vae_lpips_mmd(
                model,
                dataset,
                args.ae_steps,
                1e-4,
                args.batch_size,
                args.lpips_weight,
                args.kl_weight,
                args.mmd_weight,
                device,
            )
            with torch.no_grad():
                x0, _ = dataset.sample_clean(2000)
                x0 = x0.to(device)
                z = model.vae_encoder.encode_deterministic(x0)
                recon = F.mse_loss(model.decoder(z), x0).item()
            save_checkpoint(model, ae_name, {"recon_mse": recon})
        else:
            recon = ae_checkpoint["recon_mse"]

        model.freeze_ae()
        model.train_diffusion(
            dataset,
            n_steps=args.diff_steps,
            lr=3e-4,
            batch_size=args.batch_size,
        )
        eps = evaluate_eps(model, dataset, False, device)
        save_checkpoint(
            model,
            full_name,
            {"recon_mse": recon, "avg_loss": eps},
        )
    else:
        recon = checkpoint["recon_mse"]
        eps = checkpoint["avg_loss"]
        model.freeze_ae()

    latent_stats = compute_latent_stats(model, dataset, False, device)
    fid = compute_fid(model, dataset, args.n_fid, 64, 64, device)
    print(f"  Recon MSE: {recon:.6f}")
    print(f"  Eps loss: {eps:.6f}")
    print(f"  FID ({args.n_fid}): {fid:.4f}")

    with torch.no_grad():
        samples = model.sample(64, n_steps=500, device=device, clamp=(0.0, 1.0))
    plot_image_grid(
        samples.cpu(),
        nrow=8,
        title=f"LDM + MMD ({args.mmd_weight:g})",
        save_name=f"celeba_ldm_symmetric_{tag}.png",
        img_size=64,
    )

    result = {
        "experiment_variant": experiment_variant(),
        "method": "VAE-LDM + MMD",
        "seed": args.seed,
        "stage2_time_range": [stage2_t_min, 0.999],
        "config": {
            "mmd_weight": args.mmd_weight,
            "lpips_weight": args.lpips_weight,
            "kl_weight": args.kl_weight,
            "ae_steps": args.ae_steps,
            "diff_steps": args.diff_steps,
            "batch_size": args.batch_size,
            "n_fid": args.n_fid,
        },
        "results": {
            "recon_mse": recon,
            "eps_loss": eps,
            "fid": fid,
            "latent_stats": latent_stats,
        },
    }
    result_path = os.path.join(
        LOG_DIR, f"celeba_ldm_symmetric_{tag}_results.json"
    )
    with open(result_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"  Saved results: {result_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmd-weight", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ae-steps", type=int, default=20000)
    parser.add_argument("--diff-steps", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lpips-weight", type=float, default=0.5)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--n-fid", type=int, default=2048)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
