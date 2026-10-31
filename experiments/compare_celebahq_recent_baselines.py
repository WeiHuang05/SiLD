"""
Recent diffusability baselines on CelebA-HQ with a 32x32x8 spatial latent.

freqwarm:
  Freeze the completed VAE and warm up latent diffusion on low-pass RGB
  inputs before switching to full-frequency inputs.

scaleeq:
  Retrain the VAE with Skorokhodov et al.'s decoder scale-equivariance
  regularizer. Spatial latents and RGB targets are downsampled by the same
  randomly chosen factor, then the decoder reconstructs at that scale.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from data.image_data import CelebAHQDataset
from experiments.compare_celebahq_large_latent import (
    EMA,
    FID_AVAILABLE,
    LDMLargeLatent,
    LPIPS_AVAILABLE,
    PatchDiscriminator,
    FrechetInceptionDistance,
    hinge_d,
    hinge_g,
    plot_grid,
)
from utils.experiment_variant import experiment_variant, output_dir
from utils.frequency_filter import apply_frequency_filter

if LPIPS_AVAILABLE:
    import lpips


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = output_dir(__file__, "figures")
CKPT_DIR = output_dir(__file__, "checkpoints")
LOG_DIR = output_dir(__file__, "logs")


def save_checkpoint(model, path, **extra):
    torch.save({"model": model.state_dict(), **extra}, path)
    print(f"  Saved: {path}")


def load_checkpoint(model, path, device):
    if not os.path.exists(path):
        return None
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    print(f"  Loaded: {path}")
    return state


def compute_z_stats(model, dataset, device, batch_size):
    with torch.no_grad():
        latents = []
        for _ in range(20):
            x0, _ = dataset.sample_clean(min(batch_size, 500))
            latents.append(
                model.vae_encoder.encode_deterministic(x0.to(device))
            )
        latents = torch.cat(latents)
        model.z_mean.copy_(
            latents.mean(dim=(0, 2, 3), keepdim=True).squeeze(0)
        )
        model.z_std.copy_(
            latents.std(dim=(0, 2, 3), keepdim=True)
            .squeeze(0)
            .clamp(min=1e-6)
        )


def train_scale_equivariant_vae(
    model,
    dataset,
    *,
    n_steps,
    batch_size,
    scale_weight,
    scale_factors,
    device,
):
    perceptual_fn = None
    if LPIPS_AVAILABLE:
        perceptual_fn = lpips.LPIPS(net="vgg").to(device).eval()
        for parameter in perceptual_fn.parameters():
            parameter.requires_grad = False

    discriminator = PatchDiscriminator(3, 64, 3).to(device)
    gen_params = list(model.vae_encoder.parameters()) + list(
        model.decoder.parameters()
    )
    opt_g = torch.optim.Adam(gen_params, lr=1e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(
        discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, n_steps)
    ema = EMA(model, 0.9999)
    disc_start = min(1000, n_steps // 10)

    print(
        f"  Scale-equivariant VAE: {n_steps} steps, "
        f"weight={scale_weight}, scales={scale_factors}"
    )
    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)
        x0_img = x0.view(-1, 3, 64, 64)
        mu, logvar = model.vae_encoder(x0)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        reconstruction = model.decoder(z)
        recon_img = reconstruction.view(-1, 3, 64, 64)

        mse = F.mse_loss(reconstruction, x0)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        perceptual = (
            perceptual_fn(recon_img * 2 - 1, x0_img * 2 - 1).mean()
            if perceptual_fn is not None
            else torch.zeros((), device=device)
        )
        adversarial = (
            hinge_g(discriminator(recon_img))
            if step >= disc_start
            else torch.zeros((), device=device)
        )

        scale_factor = float(np.random.choice(scale_factors))
        reduction = int(round(1.0 / scale_factor))
        z_down = F.avg_pool2d(z, reduction, reduction)
        target_down = F.avg_pool2d(x0_img, reduction, reduction)
        # The convolutional decoder is fully spatial. Calling its network
        # directly allows the same decoder weights to operate at 1/2 or 1/4
        # spatial resolution, exactly as in the official regularizer.
        pred_down = model.decoder.net(z_down)
        scale_mse = F.mse_loss(pred_down, target_down)
        # Use the official ``ignore_lpips`` branch for the scale auxiliary
        # term. This avoids changing the meaning of LPIPS across 64/32/16px;
        # the full-resolution base objective still retains LPIPS.
        scale_perceptual = torch.zeros((), device=device)
        scale_loss = scale_mse + scale_perceptual

        loss = (
            mse
            + perceptual
            + model.kl_weight * kl
            + 0.1 * adversarial
            + scale_weight * scale_loss
        )
        opt_g.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
        opt_g.step()
        scheduler.step()

        if step >= disc_start:
            with torch.no_grad():
                mu_det, _ = model.vae_encoder(x0)
                fake = model.decoder(mu_det).view(-1, 3, 64, 64)
            d_loss = hinge_d(discriminator(x0_img), discriminator(fake))
            opt_d.zero_grad()
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            opt_d.step()
        else:
            d_loss = torch.zeros(())
        ema.update(model)

        if (step + 1) % 500 == 0:
            print(
                f"  AE [{step + 1}/{n_steps}] total={loss.item():.6f} "
                f"mse={mse.item():.6f} scale={scale_loss.item():.6f} "
                f"factor={scale_factor:g} d={d_loss.item():.6f}"
            )

    ema.apply(model)
    compute_z_stats(model, dataset, device, batch_size)


def train_diffusion(
    model,
    dataset,
    *,
    n_steps,
    batch_size,
    device,
    freqwarm=False,
    warmup_steps=0,
    cutoff=0.2,
):
    ema = EMA(model, 0.9999)
    optimizer = torch.optim.Adam(model.score_net.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)
        warm = freqwarm and step < warmup_steps
        with torch.no_grad():
            if warm:
                x_enc = apply_frequency_filter(
                    x0.view(-1, 3, 64, 64),
                    keep="low",
                    cutoff=cutoff,
                    filter_type="ideal",
                ).view(x0.shape[0], -1)
            else:
                x_enc = x0
            z0 = (
                model.vae_encoder.encode_deterministic(x_enc) - model.z_mean
            ) / model.z_std.clamp(min=1e-6)

        t = torch.rand(z0.shape[0], device=device) * 0.998 + 0.001
        alpha_bar, sigma = model.noise_schedule(t)
        eps = torch.randn_like(z0)
        z_t = (
            torch.sqrt(alpha_bar).view(-1, 1, 1, 1) * z0
            + sigma.view(-1, 1, 1, 1) * eps
        )
        loss = F.mse_loss(model.score_net(z_t, t), eps)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.score_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        ema.update(model)

        if step + 1 == warmup_steps:
            print("  FreqWarm: switching to full-frequency inputs.")
        if (step + 1) % 500 == 0:
            phase = "warm" if warm else "full"
            print(
                f"  Diff [{step + 1}/{n_steps}] phase={phase} "
                f"eps={loss.item():.6f}"
            )
    ema.apply(model)


@torch.no_grad()
def evaluate_eps(model, dataset, device):
    values = {}
    for time in [0.01, 0.1, 0.3, 0.5, 0.8]:
        x0, _ = dataset.sample_clean(2000)
        z0 = (
            model.vae_encoder.encode_deterministic(x0.to(device)) - model.z_mean
        ) / model.z_std.clamp(min=1e-6)
        t = torch.full((z0.shape[0],), time, device=device)
        alpha_bar, sigma = model.noise_schedule(t)
        eps = torch.randn_like(z0)
        z_t = (
            torch.sqrt(alpha_bar).view(-1, 1, 1, 1) * z0
            + sigma.view(-1, 1, 1, 1) * eps
        )
        values[str(time)] = F.mse_loss(
            model.score_net(z_t, t), eps
        ).item()
    return float(np.mean(list(values.values()))), values


@torch.no_grad()
def evaluate_fids(model, dataset, n_fid, device):
    if not FID_AVAILABLE:
        return float("nan"), float("nan")
    batch_size = 32
    real, reconstructed, generated = [], [], []
    for _ in range(n_fid // batch_size):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)
        real.append(
            (x0.view(-1, 3, 64, 64).clamp(0, 1) * 255)
            .to(torch.uint8)
            .cpu()
        )
        mu = model.vae_encoder.encode_deterministic(x0)
        reconstruction = model.decoder(mu)
        reconstructed.append(
            (reconstruction.view(-1, 3, 64, 64).clamp(0, 1) * 255)
            .to(torch.uint8)
            .cpu()
        )
        samples = model.sample(
            batch_size, n_steps=500, device=device, clamp=(0, 1)
        )
        generated.append(
            (samples.view(-1, 3, 64, 64).clamp(0, 1) * 255)
            .to(torch.uint8)
            .cpu()
        )
    real = torch.cat(real)[:n_fid]
    reconstructed = torch.cat(reconstructed)[:n_fid]
    generated = torch.cat(generated)[:n_fid]
    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    metric.update(real.to(device), real=True)
    metric.update(reconstructed.to(device), real=False)
    rfid = metric.compute().item()
    metric.reset()
    metric.update(real.to(device), real=True)
    metric.update(generated.to(device), real=False)
    fid = metric.compute().item()
    return rfid, fid


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CelebAHQDataset(img_size=64, max_samples=20000)
    model = LDMLargeLatent(
        3 * 64 * 64,
        (3, 64, 64),
        latent_ch=8,
        base_ch=128,
        unet_base_ch=128,
        kl_weight=1e-4,
    ).to(device)
    tag = f"celebahq_ll_{args.method}_seed{args.seed}"
    ae_path = os.path.join(CKPT_DIR, f"{tag}_ae.pt")
    full_path = os.path.join(CKPT_DIR, f"{tag}_full.pt")

    state = load_checkpoint(model, full_path, device)
    if state is None:
        if args.method == "freqwarm":
            source = os.path.join(
                PROJECT_ROOT, "checkpoints", "celebahq_ll_ldm_ae.pt"
            )
            load_checkpoint(model, source, device)
        else:
            ae_state = load_checkpoint(model, ae_path, device)
            if ae_state is None:
                train_scale_equivariant_vae(
                    model,
                    dataset,
                    n_steps=args.ae_steps,
                    batch_size=args.batch_size,
                    scale_weight=args.scale_weight,
                    scale_factors=args.scale_factors,
                    device=device,
                )
                save_checkpoint(model, ae_path)
        model.freeze_ae()
        train_diffusion(
            model,
            dataset,
            n_steps=args.diff_steps,
            batch_size=args.batch_size,
            device=device,
            freqwarm=args.method == "freqwarm",
            warmup_steps=args.warmup_steps,
            cutoff=args.cutoff,
        )
        save_checkpoint(model, full_path)
    else:
        model.freeze_ae()

    eps, eps_by_time = evaluate_eps(model, dataset, device)
    rfid, fid = evaluate_fids(model, dataset, args.n_fid, device)
    with torch.no_grad():
        grid = model.sample(64, 500, device, (0, 1)).cpu()
    plot_grid(
        grid, 8, args.method, os.path.join(FIG_DIR, f"{tag}.png")
    )
    output = {
        "method": args.method,
        "seed": args.seed,
        "experiment_variant": experiment_variant(),
        "config": vars(args),
        "results": {
            "eps_loss": eps,
            "eps_loss_by_time": eps_by_time,
            "rfid": rfid,
            "fid": fid,
        },
    }
    output_path = os.path.join(LOG_DIR, f"{tag}_results.json")
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"  Saved results: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["freqwarm", "scaleeq"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ae-steps", type=int, default=100000)
    parser.add_argument("--diff-steps", type=int, default=300000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=225000)
    parser.add_argument("--cutoff", type=float, default=0.2)
    parser.add_argument("--scale-weight", type=float, default=1.0)
    parser.add_argument(
        "--scale-factors", type=float, nargs="+", default=[0.5, 0.25]
    )
    parser.add_argument("--n-fid", type=int, default=2048)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
