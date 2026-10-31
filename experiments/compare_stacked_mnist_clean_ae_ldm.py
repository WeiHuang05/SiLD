"""Stacked MNIST: deterministic clean autoencoder + standard latent diffusion.

This controlled baseline keeps the SiLD encoder, decoder, latent dimension,
score network, optimizer budgets, and sampler, while changing Stage 1 to a
plain deterministic autoencoder trained only on clean samples:

    x0 -> encoder -> decoder -> x_hat,  loss = MSE(x_hat, x0)

There is no input corruption, stochastic reparameterization, or KL penalty.
After Stage 1, the autoencoder is frozen and Stage 2 is the ordinary LDM
latent noise-prediction objective on [0.001, 0.999].
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.image_data import StackedMNISTDataset
from experiments.compare_stacked_mnist import (
    MNISTClassifier,
    TORCH_FIDELITY_AVAILABLE,
    _save_uint8_pngs,
    classify_stacked,
    compute_fid_torch_fidelity,
    evaluate_mode_coverage,
    plot_image_grid,
)
from models.two_stage_latent import TwoStageManifoldDiffusion
from utils.experiment_variant import experiment_variant, output_dir


FIG_DIR = output_dir(__file__, "figures")
CKPT_DIR = output_dir(__file__, "checkpoints")
LOG_DIR = output_dir(__file__, "logs")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def time_tag(value):
    return f"{int(round(value * 1000)):03d}"


def load_classifier(device):
    checkpoint_path = os.path.join(
        PROJECT_ROOT,
        "checkpoints",
        "matched_stage2",
        "mnist_classifier.pt",
    )
    state = torch.load(checkpoint_path, map_location=device)
    classifier = MNISTClassifier().to(device)
    classifier.load_state_dict(state["model"])
    classifier.eval()
    print(
        f"  Loaded MNIST classifier: {checkpoint_path} "
        f"(test_acc={state.get('test_acc', float('nan')):.4f})"
    )
    return classifier


def save_checkpoint(model, path, **metadata):
    torch.save({"model": model.state_dict(), **metadata}, path)
    print(f"  Saved checkpoint: {path}")


def load_checkpoint(model, path, device):
    if not os.path.exists(path):
        return None
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    print(f"  Loaded checkpoint: {path}")
    return state


def train_clean_autoencoder(
    model, dataset, *, n_steps, lr, batch_size, input_noise_std=0.0,
):
    """Train a deterministic AE with an explicit, documented input noise std."""
    device = next(model.parameters()).device
    parameters = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    losses = []

    print(
        f"  Deterministic AE: steps={n_steps}, batch={batch_size}, lr={lr}, "
        f"input_noise_std={input_noise_std:.8f}, objective=MSE (no KL)"
    )
    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)
        x_in = x0 + input_noise_std * torch.randn_like(x0)
        _, x_hat = model.project(x_in)
        loss = F.mse_loss(x_hat, x0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if (step + 1) % 50 == 0 or step + 1 == n_steps:
            print(
                f"  AE [{step + 1}/{n_steps}] "
                f"recon_loss={loss.item():.6f}"
            )

    with torch.no_grad():
        latent_batches = []
        for _ in range(10):
            x0, _ = dataset.sample_clean(min(batch_size, 5000))
            latent_batches.append(model.encoder(x0.to(device)))
        latents = torch.cat(latent_batches, dim=0)
        model.z_mean.copy_(latents.mean(dim=0))
        model.z_std.copy_(latents.std(dim=0).clamp(min=1e-6))

    print(
        f"  Latent stats: mean_norm={model.z_mean.norm():.4f}, "
        f"std=[{model.z_std.min():.4f}, {model.z_std.max():.4f}]"
    )
    return {
        "last_loss": losses[-1],
        "loss_last_1k": float(np.mean(losses[-1000:])),
    }


@torch.no_grad()
def reconstruction_mse(model, dataset, device, n_samples=2000):
    x0, _ = dataset.sample_clean(n_samples)
    x0 = x0.to(device)
    _, x_hat = model.project(x0)
    return F.mse_loss(x_hat, x0).item()


@torch.no_grad()
def evaluate_eps_loss(model, dataset, device):
    eval_times = sorted(
        {max(model.diffusion_t_min, 0.01), 0.1, 0.3, 0.5, 0.8}
    )
    per_time = {}
    for t_eval in eval_times:
        x0, _ = dataset.sample_clean(2000)
        z0 = model.encode(x0.to(device))
        t_batch = torch.full((z0.shape[0],), t_eval, device=device)
        z_t, target = model.forward_diffusion_latent(z0, t_batch)
        prediction = model.predict_eps_latent(z_t, t_batch)
        mse = F.mse_loss(prediction, target).item()
        per_time[str(t_eval)] = mse
        print(f"  t={t_eval:.2f}: latent_eps_MSE={mse:.6f}")
    return float(np.mean(list(per_time.values()))), per_time


@torch.no_grad()
def sample_in_batches(model, n_samples, device, *, n_steps, batch_size=256):
    batches = []
    for start in range(0, n_samples, batch_size):
        size = min(batch_size, n_samples - start)
        images = model.sample(
            size,
            n_steps=n_steps,
            device=device,
            clamp=(0, 1),
        )
        batches.append(images.view(size, -1).cpu())
        if (start // batch_size + 1) % 8 == 0:
            print(f"    {min(start + batch_size, n_samples)}/{n_samples}")
    return torch.cat(batches, dim=0)


def run(args):
    if args.n_eval < args.n_fid:
        raise ValueError(f"n_eval ({args.n_eval}) must be >= n_fid ({args.n_fid})")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_shape = (3, 28, 28)
    d = 3 * 28 * 28
    mode_suffix = "" if args.stage1_noise_mode == "clean" else (
        f"_{args.stage1_noise_mode}"
    )
    tag = (
        f"tmin{time_tag(args.stage2_t_min)}{mode_suffix}_seed{args.seed}"
    )

    print("=" * 76)
    print("STACKED MNIST — CLEAN DETERMINISTIC AE + STANDARD LDM STAGE 2")
    print(f"  device={device}, seed={args.seed}")
    print(f"  experiment_variant={experiment_variant()}")
    print(
        f"  Stage 1: clean deterministic AE, {args.ae_steps} steps; "
        "no corruption, no KL"
    )
    print(
        f"  Stage 2: ordinary latent diffusion on "
        f"[{args.stage2_t_min}, 0.999], {args.diff_steps} steps"
    )
    print("=" * 76)

    dataset = StackedMNISTDataset(n_pool=60000, seed=args.seed)
    classifier = load_classifier(device)
    model = TwoStageManifoldDiffusion(
        d=d,
        latent_dim=64,
        hidden_dim_enc=512,
        hidden_dim_score=512,
        n_layers=6,
        t0=0.005,
        image_shape=image_shape,
        base_ch=64,
        diffusion_t_min=args.stage2_t_min,
    ).to(device)
    if args.stage1_noise_mode == "clean":
        stage1_noise_std = 0.0
    elif args.stage1_noise_mode == "legacy":
        stage1_noise_std = float(model.t0)
    else:
        _, vp_sigma = model.noise_schedule(
            torch.tensor(model.t0, device=device)
        )
        stage1_noise_std = float(vp_sigma.item())
    print(
        f"  Stage-1 noise mode={args.stage1_noise_mode}: "
        f"t0={model.t0}, additive std={stage1_noise_std:.8f}"
    )

    ae_path = os.path.join(CKPT_DIR, f"stacked_mnist_clean_ae_{tag}.pt")
    full_path = os.path.join(
        CKPT_DIR,
        f"stacked_mnist_clean_ae_ldm_full_{tag}.pt",
    )

    full_state = None
    if not args.force_train:
        full_state = load_checkpoint(model, full_path, device)

    if full_state is not None:
        model.freeze_stage1()
        ae_stats = full_state["ae_stats"]
        recon_mse = full_state["recon_mse"]
        stage2_stats = full_state["stage2_stats"]
    else:
        ae_state = None
        if not args.force_train:
            ae_state = load_checkpoint(model, ae_path, device)
        if ae_state is None:
            ae_stats = train_clean_autoencoder(
                model,
                dataset,
                n_steps=args.ae_steps,
                lr=args.ae_lr,
                batch_size=args.batch_size,
                input_noise_std=stage1_noise_std,
            )
            recon_mse = reconstruction_mse(model, dataset, device)
            print(f"  Clean AE reconstruction MSE: {recon_mse:.6f}")
            save_checkpoint(
                model,
                ae_path,
                ae_stats=ae_stats,
                recon_mse=recon_mse,
                config=vars(args),
            )
        else:
            ae_stats = ae_state["ae_stats"]
            recon_mse = ae_state["recon_mse"]

        model.freeze_stage1()
        stage2_losses = model.train_stage2(
            dataset,
            n_steps=args.diff_steps,
            lr=args.diff_lr,
            batch_size=args.batch_size,
        )
        stage2_stats = {
            "last_loss": stage2_losses[-1],
            "loss_last_1k": float(np.mean(stage2_losses[-1000:])),
        }
        save_checkpoint(
            model,
            full_path,
            ae_stats=ae_stats,
            recon_mse=recon_mse,
            stage2_stats=stage2_stats,
            config=vars(args),
        )

    avg_eps_loss, eps_loss_by_time = evaluate_eps_loss(model, dataset, device)

    grid = model.sample(
        64,
        n_steps=args.sample_steps,
        device=device,
        clamp=(0, 1),
    ).cpu()
    plot_image_grid(
        grid,
        nrow=8,
        title=f"CleanAE + LDM ({tag})",
        save_name=f"stacked_mnist_gen_clean_ae_ldm_{tag}.png",
    )

    print(f"\n--- Mode coverage ({args.n_eval} samples) ---")
    generated = sample_in_batches(
        model,
        args.n_eval,
        device,
        n_steps=args.sample_steps,
    )
    _, generated_modes = classify_stacked(classifier, generated, device=device)
    coverage = evaluate_mode_coverage(generated_modes)
    print(
        f"  CleanAE+LDM: modes={coverage['n_modes_covered']}/1000, "
        f"KL_unif={coverage['kl_to_uniform']:.6f}"
    )

    real_eval, real_labels = dataset.sample_clean(args.n_eval)
    real_modes = (
        real_labels[:, 0] * 100
        + real_labels[:, 1] * 10
        + real_labels[:, 2]
    )
    real_coverage = evaluate_mode_coverage(real_modes)
    _, classified_real_modes = classify_stacked(
        classifier,
        real_eval,
        device=device,
    )
    classifier_acc = (classified_real_modes == real_modes).float().mean().item()

    fidelity = {}
    if args.n_fid > 0 and TORCH_FIDELITY_AVAILABLE:
        scratch = os.path.join(
            "/tmp",
            f"stacked_mnist_clean_ae_ldm_fid_{os.getpid()}",
        )
        try:
            _save_uint8_pngs(
                real_eval[: args.n_fid].view(-1, 3, 28, 28),
                os.path.join(scratch, "real"),
                "r",
            )
            _save_uint8_pngs(
                generated[: args.n_fid].view(-1, 3, 28, 28),
                os.path.join(scratch, "clean_ae_ldm"),
                "c",
            )
            fidelity = compute_fid_torch_fidelity(
                os.path.join(scratch, "real"),
                os.path.join(scratch, "clean_ae_ldm"),
            )
            print(
                f"  CleanAE+LDM FID={fidelity['fid']:.4f}, "
                f"IS={fidelity['is_mean']:.4f}+/-{fidelity['is_std']:.4f}"
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    reference_path = os.path.join(
        PROJECT_ROOT,
        "logs",
        "sild_matches_ldm",
        f"stacked_mnist_results_t005_seed{args.seed}.json",
    )
    reference = None
    if os.path.exists(reference_path):
        with open(reference_path) as handle:
            reference = json.load(handle)

    summary = {
        "config": {
            **vars(args),
            "experiment_variant": experiment_variant(),
            "stage1_input": args.stage1_noise_mode,
            "stage1_noise_std": stage1_noise_std,
            "stage1_objective": "deterministic_denoising_mse_no_kl",
            "stage2_objective": "standard_ldm_latent_noise_prediction",
            "stage2_time_range": [args.stage2_t_min, 0.999],
        },
        "classifier_acc_on_real_stacked": classifier_acc,
        "real": real_coverage,
        "clean_ae_ldm": {
            "recon_mse": recon_mse,
            "avg_eps_loss": avg_eps_loss,
            "eps_loss_by_time": eps_loss_by_time,
            **coverage,
            **fidelity,
        },
        "train": {
            "clean_ae": ae_stats,
            "stage2": stage2_stats,
        },
        "reference": {
            "path": reference_path,
            "sild": reference["uld"] if reference else None,
            "vae_ldm": reference["ldm"] if reference else None,
        },
    }
    output_path = os.path.join(
        LOG_DIR,
        f"stacked_mnist_clean_ae_ldm_results_{tag}.json",
    )
    with open(output_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"  Saved results: {output_path}")

    print("\n" + "=" * 76)
    print("STACKED MNIST COMPARISON")
    print("=" * 76)
    print(
        f"  {'Method':<18} {'Modes':>8} {'KL':>10} "
        f"{'Recon':>12} {'Eps loss':>12} {'FID':>10}"
    )
    rows = []
    if reference:
        rows.extend(
            [
                ("SiLD", reference["uld"]),
                ("VAE-LDM", reference["ldm"]),
            ]
        )
    rows.append(("CleanAE+LDM", summary["clean_ae_ldm"]))
    for name, result in rows:
        print(
            f"  {name:<18} {result['n_modes_covered']:>8} "
            f"{result['kl_to_uniform']:>10.5f} "
            f"{result['recon_mse']:>12.6f} "
            f"{result['avg_eps_loss']:>12.5f} "
            f"{result.get('fid', float('nan')):>10.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage2-t-min", type=float, default=0.001)
    parser.add_argument("--ae-steps", type=int, default=20000)
    parser.add_argument("--diff-steps", type=int, default=80000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ae-lr", type=float, default=1e-4)
    parser.add_argument("--diff-lr", type=float, default=3e-4)
    parser.add_argument("--sample-steps", type=int, default=500)
    parser.add_argument("--n-eval", type=int, default=25600)
    parser.add_argument("--n-fid", type=int, default=10000)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument(
        "--stage1-noise-mode",
        choices=["clean", "legacy", "vp_schedule"],
        default="clean",
    )
    run(parser.parse_args())
