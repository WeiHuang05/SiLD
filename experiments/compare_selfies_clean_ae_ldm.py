"""
Clean deterministic AE + ordinary latent diffusion on molecular SELFIES.

The experiment is a causal control for SiLD:

Stage 1
    clean x0 -> deterministic encoder/decoder -> x_hat
    objective: MSE(x_hat, x0), with no corruption, KL, or auxiliary loss

Stage 2
    freeze the AE and train exactly the same latent diffusion used by the
    matched SELFIES SiLD/LDM experiments.

Usage:
    python experiments/compare_selfies_clean_ae_ldm.py --dataset qm9
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

from data.molecule_data import SelfiesDataset
from experiments.compare_selfies import SELFIES_CONFIGS
from models.two_stage_latent import TwoStageManifoldDiffusion
from utils.experiment_variant import experiment_variant, output_dir


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = output_dir(__file__, "checkpoints")
LOG_DIR = output_dir(__file__, "logs")


def metric_text(value, digits=3):
    """Format optional scalar metrics without failing at job shutdown."""
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def save_checkpoint(model, path, **extra):
    state = {"model": model.state_dict(), **extra}
    torch.save(state, path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(model, path, device):
    if not os.path.exists(path):
        return None
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    print(f"  Checkpoint loaded: {path}")
    return state


def train_clean_autoencoder(model, dataset, n_steps, lr, batch_size):
    """Train a deterministic AE on clean x0 using reconstruction MSE only."""
    device = next(model.parameters()).device
    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    losses = []

    print(
        f"  Clean deterministic AE: {n_steps} steps, lr={lr}, "
        f"batch={batch_size}; objective=MSE only"
    )
    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)
        z, x_hat = model.project(x0)
        loss = F.mse_loss(x_hat, x0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if (step + 1) % 200 == 0:
            print(f"  CleanAE [{step + 1}/{n_steps}] mse={loss.item():.6f}")

    # Use the same post-AE latent normalization as SiLD.
    with torch.no_grad():
        latent_batches = []
        for _ in range(10):
            x0, _ = dataset.sample_clean(min(batch_size, 5000))
            latent_batches.append(model.encoder(x0.to(device)))
        latents = torch.cat(latent_batches, dim=0)
        model.z_mean.copy_(latents.mean(dim=0))
        model.z_std.copy_(latents.std(dim=0).clamp(min=1e-6))

    return {
        "last_loss": losses[-1],
        "loss_last_1k": float(np.mean(losses[-1000:])),
    }


@torch.no_grad()
def reconstruction_mse(model, dataset, device, n_eval=2000):
    x0, _ = dataset.sample_clean(n_eval)
    x0 = x0.to(device)
    _, x_hat = model.project(x0)
    return F.mse_loss(x_hat, x0).item()


@torch.no_grad()
def evaluate_eps_loss(model, dataset, device, stage2_t_min):
    eval_times = sorted(
        set([max(stage2_t_min, 0.01), 0.1, 0.3, 0.5, 0.8])
    )
    by_time = {}
    for t_eval in eval_times:
        x0, _ = dataset.sample_clean(2000)
        z0 = model.encode(x0.to(device))
        t_batch = torch.full((z0.shape[0],), t_eval, device=device)
        z_t, eps = model.forward_diffusion_latent(z0, t_batch)
        eps_pred = model.predict_eps_latent(z_t, t_batch)
        value = F.mse_loss(eps_pred, eps).item()
        by_time[str(t_eval)] = value
        print(f"  t={t_eval:.2f}: eps_MSE={value:.6f}")
    return float(np.mean(list(by_time.values()))), by_time


@torch.no_grad()
def sample_in_batches(model, n_samples, device, n_steps):
    batches = []
    batch_size = 512
    for start in range(0, n_samples, batch_size):
        size = min(batch_size, n_samples - start)
        samples = model.sample(size, n_steps=n_steps, device=device)
        batches.append(samples.cpu())
        if (start // batch_size + 1) % 5 == 0:
            print(f"    {min(start + batch_size, n_samples)}/{n_samples}")
    return torch.cat(batches, dim=0)[:n_samples]


def run(args):
    cfg = SELFIES_CONFIGS[args.dataset]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 76)
    print(f"{cfg['label']} SELFIES — CLEAN AE + ORDINARY LDM STAGE 2")
    print(f"  device={device}, seed={args.seed}")
    print(f"  experiment_variant={experiment_variant()}")
    print("  Stage 1: clean deterministic AE, MSE only, no noise, no KL")
    print(
        f"  Stage 2: [{args.stage2_t_min}, 0.999], "
        f"Min-SNR-gamma={args.min_snr_gamma}"
    )
    print("=" * 76)

    dataset = SelfiesDataset(
        csv_path=cfg["csv"],
        smiles_col=cfg["smiles_col"],
        max_len=cfg["max_len"],
        seed=args.seed,
        name=cfg["label"],
    )
    d = dataset.d
    model = TwoStageManifoldDiffusion(
        d=d,
        latent_dim=cfg["latent_dim"],
        hidden_dim_enc=cfg["hidden_dim"],
        hidden_dim_score=cfg["hidden_dim"],
        n_layers=8,
        t0=0.05,
        image_shape=None,
        diffusion_t_min=args.stage2_t_min,
    ).to(device)

    tag = f"{args.dataset}_selfies_clean_ae_ldm_tmin001_seed{args.seed}"
    ae_path = os.path.join(CKPT_DIR, f"{tag}_ae.pt")
    full_path = os.path.join(CKPT_DIR, f"{tag}_full.pt")

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
            )
            recon_mse = reconstruction_mse(model, dataset, device)
            print(f"  CleanAE reconstruction MSE: {recon_mse:.6f}")
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
            min_snr_gamma=args.min_snr_gamma,
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

    avg_eps_loss, eps_loss_by_time = evaluate_eps_loss(
        model, dataset, device, args.stage2_t_min
    )

    print(f"\n--- Generating and evaluating {args.n_eval} samples ---")
    generated = sample_in_batches(
        model, args.n_eval, device, n_steps=args.sample_steps
    )
    metrics = dataset.evaluate_samples(
        generated,
        compute_diversity=True,
        compute_qed=True,
        compute_fcd=True,
        fcd_reference_max=10000,
    )

    reference_path = os.path.join(
        PROJECT_ROOT,
        "logs",
        "sild_matches_ldm",
        f"{args.dataset}_selfies_results.json",
    )
    reference = None
    if os.path.exists(reference_path):
        with open(reference_path) as handle:
            reference = json.load(handle)

    clean_result = {
        **metrics,
        "recon_mse": recon_mse,
        "avg_eps_loss": avg_eps_loss,
        "eps_loss_by_time": eps_loss_by_time,
    }
    result = {
        "dataset": cfg["label"],
        "representation": "SELFIES",
        "method": "CleanAE+LDM",
        "config": {
            **vars(args),
            "d": d,
            "vocab_size": dataset.vocab_size,
            "max_len": cfg["max_len"],
            "latent_dim": cfg["latent_dim"],
            "hidden_dim": cfg["hidden_dim"],
            "n_score_layers": 8,
            "experiment_variant": experiment_variant(),
            "stage1_input": "clean_x0",
            "stage1_objective": "deterministic_reconstruction_mse_no_kl",
            "stage2_objective": "matched_latent_noise_prediction",
            "stage2_time_range": [args.stage2_t_min, 0.999],
        },
        "methods": {"CleanAE+LDM": clean_result},
        "train": {"clean_ae": ae_stats, "stage2": stage2_stats},
        "reference": {
            "path": reference_path,
            "real_baseline": reference.get("real_baseline") if reference else None,
            "SiLD": reference["methods"].get("ULD") if reference else None,
            "VAE-LDM": reference["methods"].get("LDM") if reference else None,
            "CharRNN": reference["methods"].get("CharRNN") if reference else None,
        },
    }

    output_path = os.path.join(
        LOG_DIR, f"{args.dataset}_selfies_clean_ae_ldm_seed{args.seed}.json"
    )
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"  Saved results: {output_path}")

    print("\n" + "=" * 92)
    print(f"{cfg['label']} SELFIES CAUSAL COMPARISON")
    print("=" * 92)
    print(
        f"  {'Method':<14} {'Valid':>7} {'Unique':>8} {'Novel':>8} "
        f"{'IntDiv':>8} {'QED':>8} {'FCD':>10} {'Recon':>10}"
    )
    rows = []
    if reference:
        rows.extend(
            [
                ("SiLD", reference["methods"]["ULD"]),
                ("VAE-LDM", reference["methods"]["LDM"]),
            ]
        )
    rows.append(("CleanAE+LDM", clean_result))
    for name, values in rows:
        print(
            f"  {name:<14} {metric_text(values.get('validity')):>7} "
            f"{metric_text(values.get('uniqueness')):>8} "
            f"{metric_text(values.get('novelty')):>8} "
            f"{metric_text(values.get('internal_diversity')):>8} "
            f"{metric_text(values.get('qed_mean')):>8} "
            f"{metric_text(values.get('fcd')):>10} "
            f"{metric_text(values.get('recon_mse'), 6):>10}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=sorted(SELFIES_CONFIGS.keys()), required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage2-t-min", type=float, default=0.001)
    parser.add_argument("--ae-steps", type=int, default=40000)
    parser.add_argument("--diff-steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ae-lr", type=float, default=1e-4)
    parser.add_argument("--diff-lr", type=float, default=3e-4)
    parser.add_argument("--min-snr-gamma", type=float, default=5.0)
    parser.add_argument("--sample-steps", type=int, default=200)
    parser.add_argument("--n-eval", type=int, default=10000)
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
