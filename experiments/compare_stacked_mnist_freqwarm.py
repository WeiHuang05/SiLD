"""Stacked MNIST: LDM + FreqWarm using the completed matched-stage2 AE.

The VAE checkpoint from the seed-42 matched comparison is frozen. A fresh
latent score model is trained for the same 80K-step Stage-2 budget:

  * first 60K steps: encode RGB images low-pass filtered at r0=0.20;
  * final 20K steps: encode the original full-frequency RGB images.

The Stage-2 diffusion interval is the same [0.005, 0.999] interval recorded
for Job 18452203, so its LDM and SiLD metrics are directly comparable.
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
from models.latent_diffusion_cnn import LatentDiffusionCNN
from utils.experiment_variant import experiment_variant, output_dir
from utils.frequency_filter import apply_frequency_filter


FIG_DIR = output_dir(__file__, "figures")
CKPT_DIR = output_dir(__file__, "checkpoints")
LOG_DIR = output_dir(__file__, "logs")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_frozen_ae(model, checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.freeze_ae()
    # The Stage-1 checkpoint also contains the original, still-untrained score
    # network initialization. Keeping it gives the closest controlled comparison
    # with the completed LDM run, which started from this same checkpoint.
    print(f"  Loaded frozen VAE from: {checkpoint_path}")


def train_freqwarm(
    model,
    dataset,
    *,
    n_steps,
    warmup_steps,
    cutoff,
    filter_type,
    lr,
    batch_size,
):
    if not 0 <= warmup_steps < n_steps:
        raise ValueError(
            f"warmup_steps must be in [0, n_steps), got {warmup_steps}/{n_steps}"
        )

    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.score_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    losses = []
    warm_losses = []
    full_losses = []

    print(
        f"  FreqWarm Stage 2: t in [{model.diffusion_t_min}, {model.T}], "
        f"cutoff={cutoff}, filter={filter_type}, "
        f"warm/full={warmup_steps}/{n_steps - warmup_steps} steps"
    )

    for step in range(n_steps):
        x0, _ = dataset.sample_clean(batch_size)
        x0 = x0.to(device)
        in_warmup = step < warmup_steps

        with torch.no_grad():
            if in_warmup:
                x_for_encoding = apply_frequency_filter(
                    x0.view(-1, 3, 28, 28),
                    keep="low",
                    cutoff=cutoff,
                    filter_type=filter_type,
                ).view(x0.shape[0], -1)
            else:
                x_for_encoding = x0
            z0 = model.encode(x_for_encoding)

        t = (
            torch.rand(z0.shape[0], device=device)
            * (model.T - model.diffusion_t_min)
            + model.diffusion_t_min
        )
        z_t, eps_z = model.forward_diffusion_latent(z0, t)
        eps_z_pred = model.score_net(z_t, t)
        loss = F.mse_loss(eps_z_pred, eps_z)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.score_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        value = loss.item()
        losses.append(value)
        (warm_losses if in_warmup else full_losses).append(value)

        if step + 1 == warmup_steps:
            print("  Switching from low-pass latents to full-frequency latents.")
        if (step + 1) % 200 == 0:
            phase = "warm" if in_warmup else "full"
            print(
                f"  Diffusion [{step + 1}/{n_steps}] "
                f"phase={phase} latent_eps_loss={value:.6f}"
            )

    return {
        "last_loss": losses[-1],
        "warm_loss_last_1k": float(np.mean(warm_losses[-1000:])),
        "full_loss_last_1k": float(np.mean(full_losses[-1000:])),
    }


@torch.no_grad()
def evaluate_eps_loss(model, dataset, device):
    eval_times = sorted(
        {max(model.diffusion_t_min, 0.01), 0.1, 0.3, 0.5, 0.8}
    )
    per_time = {}
    for t_eval in eval_times:
        x0, _ = dataset.sample_clean(2000)
        x0 = x0.to(device)
        z0 = model.encode(x0)
        t_batch = torch.full((x0.shape[0],), t_eval, device=device)
        z_t, eps_z = model.forward_diffusion_latent(z0, t_batch)
        eps_z_pred = model.score_net(z_t, t_batch)
        mse = F.mse_loss(eps_z_pred, eps_z).item()
        per_time[str(t_eval)] = mse
        print(f"  t={t_eval:.2f}: full-frequency latent_eps_MSE={mse:.6f}")
    return float(np.mean(list(per_time.values()))), per_time


@torch.no_grad()
def sample_in_batches(model, n_samples, device, n_steps=500, batch_size=256):
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


def load_classifier(device):
    classifier_path = os.path.join(
        PROJECT_ROOT,
        "checkpoints",
        "matched_stage2",
        "mnist_classifier.pt",
    )
    state = torch.load(classifier_path, map_location=device)
    classifier = MNISTClassifier().to(device)
    classifier.load_state_dict(state["model"])
    classifier.eval()
    print(
        f"  Loaded MNIST classifier: {classifier_path} "
        f"(test_acc={state.get('test_acc', float('nan')):.4f})"
    )
    return classifier


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_shape = (3, 28, 28)
    d = 3 * 28 * 28
    latent_dim = 64
    ae_checkpoint = os.path.join(
        PROJECT_ROOT,
        args.baseline_variant,
        args.ae_checkpoint,
    )
    baseline_results = os.path.join(
        PROJECT_ROOT,
        args.baseline_variant,
        args.baseline_results,
    )

    print("=" * 72)
    print("STACKED MNIST — LDM + FreqWarm")
    print(f"  device={device}, seed={args.seed}")
    print(f"  experiment_variant={experiment_variant()}")
    print(f"  Stage-2 time=[{args.stage2_t_min}, 0.999]")
    print(
        f"  total_steps={args.diff_steps}, warmup_steps={args.warmup_steps}, "
        f"cutoff={args.cutoff}, filter={args.filter_type}"
    )
    print("=" * 72)

    dataset = StackedMNISTDataset(n_pool=60000, seed=args.seed)
    classifier = load_classifier(device)
    model = LatentDiffusionCNN(
        d=d,
        latent_dim=latent_dim,
        hidden_dim_enc=512,
        hidden_dim_score=512,
        n_layers=6,
        kl_weight=1e-4,
        image_shape=image_shape,
        base_ch=64,
        diffusion_t_min=args.stage2_t_min,
    ).to(device)
    load_frozen_ae(model, ae_checkpoint, device)

    full_checkpoint_path = os.path.join(
        CKPT_DIR,
        f"stacked_mnist_ldm_freqwarm_t005_seed{args.seed}.pt",
    )
    if os.path.exists(full_checkpoint_path) and not args.force_train:
        state = torch.load(full_checkpoint_path, map_location=device)
        model.load_state_dict(state["model"])
        model.freeze_ae()
        train_stats = state["train_stats"]
        print(f"  Loaded completed FreqWarm checkpoint: {full_checkpoint_path}")
    else:
        train_stats = train_freqwarm(
            model,
            dataset,
            n_steps=args.diff_steps,
            warmup_steps=args.warmup_steps,
            cutoff=args.cutoff,
            filter_type=args.filter_type,
            lr=args.lr,
            batch_size=args.batch_size,
        )
        torch.save(
            {
                "model": model.state_dict(),
                "train_stats": train_stats,
                "config": vars(args),
            },
            full_checkpoint_path,
        )
        print(f"  Saved checkpoint: {full_checkpoint_path}")

    avg_eps_loss, eps_loss_by_time = evaluate_eps_loss(model, dataset, device)

    real_probe, real_labels = dataset.sample_clean(2000)
    filtered_probe = apply_frequency_filter(
        real_probe.view(-1, 3, 28, 28).to(device),
        keep="low",
        cutoff=args.cutoff,
        filter_type=args.filter_type,
    ).view(real_probe.shape[0], -1)
    filter_mse = F.mse_loss(filtered_probe.cpu(), real_probe).item()
    _, filtered_modes = classify_stacked(
        classifier, filtered_probe.cpu(), device=device
    )
    true_modes = (
        real_labels[:, 0] * 100 + real_labels[:, 1] * 10 + real_labels[:, 2]
    )
    filtered_mode_acc = (filtered_modes == true_modes).float().mean().item()
    print(
        f"  Low-pass diagnostic: image_MSE={filter_mse:.6f}, "
        f"mode_identity_acc={filtered_mode_acc:.4f}"
    )

    grid = model.sample(64, n_steps=500, device=device, clamp=(0, 1)).cpu()
    plot_image_grid(
        grid,
        nrow=8,
        title=f"LDM + FreqWarm (seed {args.seed})",
        save_name=f"stacked_mnist_gen_ldm_freqwarm_t005_seed{args.seed}.png",
    )

    print(f"\n--- Mode coverage ({args.n_eval} samples) ---")
    generated = sample_in_batches(model, args.n_eval, device)
    _, generated_modes = classify_stacked(
        classifier, generated, device=device
    )
    coverage = evaluate_mode_coverage(generated_modes)
    print(
        f"  LDM+FreqWarm: modes={coverage['n_modes_covered']}/1000, "
        f"KL_unif={coverage['kl_to_uniform']:.6f}"
    )

    fid = {}
    if TORCH_FIDELITY_AVAILABLE:
        scratch = os.path.join(
            "/tmp", f"stacked_mnist_freqwarm_fid_{os.getpid()}"
        )
        try:
            real_fid, _ = dataset.sample_clean(args.n_fid)
            _save_uint8_pngs(
                real_fid.view(-1, 3, 28, 28),
                os.path.join(scratch, "real"),
                "r",
            )
            _save_uint8_pngs(
                generated[:args.n_fid].view(-1, 3, 28, 28),
                os.path.join(scratch, "freqwarm"),
                "f",
            )
            fid = compute_fid_torch_fidelity(
                os.path.join(scratch, "real"),
                os.path.join(scratch, "freqwarm"),
            )
            print(
                f"  LDM+FreqWarm FID={fid['fid']:.4f}, "
                f"IS={fid['is_mean']:.4f}+/-{fid['is_std']:.4f}"
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    with open(baseline_results) as handle:
        baseline = json.load(handle)

    summary = {
        "config": {
            **vars(args),
            "experiment_variant": experiment_variant(),
            "stage2_time_range": [args.stage2_t_min, 0.999],
            "ae_checkpoint": ae_checkpoint,
            "baseline_results": baseline_results,
        },
        "filter_diagnostic": {
            "mse": filter_mse,
            "mode_identity_acc": filtered_mode_acc,
        },
        "train": train_stats,
        "ldm_freqwarm": {
            "avg_eps_loss": avg_eps_loss,
            "eps_loss_by_time": eps_loss_by_time,
            **coverage,
            **fid,
        },
        "matched_job_18452203": {
            "uld": baseline["uld"],
            "ldm": baseline["ldm"],
        },
    }
    output_path = os.path.join(
        LOG_DIR,
        f"stacked_mnist_freqwarm_results_t005_seed{args.seed}.json",
    )
    with open(output_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"  Saved results: {output_path}")

    print("\n" + "=" * 72)
    print("STACKED MNIST COMPARISON")
    print("=" * 72)
    print(
        f"  {'Method':<18} {'Modes':>8} {'KL':>10} "
        f"{'Eps loss':>12} {'FID':>10}"
    )
    for name, result in (
        ("SiLD", baseline["uld"]),
        ("LDM", baseline["ldm"]),
        ("LDM+FreqWarm", summary["ldm_freqwarm"]),
    ):
        print(
            f"  {name:<18} {result['n_modes_covered']:>8} "
            f"{result['kl_to_uniform']:>10.5f} "
            f"{result['avg_eps_loss']:>12.5f} "
            f"{result.get('fid', float('nan')):>10.3f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage2-t-min", type=float, default=0.005)
    parser.add_argument("--diff-steps", type=int, default=80000)
    parser.add_argument("--warmup-steps", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--cutoff", type=float, default=0.20)
    parser.add_argument(
        "--filter-type",
        choices=["ideal", "butterworth", "gaussian"],
        default="ideal",
    )
    parser.add_argument("--n-eval", type=int, default=25600)
    parser.add_argument("--n-fid", type=int, default=10000)
    parser.add_argument(
        "--baseline-variant",
        default="checkpoints/matched_stage2",
    )
    parser.add_argument(
        "--ae-checkpoint",
        default="stacked_mnist_ldm_ae_t005_seed42.pt",
    )
    parser.add_argument(
        "--baseline-results",
        default="../../logs/matched_stage2/stacked_mnist_results_t005_seed42.json",
    )
    parser.add_argument("--force-train", action="store_true")
    run(parser.parse_args())
