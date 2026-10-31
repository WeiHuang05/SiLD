# SiLD — Score-induced Latent Diffusion

Research code for a two-stage latent diffusion model that separates *geometry
learning* from *density learning*:

- **Stage 1** trains a deterministic encoder–decoder at low noise, which
  identifies the data manifold's tangent/normal decomposition.
- **Stage 2** freezes that map and fits a latent diffusion model in the
  learned coordinates.

The comparison throughout is against an architecture-matched latent diffusion
baseline whose Stage 1 is a KL-regularized (β-VAE) bottleneck. Encoder,
decoder, latent score network, and training budget are held identical; only
the Stage-1 objective differs.

## Layout

| Path | Contents |
|---|---|
| `models/` | Two-stage model, latent-diffusion CNN baseline, score networks |
| `data/` | Dataset wrappers: images, synthetic manifolds, molecules (SMILES/SELFIES, 3D) |
| `experiments/` | One `compare_*.py` per benchmark, plus ablation and aggregation scripts |
| `run_*.sh` | Grid-engine job scripts; each wraps one `experiments/` entry point |
| `logs/` | Every reported metric, as JSON (checked in) |
| `figures/` | Paper figures and the LaTeX for the experiments section |

## Reproducing the reported numbers

Every number in the paper is committed under `logs/` as JSON, so results can be
inspected without retraining. The scripts that regenerate them:

```bash
# Synthetic: two-stage dynamics without oracle knowledge of the subspace
bash run_toy_algorithm1_no_svd.sh

# Stacked MNIST (1000-mode benchmark), 3 seeds
bash run_stacked_mnist.sh                 # SiLD + beta-VAE LDM baseline
bash run_stacked_mnist_clean_ae_ldm.sh    # the sigma -> 0 member of the family
bash run_stacked_stage1_vp_noise.sh       # VP noise-form check
python experiments/summarize_stage1_noise_sweep.py

# Molecules (SELFIES), four MoleculeNet benchmarks
bash run_selfies_qm9.sh                   # also: _hiv, _muv, _pcba
bash run_selfies_clean_ae_ldm.sh qm9      # sigma -> 0 arm
bash run_collapse_diagnosis.sh            # A/B/C pipeline decomposition

# Images
bash run_celeba.sh ; bash run_celebahq.sh ; bash run_cifar.sh
```

Figures are rebuilt with `make_paper_figure.py`, `make_paper_figure_ll.py`, and
`make_stacked_mnist_family_figure.py`.

## Not in this repository

- **Trained weights** (`checkpoints/`, ~91 GB) — regenerate with the scripts above.
- **Third-party baselines.** `external_baselines/diffusability` is a checkout of
  [snap-research/diffusability](https://github.com/snap-research/diffusability)
  and is not vendored here; clone it into `external_baselines/` if you need it.
  The GeoLDM 3D experiments likewise expect a separate GeoLDM checkout.
- **Datasets** — downloaded on first use by the wrappers in `data/`.

## Paper

`_NeurIPS_2026_Submission__Diffusion_Low_Dimension.pdf` is the submitted draft.
The experiments section source is `figures/experiment.tex` (main text) and
`figures/exp_append.tex` (appendix); note that the PDF predates the current
revision of those files, so where they disagree the `.tex` files are newer.
