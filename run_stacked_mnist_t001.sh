#! /bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -jc gtn-container_g1
#$ -ac d=nvcr-pytorch-2311
#$ -m e
#$ -M wei.huang.vr@riken.jp

set -euo pipefail

source /home/weihuang/mamba/etc/profile.d/conda.sh
conda activate diffusion

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export EXPERIMENT_VARIANT="${EXPERIMENT_VARIANT:-sild_t001}"

cd /home/weihuang/diffusion_latent
mkdir -p "logs/$EXPERIMENT_VARIANT" "figures/$EXPERIMENT_VARIANT" \
    "checkpoints/$EXPERIMENT_VARIANT"

echo "=== Stacked MNIST: SiLD Stage-1 t0 = 0.001, Stage-2 [0.001, 0.999] ==="
echo "Variant: $EXPERIMENT_VARIANT"
echo "Seed: ${SEED:?SEED must be set}"
echo "Date: $(date)"
echo "Stage 1: denoising AE, sigma_t0 = t0 = 0.001 (vs 0.005 in Table 1)"
echo "Stage 2: [t0, 0.999] = [0.001, 0.999] -- matches CleanAE+LDM and LDM-CNN"
echo "LDM-CNN arm resumes from the sild_matches_ldm checkpoints (t0-independent)"

python -u experiments/compare_stacked_mnist.py --seed "$SEED" --t0 0.001

echo "=== Done ==="
echo "Date: $(date)"
