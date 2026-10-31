#! /bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -jc gtb-container_g1
#$ -ac d=nvcr-pytorch-2311
#$ -m e
#$ -M wei.huang.vr@riken.jp

set -euo pipefail

source /home/weihuang/mamba/etc/profile.d/conda.sh
conda activate diffusion

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export EXPERIMENT_VARIANT="${EXPERIMENT_VARIANT:-clean_ae_ldm_stacked}"

cd /home/weihuang/diffusion_latent
mkdir -p "logs/$EXPERIMENT_VARIANT" "figures/$EXPERIMENT_VARIANT" \
    "checkpoints/$EXPERIMENT_VARIANT"

echo "=== Stacked MNIST: Clean deterministic AE + standard LDM Stage 2 ==="
echo "Date: $(date)"
echo "Variant: $EXPERIMENT_VARIANT"
echo "Seed: ${SEED:?SEED must be set}"
echo "Stage 1: clean x0 reconstruction, no corruption, no KL"
echo "Stage 2: standard LDM on [0.001, 0.999]"

python -u experiments/compare_stacked_mnist_clean_ae_ldm.py \
    --seed "$SEED" \
    --stage2-t-min 0.001 \
    --ae-steps 20000 \
    --diff-steps 80000 \
    --batch-size 128 \
    --sample-steps 500 \
    --n-eval 25600 \
    --n-fid 10000

echo "=== Done ==="
echo "Date: $(date)"
