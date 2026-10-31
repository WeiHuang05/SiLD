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
export EXPERIMENT_VARIANT="${EXPERIMENT_VARIANT:-freqwarm_stacked}"

cd /home/weihuang/diffusion_latent
mkdir -p "logs/$EXPERIMENT_VARIANT" "figures/$EXPERIMENT_VARIANT" \
    "checkpoints/$EXPERIMENT_VARIANT"

echo "=== Stacked MNIST: LDM + FreqWarm ==="
echo "Date: $(date)"
echo "Variant: $EXPERIMENT_VARIANT"
echo "Queue: gtb-container_g1"
echo "Stage 2: [0.005, 0.999], 60K low-pass + 20K full-frequency"

SEED="${1:-42}"

python -u experiments/compare_stacked_mnist_freqwarm.py \
    --seed "$SEED" \
    --stage2-t-min 0.005 \
    --diff-steps 80000 \
    --warmup-steps 60000 \
    --batch-size 128 \
    --cutoff 0.20 \
    --filter-type ideal \
    --ae-checkpoint "stacked_mnist_ldm_ae_t005_seed${SEED}.pt" \
    --baseline-results "../../logs/matched_stage2/stacked_mnist_results_t005_seed${SEED}.json"

echo "=== Done ==="
echo "Date: $(date)"
