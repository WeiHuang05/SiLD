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
export EXPERIMENT_VARIANT="${EXPERIMENT_VARIANT:-clean_ae_ldm_selfies}"

cd /home/weihuang/diffusion_latent
mkdir -p \
    logs/clean_ae_ldm_selfies \
    checkpoints/clean_ae_ldm_selfies

DATASET="${1:?usage: run_selfies_clean_ae_ldm.sh DATASET [SEED]}"
SEED="${2:-42}"

echo "=== SELFIES CleanAE + LDM causal control ==="
echo "Dataset: ${DATASET}; seed: ${SEED}"
echo "Date: $(date)"

python -u experiments/compare_selfies_clean_ae_ldm.py \
    --dataset "${DATASET}" \
    --seed "${SEED}"

echo "=== Done ==="
echo "Date: $(date)"
