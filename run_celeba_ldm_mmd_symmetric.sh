#! /bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -jc gtn-container_g1
#$ -ac d=nvcr-pytorch-2311
#$ -m e
#$ -M wei.huang.vr@riken.jp

source /home/weihuang/mamba/etc/profile.d/conda.sh
conda activate diffusion

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export EXPERIMENT_VARIANT=symmetric_mmd

cd /home/weihuang/diffusion_latent
mkdir -p logs/symmetric_mmd figures/symmetric_mmd checkpoints/symmetric_mmd

MMD_WEIGHT="${1:?usage: run_celeba_ldm_mmd_symmetric.sh MMD_WEIGHT [SEED]}"
SEED="${2:-42}"

echo "=== CelebA VAE-LDM symmetric MMD control ==="
echo "MMD weight: ${MMD_WEIGHT}; seed: ${SEED}"
echo "Date: $(date)"

python -u experiments/compare_celeba_ldm_mmd_symmetric.py \
    --mmd-weight "${MMD_WEIGHT}" \
    --seed "${SEED}"

echo "=== Done ==="
echo "Date: $(date)"
