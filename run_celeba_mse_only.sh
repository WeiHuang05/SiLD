#! /bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -jc gtb-container_g1
#$ -ac d=nvcr-pytorch-2311
#$ -m e
#$ -M wei.huang.vr@riken.jp

source /home/weihuang/mamba/etc/profile.d/conda.sh
conda activate diffusion

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export EXPERIMENT_VARIANT=celeba_mse_only
export MPLCONFIGDIR=/tmp/matplotlib-ce-mse-${JOB_ID}

cd /home/weihuang/diffusion_latent
mkdir -p \
    logs/celeba_mse_only \
    checkpoints/celeba_mse_only \
    figures/celeba_mse_only

METHOD="${1:?usage: run_celeba_mse_only.sh METHOD [SEED]}"
SEED="${2:-42}"
python -u experiments/compare_celeba_mse_only_symmetric.py \
    --method "${METHOD}" \
    --seed "${SEED}"
