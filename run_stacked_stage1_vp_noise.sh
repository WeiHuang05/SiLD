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
export EXPERIMENT_VARIANT=stage1_vp_noise
export MPLCONFIGDIR=/tmp/matplotlib-vp-${JOB_ID}

cd /home/weihuang/diffusion_latent
mkdir -p \
    logs/stage1_vp_noise \
    checkpoints/stage1_vp_noise \
    figures/stage1_vp_noise

SEED="${1:-42}"
python -u experiments/compare_stacked_mnist_clean_ae_ldm.py \
    --seed "${SEED}" \
    --stage1-noise-mode vp_schedule \
    --stage2-t-min 0.001
