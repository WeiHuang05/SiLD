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
export EXPERIMENT_VARIANT=celebahq_recent_baselines
export MPLCONFIGDIR=/tmp/matplotlib-recent-${JOB_ID}

cd /home/weihuang/diffusion_latent
mkdir -p \
    logs/celebahq_recent_baselines \
    checkpoints/celebahq_recent_baselines \
    figures/celebahq_recent_baselines

METHOD="${1:?usage: run_celebahq_recent_baseline.sh METHOD [SEED]}"
SEED="${2:-42}"
python -u experiments/compare_celebahq_recent_baselines.py \
    --method "${METHOD}" \
    --seed "${SEED}"
