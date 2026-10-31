#! /bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -jc gtb-container_g1
#$ -ac d=nvcr-pytorch-2311
#$ -m e
#$ -M wei.huang.vr@riken.jp

source /home/weihuang/mamba/etc/profile.d/conda.sh
conda activate diffusion

export PYTHONUNBUFFERED=1
export EXPERIMENT_VARIANT=rebuttal_toy_no_svd
export MPLCONFIGDIR=/tmp/matplotlib-toy-${JOB_ID}

cd /home/weihuang/diffusion_latent
mkdir -p logs/rebuttal_toy_no_svd figures/rebuttal_toy_no_svd

REGULARIZER="${1:?usage: run_toy_algorithm1_no_svd.sh REGULARIZER WEIGHT [SEED]}"
WEIGHT="${2:?usage: run_toy_algorithm1_no_svd.sh REGULARIZER WEIGHT [SEED]}"
SEED="${3:-42}"

python -u experiments/toy_decomposed_score.py \
    --regularizer "${REGULARIZER}" \
    --reg-weight "${WEIGHT}" \
    --seed "${SEED}"
