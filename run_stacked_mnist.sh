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

cd /home/weihuang/diffusion_latent
mkdir -p logs figures checkpoints

SEED=${SEED:-42}
T0=${T0:-0.05}

echo "=== Starting Stacked MNIST experiment (1000-mode benchmark) ==="
echo "Date: $(date)"
echo "Seed: $SEED   t0: $T0"
echo "Goal: ULD vs VAE-LDM at matched CNN AE capacity. Mode-coverage out"
echo "      of 1000 + KL_unif + FID via torch-fidelity."

python -u experiments/compare_stacked_mnist.py --seed "$SEED" --t0 "$T0"

echo "=== Done ==="
echo "Date: $(date)"
