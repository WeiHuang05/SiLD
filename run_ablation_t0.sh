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

echo "=== Starting t0 ablation on MultiManifoldLifted ==="
echo "Date: $(date)"
echo "Sweep t0 in {0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2}"
echo "Metrics: distance to true manifold + mode-coverage KL + recon MSE."

python -u experiments/ablation_t0.py

echo "=== Done ==="
echo "Date: $(date)"
