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

echo "=== Starting Multi-Manifold Lifted experiment (ULD vs VAE-LDM) ==="
echo "Date: $(date)"
echo "Goal: synthetic K=3 disjoint Swiss rolls in R^128 — show ULD recovers"
echo "      the manifold (small dist to M, correct mode coverage) where VAE"
echo "      smears across the gaps."

python -u experiments/compare_multi_manifold.py

echo "=== Done ==="
echo "Date: $(date)"
