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

echo "=== Starting CIFAR-10 v2 experiment (8x8x8 spatial latent + U-Net) ==="
echo "Date: $(date)"
echo "Goal: competitive CIFAR-10 FID for both ULD and LDM (ULD better)."
echo "Recipe: spatial latent + U-Net + GAN+LPIPS+EMA Stage 1 + EMA diff."

python -u experiments/compare_cifar_v2.py

echo "=== Done ==="
echo "Date: $(date)"
