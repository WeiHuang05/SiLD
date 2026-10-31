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

echo "=== Starting CelebA-HQ U-Net v3 experiment ==="
echo "Date: $(date)"
echo "Recipe: v2 + Min-SNR-gamma=5 loss weighting + t0=0.05."
echo "Fresh training from scratch (separate v3 checkpoints from v2)."

python -u experiments/compare_celebahq_unet_v3.py

echo "=== Done ==="
echo "Date: $(date)"
