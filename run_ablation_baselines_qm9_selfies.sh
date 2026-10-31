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
mkdir -p logs

echo "=== Ablation baselines: QM9 selfies (VAE-only + Latent GAN) ==="
echo "Date: $(date)"
python -u experiments/evaluate_ablation_baselines.py --dataset qm9 --representation selfies
echo "=== Done ==="
echo "Date: $(date)"
