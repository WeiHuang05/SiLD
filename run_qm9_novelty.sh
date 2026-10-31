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

echo "=== QM9 SELFIES ULD — Novelty Boost Sampling ==="
echo "Date: $(date)"
python -u experiments/qm9_selfies_novelty_boost.py --n_eval 10000
echo "=== Done ==="
echo "Date: $(date)"
