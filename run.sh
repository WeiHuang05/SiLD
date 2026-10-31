#! /bin/bash

#S -S /bin/bash
#$ -cwd
#$ -jc gtn-container_g1_dev
#$ -ac d=nvcr-pytorch-2311
#$ -m e
#$ -M wei.huang.vr@riken.jp

source /home/weihuang/mamba/etc/profile.d/conda.sh
conda activate diffusion

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

# -----------------------------------------------------------------------
# Diffusion on Low-Dimensional Manifolds: Two-Stage vs Latent Diffusion
# -----------------------------------------------------------------------
cd /home/weihuang/diffusion_latent

mkdir -p logs figures

echo "=== Starting diffusion manifold experiments ==="
echo "Date: $(date)"

# Run experiments: 'all', 'linear', or 'nonlinear'
python -u run_all.py ${1:-image}

echo "=== Done ==="
echo "Date: $(date)"
