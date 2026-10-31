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

echo "=== Starting QM9 SMILES experiment (molecule generation) ==="
echo "Date: $(date)"
echo "Goal: ULD vs VAE-LDM on character-level SMILES one-hot diffusion."
echo "Metrics: validity / uniqueness / novelty (via RDKit)."

python -u experiments/compare_qm9.py

echo "=== Done ==="
echo "Date: $(date)"
