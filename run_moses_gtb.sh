#! /bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -jc gtb-container_g4
#$ -ac d=nvcr-pytorch-2311
#$ -m e
#$ -M wei.huang.vr@riken.jp

source /home/weihuang/mamba/etc/profile.d/conda.sh
conda activate diffusion

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

cd /home/weihuang/diffusion_latent

echo "=== MOSES baselines on QM9/HIV/MUV/PCBA ==="
echo "Date: $(date)"
python -u experiments/moses_train_all.py --dataset all --model all --n_eval 10000
echo "=== Done ==="
echo "Date: $(date)"
