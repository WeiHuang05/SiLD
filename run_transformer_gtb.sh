#! /bin/bash

#$ -S /bin/bash
#$ -cwd
#$ -jc gtb-container_g8
#$ -ac d=nvcr-pytorch-2311
#$ -m e
#$ -M wei.huang.vr@riken.jp

source /home/weihuang/mamba/etc/profile.d/conda.sh
conda activate diffusion

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

cd /home/weihuang/diffusion_latent

echo "=== Transformer AR on HIV / MUV / PCBA (SELFIES) ==="
echo "Date: $(date)"

for ds in hiv muv pcba; do
    echo "--- $ds (SELFIES) ---"
    python -u experiments/qm9_transformer_ar.py --dataset $ds --representation selfies --n_steps 40000 --n_eval 10000
done

echo "=== Done ==="
echo "Date: $(date)"
