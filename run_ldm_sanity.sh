#! /bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gtn-container_g1
#$ -ac d=nvcr-pytorch-2311
#$ -m e
#$ -M wei.huang.vr@riken.jp

set -euo pipefail
source /home/weihuang/mamba/etc/profile.d/conda.sh
conda activate diffusion
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
cd /home/weihuang/diffusion_latent

echo "=== VAE-LDM baseline sanity: ${DS:?DS}  kl=${KL:?KL}  seed=${SD:?SD} ==="
echo "Date: $(date)"
python -u experiments/sweep_ldm_kl_selfies.py \
    --dataset "$DS" --kl-weight "$KL" --seed "$SD" \
    --n-eval 10000 --sample-steps 200 --stage2-t-min 0.001
echo "=== Done ==="
echo "Date: $(date)"
