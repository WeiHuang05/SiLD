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
export EXPERIMENT_VARIANT="${EXPERIMENT_VARIANT:-matched_stage2}"

cd /home/weihuang/diffusion_latent
mkdir -p "logs/$EXPERIMENT_VARIANT" "figures/$EXPERIMENT_VARIANT" \
    "checkpoints/$EXPERIMENT_VARIANT"

echo "=== NeurIPS paper rerun with matched Stage-2 time ==="
echo "Task: ${TASK:?TASK must be set}"
echo "Variant: $EXPERIMENT_VARIANT"
echo "Date: $(date)"

case "$TASK" in
    toy)
        python -u experiments/toy_decomposed_score.py
        ;;
    stacked_42)
        python -u experiments/compare_stacked_mnist.py --seed 42 --t0 0.005
        ;;
    stacked_43)
        python -u experiments/compare_stacked_mnist.py --seed 43 --t0 0.005
        ;;
    stacked_44)
        python -u experiments/compare_stacked_mnist.py --seed 44 --t0 0.005
        ;;
    celeba_mmd)
        python -u experiments/compare_celeba_mmd_sweep.py
        ;;
    celeba_gan)
        python -u experiments/compare_celeba_gan.py
        ;;
    selfies_qm9|selfies_hiv|selfies_muv|selfies_pcba)
        dataset="${TASK#selfies_}"
        python -u experiments/compare_selfies.py --dataset "$dataset"
        ;;
    celebahq_small)
        python -u experiments/compare_celebahq.py --config small
        ;;
    celebahq_large)
        python -u experiments/compare_celebahq.py --config large
        ;;
    celebahq_large_10x)
        python -u experiments/compare_celebahq.py --config large_10x
        ;;
    aggregate)
        python -u experiments/aggregate_stacked_mnist.py
        python -u experiments/aggregate_paper_matched.py
        ;;
    *)
        echo "Unknown TASK: $TASK" >&2
        exit 2
        ;;
esac

echo "=== Completed: $TASK ==="
echo "Date: $(date)"
