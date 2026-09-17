#!/bin/bash
#SBATCH --job-name=channeltok-reconstruct
#SBATCH --partition=cml-scavenger
#SBATCH --account=cml-scavenger
#SBATCH --qos=cml-scavenger
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
else
    REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
fi
CHECKPOINT="${CHECKPOINT:-/cmlscratch/sukriti5/mask-git-juhu/ckpt_runpod/imagenet_bsq_large_twostage_gan_continued/last.ckpt}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/exp/imagenet_bsq_large_continued.yaml}"
DATA_ROOT="${DATA_ROOT:-/fs/cml-datasets/ImageNet/ILSVRC2012}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/inference_output}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
INFERENCE_T="${INFERENCE_T:-512}"
OUTPUT_DIR="${OUTPUT_ROOT}/imagenet_bsq_last_t${INFERENCE_T}_offset${SAMPLE_OFFSET}_${SLURM_JOB_ID:-manual}"

if [[ ! -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    echo "Missing ${REPO_ROOT}/.venv. Run 'uv sync --python 3.10' first." >&2
    exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi

if [[ ! -d "${DATA_ROOT}/val" ]]; then
    echo "ImageNet validation directory not found: ${DATA_ROOT}/val" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/channeltok-${SLURM_JOB_ID:-manual}}"
mkdir -p "${TMPDIR}"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Host: $(hostname)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Data root: ${DATA_ROOT}"
echo "Output directory: ${OUTPUT_DIR}"
nvidia-smi

srun "${REPO_ROOT}/.venv/bin/python" -m tokenizer.evaluate \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --data_dir "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_samples "${NUM_SAMPLES}" \
    --sample_offset "${SAMPLE_OFFSET}" \
    --batch_size "${NUM_SAMPLES}" \
    --inference_t "${INFERENCE_T}" \
    --save_images \
    --device cuda

echo "Inference complete: ${OUTPUT_DIR}"
