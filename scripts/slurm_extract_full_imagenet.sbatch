#!/bin/bash
#SBATCH --job-name=channeltok-extract-full-imagenet
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
else
    REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"
CHECKPOINT="/fs/cml-projects/gaze/channeltok_ckpts/last.ckpt"
TOKENIZER_CONFIG="${REPO_ROOT}/configs/exp/imagenet_bsq_large_continued.yaml"
DATA_ROOT="/fs/cml-datasets/ImageNet/ILSVRC2012"
OUTPUT_DIR="/fs/cml-projects/gaze/channeltok_latents/imagenet_full_512ch"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing ${PYTHON}. Run 'uv sync --python 3.10' first." >&2
    exit 1
fi

for required in "${CHECKPOINT}" "${TOKENIZER_CONFIG}"; do
    if [[ ! -f "${required}" ]]; then
        echo "Required file not found: ${required}" >&2
        exit 1
    fi
done

for split in train val; do
    if [[ ! -d "${DATA_ROOT}/${split}" ]]; then
        echo "ImageNet split directory not found: ${DATA_ROOT}/${split}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"
if [[ ! -w "${OUTPUT_DIR}" ]]; then
    echo "Output directory is not writable: ${OUTPUT_DIR}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/channeltok-extract-${SLURM_JOB_ID:-manual}}"
mkdir -p "${TMPDIR}"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Host: $(hostname)"
echo "Repository: ${REPO_ROOT}"
echo "Output directory: ${OUTPUT_DIR}"
nvidia-smi

srun --ntasks=1 "${PYTHON}" -m tokenizer.extract_latents \
    --checkpoint "${CHECKPOINT}" \
    --config "${TOKENIZER_CONFIG}" \
    --data_dir "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --splits val train \
    --inference_t 512 \
    --batch_size 128 \
    --num_workers 16 \
    --device cuda

echo "Full ImageNet latent extraction complete."
