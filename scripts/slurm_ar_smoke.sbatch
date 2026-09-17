#!/bin/bash
#SBATCH --job-name=channeltok-ar-smoke
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

PYTHON="${REPO_ROOT}/.venv/bin/python"
CHECKPOINT="${CHECKPOINT:-/cmlscratch/sukriti5/mask-git-juhu/ckpt_runpod/imagenet_bsq_large_twostage_gan_continued/last.ckpt}"
TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-${REPO_ROOT}/configs/exp/imagenet_bsq_large_continued.yaml}"
AR_CONFIG="${AR_CONFIG:-${REPO_ROOT}/configs/generation/smoke_65k.yaml}"
DATA_ROOT="${DATA_ROOT:-/fs/cml-datasets/ImageNet/ILSVRC2012}"
LATENT_DIR="${LATENT_DIR:-${REPO_ROOT}/latents_data/ar_smoke_65k}"
NUM_EXTRACT_SAMPLES="${NUM_EXTRACT_SAMPLES:-256}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing ${PYTHON}. Run 'uv sync --python 3.10' first." >&2
    exit 1
fi

for required in "${CHECKPOINT}" "${TOKENIZER_CONFIG}" "${AR_CONFIG}"; do
    if [[ ! -f "${required}" ]]; then
        echo "Required file not found: ${required}" >&2
        exit 1
    fi
done

if [[ ! -d "${DATA_ROOT}/val" ]]; then
    echo "ImageNet validation directory not found: ${DATA_ROOT}/val" >&2
    exit 1
fi

cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/channeltok-ar-${SLURM_JOB_ID:-manual}}"
export WANDB_MODE=disabled
mkdir -p "${TMPDIR}" "${LATENT_DIR}"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Host: $(hostname)"
echo "Latent directory: ${LATENT_DIR}"
nvidia-smi

if [[ ! -f "${LATENT_DIR}/val_latents.npz" ]]; then
    echo "Extracting ${NUM_EXTRACT_SAMPLES} tokenizer sequences from ImageNet validation..."
    srun --ntasks=1 "${PYTHON}" -m tokenizer.extract_latents \
        --checkpoint "${CHECKPOINT}" \
        --config "${TOKENIZER_CONFIG}" \
        --output_dir "${LATENT_DIR}" \
        --splits val \
        --data_dir "${DATA_ROOT}" \
        --batch_size 16 \
        --num_workers 8 \
        --num_samples "${NUM_EXTRACT_SAMPLES}" \
        --device cuda
else
    echo "Reusing existing ${LATENT_DIR}/val_latents.npz"
fi

echo "Launching a fresh AR model (no --gpt-ckpt)..."
srun --ntasks=1 "${PYTHON}" -m generation.train --config "${AR_CONFIG}"

echo "AR smoke test complete."
