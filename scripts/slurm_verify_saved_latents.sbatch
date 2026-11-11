#!/bin/bash
#SBATCH --job-name=channeltok-verify-latents
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa5000:4
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
CHECKPOINT="/cmlscratch/sukriti5/mask-git-juhu/ckpt_runpod/imagenet_bsq_large_twostage_gan_continued/last.ckpt"
CONFIG="${REPO_ROOT}/configs/exp/imagenet_bsq_large_continued.yaml"
DATA_ROOT="/fs/cml-datasets/ImageNet/ILSVRC2012"
LATENTS="/fs/cml-projects/gaze/channeltok_latents/imagenet_full_512ch/val_latents.npz"
OUTPUT_DIR="/fs/cml-projects/gaze/channeltok_latents/imagenet_full_512ch/reconstruction_verification"

for required in "${PYTHON}" "${CHECKPOINT}" "${CONFIG}" "${LATENTS}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Required path not found: ${required}" >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/channeltok-verify-${SLURM_JOB_ID:-manual}}"
mkdir -p "${TMPDIR}"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Host: $(hostname)"
echo "Output directory: ${OUTPUT_DIR}"
nvidia-smi

srun --ntasks=1 "${PYTHON}" -m tokenizer.verify_saved_latents \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --data_dir "${DATA_ROOT}" \
    --latents "${LATENTS}" \
    --output_dir "${OUTPUT_DIR}" \
    --sample_indices 0 10000 20000 30000 40000 49999 \
    --extraction_batch_size 128 \
    --device cuda

echo "Saved-latent reconstruction verification complete."
