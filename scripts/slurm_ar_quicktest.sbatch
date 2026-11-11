#!/bin/bash
#SBATCH --job-name=channeltok-ar-quicktest
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
else
    REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"
TORCHRUN="${REPO_ROOT}/.venv/bin/torchrun"
CONFIG="${REPO_ROOT}/configs/generation/quick_test_full512.yaml"
LATENT_DIR="/fs/cml-projects/gaze/channeltok_latents/imagenet_full_512ch"
DECODER_CHECKPOINT="/fs/cml-projects/gaze/channeltok_ckpts/last.ckpt"

for required in "${PYTHON}" "${TORCHRUN}" "${CONFIG}" \
                "${LATENT_DIR}/train_latents.npz" "${LATENT_DIR}/val_latents.npz" \
                "${DECODER_CHECKPOINT}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Required path not found: ${required}" >&2
        exit 1
    fi
done

cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/channeltok-ar-quicktest-${SLURM_JOB_ID:-manual}}"
mkdir -p "${TMPDIR}"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Host: $(hostname)"
echo "Repository: ${REPO_ROOT}"
echo "Config: ${CONFIG}"
echo "Latents: ${LATENT_DIR}"
nvidia-smi

srun --ntasks=1 "${TORCHRUN}" --nproc_per_node=4 \
    -m generation.train \
    --config "${CONFIG}"

echo "AR quick test complete."
