#!/bin/bash
#SBATCH --job-name=channeltok-ar-precheck
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
else
    REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"
CONFIG="${REPO_ROOT}/configs/generation/quick_test_full512.yaml"
LATENT_DIR="/fs/cml-projects/gaze/channeltok_latents/imagenet_full_512ch"
DECODER_CHECKPOINT="/cmlscratch/sukriti5/mask-git-juhu/ckpt_runpod/imagenet_bsq_large_twostage_gan_continued/last.ckpt"

for required in "${PYTHON}" "${CONFIG}" \
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
export TMPDIR="${SLURM_TMPDIR:-/tmp/${USER}/channeltok-ar-precheck-${SLURM_JOB_ID:-manual}}"
mkdir -p "${TMPDIR}"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Host: $(hostname)"
echo "Config: ${CONFIG}"
nvidia-smi

monitor_gpu_memory() {
    while true; do
        date '+GPU_MEMORY %Y-%m-%d %H:%M:%S'
        nvidia-smi \
            --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits
        sleep 5
    done
}

monitor_gpu_memory &
GPU_MONITOR_PID=$!
trap 'kill "${GPU_MONITOR_PID}" 2>/dev/null || true' EXIT

set +e
"${PYTHON}" -m generation.train \
    --config "${CONFIG}" \
    --global-batch-size 16 \
    --max-train-steps 20
TRAIN_STATUS=$?
set -e

kill "${GPU_MONITOR_PID}" 2>/dev/null || true
wait "${GPU_MONITOR_PID}" 2>/dev/null || true
trap - EXIT

if [[ ${TRAIN_STATUS} -ne 0 ]]; then
    exit "${TRAIN_STATUS}"
fi

echo "AR quick-test precheck complete."
