#!/bin/bash
#SBATCH --job-name=channeltok-ar-decode
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
else
    REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"
SCRIPT="${REPO_ROOT}/scripts/decode_quicktest_samples.py"

for required in "${PYTHON}" "${SCRIPT}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Required path not found: ${required}" >&2
        exit 1
    fi
done

cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Host: $(hostname)"
nvidia-smi
"${PYTHON}" "${SCRIPT}"
