#!/bin/bash
# Two-node/one-GPU-per-node demo template. These three scheduler values
# are from Sukriti's UMIACS validation; change them for the target cluster.
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=high
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
# UMIACS validation GRES; change this to the collaborator's B100 GRES string.
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=channeltok-multinode-demo
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
PYTHON="${REPO_ROOT}/.venv/bin/python"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/generation/full_imagenet_512ch_gptl.yaml}"

# global_batch_size is divided by WORLD_SIZE before each rank builds its loader.
# For this two-rank demo, 32 means 16 images/GPU. Do not reuse a batch
# chosen for a different world size without recomputing the per-rank batch.
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
# Set MAX_TRAIN_STEPS=none after validation to run according to config epochs.
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-50}"

export REPO_ROOT PYTHON CONFIG GLOBAL_BATCH_SIZE MAX_TRAIN_STEPS
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_IB_DISABLE=1
export NCCL_CROSS_NIC=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_PORT=$(expr 10000 + $(echo -n "$SLURM_JOBID" | tail -c 4))

echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_JOB_NODELIST}"
srun --label bash -c 'echo "host=$(hostname) rank=${SLURM_PROCID} world_size=${SLURM_NTASKS} local_rank=${SLURM_LOCALID}"'

# Use srun, not torchrun, to exercise generation/utils/distributed.py's
# SLURM_PROCID branch. CUDA visibility must be assigned inside each task.
srun --label --kill-on-bad-exit=1 bash -c '
    cd "$REPO_ROOT"
    export CUDA_VISIBLE_DEVICES="$SLURM_LOCALID"
    train_args=(
        -m generation.train
        --config "$CONFIG"
        --global-batch-size "$GLOBAL_BATCH_SIZE"
    )
    if [[ "$MAX_TRAIN_STEPS" != "none" ]]; then
        train_args+=(--max-train-steps "$MAX_TRAIN_STEPS")
    fi
    exec "$PYTHON" "${train_args[@]}"
'
