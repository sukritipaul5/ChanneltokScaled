# Full-scale 512-channel ImageNet AR handoff

This is a validated channel-wise autoregressive training path for full ImageNet-1k with 512 channel tokens per image and a 65,536-code vocabulary. A required large-vocabulary fix was validated with a 3,000-step GPT-B run and merged in [`b3bdd9c`](https://github.com/sukritipaul5/ChanneltokScaled/commit/b3bdd9cb1120bc4a439c39f504d593ce3d0220e0).

## What exists and where

These are measured on Sukriti's UMIACS environment. File sizes include both the exact byte count from `ls -la` and allocated size from `du -h`; directory rows report total `du -h` usage.

| Artifact | UMIACS path | Measured size | Details |
|---|---|---:|---|
| Repository root | `/fs/cml-projects/gaze/ChanneltokScaled` | 15G directory usage | Source checkout used for all validation runs. |
| Tokenizer checkpoint | `/cmlscratch/sukriti5/mask-git-juhu/ckpt_runpod/imagenet_bsq_large_twostage_gan_continued/last.ckpt` | 1,945,272,705 bytes; 1.9G allocated | Frozen 512-channel BSQ tokenizer/decoder. Required for either latent-delivery option. |
| Tokenizer config | `/fs/cml-projects/gaze/ChanneltokScaled/configs/exp/imagenet_bsq_large_continued.yaml` | 2,796 bytes; 8.0K allocated | Matching tokenizer architecture/configuration. |
| Latents directory | `/fs/cml-projects/gaze/channeltok_latents/imagenet_full_512ch` | 1.3G directory usage | Contains the train/validation NPZ files and local reconstruction-verification artifacts. |
| Training latents | `/fs/cml-projects/gaze/channeltok_latents/imagenet_full_512ch/train_latents.npz` | 1,318,496,079 bytes; 1.3G allocated | `indices`: `(1,281,152, 512)` `uint16`; `labels`: `(1,281,152)` `int64`. |
| Validation latents | `/fs/cml-projects/gaze/channeltok_latents/imagenet_full_512ch/val_latents.npz` | 51,336,536 bytes; 49M allocated | `indices`: `(50,000, 512)` `uint16`; `labels`: `(50,000)` `int64`. |
| Validated GPT-B checkpoint | `/fs/cml-projects/gaze/ChanneltokScaled/results_full512_quicktest/004-GPT-B/checkpoints/latest.pt` | 1,626,634,862 bytes; 1.6G allocated | Reference artifact from the successful 3,000-step run; not required for the collaborator's training. |
| Decode sample grid | `/fs/cml-projects/gaze/ChanneltokScaled/results_full512_quicktest/quicktest_samples_grid.png` | 4,018,363 bytes; 3.9M allocated | Eight samples decoded at 64/128/256/512-token budgets. |
| Multi-node validation log | `/nfshomes/sukriti5/channeltok-multinode-8356534.out` | 28,144 bytes; 24K allocated | Successful two-node, one-A5000-per-node SLURM/NCCL validation. |

The training NPZ contains 1,281,152 rows, which is 15 fewer than the canonical ImageNet-1k training count of 1,281,167. That is the exact artifact validated here and should be checked explicitly after either transfer or regeneration.

**The tokenizer checkpoint is required in both delivery scenarios.** `generation/train.py` loads `decoder_ckpt` unconditionally, even when visualization is infrequent or disabled, so receiving precomputed NPZ files does not remove the decoder-checkpoint requirement. All UMIACS paths in the production configs are marked for replacement after the corresponding files are placed on the collaborator's infrastructure.

## Open decision: how the collaborator receives the latents

This is intentionally unresolved and should be decided directly with the collaborator:

1. Transfer `train_latents.npz` and `val_latents.npz` from the directory above.
2. Regenerate them with `tokenizer/extract_latents.py` against the collaborator's own ImageNet copy, using the tokenizer checkpoint and matching config above. `scripts/slurm_extract_full_imagenet.sbatch` is a working UMIACS example, but its paths and scheduler directives are site-specific.

Whichever option is chosen, verify array shape, dtype, labels length, code range, and exact image count before starting AR training.

## Reproduction before the full run

This sequence is how confidence in the code was established here, step by step. It is not ceremonial: new hardware, networking, world size, and model size can each expose a different failure, so do not skip directly to a multi-day run.

### 1. Single-GPU memory-fit precheck

Start with one GPU, a deliberately small per-GPU batch, and approximately 20 optimizer steps:

```bash
python -m generation.train \
  --config configs/generation/full_imagenet_512ch_gptl.yaml \
  --global-batch-size <safe-single-GPU-batch> \
  --max-train-steps 20
```

Use the GPT-XL config when testing GPT-XL. Watch both the training loop's allocated/reserved memory log and `nvidia-smi` through forward, backward, and optimizer steps. Preserve comfortable headroom rather than merely avoiding OOM once.

### 2. Two-node SLURM/NCCL smoke test

If the real run is multi-node, reproduce the validated two-node/one-GPU-per-node test before scaling the allocation:

```bash
sbatch scripts/slurm_multinode_template.sbatch
```

Change its partition/account/QoS/GRES and file paths for the target cluster. Confirm all of the following in the combined log:

- two different physical hostnames;
- distinct global ranks 0 and 1 with `world_size=2`;
- `NCCL ... Init COMPLETE` and connected socket rings on both nodes;
- only rank 0 creates the experiment directory;
- finite loss/accuracy metrics after backward and optimizer steps;
- clean checkpointing and exit on both ranks.

UMIACS job `8356534` passed these checks on `tron47` and `tron48`. With global batch 32 and two ranks, each A5000 processed 16 images; step 20 logged loss `11.1839`, accuracy `0.002`, and step 40 logged loss `11.1289`, accuracy `0.002`, before a clean 50-step exit.

### 3. Bounded learning run on the real hardware

Run roughly 3,000 optimizer steps using the intended model and hardware. The random-token cross-entropy reference is `ln(65536) ≈ 11.09` and chance token accuracy is about `1 / 65536 ≈ 0.0000153`. Require loss to move meaningfully below that baseline and accuracy to rise well above chance, rather than accepting only a non-crashing process.

The validation run here used GPT-B on 4×A5000: loss fell from `11.15` to `6.06`, token accuracy rose from `0.002` to `0.350`, and decoder samples showed a sane coarse-to-fine progression. That is the bar for establishing that optimization, data loading, distributed reductions, checkpointing, and decoding work together.

## GPT-L or GPT-XL

- `configs/generation/full_imagenet_512ch_gptl.yaml` uses GPT-L. GPT-L is the paper-validated architecture and is the lower-risk starting point; this repository's bug-fixed training path has already learned correctly at the exact 512-channel/65K-vocabulary representation with GPT-B.
- `configs/generation/full_imagenet_512ch_gptxl.yaml` uses GPT-XL. GPT-XL is the recommended primary target when compute permits because this experiment has roughly ten times more images than the paper's AR experiment, consistent with LlamaGen's published model-scaling trend.

This exact combination—channel tokens, full ImageNet-1k, 512 channels, and a 65K vocabulary—has not been run before at either size. If budget permits, run GPT-L and GPT-XL in parallel as a hedge rather than choosing one blindly. Memory-fit and throughput-test each model independently.

## Launch commands

First replace the three UMIACS paths marked in the chosen production config.

Single-node multi-GPU:

```bash
torchrun --nproc_per_node=<GPUS_ON_NODE> \
  -m generation.train \
  --config configs/generation/full_imagenet_512ch_gptl.yaml
```

Use `full_imagenet_512ch_gptxl.yaml` for GPT-XL.

Multi-node under SLURM should use one task per GPU (or the scheduler-specific equivalent) and `srun`, which exercises the repository's `SLURM_PROCID` initialization path. The provided template defaults to the bounded two-node smoke test:

```bash
CONFIG="$PWD/configs/generation/full_imagenet_512ch_gptl.yaml" \
GLOBAL_BATCH_SIZE=32 MAX_TRAIN_STEPS=50 \
sbatch scripts/slurm_multinode_template.sbatch
```

After the smoke test, adapt the template to the final node/GPU count and submit with `MAX_TRAIN_STEPS=none` for the production run. On a non-SLURM system, use standard `torchrun --nnodes`, `--node_rank`, `--rdzv_id`, and `--rdzv_endpoint` arguments supplied by that cluster's launcher.

## Memory tuning

`global_batch_size` is divided by `world_size` to obtain the batch loaded by each rank. The effective optimizer batch is:

```text
effective_batch_size = global_batch_size × gradient_accumulation_steps
```

Recompute both quantities whenever the GPU or world count changes. In the UMIACS multi-node validation, global batch 64 over two ranks meant 32 images/A5000 and OOM'd; global batch 32 meant 16 images/A5000 and passed. Once a safe per-rank batch is measured, increase `gradient_accumulation_steps` if a larger effective batch is desired without raising peak memory.

## Throughput and runtime extrapolation

Measure steady-state optimizer steps per second after a few hundred real steps, ignoring startup, compilation, decoder loading, and warmup. Then compute:

```text
effective_batch_size = global_batch_size × gradient_accumulation_steps
images_per_second = steady_state_steps_per_second × effective_batch_size
seconds_per_epoch = actual_number_of_training_rows / images_per_second
total_seconds = seconds_per_epoch × epochs
```

Use the actual NPZ row count, not the canonical ImageNet count by assumption. Treat the production configs' `epochs: 20` as a pre-benchmark estimate and re-derive the total runtime from the collaborator's B100 measurement.

## Resuming after interruption

Set the chosen config to:

```yaml
gpt_ckpt: "/path/to/results_dir/<run>/checkpoints/latest.pt"
resume_same_folder: true
```

Relaunch with the same distributed command and world size. `generation/train.py` restores model and optimizer state, plus EMA and scheduler state when present, and reuses the existing experiment directory.

## Multi-node NCCL troubleshooting

The working UMIACS settings are encoded in `scripts/slurm_multinode_template.sbatch`:

```bash
export NCCL_IB_DISABLE=1
export NCCL_CROSS_NIC=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_PORT=$(expr 10000 + $(echo -n "$SLURM_JOBID" | tail -c 4))
```

Each `srun` task also sets:

```bash
export CUDA_VISIBLE_DEVICES="$SLURM_LOCALID"
```

The initial UMIACS attempt failed with `ibv_reg_mr_iova2 ... Cannot allocate memory`; these tron nodes have no usable InfiniBand path, and `NCCL_IB_DISABLE=1` fixed initialization. Do not guess an interface name on a new cluster. First run `ip -4 addr show` on an allocated node and inspect NCCL's `Bootstrap`/`NET/Socket` lines. On tron, NCCL automatically selected `bond0` (`10.71.20.x`), so `NCCL_SOCKET_IFNAME` was not needed; set it only when inspection shows auto-selection is wrong.

If initialization succeeds but training OOMs, distinguish networking from per-rank batch size before changing NCCL. If initialization hangs, verify that both ranks received the same `MASTER_ADDR`, derived `MASTER_PORT`, and `WORLD_SIZE`, and that the chosen TCP interface is reachable between allocated hosts.

## Why the architecture changed from a vanilla LlamaGen checkpoint (optional background)

At a 65,539-entry model vocabulary (65,536 codes plus BOS/EOS/PAD), the original implementation had separate token-embedding and output-projection matrices and unconditionally converted the full `(batch, sequence, vocabulary)` logits tensor to fp32. The validated implementation ties `self.output.weight` to `self.tok_embeddings.weight` and leaves logits in the active autocast dtype, removing the redundant vocabulary matrix and the large transient fp32 logits copy.

The old final-layer zero initialization was removed intentionally: with tied weights it would also zero the token embeddings and break initialization. As a result, these checkpoints differ structurally and in initialization behavior from an untied vanilla LlamaGen checkpoint; external checkpoint-loading or conversion code must account for the tied embedding/output weight.
