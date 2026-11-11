# Full-scale 512-channel ImageNet AR handoff

This repository now contains a validated channel-wise autoregressive training path for full ImageNet-1k using 512 channel tokens per image and a 65,536-code tokenizer vocabulary. The required fix was merged in [`b3bdd9c`](https://github.com/sukritipaul5/ChanneltokScaled/commit/b3bdd9cb1120bc4a439c39f504d593ce3d0220e0) after a 3,000-step, 4×A5000 GPT-B run demonstrated stable learning. At this vocabulary size, the old model used separate token-embedding and output matrices and unconditionally copied the full `(batch, sequence, vocabulary)` logits tensor to fp32. The merged implementation ties the output projection to the token embedding and leaves logits in the active autocast dtype. It intentionally removes the old zero-initialized final-layer behavior because zeroing a tied output matrix would also zero the token embeddings. This makes the checkpoint architecture different from a vanilla, untied LlamaGen checkpoint.

## Prerequisites and the open latent-delivery decision

The frozen 512-channel tokenizer checkpoint used for extraction and visualization is:

```text
/cmlscratch/sukriti5/mask-git-juhu/ckpt_runpod/imagenet_bsq_large_twostage_gan_continued/last.ckpt
```

Its matching config is `configs/exp/imagenet_bsq_large_continued.yaml`. The AR input directory must contain `train_latents.npz` and `val_latents.npz` produced at 512 channels. Each row contains 512 `uint16` code indices; the code range is 0–65,535, while BOS/EOS/PAD are added by the dataset loader as IDs 65,536/65,537/65,538.

How those NPZ files will reach the collaborator is intentionally unresolved and must be decided with him directly:

1. Ship the already-extracted `train_latents.npz` and `val_latents.npz` files directly.
2. Have him regenerate them against his own ImageNet copy using `tokenizer/extract_latents.py`, the tokenizer checkpoint above, and `configs/exp/imagenet_bsq_large_continued.yaml`. `scripts/slurm_extract_full_imagenet.sbatch` is a working single-node example, but its paths and SLURM directives are cluster-specific.

After choosing either route, replace `npz_path: "/path/to/your/imagenet_full_512ch"` in the selected production config.

## Model choices

- `configs/generation/full_imagenet_512ch_gptl.yaml` uses GPT-L. GPT-L is the paper-validated architecture, and this repository's bug-fixed GPT-B quick test has already shown that the same training path learns correctly with the exact 512-channel, 65K-vocabulary representation.
- `configs/generation/full_imagenet_512ch_gptxl.yaml` uses GPT-XL. GPT-XL is the recommended primary target because this run has roughly ten times more images than the paper's AR experiment and substantially more compute is available; that recommendation follows LlamaGen's published model-scaling trend.

This exact combination—channel tokens, full ImageNet-1k, 512 channels, and a 65K code vocabulary—has not been run before at any model size. If the GPU budget permits, run GPT-L and GPT-XL in parallel as a hedge rather than choosing one blindly. Benchmark memory and throughput independently for each model; GPT-XL cannot inherit GPT-L's batch-size result by assumption.

## Launching

For a single multi-GPU node, replace `<N>` with the number of GPUs visible to the job:

```bash
torchrun --nproc_per_node=<N> -m generation.train \
  --config configs/generation/full_imagenet_512ch_gptl.yaml
```

or:

```bash
torchrun --nproc_per_node=<N> -m generation.train \
  --config configs/generation/full_imagenet_512ch_gptxl.yaml
```

Multi-node training requires the standard `torchrun` rendezvous arguments, including `--nnodes`, `--node_rank`, and `--rdzv_endpoint` (and normally `--rdzv_id`). Their concrete values and environment setup depend on the collaborator's scheduler; this repository does not configure that cluster-specific orchestration.

## Memory-fit precheck on new hardware

Do not commit immediately to the long B100 allocation with the shipped batch defaults. First run approximately 20 optimizer steps for the chosen model, for example by adding `--max-train-steps 20` to the launch command. Check the training loop's logged allocated/reserved GPU memory and independently sample `nvidia-smi` during forward, backward, and optimizer steps. Adjust `global_batch_size` until each rank has comfortable headroom, then use `gradient_accumulation_steps` to recover the desired effective batch size. The effective batch is:

```text
global_batch_size × gradient_accumulation_steps
```

Repeat this precheck independently for GPT-L and GPT-XL. The validated A5000 workflow used exactly this process: a one-GPU, per-rank batch precheck exposed the real memory profile before scaling to the four-GPU job, and the resulting full run completed without OOM. Repeating it on B100 is cheaper than assuming the shipped defaults are correct.

## Throughput and runtime extrapolation

After the memory-fit test, collect steady-state optimizer steps per second from a few hundred real steps. Ignore the first logs, which include process startup, checkpoint/decoder loading, compilation, and warmup. Compute:

```text
effective_batch_size = global_batch_size × gradient_accumulation_steps
images_per_second = steady_state_steps_per_second × effective_batch_size
seconds_per_epoch = number_of_training_images / images_per_second
total_seconds = seconds_per_epoch × epochs
```

Use the actual training image count in the delivered or regenerated NPZ. Treat `epochs: 20` as a pre-benchmark starting estimate, not gospel; re-derive the expected epoch and total runtime from the B100 measurement before reserving the long job.

## Resuming after interruption

Set these fields in the selected config:

```yaml
gpt_ckpt: "/path/to/results_dir/<run>/checkpoints/latest.pt"
resume_same_folder: true
```

Then relaunch with the same distributed command. This follows the resume path already implemented in `generation/train.py`: it restores model and optimizer state (plus EMA and scheduler state when present) and continues in the checkpoint's existing experiment directory.
