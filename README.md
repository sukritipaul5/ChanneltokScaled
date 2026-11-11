# ChannelTok

ChannelTok represents an image as an ordered sequence of latent channels. Early tokens capture global structure; later tokens add detail, so the same tokenizer supports flexible compression and channel-wise autoregressive generation.

<p align="center">
  <a href="https://channeltok.github.io">website</a> ·
  <a href="https://arxiv.org/abs/2606.04461">paper</a> ·
  <a href="#citation">citation</a>
</p>

The validated full-scale path uses 512 tokens per image, a 65,536-code vocabulary, and pre-extracted ImageNet-1k latents. See [FULL_SCALE_AR_HANDOFF.md](FULL_SCALE_AR_HANDOFF.md) for exact artifacts, measured sizes, multi-node notes, and troubleshooting.

## Install

Python 3.10 and CUDA are recommended.

```bash
git clone https://github.com/sukritipaul5/ChanneltokScaled.git
cd ChanneltokScaled

curl -LsSf https://astral.sh/uv/install.sh | sh  # skip if uv is installed
uv sync --python 3.10
```

Commands below use `.venv/bin/python` and `.venv/bin/torchrun`. Activating the environment is optional:

```bash
source .venv/bin/activate
```

## Reproduce the full ImageNet AR setup

Two production configs are provided:

- `configs/generation/full_imagenet_512ch_gptl.yaml` — GPT-L, lower-risk starting point
- `configs/generation/full_imagenet_512ch_gptxl.yaml` — GPT-XL, larger primary target

Both configs contain the validated UMIACS paths for the latent directory, tokenizer checkpoint, and tokenizer config. Change the three marked paths if the files are mounted elsewhere.

Set `WANDB_API_KEY` for logging, or set `wandb_enabled: false` in the selected config.

### 1. Check memory on one GPU

Run a short precheck before requesting a large allocation:

```bash
.venv/bin/python -m generation.train \
  --config configs/generation/full_imagenet_512ch_gptl.yaml \
  --global-batch-size 8 \
  --max-train-steps 20
```

Watch allocated/reserved GPU memory and `nvidia-smi`, then adjust the batch. Repeat independently for GPT-XL.

### 2. Run on one multi-GPU node

```bash
.venv/bin/torchrun --nproc_per_node=8 \
  -m generation.train \
  --config configs/generation/full_imagenet_512ch_gptl.yaml
```

`global_batch_size / world_size` is the batch loaded by each rank. Recompute it whenever the GPU count changes; use gradient accumulation to increase effective batch size without increasing peak memory.

### 3. Validate multi-node launch

Edit the scheduler and GRES directives in the template, then run its bounded two-node demo:

```bash
sbatch scripts/slurm_multinode_template.sbatch
```

A passing log shows distinct global ranks, different hostnames, NCCL initialization on every node, finite loss/accuracy, rank-0-only checkpoint setup, and a clean exit. After validation, adapt the template to the final allocation and set `MAX_TRAIN_STEPS=none`.

## Extract latents

Latents are generated with `tokenizer.extract_latents`. The included SLURM job is the reference full-ImageNet extraction:

```bash
sbatch scripts/slurm_extract_full_imagenet.sbatch
```

Update its checkpoint, ImageNet root, output directory, and scheduler directives first. The output directory must contain `train_latents.npz` and `val_latents.npz` for AR training.

## Tokenizer evaluation

ImageNet should use torchvision `ImageFolder` layout:

```text
/path/to/imagenet/
├── train/<class>/*.JPEG
└── val/<class>/*.JPEG
```

Evaluate a checkpoint at one channel budget:

```bash
.venv/bin/python -m tokenizer.evaluate \
  --checkpoint /path/to/tokenizer.ckpt \
  --config /path/to/tokenizer.yaml \
  --data_dir /path/to/imagenet \
  --inference_t 512
```

Evaluate several budgets:

```bash
.venv/bin/python -m tokenizer.evaluate_quantizer \
  --ckpt /path/to/tokenizer.ckpt \
  --config /path/to/tokenizer.yaml \
  --data_dir /path/to/imagenet \
  --budgets 64 128 256 512
```

## Tokenizer training

Set `data.data_dir` and `output_dir` in `configs/tokenizer/channeltok_flex.yaml`, then run:

```bash
.venv/bin/torchrun --nproc_per_node=8 \
  -m tokenizer.train \
  --config configs/tokenizer/channeltok_flex.yaml
```

## Repository map

```text
tokenizer/    tokenizer training, evaluation, and latent extraction
generation/   autoregressive training, sampling, and distributed setup
models/       tokenizer architecture and quantizers
configs/      tokenizer, generation, scale, and experiment configs
scripts/      SLURM jobs and evaluation utilities
utils/        checkpoint, extraction, loss, and schedule helpers
```

Optional WebDataset/DALI support is installed with `uv sync --extra dali`. Standard ImageFolder loading requires no optional dependencies.

## Citation

```bibtex
@article{paul2026channeltok,
  title={ChannelTok: Efficient Flexible-Length Vision Tokenization},
  author={Paul, Sukriti and Bansal, Arpit and Goldstein, Tom},
  year={2026},
  eprint={2606.04461},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
