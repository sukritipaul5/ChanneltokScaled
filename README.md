# ChannelTok: Efficient Flexible-Length Vision Tokenization

<p align="center">
  <a href="https://channeltok.github.io">Website</a> |
  <a href="https://arxiv.org/abs/2606.04461">arXiv</a> |
  <a href="#citation">BibTeX</a>
</p>

<!-- TODO: Add results figure -->
<!-- ![ChannelTok results](./assets/channeltok_results.png) -->

## Overview

ChannelTok treats each latent channel as a visual token, enabling:
- **Flexible compression**: retain first k channels at inference for quality-speed tradeoff
- **Coarse-to-fine hierarchy**: early channels encode global structure, later channels refine details
- **Variable-length AR generation**: channel ordering maps directly to autoregressive factorization

Our model achieves **rFID 2.92** while being **8.6x faster** in decoding and **2.1x smaller** (159M params) than the next-best flexible tokenizer.

Architecture: CNN-Transformer hybrid encoder-decoder with Binary Spherical Quantization (BSQ).

## Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/sukritipaul5/Channeltok.git
cd Channeltok
uv sync
```

## Quick Start

### Inference / Evaluation

```bash
# Evaluate ChannelTok (flexible) at 256 tokens
python -m tokenizer.evaluate \
  --checkpoint checkpoints/channeltok/best.ckpt \
  --config configs/tokenizer/channeltok_flex.yaml \
  --data_dir /path/to/imagenet \
  --inference_t 256

# Multi-budget evaluation (32, 64, 128, 256, 512 tokens)
python -m tokenizer.evaluate_quantizer \
  --ckpt checkpoints/channeltok/best.ckpt \
  --config configs/tokenizer/channeltok_flex.yaml \
  --data_dir /path/to/imagenet \
  --budgets 32 64 128 256 512
```

### Tokenizer Training

```bash
# Single node, multi-GPU
torchrun --nproc_per_node=8 -m tokenizer.train \
  --config configs/tokenizer/channeltok_flex.yaml \
  --data_dir /path/to/imagenet
```

### Autoregressive Generation

The AR pipeline uses pre-extracted token indices from the tokenizer.

```bash
# Step 1: Extract latents
python -m tokenizer.extract_latents \
  --checkpoint checkpoints/channeltok/best.ckpt \
  --config configs/tokenizer/channeltok_flex.yaml \
  --data_dir /path/to/imagenet \
  --output_dir ./latents_data/imagenet

# Step 2: Train AR model (GPT-L) on extracted latents
torchrun --nproc_per_node=8 -m generation.train \
  --config configs/generation/gpt_l.yaml

# Step 3: Generate images
python -m generation.sample \
  --config configs/generation/gpt_l.yaml \
  --ckpt checkpoints/generation/gpt_l/latest.pt
```

## Project Structure

```
models/                  # Model definitions
  vqgan_ae.py            # Encoder, Decoder, VQGANAutoencoder
  helper.py              # ResidualBlock, NonLocalBlock, etc.
  fsq.py, lfq.py         # Quantizers (FSQ, LFQ/BSQ)
  latent_mask_head.py     # Adaptive channel masking
  discriminator.py        # PatchGAN discriminator

tokenizer/               # Tokenizer training, evaluation, extraction
  train.py               # Training script
  evaluate.py            # Reconstruction evaluation (PSNR, LPIPS, rFID)
  evaluate_quantizer.py  # Multi-budget quantizer evaluation
  extract_latents.py     # Latent extraction for AR training

generation/              # Autoregressive image generation (LlamaGen)
  train.py               # AR model training (DDP)
  sample.py              # Variable-length image generation
  decode_utils.py        # Token-to-image reconstruction
  models/gpt.py          # GPT transformer architecture
  models/generate.py     # Sampling utilities (top-k, nucleus)
  dataset/imagenet_npz.py # Pre-extracted latent dataset loader

utils/                   # Shared utilities
  checkpoint.py          # Config and checkpoint loading
  latent_extraction.py   # Extraction logic
  losses.py              # Loss functions
  lr_schedule.py         # Learning rate schedulers

data.py                  # Data loader factory (ImageFolder, DALI/WebDataset)

configs/
  tokenizer/             # Tokenizer configs (channeltok_flex, channeltok_base)
  generation/            # AR generation configs (GPT-B, GPT-L)
  quantizers/            # Quantizer ablation configs (BSQ, FSQ, LFQ)
  scale/                 # Model scale configs
  sampling/              # Sampling strategy configs
```

## Data Loading

Two data formats are supported:

**ImageFolder** (default) — standard torchvision format, works everywhere:
```bash
python -m tokenizer.evaluate --data_dir /path/to/imagenet ...
# Directory should contain train/ and val/ subdirs with class folders
```

**WebDataset + DALI** — GPU-accelerated, for high-throughput training:
```bash
# Requires: uv sync --extra dali
export IMAGENET_WDS_ROOT=/path/to/imagenet-1k-wds/
python -m tokenizer.evaluate --dataset imagenet_wds ...
```

All CLI scripts accept `--dataset {imagefolder,imagenet_wds}` and `--data_dir` overrides.

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
