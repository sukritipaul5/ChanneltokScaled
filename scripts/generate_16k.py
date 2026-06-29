#!/usr/bin/env python3
"""
Generate images from trained GPT-B 16K model.
Two modes:
  1. Unconditional: generate all 256 tokens from scratch
  2. Completion: given first N% of real tokens, generate the rest

Usage:
  cd /mnt/data/mask-git-juhu
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 uv run python scripts/generate_16k.py
"""

import torch
import numpy as np
import os

from PIL import Image
from generation.decode_utils import reconstruct_from_channelwise_indices, denorm_to_uint8
from generation.train import load_decoder_model
from generation.models.gpt import GPT_models
from utils.checkpoint import load_config as load_decoder_config


# ---- Config ----
CKPT_PATH = "llamagen_flextok/results_imagenet100k_16k/000-GPT-B/checkpoints/latest.pt"
DECODER_CKPT = "checkpoints/imagenet100k_16k_tokenizer/checkpoints/best.ckpt"
DECODER_CONFIG = "configs/num_channels/ch_256_16k"
LATENTS_PATH = "latents_data/imagenet100k_16k/val_latents.npz"
OUTPUT_DIR = "temp_output/16k_generation_final"

NUM_UNCOND = 30          # unconditional images
NUM_COMPLETION = 5       # images per completion ratio
COMPLETION_RATIOS = [0.10, 0.20, 0.25, 0.50]  # 10%, 20%, 25%, 50%
TEMPERATURE = 0.85
TOP_K = 200
TOTAL_TOKENS = 256
DEVICE = "cuda"


def load_gpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = ckpt['args']
    print(f"Checkpoint: epoch={ckpt['epoch']}, step={ckpt['steps']}")

    gpt = GPT_models['GPT-B'](
        vocab_size=args.pad_token_id + 1,
        block_size=args.max_sequence_length,
        num_classes=1000, cls_token_num=0, model_type='c2i',
        resid_dropout_p=0.0, ffn_dropout_p=0.0,
        drop_path_rate=0.0, token_dropout_p=0.0,
    )
    sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    gpt.load_state_dict(sd, strict=True)
    gpt.eval().to(device)
    print(f"GPT-B loaded: {sum(p.numel() for p in gpt.parameters())/1e6:.1f}M params")
    return gpt, args


def sample_tokens(gpt, prefix, num_to_generate, vocab_size, device,
                  temperature=0.85, top_k=200):
    """Autoregressively sample tokens given a prefix."""
    generated = list(prefix)
    with torch.no_grad():
        for _ in range(num_to_generate):
            idx = torch.tensor([generated], dtype=torch.long, device=device)
            dummy_cond = torch.zeros(1, dtype=torch.long, device=device)
            logits, _ = gpt(idx=idx, cond_idx=dummy_cond)
            logits_last = logits[0, -1, :vocab_size] / temperature
            probs = torch.softmax(logits_last, dim=-1)
            topk_vals, topk_idx = torch.topk(probs, top_k)
            topk_probs = topk_vals / topk_vals.sum()
            sampled = topk_idx[torch.multinomial(topk_probs, 1)]
            generated.append(sampled.item())
    return generated


def decode_tokens(decoder, token_ids, device):
    """Decode token indices to image."""
    indices = torch.tensor(token_ids, dtype=torch.long)
    recon, _ = reconstruct_from_channelwise_indices(
        decoder, indices, latent_hw=(4, 4), device=device
    )
    return denorm_to_uint8(recon)


def main():
    os.makedirs(f"{OUTPUT_DIR}/unconditional", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/completion", exist_ok=True)

    # Load models
    gpt, args = load_gpt(CKPT_PATH, DEVICE)
    decoder_cfg = load_decoder_config(DECODER_CONFIG)
    decoder = load_decoder_model(DECODER_CKPT, decoder_cfg, DEVICE)
    print("Decoder loaded")

    bos_id = args.bos_token_id
    vocab = args.vocab_size

    # ---- Part 1: Unconditional Generation ----
    print(f"\n{'='*60}")
    print(f"Generating {NUM_UNCOND} unconditional images (full 256 tokens)")
    print(f"{'='*60}")

    for i in range(NUM_UNCOND):
        torch.manual_seed(200 + i)
        tokens = sample_tokens(gpt, [bos_id], TOTAL_TOKENS, vocab, DEVICE,
                               TEMPERATURE, TOP_K)
        img = decode_tokens(decoder, tokens[1:], DEVICE)  # skip BOS
        Image.fromarray(img).save(f"{OUTPUT_DIR}/unconditional/gen_{i:02d}.png")
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{NUM_UNCOND}")

    # ---- Part 2: Completion from real tokens ----
    print(f"\n{'='*60}")
    print(f"Generating completions at {COMPLETION_RATIOS}")
    print(f"{'='*60}")

    # Load real val latents
    data = np.load(LATENTS_PATH)
    val_indices = data['indices']  # [5000, 256]

    for ratio in COMPLETION_RATIOS:
        n_given = int(TOTAL_TOKENS * ratio)
        n_generate = TOTAL_TOKENS - n_given
        ratio_name = f"{int(ratio*100)}pct"
        ratio_dir = f"{OUTPUT_DIR}/completion/{ratio_name}"
        os.makedirs(ratio_dir, exist_ok=True)

        print(f"\n  Ratio {ratio_name}: {n_given} given + {n_generate} generated = 256 tokens")

        for i in range(NUM_COMPLETION):
            # Pick a val image (spread across dataset)
            img_idx = i * 200
            real_tokens = val_indices[img_idx].tolist()

            torch.manual_seed(300 + i)

            # Prefix: BOS + first n_given real tokens
            prefix = [bos_id] + real_tokens[:n_given]
            tokens = sample_tokens(gpt, prefix, n_generate, vocab, DEVICE,
                                   TEMPERATURE, TOP_K)
            generated_tokens = tokens[1:]  # skip BOS

            # Also decode the full real image for comparison
            real_img = decode_tokens(decoder, real_tokens, DEVICE)

            # Decode the completion (given real prefix + generated rest)
            completed_img = decode_tokens(decoder, generated_tokens, DEVICE)

            # Decode just the prefix (padded with zeros for missing channels)
            prefix_only = real_tokens[:n_given]
            prefix_img = decode_tokens(decoder, prefix_only, DEVICE)

            # Save: real | prefix_only | completed
            combined = np.concatenate([real_img, prefix_img, completed_img], axis=1)
            Image.fromarray(combined).save(f"{ratio_dir}/comp_{i:02d}.png")

            # Also save individual completed
            Image.fromarray(completed_img).save(f"{ratio_dir}/completed_{i:02d}.png")

        print(f"  Saved to {ratio_dir}/")

    print(f"\n{'='*60}")
    print(f"Done! All images at {OUTPUT_DIR}/")
    print(f"  unconditional/  - {NUM_UNCOND} images from scratch")
    print(f"  completion/     - {NUM_COMPLETION} images per ratio")
    print(f"    10pct/ - given 26 tokens, generated 230")
    print(f"    20pct/ - given 51 tokens, generated 205")
    print(f"    25pct/ - given 64 tokens, generated 192")
    print(f"    50pct/ - given 128 tokens, generated 128")
    print(f"  Each completion image: real | prefix_decode | completed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
