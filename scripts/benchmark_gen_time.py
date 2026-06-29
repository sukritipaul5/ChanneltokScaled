#!/usr/bin/env python3
"""
Benchmark generation time vs token budget.
Generates 50 images at each budget and measures wall-clock time.

Usage:
  cd /mnt/data/mask-git-juhu
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 uv run python scripts/benchmark_gen_time.py
"""

import torch
import time

from generation.decode_utils import reconstruct_from_channelwise_indices, denorm_to_uint8
from generation.train import load_decoder_model
from generation.models.gpt import GPT_models
from utils.checkpoint import load_config as load_decoder_config

CKPT_PATH = "llamagen_flextok/results_imagenet100k_16k/001-GPT-L/checkpoints/latest.pt"
DECODER_CKPT = "checkpoints/imagenet100k_16k_tokenizer/checkpoints/best.ckpt"
DECODER_CONFIG = "configs/num_channels/ch_256_16k"

BUDGETS = [32, 64, 128, 192, 256]
NUM_IMAGES = 50
TEMPERATURE = 0.85
TOP_K = 200
DEVICE = "cuda"


def main():
    # Load models once
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    args = ckpt['args']

    gpt = GPT_models['GPT-L'](
        vocab_size=args.pad_token_id + 1,
        block_size=args.max_sequence_length,
        num_classes=1000, cls_token_num=0, model_type='c2i',
        resid_dropout_p=0.0, ffn_dropout_p=0.0,
        drop_path_rate=0.0, token_dropout_p=0.0,
    )
    sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    gpt.load_state_dict(sd, strict=True)
    gpt.eval().to(DEVICE)
    del ckpt

    decoder_cfg = load_decoder_config(DECODER_CONFIG)
    decoder = load_decoder_model(DECODER_CKPT, decoder_cfg, DEVICE)

    bos_id = args.bos_token_id
    vocab = args.vocab_size

    print(f"{'Budget':>8} | {'Total (s)':>10} | {'Per Image (s)':>13} | {'Speedup vs 256':>14}")
    print("-" * 60)

    results = {}

    for budget in BUDGETS:
        # Warmup
        torch.manual_seed(0)
        generated = [bos_id]
        with torch.no_grad():
            for _ in range(min(budget, 8)):
                idx = torch.tensor([generated], dtype=torch.long, device=DEVICE)
                dummy = torch.zeros(1, dtype=torch.long, device=DEVICE)
                logits, _ = gpt(idx=idx, cond_idx=dummy)
                logits_last = logits[0, -1, :vocab] / TEMPERATURE
                probs = torch.softmax(logits_last, dim=-1)
                topk_vals, topk_idx = torch.topk(probs, TOP_K)
                topk_probs = topk_vals / topk_vals.sum()
                sampled = topk_idx[torch.multinomial(topk_probs, 1)]
                generated.append(sampled.item())
        torch.cuda.synchronize()

        # Benchmark: generation + decoding
        t0 = time.time()
        for i in range(NUM_IMAGES):
            torch.manual_seed(1000 + i)
            generated = [bos_id]

            with torch.no_grad():
                for _ in range(budget):
                    idx = torch.tensor([generated], dtype=torch.long, device=DEVICE)
                    dummy = torch.zeros(1, dtype=torch.long, device=DEVICE)
                    logits, _ = gpt(idx=idx, cond_idx=dummy)
                    logits_last = logits[0, -1, :vocab] / TEMPERATURE
                    probs = torch.softmax(logits_last, dim=-1)
                    topk_vals, topk_idx = torch.topk(probs, TOP_K)
                    topk_probs = topk_vals / topk_vals.sum()
                    sampled = topk_idx[torch.multinomial(topk_probs, 1)]
                    generated.append(sampled.item())

            # Decode
            indices = torch.tensor(generated[1:], dtype=torch.long)
            with torch.no_grad():
                recon, _ = reconstruct_from_channelwise_indices(
                    decoder, indices, latent_hw=(4, 4), device=DEVICE
                )
            _ = denorm_to_uint8(recon)

        torch.cuda.synchronize()
        elapsed = time.time() - t0
        per_image = elapsed / NUM_IMAGES
        results[budget] = per_image

    # Print results
    base_time = results[256]
    for budget in BUDGETS:
        per_img = results[budget]
        speedup = base_time / per_img
        total = per_img * NUM_IMAGES
        print(f"{budget:>8} | {total:>10.2f} | {per_img:>13.3f} | {speedup:>13.2f}x")

    print(f"\nBenchmark: {NUM_IMAGES} images per budget on single GPU ({DEVICE})")
    print(f"Model: GPT-L (342.9M params)")


if __name__ == "__main__":
    main()
