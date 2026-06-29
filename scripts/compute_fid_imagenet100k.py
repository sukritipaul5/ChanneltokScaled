#!/usr/bin/env python3
"""
Compute FID for GPT-L on ImageNet-100k.
1. Decode val latents → reference images
2. Generate 5K images across multiple GPUs
3. Compute FID using clean-fid

Usage:
  cd /mnt/data/mask-git-juhu
  PYTHONPATH=. uv run python scripts/compute_fid_imagenet100k.py
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import os, sys, time

from PIL import Image
from generation.decode_utils import reconstruct_from_channelwise_indices, denorm_to_uint8
from generation.train import load_decoder_model
from generation.models.gpt import GPT_models
from utils.checkpoint import load_config as load_decoder_config

# ---- Config ----
CKPT_PATH = "llamagen_flextok/results_imagenet100k_16k/001-GPT-L/checkpoints/latest.pt"
DECODER_CKPT = "checkpoints/imagenet100k_16k_tokenizer/checkpoints/best.ckpt"
DECODER_CONFIG = "configs/num_channels/ch_256_16k"
VAL_LATENTS = "latents_data/imagenet100k_16k/val_latents.npz"

REF_DIR = "temp_output/fid_eval/reference"
GEN_DIR = "temp_output/fid_eval/generated"
NUM_GENERATE = 5000
NUM_GPUS = 8
TEMPERATURE = 0.85
TOP_K = 200


def decode_val_images():
    """Decode val latents through the tokenizer to get reference images."""
    print("Step 1: Decoding val latents to reference images...")
    os.makedirs(REF_DIR, exist_ok=True)

    decoder_cfg = load_decoder_config(DECODER_CONFIG)
    decoder = load_decoder_model(DECODER_CKPT, decoder_cfg, 'cuda:0')

    data = np.load(VAL_LATENTS)
    val_indices = data['indices']  # [5000, 256]
    n = len(val_indices)
    print(f"  {n} val samples to decode")

    for i in range(n):
        indices = torch.tensor(val_indices[i], dtype=torch.long)
        with torch.no_grad():
            recon, _ = reconstruct_from_channelwise_indices(
                decoder, indices, latent_hw=(4, 4), device='cuda:0'
            )
        img = denorm_to_uint8(recon)
        Image.fromarray(img).save(f"{REF_DIR}/{i:05d}.png")
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n}")

    print(f"  Reference images saved to {REF_DIR}/")
    return n


def generate_on_gpu(gpu_id, start_idx, count, ckpt_path):
    """Generate images on a single GPU."""
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)

    # Load GPT-L
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
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
    gpt.eval().to(device)
    del ckpt

    # Load decoder
    decoder_cfg = load_decoder_config(DECODER_CONFIG)
    decoder = load_decoder_model(DECODER_CKPT, decoder_cfg, device)

    bos_id = args.bos_token_id
    vocab = args.vocab_size

    for i in range(count):
        global_idx = start_idx + i
        torch.manual_seed(5000 + global_idx)
        generated = [bos_id]

        with torch.no_grad():
            for _ in range(256):
                idx = torch.tensor([generated], dtype=torch.long, device=device)
                dummy = torch.zeros(1, dtype=torch.long, device=device)
                logits, _ = gpt(idx=idx, cond_idx=dummy)
                logits_last = logits[0, -1, :vocab] / TEMPERATURE
                probs = torch.softmax(logits_last, dim=-1)
                topk_vals, topk_idx = torch.topk(probs, TOP_K)
                topk_probs = topk_vals / topk_vals.sum()
                sampled = topk_idx[torch.multinomial(topk_probs, 1)]
                generated.append(sampled.item())

        indices = torch.tensor(generated[1:], dtype=torch.long)
        with torch.no_grad():
            recon, _ = reconstruct_from_channelwise_indices(
                decoder, indices, latent_hw=(4, 4), device=device
            )
        img = denorm_to_uint8(recon)
        Image.fromarray(img).save(f"{GEN_DIR}/{global_idx:05d}.png")

        if (i + 1) % 100 == 0:
            print(f"  GPU {gpu_id}: {i+1}/{count}")

    print(f"  GPU {gpu_id}: done ({count} images)")


def generate_all():
    """Generate 5K images across multiple GPUs using multiprocessing."""
    print(f"\nStep 2: Generating {NUM_GENERATE} images across {NUM_GPUS} GPUs...")
    os.makedirs(GEN_DIR, exist_ok=True)

    per_gpu = NUM_GENERATE // NUM_GPUS
    remainder = NUM_GENERATE % NUM_GPUS

    processes = []
    start_idx = 0
    for gpu_id in range(NUM_GPUS):
        count = per_gpu + (1 if gpu_id < remainder else 0)
        p = mp.Process(target=generate_on_gpu, args=(gpu_id, start_idx, count, CKPT_PATH))
        p.start()
        processes.append(p)
        start_idx += count

    for p in processes:
        p.join()

    actual = len([f for f in os.listdir(GEN_DIR) if f.endswith('.png')])
    print(f"  Generated {actual} images in {GEN_DIR}/")
    return actual


def compute_fid():
    """Compute FID between reference and generated images."""
    print("\nStep 3: Computing FID...")
    from cleanfid import fid

    score = fid.compute_fid(REF_DIR, GEN_DIR)
    print(f"\n{'='*50}")
    print(f"  FID = {score:.2f}")
    print(f"  Reference: {REF_DIR} ({len(os.listdir(REF_DIR))} images)")
    print(f"  Generated: {GEN_DIR} ({len(os.listdir(GEN_DIR))} images)")
    print(f"{'='*50}")
    return score


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    t0 = time.time()

    # Step 1: Decode val images
    n_ref = decode_val_images()

    # Step 2: Generate
    n_gen = generate_all()

    # Step 3: FID
    score = compute_fid()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
