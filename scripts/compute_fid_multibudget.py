#!/usr/bin/env python3
"""
Compute FID at multiple token budgets (64, 128, 192, 256).
Assumes reference images already exist at temp_output/fid_eval/reference/

Usage:
  cd /mnt/data/mask-git-juhu
  PYTHONPATH=. uv run python scripts/compute_fid_multibudget.py --budget 64
  PYTHONPATH=. uv run python scripts/compute_fid_multibudget.py --budget 128
"""

import torch
import torch.multiprocessing as mp
import numpy as np
import os, sys, time, argparse

from PIL import Image
from generation.decode_utils import reconstruct_from_channelwise_indices, denorm_to_uint8
from generation.train import load_decoder_model
from generation.models.gpt import GPT_models
from utils.checkpoint import load_config as load_decoder_config

CKPT_PATH = "llamagen_flextok/results_imagenet100k_16k/001-GPT-L/checkpoints/latest.pt"
DECODER_CKPT = "checkpoints/imagenet100k_16k_tokenizer/checkpoints/best.ckpt"
DECODER_CONFIG = "configs/num_channels/ch_256_16k"
REF_DIR = "temp_output/fid_eval/reference"
NUM_GENERATE = 5000
TEMPERATURE = 0.85
TOP_K = 200


def generate_on_gpu(gpu_id, start_idx, count, budget, gen_dir, ckpt_path):
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(device)

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

    decoder_cfg = load_decoder_config(DECODER_CONFIG)
    decoder = load_decoder_model(DECODER_CKPT, decoder_cfg, device)

    bos_id = args.bos_token_id
    vocab = args.vocab_size

    for i in range(count):
        global_idx = start_idx + i
        torch.manual_seed(5000 + global_idx)
        generated = [bos_id]

        with torch.no_grad():
            for _ in range(budget):
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
        Image.fromarray(img).save(f"{gen_dir}/{global_idx:05d}.png")

        if (i + 1) % 100 == 0:
            print(f"  GPU {gpu_id}: {i+1}/{count}")

    print(f"  GPU {gpu_id}: done ({count} images)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, required=True, help="Token budget (64, 128, 192, 256)")
    parser.add_argument("--num-gpus", type=int, default=8)
    args = parser.parse_args()

    budget = args.budget
    num_gpus = args.num_gpus
    gen_dir = f"temp_output/fid_eval/generated_{budget}tok"

    print(f"Generating {NUM_GENERATE} images at {budget} tokens across {num_gpus} GPUs...")
    os.makedirs(gen_dir, exist_ok=True)

    mp.set_start_method('spawn', force=True)
    t0 = time.time()

    per_gpu = NUM_GENERATE // num_gpus
    remainder = NUM_GENERATE % num_gpus

    processes = []
    start_idx = 0
    for gpu_id in range(num_gpus):
        count = per_gpu + (1 if gpu_id < remainder else 0)
        p = mp.Process(target=generate_on_gpu,
                       args=(gpu_id, start_idx, count, budget, gen_dir, CKPT_PATH))
        p.start()
        processes.append(p)
        start_idx += count

    for p in processes:
        p.join()

    actual = len([f for f in os.listdir(gen_dir) if f.endswith('.png')])
    gen_time = time.time() - t0
    print(f"Generated {actual} images in {gen_time/60:.1f} min")

    # Compute FID
    print(f"\nComputing FID: {gen_dir} vs {REF_DIR}...")
    from cleanfid import fid
    score = fid.compute_fid(REF_DIR, gen_dir)

    print(f"\n{'='*50}")
    print(f"  Budget: {budget} tokens")
    print(f"  FID = {score:.2f}")
    print(f"  Reference: {REF_DIR} ({len(os.listdir(REF_DIR))} images)")
    print(f"  Generated: {gen_dir} ({actual} images)")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
