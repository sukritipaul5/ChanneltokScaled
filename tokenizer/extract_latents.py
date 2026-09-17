#!/usr/bin/env python3
"""
CLI entry point for extracting token indices from a trained tokenizer.

Loads model via ``utils.checkpoint``, runs extraction via
``utils.latent_extraction``, and saves compressed NPZ per split.

Usage:
    python -m tokenizer.extract_latents \
        --checkpoint path/to/ckpt \
        --config configs/tokenizer/flexible.yaml \
        --output_dir ./latents_data/out \
        --splits train val \
        --batch_size 256 \
        --num_workers 8 \
        --save_samples
"""

import os
import argparse
import time
import numpy as np
import torch
from pathlib import Path

from utils.checkpoint import load_config, load_checkpoint
from utils.latent_extraction import (
    IndexExtractor,
    extract_latents,
    save_latents_npz,
    validate_first_batch,
)
from data import get_data_loader


def main():
    parser = argparse.ArgumentParser(
        description="Extract token indices from a trained tokenizer"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to tokenizer checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output_dir", type=str, default="./latents_data/out", help="Output directory")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Dataset splits to process")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for extraction")
    parser.add_argument("--num_workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Maximum images to extract from each split")
    parser.add_argument("--inference_t", type=int, default=None, help="Fixed token budget (None = all channels)")
    parser.add_argument("--save_samples", action="store_true", help="Save reconstruction samples")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to ImageNet (sets dataset to imagefolder)")
    parser.add_argument("--dataset", type=str, default=None, choices=["imagefolder", "imagenet_wds"],
                        help="Override dataset type from config")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if None)")
    args = parser.parse_args()

    # ---- device ----
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ---- config ----
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    if args.dataset is not None:
        config.data.dataset = args.dataset
    if args.data_dir is not None:
        config.data.dataset = 'imagefolder'
        config.data.data_dir = args.data_dir
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.num_workers

    # ---- model ----
    print(f"Loading checkpoint from {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, config, device=str(device))
    model = model.to(device)

    # Print model info
    if hasattr(model, "quantizer"):
        print(f"Quantizer type: {getattr(model, 'quantizer_type', 'unknown')}")
        if hasattr(model.quantizer, "codebook_size"):
            print(f"Codebook size: {model.quantizer.codebook_size}")
        if hasattr(model.quantizer, "num_codebooks"):
            print(f"Number of codebooks: {model.quantizer.num_codebooks}")

    # ---- DataParallel when multiple GPUs are available ----
    extractor_model = model
    num_gpus = torch.cuda.device_count()
    use_dp = num_gpus > 1 and device.type == "cuda"

    if use_dp and args.inference_t is not None:
        # Wrap for DataParallel compatibility (can't gather dicts)
        extractor = IndexExtractor(model, inference_t=args.inference_t)
        extractor = torch.nn.DataParallel(extractor)
        print(f"Using DataParallel on {num_gpus} GPUs")
    else:
        extractor = None  # Will use extract_latents' internal path

    # ---- output dir ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- process each split ----
    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print("=" * 60)

        # Dataloader
        print(f"Loading {split} dataloader...")
        dataloader = get_data_loader(
            config=config,
            device=device,
            distributed=False,
            local_rank=0,
            world_size=1,
            split=split,
            return_labels=True,
        )

        # Sample dir
        sample_dir = None
        if args.save_samples:
            sample_dir = str(output_dir / "samples" / split)

        # Extract
        t0 = time.time()

        if extractor is not None:
            # DataParallel path: iterate manually
            all_indices = []
            all_labels = []
            all_names = []
            validated = False

            from utils.latent_extraction import _unpack_batch

            with torch.no_grad():
                num_extracted = 0
                for batch_idx, batch in enumerate(
                    __import__("tqdm").tqdm(dataloader, desc=f"  {split}")
                ):
                    images, labels, keys = _unpack_batch(batch)

                    if args.num_samples is not None:
                        remaining = args.num_samples - num_extracted
                        if remaining <= 0:
                            break
                        if images.size(0) > remaining:
                            images = images[:remaining]
                            if labels is not None:
                                labels = labels[:remaining]
                            if isinstance(keys, list):
                                keys = keys[:remaining]

                    images = images.to(device, non_blocking=True)

                    indices = extractor(images)  # [B, T]

                    if not validated:
                        validate_first_batch(indices)
                        validated = True

                    all_indices.append(indices.cpu().numpy().astype(np.int32))

                    if labels is not None:
                        if isinstance(labels, torch.Tensor):
                            labels = labels.cpu().numpy()
                        if labels.ndim > 1:
                            labels = labels.reshape(-1)
                        all_labels.append(labels.astype(np.int64))
                    else:
                        all_labels.append(np.full(images.size(0), -1, dtype=np.int64))

                    if keys is None:
                        keys = [f"img_{batch_idx}_{i}" for i in range(images.size(0))]
                    if isinstance(keys, list):
                        all_names.extend(keys)
                    else:
                        all_names.extend([keys] * images.size(0))

                    num_extracted += images.size(0)

            indices = np.concatenate(all_indices, axis=0)
            labels_np = np.concatenate(all_labels, axis=0)
            names = all_names
        else:
            # Single-GPU path using extract_latents
            indices, names, labels_np = extract_latents(
                dataloader,
                extractor_model,
                device,
                inference_t=args.inference_t,
                save_samples=args.save_samples,
                sample_dir=sample_dir,
                num_samples=args.num_samples,
            )

        elapsed = time.time() - t0
        print(f"  Elapsed: {elapsed:.1f}s ({len(indices) / elapsed:.0f} img/s)")

        # ---- save ----
        npz_path = output_dir / f"{split}_latents.npz"

        codebook_size = None
        if hasattr(model, "quantizer") and hasattr(model.quantizer, "codebook_size"):
            codebook_size = model.quantizer.codebook_size

        save_latents_npz(
            path=str(npz_path),
            indices=indices,
            labels=labels_np,
            names=names,
            codebook_size=codebook_size,
            num_tokens_per_image=indices.shape[1] if indices.ndim == 2 else None,
        )

        # ---- statistics ----
        size_mb = os.path.getsize(npz_path) / (1024 * 1024)
        unique_tokens = len(np.unique(indices))
        print(f"\n  Saved {split} latents -> {npz_path}")
        print(f"    Shape          : {indices.shape}")
        print(f"    Dtype          : {indices.dtype}")
        print(f"    File size      : {size_mb:.2f} MB")
        print(f"    Index range    : [{indices.min()}, {indices.max()}]")
        print(f"    Unique tokens  : {unique_tokens}")
        print(f"    Mean / Std     : {indices.mean():.1f} / {indices.std():.1f}")

    print(f"\n{'='*60}")
    print(f"Extraction complete. Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
