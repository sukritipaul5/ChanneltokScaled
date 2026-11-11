#!/usr/bin/env python3
"""Verify saved ChannelTok indices by decoding and re-encoding val images."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image
from torchvision import datasets, transforms

from tokenizer.evaluate import calculate_metrics
from utils.checkpoint import load_checkpoint, load_config
from utils.latent_extraction import IndexExtractor


def save_image(tensor, path):
    image = ((tensor.detach().cpu() + 1.0) * 0.5).clamp(0, 1)
    array = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def main():
    parser = argparse.ArgumentParser(
        description="Decode saved NPZ indices and compare them with source images"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--latents", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_indices", type=int, nargs="+", required=True)
    parser.add_argument("--extraction_batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    model = load_checkpoint(args.checkpoint, config, device=str(device))
    model.eval()

    img_size = int(config.data.img_size)
    val_resize = int(getattr(config.data, "val_resize", 256))
    transform = transforms.Compose(
        [
            transforms.Resize(val_resize, antialias=True),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )
    dataset = datasets.ImageFolder(Path(args.data_dir) / "val", transform=transform)

    positions = args.sample_indices
    if min(positions) < 0 or max(positions) >= len(dataset):
        raise ValueError(f"Sample positions must be within [0, {len(dataset) - 1}]")

    with np.load(args.latents, mmap_mode="r", allow_pickle=False) as data:
        saved_indices_np = data["indices"][positions].astype(np.int64)
        saved_labels = data["labels"][positions].astype(np.int64)

    samples = [dataset[position] for position in positions]
    originals = torch.stack([sample[0] for sample in samples]).to(device)
    source_labels = np.asarray([sample[1] for sample in samples], dtype=np.int64)
    if not np.array_equal(saved_labels, source_labels):
        raise AssertionError(
            f"Label mismatch: saved={saved_labels.tolist()} source={source_labels.tolist()}"
        )

    saved_indices = torch.from_numpy(saved_indices_np).to(device=device, dtype=torch.long)
    extractor = IndexExtractor(model, inference_t=512)
    if torch.cuda.device_count() > 1:
        extractor = torch.nn.DataParallel(extractor)
    print(
        f"Re-encoding with batch size {args.extraction_batch_size} "
        f"on {torch.cuda.device_count()} GPU(s)"
    )

    with torch.no_grad():
        reencoded_rows = []
        for position in positions:
            batch_start = (position // args.extraction_batch_size) * args.extraction_batch_size
            batch_stop = min(batch_start + args.extraction_batch_size, len(dataset))
            batch = torch.stack(
                [dataset[index][0] for index in range(batch_start, batch_stop)]
            ).to(device, non_blocking=True)
            batch_indices = extractor(batch).reshape(len(batch), 512)
            reencoded_rows.append(batch_indices[position - batch_start])
        reencoded_indices = torch.stack(reencoded_rows)

        codes = model.quantizer.indices_to_codes(saved_indices, project_out=True)
        expected_codes_shape = (len(positions), 512, 16)
        if tuple(codes.shape) != expected_codes_shape:
            raise AssertionError(
                f"Expected decoded code shape {expected_codes_shape}, got {tuple(codes.shape)}"
            )
        quantized = codes.reshape(len(positions), 512, 4, 4)
        saved_recon = model.decoder(quantized)

        direct_codes = model.quantizer.indices_to_codes(
            reencoded_indices, project_out=True
        )
        direct_recon = model.decoder(direct_codes.reshape(len(positions), 512, 4, 4))

    token_matches = (reencoded_indices == saved_indices).sum(dim=1)
    decoder_max_abs_delta = (saved_recon - direct_recon).abs().flatten(1).max(dim=1).values

    results = []
    for sample_number, position in enumerate(positions):
        metrics = calculate_metrics(
            originals[sample_number], saved_recon[sample_number], lpips_fn=None
        )
        record = {
            "position": position,
            "class_index": int(source_labels[sample_number]),
            "matching_tokens": int(token_matches[sample_number].item()),
            "total_tokens": 512,
            "decoder_max_abs_delta": float(decoder_max_abs_delta[sample_number].item()),
            **metrics,
        }
        results.append(record)
        save_image(originals[sample_number], output_dir / f"sample_{position:05d}_original.png")
        save_image(saved_recon[sample_number], output_dir / f"sample_{position:05d}_reconstruction.png")
        print(json.dumps(record, sort_keys=True))

    originals_01 = ((originals.detach().cpu() + 1.0) * 0.5).clamp(0, 1)
    recon_01 = ((saved_recon.detach().cpu() + 1.0) * 0.5).clamp(0, 1)
    grid = vutils.make_grid(
        torch.cat([originals_01, recon_01], dim=0),
        nrow=len(positions),
        normalize=False,
    )
    grid_array = (grid.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(grid_array).save(output_dir / "comparison_grid.png")

    summary = {
        "samples": results,
        "all_labels_match": True,
        "all_tokens_match": all(item["matching_tokens"] == 512 for item in results),
        "all_decodes_match_direct_forward": all(
            item["decoder_max_abs_delta"] <= 1e-6 for item in results
        ),
        "mean_psnr": float(np.mean([item["psnr"] for item in results])),
        "mean_ssim": float(np.mean([item["ssim"] for item in results])),
        "mean_l1": float(np.mean([item["l1"] for item in results])),
    }
    with open(output_dir / "verification.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print("SUMMARY=" + json.dumps(summary, sort_keys=True))
    if not summary["all_tokens_match"]:
        raise AssertionError("At least one source image did not reproduce its saved indices")
    if not summary["all_decodes_match_direct_forward"]:
        raise AssertionError("Saved-index decode differs from direct tokenizer reconstruction")


if __name__ == "__main__":
    main()
