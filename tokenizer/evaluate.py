#!/usr/bin/env python3
"""
Evaluate reconstruction quality of a trained ChannelTok tokenizer.
"""

import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.utils as vutils

# Import model/data loaders and shared checkpoint utilities
from data import get_data_loader
from utils.checkpoint import load_config, load_checkpoint


def calculate_metrics(original, reconstructed, lpips_fn=None):
    """Calculate per-image reconstruction metrics.

    Args:
        original: Tensor [C, H, W] in [-1, 1].
        reconstructed: Tensor [C, H, W] in [-1, 1].
        lpips_fn: Optional LPIPS metric (torchmetrics). If None, LPIPS is skipped.

    Returns:
        Dict with mse, psnr, l1, ssim, and optionally lpips.
    """
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn

    # [0, 1] range for MSE/PSNR
    orig_01 = (original + 1.0) * 0.5
    recon_01 = (reconstructed + 1.0) * 0.5

    mse = F.mse_loss(recon_01, orig_01)
    psnr = (10 * torch.log10(1 / mse)).item() if mse > 0 else float('inf')
    l1 = F.l1_loss(recon_01, orig_01).item()

    # SSIM expects [B, C, H, W]
    ssim_val = ssim_fn(
        recon_01.unsqueeze(0), orig_01.unsqueeze(0), data_range=1.0
    ).item()

    metrics = {'mse': mse.item(), 'psnr': psnr, 'l1': l1, 'ssim': ssim_val}

    # LPIPS (expects [-1, 1] input)
    if lpips_fn is not None:
        lpips_val = lpips_fn(
            reconstructed.unsqueeze(0), original.unsqueeze(0)
        ).item()
        metrics['lpips'] = lpips_val

    return metrics


def save_reconstruction_grid(original, reconstructed, save_path, num_images=8):
    """Save grid of original and reconstructed images"""
    with torch.no_grad():
        # Take first num_images
        n = min(num_images, original.size(0))
        orig_viz = original[:n].cpu()
        recon_viz = reconstructed[:n].cpu()

        # Convert from [-1, 1] to [0, 1]
        orig_viz = (orig_viz + 1.0) / 2.0
        recon_viz = (recon_viz + 1.0) / 2.0

        # Clamp
        orig_viz = orig_viz.clamp(0, 1)
        recon_viz = recon_viz.clamp(0, 1)

        # Create grid: top row = original, bottom row = reconstructed
        combined = torch.cat([orig_viz, recon_viz], dim=0)
        grid = vutils.make_grid(combined, nrow=n, normalize=False, scale_each=False)

        # Save
        grid_np = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(grid_np).save(save_path)


def save_image(image, save_path):
    """Save image"""
    image = image.cpu().detach()
    image = (image + 1.0) / 2.0
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0)* 255
    image = image.numpy().astype(np.uint8)
    Image.fromarray(image).save(save_path)
    print(f"Saved image to {save_path}")


def _save_png(tensor_01, path):
    """Save a [C, H, W] tensor in [0, 1] as a lossless PNG."""
    arr = (tensor_01.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def main():
    parser = argparse.ArgumentParser(description='Evaluate VQGAN autoencoder reconstruction')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./inference_output', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples (None = all)')
    parser.add_argument('--sample_offset', type=int, default=0,
                        help='Skip this many validation samples before evaluation')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size')
    parser.add_argument('--save_images', action='store_true', help='Save reconstruction grids')
    parser.add_argument('--inference_t', type=int, default=None, help='Number of channels (compression level)')
    parser.add_argument('--compute_fid', action='store_true',
                        help='Save PNGs and compute rFID via clean-fid (slow, needs all samples)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to ImageNet (with train/ and val/ subdirs). Sets dataset to imagefolder.')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['imagefolder', 'imagenet_wds'],
                        help='Override dataset type from config')
    parser.add_argument('--device', type=str, default=None, help='Device (auto-detect if None)')

    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    if args.dataset is not None:
        config.data.dataset = args.dataset
    if args.data_dir is not None:
        config.data.dataset = 'imagefolder'
        config.data.data_dir = args.data_dir
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size

    os.makedirs(args.output_dir, exist_ok=True)

    # Detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint from {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, config, device=str(device))

    # LPIPS metric (VGG backbone, matches paper protocol using standalone lpips package)
    import lpips as lpips_lib
    lpips_fn = lpips_lib.LPIPS(net='vgg').to(device).eval()

    # Get validation dataloader
    print(f"Loading validation data (dataset: {config.data.dataset})")
    val_loader = get_data_loader(
        config=config,
        device=device,
        distributed=False,
        local_rank=0,
        world_size=1,
        split='val'
    )

    # Prepare rFID directories
    if args.compute_fid:
        orig_dir = os.path.join(args.output_dir, 'originals')
        recon_dir = os.path.join(args.output_dir, 'reconstructions')
        os.makedirs(orig_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)

    # Run inference
    print(f"\nRunning inference...")
    if args.inference_t is not None:
        print(f"  Using inference_t={args.inference_t} (fixed channel masking)")

    all_metrics = []
    num_processed = 0
    num_skipped = 0
    global_img_idx = 0
    grid_saved = False

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Processing")):
            if args.num_samples is not None and num_processed >= args.num_samples:
                break

            # Extract images (handle DALI / tuple / tensor)
            if isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], dict):
                images = batch[0]['data']
            elif isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            # Select a later deterministic slice without changing dataset order.
            if num_skipped < args.sample_offset:
                remaining_to_skip = args.sample_offset - num_skipped
                if remaining_to_skip >= images.size(0):
                    num_skipped += images.size(0)
                    continue
                images = images[remaining_to_skip:]
                num_skipped += remaining_to_skip

            images = images.to(device)

            # Forward pass
            output = model(images, inference_t=args.inference_t)
            reconstructed = output[0] if isinstance(output, tuple) else output

            # Per-image metrics
            for i in range(images.size(0)):
                if args.num_samples is not None and num_processed >= args.num_samples:
                    break

                metrics = calculate_metrics(images[i], reconstructed[i], lpips_fn=lpips_fn)
                all_metrics.append(metrics)
                num_processed += 1

                # Save PNGs for rFID
                if args.compute_fid:
                    orig_01 = (images[i] + 1.0) * 0.5
                    recon_01 = (reconstructed[i] + 1.0) * 0.5
                    _save_png(orig_01, os.path.join(orig_dir, f'{global_img_idx:06d}.png'))
                    _save_png(recon_01, os.path.join(recon_dir, f'{global_img_idx:06d}.png'))

                global_img_idx += 1

            # Save reconstruction grid from first batch
            if args.save_images and not grid_saved:
                save_path = os.path.join(args.output_dir, 'reconstruction_grid.png')
                save_reconstruction_grid(images, reconstructed, save_path)
                print(f"Saved reconstruction grid to {save_path}")
                grid_saved = True

    # Aggregate metrics
    avg = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}

    # Print results
    print("\n" + "=" * 60)
    print("RECONSTRUCTION METRICS:")
    print("=" * 60)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Config     : {args.config}")
    print(f"Samples    : {num_processed}")
    if args.inference_t is not None:
        print(f"Inference_t: {args.inference_t}")
    print(f"\n  MSE      : {avg['mse']:.6f}")
    print(f"  PSNR     : {avg['psnr']:.2f} dB")
    print(f"  L1       : {avg['l1']:.4f}")
    print(f"  SSIM     : {avg['ssim']:.4f}")
    if 'lpips' in avg:
        print(f"  LPIPS    : {avg['lpips']:.4f}")

    # Compute rFID
    if args.compute_fid:
        print("\nComputing rFID (clean-fid)...")
        from cleanfid import fid as cleanfid
        rfid = cleanfid.compute_fid(orig_dir, recon_dir)
        avg['rfid'] = rfid
        print(f"  rFID     : {rfid:.2f}")

    print("=" * 60)

    # Save metrics
    metrics_file = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Samples: {num_processed}\n")
        f.write(f"Sample offset: {args.sample_offset}\n")
        if args.inference_t is not None:
            f.write(f"Inference_t: {args.inference_t}\n")
        for k, v in avg.items():
            f.write(f"{k}: {v}\n")

    print(f"Metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()
