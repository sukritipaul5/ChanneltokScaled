"""
Evaluate a trained tokenizer checkpoint on ImageNet-100k val set.
Reports PSNR, L1, L2 averaged over val set at multiple token budgets.

Usage:
    python -m tokenizer.evaluate_quantizer \
        --config configs/quantizers/fsq.yaml \
        --ckpt outputs/quantizer_ablation/fsq/checkpoints/best.ckpt
"""

import os
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

from data import get_data_loader
from utils.checkpoint import load_config, load_checkpoint


def evaluate(model, val_loader, budgets, device):
    """Evaluate model at multiple token budgets. Returns dict of budget -> metrics."""
    results = {}

    for budget in budgets:
        psnr_sum = 0.0
        l1_sum = 0.0
        l2_sum = 0.0
        count = 0

        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device)

            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                decoded, _ = model(images, inference_t=budget)

            # Denormalize to [0, 1]
            orig_01 = (images.float() + 1.0) * 0.5
            dec_01 = (decoded.float() + 1.0) * 0.5
            dec_01 = dec_01.clamp(0, 1)

            # Per-batch metrics
            mse = F.mse_loss(dec_01, orig_01)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            l1 = F.l1_loss(dec_01, orig_01)
            l2 = torch.sqrt(mse)

            psnr_sum += psnr.item() * images.shape[0]
            l1_sum += l1.item() * images.shape[0]
            l2_sum += l2.item() * images.shape[0]
            count += images.shape[0]

        results[budget] = {
            'psnr': psnr_sum / count,
            'l1': l1_sum / count,
            'l2': l2_sum / count,
        }
        print(f"  Budget {budget:>3d}: PSNR={results[budget]['psnr']:.2f}, L1={results[budget]['l1']:.4f}, L2={results[budget]['l2']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--budgets', type=int, nargs='+', default=[32, 64, 128, 256])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--data_dir', type=str, default=None, help='Path to ImageNet (sets dataset to imagefolder)')
    parser.add_argument('--dataset', type=str, default=None, choices=['imagefolder', 'imagenet_wds'],
                        help='Override dataset type from config')
    parser.add_argument('--output', type=str, default=None, help='Output file for results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = load_config(args.config)
    if args.dataset is not None:
        config.data.dataset = args.dataset
    if args.data_dir is not None:
        config.data.dataset = 'imagefolder'
        config.data.data_dir = args.data_dir
    config.data.batch_size = args.batch_size
    config.data.num_workers = 4

    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Quantizer: {config.model.get('quantizer', 'FSQ')}")
    print(f"Budgets: {args.budgets}")

    model = load_checkpoint(args.ckpt, config, device=str(device))

    val_loader = get_data_loader(
        config=config,
        device=device,
        distributed=False,
        local_rank=0,
        world_size=1,
        split='val',
        deterministic=True
    )

    print(f"\nEvaluating on val set...")
    results = evaluate(model, val_loader, args.budgets, device)

    # Print summary table
    ckpt_name = Path(args.ckpt).stem
    quantizer = config.model.get('quantizer', 'FSQ')
    print(f"\n{'='*60}")
    print(f"Results: {quantizer} ({ckpt_name})")
    print(f"{'='*60}")
    print(f"{'Budget':>8s} {'PSNR':>8s} {'L1':>10s} {'L2':>10s}")
    print(f"{'-'*38}")
    for budget in args.budgets:
        r = results[budget]
        print(f"{budget:>8d} {r['psnr']:>8.2f} {r['l1']:>10.4f} {r['l2']:>10.4f}")

    # Save results if output path specified
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Config: {args.config}\n")
            f.write(f"Checkpoint: {args.ckpt}\n")
            f.write(f"Quantizer: {quantizer} ({ckpt_name})\n")
            f.write(f"{'='*60}\n")
            f.write(f"{'Budget':>8s} {'PSNR':>8s} {'L1':>10s} {'L2':>10s}\n")
            f.write(f"{'-'*38}\n")
            for budget in args.budgets:
                r = results[budget]
                f.write(f"{budget:>8d} {r['psnr']:>8.2f} {r['l1']:>10.4f} {r['l2']:>10.4f}\n")
        print(f"\nResults appended to {args.output}")


if __name__ == '__main__':
    main()
