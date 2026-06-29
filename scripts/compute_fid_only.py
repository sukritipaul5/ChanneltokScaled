#!/usr/bin/env python3
"""
Compute FID between generated and reference folders.
Run this AFTER generation is done (separate from multiprocessing generation).

Usage:
  cd /mnt/data/mask-git-juhu
  uv run python compute_fid_only.py --gen-dir temp_output/fid_eval/generated_64tok
"""
import argparse, os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-dir", type=str, required=True)
    parser.add_argument("--ref-dir", type=str, default="temp_output/fid_eval/reference")
    args = parser.parse_args()

    n_gen = len([f for f in os.listdir(args.gen_dir) if f.endswith('.png')])
    n_ref = len([f for f in os.listdir(args.ref_dir) if f.endswith('.png')])
    print(f"Reference: {n_ref} images in {args.ref_dir}")
    print(f"Generated: {n_gen} images in {args.gen_dir}")

    from cleanfid import fid
    score = fid.compute_fid(args.ref_dir, args.gen_dir, num_workers=0)

    print(f"\n{'='*50}")
    print(f"  FID = {score:.2f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
