#!/usr/bin/env python3
"""
Generate paper visualization strips comparing flex vs baseline tokenizer
at multiple channel budgets.

Output: horizontal strips [Original | 32 | 48 | 64 | 112 | 128 | 256 | 512]
for each tokenizer, plus a paired version (both stacked vertically).
"""

import torch
import os
import argparse
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from omegaconf import OmegaConf


# ── Model loading (from eval_imagenet.py) ────────────────────────────────────

def load_config(config_path):
    if not config_path.endswith(".yaml"):
        config_path += ".yaml"
    config = OmegaConf.load(config_path)
    if "defaults" in config:
        base_configs = []
        for default in config.defaults:
            if isinstance(default, str):
                base_config_path = f"configs/{default}.yaml"
                if os.path.exists(base_config_path):
                    base_configs.append(OmegaConf.load(base_config_path))
        if base_configs:
            config = OmegaConf.merge(*base_configs, config)
    return config


def load_checkpoint(checkpoint_path, config):
    from models import get_model_class
    model_class = get_model_class(config.model.name)
    model = model_class(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if "perceptual_loss" in k or "discriminator" in k or "lpips" in k:
            continue
        new_key = k
        if new_key.startswith("module."):
            new_key = new_key[7:]
        if new_key.startswith("model."):
            new_key = new_key[6:]
        if new_key.startswith("_orig_mod."):
            new_key = new_key[10:]
        if new_key.startswith(("encoder.", "decoder.", "quantizer.", "latent_mask_head.", "ch_indices")):
            new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model


# ── Image utilities ──────────────────────────────────────────────────────────

def tensor_to_pil(t):
    """Convert [-1,1] CHW tensor to PIL Image."""
    t = (t.cpu().clamp(-1, 1) + 1.0) / 2.0
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def make_strip(images_pil, labels=None, label_height=20):
    """Create a horizontal strip from a list of PIL images, with optional labels."""
    w, h = images_pil[0].size
    strip_h = h + (label_height if labels else 0)
    strip = Image.new("RGB", (w * len(images_pil), strip_h), (255, 255, 255))
    draw = ImageDraw.Draw(strip)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    for i, img in enumerate(images_pil):
        y_offset = label_height if labels else 0
        strip.paste(img, (i * w, y_offset))
        if labels:
            text = labels[i]
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            tx = i * w + (w - tw) // 2
            draw.text((tx, 2), text, fill=(0, 0, 0), font=font)

    return strip


def stack_strips(strip_top, strip_bottom, gap=4):
    """Stack two strips vertically with a small gap."""
    w = max(strip_top.width, strip_bottom.width)
    h = strip_top.height + strip_bottom.height + gap
    out = Image.new("RGB", (w, h), (255, 255, 255))
    out.paste(strip_top, (0, 0))
    out.paste(strip_bottom, (0, strip_top.height + gap))
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate paper reconstruction strips")
    parser.add_argument("--val_dir", type=str, default="data/val.X",
                        help="Path to ImageNet validation folder")
    parser.add_argument("--output_dir", type=str, default="paper_visualizations",
                        help="Output directory")
    parser.add_argument("--flex_ckpt", type=str,
                        default="checkpoints/flex_tokenizer_final/checkpoints/best.ckpt")
    parser.add_argument("--flex_config", type=str,
                        default="checkpoints/flex_tokenizer_final/checkpoints/config.yaml")
    parser.add_argument("--baseline_ckpt", type=str,
                        default="checkpoints/imagenet_no_latent_mask/checkpoints/best.ckpt")
    parser.add_argument("--baseline_config", type=str,
                        default="checkpoints/imagenet_no_latent_mask/checkpoints/config.yaml")
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[32, 48, 64, 112, 128, 256, 512])
    parser.add_argument("--num_images", type=int, default=1000,
                        help="Total images to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Output dirs ──
    flex_dir = os.path.join(args.output_dir, "flex")
    baseline_dir = os.path.join(args.output_dir, "baseline")
    paired_dir = os.path.join(args.output_dir, "paired")
    os.makedirs(flex_dir, exist_ok=True)
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(paired_dir, exist_ok=True)

    # ── Load dataset ──
    transform = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = ImageFolder(args.val_dir, transform=transform)
    print(f"Dataset: {len(dataset)} images, {len(dataset.classes)} classes")

    # Sample evenly across classes
    per_class = max(1, args.num_images // len(dataset.classes))
    class_to_indices = {}
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices.setdefault(label, []).append(idx)

    selected = []
    for label in sorted(class_to_indices.keys()):
        indices = class_to_indices[label]
        random.shuffle(indices)
        selected.extend(indices[:per_class])
    random.shuffle(selected)
    selected = selected[:args.num_images]
    print(f"Selected {len(selected)} images ({per_class} per class)")

    # ── Load models ──
    print("Loading flex tokenizer...")
    flex_cfg = load_config(args.flex_config)
    flex_model = load_checkpoint(args.flex_ckpt, flex_cfg).to(device)

    print("Loading baseline tokenizer...")
    base_cfg = load_config(args.baseline_config)
    base_model = load_checkpoint(args.baseline_ckpt, base_cfg).to(device)

    labels = ["Original"] + [str(b) for b in args.budgets]

    # ── Process images ──
    print(f"Generating strips for {len(selected)} images...")
    with torch.no_grad():
        for i, idx in enumerate(tqdm(selected)):
            img_tensor, label = dataset[idx]
            img_path = dataset.samples[idx][0]
            class_name = os.path.basename(os.path.dirname(img_path))
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            fname = f"{i:04d}_{class_name}_{img_name}.png"

            img_batch = img_tensor.unsqueeze(0).to(device)
            orig_pil = tensor_to_pil(img_tensor)

            for tag, model, out_dir in [
                ("flex", flex_model, flex_dir),
                ("baseline", base_model, baseline_dir),
            ]:
                recon_pils = [orig_pil]
                for budget in args.budgets:
                    output = model(img_batch, inference_t=budget)
                    recon = output[0] if isinstance(output, tuple) else output
                    recon_pils.append(tensor_to_pil(recon[0]))
                strip = make_strip(recon_pils, labels=labels)
                strip.save(os.path.join(out_dir, fname))

            # Paired: flex on top, baseline below
            flex_strip = Image.open(os.path.join(flex_dir, fname))
            base_strip = Image.open(os.path.join(baseline_dir, fname))
            paired = stack_strips(flex_strip, base_strip)
            paired.save(os.path.join(paired_dir, fname))

    print(f"Done! Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
