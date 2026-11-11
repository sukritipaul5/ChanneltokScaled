#!/usr/bin/env python3
"""Sample a completed full-512 AR quick-test checkpoint and decode a grid."""

import argparse
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from generation.decode_utils import denorm_to_uint8, reconstruct_from_channelwise_indices
from generation.models.gpt import GPT_models
from utils.checkpoint import load_checkpoint, load_config as load_decoder_config


def saved_arg(saved_args, name, default=None):
    if isinstance(saved_args, dict):
        return saved_args.get(name, default)
    return getattr(saved_args, name, default)


def find_checkpoint(pattern):
    candidates = []
    for path in glob(pattern):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        candidates.append((int(checkpoint.get("steps", -1)), Path(path).stat().st_mtime, path, checkpoint))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints matched {pattern}")
    _, _, path, checkpoint = max(candidates, key=lambda item: (item[0], item[1]))
    return Path(path), checkpoint


def build_model(checkpoint, device):
    args = checkpoint["args"]
    pad_token_id = int(saved_arg(args, "pad_token_id"))
    target_seq_length = int(saved_arg(args, "target_seq_length", 0))
    max_sequence_length = int(saved_arg(args, "max_sequence_length", 512))
    if target_seq_length and int(target_seq_length**0.5) ** 2 == target_seq_length:
        block_size = target_seq_length
    else:
        block_size = max_sequence_length

    drop_path_rate = float(saved_arg(args, "drop_path_rate", 0.0))
    dropout_p = 0.0 if drop_path_rate > 0 else float(saved_arg(args, "dropout_p", 0.1))
    model = GPT_models[saved_arg(args, "gpt_model")](
        vocab_size=pad_token_id + 1,
        block_size=block_size,
        num_classes=int(saved_arg(args, "num_classes")),
        cls_token_num=int(saved_arg(args, "cls_token_num")),
        model_type=saved_arg(args, "gpt_type"),
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=drop_path_rate,
        token_dropout_p=float(saved_arg(args, "token_dropout_p", 0.1)),
        use_learned_pos_emb=bool(saved_arg(args, "use_learned_pos_emb", False)),
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), args


@torch.no_grad()
def sample_sequence(model, bos_token_id, eos_token_id, codebook_size, device):
    generated = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
    for _ in range(512):
        dummy_cond = torch.zeros(1, dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(idx=generated, cond_idx=dummy_cond)
        probs = torch.softmax(logits[:, -1, :] / 1.0, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        if next_token.item() == eos_token_id:
            break
        generated = torch.cat([generated, next_token], dim=1)

    tokens = generated[0, 1:]
    return tokens[tokens < codebook_size].detach().cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-pattern",
        default="results_full512_quicktest/*/checkpoints/latest.pt",
    )
    parser.add_argument(
        "--output",
        default="results_full512_quicktest/quicktest_samples_grid.png",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for sampling and decoding")
    device = torch.device("cuda")
    checkpoint_path, checkpoint = find_checkpoint(args.checkpoint_pattern)
    if int(checkpoint.get("steps", -1)) < 3000:
        raise RuntimeError(
            f"Newest/highest-step checkpoint is only step {checkpoint.get('steps')}: {checkpoint_path}"
        )
    model, train_args = build_model(checkpoint, device)

    decoder_config_path = saved_arg(train_args, "decoder_config")
    decoder_checkpoint_path = saved_arg(train_args, "decoder_ckpt")
    decoder_config = load_decoder_config(decoder_config_path)
    decoder = load_checkpoint(decoder_checkpoint_path, decoder_config, device=device).eval()

    bos_token_id = int(saved_arg(train_args, "bos_token_id"))
    eos_token_id = int(saved_arg(train_args, "eos_token_id"))
    codebook_size = int(saved_arg(train_args, "vocab_size"))
    latent_size = int(saved_arg(train_args, "image_size", 256)) // int(
        saved_arg(train_args, "downsample_size", 64)
    )
    budgets = (64, 128, 256, 512)
    rows = []
    lengths = []
    for sample_index in range(8):
        tokens = sample_sequence(model, bos_token_id, eos_token_id, codebook_size, device)
        lengths.append(int(tokens.numel()))
        images = []
        for budget in budgets:
            truncated = tokens[:budget]
            if truncated.numel() == 0:
                raise RuntimeError(f"Sample {sample_index} produced no codebook tokens")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                reconstruction, _ = reconstruct_from_channelwise_indices(
                    decoder,
                    truncated,
                    latent_hw=(latent_size, latent_size),
                    total_channels=None,
                    device=device,
                )
            images.append(denorm_to_uint8(reconstruction))
        rows.append(np.concatenate(images, axis=1))
        print(f"sample={sample_index} valid_tokens={tokens.numel()}", flush=True)

    grid = np.concatenate(rows, axis=0)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(output_path)
    print(f"checkpoint={checkpoint_path}")
    print(f"steps={checkpoint.get('steps')}")
    print(f"sample_lengths={lengths}")
    print(f"saved={output_path.resolve()}")


if __name__ == "__main__":
    main()
