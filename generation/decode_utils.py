"""Decoder utility functions for reconstructing images from channelwise FSQ indices.

These helpers are used by both the training loop (generation/train.py) and
standalone scripts (scripts/generate_16k.py, scripts/compute_fid_*.py).
"""

import torch
import torch.nn as nn


@torch.no_grad()
def reconstruct_from_channelwise_indices(
    model,
    indices_1d: torch.LongTensor,
    latent_hw=(4, 4),
    total_channels=None,
    device="cuda",
):
    """
    indices_1d: LongTensor (t,) one FSQ index per *channel* for channels [0..t-1]
    latent_hw: spatial size of the latent (H,W) -> usually (4,4) so H*W=16
    """
    if not hasattr(model, "quantizer"):
        raise RuntimeError("Model has no .quantizer")
    q = model.quantizer

    # Auto-detect total_channels from decoder's first conv layer
    if total_channels is None:
        total_channels = 512  # default fallback
        if hasattr(model, 'decoder'):
            for m in model.decoder.modules():
                if isinstance(m, torch.nn.Conv2d):
                    total_channels = m.in_channels
                    break

    H, W = latent_hw
    if H * W != 16:
        raise ValueError(
            f"FSQ projection output dim is 16, but H*W={H*W}. "
            f"Set --latent_hw 4 4 (or match your training)."
        )

    t = indices_1d.numel()
    idx = indices_1d.view(1, t, 1).to(device)  # (1, t, 1)

    codes = q.indices_to_codes(idx, project_out=True)  # shape varies by impl

    # --- normalize to (1, 16, t) ---
    c = codes
    # ensure batch dim at front
    if c.dim() == 3 and c.shape[0] != 1:
        # likely (16, t, 1) or (16, t)
        c = c.unsqueeze(0)  # add batch: (1, 16, t, 1) or (1, 16, t)
    elif c.dim() == 2:
        # (16, t) or (t, 16) -> add batch
        c = c.unsqueeze(0)

    # squeeze any trailing singleton spatial
    if c.dim() == 4 and c.shape[-1] == 1:
        c = c.squeeze(-1)

    # now c is 3D: (B?, A, B) in some order; find which axis is 16 and which is t
    if c.dim() != 3:
        raise ValueError(f"Unexpected codes shape after squeeze: {tuple(c.shape)}")

    bsz = c.shape[0]
    axes = list(c.shape)
    # locate 16 and t
    try:
        ax_16 = [i for i, s in enumerate(axes) if s == 16][0]
    except IndexError:
        raise ValueError(f"Could not find a 16-dim in codes shape {axes}")
    try:
        ax_t = [i for i, s in enumerate(axes) if s == t][0]
    except IndexError:
        raise ValueError(f"Could not find a t={t} dim in codes shape {axes}")

    # move to (B, 16, t)
    order = [0, 1, 2]
    # current positions of (B, 16, t) among dims (0,1,2)
    # make sure batch is dim 0
    if bsz != 1:
        raise ValueError("Batch size not 1 -- unexpected for indices-based decode.")
    # we already have batch at 0 due to unsqueeze rules above

    # if 16 or t are not at (1,2), permute
    if (ax_16, ax_t) != (1, 2):
        # build mapping from current (0,1,2) to desired (0,1,2) where 1->ax_16 and 2->ax_t
        inv = [0, ax_16, ax_t]
        # invert mapping to get permute order
        perm = [inv.index(i) for i in range(3)]
        c = c.permute(*perm).contiguous()

    if c.shape != (1, 16, t):
        raise ValueError(f"Normalized codes shape expected (1,16,t), got {tuple(c.shape)}")

    codes_16_t = c  # (1, 16, t)

    # --- place first t channels; zero the rest ---
    codes_16_all = torch.zeros(1, 16, total_channels, device=device)
    codes_16_all[:, :, :t] = codes_16_t  # fill first t, others remain zero

    # (1,16,512) -> (1,512,16) -> (1,512,4,4)
    quant_latent = codes_16_all.permute(0, 2, 1).contiguous().view(1, total_channels, H, W)

    # decode
    if hasattr(model, "decode"):
        recon = model.decode(quant_latent)
    else:
        recon = model.decoder(quant_latent)

    return recon, quant_latent


def denorm_to_uint8(x):
    if x.dim() == 4:
        x = x[0]
    # assume [-1,1] by default
    if x.min() < -0.01:
        x = (x * 0.5 + 0.5).clamp(0, 1)
    else:
        x = x.clamp(0, 1)
    return (x * 255).round().byte().permute(1, 2, 0).cpu().numpy()
