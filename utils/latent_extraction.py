"""Latent extraction utilities for converting images to token indices.

Consolidates logic from extract_imagenet_latents.py and
extract_imagenet100k_latents.py into reusable components.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import torchvision.utils as vutils


class IndexExtractor(nn.Module):
    """Wraps a VQGANAutoencoder to return only quantized indices.

    Needed for ``DataParallel`` which cannot gather dicts with mixed types.

    Args:
        model: A VQGAN autoencoder with ``forward(x, inference_t=..., return_indices=True)``.
        inference_t: Number of channels to use (``None`` = all).
    """

    def __init__(self, model, inference_t=None):
        super().__init__()
        self.model = model
        self.inference_t = inference_t

    def forward(self, x):
        _, info = self.model(x, inference_t=self.inference_t, return_indices=True)
        return info["indices"]


# ---------------------------------------------------------------------------
# Batch unpacking
# ---------------------------------------------------------------------------

def _unpack_batch(batch):
    """Handle different batch formats and return (images, labels, keys).

    Supported formats:
    - DALI: ``[{'data': tensor, 'label': tensor}]``
    - Standard tuple: ``(images, labels)``
    - Plain tensor

    Returns:
        images: Tensor of shape ``[B, C, H, W]``.
        labels: Tensor of shape ``[B]`` or ``None``.
        keys: List of string identifiers (may be ``None``).
    """
    labels = None
    keys = None

    if isinstance(batch, (tuple, list)) and len(batch) >= 1:
        if isinstance(batch[0], dict):
            # DALI format: list of dicts
            images = batch[0]["data"]
            if "label" in batch[0]:
                labels = batch[0]["label"]
                if hasattr(labels, "shape"):
                    labels = labels.reshape(-1)
            keys = batch[0].get("__key__", None)
        else:
            # Standard (image, label, ...) tuple
            images = batch[0]
            if len(batch) >= 2:
                labels = batch[1]
    elif isinstance(batch, dict):
        # WebDataset format
        images = batch.get("jpg", batch.get("png"))
        labels = batch.get("cls", batch.get("label"))
        keys = batch.get("__key__")
    else:
        images = batch

    return images, labels, keys


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def _save_reconstruction_samples(model, images, sample_dir, num_samples=8):
    """Save original and reconstructed images for visual verification."""
    with torch.no_grad():
        n = min(num_samples, images.size(0))
        sample_images = images[:n]

        output = model(sample_images, inference_t=512)
        reconstructed = output[0] if isinstance(output, tuple) else output

        sample_01 = (sample_images + 1.0) / 2.0
        recon_01 = (reconstructed + 1.0) / 2.0
        sample_01 = sample_01.clamp(0, 1)
        recon_01 = recon_01.clamp(0, 1)

        for i in range(n):
            orig_np = (sample_01[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(orig_np).save(os.path.join(sample_dir, f"original_{i:03d}.png"))

            recon_np = (recon_01[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(recon_np).save(os.path.join(sample_dir, f"reconstruction_{i:03d}.png"))

        combined = torch.cat([sample_01, recon_01], dim=0)
        grid = vutils.make_grid(combined, nrow=n, normalize=False, scale_each=False)
        grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(grid_np).save(os.path.join(sample_dir, "comparison_grid.png"))

        print(f"Saved {n} reconstruction samples to {sample_dir}")


def extract_latents(
    dataloader,
    model,
    device,
    inference_t=None,
    save_samples=False,
    sample_dir=None,
    num_samples=None,
):
    """Extract token indices from all images in a dataloader.

    When *inference_t* is not ``None`` the model is wrapped in
    :class:`IndexExtractor` so ``DataParallel`` can be used.  When it is
    ``None`` the raw encoder + quantizer path is used instead.

    Args:
        dataloader: PyTorch DataLoader yielding batches of images.
        model: VQGAN autoencoder (unwrapped -- not inside DataParallel).
        device: Torch device.
        inference_t: Fixed token budget (number of channels).  ``None`` = all.
        save_samples: If ``True``, save reconstruction PNGs from the first batch.
        sample_dir: Directory for sample images (required when *save_samples*).
        num_samples: Optional maximum number of images to extract.

    Returns:
        ``(indices_np, names_list, labels_np)``
        - *indices_np*: int32 array of shape ``[N, T]``
        - *names_list*: list of string identifiers
        - *labels_np*: int64 array of shape ``[N]``
    """
    all_indices = []
    all_labels = []
    all_names = []
    samples_saved = False
    num_extracted = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting latents")):
            images, labels, keys = _unpack_batch(batch)

            if num_samples is not None:
                remaining = num_samples - num_extracted
                if remaining <= 0:
                    break
                if images.size(0) > remaining:
                    images = images[:remaining]
                    if labels is not None:
                        labels = labels[:remaining]
                    if isinstance(keys, list):
                        keys = keys[:remaining]

            images = images.to(device)

            # Build default keys when the dataloader does not supply them
            if keys is None:
                keys = [f"img_{batch_idx}_{i}" for i in range(images.size(0))]

            # ------ encode + quantize ------
            if inference_t is not None:
                # Use full forward with return_indices
                _, info = model(images, inference_t=inference_t, return_indices=True)
                indices = info["indices"]
            else:
                # Fallback: raw encoder -> quantizer
                encoded = model.encoder(images)
                if hasattr(model, "quantizer"):
                    _, indices, _ = model.quantizer(encoded)
                else:
                    raise RuntimeError("Model has no quantizer attribute")

            # Normalise shape to [B, T]
            if indices.dim() == 3:  # [B, H, W] spatial quantization
                indices = indices.reshape(indices.size(0), -1)

            indices_np = indices.cpu().numpy().astype(np.int32)
            all_indices.append(indices_np)

            # Labels
            if labels is not None:
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                if labels.ndim > 1:
                    labels = labels.reshape(-1)
                all_labels.append(labels.astype(np.int64))
            else:
                all_labels.append(np.full(images.size(0), -1, dtype=np.int64))

            # Names
            if isinstance(keys, list):
                all_names.extend(keys)
            else:
                all_names.extend([keys] * images.size(0))

            # Optional reconstruction samples
            if save_samples and not samples_saved and sample_dir is not None:
                os.makedirs(sample_dir, exist_ok=True)
                _save_reconstruction_samples(model, images, sample_dir)
                samples_saved = True

            num_extracted += images.size(0)

    indices_np = np.concatenate(all_indices, axis=0)
    labels_np = np.concatenate(all_labels, axis=0) if all_labels else np.full(len(indices_np), -1, dtype=np.int64)

    return indices_np, all_names, labels_np


# ---------------------------------------------------------------------------
# Saving / validation
# ---------------------------------------------------------------------------

def save_latents_npz(path, indices, labels, names, codebook_size=None, num_tokens_per_image=None):
    """Save extracted latents as a compressed NPZ file.

    Args:
        path: Output file path (should end in ``.npz``).
        indices: int32 array ``[N, T]``.
        labels: int64 array ``[N]``.
        names: List or array of string identifiers.
        codebook_size: Optional codebook size metadata.
        num_tokens_per_image: Optional tokens-per-image metadata.
    """
    save_kwargs = dict(
        indices=indices,
        labels=labels,
        names=np.array(names) if not isinstance(names, np.ndarray) else names,
        shape=np.array(indices.shape),
    )
    if codebook_size is not None:
        save_kwargs["codebook_size"] = np.int64(codebook_size)
    if num_tokens_per_image is not None:
        save_kwargs["num_tokens_per_image"] = np.int64(num_tokens_per_image)

    np.savez_compressed(path, **save_kwargs)


def validate_first_batch(indices):
    """Sanity-check the first batch of indices for range and uniqueness.

    Args:
        indices: Tensor or ndarray of token indices from a single batch.

    Raises:
        AssertionError: If indices are out of a reasonable range.
    """
    if isinstance(indices, np.ndarray):
        indices = torch.from_numpy(indices)

    unique = torch.unique(indices)
    n_unique = len(unique)
    lo, hi = indices.min().item(), indices.max().item()

    print(f"  [validation] first batch: {tuple(indices.shape)}, "
          f"range [{lo}, {hi}], unique values: {n_unique}")

    assert hi <= 65535, f"Index out of range: max={hi}"
    assert lo >= 0, f"Negative index: min={lo}"

    if n_unique < 100:
        print(f"  WARNING: only {n_unique} unique values in first batch -- may indicate a problem")
    else:
        print(f"  OK: {n_unique} unique values (> 100 threshold)")
