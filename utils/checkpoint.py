"""Consolidated checkpoint loading utilities.

Replaces duplicated load_config / load_checkpoint scattered across
inference.py, extract_imagenet100k_latents.py, and others.
"""

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> OmegaConf:
    """Load a YAML config, resolving ``defaults:`` by merging base configs."""
    config_path = Path(config_path).expanduser().resolve()
    config = OmegaConf.load(config_path)

    defaults = config.get("defaults", [])
    if defaults:
        project_root = Path(__file__).resolve().parents[1]
        config_dir = config_path.parent

        base_cfgs = []
        for entry in defaults:
            resolved = _resolve_default_path(entry, config_dir, project_root)
            if resolved is not None:
                base_cfgs.append(OmegaConf.load(resolved))

        if base_cfgs:
            config = OmegaConf.merge(*base_cfgs, config)

    return config


def _resolve_default_path(entry: str, config_dir: Path, project_root: Path):
    """Try ``configs/{entry}.yaml``, then siblings of the config file."""
    if not isinstance(entry, str):
        return None

    candidates = [entry] if entry.endswith(".yaml") else [f"{entry}.yaml", entry]

    for name in candidates:
        p = Path(name)
        if p.is_absolute() and p.exists():
            return p
        local = (config_dir / name).resolve()
        if local.exists():
            return local
        repo = (project_root / "configs" / name).resolve()
        if repo.exists():
            return repo
    return None


# ---------------------------------------------------------------------------
# State-dict cleaning
# ---------------------------------------------------------------------------

_SKIP_SUBSTRINGS = ("perceptual_loss", "discriminator", "lpips")
_VALID_PREFIXES = ("encoder.", "decoder.", "quantizer.", "latent_mask_head.", "ch_indices")
_WRAPPER_PREFIXES = ("module.", "model.", "_orig_mod.")


def clean_state_dict(state_dict: dict) -> dict:
    """Strip wrapper prefixes and filter non-model keys from a state dict."""
    cleaned = {}
    for key, value in state_dict.items():
        if any(s in key for s in _SKIP_SUBSTRINGS):
            continue

        new_key = key
        for prefix in _WRAPPER_PREFIXES:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]

        if new_key.startswith(_VALID_PREFIXES):
            cleaned[new_key] = value

    return cleaned


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: str,
    config,
    device: str = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Instantiate a model from *config* and load weights from *checkpoint_path*.

    Returns the model in eval mode on *device*.
    """
    from models import get_model_class

    model_class = get_model_class(config.model.name)
    model = model_class(config)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = clean_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()
    return model
