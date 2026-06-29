import torch
import torch.nn.functional as F


def _mse_loss(recon: torch.Tensor, target: torch.Tensor, **_) -> torch.Tensor:
    return F.mse_loss(recon, target)


def _l1_loss(recon: torch.Tensor, target: torch.Tensor, **_) -> torch.Tensor:
    return F.l1_loss(recon, target)


def _perceptual_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    perceptual_fn,
    **_
) -> torch.Tensor:
    if perceptual_fn is None:
        raise ValueError("Perceptual loss requested but perceptual_fn is None.")
    return perceptual_fn(recon, target)



def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Standard hinge loss for the discriminator."""
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def hinge_g_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    """Standard hinge loss for the generator."""
    return -torch.mean(logits_fake)


def vanilla_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Vanilla GAN discriminator loss with softplus stabilization."""
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_g_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    """Vanilla GAN generator loss with softplus stabilization."""
    return torch.mean(F.softplus(-logits_fake))


def _adversarial_g_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    discriminator=None,
    current_epoch: int = 0,
    gan_start_epoch: int = 0,
    **_
) -> torch.Tensor:
    """
    Generator adversarial loss with epoch-based activation.

    Args:
        recon: Reconstructed images
        target: Target images (unused, for API consistency)
        discriminator: Discriminator network
        current_epoch: Current training epoch
        gan_start_epoch: Epoch to start adversarial training

    Returns:
        Generator adversarial loss (0.0 if not yet active)
    """
    if discriminator is None or current_epoch < gan_start_epoch:
        return torch.tensor(0.0, device=recon.device, dtype=recon.dtype)

    logits_fake = discriminator(recon)
    return hinge_g_loss(logits_fake)


# def gram_matrix(features: torch.Tensor) -> torch.Tensor:
#     """Compute Gram matrix for features shaped [B, C, H, W], normalized by elements per map."""
#     if features.dim() != 4:
#         raise ValueError(f"Expected 4D tensor for Gram computation, got shape {tuple(features.shape)}")
#     b, c, h, w = features.shape
#     reshaped = features.view(b, c, h * w)
#     reshaped = reshaped.float()
#     gram = torch.bmm(reshaped, reshaped.transpose(1, 2))
#     return gram / (c * h * w)



def l1_charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6, **_) -> torch.Tensor:
    diff = pred - target
    error = torch.sqrt(diff.pow(2) + eps)
    return error.mean()



def gram_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6, **_) -> torch.Tensor:
    """Gram loss with numerical stability."""
    def compute_gram(x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1)
        # Normalize each feature map
        x = x / (x.norm(dim=2, keepdim=True) + eps)
        gram = torch.bmm(x, x.transpose(1, 2)) / (h * w + eps)
        return gram
    
    gram_pred = compute_gram(pred)
    gram_target = compute_gram(target).detach()  # Detach target
    
    return F.mse_loss(gram_pred, gram_target)


LOSS_REGISTRY = {
    "l2": _mse_loss,
    "mse": _mse_loss,
    "l1": _l1_loss,
    "mae": _l1_loss,
    "perceptual": _perceptual_loss,
    "lpips": _perceptual_loss,
    "gram": gram_loss,
    "l1_charbonnier": l1_charbonnier_loss,
    "adversarial": _adversarial_g_loss,
}
