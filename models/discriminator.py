"""Simple PatchGAN discriminator for adversarial training"""
import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    Simple PatchGAN discriminator for image reconstruction.
    Uses GroupNorm instead of BatchNorm for better DDP compatibility.
    """
    def __init__(self, input_channels=3, base_channels=64, n_layers=3):
        super().__init__()
        layers = []

        # First layer: no normalization
        layers.extend([
            nn.Conv2d(input_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])

        # Intermediate layers with GroupNorm (no running stats, DDP-friendly)
        nf = base_channels
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            layers.extend([
                nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(num_groups=min(32, nf), num_channels=nf),
                nn.LeakyReLU(0.2, inplace=True)
            ])

        # Final layers
        nf_prev = nf
        nf = min(nf * 2, 512)
        layers.extend([
            nn.Conv2d(nf_prev, nf, kernel_size=4, stride=1, padding=1),
            nn.GroupNorm(num_groups=min(32, nf), num_channels=nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1)
        ])

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.GroupNorm):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)
