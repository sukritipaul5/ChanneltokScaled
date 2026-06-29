import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from .helper import ResidualBlock, NonLocalBlock, DownSampleBlock, GroupNorm, Swish, UpSampleBlock
    from .fsq import FSQQuantizer
    from .fsq2 import FSQQuantizer2
    from .lfq import LFQ
    from .latent_mask_head import LatentMaskHead
except ImportError:  # Allows running as a script (python models/vqgan_ae.py ...)
    from models.helper import ResidualBlock, NonLocalBlock, DownSampleBlock, GroupNorm, Swish, UpSampleBlock
    from models.fsq import FSQQuantizer
    from models.fsq2 import FSQQuantizer2
    from models.lfq import LFQ
    from models.latent_mask_head import LatentMaskHead


def register_contiguous_gradients(module: nn.Module, force: bool = False) -> None:
    for param in module.parameters():
        if not force and getattr(param, "_contiguous_hooked", False):
            continue
        param.register_hook(lambda grad: grad.contiguous() if grad is not None else grad)
        param._contiguous_hooked = True


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        channels = getattr(
            config.model,
            'encoder_channels',
            [128, 128, 128, 256, 256, 512]
        )
        attn_resolutions = getattr(
            config.model,
            'encoder_attn_resolutions',
            getattr(config.model, 'attn_resolutions', [16])
        )
        if not getattr(config.model, 'use_intermediate_attn', True):
            attn_resolutions = []
        num_res_blocks = 2
        self.use_flash_attn = getattr(config.model, 'use_flash_attn', False)
        self.flash_attn_heads = getattr(config.model, 'flash_attn_heads', 8) 
        self.use_final_attn = getattr(config.model, 'use_final_attn', True)

        input_resolution = getattr(config.model, 'input_resolution', 256)
        layers = [nn.Conv2d(config.model.image_channels, channels[0], 3, 1, 1)]
        resolution = input_resolution
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(
                        in_channels,
                        use_flash_attn=self.use_flash_attn,
                        num_heads=self.flash_attn_heads
                    ))
            if i != len(channels) - 2:
                layers.append(DownSampleBlock(channels[i+1]))
                layers.append(Swish())
                resolution //= 2
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        
        if self.use_final_attn:
            layers.append(NonLocalBlock(
                channels[-1],
                use_flash_attn=self.use_flash_attn,
                num_heads=self.flash_attn_heads
            ))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], config.model.latent_dim, 3, 1, 1))
        self.model = nn.Sequential(*layers)
        
        self.default_spatial_size = resolution
        # Check if we need to adjust spatial dimensions
        self.target_spatial_size = getattr(config.model, 'target_spatial_size', 16)
        self.adjust_spatial = self.target_spatial_size != self.default_spatial_size
        
        if self.adjust_spatial:
            current_size = self.default_spatial_size
            target = self.target_spatial_size
            if current_size > target:
                spatial_layers = []
                while current_size > target:
                    if current_size % 2 != 0:
                        raise ValueError(f"Cannot downsample from {self.default_spatial_size} to {target}")
                    spatial_layers.append(DownSampleBlock(config.model.latent_dim))
                    spatial_layers.append(Swish())
                    current_size //= 2
                self.spatial_adjust = nn.Sequential(*spatial_layers)
            else:
                scale = target // current_size
                self.spatial_adjust = nn.ConvTranspose2d(
                    config.model.latent_dim,
                    config.model.latent_dim,
                    kernel_size=scale * 2,
                    stride=scale,
                    padding=scale // 2
                )

    def forward(self, x):
        x = self.model(x)
        
        # Apply spatial adjustment if needed
        if self.adjust_spatial:
            x = self.spatial_adjust(x)
            
        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.use_decoder_mask = getattr(config.model, 'use_decoder_mask', False)
        self.use_flash_attn = getattr(config.model, 'use_flash_attn', False)
        self.flash_attn_heads = getattr(config.model, 'flash_attn_heads', 8)
        self.use_final_attn = getattr(config.model, 'use_final_attn', True)
        
        attn_resolutions = getattr(
            config.model,
            'decoder_attn_resolutions',
            getattr(config.model, 'attn_resolutions', [16])
        )
        if not getattr(config.model, 'use_intermediate_attn', True):
            attn_resolutions = []
        ch_mult = getattr(
            config.model,
            'decoder_channels',
            [128, 128, 256, 256, 512]
        )
        num_resolutions = len(ch_mult)
        block_in = ch_mult[num_resolutions-1]
        
        # Default latent spatial size is 16x16
        self.default_spatial_size = 16
        
        # Get target spatial size from config if provided, otherwise use default
        self.target_spatial_size = getattr(config.model, 'target_spatial_size', self.default_spatial_size)
        
        # Calculate the starting resolution based on target spatial size
        curr_res = self.target_spatial_size
        
        # Calculate how many additional upsampling layers we need to get back to 16x16
        # This should be symmetric with encoder's spatial adjustment
        if self.target_spatial_size == 4:  # Need 2 steps: 4→8→16
            self.extra_upsample_needed = 2
        elif self.target_spatial_size == 8:  # Need 1 step: 8→16
            self.extra_upsample_needed = 1
        else:  # target_spatial_size == 16, no extra upsampling needed
            self.extra_upsample_needed = 0
        
        # For channel-wise mask, we need to double the input channels when mask is used
        # This is because we're concatenating a mask with the same channel dimension as the latent
        in_channels = config.model.latent_dim * 2 if self.use_decoder_mask else config.model.latent_dim

        
        layers = [
            nn.Conv2d(in_channels, block_in, kernel_size=3, stride=1, padding=1),
            ResidualBlock(block_in, block_in)]
        
        if self.use_final_attn:
            layers.append(NonLocalBlock(
                block_in,
                use_flash_attn=self.use_flash_attn,
                num_heads=self.flash_attn_heads
            ))

        layers.append(ResidualBlock(block_in, block_in))
        
        
        # Add extra upsampling layers if needed to compensate for smaller latent spatial size
        for _ in range(self.extra_upsample_needed):
            layers.append(UpSampleBlock(block_in))
            layers.append(Swish())
            curr_res = curr_res * 2

        for i in reversed(range(num_resolutions)):
            block_out = ch_mult[i]
            for i_block in range(3):
                layers.append(ResidualBlock(block_in, block_out))
                block_in = block_out
                if curr_res in attn_resolutions:
                    layers.append(NonLocalBlock(
                        block_in,
                        use_flash_attn=self.use_flash_attn,
                        num_heads=self.flash_attn_heads
                    ))
            if i != 0:
                layers.append(UpSampleBlock(block_in))
                layers.append(Swish())
                curr_res = curr_res * 2

        layers.append(GroupNorm(block_in))
        layers.append(Swish())
        layers.append(nn.Conv2d(block_in, config.model.image_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, x, mask=None):
        if self.use_decoder_mask:
            if mask is None:
                raise ValueError("Decoder mask is enabled, but no mask was provided.")
            
            # Check mask dimensions and expand to match spatial dimensions if needed
            if mask.dim() == 2:  # [B, C]
                # Expand from [B, C] to [B, C, H, W]
                mask = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
            elif mask.dim() == 4 and mask.shape[2] == 1 and mask.shape[3] == 1:  # [B, C, 1, 1]
                # Expand from [B, C, 1, 1] to [B, C, H, W]
                mask = mask.expand(-1, -1, x.shape[2], x.shape[3])
            elif mask.dim() != 4 or mask.shape[2] != x.shape[2] or mask.shape[3] != x.shape[3]:
                # If dimensions don't match, raise an error
                raise ValueError(f"Mask shape {mask.shape} doesn't match expected spatial dimensions {x.shape}")
                
            # Concatenate along channel dimension
            x = torch.cat([x, mask], dim=1)

        # Log shape information in debug mode
        if hasattr(self, 'debug') and self.debug:
            print(f"Decoder input shape: {x.shape}")
        
        # Process through the model
        x = self.model(x)
        
        # Log output shape in debug mode
        if hasattr(self, 'debug') and self.debug:
            print(f"Decoder output shape: {x.shape}")
            
        return x

class VQGANAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Ensure image_channels is set in config.model, default to 3 if not present
        if not hasattr(config.model, 'image_channels'):
            config.model.image_channels = 3

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        register_contiguous_gradients(self.encoder)
        register_contiguous_gradients(self.decoder)

        self.config=config #added

        # Quantizer setup: FSQ (default for backward compatibility), LFQ, or BSQ
        quantizer_type = getattr(config.model, 'quantizer', 'FSQ')  # Default to FSQ
        self.quantizer_type = quantizer_type

        if quantizer_type == 'FSQ':
            self.quantizer = FSQQuantizer(
                levels=config.model.quantizer_levels,
                dim=config.model.quantizer_dim,
                num_codebooks=config.model.num_codebooks,
                keep_num_codebooks_dim=config.model.keep_num_codebooks_dim,
                scale=config.model.quantizer_scale,
                quantize_channels=config.model.quantize_channels,
                use_conv_projections=getattr(config.model, 'use_conv_projections', False)
            )
            register_contiguous_gradients(self.quantizer)

        elif quantizer_type == 'FSQ2':
            self.quantizer = FSQQuantizer2(
                levels=config.model.quantizer_levels,
                dim=config.model.quantizer_dim,
                num_codebooks=config.model.num_codebooks,
                keep_num_codebooks_dim=config.model.keep_num_codebooks_dim,
                scale=config.model.quantizer_scale,
                quantize_channels=config.model.quantize_channels,
            )


        elif quantizer_type in ['LFQ', 'BSQ']:
            # LFQ or BSQ (Binary Spherical Quantization = LFQ with spherical=True)
            self.quantizer = LFQ(
                dim=config.model.quantizer_dim,
                codebook_size=getattr(config.model, 'codebook_size', 8192),
                entropy_loss_weight=getattr(config.model, 'entropy_loss_weight', 0.1),
                commitment_loss_weight=getattr(config.model, 'commitment_loss_weight', 0.25),
                diversity_gamma=getattr(config.model, 'diversity_gamma', 1.0),
                spherical=(quantizer_type == 'BSQ'),  # BSQ uses spherical quantization
                num_codebooks=config.model.num_codebooks,
                keep_num_codebooks_dim=config.model.keep_num_codebooks_dim,
                quantize_channels=config.model.quantize_channels,
            )
            #register_contiguous_gradients(self.quantizer)
        else:
            raise ValueError(f"Unknown quantizer type: {quantizer_type}. Supported: FSQ, LFQ, BSQ")

        print(f"VQGANAutoencoder initialized with quantizer: {quantizer_type}")

        #Flexible length tok. module
        self.use_latent_mask = config.model.get('use_latent_mask', False)

        if self.use_latent_mask:
            self.latent_mask_head = LatentMaskHead(
                latent_dim=config.model.latent_dim,
                t_min=config.model.get('t_min', 0.1),
                t_max=config.model.get('t_max', 1.0),
                sampling_strategy=config.model.get('sampling_strategy', 'uniform')
            )
            # Pre-allocate channel indices for masking to avoid recreation every forward pass
            self.register_buffer('ch_indices', torch.arange(config.model.latent_dim).view(1, -1, 1, 1), persistent=False)
        else:
            self.latent_mask_head = None
        print(f"VQGANAutoencoder initialized with latent masking: {self.use_latent_mask}")


    def _resolve_mask_toggle(self, mask_toggle, batch_size, device, dtype):
        """Normalize mask_toggle to a broadcastable tensor."""
        if mask_toggle is None:
            return torch.zeros((batch_size, 1, 1, 1), device=device, dtype=dtype)

        toggle = mask_toggle.to(device=device, dtype=dtype)

        if toggle.dim() == 0:
            toggle = toggle.view(1, 1, 1, 1)
        elif toggle.dim() == 1:
            toggle = toggle.view(-1, 1, 1, 1)
        elif toggle.dim() == 2:
            toggle = toggle.view(toggle.shape[0], toggle.shape[1], 1, 1)
        elif toggle.dim() != 4:
            raise ValueError(f"Unsupported mask_toggle shape: {toggle.shape}")

        batch_dim = toggle.shape[0]
        if batch_dim == 1 and batch_size > 1:
            toggle = toggle.expand(batch_size, -1, -1, -1)
        elif batch_dim not in (1, batch_size):
            raise ValueError(
                f"mask_toggle batch dimension {batch_dim} does not match input batch size {batch_size}"
            )

        return toggle.clamp(0.0, 1.0)

    def _apply_latent_mask(self, encoded, mask_toggle_tensor):
        """Apply adaptive channel masking prior to quantization."""
        masked_encoded, mask, t_values = self.latent_mask_head(encoded)
        B, C, H, W = masked_encoded.shape

        t_int = t_values.to(dtype=torch.long).clamp(min=0, max=C)
        mask_bool = (self.ch_indices < t_int.view(B, 1, 1, 1))

        masked_input = torch.where(mask_bool, masked_encoded, masked_encoded.detach())
        quantizer_input = mask_toggle_tensor * masked_input + (1.0 - mask_toggle_tensor) * encoded

        mask_scale = mask_toggle_tensor * mask_bool.to(encoded.dtype) + (1.0 - mask_toggle_tensor)
        mask_active = mask_toggle_tensor.detach().amax()
        mask_info = {
            't_values': t_values.detach(),
            'adaptive_quantization': mask_active,
            'mask_toggle': mask_toggle_tensor.detach()
        }

        return quantizer_input, mask_scale, mask_info

    def forward(self, x, mask_toggle=None, inference_t=None,return_indices=False):
        encoded = self.encoder(x)
        commit_loss = torch.tensor(0.0, device=x.device)
        quantizer_aux_loss = torch.tensor(0.0, device=x.device)
        indices = None
        mask_info = None

        if hasattr(self, 'quantizer'):
            if inference_t is not None:
                B, C, H, W = encoded.shape

                if torch.is_tensor(inference_t):
                    t_tensor = inference_t.to(device=encoded.device)
                else:
                    t_tensor = torch.tensor(inference_t, device=encoded.device)

                if t_tensor.dim() == 0:
                    t_tensor = t_tensor.view(1)
                if t_tensor.shape[0] not in (1, B):
                    raise ValueError(
                        f"inference_t batch dimension {t_tensor.shape[0]} does not match batch size {B}"
                    )
                if t_tensor.shape[0] == 1 and B > 1:
                    t_tensor = t_tensor.expand(B)

                t_tensor = t_tensor.to(dtype=torch.long).clamp(min=0, max=C)

                if self.latent_mask_head is not None:
                    mask = self.latent_mask_head.build_mask_from_t(t_tensor, (H, W))
                    mask = mask.to(device=encoded.device, dtype=encoded.dtype)
                else:
                    if not hasattr(self, 'ch_indices'):
                        self.register_buffer(
                            'ch_indices',
                            torch.arange(C, device=encoded.device).view(1, -1, 1, 1),
                            persistent=False
                        )
                    elif self.ch_indices.shape[1] != C:
                        self.ch_indices = torch.arange(C, device=encoded.device).view(1, -1, 1, 1)

                    mask_bool = (self.ch_indices < t_tensor.view(B, 1, 1, 1))
                    mask = mask_bool.to(dtype=encoded.dtype).expand(-1, -1, H, W)

                quantizer_input = encoded * mask
                mask_scale = mask
                mask_info = {
                    'inference_t': t_tensor.detach(),
                    'adaptive_quantization': False,
                    'mask_scale': mask.detach()
                }
            elif self.latent_mask_head is not None:
                mask_toggle_tensor = self._resolve_mask_toggle(
                    mask_toggle,
                    batch_size=encoded.shape[0],
                    device=encoded.device,
                    dtype=encoded.dtype
                )
                quantizer_input, mask_scale, mask_info = self._apply_latent_mask(encoded, mask_toggle_tensor)
            else:
                quantizer_input = encoded
                mask_scale = encoded.new_tensor(1.0)
            

            quantized, indices, quantizer_aux_loss = self.quantizer(quantizer_input)
            quantized = quantized * mask_scale
            indices = indices.detach()
        else:
            quantized = encoded
            mask_scale = encoded.new_tensor(1.0)

        decoded = self.decoder(quantized)

        info = {
            'commit_loss': commit_loss,
            'quantizer_aux_loss': quantizer_aux_loss,
            'mask_info': mask_info,
        }
        if return_indices:
            info['indices'] = indices.detach()
        return decoded, info


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from omegaconf import OmegaConf

    def _resolve_default_path(default_entry, config_dir: Path, project_root: Path):
        """Try to resolve a default entry like 'base' to an actual YAML file."""
        if not isinstance(default_entry, str):
            return None

        candidates = []
        if default_entry.endswith(".yaml"):
            candidates.append(default_entry)
        else:
            candidates.extend([f"{default_entry}.yaml", default_entry])

        for name in candidates:
            path = Path(name)
            if path.is_absolute() and path.exists():
                return path
            local_candidate = (config_dir / name).resolve()
            if local_candidate.exists():
                return local_candidate
            repo_candidate = (project_root / "configs" / name).resolve()
            if repo_candidate.exists():
                return repo_candidate
        return None

    def _load_config(config_path: Path):
        config = OmegaConf.load(config_path)
        defaults = config.get("defaults", [])
        if defaults:
            base_cfgs = []
            project_root = Path(__file__).resolve().parents[1]
            for default in defaults:
                default_path = _resolve_default_path(default, config_path.parent, project_root)
                if default_path is not None:
                    base_cfgs.append(OmegaConf.load(default_path))
            if base_cfgs:
                config = OmegaConf.merge(*base_cfgs, config)
        return config

    def _count_parameters(module: nn.Module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    parser = argparse.ArgumentParser(description="Load VQGAN config and print parameter counts.")
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to the YAML config file to load.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_config(cfg_path)

    model = VQGANAutoencoder(cfg)
    total_params, trainable_params = _count_parameters(model)
    print(f"Loaded config: {cfg_path}")
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")

    input_image = torch.randn(4, 3, 256, 256).to("cuda")
    model = model.to("cuda")
    output = model(input_image)
    print(output[0].shape)
    # print the model architecture