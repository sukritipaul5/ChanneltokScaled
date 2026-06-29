import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class LatentMaskHead(nn.Module):
    def __init__(self, latent_dim, t_min=0.1, t_max=1.0, sampling_strategy='uniform'):
        """
        Flexible length tokenizer that masks channels in the latent space.
        For each sample in the batch, it randomly selects a number of channels to keep
        between t_min * latent_dim and t_max * latent_dim.

        Args:
            latent_dim: The embedding dimension of the latent space (total number of channels)
            t_min: Minimum fraction of channels to keep (default: 0.1)
            t_max: Maximum fraction of channels to keep (default: 1.0)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.t_min = t_min
        self.t_max = t_max

        # Pre-allocate channel indices buffer to avoid recreation every forward pass
        self.register_buffer('channel_indices', torch.arange(latent_dim), persistent=False)
        self.sampling_strategy = sampling_strategy
        
    def forward(self, x):
        """
        Apply channel masking to the input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, latent_dim, height, width]
                for CNN-style latents
        
        Returns:
            masked_x: Tensor with the same shape as x but with some channels masked
            mask: The binary mask applied to the tensor
            t_values: The number of channels kept for each sample in the batch
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Determine shape format
        if len(x.shape) == 4:  # CNN format: [batch_size, latent_dim, height, width]
            B, C, H, W = x.shape
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}. Expected 4D tensor [B, C, H, W]")
        
        # Sample t_fractions for the batch - vectorized approach for torch.compile compatibility
        if self.sampling_strategy == 'uniform':
            t_fractions = torch.empty(batch_size, device=device).uniform_(self.t_min, self.t_max)

        elif self.sampling_strategy == 'bias_lower':
            # 75% of samples from lower 25% of range
            span = self.t_max - self.t_min
            selector = torch.rand(batch_size, device=device)
            interval_noise = torch.rand(batch_size, device=device)
            lower_width = 0.25 * span
            upper_width = span - lower_width
            lower_samples = self.t_min + interval_noise * lower_width
            upper_start = self.t_min + lower_width
            upper_samples = upper_start + interval_noise * upper_width
            t_fractions = torch.where(selector < 0.75, lower_samples, upper_samples)


        elif self.sampling_strategy == 'bias_higher':
            # 75% of samples from upper 25% of range
            span = self.t_max - self.t_min
            selector = torch.rand(batch_size, device=device)
            interval_noise = torch.rand(batch_size, device=device)
            lower_width = 0.75 * span  # start of upper 25%
            upper_width = span - lower_width
            lower_samples = self.t_min + interval_noise * lower_width
            upper_start = self.t_min + lower_width
            upper_samples = upper_start + interval_noise * upper_width
            t_fractions = torch.where(selector < 0.75, upper_samples, lower_samples)
            

        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        # Calculate number of channels to keep for each sample (vectorized)
        # Clamp to ensure at least 1 channel is kept
        num_channels_to_keep = torch.clamp(
            (t_fractions * self.latent_dim).long(), 
            min=1, 
            max=self.latent_dim
        )  # Shape: [batch_size]

        # Create masks efficiently using broadcasting
        # Use pre-allocated channel indices buffer
        # Broadcast comparison: [batch_size, 1] < [1, latent_dim] -> [batch_size, latent_dim]
        mask = (self.channel_indices.unsqueeze(0) < num_channels_to_keep.unsqueeze(1)).float()
        
        # Expand mask to [batch_size, latent_dim, height, width]
        # Use expand (creates view, not copy) for memory efficiency
        mask = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        
        # Apply the mask to the input tensor
        masked_x = x * mask

        # Return tensor for t_values to avoid torch.compile issues with tolist()
        # Detach mask to break reference chain and prevent memory accumulation
        # The calling code can convert to list if needed for logging
        return masked_x, mask.detach(), num_channels_to_keep
    
    def log_metrics(self, t_values, logger=None, step=None):
        """
        Log masking metrics.
        
        Args:
            t_values: Tensor or list of t values (number of channels kept) for the batch
            logger: Logger instance (e.g., wandb)
            step: Current training step
        """
        if not logger:
            return
            
        # Convert tensor to list if needed
        if hasattr(t_values, 'tolist'):
            t_values_list = t_values.tolist()
        else:
            t_values_list = t_values
            
        # Calculate statistics
        avg_channels = sum(t_values_list) / len(t_values_list)
        min_channels = min(t_values_list)
        max_channels = max(t_values_list)
        avg_percentage = 100 * avg_channels / self.latent_dim
        
        # Log to wandb if available
        if hasattr(logger, 'log'):
            logger.log({
                "latent_mask/avg_channels": avg_channels,
                "latent_mask/min_channels": min_channels,
                "latent_mask/max_channels": max_channels,
                "latent_mask/avg_percentage": avg_percentage
            }, step=step)

    def build_mask_from_t(self, t_values, spatial_shape):
        """
        Deterministically build a binary mask using explicit channel counts.

        Args:
            t_values: Tensor or scalar specifying number of channels to keep per sample.
            spatial_shape: Tuple for spatial dims (H, W); use empty tuple for latent vectors.

        Returns:
            mask: Float tensor shaped [B, C, *spatial_shape] with ones for active channels.
        """
        if not torch.is_tensor(t_values):
            t_values = torch.tensor(t_values, device=self.channel_indices.device, dtype=torch.long)

        if t_values.dim() == 0:
            t_values = t_values.view(1)

        t_values = t_values.to(device=self.channel_indices.device, dtype=torch.long)
        batch_size = t_values.shape[0]

        # Expand or clamp to latent dimension bounds
        t_values = torch.clamp(t_values, min=0, max=self.latent_dim)

        mask = (self.channel_indices.unsqueeze(0) < t_values.view(batch_size, 1)).float()
        mask = mask.unsqueeze(-1).unsqueeze(-1)
        if spatial_shape:
            mask = mask.expand(-1, -1, *spatial_shape)

        return mask
