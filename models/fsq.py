import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor, int32
from typing import List, Optional
import math

from einops import rearrange

"""
Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505
Code adapted from Jax version in Appendix A.1
"""



# helper functions

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

# tensor helpers

def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()

# main class

class FSQQuantizer(Module):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks = 1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        quantize_channels: Optional[bool] = False,
        use_conv_projections: bool = False
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent = False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent = False)

        self.scale = scale
        self.quantize_channels = quantize_channels

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.has_projections = has_projections
        self.use_conv_projections = use_conv_projections and has_projections

        if has_projections:
            if self.use_conv_projections:
                self.project_in_module = nn.Conv1d(self.dim, effective_codebook_dim, kernel_size=1, bias=True)
                self.project_out_module = nn.Conv1d(effective_codebook_dim, self.dim, kernel_size=1, bias=True)
                self._register_conv_grad_hooks(self.project_in_module)
                self._register_conv_grad_hooks(self.project_out_module)
            else:
                self.project_in_module = nn.Linear(self.dim, effective_codebook_dim)
                self.project_out_module = nn.Linear(effective_codebook_dim, self.dim)
        else:
            self.project_in_module = nn.Identity()
            self.project_out_module = nn.Identity()

        self.codebook_size = self._levels.prod().item()
 

        #No need to save codebook-> deterministic!
        #implicit_codebook = self.indices_to_codes(torch.arange(self.codebook_size), project_out = False)
        #self.register_buffer("implicit_codebook", implicit_codebook, persistent = False)

    def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: Tensor) -> Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2 # Renormalize to [-1, 1].
        return quantized / half_width

    @staticmethod
    def _register_conv_grad_hooks(module: nn.Conv1d) -> None:
        module.weight.register_hook(lambda grad: grad.contiguous() if grad is not None else grad)
        if module.bias is not None:
            module.bias.register_hook(lambda grad: grad.contiguous() if grad is not None else grad)

    def _project_in(self, x: Tensor) -> Tensor:
        if not self.has_projections:
            return x
        if self.use_conv_projections:
            if x.ndim == 3:
                x = x.transpose(1, 2).contiguous()
                x = self.project_in_module(x)
                return x.transpose(1, 2).contiguous()
            orig_shape = x.shape
            last_dim = orig_shape[-1]
            x = x.reshape(-1, last_dim).unsqueeze(-1)
            x = self.project_in_module(x)
            new_last_dim = x.shape[1]
            x = x.squeeze(-1).reshape(*orig_shape[:-1], new_last_dim)
            return x
        return self.project_in_module(x)

    def _project_out(self, x: Tensor) -> Tensor:
        if not self.has_projections:
            return x
        if self.use_conv_projections:
            if x.ndim == 3:
                x = x.transpose(1, 2).contiguous()
                x = self.project_out_module(x)
                return x.transpose(1, 2).contiguous()
            orig_shape = x.shape
            last_dim = orig_shape[-1]
            x = x.reshape(-1, last_dim).unsqueeze(-1)
            x = self.project_out_module(x)
            new_last_dim = x.shape[1]
            x = x.squeeze(-1).reshape(*orig_shape[:-1], new_last_dim)
            return x
        return self.project_out_module(x)
    
    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width
    
    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width
    
    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)
    
    def indices_to_codes(
        self,
        indices: Tensor,
        project_out = True
    ) -> Tensor:
        """Inverse of `codes_to_indices`."""

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        indices = rearrange(indices, '... -> ... 1')
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, '... c d -> ... (c d)')

        if project_out:
            codes = self._project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def forward(self, z: Tensor,mask=None) -> Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        mask - not used.
        """

        batch_size = z.shape[0]
        original_channels = z.shape[1] if z.ndim > 1 else self.dim
        is_img_or_video = z.ndim >= 4
        spatial_shape = ()
        channelwise = self.quantize_channels and is_img_or_video

        if is_img_or_video:
            channel_dim = z.shape[1]
            spatial_shape = z.shape[2:]
            spatial_prod = math.prod(spatial_shape)            

            if channelwise:
                # z = z.reshape(batch_size, channel_dim, spatial_prod)
                z = rearrange(z, 'b c ... -> b c (...)')
            else:
                #z = z.reshape(batch_size, channel_dim, spatial_prod).transpose(1, 2)
                z = rearrange(z, 'b c ... -> b (...) c')

        assert z.shape[-1] == self.dim, f'expected dimension of {self.dim} but found dimension of {z.shape[-1]}'
        

        z = self._project_in(z) #project spatial from 16x16 to 6
        z = rearrange(z, 'b n (c d) -> b n c d', c = self.num_codebooks) #8,512,1,6
        
        #bins the numbers into points on a line brt -1,1
        codes = self.quantize(z) 
        indices = self.codes_to_indices(codes)#8,512,1

        codes = rearrange(codes, 'b n c d -> b n (c d)') ##8,512,6
        # reconstitute image or video dimensions
        out = self._project_out(codes)

        if channelwise:
            #out = out.contiguous().view(batch_size, original_channels, *spatial_shape)
            out = rearrange(out, 'b c (h w) -> b c h w', h=spatial_shape[0], w=spatial_shape[1])
        elif is_img_or_video:
            out = rearrange(out, 'b (h w) d -> b d h w', h=spatial_shape[0], w=spatial_shape[1])
            indices = rearrange(indices, 'b (h w) d -> b d h w', h=spatial_shape[0], w=spatial_shape[1])            
            # out = out.transpose(1, 2).contiguous()
            # out = out.view(batch_size, self.dim, *spatial_shape)
            # indices = indices.reshape(batch_size, *spatial_shape, indices.shape[-1])
        return out, indices, 0

if __name__ == '__main__':
    levels = [8,5,5,5] # see 4.1 and A.4.1 in the paper


    print("Testing non-channelwise mode:")
    quantizer = FSQQuantizer(levels=levels, dim=512, num_codebooks=1, quantize_channels=False)
    x = torch.randn(1, 512, 4, 4) # 4 since there are 4 levels
    xhat, indices,_ = quantizer(x)
    print(xhat.shape,indices.shape)    # channelwise returns (B, C, H, W) and (B, C)
    print('*'*100)

    print("Testing channelwise mode:")
    quantizer = FSQQuantizer(levels=levels, dim=16, num_codebooks=1, quantize_channels=True,use_conv_projections=True)
    x = torch.randn(1, 512, 4, 4) # 4 since there are 4 levels
    xhat, indices,_ = quantizer(x)
    print(xhat.shape,indices.shape)    # channelwise returns (B, C, H, W) and (B, C)
    # print(indices.shape) # (1, 1024)    - (batch, seq)
