import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.block(x) + self.channel_up(x)
        else:
            return x + self.block(x)


class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.)
        return self.conv(x)


class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, use_flash_attn: bool = False, num_heads: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.use_flash_attn = use_flash_attn
        self.num_heads = num_heads

        self.norm = GroupNorm(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.k = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.v = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

        if self.use_flash_attn:
            assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
            self.head_dim = in_channels // num_heads

    def _flash_attn(self, q, k, v):
        b, _, h, w = q.shape
        hw = h * w

        def reshape_heads(t):
            t = t.view(b, self.num_heads, self.head_dim, hw)
            return t.permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn_out = attn_out.permute(0, 1, 3, 2).contiguous()
        attn_out = attn_out.view(b, self.in_channels, h, w)
        return attn_out

    def _classic_attn(self, q, k, v):
        b, c, h, w = q.shape
        hw = h * w
        q_flat = q.reshape(b, c, hw)
        k_flat = k.reshape(b, c, hw)
        v_flat = v.reshape(b, c, hw)
        attn = torch.bmm(q_flat.transpose(1, 2), k_flat)
        attn = attn * (c ** (-0.5))
        attn = F.softmax(attn, dim=2)
        out = torch.bmm(v_flat, attn.transpose(1, 2))
        return out.reshape(b, c, h, w).contiguous()

    def forward(self, x):
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        if self.use_flash_attn:
            A = self._flash_attn(q, k, v)
        else:
            A = self._classic_attn(q, k, v)

        A = self.proj_out(A)
        return x + A


class GroupNorm(nn.Module):
    def __init__(self, in_channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
