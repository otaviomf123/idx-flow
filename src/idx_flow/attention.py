"""
Attention layers for HEALPix Grid Processing.

This module implements self-attention layers for processing spatial data
on spherical HEALPix grids.
"""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SpatialSelfAttention(nn.Module):
    """
    Self-Attention layer for spatial data on HEALPix grids.

    Applies multi-head self-attention across the spatial dimension,
    allowing each spatial point to attend to all other points.

    Note: This has O(N^2) complexity in the number of spatial points.
    For large grids, consider using local attention variants.

    Args:
        embed_dim: Total dimension of the model (must be divisible by num_heads).
        num_heads: Number of attention heads.
        dropout: Dropout probability on attention weights. Default: 0.0.
        bias: Whether to include bias in projections. Default: True.

    Shape:
        - Input: [B, N, embed_dim]
        - Output: [B, N, embed_dim]

    Example:
        >>> attn = SpatialSelfAttention(embed_dim=64, num_heads=8)
        >>> x = torch.randn(4, 768, 64)  # Small grid for attention
        >>> y = attn(x)
        >>> print(y.shape)  # torch.Size([4, 768, 64])
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of spatial self-attention.

        Args:
            x: Input tensor of shape [B, N, embed_dim].

        Returns:
            Output tensor of shape [B, N, embed_dim].
        """
        batch_size, num_points, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [B, N, 3 * embed_dim]
        qkv = qkv.reshape(batch_size, num_points, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).reshape(batch_size, num_points, self.embed_dim)

        # Output projection
        out = self.out_proj(out)

        return out

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}"
