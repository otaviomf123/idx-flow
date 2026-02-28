"""
Attention layers for HEALPix Grid Processing.

This module implements self-attention layers for processing spatial data
on spherical HEALPix grids. Supports PyTorch's scaled_dot_product_attention
(SDPA) for automatic FlashAttention 2 / memory-efficient dispatch on
compatible hardware.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# SDPA is available since PyTorch 2.0
_SDPA_AVAILABLE = hasattr(F, "scaled_dot_product_attention")

AttnBackend = Literal["auto", "sdpa", "manual"]


class SpatialSelfAttention(nn.Module):
    """
    Self-Attention layer for spatial data on HEALPix grids.

    Applies multi-head self-attention across the spatial dimension,
    allowing each spatial point to attend to all other points.

    When ``attn_backend="auto"`` (default) and PyTorch >= 2.0,
    attention is computed via ``F.scaled_dot_product_attention``, which
    automatically selects the fastest available kernel:

    - **FlashAttention 2** on Ampere+ GPUs with float16/bfloat16
    - **Memory-efficient attention** via xFormers-style backend
    - **Math fallback** on CPU or unsupported dtypes

    Set ``attn_backend="manual"`` to force the explicit
    matmul-softmax-matmul path (always available).

    Note: Complexity is O(N^2) in the number of spatial points
    regardless of backend. FlashAttention 2 reduces the constant
    factor and memory usage but does not change asymptotic complexity.

    Args:
        embed_dim: Total dimension of the model (must be divisible by num_heads).
        num_heads: Number of attention heads.
        dropout: Dropout probability on attention weights. Default: 0.0.
        bias: Whether to include bias in projections. Default: True.
        attn_backend: Attention computation backend. Default: "auto".
            - ``"auto"``: Use SDPA when available (PyTorch >= 2.0), else manual.
            - ``"sdpa"``: Force SDPA (raises if unavailable).
            - ``"manual"``: Force explicit matmul-softmax-matmul.

    Shape:
        - Input: [B, N, embed_dim]
        - Output: [B, N, embed_dim]

    Example:
        >>> attn = SpatialSelfAttention(embed_dim=64, num_heads=8)
        >>> x = torch.randn(4, 768, 64)
        >>> y = attn(x)
        >>> print(y.shape)  # torch.Size([4, 768, 64])
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        attn_backend: AttnBackend = "auto",
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        if attn_backend == "sdpa" and not _SDPA_AVAILABLE:
            raise RuntimeError(
                "attn_backend='sdpa' requires PyTorch >= 2.0. "
                f"Current version: {torch.__version__}"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_backend = attn_backend

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self._dropout_p = dropout

    def _use_sdpa(self) -> bool:
        """Determine whether to use SDPA for the forward pass."""
        if self.attn_backend == "manual":
            return False
        if self.attn_backend == "sdpa":
            return True
        # "auto": use SDPA when available
        return _SDPA_AVAILABLE

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
        q, k, v = qkv.unbind(0)

        if self._use_sdpa():
            dropout_p = self._dropout_p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p
            )  # [B, num_heads, N, head_dim]
        else:
            # Manual attention: matmul -> scale -> softmax -> dropout -> matmul
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = attn @ v  # [B, num_heads, N, head_dim]

        out = out.transpose(1, 2).reshape(batch_size, num_points, self.embed_dim)

        # Output projection
        out = self.out_proj(out)

        return out

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"attn_backend='{self.attn_backend}'"
        )
