"""
Vision Transformer layers for HEALPix Grid Processing.

This module implements Vision Transformer (ViT) components adapted for
spatial data on spherical HEALPix grids: patch embedding, transformer
encoder blocks, and the full ViT architecture.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor

from idx_flow.attention import AttnBackend, SpatialSelfAttention
from idx_flow.functional import ActivationType, InitMethod, get_activation, get_initializer


class SpatialPatchEmbedding(nn.Module):
    """
    Spatial Patch Embedding layer for HEALPix grids.

    This layer creates patch embeddings from local neighborhoods on the
    spherical grid using precomputed connection indices. It gathers k neighbor
    features for each output point, flattens the neighborhood into a single
    vector, and projects it to the embedding dimension via a linear layer.

    This is the spatial analog of the patch embedding in Vision Transformers
    (ViT), adapted for non-Euclidean domains. Instead of extracting fixed-size
    2D image patches, it uses precomputed topology (connection indices) to
    define local patches on the sphere.

    The operation:
        1. Gather: Collect k neighbor features for each output point: [B, N_out, k, C_in]
        2. Flatten: Reshape neighborhood to vector: [B, N_out, k * C_in]
        3. Project: Linear projection to embedding dimension: [B, N_out, embed_dim]

    Mathematically:
        E[b,p,:] = W_proj * [X[b, idx[p,0], :] || ... || X[b, idx[p,k-1], :]] + b_proj

    Literature Context:
        Adapted from the Vision Transformer (ViT) architecture:

        - Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021).
          "An Image is Worth 16x16 Words: Transformers for Image Recognition
          at Scale." In ICLR 2021.
          arXiv: 2010.11929

    Args:
        output_points: Number of spatial points in the output (number of patches).
        connection_indices: Integer array of shape [output_points, kernel_size]
            containing indices of input pixels for each output patch.
        embed_dim: Dimension of the patch embedding vectors.
        bias: Whether to include a bias term in the projection. Default is True.
        weight_init: Weight initialization method. Default is "xavier_uniform".
        weight_init_gain: Gain for xavier/orthogonal initialization.

    Attributes:
        projection: Learnable linear projection from flattened patch to embed_dim.

    Shape:
        - Input: [B, N_in, C_in] where B is batch size, N_in is input points,
          C_in is input channels.
        - Output: [B, N_out, embed_dim] where N_out is output_points.

    Example:
        >>> from idx_flow.utils import compute_connection_indices
        >>> indices, _ = compute_connection_indices(
        ...     nside_in=64, nside_out=32, k=9
        ... )
        >>> patch_embed = SpatialPatchEmbedding(
        ...     output_points=12 * 32**2,
        ...     connection_indices=indices,
        ...     embed_dim=128,
        ... )
        >>> x = torch.randn(8, 12 * 64**2, 16)
        >>> embeddings = patch_embed(x)
        >>> print(embeddings.shape)  # torch.Size([8, 12288, 128])

    See Also:
        - :class:`SpatialConv`: Linear convolution (einsum-based kernel)
        - :class:`SpatialMLP`: MLP kernel for non-linear local processing
        - :class:`SpatialViT`: Full Vision Transformer using this embedding
    """

    def __init__(
        self,
        output_points: int,
        connection_indices: NDArray[np.int64],
        embed_dim: int = 128,
        bias: bool = True,
        weight_init: InitMethod = "xavier_uniform",
        weight_init_gain: float = 1.0,
    ) -> None:
        super().__init__()

        self.output_points = output_points
        self.kernel_size = connection_indices.shape[1]
        self.embed_dim = embed_dim
        self.use_bias = bias
        self.weight_init = weight_init
        self.weight_init_gain = weight_init_gain

        # Register connection indices as buffer (non-trainable, saved with model)
        self.register_buffer(
            "connection_indices",
            torch.from_numpy(connection_indices.astype(np.int64)),
        )

        # Projection layer will be lazily initialized
        self.projection: Optional[nn.Linear] = None
        self._initialized = False

    def _initialize_parameters(self, in_channels: int) -> None:
        """Initialize projection layer based on input channels."""
        input_dim = self.kernel_size * in_channels
        self.projection = nn.Linear(input_dim, self.embed_dim, bias=self.use_bias)

        init_fn = get_initializer(self.weight_init, gain=self.weight_init_gain)
        init_fn(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

        self._initialized = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of spatial patch embedding.

        Args:
            x: Input tensor of shape [B, N_in, C_in].

        Returns:
            Output tensor of shape [B, N_out, embed_dim].
        """
        batch_size, input_points, in_channels = x.shape

        # Lazy initialization of parameters
        if not self._initialized:
            self._initialize_parameters(in_channels)
            if self.projection is not None:
                self.projection = self.projection.to(x.device)

        # Gather neighbor features: [B, N_out, kernel_size, C_in]
        neighbors = x[:, self.connection_indices, :]

        # Flatten patches: [B, N_out, kernel_size * C_in]
        patches = neighbors.reshape(
            batch_size, self.output_points, self.kernel_size * in_channels
        )

        # Project to embedding dimension: [B, N_out, embed_dim]
        embeddings = self.projection(patches)

        return embeddings

    def extra_repr(self) -> str:
        """Return a string representation of layer parameters."""
        return (
            f"output_points={self.output_points}, "
            f"kernel_size={self.kernel_size}, "
            f"embed_dim={self.embed_dim}, "
            f"bias={self.use_bias}, "
            f"weight_init='{self.weight_init}'"
        )


class SpatialTransformerBlock(nn.Module):
    """
    Transformer Encoder Block for spatial data on HEALPix grids.

    Implements a standard pre-norm Transformer encoder block with multi-head
    self-attention and a position-wise feed-forward network (FFN). Uses
    residual connections and layer normalization following the pre-norm
    convention (LN before attention/FFN rather than after).

    The operation:
        1. Pre-norm: LayerNorm -> Multi-Head Self-Attention -> Residual
        2. Pre-norm: LayerNorm -> Feed-Forward Network -> Residual

    Mathematically:
        h = x + MHSA(LN(x))
        y = h + FFN(LN(h))

    where:
        - MHSA: Multi-Head Self-Attention with scaled dot-product
        - FFN: Two-layer MLP with activation (Linear -> Act -> Dropout -> Linear -> Dropout)
        - LN: Layer Normalization per spatial point

    Note: The self-attention has O(N^2) complexity in the number of spatial
    points. For large grids, consider using this after spatial downsampling.

    Args:
        embed_dim: Embedding dimension (must be divisible by num_heads).
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of FFN hidden dimension to embed_dim. Default is 4.0.
        dropout: Dropout probability in attention and FFN. Default is 0.0.
        activation: Activation function for the FFN. Default is "gelu".
        bias: Whether to include bias in linear projections. Default is True.
        norm_eps: Epsilon for layer normalization. Default is 1e-6.
        attn_backend: Attention backend. ``"auto"`` uses SDPA/FlashAttention 2
            when available, ``"manual"`` forces explicit matmul path.
            Default is ``"auto"``.

    Shape:
        - Input: [B, N, embed_dim]
        - Output: [B, N, embed_dim]

    Example:
        >>> block = SpatialTransformerBlock(
        ...     embed_dim=128,
        ...     num_heads=8,
        ...     mlp_ratio=4.0,
        ...     dropout=0.1,
        ... )
        >>> x = torch.randn(8, 768, 128)
        >>> y = block(x)
        >>> print(y.shape)  # torch.Size([8, 768, 128])

    See Also:
        - :class:`SpatialSelfAttention`: Standalone multi-head self-attention
        - :class:`SpatialViT`: Full Vision Transformer using this block
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        activation: ActivationType = "gelu",
        bias: bool = True,
        norm_eps: float = 1e-6,
        attn_backend: AttnBackend = "auto",
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # Pre-norm layer normalization
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)

        # Multi-head self-attention
        self.attn = SpatialSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            attn_backend=attn_backend,
        )

        # Feed-forward network
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim, bias=bias),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the transformer block.

        Args:
            x: Input tensor of shape [B, N, embed_dim].

        Returns:
            Output tensor of shape [B, N, embed_dim].
        """
        # Self-attention with pre-norm and residual
        x = x + self.attn(self.norm1(x))

        # FFN with pre-norm and residual
        x = x + self.ffn(self.norm2(x))

        return x

    def extra_repr(self) -> str:
        """Return a string representation of layer parameters."""
        return (
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"mlp_ratio={self.mlp_ratio}"
        )


class SpatialViT(nn.Module):
    """
    Vision Transformer (ViT) for spatial data on HEALPix grids.

    This layer implements a complete Vision Transformer adapted for spherical
    data discretized on HEALPix grids. It combines index-based patch embedding
    with Transformer encoder blocks, bridging the structure-compilation
    philosophy of this library with the global attention mechanism of ViT.

    The architecture:
        1. **Patch Embedding**: Uses precomputed connection indices to gather
           local neighborhoods and project them to embedding vectors.
        2. **Positional Encoding**: Learnable positional embeddings added to
           each spatial point to encode location on the sphere.
        3. **Transformer Encoder**: Stack of N Transformer blocks, each with
           multi-head self-attention and feed-forward network.
        4. **Output Projection**: Optional linear projection to desired output
           dimension.

    The operation:
        1. Embed: E = PatchEmbed(X) + PosEmbed     [B, N_out, embed_dim]
        2. Encode: Z = TransformerBlock_N(...TransformerBlock_1(E))
        3. Project: Y = Linear(LN(Z))              [B, N_out, output_dim]

    Complexity:
        - Patch embedding: O(N_out * k * C_in) -- linear in output points
        - Self-attention per block: O(N_out^2 * embed_dim) -- quadratic in points
        - Total: O(depth * N_out^2 * embed_dim)
        For large grids, use spatial downsampling (via connection indices from
        higher to lower resolution) to reduce N_out before the transformer.

    Literature Context:
        Adapted from:

        - Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021).
          "An Image is Worth 16x16 Words: Transformers for Image Recognition
          at Scale." In ICLR 2021.
          arXiv: 2010.11929

    Args:
        output_points: Number of spatial points (patches) in the output.
        connection_indices: Integer array of shape [output_points, kernel_size]
            containing indices of input pixels for each output patch.
        embed_dim: Dimension of the patch embeddings. Default is 128.
        depth: Number of Transformer encoder blocks. Default is 4.
        num_heads: Number of attention heads per block. Default is 8.
        mlp_ratio: Ratio of FFN hidden dim to embed_dim. Default is 4.0.
        output_dim: Output feature dimension. If None, equals embed_dim.
        dropout: Dropout probability for attention and FFN. Default is 0.0.
        activation: Activation function for FFN layers. Default is "gelu".
        bias: Whether to include bias in linear projections. Default is True.
        weight_init: Weight initialization method for patch embedding.
            Default is "xavier_uniform".
        norm_eps: Epsilon for layer normalization. Default is 1e-6.
        attn_backend: Attention backend for all transformer blocks.
            ``"auto"`` uses SDPA/FlashAttention 2 when available,
            ``"manual"`` forces explicit matmul path.
            Default is ``"auto"``.

    Shape:
        - Input: [B, N_in, C_in] where B is batch size, N_in is input points,
          C_in is input channels.
        - Output: [B, N_out, output_dim] where N_out is output_points.

    Example:
        >>> from idx_flow.utils import compute_connection_indices
        >>> indices, _ = compute_connection_indices(
        ...     nside_in=16, nside_out=8, k=9
        ... )
        >>> vit = SpatialViT(
        ...     output_points=12 * 8**2,
        ...     connection_indices=indices,
        ...     embed_dim=64,
        ...     depth=4,
        ...     num_heads=4,
        ...     output_dim=32,
        ...     dropout=0.1,
        ... )
        >>> x = torch.randn(4, 12 * 16**2, 8)
        >>> y = vit(x)
        >>> print(y.shape)  # torch.Size([4, 768, 32])

    See Also:
        - :class:`SpatialPatchEmbedding`: Patch embedding using connection indices
        - :class:`SpatialTransformerBlock`: Single transformer encoder block
        - :class:`SpatialSelfAttention`: Standalone multi-head self-attention
        - :class:`SpatialConv`: Linear convolution with O(N) complexity
        - :class:`SpatialMLP`: MLP kernel for local non-linear processing
    """

    def __init__(
        self,
        output_points: int,
        connection_indices: NDArray[np.int64],
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        output_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: ActivationType = "gelu",
        bias: bool = True,
        weight_init: InitMethod = "xavier_uniform",
        norm_eps: float = 1e-6,
        attn_backend: AttnBackend = "auto",
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.output_points = output_points
        self.kernel_size = connection_indices.shape[1]
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.output_dim = output_dim if output_dim is not None else embed_dim

        # Patch embedding (uses connection indices, lazily initialized)
        self.patch_embed = SpatialPatchEmbedding(
            output_points=output_points,
            connection_indices=connection_indices,
            embed_dim=embed_dim,
            bias=bias,
            weight_init=weight_init,
        )

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, output_points, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Embedding dropout
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            SpatialTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                activation=activation,
                bias=bias,
                norm_eps=norm_eps,
                attn_backend=attn_backend,
            )
            for _ in range(depth)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)

        # Output projection (if output_dim differs from embed_dim)
        if self.output_dim != embed_dim:
            self.output_proj = nn.Linear(embed_dim, self.output_dim, bias=bias)
            init_fn = get_initializer(weight_init)
            init_fn(self.output_proj.weight)
            if self.output_proj.bias is not None:
                nn.init.zeros_(self.output_proj.bias)
        else:
            self.output_proj = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Spatial Vision Transformer.

        Args:
            x: Input tensor of shape [B, N_in, C_in].

        Returns:
            Output tensor of shape [B, N_out, output_dim].
        """
        # Patch embedding: [B, N_in, C_in] -> [B, N_out, embed_dim]
        x = self.patch_embed(x)

        # Add positional encoding
        x = x + self.pos_embed
        x = self.embed_dropout(x)

        # Transformer encoder blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        # Output projection
        if self.output_proj is not None:
            x = self.output_proj(x)

        return x

    def extra_repr(self) -> str:
        """Return a string representation of layer parameters."""
        return (
            f"output_points={self.output_points}, "
            f"kernel_size={self.kernel_size}, "
            f"embed_dim={self.embed_dim}, "
            f"depth={self.depth}, "
            f"num_heads={self.num_heads}, "
            f"output_dim={self.output_dim}"
        )
