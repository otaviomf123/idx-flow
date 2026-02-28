"""
idx-flow: Index-based Spherical Convolutions for HEALPix Grids in PyTorch.

Docs: https://idx-flow.readthedocs.io/en/latest/

Based on: Atmospheric Data Compression and Reconstruction Using Spherical GANs
(DOI: 10.1109/IJCNN64981.2025.11227156)

Modules:
    conv           -- SpatialConv, SpatialTransposeConv, SpatialUpsampling
    mlp            -- SpatialMLP, GlobalMLP
    norm           -- SpatialBatchNorm, SpatialLayerNorm, SpatialInstanceNorm, SpatialGroupNorm
    regularization -- SpatialDropout, ChannelDropout
    attention      -- SpatialSelfAttention
    vit            -- SpatialPatchEmbedding, SpatialTransformerBlock, SpatialViT
    pooling        -- SpatialPooling, Squeeze, Unsqueeze
    functional     -- get_initializer, get_activation, type aliases
    utils          -- hp_distance, get_weights, compute_connection_indices

All public names are re-exported here for convenience::

    from idx_flow import SpatialConv, compute_connection_indices
"""

__version__ = "0.1.0"
__author__ = "Otavio Medeiros Feitosa"

# Core spatial layers
from idx_flow.conv import (
    SpatialConv,
    SpatialTransposeConv,
    SpatialUpsampling,
)

# Pooling layers
from idx_flow.pooling import (
    SpatialPooling,
    Squeeze,
    Unsqueeze,
)

# MLP layers
from idx_flow.mlp import (
    SpatialMLP,
    GlobalMLP,
)

# Normalization layers
from idx_flow.norm import (
    SpatialBatchNorm,
    SpatialLayerNorm,
    SpatialInstanceNorm,
    SpatialGroupNorm,
)

# Regularization layers
from idx_flow.regularization import (
    SpatialDropout,
    ChannelDropout,
)

# Attention layers
from idx_flow.attention import (
    SpatialSelfAttention,
)

# Vision Transformer layers
from idx_flow.vit import (
    SpatialPatchEmbedding,
    SpatialTransformerBlock,
    SpatialViT,
)

# Initialization and activation utilities
from idx_flow.functional import (
    get_initializer,
    get_activation,
)

# Type aliases
from idx_flow.functional import (
    InterpolationMethod,
    PoolingMethod,
    InitMethod,
    ActivationType,
)

# Utility functions
from idx_flow.utils import (
    hp_distance,
    get_weights,
    compute_connection_indices,
    get_healpix_resolution_info,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core spatial layers
    "SpatialConv",
    "SpatialTransposeConv",
    "SpatialUpsampling",
    "SpatialPooling",
    # MLP layers
    "SpatialMLP",
    "GlobalMLP",
    # Normalization layers
    "SpatialBatchNorm",
    "SpatialLayerNorm",
    "SpatialInstanceNorm",
    "SpatialGroupNorm",
    # Regularization layers
    "SpatialDropout",
    "ChannelDropout",
    # Attention layers
    "SpatialSelfAttention",
    # Vision Transformer layers
    "SpatialPatchEmbedding",
    "SpatialTransformerBlock",
    "SpatialViT",
    # Utility layers
    "Squeeze",
    "Unsqueeze",
    # Initialization and activation utilities
    "get_initializer",
    "get_activation",
    # Type aliases
    "InterpolationMethod",
    "PoolingMethod",
    "InitMethod",
    "ActivationType",
    # Utility functions
    "hp_distance",
    "get_weights",
    "compute_connection_indices",
    "get_healpix_resolution_info",
]
