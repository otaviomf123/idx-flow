"""
idx-flow: Index-based Spherical Convolutions for HEALPix Grids in PyTorch.

This library provides efficient neural network layers for processing data on
spherical HEALPix grids using index-based convolutions. The architecture
achieves O(N) computational complexity while preserving the equal-area
properties essential for atmospheric and spherical data analysis.

Documentation: https://idx-flow.readthedocs.io/en/latest/

Structure Compilation Philosophy:
    This library decouples topology from computation. Connection indices
    (topology) are precomputed once and stored as buffers, while learnable
    weights (computation) are applied at runtime. This enables:
    - O(N) complexity instead of O(N^2) for neighbor lookups
    - Flexible architecture design with reusable index structures
    - Efficient memory usage through shared topology buffers

Architecture based on the paper:
    Atmospheric Data Compression and Reconstruction Using Spherical GANS.
    DOI: 10.1109/IJCNN64981.2025.11227156

Main Components:
    Core Spatial Layers:
        - SpatialConv: Spatial convolution for downsampling
        - SpatialTransposeConv: Transpose convolution for upsampling
        - SpatialUpsampling: Distance-based interpolation upsampling
        - SpatialPooling: Pooling operations (mean, max, sum)

    MLP Layers:
        - SpatialMLP: Multi-layer perceptron with spatial gathering
        - GlobalMLP: Channel-wise MLP without spatial mixing

    Normalization Layers:
        - SpatialBatchNorm: Batch normalization for spatial data
        - SpatialLayerNorm: Layer normalization for spatial data
        - SpatialInstanceNorm: Instance normalization for spatial data
        - SpatialGroupNorm: Group normalization for spatial data

    Regularization Layers:
        - SpatialDropout: Drops entire spatial locations
        - ChannelDropout: Drops entire channels

    Attention Layers:
        - SpatialSelfAttention: Multi-head self-attention

    Utility Layers:
        - Squeeze: Global spatial aggregation to vector
        - Unsqueeze: Broadcast vector to spatial dimension

    Utilities:
        - hp_distance: Compute neighbor indices and distances between grids
        - get_weights: Calculate interpolation weights from distances
        - compute_connection_indices: Convenience function for layer setup
        - get_healpix_resolution_info: Get resolution information
        - get_initializer: Get weight initialization function
        - get_activation: Get activation module by name

Example:
    >>> import torch
    >>> from idx_flow import SpatialConv, compute_connection_indices
    >>>
    >>> # Compute connection indices for downsampling
    >>> indices, distances = compute_connection_indices(
    ...     nside_in=64, nside_out=32, k=4
    ... )
    >>>
    >>> # Create spatial convolution layer with kaiming initialization
    >>> conv = SpatialConv(
    ...     output_points=12 * 32**2,
    ...     connection_indices=indices,
    ...     filters=64,
    ...     weight_init="kaiming_normal"
    ... )
    >>>
    >>> # Forward pass
    >>> x = torch.randn(8, 12 * 64**2, 32)  # [batch, points, channels]
    >>> y = conv(x)
    >>> print(y.shape)  # torch.Size([8, 12288, 64])

Author: Otavio Medeiros Feitosa
Institution: National Institute for Space Research (INPE)
"""

__version__ = "0.1.0"
__author__ = "Otavio Medeiros Feitosa"

# Core spatial layers
from idx_flow.layers import (
    SpatialConv,
    SpatialTransposeConv,
    SpatialUpsampling,
    SpatialPooling,
)

# MLP layers
from idx_flow.layers import (
    SpatialMLP,
    GlobalMLP,
)

# Normalization layers
from idx_flow.layers import (
    SpatialBatchNorm,
    SpatialLayerNorm,
    SpatialInstanceNorm,
    SpatialGroupNorm,
)

# Regularization layers
from idx_flow.layers import (
    SpatialDropout,
    ChannelDropout,
)

# Attention layers
from idx_flow.layers import (
    SpatialSelfAttention,
)

# Utility layers
from idx_flow.layers import (
    Squeeze,
    Unsqueeze,
)

# Initialization and activation utilities
from idx_flow.layers import (
    get_initializer,
    get_activation,
)

# Type aliases
from idx_flow.layers import (
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
