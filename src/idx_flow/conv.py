"""
Spatial Convolution layers for HEALPix Grid Processing.

This module implements convolution and upsampling layers for processing data
on spherical HEALPix grids using index-based convolutions.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor

from idx_flow.functional import (
    InitMethod,
    InterpolationMethod,
    get_initializer,
)


class SpatialConv(nn.Module):
    """
    Spatial Convolution layer for downsampling on HEALPix grids.

    This layer performs convolution on spherical data discretized using the
    HEALPix tessellation scheme. It uses precomputed connection indices to
    gather features from neighboring pixels and applies learnable kernels
    for spatial feature transformation.

    The operation follows:
        1. Gather: Collect features from k neighbors for each output point
        2. Transform: Apply learnable kernel weights
        3. Aggregate: Sum weighted contributions with bias

    Mathematically:
        Y[b,p,f] = sum_k sum_c X[b, idx[p,k], c] * W[k,c,f] + bias[f]

    Args:
        output_points: Number of spatial points in the output tensor.
        connection_indices: Integer array of shape [output_points, kernel_size]
            containing indices of input pixels that connect to each output pixel.
        kernel_weights: Optional float array of shape [output_points, kernel_size]
            containing distance-based weights for each connection. If provided,
            neighbor features are scaled by these weights before convolution.
        filters: Number of output feature channels (filters).
        bias: Whether to include a bias term. Default is True.
        weight_init: Weight initialization method. Default is "xavier_uniform".
        weight_init_gain: Gain for xavier/orthogonal initialization.
        bias_init: Bias initialization value. Default is 0.0.

    Attributes:
        kernel: Learnable weight tensor of shape [kernel_size, in_channels, filters].
        bias_param: Learnable bias tensor of shape [filters] if bias=True.

    Shape:
        - Input: [B, N_in, C_in] where B is batch size, N_in is input points,
          C_in is input channels.
        - Output: [B, N_out, filters] where N_out is output_points.

    Example:
        >>> from idx_flow.utils import compute_connection_indices
        >>> indices, distances = compute_connection_indices(
        ...     nside_in=64, nside_out=32, k=4
        ... )
        >>> conv = SpatialConv(
        ...     output_points=12 * 32**2,
        ...     connection_indices=indices,
        ...     filters=64,
        ...     weight_init="kaiming_normal"
        ... )
        >>> x = torch.randn(8, 12 * 64**2, 32)
        >>> y = conv(x)
        >>> print(y.shape)  # torch.Size([8, 12288, 64])

    Notes:
        - Connection indices must be precomputed using hp_distance or similar.
        - The layer maintains O(N) complexity per forward pass.
        - Input channels are inferred from the first forward pass.
    """

    def __init__(
        self,
        output_points: int,
        connection_indices: NDArray[np.int64],
        kernel_weights: Optional[NDArray[np.float64]] = None,
        filters: int = 32,
        bias: bool = True,
        weight_init: InitMethod = "xavier_uniform",
        weight_init_gain: float = 1.0,
        bias_init: float = 0.0,
    ) -> None:
        super().__init__()

        self.output_points = output_points
        self.filters = filters
        self.kernel_size = connection_indices.shape[1]
        self.use_bias = bias
        self.weight_init = weight_init
        self.weight_init_gain = weight_init_gain
        self.bias_init = bias_init

        # Register connection indices as buffer (non-trainable, saved with model)
        self.register_buffer(
            "connection_indices",
            torch.from_numpy(connection_indices.astype(np.int64)),
        )

        # Register optional kernel weights
        if kernel_weights is not None:
            weights_tensor = torch.from_numpy(kernel_weights.astype(np.float32))
            self.register_buffer("kernel_weights", weights_tensor.unsqueeze(-1))
        else:
            self.register_buffer("kernel_weights", None)

        # Learnable parameters will be lazily initialized
        self.kernel: Optional[nn.Parameter] = None
        self.bias_param: Optional[nn.Parameter] = None
        self._initialized = False

    def _initialize_parameters(self, in_channels: int) -> None:
        """Initialize learnable parameters based on input channels."""
        self.kernel = nn.Parameter(
            torch.empty(self.kernel_size, in_channels, self.filters)
        )
        init_fn = get_initializer(self.weight_init, gain=self.weight_init_gain)
        init_fn(self.kernel)

        if self.use_bias:
            self.bias_param = nn.Parameter(
                torch.full((self.filters,), self.bias_init)
            )

        self._initialized = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the spatial convolution.

        Args:
            x: Input tensor of shape [B, N_in, C_in].

        Returns:
            Output tensor of shape [B, N_out, filters].
        """
        batch_size, input_points, in_channels = x.shape

        # Lazy initialization of parameters
        if not self._initialized:
            self._initialize_parameters(in_channels)
            if self.kernel is not None:
                self.kernel = nn.Parameter(self.kernel.to(x.device))
            if self.bias_param is not None:
                self.bias_param = nn.Parameter(self.bias_param.to(x.device))

        # Gather neighbor features: [B, N_out, kernel_size, C_in]
        neighbors = x[:, self.connection_indices, :]

        # Apply optional distance-based weights
        if self.kernel_weights is not None:
            neighbors = neighbors * self.kernel_weights

        # Spatial convolution using einsum
        output = torch.einsum("bpkc,kcf->bpf", neighbors, self.kernel)

        # Add bias
        if self.bias_param is not None:
            output = output + self.bias_param

        return output

    def extra_repr(self) -> str:
        """Return a string representation of layer parameters."""
        return (
            f"output_points={self.output_points}, "
            f"kernel_size={self.kernel_size}, "
            f"filters={self.filters}, "
            f"bias={self.use_bias}, "
            f"weight_init='{self.weight_init}'"
        )


class SpatialTransposeConv(nn.Module):
    """
    Spatial Transpose Convolution layer for upsampling on HEALPix grids.

    This layer performs transposed (deconvolution) operations for upsampling
    spatial resolution on HEALPix grids. It maps features from a lower
    resolution grid to a higher resolution grid using precomputed connection
    indices.

    Args:
        output_points: Number of spatial points in the output (higher resolution).
        connection_indices: Integer array of shape [output_points, kernel_size]
            containing indices of input pixels for each output pixel.
        kernel_weights: Optional float array of shape [output_points, kernel_size]
            containing distance-based weights for each connection.
        filters: Number of output feature channels.
        bias: Whether to include a bias term. Default is True.
        weight_init: Weight initialization method. Default is "xavier_uniform".
        weight_init_gain: Gain for xavier/orthogonal initialization.
        bias_init: Bias initialization value. Default is 0.0.

    Shape:
        - Input: [B, N_in, C_in] where N_in is the lower resolution.
        - Output: [B, N_out, filters] where N_out is output_points (higher res).

    Example:
        >>> from idx_flow.utils import compute_connection_indices
        >>> indices, distances, weights = compute_connection_indices(
        ...     nside_in=32, nside_out=64, k=4, return_weights=True
        ... )
        >>> transpose_conv = SpatialTransposeConv(
        ...     output_points=12 * 64**2,
        ...     connection_indices=indices,
        ...     kernel_weights=weights,
        ...     filters=32,
        ...     weight_init="orthogonal"
        ... )
        >>> x = torch.randn(8, 12 * 32**2, 64)
        >>> y = transpose_conv(x)
        >>> print(y.shape)  # torch.Size([8, 49152, 32])
    """

    def __init__(
        self,
        output_points: int,
        connection_indices: NDArray[np.int64],
        kernel_weights: Optional[NDArray[np.float64]] = None,
        filters: int = 32,
        bias: bool = True,
        weight_init: InitMethod = "xavier_uniform",
        weight_init_gain: float = 1.0,
        bias_init: float = 0.0,
    ) -> None:
        super().__init__()

        self.output_points = output_points
        self.filters = filters
        self.kernel_size = connection_indices.shape[1]
        self.use_bias = bias
        self.weight_init = weight_init
        self.weight_init_gain = weight_init_gain
        self.bias_init = bias_init

        self.register_buffer(
            "connection_indices",
            torch.from_numpy(connection_indices.astype(np.int64)),
        )

        if kernel_weights is not None:
            weights_tensor = torch.from_numpy(kernel_weights.astype(np.float32))
            self.register_buffer("kernel_weights", weights_tensor.unsqueeze(-1))
        else:
            self.register_buffer("kernel_weights", None)

        self.kernel: Optional[nn.Parameter] = None
        self.bias_param: Optional[nn.Parameter] = None
        self._initialized = False

    def _initialize_parameters(self, in_channels: int) -> None:
        """Initialize learnable parameters based on input channels."""
        self.kernel = nn.Parameter(
            torch.empty(self.kernel_size, in_channels, self.filters)
        )
        init_fn = get_initializer(self.weight_init, gain=self.weight_init_gain)
        init_fn(self.kernel)

        if self.use_bias:
            self.bias_param = nn.Parameter(
                torch.full((self.filters,), self.bias_init)
            )

        self._initialized = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the spatial transpose convolution.

        Args:
            x: Input tensor of shape [B, N_in, C_in].

        Returns:
            Output tensor of shape [B, N_out, filters].
        """
        batch_size, input_points, in_channels = x.shape

        if not self._initialized:
            self._initialize_parameters(in_channels)
            if self.kernel is not None:
                self.kernel = nn.Parameter(self.kernel.to(x.device))
            if self.bias_param is not None:
                self.bias_param = nn.Parameter(self.bias_param.to(x.device))

        neighbors = x[:, self.connection_indices, :]

        if self.kernel_weights is not None:
            neighbors = neighbors * self.kernel_weights

        output = torch.einsum("bpkc,kcf->bpf", neighbors, self.kernel)

        if self.bias_param is not None:
            output = output + self.bias_param

        return output

    def extra_repr(self) -> str:
        """Return a string representation of layer parameters."""
        return (
            f"output_points={self.output_points}, "
            f"kernel_size={self.kernel_size}, "
            f"filters={self.filters}, "
            f"bias={self.use_bias}, "
            f"weight_init='{self.weight_init}'"
        )


class SpatialUpsampling(nn.Module):
    """
    Spatial Upsampling layer using distance-based interpolation.

    This layer performs upsampling on HEALPix grids using precomputed
    interpolation weights based on geodesic distances. Unlike SpatialTransposeConv,
    this layer does not have learnable parameters and performs pure
    distance-weighted interpolation.

    Interpolation methods:
        - "linear": Weight = max(0, 1 - distance / kernel_radius)
        - "idw": Inverse distance weighting, Weight = 1 / (distance^2 + eps)
        - "gaussian": Weight = exp(-0.5 * (distance / kernel_radius)^2)

    Args:
        output_points: Number of spatial points in the output (higher resolution).
        connection_indices: Integer array of shape [output_points, kernel_size]
            containing indices of input pixels for each output pixel.
        distances: Float array of shape [output_points, kernel_size] containing
            geodesic distances to each neighbor.
        interpolation: Interpolation method. One of "linear", "idw", "gaussian".
        kernel_radius: Radius for interpolation kernel. If None, uses the
            maximum distance. Only used for "linear" and "gaussian" methods.

    Shape:
        - Input: [B, N_in, C_in]
        - Output: [B, N_out, C_in] (channels preserved)

    Example:
        >>> from idx_flow.utils import compute_connection_indices
        >>> indices, distances = compute_connection_indices(
        ...     nside_in=32, nside_out=64, k=4
        ... )
        >>> upsample = SpatialUpsampling(
        ...     output_points=12 * 64**2,
        ...     connection_indices=indices,
        ...     distances=distances,
        ...     interpolation="idw"
        ... )
        >>> x = torch.randn(8, 12 * 32**2, 32)
        >>> y = upsample(x)
        >>> print(y.shape)  # torch.Size([8, 49152, 32])

    Notes:
        - This layer has no learnable parameters.
        - Output channels equal input channels (no feature transformation).
        - Weights are precomputed and stored as buffers for efficiency.
    """

    def __init__(
        self,
        output_points: int,
        connection_indices: NDArray[np.int64],
        distances: NDArray[np.float64],
        interpolation: InterpolationMethod = "linear",
        kernel_radius: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.output_points = output_points
        self.kernel_size = connection_indices.shape[1]
        self.interpolation = interpolation
        self.kernel_radius = kernel_radius if kernel_radius else float(np.max(distances))

        self.register_buffer(
            "connection_indices",
            torch.from_numpy(connection_indices.astype(np.int64)),
        )

        weights = self._compute_weights(distances)
        weights = weights / (np.sum(weights, axis=-1, keepdims=True) + 1e-10)
        self.register_buffer(
            "interpolation_weights",
            torch.from_numpy(weights.astype(np.float32)).unsqueeze(-1),
        )

    def _compute_weights(self, distances: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute interpolation weights based on the chosen method."""
        if self.interpolation == "linear":
            norm_distances = distances / self.kernel_radius
            weights = np.maximum(0.0, 1.0 - norm_distances)
        elif self.interpolation == "idw":
            epsilon = 1e-10
            weights = 1.0 / (np.power(distances + epsilon, 2))
        elif self.interpolation == "gaussian":
            weights = np.exp(-0.5 * np.square(distances / self.kernel_radius))
        else:
            raise ValueError(
                f"Unsupported interpolation method: '{self.interpolation}'. "
                f"Choose from: 'linear', 'idw', 'gaussian'"
            )
        return weights

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of spatial upsampling.

        Args:
            x: Input tensor of shape [B, N_in, C_in].

        Returns:
            Output tensor of shape [B, N_out, C_in].
        """
        neighbors = x[:, self.connection_indices, :]
        output = torch.sum(neighbors * self.interpolation_weights, dim=2)
        return output

    def extra_repr(self) -> str:
        """Return a string representation of layer parameters."""
        return (
            f"output_points={self.output_points}, "
            f"kernel_size={self.kernel_size}, "
            f"interpolation='{self.interpolation}', "
            f"kernel_radius={self.kernel_radius:.2f}"
        )
