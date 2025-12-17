"""
Spatial Neural Network Layers for HEALPix Grid Processing.

This module implements PyTorch layers for processing data on spherical HEALPix
grids using index-based convolutions. The layers enable efficient O(N) spatial
operations while preserving the equal-area properties essential for spherical
data analysis.

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

The spatial convolution operation is defined by:
    Y[b,i,f] = sum_k sum_c N[b,i,k,c] * W[k,c,f] + b[f]

where:
    - Y[b,i,f]: Output for batch b, spatial point i, feature channel f
    - N[b,i,k,c]: Neighbor features for batch b, point i, k-th neighbor, channel c
    - W[k,c,f]: Learnable weight for k-th neighbor, input channel c, output channel f
    - b[f]: Bias term for output channel f

Author: Otavio Medeiros Feitosa
Institution: National Institute for Space Research (INPE)
"""

from typing import Callable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor


# =============================================================================
# Type Aliases
# =============================================================================

InterpolationMethod = Literal["linear", "idw", "gaussian"]
PoolingMethod = Literal["mean", "max", "sum"]
InitMethod = Literal["xavier_uniform", "xavier_normal", "kaiming_uniform",
                     "kaiming_normal", "orthogonal", "normal", "uniform", "zeros"]
ActivationType = Literal["relu", "selu", "leaky_relu", "gelu", "elu", "tanh",
                         "sigmoid", "swish", "mish", "linear"]


# =============================================================================
# Initialization Utilities
# =============================================================================


def get_initializer(
    method: InitMethod,
    gain: float = 1.0,
    nonlinearity: str = "leaky_relu",
    mean: float = 0.0,
    std: float = 0.02,
    a: float = 0.0,
    b: float = 1.0,
) -> Callable[[Tensor], Tensor]:
    """
    Get weight initialization function.

    Args:
        method: Initialization method name.
        gain: Gain factor for xavier/orthogonal initialization.
        nonlinearity: Nonlinearity for kaiming initialization.
        mean: Mean for normal initialization.
        std: Standard deviation for normal initialization.
        a: Lower bound for uniform initialization.
        b: Upper bound for uniform initialization.

    Returns:
        Initialization function that takes a tensor and initializes it in-place.

    Raises:
        ValueError: If method is not recognized.
    """
    def init_fn(tensor: Tensor) -> Tensor:
        if method == "xavier_uniform":
            return nn.init.xavier_uniform_(tensor, gain=gain)
        elif method == "xavier_normal":
            return nn.init.xavier_normal_(tensor, gain=gain)
        elif method == "kaiming_uniform":
            return nn.init.kaiming_uniform_(tensor, a=0, mode="fan_in", nonlinearity=nonlinearity)
        elif method == "kaiming_normal":
            return nn.init.kaiming_normal_(tensor, a=0, mode="fan_in", nonlinearity=nonlinearity)
        elif method == "orthogonal":
            return nn.init.orthogonal_(tensor, gain=gain)
        elif method == "normal":
            return nn.init.normal_(tensor, mean=mean, std=std)
        elif method == "uniform":
            return nn.init.uniform_(tensor, a=a, b=b)
        elif method == "zeros":
            return nn.init.zeros_(tensor)
        else:
            raise ValueError(
                f"Unknown initialization method: '{method}'. "
                f"Choose from: xavier_uniform, xavier_normal, kaiming_uniform, "
                f"kaiming_normal, orthogonal, normal, uniform, zeros"
            )
    return init_fn


def get_activation(name: Optional[ActivationType]) -> nn.Module:
    """
    Get activation module by name.

    Args:
        name: Activation function name. If None, returns Identity.

    Returns:
        PyTorch activation module.

    Raises:
        ValueError: If activation name is not recognized.
    """
    activations = {
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "swish": nn.SiLU,  # SiLU is the same as Swish
        "mish": nn.Mish,
        "linear": nn.Identity,
        None: nn.Identity,
    }

    if name not in activations:
        raise ValueError(
            f"Unknown activation: '{name}'. "
            f"Choose from: {list(activations.keys())}"
        )

    act_class = activations[name]
    return act_class() if callable(act_class) else act_class


# =============================================================================
# Core Spatial Layers
# =============================================================================


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


# =============================================================================
# Enhanced MLP Layers
# =============================================================================


class SpatialMLP(nn.Module):
    """
    Spatial Multi-Layer Perceptron for HEALPix grids.

    This layer processes spatial connections through multiple dense layers,
    enabling more complex non-linear transformations of neighborhood features.
    It gathers features from neighboring pixels and processes them through
    an MLP stack with optional dropout, batch normalization, and residual
    connections.

    The operation:
        1. Gather k neighbor features for each output point
        2. Flatten the neighbor features: [B, N_out, k * C_in]
        3. Process through MLP layers with specified activations
        4. Output: [B, N_out, hidden_units[-1]]

    Args:
        output_points: Number of spatial points in the output.
        connection_indices: Integer array of shape [output_points, kernel_size]
            containing indices of input pixels for each output pixel.
        hidden_units: List of hidden layer dimensions. The last value
            determines the output feature dimension.
        activations: List of activation function names, one per hidden layer.
            Options: "relu", "selu", "leaky_relu", "gelu", "elu", "tanh",
            "sigmoid", "swish", "mish", "linear".
        dropout: Dropout probability applied after each layer (except last).
            Default is 0.0 (no dropout).
        use_batch_norm: Whether to apply batch normalization after each layer.
            Default is False.
        residual: Whether to add residual connection if input/output dims match.
            Default is False.
        weight_init: Weight initialization method. Default is "xavier_uniform".

    Shape:
        - Input: [B, N_in, C_in]
        - Output: [B, N_out, hidden_units[-1]]

    Example:
        >>> from idx_flow.utils import compute_connection_indices
        >>> indices, _ = compute_connection_indices(
        ...     nside_in=64, nside_out=32, k=4
        ... )
        >>> mlp = SpatialMLP(
        ...     output_points=12 * 32**2,
        ...     connection_indices=indices,
        ...     hidden_units=[64, 64, 32],
        ...     activations=["gelu", "gelu", "linear"],
        ...     dropout=0.1,
        ...     use_batch_norm=True
        ... )
        >>> x = torch.randn(8, 12 * 64**2, 16)
        >>> y = mlp(x)
        >>> print(y.shape)  # torch.Size([8, 12288, 32])
    """

    def __init__(
        self,
        output_points: int,
        connection_indices: NDArray[np.int64],
        hidden_units: Sequence[int] = (32, 32, 32),
        activations: Sequence[Optional[ActivationType]] = ("linear", "linear", "linear"),
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        residual: bool = False,
        weight_init: InitMethod = "xavier_uniform",
    ) -> None:
        super().__init__()

        if len(hidden_units) != len(activations):
            raise ValueError(
                f"Length of hidden_units ({len(hidden_units)}) must match "
                f"length of activations ({len(activations)})"
            )

        self.output_points = output_points
        self.kernel_size = connection_indices.shape[1]
        self.hidden_units = list(hidden_units)
        self.output_channels = hidden_units[-1]
        self.activations_names = list(activations)
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm
        self.use_residual = residual
        self.weight_init = weight_init

        self.register_buffer(
            "connection_indices",
            torch.from_numpy(connection_indices.astype(np.int64)),
        )

        # Build activation functions
        self.activation_fns = nn.ModuleList([
            get_activation(act_name) for act_name in activations
        ])

        # MLP layers will be lazily initialized
        self.mlp_layers: Optional[nn.ModuleList] = None
        self.bn_layers: Optional[nn.ModuleList] = None
        self.dropout_layers: Optional[nn.ModuleList] = None
        self.residual_proj: Optional[nn.Linear] = None
        self._initialized = False
        self._input_dim: Optional[int] = None

    def _initialize_parameters(self, in_channels: int) -> None:
        """Initialize MLP layers based on input channels."""
        self.mlp_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if self.use_batch_norm else None
        self.dropout_layers = nn.ModuleList() if self.dropout_rate > 0 else None

        input_dim = self.kernel_size * in_channels
        self._input_dim = input_dim
        init_fn = get_initializer(self.weight_init)

        for i, hidden_dim in enumerate(self.hidden_units):
            layer = nn.Linear(input_dim if i == 0 else self.hidden_units[i - 1], hidden_dim)
            init_fn(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            self.mlp_layers.append(layer)

            if self.use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

            if self.dropout_rate > 0 and i < len(self.hidden_units) - 1:
                self.dropout_layers.append(nn.Dropout(self.dropout_rate))

        # Residual projection if dimensions don't match
        if self.use_residual and input_dim != self.output_channels:
            self.residual_proj = nn.Linear(input_dim, self.output_channels)
            init_fn(self.residual_proj.weight)

        self._initialized = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the spatial MLP.

        Args:
            x: Input tensor of shape [B, N_in, C_in].

        Returns:
            Output tensor of shape [B, N_out, hidden_units[-1]].
        """
        batch_size, input_points, in_channels = x.shape

        if not self._initialized:
            self._initialize_parameters(in_channels)
            if self.mlp_layers is not None:
                self.mlp_layers = self.mlp_layers.to(x.device)
            if self.bn_layers is not None:
                self.bn_layers = self.bn_layers.to(x.device)
            if self.dropout_layers is not None:
                self.dropout_layers = self.dropout_layers.to(x.device)
            if self.residual_proj is not None:
                self.residual_proj = self.residual_proj.to(x.device)

        # Gather neighbor features: [B, N_out, kernel_size, C_in]
        neighbors = x[:, self.connection_indices, :]

        # Reshape for MLP: [B * N_out, kernel_size * C_in]
        mlp_input = neighbors.reshape(
            batch_size * self.output_points, self.kernel_size * in_channels
        )

        # Store for residual
        residual = mlp_input if self.use_residual else None

        # Process through MLP layers
        out = mlp_input
        for i, layer in enumerate(self.mlp_layers):
            out = layer(out)

            if self.use_batch_norm and self.bn_layers is not None:
                out = self.bn_layers[i](out)

            out = self.activation_fns[i](out)

            if self.dropout_layers is not None and i < len(self.dropout_layers):
                out = self.dropout_layers[i](out)

        # Add residual connection
        if self.use_residual and residual is not None:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            out = out + residual

        # Reshape output: [B, N_out, output_channels]
        output = out.reshape(batch_size, self.output_points, self.output_channels)

        return output

    def extra_repr(self) -> str:
        """Return a string representation of layer parameters."""
        return (
            f"output_points={self.output_points}, "
            f"kernel_size={self.kernel_size}, "
            f"hidden_units={self.hidden_units}, "
            f"dropout={self.dropout_rate}, "
            f"batch_norm={self.use_batch_norm}, "
            f"residual={self.use_residual}"
        )


class GlobalMLP(nn.Module):
    """
    Global MLP for channel-wise transformations on spatial data.

    This layer applies a shared MLP to each spatial point independently,
    transforming the feature channels without spatial mixing. Useful for
    pointwise feature transformations in encoder-decoder architectures.

    The operation applies the same MLP to each of the N spatial points:
        Y[b,n,:] = MLP(X[b,n,:]) for all n

    Args:
        hidden_units: List of hidden layer dimensions. The last value
            determines the output feature dimension.
        activations: List of activation function names, one per hidden layer.
        dropout: Dropout probability applied after each layer (except last).
        use_batch_norm: Whether to apply batch normalization.
        residual: Whether to add residual connection if input/output dims match.
        weight_init: Weight initialization method.

    Shape:
        - Input: [B, N, C_in]
        - Output: [B, N, hidden_units[-1]]

    Example:
        >>> mlp = GlobalMLP(
        ...     hidden_units=[64, 128, 64],
        ...     activations=["gelu", "gelu", "linear"],
        ...     dropout=0.1,
        ...     residual=True
        ... )
        >>> x = torch.randn(8, 12288, 32)
        >>> # First call initializes based on input channels
        >>> y = mlp(x)
        >>> print(y.shape)  # torch.Size([8, 12288, 64])
    """

    def __init__(
        self,
        hidden_units: Sequence[int] = (64, 64),
        activations: Sequence[Optional[ActivationType]] = ("gelu", "linear"),
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        residual: bool = False,
        weight_init: InitMethod = "xavier_uniform",
    ) -> None:
        super().__init__()

        if len(hidden_units) != len(activations):
            raise ValueError(
                f"Length of hidden_units ({len(hidden_units)}) must match "
                f"length of activations ({len(activations)})"
            )

        self.hidden_units = list(hidden_units)
        self.output_channels = hidden_units[-1]
        self.activations_names = list(activations)
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm
        self.use_residual = residual
        self.weight_init = weight_init

        # Build activation functions
        self.activation_fns = nn.ModuleList([
            get_activation(act_name) for act_name in activations
        ])

        # Layers will be lazily initialized
        self.mlp_layers: Optional[nn.ModuleList] = None
        self.bn_layers: Optional[nn.ModuleList] = None
        self.dropout_layers: Optional[nn.ModuleList] = None
        self.residual_proj: Optional[nn.Linear] = None
        self._initialized = False
        self._in_channels: Optional[int] = None

    def _initialize_parameters(self, in_channels: int) -> None:
        """Initialize MLP layers based on input channels."""
        self._in_channels = in_channels
        self.mlp_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if self.use_batch_norm else None
        self.dropout_layers = nn.ModuleList() if self.dropout_rate > 0 else None

        init_fn = get_initializer(self.weight_init)
        prev_dim = in_channels

        for i, hidden_dim in enumerate(self.hidden_units):
            layer = nn.Linear(prev_dim, hidden_dim)
            init_fn(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            self.mlp_layers.append(layer)
            prev_dim = hidden_dim

            if self.use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(hidden_dim))

            if self.dropout_rate > 0 and i < len(self.hidden_units) - 1:
                self.dropout_layers.append(nn.Dropout(self.dropout_rate))

        # Residual projection if dimensions don't match
        if self.use_residual and in_channels != self.output_channels:
            self.residual_proj = nn.Linear(in_channels, self.output_channels)
            init_fn(self.residual_proj.weight)

        self._initialized = True

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the global MLP.

        Args:
            x: Input tensor of shape [B, N, C_in].

        Returns:
            Output tensor of shape [B, N, hidden_units[-1]].
        """
        batch_size, num_points, in_channels = x.shape

        if not self._initialized:
            self._initialize_parameters(in_channels)
            if self.mlp_layers is not None:
                self.mlp_layers = self.mlp_layers.to(x.device)
            if self.bn_layers is not None:
                self.bn_layers = self.bn_layers.to(x.device)
            if self.dropout_layers is not None:
                self.dropout_layers = self.dropout_layers.to(x.device)
            if self.residual_proj is not None:
                self.residual_proj = self.residual_proj.to(x.device)

        # Reshape for processing: [B * N, C_in]
        out = x.reshape(batch_size * num_points, in_channels)
        residual = out if self.use_residual else None

        # Process through MLP layers
        for i, layer in enumerate(self.mlp_layers):
            out = layer(out)

            if self.use_batch_norm and self.bn_layers is not None:
                out = self.bn_layers[i](out)

            out = self.activation_fns[i](out)

            if self.dropout_layers is not None and i < len(self.dropout_layers):
                out = self.dropout_layers[i](out)

        # Add residual connection
        if self.use_residual and residual is not None:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            out = out + residual

        # Reshape output: [B, N, output_channels]
        output = out.reshape(batch_size, num_points, self.output_channels)

        return output

    def extra_repr(self) -> str:
        """Return a string representation of layer parameters."""
        return (
            f"hidden_units={self.hidden_units}, "
            f"dropout={self.dropout_rate}, "
            f"batch_norm={self.use_batch_norm}, "
            f"residual={self.use_residual}"
        )


# =============================================================================
# Pooling and Normalization Layers
# =============================================================================


class SpatialPooling(nn.Module):
    """
    Spatial Pooling layer for HEALPix grids.

    Performs pooling operations (mean, max, or sum) over local neighborhoods
    on the spherical grid. This is a non-learnable layer useful for
    downsampling with simple aggregation.

    Args:
        output_points: Number of spatial points in the output.
        connection_indices: Integer array of shape [output_points, kernel_size].
        pool_type: Type of pooling operation. One of "mean", "max", "sum".

    Shape:
        - Input: [B, N_in, C_in]
        - Output: [B, N_out, C_in] (channels preserved)

    Example:
        >>> from idx_flow.utils import compute_connection_indices
        >>> indices, _ = compute_connection_indices(
        ...     nside_in=64, nside_out=32, k=4
        ... )
        >>> pool = SpatialPooling(
        ...     output_points=12 * 32**2,
        ...     connection_indices=indices,
        ...     pool_type="mean"
        ... )
        >>> x = torch.randn(8, 12 * 64**2, 32)
        >>> y = pool(x)
        >>> print(y.shape)  # torch.Size([8, 12288, 32])
    """

    def __init__(
        self,
        output_points: int,
        connection_indices: NDArray[np.int64],
        pool_type: PoolingMethod = "mean",
    ) -> None:
        super().__init__()

        self.output_points = output_points
        self.kernel_size = connection_indices.shape[1]
        self.pool_type = pool_type

        self.register_buffer(
            "connection_indices",
            torch.from_numpy(connection_indices.astype(np.int64)),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of spatial pooling.

        Args:
            x: Input tensor of shape [B, N_in, C_in].

        Returns:
            Output tensor of shape [B, N_out, C_in].
        """
        neighbors = x[:, self.connection_indices, :]

        if self.pool_type == "mean":
            output = torch.mean(neighbors, dim=2)
        elif self.pool_type == "max":
            output = torch.max(neighbors, dim=2)[0]
        elif self.pool_type == "sum":
            output = torch.sum(neighbors, dim=2)
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

        return output

    def extra_repr(self) -> str:
        """Return a string representation of layer parameters."""
        return (
            f"output_points={self.output_points}, "
            f"kernel_size={self.kernel_size}, "
            f"pool_type='{self.pool_type}'"
        )


class SpatialBatchNorm(nn.Module):
    """
    Batch Normalization for spatial data on HEALPix grids.

    Applies batch normalization across the spatial and batch dimensions,
    normalizing per channel.

    Args:
        num_features: Number of feature channels.
        eps: Small constant for numerical stability. Default: 1e-5.
        momentum: Momentum for running statistics. Default: 0.1.
        affine: Whether to include learnable affine parameters. Default: True.

    Shape:
        - Input: [B, N, C]
        - Output: [B, N, C]

    Example:
        >>> bn = SpatialBatchNorm(num_features=64)
        >>> x = torch.randn(8, 12288, 64)
        >>> y = bn(x)
        >>> print(y.shape)  # torch.Size([8, 12288, 64])
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of spatial batch normalization.

        Args:
            x: Input tensor of shape [B, N, C].

        Returns:
            Output tensor of shape [B, N, C].
        """
        x = x.transpose(1, 2)  # [B, C, N]
        x = self.bn(x)
        x = x.transpose(1, 2)  # [B, N, C]
        return x


class SpatialLayerNorm(nn.Module):
    """
    Layer Normalization for spatial data on HEALPix grids.

    Applies layer normalization across the feature dimension for each
    spatial point independently. Unlike BatchNorm, this normalizes
    across features rather than across the batch.

    Args:
        num_features: Number of feature channels.
        eps: Small constant for numerical stability. Default: 1e-6.
        elementwise_affine: Whether to include learnable affine parameters.

    Shape:
        - Input: [B, N, C]
        - Output: [B, N, C]

    Example:
        >>> ln = SpatialLayerNorm(num_features=64)
        >>> x = torch.randn(8, 12288, 64)
        >>> y = ln(x)
        >>> print(y.shape)  # torch.Size([8, 12288, 64])
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(num_features, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of spatial layer normalization.

        Args:
            x: Input tensor of shape [B, N, C].

        Returns:
            Output tensor of shape [B, N, C].
        """
        return self.ln(x)


class SpatialInstanceNorm(nn.Module):
    """
    Instance Normalization for spatial data on HEALPix grids.

    Applies instance normalization across the spatial dimension for each
    channel independently. Useful for style transfer and generative models.

    Args:
        num_features: Number of feature channels.
        eps: Small constant for numerical stability. Default: 1e-5.
        momentum: Momentum for running statistics. Default: 0.1.
        affine: Whether to include learnable affine parameters. Default: False.

    Shape:
        - Input: [B, N, C]
        - Output: [B, N, C]

    Example:
        >>> instnorm = SpatialInstanceNorm(num_features=64, affine=True)
        >>> x = torch.randn(8, 12288, 64)
        >>> y = instnorm(x)
        >>> print(y.shape)  # torch.Size([8, 12288, 64])
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
    ) -> None:
        super().__init__()
        self.instance_norm = nn.InstanceNorm1d(
            num_features, eps=eps, momentum=momentum, affine=affine
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of spatial instance normalization.

        Args:
            x: Input tensor of shape [B, N, C].

        Returns:
            Output tensor of shape [B, N, C].
        """
        x = x.transpose(1, 2)  # [B, C, N]
        x = self.instance_norm(x)
        x = x.transpose(1, 2)  # [B, N, C]
        return x


class SpatialGroupNorm(nn.Module):
    """
    Group Normalization for spatial data on HEALPix grids.

    Divides channels into groups and normalizes within each group.
    Provides a middle ground between LayerNorm and InstanceNorm.

    Args:
        num_groups: Number of groups to divide channels into.
        num_channels: Number of feature channels (must be divisible by num_groups).
        eps: Small constant for numerical stability. Default: 1e-5.
        affine: Whether to include learnable affine parameters. Default: True.

    Shape:
        - Input: [B, N, C]
        - Output: [B, N, C]

    Example:
        >>> gn = SpatialGroupNorm(num_groups=8, num_channels=64)
        >>> x = torch.randn(8, 12288, 64)
        >>> y = gn(x)
        >>> print(y.shape)  # torch.Size([8, 12288, 64])
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of spatial group normalization.

        Args:
            x: Input tensor of shape [B, N, C].

        Returns:
            Output tensor of shape [B, N, C].
        """
        x = x.transpose(1, 2)  # [B, C, N]
        x = self.gn(x)
        x = x.transpose(1, 2)  # [B, N, C]
        return x


# =============================================================================
# Regularization Layers
# =============================================================================


class SpatialDropout(nn.Module):
    """
    Spatial Dropout for HEALPix grid data.

    Drops entire spatial locations (all channels for selected points) during
    training. This encourages the model to learn spatially robust features.

    Args:
        p: Probability of dropping a spatial location. Default: 0.1.

    Shape:
        - Input: [B, N, C]
        - Output: [B, N, C]

    Example:
        >>> dropout = SpatialDropout(p=0.2)
        >>> x = torch.randn(8, 12288, 64)
        >>> y = dropout(x)  # During training, some spatial points are zeroed
        >>> print(y.shape)  # torch.Size([8, 12288, 64])
    """

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Dropout probability must be between 0 and 1, got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of spatial dropout.

        Args:
            x: Input tensor of shape [B, N, C].

        Returns:
            Output tensor of shape [B, N, C].
        """
        if not self.training or self.p == 0.0:
            return x

        batch_size, num_points, num_channels = x.shape

        # Create dropout mask for spatial dimension: [B, N, 1]
        mask = torch.bernoulli(
            torch.full((batch_size, num_points, 1), 1 - self.p, device=x.device)
        )

        # Scale and apply mask
        return x * mask / (1 - self.p)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class ChannelDropout(nn.Module):
    """
    Channel Dropout for HEALPix grid data.

    Drops entire channels (all spatial points for selected channels) during
    training. This encourages the model to learn channel-robust features.

    Args:
        p: Probability of dropping a channel. Default: 0.1.

    Shape:
        - Input: [B, N, C]
        - Output: [B, N, C]

    Example:
        >>> dropout = ChannelDropout(p=0.2)
        >>> x = torch.randn(8, 12288, 64)
        >>> y = dropout(x)  # During training, some channels are zeroed
        >>> print(y.shape)  # torch.Size([8, 12288, 64])
    """

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Dropout probability must be between 0 and 1, got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of channel dropout.

        Args:
            x: Input tensor of shape [B, N, C].

        Returns:
            Output tensor of shape [B, N, C].
        """
        if not self.training or self.p == 0.0:
            return x

        batch_size, num_points, num_channels = x.shape

        # Create dropout mask for channel dimension: [B, 1, C]
        mask = torch.bernoulli(
            torch.full((batch_size, 1, num_channels), 1 - self.p, device=x.device)
        )

        # Scale and apply mask
        return x * mask / (1 - self.p)

    def extra_repr(self) -> str:
        return f"p={self.p}"


# =============================================================================
# Attention Layers
# =============================================================================


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


# =============================================================================
# Utility Layers
# =============================================================================


class Squeeze(nn.Module):
    """
    Squeeze layer that reduces spatial dimension to a single vector.

    Performs global aggregation over all spatial points using mean, max,
    or sum pooling.

    Args:
        reduction: Reduction method. One of "mean", "max", "sum".

    Shape:
        - Input: [B, N, C]
        - Output: [B, C]

    Example:
        >>> squeeze = Squeeze(reduction="mean")
        >>> x = torch.randn(8, 12288, 64)
        >>> y = squeeze(x)
        >>> print(y.shape)  # torch.Size([8, 64])
    """

    def __init__(self, reduction: Literal["mean", "max", "sum"] = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of squeeze.

        Args:
            x: Input tensor of shape [B, N, C].

        Returns:
            Output tensor of shape [B, C].
        """
        if self.reduction == "mean":
            return torch.mean(x, dim=1)
        elif self.reduction == "max":
            return torch.max(x, dim=1)[0]
        elif self.reduction == "sum":
            return torch.sum(x, dim=1)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

    def extra_repr(self) -> str:
        return f"reduction='{self.reduction}'"


class Unsqueeze(nn.Module):
    """
    Unsqueeze layer that broadcasts a vector to all spatial points.

    Takes a feature vector and replicates it across the spatial dimension.

    Args:
        num_points: Number of spatial points to broadcast to.

    Shape:
        - Input: [B, C]
        - Output: [B, num_points, C]

    Example:
        >>> unsqueeze = Unsqueeze(num_points=12288)
        >>> x = torch.randn(8, 64)
        >>> y = unsqueeze(x)
        >>> print(y.shape)  # torch.Size([8, 12288, 64])
    """

    def __init__(self, num_points: int) -> None:
        super().__init__()
        self.num_points = num_points

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of unsqueeze.

        Args:
            x: Input tensor of shape [B, C].

        Returns:
            Output tensor of shape [B, num_points, C].
        """
        return x.unsqueeze(1).expand(-1, self.num_points, -1)

    def extra_repr(self) -> str:
        return f"num_points={self.num_points}"
