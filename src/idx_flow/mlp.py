"""
MLP layers for HEALPix Grid Processing.

This module implements Spatial MLP and Global MLP layers for processing
data on spherical HEALPix grids.
"""

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor

from idx_flow.functional import ActivationType, InitMethod, get_activation, get_initializer


class SpatialMLP(nn.Module):
    """
    Spatial Multi-Layer Perceptron for local non-linear processing on HEALPix grids.

    This layer replaces the standard linear convolution kernel (matrix multiplication)
    with a shared **Dense Neural Network (MLP)** applied to the flattened neighborhood
    vector. This architectural choice is inspired by principles from Geometric Deep
    Learning, which extends neural network operations to non-Euclidean domains such
    as manifolds and graphs.

    Architectural Difference from SpatialConv:
        - **SpatialConv**: Applies a linear transformation to neighborhood features.
          The output is a weighted sum: Y = sum_k(W_k * X_k) + b, which can only
          capture linear relationships between neighbors.

        - **SpatialMLP**: Concatenates neighborhood features into a single vector
          and processes it through a multi-layer perceptron with non-linear
          activations: Y = MLP([X_1 || X_2 || ... || X_k]). This allows the model
          to learn **complex, non-linear interactions** between neighbors.

    Trade-offs:
        Benefits:
            - Significantly higher **representation capacity** and expressivity
            - Can approximate arbitrary non-linear functions over local patches
            - Better suited for learning complex spatial patterns on manifolds
            - Supports dropout, batch normalization, and residual connections

        Costs:
            - Higher computational cost (FLOPs) due to dense layer operations
            - Increased memory usage for storing MLP parameters
            - May require more training data to avoid overfitting

    The operation:
        1. Gather k neighbor features for each output point: [B, N_out, k, C_in]
        2. Flatten the neighbor features: [B, N_out, k * C_in]
        3. Process through shared MLP layers with specified activations
        4. Output: [B, N_out, hidden_units[-1]]

    Literature Context:
        This approach draws from Geometric Deep Learning research on extending
        neural networks to non-Euclidean domains:

        - Bronstein, M. M., Bruna, J., LeCun, Y., Szlam, A., & Vandergheynst, P.
          (2017). "Geometric deep learning: Going beyond Euclidean data."
          IEEE Signal Processing Magazine, 34(4), 18-42.
          DOI: 10.1109/MSP.2017.2693418

        - Masci, J., Boscaini, D., Bronstein, M. M., & Vandergheynst, P. (2015).
          "Geodesic convolutional neural networks on Riemannian manifolds."
          In Proceedings of the IEEE ICCV Workshops, pp. 37-45.
          DOI: 10.1109/ICCVW.2015.112

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

    See Also:
        - :class:`SpatialConv`: Linear convolution with lower computational cost
        - :class:`GlobalMLP`: Channel-wise MLP without spatial neighbor gathering
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
