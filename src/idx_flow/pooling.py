"""
Pooling and utility layers for HEALPix Grid Processing.

This module implements pooling operations and spatial dimension
manipulation layers for data on spherical HEALPix grids.
"""

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor

from idx_flow.functional import PoolingMethod


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
