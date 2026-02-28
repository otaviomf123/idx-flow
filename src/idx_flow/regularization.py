"""
Regularization layers for HEALPix Grid Processing.

This module implements dropout variants adapted for spatial data
on spherical HEALPix grids.
"""

import torch
import torch.nn as nn
from torch import Tensor


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
