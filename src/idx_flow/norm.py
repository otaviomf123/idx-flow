"""
Normalization layers for HEALPix Grid Processing.

This module implements normalization layers adapted for spatial data
on spherical HEALPix grids.
"""

import torch.nn as nn
from torch import Tensor


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
