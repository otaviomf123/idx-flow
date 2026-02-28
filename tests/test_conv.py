"""
Tests for idx-flow spatial convolution layers.
"""

import numpy as np
import pytest
import torch

from idx_flow import (
    SpatialConv,
    SpatialTransposeConv,
    SpatialUpsampling,
)


class TestSpatialConv:
    """Tests for SpatialConv layer."""

    def test_init(self, small_connection_indices):
        """Test layer initialization."""
        layer = SpatialConv(
            output_points=100,
            connection_indices=small_connection_indices,
            filters=32,
        )
        assert layer.output_points == 100
        assert layer.kernel_size == 4
        assert layer.filters == 32

    def test_forward_shape(self, small_connection_indices):
        """Test forward pass output shape."""
        layer = SpatialConv(
            output_points=100,
            connection_indices=small_connection_indices,
            filters=64,
        )

        batch_size = 8
        input_points = 400
        in_channels = 16

        x = torch.randn(batch_size, input_points, in_channels)
        y = layer(x)

        assert y.shape == (batch_size, 100, 64)

    def test_forward_with_kernel_weights(self, small_connection_indices, small_weights):
        """Test forward pass with kernel weights."""
        layer = SpatialConv(
            output_points=100,
            connection_indices=small_connection_indices,
            kernel_weights=small_weights,
            filters=32,
        )

        x = torch.randn(4, 400, 8)
        y = layer(x)

        assert y.shape == (4, 100, 32)

    def test_gradient_flow(self, small_connection_indices):
        """Test that gradients flow through the layer."""
        layer = SpatialConv(
            output_points=100,
            connection_indices=small_connection_indices,
            filters=32,
        )

        x = torch.randn(4, 400, 16, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_no_bias(self, small_connection_indices):
        """Test layer without bias."""
        layer = SpatialConv(
            output_points=100,
            connection_indices=small_connection_indices,
            filters=32,
            bias=False,
        )

        x = torch.randn(4, 400, 16)
        y = layer(x)

        assert layer.bias_param is None
        assert y.shape == (4, 100, 32)


class TestSpatialTransposeConv:
    """Tests for SpatialTransposeConv layer."""

    def test_init(self, small_connection_indices):
        """Test layer initialization."""
        layer = SpatialTransposeConv(
            output_points=100,
            connection_indices=small_connection_indices,
            filters=32,
        )
        assert layer.output_points == 100
        assert layer.filters == 32

    def test_forward_shape(self, small_connection_indices):
        """Test forward pass output shape (upsampling)."""
        # For upsampling: indices map from high-res output to low-res input
        layer = SpatialTransposeConv(
            output_points=100,
            connection_indices=small_connection_indices,
            filters=64,
        )

        x = torch.randn(4, 400, 32)
        y = layer(x)

        assert y.shape == (4, 100, 64)

    def test_forward_with_weights(self, small_connection_indices, small_weights):
        """Test forward pass with kernel weights."""
        layer = SpatialTransposeConv(
            output_points=100,
            connection_indices=small_connection_indices,
            kernel_weights=small_weights,
            filters=32,
        )

        x = torch.randn(4, 400, 16)
        y = layer(x)

        assert y.shape == (4, 100, 32)

    def test_gradient_flow(self, small_connection_indices):
        """Test gradient flow through transpose conv."""
        layer = SpatialTransposeConv(
            output_points=100,
            connection_indices=small_connection_indices,
            filters=32,
        )

        x = torch.randn(4, 400, 16, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None


class TestSpatialUpsampling:
    """Tests for SpatialUpsampling layer."""

    def test_init_linear(self, small_connection_indices, small_distances):
        """Test initialization with linear interpolation."""
        layer = SpatialUpsampling(
            output_points=100,
            connection_indices=small_connection_indices,
            distances=small_distances,
            interpolation="linear",
        )
        assert layer.interpolation == "linear"

    def test_init_idw(self, small_connection_indices, small_distances):
        """Test initialization with IDW interpolation."""
        layer = SpatialUpsampling(
            output_points=100,
            connection_indices=small_connection_indices,
            distances=small_distances,
            interpolation="idw",
        )
        assert layer.interpolation == "idw"

    def test_init_gaussian(self, small_connection_indices, small_distances):
        """Test initialization with Gaussian interpolation."""
        layer = SpatialUpsampling(
            output_points=100,
            connection_indices=small_connection_indices,
            distances=small_distances,
            interpolation="gaussian",
        )
        assert layer.interpolation == "gaussian"

    def test_forward_preserves_channels(self, small_connection_indices, small_distances):
        """Test that upsampling preserves channel count."""
        layer = SpatialUpsampling(
            output_points=100,
            connection_indices=small_connection_indices,
            distances=small_distances,
            interpolation="idw",
        )

        in_channels = 32
        x = torch.randn(4, 400, in_channels)
        y = layer(x)

        assert y.shape == (4, 100, in_channels)

    def test_invalid_interpolation(self, small_connection_indices, small_distances):
        """Test that invalid interpolation method raises error."""
        with pytest.raises(ValueError, match="Unsupported interpolation"):
            SpatialUpsampling(
                output_points=100,
                connection_indices=small_connection_indices,
                distances=small_distances,
                interpolation="invalid_method",
            )
