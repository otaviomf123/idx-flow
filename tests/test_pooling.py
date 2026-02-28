"""
Tests for idx-flow pooling layers.
"""

import torch

from idx_flow import SpatialPooling


class TestSpatialPooling:
    """Tests for SpatialPooling layer."""

    def test_mean_pooling(self, small_connection_indices):
        """Test mean pooling."""
        layer = SpatialPooling(
            output_points=100,
            connection_indices=small_connection_indices,
            pool_type="mean",
        )

        x = torch.randn(4, 400, 32)
        y = layer(x)

        assert y.shape == (4, 100, 32)

    def test_max_pooling(self, small_connection_indices):
        """Test max pooling."""
        layer = SpatialPooling(
            output_points=100,
            connection_indices=small_connection_indices,
            pool_type="max",
        )

        x = torch.randn(4, 400, 32)
        y = layer(x)

        assert y.shape == (4, 100, 32)

    def test_sum_pooling(self, small_connection_indices):
        """Test sum pooling."""
        layer = SpatialPooling(
            output_points=100,
            connection_indices=small_connection_indices,
            pool_type="sum",
        )

        x = torch.randn(4, 400, 32)
        y = layer(x)

        assert y.shape == (4, 100, 32)

    def test_preserves_channels(self, small_connection_indices):
        """Test that pooling preserves channel count."""
        layer = SpatialPooling(
            output_points=100,
            connection_indices=small_connection_indices,
            pool_type="mean",
        )

        for channels in [1, 16, 64, 128]:
            x = torch.randn(2, 400, channels)
            y = layer(x)
            assert y.shape[-1] == channels
