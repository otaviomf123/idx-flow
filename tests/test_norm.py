"""
Tests for idx-flow normalization layers.
"""

import torch

from idx_flow import SpatialBatchNorm


class TestSpatialBatchNorm:
    """Tests for SpatialBatchNorm layer."""

    def test_init(self):
        """Test layer initialization."""
        layer = SpatialBatchNorm(num_features=64)
        assert layer.bn.num_features == 64

    def test_forward_shape(self):
        """Test forward pass preserves shape."""
        layer = SpatialBatchNorm(num_features=32)

        x = torch.randn(8, 1000, 32)
        y = layer(x)

        assert y.shape == x.shape

    def test_training_mode(self):
        """Test layer in training mode."""
        layer = SpatialBatchNorm(num_features=32)
        layer.train()

        x = torch.randn(8, 1000, 32)
        y = layer(x)

        assert y.shape == x.shape

    def test_eval_mode(self):
        """Test layer in evaluation mode."""
        layer = SpatialBatchNorm(num_features=32)

        # Run some training batches to update running stats
        layer.train()
        for _ in range(10):
            x = torch.randn(8, 1000, 32)
            _ = layer(x)

        # Switch to eval mode
        layer.eval()
        x = torch.randn(4, 1000, 32)
        y = layer(x)

        assert y.shape == x.shape
