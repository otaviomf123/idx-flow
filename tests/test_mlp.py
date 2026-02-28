"""
Tests for idx-flow MLP layers.
"""

import pytest
import torch

from idx_flow import SpatialMLP


class TestSpatialMLP:
    """Tests for SpatialMLP layer."""

    def test_init(self, small_connection_indices):
        """Test layer initialization."""
        layer = SpatialMLP(
            output_points=100,
            connection_indices=small_connection_indices,
            hidden_units=[64, 32, 16],
            activations=["relu", "relu", "linear"],
        )
        assert layer.output_channels == 16
        assert len(layer.hidden_units) == 3

    def test_forward_shape(self, small_connection_indices):
        """Test forward pass output shape."""
        layer = SpatialMLP(
            output_points=100,
            connection_indices=small_connection_indices,
            hidden_units=[64, 32],
            activations=["selu", "linear"],
        )

        x = torch.randn(4, 400, 16)
        y = layer(x)

        assert y.shape == (4, 100, 32)

    def test_all_activations(self, small_connection_indices):
        """Test all supported activation functions."""
        activations = ["relu", "selu", "leaky_relu", "tanh", "sigmoid", "linear"]

        for act in activations:
            layer = SpatialMLP(
                output_points=100,
                connection_indices=small_connection_indices,
                hidden_units=[32],
                activations=[act],
            )
            x = torch.randn(2, 400, 8)
            y = layer(x)
            assert y.shape == (2, 100, 32)

    def test_mismatched_hidden_activations(self, small_connection_indices):
        """Test that mismatched lengths raise error."""
        with pytest.raises(ValueError, match="Length of hidden_units"):
            SpatialMLP(
                output_points=100,
                connection_indices=small_connection_indices,
                hidden_units=[64, 32, 16],
                activations=["relu", "relu"],  # Wrong length
            )

    def test_invalid_activation(self, small_connection_indices):
        """Test that invalid activation raises error."""
        with pytest.raises(ValueError, match="Unknown activation"):
            SpatialMLP(
                output_points=100,
                connection_indices=small_connection_indices,
                hidden_units=[32],
                activations=["invalid_act"],
            )
