"""
Tests for idx-flow spatial layers.

This module contains comprehensive tests for all spatial neural network layers
implemented in the idx-flow package.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from idx_flow import (
    SpatialBatchNorm,
    SpatialConv,
    SpatialMLP,
    SpatialPooling,
    SpatialTransposeConv,
    SpatialUpsampling,
)


# Test fixtures
@pytest.fixture
def small_connection_indices():
    """Create small connection indices for testing."""
    output_points = 100
    kernel_size = 4
    input_points = 400
    # Random indices (simulating neighbor connections)
    np.random.seed(42)
    indices = np.random.randint(0, input_points, size=(output_points, kernel_size))
    return indices.astype(np.int64)


@pytest.fixture
def small_distances():
    """Create small distance array for testing."""
    output_points = 100
    kernel_size = 4
    np.random.seed(42)
    distances = np.random.uniform(50, 500, size=(output_points, kernel_size))
    return distances.astype(np.float64)


@pytest.fixture
def small_weights(small_distances):
    """Create normalized weights from distances."""
    weights = 1.0 / (small_distances**2 + 1e-10)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights.astype(np.float64)


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


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_encoder_decoder_pipeline(self, small_connection_indices, small_weights):
        """Test a simple encoder-decoder pipeline."""
        # Encoder indices (downsampling)
        enc_indices = small_connection_indices

        # Decoder indices (upsampling - inverse mapping)
        np.random.seed(123)
        dec_indices = np.random.randint(0, 100, size=(400, 4)).astype(np.int64)
        dec_weights = np.random.uniform(0.1, 1.0, size=(400, 4)).astype(np.float64)
        dec_weights = dec_weights / np.sum(dec_weights, axis=1, keepdims=True)

        # Build encoder-decoder
        encoder = SpatialConv(
            output_points=100,
            connection_indices=enc_indices,
            filters=64,
        )

        decoder = SpatialTransposeConv(
            output_points=400,
            connection_indices=dec_indices,
            kernel_weights=dec_weights,
            filters=16,
        )

        # Forward pass
        x = torch.randn(4, 400, 16)
        latent = encoder(x)
        reconstruction = decoder(latent)

        assert latent.shape == (4, 100, 64)
        assert reconstruction.shape == (4, 400, 16)

    def test_model_save_load(self, small_connection_indices, tmp_path):
        """Test saving and loading a model."""
        layer = SpatialConv(
            output_points=100,
            connection_indices=small_connection_indices,
            filters=32,
        )

        # Initialize weights by running forward pass
        x = torch.randn(2, 400, 16)
        _ = layer(x)

        # Save model
        save_path = tmp_path / "model.pt"
        torch.save(layer.state_dict(), save_path)

        # Create new layer and load weights
        new_layer = SpatialConv(
            output_points=100,
            connection_indices=small_connection_indices,
            filters=32,
        )
        # Initialize the new layer
        _ = new_layer(x)

        # Load state dict
        new_layer.load_state_dict(torch.load(save_path))

        # Verify outputs match
        layer.eval()
        new_layer.eval()

        y1 = layer(x)
        y2 = new_layer(x)

        assert torch.allclose(y1, y2)

    def test_gpu_support(self, small_connection_indices):
        """Test GPU support if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")

        layer = SpatialConv(
            output_points=100,
            connection_indices=small_connection_indices,
            filters=32,
        )
        layer = layer.to(device)

        x = torch.randn(4, 400, 16, device=device)
        y = layer(x)

        assert y.device.type == "cuda"
        assert y.shape == (4, 100, 32)


class TestUtils:
    """Tests for utility functions."""

    def test_hp_distance_import(self):
        """Test hp_distance import (may skip if healpy not available)."""
        try:
            from idx_flow import hp_distance
        except ImportError:
            pytest.skip("healpy not installed")

    def test_get_weights(self):
        """Test get_weights function."""
        from idx_flow import get_weights

        distances = np.array([[100.0, 200.0, 300.0], [150.0, 250.0, 350.0]])

        # Test all methods
        for method in ["inverse_square", "gaussian", "exponential", "tricube"]:
            weights = get_weights(distances, method=method)
            assert weights.shape == distances.shape
            # Weights should sum to 1 per row
            np.testing.assert_allclose(weights.sum(axis=1), [1.0, 1.0], rtol=1e-5)

    def test_get_weights_invalid_method(self):
        """Test get_weights with invalid method."""
        from idx_flow import get_weights

        distances = np.array([[100.0, 200.0]])
        with pytest.raises(ValueError, match="Unsupported weighting method"):
            get_weights(distances, method="invalid")

    def test_get_healpix_resolution_info(self):
        """Test get_healpix_resolution_info function."""
        from idx_flow import get_healpix_resolution_info

        info = get_healpix_resolution_info(nside=64)

        assert info["nside"] == 64
        assert info["npix"] == 12 * 64**2
        assert info["resolution_deg"] > 0
        assert info["resolution_km"] > 0
        assert info["area_sr"] > 0
        assert info["area_km2"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
