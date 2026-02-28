"""
End-to-end integration tests for idx-flow.
"""

import numpy as np
import pytest
import torch

from idx_flow import (
    SpatialConv,
    SpatialTransposeConv,
)


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
