"""
Tests for idx-flow Vision Transformer spatial layers.

This module contains comprehensive tests for the Vision Transformer layers
implemented in the idx-flow package: SpatialPatchEmbedding,
SpatialTransformerBlock, and SpatialViT.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from idx_flow import (
    SpatialPatchEmbedding,
    SpatialTransformerBlock,
    SpatialViT,
)


class TestSpatialPatchEmbedding:
    """Tests for SpatialPatchEmbedding layer."""

    def test_init(self, small_connection_indices):
        """Test layer initialization."""
        layer = SpatialPatchEmbedding(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
        )
        assert layer.output_points == 100
        assert layer.kernel_size == 4
        assert layer.embed_dim == 64
        assert not layer._initialized

    def test_forward_shape(self, small_connection_indices):
        """Test forward pass output shape."""
        layer = SpatialPatchEmbedding(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=128,
        )

        batch_size = 8
        input_points = 400
        in_channels = 16

        x = torch.randn(batch_size, input_points, in_channels)
        y = layer(x)

        assert y.shape == (batch_size, 100, 128)

    def test_lazy_initialization(self, small_connection_indices):
        """Test that parameters are lazily initialized on first forward."""
        layer = SpatialPatchEmbedding(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
        )
        assert layer.projection is None

        x = torch.randn(2, 400, 16)
        _ = layer(x)

        assert layer.projection is not None
        assert layer._initialized
        # Input dim should be kernel_size * in_channels = 4 * 16 = 64
        assert layer.projection.in_features == 64
        assert layer.projection.out_features == 64

    def test_no_bias(self, small_connection_indices):
        """Test layer without bias."""
        layer = SpatialPatchEmbedding(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
            bias=False,
        )

        x = torch.randn(4, 400, 16)
        y = layer(x)

        assert layer.projection.bias is None
        assert y.shape == (4, 100, 64)

    def test_gradient_flow(self, small_connection_indices):
        """Test that gradients flow through the layer."""
        layer = SpatialPatchEmbedding(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
        )

        x = torch.randn(4, 400, 16, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_different_embed_dims(self, small_connection_indices):
        """Test with various embedding dimensions."""
        for embed_dim in [32, 64, 128, 256]:
            layer = SpatialPatchEmbedding(
                output_points=100,
                connection_indices=small_connection_indices,
                embed_dim=embed_dim,
            )
            x = torch.randn(2, 400, 8)
            y = layer(x)
            assert y.shape == (2, 100, embed_dim)

    def test_larger_kernel(self, large_kernel_indices):
        """Test with larger kernel size."""
        layer = SpatialPatchEmbedding(
            output_points=100,
            connection_indices=large_kernel_indices,
            embed_dim=128,
        )
        assert layer.kernel_size == 9

        x = torch.randn(4, 400, 16)
        y = layer(x)
        assert y.shape == (4, 100, 128)

    def test_extra_repr(self, small_connection_indices):
        """Test string representation."""
        layer = SpatialPatchEmbedding(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
        )
        repr_str = layer.extra_repr()
        assert "output_points=100" in repr_str
        assert "embed_dim=64" in repr_str


class TestSpatialTransformerBlock:
    """Tests for SpatialTransformerBlock layer."""

    def test_init(self):
        """Test block initialization."""
        block = SpatialTransformerBlock(
            embed_dim=64,
            num_heads=4,
        )
        assert block.embed_dim == 64
        assert block.num_heads == 4

    def test_forward_shape(self):
        """Test forward pass preserves shape."""
        block = SpatialTransformerBlock(
            embed_dim=64,
            num_heads=4,
        )

        x = torch.randn(4, 100, 64)
        y = block(x)

        assert y.shape == (4, 100, 64)

    def test_forward_different_sequence_lengths(self):
        """Test with various spatial point counts."""
        block = SpatialTransformerBlock(
            embed_dim=64,
            num_heads=4,
        )

        for num_points in [50, 100, 200]:
            x = torch.randn(2, num_points, 64)
            y = block(x)
            assert y.shape == (2, num_points, 64)

    def test_gradient_flow(self):
        """Test gradient flow through the block."""
        block = SpatialTransformerBlock(
            embed_dim=64,
            num_heads=4,
        )

        x = torch.randn(4, 100, 64, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_with_dropout(self):
        """Test block with dropout."""
        block = SpatialTransformerBlock(
            embed_dim=64,
            num_heads=4,
            dropout=0.1,
        )

        block.train()
        x = torch.randn(4, 100, 64)
        y = block(x)
        assert y.shape == (4, 100, 64)

    def test_mlp_ratio(self):
        """Test custom MLP ratio."""
        block = SpatialTransformerBlock(
            embed_dim=64,
            num_heads=4,
            mlp_ratio=2.0,
        )
        assert block.mlp_ratio == 2.0

        x = torch.randn(4, 100, 64)
        y = block(x)
        assert y.shape == (4, 100, 64)

    def test_invalid_embed_dim(self):
        """Test that invalid embed_dim raises error."""
        with pytest.raises(ValueError, match="embed_dim .* must be divisible"):
            SpatialTransformerBlock(
                embed_dim=65,
                num_heads=4,
            )

    def test_residual_connection(self):
        """Test that residual connections are effective."""
        block = SpatialTransformerBlock(
            embed_dim=64,
            num_heads=4,
        )
        block.eval()

        x = torch.randn(2, 50, 64)
        y = block(x)

        # Output should not be zero (residual adds identity)
        assert not torch.allclose(y, torch.zeros_like(y))

    def test_different_activations(self):
        """Test with different FFN activation functions."""
        for act in ["relu", "gelu", "swish", "selu"]:
            block = SpatialTransformerBlock(
                embed_dim=64,
                num_heads=4,
                activation=act,
            )
            x = torch.randn(2, 50, 64)
            y = block(x)
            assert y.shape == (2, 50, 64)

    def test_extra_repr(self):
        """Test string representation."""
        block = SpatialTransformerBlock(
            embed_dim=128,
            num_heads=8,
            mlp_ratio=4.0,
        )
        repr_str = block.extra_repr()
        assert "embed_dim=128" in repr_str
        assert "num_heads=8" in repr_str


class TestSpatialViT:
    """Tests for SpatialViT layer."""

    def test_init(self, small_connection_indices):
        """Test full ViT initialization."""
        vit = SpatialViT(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
            depth=2,
            num_heads=4,
        )
        assert vit.output_points == 100
        assert vit.embed_dim == 64
        assert vit.depth == 2
        assert vit.num_heads == 4
        assert len(vit.blocks) == 2

    def test_forward_shape(self, small_connection_indices):
        """Test forward pass output shape."""
        vit = SpatialViT(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
            depth=2,
            num_heads=4,
        )

        x = torch.randn(4, 400, 16)
        y = vit(x)

        assert y.shape == (4, 100, 64)

    def test_forward_with_output_dim(self, small_connection_indices):
        """Test forward pass with different output dimension."""
        vit = SpatialViT(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
            depth=2,
            num_heads=4,
            output_dim=32,
        )

        x = torch.randn(4, 400, 16)
        y = vit(x)

        assert y.shape == (4, 100, 32)
        assert vit.output_proj is not None

    def test_forward_output_dim_equals_embed(self, small_connection_indices):
        """Test that no projection is created when output_dim equals embed_dim."""
        vit = SpatialViT(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
            depth=2,
            num_heads=4,
            output_dim=64,
        )

        assert vit.output_proj is None

        x = torch.randn(4, 400, 16)
        y = vit(x)
        assert y.shape == (4, 100, 64)

    def test_gradient_flow(self, small_connection_indices):
        """Test that gradients flow through the entire ViT."""
        vit = SpatialViT(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
            depth=2,
            num_heads=4,
            output_dim=32,
        )

        x = torch.randn(4, 400, 16, requires_grad=True)
        y = vit(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_to_all_parameters(self, small_connection_indices):
        """Test that gradients reach all learnable parameters."""
        vit = SpatialViT(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
            depth=2,
            num_heads=4,
            output_dim=32,
        )

        x = torch.randn(4, 400, 16)
        y = vit(x)
        loss = y.sum()
        loss.backward()

        for name, param in vit.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_with_dropout(self, small_connection_indices):
        """Test ViT with dropout."""
        vit = SpatialViT(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
            depth=2,
            num_heads=4,
            dropout=0.1,
        )

        vit.train()
        x = torch.randn(4, 400, 16)
        y = vit(x)
        assert y.shape == (4, 100, 64)

    def test_deterministic_eval(self, small_connection_indices):
        """Test that eval mode is deterministic."""
        vit = SpatialViT(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
            depth=2,
            num_heads=4,
            dropout=0.1,
        )

        # Initialize parameters
        x = torch.randn(2, 400, 16)
        _ = vit(x)

        vit.eval()
        y1 = vit(x)
        y2 = vit(x)

        assert torch.allclose(y1, y2)

    def test_various_depths(self, small_connection_indices):
        """Test ViT with different numbers of transformer blocks."""
        for depth in [1, 2, 4, 6]:
            vit = SpatialViT(
                output_points=100,
                connection_indices=small_connection_indices,
                embed_dim=64,
                depth=depth,
                num_heads=4,
            )
            assert len(vit.blocks) == depth

            x = torch.randn(2, 400, 8)
            y = vit(x)
            assert y.shape == (2, 100, 64)

    def test_invalid_embed_dim(self, small_connection_indices):
        """Test that invalid embed_dim raises error."""
        with pytest.raises(ValueError, match="embed_dim .* must be divisible"):
            SpatialViT(
                output_points=100,
                connection_indices=small_connection_indices,
                embed_dim=65,
                depth=2,
                num_heads=4,
            )

    def test_positional_embedding_shape(self, small_connection_indices):
        """Test that positional embedding has correct shape."""
        vit = SpatialViT(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
            depth=2,
            num_heads=4,
        )
        assert vit.pos_embed.shape == (1, 100, 64)

    def test_model_save_load(self, small_connection_indices, tmp_path):
        """Test saving and loading a ViT model."""
        vit = SpatialViT(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
            depth=2,
            num_heads=4,
            output_dim=32,
        )

        # Initialize by running forward pass
        x = torch.randn(2, 400, 16)
        _ = vit(x)

        # Save model
        save_path = tmp_path / "vit_model.pt"
        torch.save(vit.state_dict(), save_path)

        # Create new model and load weights
        new_vit = SpatialViT(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
            depth=2,
            num_heads=4,
            output_dim=32,
        )
        _ = new_vit(x)

        new_vit.load_state_dict(torch.load(save_path))

        # Verify outputs match
        vit.eval()
        new_vit.eval()

        y1 = vit(x)
        y2 = new_vit(x)

        assert torch.allclose(y1, y2)

    def test_parameter_count(self, small_connection_indices):
        """Test that model has the expected number of parameter groups."""
        vit = SpatialViT(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
            depth=2,
            num_heads=4,
            output_dim=32,
        )

        # Initialize
        x = torch.randn(2, 400, 8)
        _ = vit(x)

        total_params = sum(p.numel() for p in vit.parameters() if p.requires_grad)
        assert total_params > 0

    def test_extra_repr(self, small_connection_indices):
        """Test string representation."""
        vit = SpatialViT(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
            depth=2,
            num_heads=4,
            output_dim=32,
        )
        repr_str = vit.extra_repr()
        assert "output_points=100" in repr_str
        assert "embed_dim=64" in repr_str
        assert "depth=2" in repr_str
        assert "output_dim=32" in repr_str

    def test_with_larger_kernel(self, large_kernel_indices):
        """Test ViT with larger kernel size for patch embedding."""
        vit = SpatialViT(
            output_points=100,
            connection_indices=large_kernel_indices,
            embed_dim=64,
            depth=2,
            num_heads=4,
        )
        assert vit.kernel_size == 9

        x = torch.randn(4, 400, 8)
        y = vit(x)
        assert y.shape == (4, 100, 64)


class TestEndToEndViT:
    """End-to-end integration tests for ViT layers."""

    def test_vit_encoder_pipeline(self, small_connection_indices):
        """Test ViT as part of an encoder pipeline with SpatialConv."""
        from idx_flow import SpatialConv

        # First stage: convolution downsampling
        np.random.seed(99)
        conv_indices = np.random.randint(0, 800, size=(400, 4)).astype(np.int64)

        conv = SpatialConv(
            output_points=400,
            connection_indices=conv_indices,
            filters=32,
        )

        # Second stage: ViT processing
        vit = SpatialViT(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
            depth=2,
            num_heads=4,
            output_dim=32,
        )

        # Forward pass
        x = torch.randn(4, 800, 8)
        h = conv(x)
        y = vit(h)

        assert h.shape == (4, 400, 32)
        assert y.shape == (4, 100, 32)

    def test_vit_with_squeeze(self, small_connection_indices):
        """Test ViT followed by Squeeze for classification-like tasks."""
        from idx_flow import Squeeze

        vit = SpatialViT(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
            depth=2,
            num_heads=4,
            output_dim=32,
        )
        squeeze = Squeeze(reduction="mean")
        classifier = nn.Linear(32, 10)

        x = torch.randn(4, 400, 16)
        features = vit(x)
        pooled = squeeze(features)
        logits = classifier(pooled)

        assert features.shape == (4, 100, 32)
        assert pooled.shape == (4, 32)
        assert logits.shape == (4, 10)

    def test_full_gradient_pipeline(self, small_connection_indices):
        """Test full gradient flow through a ViT pipeline."""
        from idx_flow import Squeeze

        vit = SpatialViT(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
            depth=2,
            num_heads=4,
            output_dim=32,
        )
        squeeze = Squeeze(reduction="mean")
        head = nn.Linear(32, 10)

        x = torch.randn(4, 400, 16, requires_grad=True)
        y = head(squeeze(vit(x)))
        loss = y.sum()
        loss.backward()

        assert x.grad is not None

    def test_gpu_support(self, small_connection_indices):
        """Test GPU support if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")

        vit = SpatialViT(
            output_points=100,
            connection_indices=small_connection_indices,
            embed_dim=64,
            depth=2,
            num_heads=4,
        )
        vit = vit.to(device)

        x = torch.randn(4, 400, 16, device=device)
        y = vit(x)

        assert y.device.type == "cuda"
        assert y.shape == (4, 100, 64)
