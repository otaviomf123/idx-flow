Tutorial
========

This tutorial covers advanced usage of idx-flow for building spherical neural networks.

Understanding HEALPix Grids
---------------------------

HEALPix (Hierarchical Equal Area isoLatitude Pixelization) divides the sphere into
equal-area pixels. The resolution is controlled by the ``nside`` parameter:

- Number of pixels: ``npix = 12 * nside^2``
- Common values: nside = 32, 64, 128, 256

.. code-block:: python

   from idx_flow import get_healpix_resolution_info

   for nside in [32, 64, 128, 256]:
       info = get_healpix_resolution_info(nside)
       print(f"nside={nside}: {info['npix']:,} pixels, "
             f"{info['resolution_km']:.1f} km resolution")

Output::

   nside=32: 12,288 pixels, 110.4 km resolution
   nside=64: 49,152 pixels, 55.2 km resolution
   nside=128: 196,608 pixels, 27.6 km resolution
   nside=256: 786,432 pixels, 13.8 km resolution

Multi-Resolution Encoder-Decoder
--------------------------------

A common architecture progressively reduces resolution in the encoder and
increases it in the decoder:

.. code-block:: python

   import torch
   import torch.nn as nn
   from idx_flow import (
       SpatialConv,
       SpatialTransposeConv,
       SpatialBatchNorm,
       SpatialLayerNorm,
       GlobalMLP,
       Squeeze,
       compute_connection_indices
   )

   class MultiScaleUNet(nn.Module):
       """U-Net style architecture for spherical data."""

       def __init__(self, in_channels=5, base_filters=32):
           super().__init__()

           # Resolution levels: 128 -> 64 -> 32 -> 16 -> 32 -> 64 -> 128
           # Compute all connection indices
           self.idx_128_64, _ = compute_connection_indices(128, 64, k=4)
           self.idx_64_32, _ = compute_connection_indices(64, 32, k=4)
           self.idx_32_16, _ = compute_connection_indices(32, 16, k=4)

           idx_16_32, _, w_16_32 = compute_connection_indices(16, 32, k=4, return_weights=True)
           idx_32_64, _, w_32_64 = compute_connection_indices(32, 64, k=4, return_weights=True)
           idx_64_128, _, w_64_128 = compute_connection_indices(64, 128, k=4, return_weights=True)

           f = base_filters

           # Encoder
           self.enc1 = self._make_block(12*64**2, self.idx_128_64, f)
           self.enc2 = self._make_block(12*32**2, self.idx_64_32, f*2)
           self.enc3 = self._make_block(12*16**2, self.idx_32_16, f*4)

           # Bottleneck
           self.bottleneck = GlobalMLP(
               hidden_units=[f*4, f*8, f*4],
               activations=["gelu", "gelu", "linear"],
               residual=True
           )

           # Decoder (with skip connections)
           self.dec3 = self._make_up_block(12*32**2, idx_16_32, w_16_32, f*2)
           self.dec2 = self._make_up_block(12*64**2, idx_32_64, w_32_64, f)
           self.dec1 = SpatialTransposeConv(12*128**2, idx_64_128, w_64_128, filters=in_channels)

           # Projection layers for skip connections
           self.skip3 = nn.Linear(f*2, f*2)
           self.skip2 = nn.Linear(f, f)

       def _make_block(self, output_points, indices, filters):
           return nn.Sequential(
               SpatialConv(output_points, indices, filters=filters, weight_init="kaiming_normal"),
               SpatialBatchNorm(filters),
               nn.GELU()
           )

       def _make_up_block(self, output_points, indices, weights, filters):
           return nn.Sequential(
               SpatialTransposeConv(output_points, indices, weights, filters=filters),
               SpatialBatchNorm(filters),
               nn.GELU()
           )

       def forward(self, x):
           # Encoder with skip connections
           e1 = self.enc1(x)      # 128 -> 64
           e2 = self.enc2(e1)     # 64 -> 32
           e3 = self.enc3(e2)     # 32 -> 16

           # Bottleneck
           b = self.bottleneck(e3)

           # Decoder with skip connections
           d3 = self.dec3(b) + self.skip3(e2)     # 16 -> 32
           d2 = self.dec2(d3) + self.skip2(e1)    # 32 -> 64
           d1 = self.dec1(d2)                      # 64 -> 128

           return d1

Using Attention for Global Context
----------------------------------

For tasks requiring global context, use ``SpatialSelfAttention``:

.. code-block:: python

   from idx_flow import SpatialSelfAttention, SpatialLayerNorm
   import torch.nn as nn

   class AttentionBlock(nn.Module):
       """Transformer-style attention block for spatial data."""

       def __init__(self, embed_dim, num_heads, dropout=0.1):
           super().__init__()
           self.norm1 = SpatialLayerNorm(embed_dim)
           self.attn = SpatialSelfAttention(embed_dim, num_heads, dropout)
           self.norm2 = SpatialLayerNorm(embed_dim)
           self.mlp = nn.Sequential(
               nn.Linear(embed_dim, embed_dim * 4),
               nn.GELU(),
               nn.Dropout(dropout),
               nn.Linear(embed_dim * 4, embed_dim),
               nn.Dropout(dropout)
           )

       def forward(self, x):
           # Pre-norm architecture
           x = x + self.attn(self.norm1(x))
           x = x + self.mlp(self.norm2(x))
           return x

   # Use on low-resolution bottleneck (attention is O(N^2))
   attn_block = AttentionBlock(embed_dim=128, num_heads=8)

Regularization Techniques
-------------------------

idx-flow provides specialized dropout layers:

.. code-block:: python

   from idx_flow import SpatialDropout, ChannelDropout
   import torch

   # SpatialDropout: drops entire spatial locations
   spatial_drop = SpatialDropout(p=0.1)

   # ChannelDropout: drops entire channels
   channel_drop = ChannelDropout(p=0.1)

   x = torch.randn(4, 1000, 64)

   # In training mode
   spatial_drop.train()
   y = spatial_drop(x)  # Some spatial locations zeroed

   channel_drop.train()
   y = channel_drop(x)  # Some channels zeroed

   # In eval mode - no dropout
   spatial_drop.eval()
   y = spatial_drop(x)  # Identity operation

Global Feature Extraction
-------------------------

Use ``Squeeze`` to extract global features from spatial data:

.. code-block:: python

   from idx_flow import Squeeze, Unsqueeze, GlobalMLP
   import torch
   import torch.nn as nn

   class GlobalEncoder(nn.Module):
       """Extract global features from spherical data."""

       def __init__(self, in_channels, latent_dim):
           super().__init__()
           self.squeeze = Squeeze(reduction="mean")
           self.mlp = nn.Sequential(
               nn.Linear(in_channels, latent_dim * 2),
               nn.GELU(),
               nn.Linear(latent_dim * 2, latent_dim)
           )

       def forward(self, x):
           # x: [B, N, C] -> [B, C]
           global_feat = self.squeeze(x)
           # [B, C] -> [B, latent_dim]
           return self.mlp(global_feat)

   class GlobalDecoder(nn.Module):
       """Broadcast global features to spatial grid."""

       def __init__(self, latent_dim, out_channels, num_points):
           super().__init__()
           self.unsqueeze = Unsqueeze(num_points)
           self.mlp = GlobalMLP(
               hidden_units=[latent_dim * 2, out_channels],
               activations=["gelu", "linear"]
           )

       def forward(self, z):
           # z: [B, latent_dim] -> [B, N, latent_dim]
           spatial = self.unsqueeze(z)
           # [B, N, latent_dim] -> [B, N, out_channels]
           return self.mlp(spatial)

Training Tips
-------------

1. **Initialization**: Use ``kaiming_normal`` for GELU/ReLU activations
2. **Normalization**: ``SpatialLayerNorm`` works well with attention
3. **Learning Rate**: Start with 1e-4 for Adam optimizer
4. **Batch Size**: Larger batches help with BatchNorm stability

.. code-block:: python

   import torch.optim as optim

   model = MultiScaleUNet(in_channels=5)
   optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
   scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

   for epoch in range(100):
       # Training loop...
       scheduler.step()
