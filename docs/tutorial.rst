Tutorial
========

This tutorial covers multi-scale architectures, attention, Vision Transformers,
and regularization with idx-flow.

HEALPix Grids
--------------

HEALPix divides the sphere into equal-area pixels. Resolution is controlled by
``nside``; pixel count is ``12 * nside^2``.

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

Multi-Resolution U-Net
-----------------------

.. code-block:: python

   import torch
   import torch.nn as nn
   from idx_flow import (
       SpatialConv,
       SpatialTransposeConv,
       SpatialBatchNorm,
       GlobalMLP,
       compute_connection_indices
   )

   class MultiScaleUNet(nn.Module):
       def __init__(self, in_channels=5, base_filters=32):
           super().__init__()

           # 128 -> 64 -> 32 -> 16 -> 32 -> 64 -> 128
           self.idx_128_64, _ = compute_connection_indices(128, 64, k=4)
           self.idx_64_32, _ = compute_connection_indices(64, 32, k=4)
           self.idx_32_16, _ = compute_connection_indices(32, 16, k=4)

           idx_16_32, _, w_16_32 = compute_connection_indices(16, 32, k=4, return_weights=True)
           idx_32_64, _, w_32_64 = compute_connection_indices(32, 64, k=4, return_weights=True)
           idx_64_128, _, w_64_128 = compute_connection_indices(64, 128, k=4, return_weights=True)

           f = base_filters

           # Encoder
           self.enc1 = self._down_block(12*64**2, self.idx_128_64, f)
           self.enc2 = self._down_block(12*32**2, self.idx_64_32, f*2)
           self.enc3 = self._down_block(12*16**2, self.idx_32_16, f*4)

           # Bottleneck
           self.bottleneck = GlobalMLP(
               hidden_units=[f*4, f*8, f*4],
               activations=["gelu", "gelu", "linear"],
               residual=True
           )

           # Decoder (with skip connections)
           self.dec3 = self._up_block(12*32**2, idx_16_32, w_16_32, f*2)
           self.dec2 = self._up_block(12*64**2, idx_32_64, w_32_64, f)
           self.dec1 = SpatialTransposeConv(12*128**2, idx_64_128, w_64_128, filters=in_channels)

           self.skip3 = nn.Linear(f*2, f*2)
           self.skip2 = nn.Linear(f, f)

       def _down_block(self, output_points, indices, filters):
           return nn.Sequential(
               SpatialConv(output_points, indices, filters=filters, weight_init="kaiming_normal"),
               SpatialBatchNorm(filters),
               nn.GELU()
           )

       def _up_block(self, output_points, indices, weights, filters):
           return nn.Sequential(
               SpatialTransposeConv(output_points, indices, weights, filters=filters),
               SpatialBatchNorm(filters),
               nn.GELU()
           )

       def forward(self, x):
           e1 = self.enc1(x)
           e2 = self.enc2(e1)
           e3 = self.enc3(e2)

           b = self.bottleneck(e3)

           d3 = self.dec3(b) + self.skip3(e2)
           d2 = self.dec2(d3) + self.skip2(e1)
           d1 = self.dec1(d2)
           return d1

Attention
---------

``SpatialSelfAttention`` is O(N^2) in spatial points, so use it on
low-resolution bottlenecks.

.. code-block:: python

   from idx_flow import SpatialSelfAttention, SpatialLayerNorm
   import torch.nn as nn

   class AttentionBlock(nn.Module):
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
           x = x + self.attn(self.norm1(x))
           x = x + self.mlp(self.norm2(x))
           return x

   block = AttentionBlock(embed_dim=128, num_heads=8)

Vision Transformer
------------------

``SpatialViT`` wraps patch embedding, positional encoding, and a stack of
transformer blocks into a single module:

.. code-block:: python

   import torch
   from idx_flow import SpatialViT, Squeeze, compute_connection_indices
   import torch.nn as nn

   indices, _ = compute_connection_indices(nside_in=16, nside_out=8, k=9)

   vit = SpatialViT(
       output_points=12 * 8**2,
       connection_indices=indices,
       embed_dim=64,
       depth=4,
       num_heads=4,
       output_dim=32,
       dropout=0.1,
   )

   squeeze = Squeeze(reduction="mean")
   classifier = nn.Linear(32, 10)

   x = torch.randn(4, 12 * 16**2, 8)
   features = vit(x)         # [4, 768, 32]
   pooled = squeeze(features) # [4, 32]
   logits = classifier(pooled) # [4, 10]

Regularization
--------------

.. code-block:: python

   from idx_flow import SpatialDropout, ChannelDropout
   import torch

   spatial_drop = SpatialDropout(p=0.1)   # zeros entire spatial locations
   channel_drop = ChannelDropout(p=0.1)   # zeros entire channels

   x = torch.randn(4, 1000, 64)

   spatial_drop.train()
   y = spatial_drop(x)

   spatial_drop.eval()
   y = spatial_drop(x)  # identity in eval mode

Global Feature Extraction
-------------------------

``Squeeze`` reduces ``[B, N, C]`` to ``[B, C]``; ``Unsqueeze`` broadcasts
``[B, C]`` back to ``[B, N, C]``.

.. code-block:: python

   from idx_flow import Squeeze, Unsqueeze, GlobalMLP
   import torch
   import torch.nn as nn

   class GlobalEncoder(nn.Module):
       def __init__(self, in_channels, latent_dim):
           super().__init__()
           self.squeeze = Squeeze(reduction="mean")
           self.mlp = nn.Sequential(
               nn.Linear(in_channels, latent_dim * 2),
               nn.GELU(),
               nn.Linear(latent_dim * 2, latent_dim)
           )

       def forward(self, x):
           return self.mlp(self.squeeze(x))

   class GlobalDecoder(nn.Module):
       def __init__(self, latent_dim, out_channels, num_points):
           super().__init__()
           self.unsqueeze = Unsqueeze(num_points)
           self.mlp = GlobalMLP(
               hidden_units=[latent_dim * 2, out_channels],
               activations=["gelu", "linear"]
           )

       def forward(self, z):
           return self.mlp(self.unsqueeze(z))

Training Tips
-------------

- Use ``kaiming_normal`` init for GELU/ReLU activations.
- ``SpatialLayerNorm`` pairs well with attention blocks.
- Start with lr=1e-4 for AdamW.
- Larger batches stabilize BatchNorm statistics.

.. code-block:: python

   import torch.optim as optim

   model = MultiScaleUNet(in_channels=5)
   optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
   scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

   for epoch in range(100):
       # training loop ...
       scheduler.step()
