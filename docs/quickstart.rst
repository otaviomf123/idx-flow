Quick Start
===========

Concepts
--------

idx-flow works with **precomputed connection indices** that define which input
pixels connect to each output pixel on the HEALPix grid. Layers then apply
learnable (or fixed) transformations over these connections.

1. Use ``compute_connection_indices`` to get the topology.
2. Pass the indices to a layer (``SpatialConv``, ``SpatialMLP``, etc.).
3. Run the forward pass -- shapes are ``[B, N, C]`` throughout.

A Simple Encoder
----------------

.. code-block:: python

   import torch
   import torch.nn as nn
   from idx_flow import SpatialConv, SpatialBatchNorm, compute_connection_indices

   class SimpleEncoder(nn.Module):
       def __init__(self):
           super().__init__()

           indices, _ = compute_connection_indices(64, 32, k=4)

           self.conv = SpatialConv(
               output_points=12 * 32**2,
               connection_indices=indices,
               filters=64,
               weight_init="kaiming_normal"
           )
           self.bn = SpatialBatchNorm(64)
           self.activation = nn.GELU()

       def forward(self, x):
           return self.activation(self.bn(self.conv(x)))

   model = SimpleEncoder()
   x = torch.randn(4, 12 * 64**2, 32)
   y = model(x)
   print(f"Input: {x.shape} -> Output: {y.shape}")
   # Input: [4, 49152, 32] -> Output: [4, 12288, 64]

Autoencoder
-----------

.. code-block:: python

   import torch
   import torch.nn as nn
   from idx_flow import (
       SpatialConv,
       SpatialTransposeConv,
       SpatialBatchNorm,
       compute_connection_indices
   )

   class SphericalAutoencoder(nn.Module):
       def __init__(self, in_channels=5, latent_dim=64):
           super().__init__()

           idx_64_32, _ = compute_connection_indices(64, 32, k=4)
           idx_32_16, _ = compute_connection_indices(32, 16, k=4)

           idx_16_32, _, w_16_32 = compute_connection_indices(
               16, 32, k=4, return_weights=True
           )
           idx_32_64, _, w_32_64 = compute_connection_indices(
               32, 64, k=4, return_weights=True
           )

           # Encoder
           self.enc1 = SpatialConv(12*32**2, idx_64_32, filters=32)
           self.bn1 = SpatialBatchNorm(32)
           self.enc2 = SpatialConv(12*16**2, idx_32_16, filters=latent_dim)
           self.bn2 = SpatialBatchNorm(latent_dim)

           # Decoder
           self.dec1 = SpatialTransposeConv(12*32**2, idx_16_32, w_16_32, filters=32)
           self.bn3 = SpatialBatchNorm(32)
           self.dec2 = SpatialTransposeConv(12*64**2, idx_32_64, w_32_64, filters=in_channels)

           self.act = nn.GELU()

       def encode(self, x):
           x = self.act(self.bn1(self.enc1(x)))
           x = self.act(self.bn2(self.enc2(x)))
           return x

       def decode(self, z):
           x = self.act(self.bn3(self.dec1(z)))
           x = self.dec2(x)
           return x

       def forward(self, x):
           return self.decode(self.encode(x))

   model = SphericalAutoencoder(in_channels=5)
   x = torch.randn(2, 12*64**2, 5)
   y = model(x)
   print(f"Input: {x.shape} -> Output: {y.shape}")

Weight Initialization
---------------------

.. code-block:: python

   from idx_flow import SpatialConv
   import numpy as np

   indices = np.random.randint(0, 100, (50, 4))

   conv_xavier  = SpatialConv(50, indices, filters=32, weight_init="xavier_uniform")
   conv_kaiming = SpatialConv(50, indices, filters=32, weight_init="kaiming_normal")
   conv_ortho   = SpatialConv(50, indices, filters=32, weight_init="orthogonal")

SpatialMLP
----------

``SpatialMLP`` supports dropout, batch normalization, and residual connections:

.. code-block:: python

   from idx_flow import SpatialMLP
   import numpy as np
   import torch

   indices = np.random.randint(0, 100, (50, 4))

   mlp = SpatialMLP(
       output_points=50,
       connection_indices=indices,
       hidden_units=[64, 128, 64],
       activations=["gelu", "gelu", "linear"],
       dropout=0.1,
       use_batch_norm=True,
       residual=True,
       weight_init="kaiming_normal"
   )

   x = torch.randn(4, 100, 32)
   y = mlp(x)
   print(f"Output shape: {y.shape}")  # [4, 50, 64]

Next Steps
----------

- :doc:`tutorial` -- multi-scale architectures, attention, ViT, regularization
- :doc:`api/layers` -- full API reference
- :doc:`api/utils` -- utility functions
