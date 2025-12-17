Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[0.1.0] - 2025-01-XX
--------------------

Initial release of idx-flow.

Added
^^^^^

**Core Spatial Layers**

- ``SpatialConv``: Spatial convolution for downsampling on HEALPix grids
- ``SpatialTransposeConv``: Transpose convolution for upsampling
- ``SpatialUpsampling``: Distance-based interpolation upsampling
- ``SpatialPooling``: Mean, max, and sum pooling operations

**MLP Layers**

- ``SpatialMLP``: Multi-layer perceptron with spatial gathering, dropout, batch norm, and residual connections
- ``GlobalMLP``: Channel-wise MLP for pointwise transformations

**Normalization Layers**

- ``SpatialBatchNorm``: Batch normalization for spatial data
- ``SpatialLayerNorm``: Layer normalization for spatial data
- ``SpatialInstanceNorm``: Instance normalization for generative models
- ``SpatialGroupNorm``: Group normalization

**Regularization Layers**

- ``SpatialDropout``: Drops entire spatial locations
- ``ChannelDropout``: Drops entire channels

**Attention Layers**

- ``SpatialSelfAttention``: Multi-head self-attention for spatial data

**Utility Layers**

- ``Squeeze``: Global spatial aggregation to vector
- ``Unsqueeze``: Broadcast vector to spatial dimension

**Utility Functions**

- ``compute_connection_indices``: Convenience function for computing indices and weights
- ``hp_distance``: Compute neighbor indices and geodesic distances
- ``get_weights``: Calculate interpolation weights from distances
- ``get_healpix_resolution_info``: Get HEALPix resolution information
- ``get_initializer``: Get weight initialization functions
- ``get_activation``: Get activation modules by name

**Weight Initialization Options**

- Xavier (uniform and normal)
- Kaiming (uniform and normal)
- Orthogonal
- Normal and uniform distributions

**Activation Functions**

- ReLU, SELU, Leaky ReLU, GELU, ELU
- Tanh, Sigmoid, Swish, Mish
