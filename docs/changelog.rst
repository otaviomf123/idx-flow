Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[0.2.0] - 2025-XX-XX
--------------------

Changed
^^^^^^^

- Split ``layers.py`` into separate modules by responsibility:
  ``conv``, ``mlp``, ``norm``, ``regularization``, ``attention``,
  ``vit``, ``pooling``, ``functional``.
  All public names remain importable from ``idx_flow`` directly.
- Split ``test_layers.py`` into per-module test files with shared
  fixtures in ``conftest.py``.
- Removed ``layers.py`` and ``test_vit_layers.py``.

[0.1.0] - 2025-01-XX
--------------------

Initial release.

Added
^^^^^

**Layers**

- ``SpatialConv`` -- spatial convolution for downsampling
- ``SpatialTransposeConv`` -- transpose convolution for upsampling
- ``SpatialUpsampling`` -- distance-based interpolation upsampling
- ``SpatialPooling`` -- mean, max, and sum pooling
- ``SpatialMLP`` -- MLP with spatial gathering, dropout, batch norm, residual connections
- ``GlobalMLP`` -- channel-wise MLP for pointwise transformations
- ``SpatialBatchNorm``, ``SpatialLayerNorm``, ``SpatialInstanceNorm``, ``SpatialGroupNorm``
- ``SpatialDropout``, ``ChannelDropout``
- ``SpatialSelfAttention`` -- multi-head self-attention
- ``SpatialPatchEmbedding``, ``SpatialTransformerBlock``, ``SpatialViT``
- ``Squeeze``, ``Unsqueeze``

**Utilities**

- ``compute_connection_indices``
- ``hp_distance``
- ``get_weights``
- ``get_healpix_resolution_info``
- ``get_initializer``, ``get_activation``

**Initialization options**: Xavier, Kaiming, orthogonal, normal, uniform, zeros.

**Activations**: ReLU, SELU, Leaky ReLU, GELU, ELU, Tanh, Sigmoid, Swish, Mish.
