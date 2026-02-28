Layers API Reference
====================

All layers are importable directly from ``idx_flow``. Internally they live in
separate modules (``conv``, ``mlp``, ``norm``, ``regularization``, ``attention``,
``vit``, ``pooling``, ``functional``).

.. contents:: Table of Contents
   :local:
   :depth: 2

Convolution Layers
------------------

SpatialConv
^^^^^^^^^^^

.. autoclass:: idx_flow.SpatialConv
   :members:
   :undoc-members:
   :show-inheritance:

SpatialTransposeConv
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: idx_flow.SpatialTransposeConv
   :members:
   :undoc-members:
   :show-inheritance:

SpatialUpsampling
^^^^^^^^^^^^^^^^^

.. autoclass:: idx_flow.SpatialUpsampling
   :members:
   :undoc-members:
   :show-inheritance:

MLP Layers
----------

SpatialMLP
^^^^^^^^^^

.. autoclass:: idx_flow.SpatialMLP
   :members:
   :undoc-members:
   :show-inheritance:

GlobalMLP
^^^^^^^^^

.. autoclass:: idx_flow.GlobalMLP
   :members:
   :undoc-members:
   :show-inheritance:

Normalization Layers
--------------------

SpatialBatchNorm
^^^^^^^^^^^^^^^^

.. autoclass:: idx_flow.SpatialBatchNorm
   :members:
   :undoc-members:
   :show-inheritance:

SpatialLayerNorm
^^^^^^^^^^^^^^^^

.. autoclass:: idx_flow.SpatialLayerNorm
   :members:
   :undoc-members:
   :show-inheritance:

SpatialInstanceNorm
^^^^^^^^^^^^^^^^^^^

.. autoclass:: idx_flow.SpatialInstanceNorm
   :members:
   :undoc-members:
   :show-inheritance:

SpatialGroupNorm
^^^^^^^^^^^^^^^^

.. autoclass:: idx_flow.SpatialGroupNorm
   :members:
   :undoc-members:
   :show-inheritance:

Regularization Layers
---------------------

SpatialDropout
^^^^^^^^^^^^^^

.. autoclass:: idx_flow.SpatialDropout
   :members:
   :undoc-members:
   :show-inheritance:

ChannelDropout
^^^^^^^^^^^^^^

.. autoclass:: idx_flow.ChannelDropout
   :members:
   :undoc-members:
   :show-inheritance:

Attention Layers
----------------

SpatialSelfAttention
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: idx_flow.SpatialSelfAttention
   :members:
   :undoc-members:
   :show-inheritance:

Vision Transformer Layers
-------------------------

SpatialPatchEmbedding
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: idx_flow.SpatialPatchEmbedding
   :members:
   :undoc-members:
   :show-inheritance:

SpatialTransformerBlock
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: idx_flow.SpatialTransformerBlock
   :members:
   :undoc-members:
   :show-inheritance:

SpatialViT
^^^^^^^^^^

.. autoclass:: idx_flow.SpatialViT
   :members:
   :undoc-members:
   :show-inheritance:

Pooling and Utility Layers
--------------------------

SpatialPooling
^^^^^^^^^^^^^^

.. autoclass:: idx_flow.SpatialPooling
   :members:
   :undoc-members:
   :show-inheritance:

Squeeze
^^^^^^^

.. autoclass:: idx_flow.Squeeze
   :members:
   :undoc-members:
   :show-inheritance:

Unsqueeze
^^^^^^^^^

.. autoclass:: idx_flow.Unsqueeze
   :members:
   :undoc-members:
   :show-inheritance:

Functional Utilities
--------------------

get_initializer
^^^^^^^^^^^^^^^

.. autofunction:: idx_flow.get_initializer

get_activation
^^^^^^^^^^^^^^

.. autofunction:: idx_flow.get_activation

Type Aliases
^^^^^^^^^^^^

.. py:data:: idx_flow.InitMethod

   Weight initialization methods:
   ``"xavier_uniform"``, ``"xavier_normal"``, ``"kaiming_uniform"``,
   ``"kaiming_normal"``, ``"orthogonal"``, ``"normal"``, ``"uniform"``, ``"zeros"``

.. py:data:: idx_flow.ActivationType

   Activation functions:
   ``"relu"``, ``"selu"``, ``"leaky_relu"``, ``"gelu"``, ``"elu"``,
   ``"tanh"``, ``"sigmoid"``, ``"swish"``, ``"mish"``, ``"linear"``

.. py:data:: idx_flow.InterpolationMethod

   Interpolation methods:
   ``"linear"``, ``"idw"``, ``"gaussian"``

.. py:data:: idx_flow.PoolingMethod

   Pooling methods:
   ``"mean"``, ``"max"``, ``"sum"``
