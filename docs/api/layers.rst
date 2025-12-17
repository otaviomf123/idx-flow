Layers API Reference
====================

This module provides PyTorch layers for processing data on spherical HEALPix grids
using index-based convolutions.

.. contents:: Table of Contents
   :local:
   :depth: 2

Core Spatial Layers
-------------------

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

SpatialPooling
^^^^^^^^^^^^^^

.. autoclass:: idx_flow.SpatialPooling
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

Utility Layers
--------------

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

Initialization and Activation Utilities
---------------------------------------

get_initializer
^^^^^^^^^^^^^^^

.. autofunction:: idx_flow.get_initializer

get_activation
^^^^^^^^^^^^^^

.. autofunction:: idx_flow.get_activation

Type Aliases
------------

The following type aliases are available for type hints:

.. py:data:: idx_flow.InitMethod

   Literal type for weight initialization methods:
   ``"xavier_uniform"``, ``"xavier_normal"``, ``"kaiming_uniform"``,
   ``"kaiming_normal"``, ``"orthogonal"``, ``"normal"``, ``"uniform"``, ``"zeros"``

.. py:data:: idx_flow.ActivationType

   Literal type for activation functions:
   ``"relu"``, ``"selu"``, ``"leaky_relu"``, ``"gelu"``, ``"elu"``,
   ``"tanh"``, ``"sigmoid"``, ``"swish"``, ``"mish"``, ``"linear"``

.. py:data:: idx_flow.InterpolationMethod

   Literal type for interpolation methods:
   ``"linear"``, ``"idw"``, ``"gaussian"``

.. py:data:: idx_flow.PoolingMethod

   Literal type for pooling methods:
   ``"mean"``, ``"max"``, ``"sum"``
