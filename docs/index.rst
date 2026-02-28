idx-flow Documentation
======================

**Index-based Spherical Convolutions for HEALPix Grids in PyTorch**

.. image:: https://badge.fury.io/py/idx-flow.svg
   :target: https://badge.fury.io/py/idx-flow
   :alt: PyPI version

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

----

PyTorch layers for O(N) spherical convolutions on HEALPix grids.
Topology (connection indices) is precomputed once and stored as buffers;
learnable weights are applied at runtime.

Citation
--------

.. important::

   If you use this library in your research, please cite:

   **Atmospheric Data Compression and Reconstruction Using Spherical GANs**

   Otavio Medeiros Feitosa, Haroldo F. de Campos Velho, Saulo R. Freitas,
   Juliana Aparecida Anochi, Angel Dominguez Chovert, Cesar M. L. de Oliveira Junior

   *International Joint Conference on Neural Networks (IJCNN), 2025*

   **DOI:** `10.1109/IJCNN64981.2025.11227156 <https://doi.org/10.1109/IJCNN64981.2025.11227156>`_

   .. code-block:: bibtex

      @inproceedings{feitosa2025atmospheric,
         title={Atmospheric Data Compression and Reconstruction Using Spherical GANs},
         author={Feitosa, Otavio Medeiros and de Campos Velho, Haroldo F. and
                 Freitas, Saulo R. and Anochi, Juliana Aparecida and
                 Chovert, Angel Dominguez and de Oliveira Junior, Cesar M. L.},
         booktitle={International Joint Conference on Neural Networks (IJCNN)},
         year={2025},
         organization={IEEE},
         doi={10.1109/IJCNN64981.2025.11227156}
      }

Quick Example
-------------

.. code-block:: python

   import torch
   from idx_flow import SpatialConv, compute_connection_indices

   indices, distances = compute_connection_indices(
       nside_in=64, nside_out=32, k=4
   )

   conv = SpatialConv(
       output_points=12 * 32**2,
       connection_indices=indices,
       filters=64,
       weight_init="kaiming_normal"
   )

   x = torch.randn(8, 12 * 64**2, 32)  # [batch, points, channels]
   y = conv(x)
   print(y.shape)  # torch.Size([8, 12288, 64])

Package Layout
--------------

All public names are re-exported from ``idx_flow`` directly::

   from idx_flow import SpatialConv, SpatialViT, SpatialMLP

Internally the code is organized as:

- ``conv`` -- SpatialConv, SpatialTransposeConv, SpatialUpsampling
- ``mlp`` -- SpatialMLP, GlobalMLP
- ``norm`` -- SpatialBatchNorm, SpatialLayerNorm, SpatialInstanceNorm, SpatialGroupNorm
- ``regularization`` -- SpatialDropout, ChannelDropout
- ``attention`` -- SpatialSelfAttention
- ``vit`` -- SpatialPatchEmbedding, SpatialTransformerBlock, SpatialViT
- ``pooling`` -- SpatialPooling, Squeeze, Unsqueeze
- ``functional`` -- get_initializer, get_activation, type aliases
- ``utils`` -- hp_distance, get_weights, compute_connection_indices

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorial

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/layers
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog
   contributing

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Acknowledgments
---------------

- Monan Project, CEMPA Project, LAMCAD, PGMet
- CNPq (processes 422614/2021-1 and 315349/2023-9)
- National Institute for Space Research (INPE)
