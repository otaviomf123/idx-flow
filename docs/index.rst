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

**idx-flow** is a PyTorch library for efficient neural network operations on spherical
data using HEALPix tessellation. This library implements index-based spherical
convolutions that achieve **O(N) computational complexity** while preserving the
equal-area properties essential for atmospheric and geophysical data analysis.

Citation
--------

.. important::

   If you use this library in your research, please cite the following paper:

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

Structure Compilation Philosophy
--------------------------------

The **idx-flow** library decouples **topology** from **computation**:

- **Connection indices** (topology) are precomputed once and stored as buffers
- **Learnable weights** (computation) are applied at runtime

This architectural design enables:

- **O(N) complexity** instead of O(N^2) for neighbor lookups
- **Flexible architecture design** with reusable index structures
- **Efficient memory usage** through shared topology buffers

Key Features
------------

- **Efficient O(N) Complexity**: Precomputed neighbor indices enable linear-time convolutions
- **HEALPix Native**: Built for the Hierarchical Equal Area isoLatitude Pixelization scheme
- **PyTorch Integration**: Seamless integration with PyTorch models and training pipelines
- **Flexible Architecture**: Support for encoder-decoder networks, GANs, and custom architectures
- **Multiple Layer Types**: Convolution, transpose convolution, upsampling, MLP, attention, and more

Quick Start
-----------

Installation
^^^^^^^^^^^^

.. code-block:: bash

   pip install idx-flow

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   import torch
   from idx_flow import SpatialConv, compute_connection_indices

   # Compute connection indices for downsampling (nside 64 -> 32)
   indices, distances = compute_connection_indices(
       nside_in=64, nside_out=32, k=4
   )

   # Create spatial convolution layer
   conv = SpatialConv(
       output_points=12 * 32**2,  # 12288 pixels
       connection_indices=indices,
       filters=64,
       weight_init="kaiming_normal"
   )

   # Forward pass
   x = torch.randn(8, 12 * 64**2, 32)  # [batch, points, channels]
   y = conv(x)
   print(y.shape)  # torch.Size([8, 12288, 64])

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

- **Monan Project**, **CEMPA Project**, **LAMCAD**, and **PGMet**
- CNPq grants (processes 422614/2021-1, and 315349/2023-9)
- National Institute for Space Research (INPE)
