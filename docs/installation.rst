Installation
============

Requirements
------------

- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- healpy >= 1.15.0
- scikit-learn >= 0.24.0

From PyPI
---------

.. code-block:: bash

   pip install idx-flow

Upgrading from a previous version:

.. code-block:: bash

   pip install --upgrade idx-flow

From Source
-----------

.. code-block:: bash

   git clone https://github.com/otaviomf123/idx-flow.git
   cd idx-flow
   pip install -e .

With dev dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

Verifying
---------

.. code-block:: python

   import torch
   from idx_flow import SpatialConv, compute_connection_indices

   indices, distances = compute_connection_indices(
       nside_in=16, nside_out=8, k=4
   )

   conv = SpatialConv(
       output_points=12 * 8**2,
       connection_indices=indices,
       filters=32
   )

   x = torch.randn(2, 12 * 16**2, 16)
   y = conv(x)
   print(f"Output shape: {y.shape}")  # [2, 768, 32]

GPU
---

idx-flow uses GPU automatically when CUDA is available:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")

   if torch.cuda.is_available():
       from idx_flow import SpatialConv
       import numpy as np

       indices = np.random.randint(0, 100, (50, 4))
       conv = SpatialConv(50, indices, filters=32).cuda()
       x = torch.randn(2, 100, 16).cuda()
       y = conv(x)
       print(f"Output device: {y.device}")
