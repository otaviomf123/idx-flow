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

The simplest way to install idx-flow is via pip:

.. code-block:: bash

   pip install idx-flow

From Source
-----------

To install the latest development version from source:

.. code-block:: bash

   git clone https://github.com/otaviomf123/idx-flow.git
   cd idx-flow
   pip install -e .

For development with all optional dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

Verifying Installation
----------------------

After installation, you can verify that idx-flow is working correctly:

.. code-block:: python

   import torch
   from idx_flow import SpatialConv, compute_connection_indices

   # Compute connection indices
   indices, distances = compute_connection_indices(
       nside_in=16, nside_out=8, k=4
   )

   # Create a simple layer
   conv = SpatialConv(
       output_points=12 * 8**2,
       connection_indices=indices,
       filters=32
   )

   # Test forward pass
   x = torch.randn(2, 12 * 16**2, 16)
   y = conv(x)
   print(f"Output shape: {y.shape}")  # Should be [2, 768, 32]

Optional: GPU Support
---------------------

idx-flow automatically uses GPU if CUDA is available. To check:

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
