Utilities API Reference
=======================

This module provides utility functions for computing HEALPix grid connections,
distances, and interpolation weights.

.. contents:: Table of Contents
   :local:
   :depth: 2

Connection Index Functions
--------------------------

compute_connection_indices
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: idx_flow.compute_connection_indices

hp_distance
^^^^^^^^^^^

.. autofunction:: idx_flow.hp_distance

Weighting Functions
-------------------

get_weights
^^^^^^^^^^^

.. autofunction:: idx_flow.get_weights

HEALPix Information
-------------------

get_healpix_resolution_info
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: idx_flow.get_healpix_resolution_info

Usage Examples
--------------

Computing Connection Indices for Downsampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from idx_flow import compute_connection_indices

   # Downsample from nside=64 to nside=32 with 4 neighbors
   indices, distances = compute_connection_indices(
       nside_in=64,
       nside_out=32,
       k=4
   )

   print(f"Indices shape: {indices.shape}")  # (12288, 4)
   print(f"Distances shape: {distances.shape}")  # (12288, 4)

Computing Connection Indices for Upsampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from idx_flow import compute_connection_indices

   # Upsample from nside=32 to nside=64 with weights
   indices, distances, weights = compute_connection_indices(
       nside_in=32,
       nside_out=64,
       k=4,
       return_weights=True,
       weight_method="inverse_square"
   )

   print(f"Indices shape: {indices.shape}")  # (49152, 4)
   print(f"Weights shape: {weights.shape}")  # (49152, 4)

Getting HEALPix Resolution Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from idx_flow import get_healpix_resolution_info

   info = get_healpix_resolution_info(nside=256)
   print(f"Number of pixels: {info['npix']}")
   print(f"Resolution (deg): {info['resolution_deg']:.3f}")
   print(f"Resolution (km): {info['resolution_km']:.2f}")
   print(f"Pixel area (km^2): {info['area_km2']:.2f}")
