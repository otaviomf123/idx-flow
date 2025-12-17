"""
Utility functions for HEALPix grid operations and neighbor computation.

This module provides functions for computing geodesic distances and neighbor
indices between HEALPix grids at different resolutions, as well as various
distance-based weighting schemes for spatial interpolation.

Architecture based on the paper:
    Atmospheric Data Compression and Reconstruction Using Spherical GANs.
    DOI: 10.1109/IJCNN64981.2025.11227156

Author: Otavio Medeiros Feitosa
Institution: National Institute for Space Research (INPE)
"""

from typing import Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

try:
    import healpy as hp
    from sklearn.neighbors import BallTree

    HEALPY_AVAILABLE = True
except ImportError:
    HEALPY_AVAILABLE = False


WeightingMethod = Literal["inverse_square", "gaussian", "exponential", "tricube"]


def hp_distance(
    nside_in: int,
    nside_out: int,
    k: int,
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Calculate distances and neighbor indices between HEALPix grids.

    Computes the k-nearest neighbors from an input HEALPix grid to an output
    HEALPix grid using geodesic (haversine) distance on the sphere. This
    function is essential for constructing the connection indices used in
    spatial convolution layers.

    Args:
        nside_in: Input grid resolution parameter (nside). The number of pixels
            is 12 * nside_in^2.
        nside_out: Output grid resolution parameter (nside). The number of pixels
            is 12 * nside_out^2.
        k: Number of nearest neighbors to find for each output pixel.

    Returns:
        A tuple containing:
            - indices: Integer array of shape [N_out, k] containing the indices
              of the k-nearest input pixels for each output pixel.
            - distances_km: Float array of shape [N_out, k] containing the
              geodesic distances in kilometers to each neighbor.

    Raises:
        ImportError: If healpy or scikit-learn is not installed.
        ValueError: If nside_in, nside_out, or k are not positive integers.

    Example:
        >>> indices, distances = hp_distance(nside_in=64, nside_out=32, k=4)
        >>> print(indices.shape)  # (12288, 4) for nside_out=32
        >>> print(distances.shape)  # (12288, 4)

    Notes:
        - The Earth radius used for distance calculation is 6371 km.
        - Grid construction has O(N log N) complexity due to BallTree.
        - For same-resolution grids (nside_in == nside_out), the first
          neighbor will be the pixel itself with distance 0.
    """
    if not HEALPY_AVAILABLE:
        raise ImportError(
            "healpy and scikit-learn are required for hp_distance. "
            "Install them with: pip install healpy scikit-learn"
        )

    if nside_in <= 0 or nside_out <= 0 or k <= 0:
        raise ValueError("nside_in, nside_out, and k must be positive integers")

    # Input HEALPix grid parameters
    npix_in = hp.nside2npix(nside_in)
    pixel_indices_in = np.arange(npix_in)
    theta_rad_in, phi_rad_in = hp.pix2ang(nside_in, pixel_indices_in)

    # Convert to latitude/longitude (degrees)
    lat_in = np.rad2deg(theta_rad_in) - 90.0
    lon_in = np.rad2deg(phi_rad_in)
    lon_in = np.where(lon_in > 180.0, lon_in - 360.0, lon_in)

    # Output HEALPix grid parameters
    npix_out = hp.nside2npix(nside_out)
    pixel_indices_out = np.arange(npix_out)
    theta_rad_out, phi_rad_out = hp.pix2ang(nside_out, pixel_indices_out)

    # Convert to latitude/longitude (degrees)
    lat_out = np.rad2deg(theta_rad_out) - 90.0
    lon_out = np.rad2deg(phi_rad_out)
    lon_out = np.where(lon_out > 180.0, lon_out - 360.0, lon_out)

    # Build coordinate arrays in radians for haversine metric
    coords_in = np.column_stack(
        [np.deg2rad(lat_in.ravel()), np.deg2rad(lon_in.ravel())]
    )
    coords_out = np.column_stack(
        [np.deg2rad(lat_out.ravel()), np.deg2rad(lon_out.ravel())]
    )

    # Query nearest neighbors using BallTree with haversine metric
    tree = BallTree(coords_in, metric="haversine")
    distances_rad, indices = tree.query(coords_out, k=k)

    # Convert distances from radians to kilometers (Earth radius = 6371 km)
    distances_km = distances_rad * 6371.0

    return indices.astype(np.int64), distances_km.astype(np.float64)


def get_weights(
    distances: NDArray[np.float64],
    method: WeightingMethod = "inverse_square",
    sigma_factor: float = 0.5,
    epsilon: float = 1e-10,
) -> NDArray[np.float64]:
    """
    Calculate interpolation weights based on distances.

    Computes normalized weights for spatial interpolation using various
    distance-based weighting schemes. These weights are used in upsampling
    and convolution operations to aggregate features from neighboring pixels.

    Args:
        distances: Array of distances with shape [N, k] where N is the number
            of points and k is the number of neighbors.
        method: Weighting method to use. Options are:
            - "inverse_square": Inverse distance weighting (IDW) with power 2.
              Weight = 1 / (distance^2 + epsilon).
            - "gaussian": Gaussian kernel weighting.
              Weight = exp(-0.5 * (distance / sigma)^2).
            - "exponential": Exponential decay weighting.
              Weight = exp(-distance / scale).
            - "tricube": Tricube kernel weighting.
              Weight = (1 - (distance / max_distance)^3)^3.
        sigma_factor: Factor to multiply the mean distance for computing
            sigma (gaussian) or scale (exponential). Default is 0.5.
        epsilon: Small constant to prevent division by zero in inverse_square
            method. Default is 1e-10.

    Returns:
        Normalized weights array with same shape as distances. Weights sum
        to 1.0 along the last axis (neighbors dimension).

    Raises:
        ValueError: If an unsupported weighting method is specified.

    Example:
        >>> distances = np.array([[100.0, 200.0, 300.0], [150.0, 250.0, 350.0]])
        >>> weights = get_weights(distances, method="inverse_square")
        >>> print(weights.sum(axis=1))  # [1.0, 1.0]

    Notes:
        - All methods produce normalized weights that sum to 1 per point.
        - The inverse_square method is most common for IDW interpolation.
        - Gaussian and exponential methods use adaptive scaling based on
          the mean distance for each point.
    """
    if method == "inverse_square":
        weights = 1.0 / (distances**2 + epsilon)

    elif method == "gaussian":
        sigma = np.mean(distances, axis=1, keepdims=True) * sigma_factor
        sigma = np.maximum(sigma, epsilon)  # Prevent zero sigma
        weights = np.exp(-0.5 * (distances / sigma) ** 2)

    elif method == "exponential":
        scale = np.mean(distances, axis=1, keepdims=True) * sigma_factor
        scale = np.maximum(scale, epsilon)  # Prevent zero scale
        weights = np.exp(-distances / scale)

    elif method == "tricube":
        max_dist = np.max(distances, axis=1, keepdims=True)
        max_dist = np.maximum(max_dist, epsilon)  # Prevent zero max_dist
        normalized_dist = distances / max_dist
        weights = (1.0 - normalized_dist**3) ** 3

    else:
        raise ValueError(
            f"Unsupported weighting method: '{method}'. "
            f"Choose from: 'inverse_square', 'gaussian', 'exponential', 'tricube'"
        )

    # Normalize weights to sum to 1 along the neighbor axis
    weight_sums = np.sum(weights, axis=1, keepdims=True)
    weights = weights / np.maximum(weight_sums, epsilon)

    return weights.astype(np.float64)


def compute_connection_indices(
    nside_in: int,
    nside_out: int,
    k: int,
    return_weights: bool = False,
    weight_method: WeightingMethod = "inverse_square",
) -> Union[
    Tuple[NDArray[np.int64], NDArray[np.float64]],
    Tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]],
]:
    """
    Compute connection indices and optionally weights for spatial operations.

    This is a convenience function that combines hp_distance and get_weights
    for setting up spatial convolution or upsampling layers.

    Args:
        nside_in: Input grid resolution parameter (nside).
        nside_out: Output grid resolution parameter (nside).
        k: Number of nearest neighbors.
        return_weights: If True, also return computed interpolation weights.
        weight_method: Weighting method if return_weights is True.

    Returns:
        If return_weights is False:
            Tuple of (indices, distances_km)
        If return_weights is True:
            Tuple of (indices, distances_km, weights)

    Example:
        >>> # For downsampling (convolution)
        >>> indices, distances = compute_connection_indices(
        ...     nside_in=64, nside_out=32, k=4
        ... )
        >>> # For upsampling with weights
        >>> indices, distances, weights = compute_connection_indices(
        ...     nside_in=32, nside_out=64, k=4, return_weights=True
        ... )
    """
    indices, distances = hp_distance(nside_in, nside_out, k)

    if return_weights:
        weights = get_weights(distances, method=weight_method)
        return indices, distances, weights

    return indices, distances


def get_healpix_resolution_info(nside: int) -> dict:
    """
    Get information about a HEALPix resolution.

    Args:
        nside: HEALPix nside parameter.

    Returns:
        Dictionary containing:
            - nside: The nside parameter.
            - npix: Total number of pixels (12 * nside^2).
            - resolution_deg: Approximate resolution in degrees.
            - resolution_km: Approximate resolution in kilometers.
            - area_sr: Area per pixel in steradians.
            - area_km2: Approximate area per pixel in km^2.

    Example:
        >>> info = get_healpix_resolution_info(256)
        >>> print(f"Resolution: {info['resolution_deg']:.2f} degrees")
    """
    if not HEALPY_AVAILABLE:
        # Compute without healpy
        npix = 12 * nside * nside
        resolution_rad = np.sqrt(4 * np.pi / npix)
        resolution_deg = np.rad2deg(resolution_rad)
    else:
        npix = hp.nside2npix(nside)
        resolution_rad = hp.nside2resol(nside)
        resolution_deg = np.rad2deg(resolution_rad)

    # Earth radius in km
    earth_radius_km = 6371.0

    # Compute area per pixel
    area_sr = 4 * np.pi / npix  # steradians
    area_km2 = area_sr * earth_radius_km**2

    # Resolution in km
    resolution_km = resolution_deg * (np.pi / 180.0) * earth_radius_km

    return {
        "nside": nside,
        "npix": npix,
        "resolution_deg": resolution_deg,
        "resolution_km": resolution_km,
        "area_sr": area_sr,
        "area_km2": area_km2,
    }
