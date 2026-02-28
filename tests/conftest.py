"""
Shared test fixtures for idx-flow tests.
"""

import numpy as np
import pytest


@pytest.fixture
def small_connection_indices():
    """Create small connection indices for testing."""
    output_points = 100
    kernel_size = 4
    input_points = 400
    # Random indices (simulating neighbor connections)
    np.random.seed(42)
    indices = np.random.randint(0, input_points, size=(output_points, kernel_size))
    return indices.astype(np.int64)


@pytest.fixture
def small_distances():
    """Create small distance array for testing."""
    output_points = 100
    kernel_size = 4
    np.random.seed(42)
    distances = np.random.uniform(50, 500, size=(output_points, kernel_size))
    return distances.astype(np.float64)


@pytest.fixture
def small_weights(small_distances):
    """Create normalized weights from distances."""
    weights = 1.0 / (small_distances**2 + 1e-10)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights.astype(np.float64)


@pytest.fixture
def large_kernel_indices():
    """Create connection indices with larger kernel for patch embedding."""
    output_points = 100
    kernel_size = 9
    input_points = 400
    np.random.seed(42)
    indices = np.random.randint(0, input_points, size=(output_points, kernel_size))
    return indices.astype(np.int64)
