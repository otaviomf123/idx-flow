"""
Tests for idx-flow utility functions.

This module contains tests for the utility functions in idx_flow.utils.
"""

import numpy as np
import pytest

from idx_flow.utils import get_healpix_resolution_info, get_weights


class TestGetWeights:
    """Tests for get_weights function."""

    def test_inverse_square(self):
        """Test inverse square weighting."""
        distances = np.array([[100.0, 200.0, 400.0], [50.0, 100.0, 200.0]])
        weights = get_weights(distances, method="inverse_square")

        assert weights.shape == (2, 3)
        # Weights should sum to 1
        np.testing.assert_allclose(weights.sum(axis=1), [1.0, 1.0], rtol=1e-5)
        # Closer points should have higher weights
        assert weights[0, 0] > weights[0, 1] > weights[0, 2]

    def test_gaussian(self):
        """Test Gaussian weighting."""
        distances = np.array([[100.0, 200.0, 300.0]])
        weights = get_weights(distances, method="gaussian")

        assert weights.shape == (1, 3)
        np.testing.assert_allclose(weights.sum(axis=1), [1.0], rtol=1e-5)

    def test_exponential(self):
        """Test exponential weighting."""
        distances = np.array([[100.0, 200.0, 300.0]])
        weights = get_weights(distances, method="exponential")

        assert weights.shape == (1, 3)
        np.testing.assert_allclose(weights.sum(axis=1), [1.0], rtol=1e-5)

    def test_tricube(self):
        """Test tricube weighting."""
        distances = np.array([[100.0, 200.0, 300.0]])
        weights = get_weights(distances, method="tricube")

        assert weights.shape == (1, 3)
        np.testing.assert_allclose(weights.sum(axis=1), [1.0], rtol=1e-5)

    def test_invalid_method(self):
        """Test invalid weighting method."""
        distances = np.array([[100.0, 200.0]])
        with pytest.raises(ValueError, match="Unsupported weighting method"):
            get_weights(distances, method="invalid")

    def test_zero_distances(self):
        """Test handling of zero distances."""
        distances = np.array([[0.0, 100.0, 200.0]])
        # Should not raise with epsilon protection
        weights = get_weights(distances, method="inverse_square")
        assert not np.any(np.isnan(weights))
        assert not np.any(np.isinf(weights))

    def test_sigma_factor(self):
        """Test sigma_factor parameter."""
        distances = np.array([[100.0, 200.0, 300.0]])
        weights1 = get_weights(distances, method="gaussian", sigma_factor=0.5)
        weights2 = get_weights(distances, method="gaussian", sigma_factor=1.0)

        # Different sigma factors should give different weights
        assert not np.allclose(weights1, weights2)


class TestGetHealpixResolutionInfo:
    """Tests for get_healpix_resolution_info function."""

    def test_basic_info(self):
        """Test basic resolution info."""
        info = get_healpix_resolution_info(nside=64)

        assert info["nside"] == 64
        assert info["npix"] == 12 * 64**2
        assert isinstance(info["resolution_deg"], float)
        assert isinstance(info["resolution_km"], float)
        assert isinstance(info["area_sr"], float)
        assert isinstance(info["area_km2"], float)

    def test_known_values(self):
        """Test known resolution values."""
        info = get_healpix_resolution_info(nside=1)
        assert info["npix"] == 12

        info = get_healpix_resolution_info(nside=2)
        assert info["npix"] == 48

        info = get_healpix_resolution_info(nside=256)
        assert info["npix"] == 786432

    def test_resolution_scaling(self):
        """Test that higher nside gives higher resolution (smaller pixels)."""
        info_low = get_healpix_resolution_info(nside=32)
        info_high = get_healpix_resolution_info(nside=64)

        assert info_high["resolution_deg"] < info_low["resolution_deg"]
        assert info_high["resolution_km"] < info_low["resolution_km"]
        assert info_high["area_km2"] < info_low["area_km2"]

    def test_total_area_consistency(self):
        """Test that total area is approximately 4*pi steradians."""
        for nside in [16, 32, 64, 128]:
            info = get_healpix_resolution_info(nside)
            total_area = info["npix"] * info["area_sr"]
            np.testing.assert_allclose(total_area, 4 * np.pi, rtol=1e-5)


class TestHpDistance:
    """Tests for hp_distance function (requires healpy)."""

    @pytest.fixture
    def check_healpy(self):
        """Check if healpy is available."""
        try:
            import healpy

            return True
        except ImportError:
            pytest.skip("healpy not installed")
            return False

    def test_same_resolution(self, check_healpy):
        """Test hp_distance with same input/output resolution."""
        from idx_flow.utils import hp_distance

        indices, distances = hp_distance(nside_in=8, nside_out=8, k=4)

        npix = 12 * 8**2
        assert indices.shape == (npix, 4)
        assert distances.shape == (npix, 4)

        # First neighbor should be the pixel itself (distance 0)
        np.testing.assert_allclose(distances[:, 0], 0.0, atol=1e-10)

    def test_downsampling(self, check_healpy):
        """Test hp_distance for downsampling."""
        from idx_flow.utils import hp_distance

        indices, distances = hp_distance(nside_in=16, nside_out=8, k=4)

        npix_out = 12 * 8**2
        npix_in = 12 * 16**2

        assert indices.shape == (npix_out, 4)
        assert distances.shape == (npix_out, 4)
        assert np.all(indices >= 0)
        assert np.all(indices < npix_in)
        assert np.all(distances >= 0)

    def test_upsampling(self, check_healpy):
        """Test hp_distance for upsampling."""
        from idx_flow.utils import hp_distance

        indices, distances = hp_distance(nside_in=8, nside_out=16, k=4)

        npix_out = 12 * 16**2
        npix_in = 12 * 8**2

        assert indices.shape == (npix_out, 4)
        assert distances.shape == (npix_out, 4)
        assert np.all(indices >= 0)
        assert np.all(indices < npix_in)

    def test_invalid_parameters(self, check_healpy):
        """Test hp_distance with invalid parameters."""
        from idx_flow.utils import hp_distance

        with pytest.raises(ValueError):
            hp_distance(nside_in=0, nside_out=8, k=4)

        with pytest.raises(ValueError):
            hp_distance(nside_in=8, nside_out=-1, k=4)

        with pytest.raises(ValueError):
            hp_distance(nside_in=8, nside_out=8, k=0)


class TestComputeConnectionIndices:
    """Tests for compute_connection_indices function."""

    @pytest.fixture
    def check_healpy(self):
        """Check if healpy is available."""
        try:
            import healpy

            return True
        except ImportError:
            pytest.skip("healpy not installed")
            return False

    def test_without_weights(self, check_healpy):
        """Test compute_connection_indices without weights."""
        from idx_flow.utils import compute_connection_indices

        indices, distances = compute_connection_indices(
            nside_in=8, nside_out=4, k=4
        )

        npix_out = 12 * 4**2
        assert indices.shape == (npix_out, 4)
        assert distances.shape == (npix_out, 4)

    def test_with_weights(self, check_healpy):
        """Test compute_connection_indices with weights."""
        from idx_flow.utils import compute_connection_indices

        indices, distances, weights = compute_connection_indices(
            nside_in=8, nside_out=4, k=4, return_weights=True
        )

        npix_out = 12 * 4**2
        assert indices.shape == (npix_out, 4)
        assert distances.shape == (npix_out, 4)
        assert weights.shape == (npix_out, 4)

        # Weights should sum to 1
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(npix_out), rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
