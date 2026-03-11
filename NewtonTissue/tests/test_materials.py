"""Tests for material property definitions."""

import numpy as np
import pytest

from newton_tissue import IsotropicMaterial, HeterogeneousMaterial, PROSTATE_PERIPHERAL


class TestIsotropicMaterial:
    def test_from_young_poisson(self):
        mat = IsotropicMaterial(E=20e3, nu=0.48)
        # mu = E / (2*(1+nu)) = 20000 / (2*1.48) = 6756.76...
        np.testing.assert_allclose(mat.mu, 20e3 / (2 * 1.48), rtol=1e-10)
        # lam = E*nu / ((1+nu)*(1-2*nu)) = 20000*0.48 / (1.48*0.04)
        np.testing.assert_allclose(mat.lam, 20e3 * 0.48 / (1.48 * 0.04), rtol=1e-10)

    def test_from_lame(self):
        mat = IsotropicMaterial(mu=6757.0, lam=162162.0, density=1040.0)
        assert mat.mu == 6757.0
        assert mat.lam == 162162.0

    def test_round_trip(self):
        """Create from (E, nu), convert to Lamé, back to (E, nu)."""
        mat = IsotropicMaterial(E=100e3, nu=0.3, density=2000.0)
        np.testing.assert_allclose(mat.E, 100e3, rtol=1e-10)
        np.testing.assert_allclose(mat.nu, 0.3, rtol=1e-10)

    def test_nearly_incompressible(self):
        mat = IsotropicMaterial(E=20e3, nu=0.499)
        assert mat.lam > mat.mu * 100  # lambda >> mu for nearly incompressible

    def test_invalid_poisson_high(self):
        with pytest.raises(ValueError):
            IsotropicMaterial(E=1e3, nu=0.5)

    def test_invalid_poisson_low(self):
        with pytest.raises(ValueError):
            IsotropicMaterial(E=1e3, nu=-1.1)

    def test_invalid_young(self):
        with pytest.raises(ValueError):
            IsotropicMaterial(E=-1e3, nu=0.3)

    def test_must_specify_one_pair(self):
        with pytest.raises(ValueError):
            IsotropicMaterial(E=1e3, nu=0.3, mu=500, lam=1000)
        with pytest.raises(ValueError):
            IsotropicMaterial()

    def test_to_lame_arrays(self):
        mat = IsotropicMaterial(E=20e3, nu=0.3)
        k_mu, k_lam = mat.to_lame_arrays(100)
        assert k_mu.shape == (100,)
        assert k_lam.shape == (100,)
        np.testing.assert_allclose(k_mu, mat.mu, rtol=1e-6)
        np.testing.assert_allclose(k_lam, mat.lam, rtol=1e-6)


class TestHeterogeneousMaterial:
    def test_per_element_arrays(self):
        k_mu = np.array([6000, 7000, 8000], dtype=np.float32)
        k_lam = np.array([100000, 120000, 140000], dtype=np.float32)
        mat = HeterogeneousMaterial(k_mu=k_mu, k_lambda=k_lam, density=1040.0)
        assert mat.num_elements == 3
        mu_out, lam_out = mat.to_lame_arrays(3)
        np.testing.assert_array_equal(mu_out, k_mu)
        np.testing.assert_array_equal(lam_out, k_lam)

    def test_wrong_element_count(self):
        k_mu = np.array([6000, 7000], dtype=np.float32)
        k_lam = np.array([100000, 120000], dtype=np.float32)
        mat = HeterogeneousMaterial(k_mu=k_mu, k_lambda=k_lam)
        with pytest.raises(ValueError, match="Expected 2 elements"):
            mat.to_lame_arrays(5)

    def test_mismatched_shapes(self):
        with pytest.raises(ValueError):
            HeterogeneousMaterial(
                k_mu=np.array([1, 2, 3], dtype=np.float32),
                k_lambda=np.array([1, 2], dtype=np.float32),
            )


class TestPresets:
    def test_prostate_peripheral(self):
        mat = PROSTATE_PERIPHERAL
        # E=20kPa, nu=0.48
        np.testing.assert_allclose(mat.E, 20e3, rtol=1e-10)
        np.testing.assert_allclose(mat.nu, 0.48, rtol=1e-10)
        assert mat.density == 1040.0
        # mu ≈ 6756.76 Pa
        np.testing.assert_allclose(mat.mu, 20e3 / (2 * 1.48), rtol=1e-6)
