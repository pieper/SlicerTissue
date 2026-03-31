"""Tests for material property definitions."""

import numpy as np
import pytest

from newton_tissue import (
    IsotropicMaterial, HeterogeneousMaterial, NodalMaterial,
    AnisotropicMaterial, PROSTATE_PERIPHERAL,
)


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


class TestNodalMaterial:
    # Two tets sharing node 0: tet0=[0,1,2,3], tet1=[0,2,3,4]
    # Nodes 0..4 with linearly increasing mu/lam
    _k_mu = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.float32)
    _k_lam = np.array([10000, 20000, 30000, 40000, 50000], dtype=np.float32)
    _elements = np.array([[0, 1, 2, 3], [0, 2, 3, 4]], dtype=np.int32)

    def _make_mat(self):
        return NodalMaterial(k_mu=self._k_mu, k_lambda=self._k_lam)

    def test_basic_construction(self):
        mat = self._make_mat()
        assert mat.num_nodes == 5
        np.testing.assert_array_equal(mat.get_nodal_mu(), self._k_mu)
        np.testing.assert_array_equal(mat.get_nodal_lambda(), self._k_lam)

    def test_get_mu_get_lambda(self):
        mat = self._make_mat()
        np.testing.assert_array_equal(mat.get_mu(), self._k_mu)
        np.testing.assert_array_equal(mat.get_lambda(), self._k_lam)

    def test_to_lame_arrays_averages_nodes(self):
        mat = self._make_mat()
        k_mu_e, k_lam_e = mat.to_lame_arrays(2, self._elements)
        # tet0 nodes [0,1,2,3]: mu mean = (1000+2000+3000+4000)/4 = 2500
        assert k_mu_e.shape == (2,)
        np.testing.assert_allclose(k_mu_e[0], 2500.0, rtol=1e-6)
        # tet1 nodes [0,2,3,4]: mu mean = (1000+3000+4000+5000)/4 = 3250
        np.testing.assert_allclose(k_mu_e[1], 3250.0, rtol=1e-6)
        # lam tet0: (10000+20000+30000+40000)/4 = 25000
        np.testing.assert_allclose(k_lam_e[0], 25000.0, rtol=1e-6)

    def test_to_lame_arrays_requires_elements(self):
        mat = self._make_mat()
        with pytest.raises(ValueError, match="requires the 'elements' array"):
            mat.to_lame_arrays(2, elements=None)

    def test_to_lame_arrays_wrong_count(self):
        mat = self._make_mat()
        with pytest.raises(ValueError, match="Expected 5"):
            mat.to_lame_arrays(5, self._elements)

    def test_mismatched_shapes(self):
        with pytest.raises(ValueError):
            NodalMaterial(
                k_mu=np.array([1000, 2000], dtype=np.float32),
                k_lambda=np.array([10000], dtype=np.float32),
            )

    def test_nonpositive_mu(self):
        with pytest.raises(ValueError, match="positive"):
            NodalMaterial(
                k_mu=np.array([1000, 0], dtype=np.float32),
                k_lambda=np.array([10000, 20000], dtype=np.float32),
            )

    def test_density_scalar(self):
        mat = NodalMaterial(k_mu=self._k_mu, k_lambda=self._k_lam, density=1040.0)
        assert mat.get_density() == 1040.0

    def test_density_per_node(self):
        density = np.array([1000, 1010, 1020, 1030, 1040], dtype=np.float32)
        mat = NodalMaterial(k_mu=self._k_mu, k_lambda=self._k_lam, density=density)
        np.testing.assert_array_equal(mat.get_density(), density)

    def test_repr(self):
        mat = self._make_mat()
        r = repr(mat)
        assert "NodalMaterial" in r
        assert "num_nodes=5" in r

    def test_isotropic_to_lame_arrays_ignores_elements(self):
        """IsotropicMaterial should still work without elements argument."""
        mat = IsotropicMaterial(E=20e3, nu=0.3)
        k_mu, k_lam = mat.to_lame_arrays(10)
        assert k_mu.shape == (10,)

    def test_heterogeneous_to_lame_arrays_ignores_elements(self):
        """HeterogeneousMaterial should still work without elements argument."""
        k_mu = np.array([1000, 2000], dtype=np.float32)
        k_lam = np.array([10000, 20000], dtype=np.float32)
        mat = HeterogeneousMaterial(k_mu=k_mu, k_lambda=k_lam)
        mu_out, lam_out = mat.to_lame_arrays(2)
        np.testing.assert_array_equal(mu_out, k_mu)


class TestAnisotropicMaterial:
    _n = 4
    _k_mu    = np.array([1000, 2000, 3000, 4000], dtype=np.float32)
    _k_lam   = np.array([10000, 20000, 30000, 40000], dtype=np.float32)
    _k1      = np.array([500, 600, 700, 800], dtype=np.float32)
    _k2      = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    _fibers  = np.array([[1,0,0],[0,1,0],[0,0,1],[0.707,0,0.707]], dtype=np.float32)

    def _make(self):
        return AnisotropicMaterial(
            k_mu=self._k_mu, k_lambda=self._k_lam,
            k1=self._k1, k2=self._k2, fiber_dirs=self._fibers)

    def test_construction(self):
        mat = self._make()
        assert mat.num_elements == self._n

    def test_to_lame_arrays_returns_isotropic(self):
        mat = self._make()
        k_mu, k_lam = mat.to_lame_arrays(self._n)
        np.testing.assert_array_equal(k_mu, self._k_mu)
        np.testing.assert_array_equal(k_lam, self._k_lam)

    def test_get_k1_k2(self):
        mat = self._make()
        np.testing.assert_array_equal(mat.get_k1(), self._k1)
        np.testing.assert_array_equal(mat.get_k2(), self._k2)

    def test_get_fiber_dirs_unit(self):
        mat = self._make()
        dirs = mat.get_fiber_dirs()
        assert dirs.shape == (self._n, 3)
        norms = np.linalg.norm(dirs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_fiber_normalisation_warns(self):
        non_unit = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4], [1, 1, 0]], dtype=np.float32)
        with pytest.warns(UserWarning, match="unit vectors"):
            mat = AnisotropicMaterial(
                k_mu=self._k_mu, k_lambda=self._k_lam,
                k1=self._k1, k2=self._k2, fiber_dirs=non_unit)
        norms = np.linalg.norm(mat.get_fiber_dirs(), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_uniform_classmethod(self):
        mat = AnisotropicMaterial.uniform(
            num_elements=5, E=20e3, nu=0.45, k1=1000.0, k2=2.0,
            fiber_dir=[1.0, 0.0, 0.0])
        assert mat.num_elements == 5
        assert mat.get_k1().shape == (5,)
        np.testing.assert_allclose(mat.get_k1(), 1000.0)
        np.testing.assert_allclose(mat.get_k2(), 2.0)
        dirs = mat.get_fiber_dirs()
        np.testing.assert_allclose(dirs, np.tile([1,0,0], (5,1)), atol=1e-6)

    def test_mismatched_shapes(self):
        with pytest.raises(ValueError):
            AnisotropicMaterial(
                k_mu=self._k_mu, k_lambda=self._k_lam[:-1],
                k1=self._k1, k2=self._k2, fiber_dirs=self._fibers)

    def test_k2_positive_required(self):
        bad_k2 = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)  # zero
        with pytest.raises(ValueError, match="k2"):
            AnisotropicMaterial(
                k_mu=self._k_mu, k_lambda=self._k_lam,
                k1=self._k1, k2=bad_k2, fiber_dirs=self._fibers)

    def test_k1_nonneg_required(self):
        bad_k1 = np.array([500, -100, 700, 800], dtype=np.float32)
        with pytest.raises(ValueError, match="k1"):
            AnisotropicMaterial(
                k_mu=self._k_mu, k_lambda=self._k_lam,
                k1=bad_k1, k2=self._k2, fiber_dirs=self._fibers)

    def test_wrong_element_count(self):
        mat = self._make()
        with pytest.raises(ValueError, match="Expected"):
            mat.to_lame_arrays(99)

    def test_zero_k1_gives_no_fiber_term(self):
        """When k1=0, aniso material is equivalent to isotropic at this API level."""
        zero_k1 = np.zeros(self._n, dtype=np.float32)
        mat = AnisotropicMaterial(
            k_mu=self._k_mu, k_lambda=self._k_lam,
            k1=zero_k1, k2=self._k2, fiber_dirs=self._fibers)
        np.testing.assert_array_equal(mat.get_k1(), 0.0)
        # to_lame_arrays still returns isotropic base
        k_mu, k_lam = mat.to_lame_arrays(self._n)
        np.testing.assert_array_equal(k_mu, self._k_mu)

    def test_density_default(self):
        mat = self._make()
        assert mat.get_density() == 1000.0

    def test_density_custom(self):
        mat = AnisotropicMaterial(
            k_mu=self._k_mu, k_lambda=self._k_lam,
            k1=self._k1, k2=self._k2, fiber_dirs=self._fibers,
            density=1060.0)
        assert mat.get_density() == 1060.0

    def test_repr(self):
        r = repr(self._make())
        assert "AnisotropicMaterial" in r
        assert "num_elements=4" in r


class TestPresets:
    def test_prostate_peripheral(self):
        mat = PROSTATE_PERIPHERAL
        # E=20kPa, nu=0.48
        np.testing.assert_allclose(mat.E, 20e3, rtol=1e-10)
        np.testing.assert_allclose(mat.nu, 0.48, rtol=1e-10)
        assert mat.density == 1040.0
        # mu ≈ 6756.76 Pa
        np.testing.assert_allclose(mat.mu, 20e3 / (2 * 1.48), rtol=1e-6)
