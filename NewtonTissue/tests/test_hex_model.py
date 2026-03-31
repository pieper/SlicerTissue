"""Tests for HexTissueModel — geometry, materials, BCs (no GPU required)."""

import numpy as np
import pytest

from newton_tissue import (
    HexTissueModel,
    IsotropicMaterial,
    HeterogeneousMaterial,
    NodalMaterial,
    FixedByBox,
    Gravity,
    PROSTATE_PERIPHERAL,
)


class TestHexTissueModelBasic:
    def test_construction(self):
        m = HexTissueModel(res=(4, 4, 4))
        assert m.res == (4, 4, 4)
        assert m.num_elements == 64

    def test_num_elements(self):
        m = HexTissueModel(res=(2, 3, 5))
        assert m.num_elements == 30

    def test_element_size(self):
        m = HexTissueModel(
            res=(4, 8, 2),
            bounds_lo=(0, 0, 0),
            bounds_hi=(0.04, 0.08, 0.02),
        )
        dx, dy, dz = m.element_size
        np.testing.assert_allclose(dx, 0.01)
        np.testing.assert_allclose(dy, 0.01)
        np.testing.assert_allclose(dz, 0.01)

    def test_default_material(self):
        m = HexTissueModel(res=(2, 2, 2))
        assert isinstance(m.material, IsotropicMaterial)

    def test_repr(self):
        m = HexTissueModel(res=(2, 3, 4), material=PROSTATE_PERIPHERAL)
        r = repr(m)
        assert "HexTissueModel" in r
        assert "elements=24" in r

    def test_validate_ok(self):
        m = HexTissueModel(res=(2, 2, 2))
        assert m.validate() == []

    def test_validate_bad_res(self):
        m = HexTissueModel.__new__(HexTissueModel)
        m._res = (0, 2, 2)
        m._bounds_lo = (0, 0, 0)
        m._bounds_hi = (1, 1, 1)
        m._material = IsotropicMaterial(E=1e3, nu=0.3)
        m._boundary_conditions = []
        m._loading_conditions = []
        warns = m.validate()
        assert any("resolution" in w.lower() for w in warns)

    def test_validate_wrong_het_material(self):
        het = HeterogeneousMaterial(
            k_mu=np.ones(10, dtype=np.float32),
            k_lambda=np.ones(10, dtype=np.float32),
        )
        m = HexTissueModel(res=(2, 2, 2), material=het)  # 8 elements, not 10
        warns = m.validate()
        assert any("HeterogeneousMaterial" in w for w in warns)


class TestHexTissueModelMaterials:
    def test_isotropic_lame_arrays(self):
        mat = IsotropicMaterial(E=20e3, nu=0.45)
        m = HexTissueModel(res=(2, 2, 2), material=mat)
        k_mu, k_lam = m.build_element_lame_arrays()
        assert k_mu.shape == (8,)
        np.testing.assert_allclose(k_mu, mat.mu, rtol=1e-5)

    def test_heterogeneous_lame_arrays(self):
        n = 8
        k_mu = np.linspace(1000, 8000, n, dtype=np.float32)
        k_lam = np.linspace(10000, 80000, n, dtype=np.float32)
        mat = HeterogeneousMaterial(k_mu=k_mu, k_lambda=k_lam)
        m = HexTissueModel(res=(2, 2, 2), material=mat)
        out_mu, out_lam = m.build_element_lame_arrays()
        np.testing.assert_array_equal(out_mu, k_mu)
        np.testing.assert_array_equal(out_lam, k_lam)

    def test_nodal_lame_arrays_averaging(self):
        # 2x2x2 grid has 3x3x3 = 27 corner nodes
        n_nodes = 27
        k_mu = np.ones(n_nodes, dtype=np.float32) * 5000.0
        k_lam = np.ones(n_nodes, dtype=np.float32) * 50000.0
        mat = NodalMaterial(k_mu=k_mu, k_lambda=k_lam)
        m = HexTissueModel(res=(2, 2, 2), material=mat)
        out_mu, out_lam = m.build_element_lame_arrays()
        # Uniform field → all elements get the same value
        assert out_mu.shape == (8,)
        np.testing.assert_allclose(out_mu, 5000.0, rtol=1e-5)
        np.testing.assert_allclose(out_lam, 50000.0, rtol=1e-5)


class TestLayeredBlock:
    def test_layered_block_construction(self):
        liver = IsotropicMaterial(E=10e3, nu=0.45, density=1060.0)
        muscle = IsotropicMaterial(E=60e3, nu=0.40, density=1050.0)
        m = HexTissueModel.layered_block(
            res=(4, 8, 4),
            size=0.10,
            layers=[
                (0.00, 0.05, liver),
                (0.05, 0.10, muscle),
            ],
        )
        assert m.num_elements == 128
        assert isinstance(m.material, HeterogeneousMaterial)

    def test_layered_block_has_fixed_bottom_bc(self):
        mat = IsotropicMaterial(E=10e3, nu=0.45)
        m = HexTissueModel.layered_block(
            res=(2, 4, 2), size=0.10,
            layers=[(0.0, 0.10, mat)],
            fixed_bottom=True,
        )
        assert len(m.boundary_conditions) == 1
        assert isinstance(m.boundary_conditions[0], FixedByBox)

    def test_layered_block_no_bc(self):
        mat = IsotropicMaterial(E=10e3, nu=0.45)
        m = HexTissueModel.layered_block(
            res=(2, 2, 2), size=0.1,
            layers=[(0.0, 0.1, mat)],
            fixed_bottom=False,
        )
        assert len(m.boundary_conditions) == 0

    def test_layered_block_material_classification(self):
        """Bottom half liver, top half muscle — check per-element mu values."""
        liver = IsotropicMaterial(E=10e3, nu=0.45)
        muscle = IsotropicMaterial(E=60e3, nu=0.40)
        # 1x4x1 grid along y: elements j=0,1 in liver, j=2,3 in muscle
        m = HexTissueModel.layered_block(
            res=(1, 4, 1),
            size=0.10,
            layers=[(0.00, 0.05, liver), (0.05, 0.10, muscle)],
            fixed_bottom=False,
        )
        k_mu, _ = m.build_element_lame_arrays()
        # Elements 0,1: liver mu
        np.testing.assert_allclose(k_mu[0], liver.mu, rtol=1e-5)
        np.testing.assert_allclose(k_mu[1], liver.mu, rtol=1e-5)
        # Elements 2,3: muscle mu
        np.testing.assert_allclose(k_mu[2], muscle.mu, rtol=1e-5)
        np.testing.assert_allclose(k_mu[3], muscle.mu, rtol=1e-5)
