"""Tests for TissueModel: mesh assembly, volumes, masses."""

import numpy as np
import pytest

from newton_tissue import TissueModel, IsotropicMaterial, FixedBC
from tests.conftest import make_single_tet, make_cantilever_mesh


class TestSingleTet:
    def test_shape(self, single_tet_model):
        assert single_tet_model.num_nodes == 4
        assert single_tet_model.num_elements == 1

    def test_volume(self, single_tet_model):
        vols = single_tet_model.compute_element_volumes()
        # Volume of tet with vertices at origin, (1,0,0), (0,1,0), (0,0,1)
        # = |det([[1,0,0],[0,1,0],[0,0,1]])| / 6 = 1/6
        np.testing.assert_allclose(vols[0], 1.0 / 6.0, rtol=1e-12)


class TestCubeMesh:
    def test_total_volume(self):
        """A 1x1x1 cube meshed into tets should have total volume 1.0."""
        nodes, elements = make_cantilever_mesh(nx=1, ny=1, nz=1, Lx=1.0, Ly=1.0, Lz=1.0)
        mat = IsotropicMaterial(E=1e3, nu=0.3)
        model = TissueModel(nodes=nodes, elements=elements, material=mat)
        vols = model.compute_element_volumes()
        np.testing.assert_allclose(vols.sum(), 1.0, rtol=1e-10)


class TestLumpedMasses:
    def test_mass_conservation(self, single_tet_model):
        """Total lumped mass should equal density * total volume."""
        masses = single_tet_model.compute_lumped_masses()
        density = single_tet_model.material.get_density()
        total_vol = single_tet_model.compute_element_volumes().sum()
        np.testing.assert_allclose(masses.sum(), density * total_vol, rtol=1e-12)

    def test_cantilever_mass(self, cantilever_model):
        masses = cantilever_model.compute_lumped_masses()
        density = cantilever_model.material.get_density()
        total_vol = cantilever_model.compute_element_volumes().sum()
        np.testing.assert_allclose(masses.sum(), density * total_vol, rtol=1e-10)


class TestUnitScale:
    def test_mm_to_meters(self):
        """unit_scale=0.001 should convert mm coordinates to meters."""
        nodes_mm = np.array(
            [[0, 0, 0], [1000, 0, 0], [0, 1000, 0], [0, 0, 1000]],
            dtype=np.float64,
        )
        elements = np.array([[0, 1, 2, 3]], dtype=np.int32)
        mat = IsotropicMaterial(E=1e3, nu=0.3)
        model = TissueModel(nodes=nodes_mm, elements=elements, material=mat, unit_scale=0.001)
        # After scaling, nodes should be in meters
        np.testing.assert_allclose(model.nodes[1, 0], 1.0, rtol=1e-12)


class TestValidation:
    def test_degenerate_tet(self):
        """Coplanar nodes should trigger a degenerate tet warning."""
        nodes = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0.5, 0]],
            dtype=np.float64,
        )
        elements = np.array([[0, 1, 2, 3]], dtype=np.int32)
        mat = IsotropicMaterial(E=1e3, nu=0.3)
        model = TissueModel(nodes=nodes, elements=elements, material=mat)
        warnings = model.validate()
        assert any("degenerate" in w.lower() for w in warnings)

    def test_poisson_locking_warning(self, single_tet_model):
        """Near-incompressible material should warn about volumetric locking."""
        warnings = single_tet_model.validate()
        assert any("locking" in w.lower() for w in warnings)

    def test_invalid_shape(self):
        with pytest.raises(ValueError, match="shape"):
            TissueModel(
                nodes=np.zeros((4, 2)),
                elements=np.array([[0, 1, 2, 3]]),
                material=IsotropicMaterial(E=1e3, nu=0.3),
            )
