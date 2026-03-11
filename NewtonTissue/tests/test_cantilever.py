"""Integration test: cantilever beam under gravity.

This is the canonical FEM verification test. A beam fixed at one end
(x=0) deforms under its own weight. The analytical tip deflection from
Euler-Bernoulli beam theory provides a reference solution.

The analytical tests are skipped until the Newton solver backend is
connected, but the setup and dummy-solve tests run immediately.
"""

import numpy as np
import pytest

from newton_tissue import TissueModel, TissueSolver, IsotropicMaterial, FixedByBox, Gravity
from tests.conftest import make_cantilever_mesh


# Beam parameters
Lx, Ly, Lz = 1.0, 0.1, 0.1  # meters
E_steel = 200e9  # Pa
nu_steel = 0.3
rho_steel = 7800.0  # kg/m^3
g = 9.81  # m/s^2


def make_cantilever(nx=10, ny=2, nz=2):
    nodes, elements = make_cantilever_mesh(nx, ny, nz, Lx, Ly, Lz)
    steel = IsotropicMaterial(E=E_steel, nu=nu_steel, density=rho_steel)
    bc = FixedByBox([-0.001, -0.001, -0.001], [0.001, Ly + 0.001, Lz + 0.001])
    model = TissueModel(
        nodes=nodes,
        elements=elements,
        material=steel,
        boundary_conditions=[bc],
        loading_conditions=[Gravity()],
    )
    return model


class TestCantileverSetup:
    def test_mesh_dimensions(self):
        model = make_cantilever()
        assert model.num_elements == 10 * 2 * 2 * 5  # 200 tets

    def test_fixed_nodes(self):
        model = make_cantilever()
        fixed = model.fixed_node_indices
        # All nodes at x=0 should be fixed
        x_zero_mask = model.nodes[:, 0] < 0.001
        expected = np.nonzero(x_zero_mask)[0]
        np.testing.assert_array_equal(fixed, expected)

    def test_total_volume(self):
        model = make_cantilever()
        vols = model.compute_element_volumes()
        expected_vol = Lx * Ly * Lz
        np.testing.assert_allclose(vols.sum(), expected_vol, rtol=1e-10)

    def test_total_mass(self):
        model = make_cantilever()
        masses = model.compute_lumped_masses()
        expected_mass = rho_steel * Lx * Ly * Lz
        np.testing.assert_allclose(masses.sum(), expected_mass, rtol=1e-10)

    def test_gravity_forces(self):
        model = make_cantilever()
        forces = model.assemble_forces()
        total_weight = rho_steel * Lx * Ly * Lz * g
        # Total force in -y should equal weight
        np.testing.assert_allclose(-forces[:, 1].sum(), total_weight, rtol=1e-10)
        # No force in x or z
        np.testing.assert_allclose(forces[:, 0].sum(), 0.0, atol=1e-10)
        np.testing.assert_allclose(forces[:, 2].sum(), 0.0, atol=1e-10)


class TestCantileverDummySolve:
    def test_dummy_solve(self):
        model = make_cantilever()
        solver = TissueSolver(model)
        result = solver.solve_static_dummy()
        assert result.converged
        assert result.positions.shape == (model.num_nodes, 3)

    def test_dummy_step(self):
        model = make_cantilever()
        solver = TissueSolver(model)
        result = solver.step_dummy()
        assert result.positions.shape == (model.num_nodes, 3)


@pytest.mark.skip(reason="Requires Newton/Warp GPU backend")
class TestCantileverAnalytical:
    """Compare FEM results to Euler-Bernoulli beam theory.

    Analytical tip deflection for a cantilever under self-weight:
        delta = (rho * g * A * L^4) / (8 * E * I)
    where:
        A = Ly * Lz (cross-section area)
        I = Lz * Ly^3 / 12 (second moment of area, bending about z)
    """

    def test_tip_deflection(self):
        model = make_cantilever(nx=20, ny=4, nz=4)  # finer mesh
        solver = TissueSolver(model, iterations=50)
        result = solver.solve_static(max_frames=5000, tol=1e-8)

        A = Ly * Lz
        I = Lz * Ly**3 / 12.0
        delta_analytical = (rho_steel * g * A * Lx**4) / (8 * E_steel * I)

        # Find tip nodes (x ≈ Lx)
        tip_mask = model.nodes[:, 0] > Lx - 0.001
        tip_displacements = result.displacements[tip_mask]
        tip_deflection = -tip_displacements[:, 1].mean()  # -y displacement

        # Allow 10% error for coarse mesh
        np.testing.assert_allclose(
            tip_deflection, delta_analytical, rtol=0.1,
            err_msg=f"Tip deflection {tip_deflection:.6e} vs analytical {delta_analytical:.6e}",
        )

    def test_patch_test(self):
        """Single element under uniform stress should give exact constant stress.

        Apply uniform traction on one face, fix the opposite face.
        The stress should be constant throughout the element.
        """
        # This verifies the element formulation is correct for constant strain
        nodes = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
        )
        elements = np.array([[0, 1, 2, 3]], dtype=np.int32)
        from newton_tissue import FixedBC, PointForce

        mat = IsotropicMaterial(E=1e6, nu=0.3, density=1000.0)
        model = TissueModel(
            nodes=nodes,
            elements=elements,
            material=mat,
            boundary_conditions=[FixedBC([0])],
            loading_conditions=[PointForce([1], [1000.0, 0.0, 0.0])],
        )
        solver = TissueSolver(model)
        result = solver.solve_static()
        stress = result.von_mises_stress()
        assert stress is not None
        # Stress should be approximately uniform
        np.testing.assert_allclose(
            stress, stress.mean(), rtol=0.01,
            err_msg="Patch test failed: stress is not uniform",
        )
