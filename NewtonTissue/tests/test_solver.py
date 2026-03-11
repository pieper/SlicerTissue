"""Tests for the TissueSolver interface."""

import numpy as np
import pytest

from newton_tissue import TissueSolver, SimulationResults


class TestSolverCreation:
    def test_create(self, single_tet_model):
        solver = TissueSolver(single_tet_model)
        assert solver.model is single_tet_model
        assert solver.time == 0.0

    def test_create_with_params(self, single_tet_model):
        solver = TissueSolver(
            single_tet_model, dt=0.01, num_substeps=5, iterations=20
        )
        assert solver._dt == 0.01
        assert solver._num_substeps == 5
        assert solver._iterations == 20


class TestStepDummy:
    def test_returns_results(self, single_tet_model):
        solver = TissueSolver(single_tet_model)
        result = solver.step_dummy()
        assert isinstance(result, SimulationResults)
        assert result.positions.shape == (4, 3)
        assert result.displacements.shape == (4, 3)
        assert result.velocities.shape == (4, 3)

    def test_time_advances(self, single_tet_model):
        solver = TissueSolver(single_tet_model, dt=0.01, num_substeps=5)
        solver.step_dummy()
        np.testing.assert_allclose(solver.time, 0.05, rtol=1e-12)
        solver.step_dummy()
        np.testing.assert_allclose(solver.time, 0.10, rtol=1e-12)


class TestStepRuns:
    def test_step_runs(self, single_tet_model):
        solver = TissueSolver(single_tet_model)
        result = solver.step()
        assert isinstance(result, SimulationResults)
        assert result.positions.shape == (4, 3)

    def test_solve_static_runs(self, single_tet_model):
        solver = TissueSolver(single_tet_model)
        result = solver.solve_static(max_frames=5)
        assert isinstance(result, SimulationResults)


class TestSolveStaticDummy:
    def test_returns_results(self, single_tet_model):
        solver = TissueSolver(single_tet_model)
        result = solver.solve_static_dummy()
        assert result.converged is True
        assert result.positions.shape == (4, 3)
        assert result.velocities is None  # static solve has no velocities

    def test_has_forces(self, cantilever_model):
        solver = TissueSolver(cantilever_model)
        result = solver.solve_static_dummy()
        assert result.forces is not None
        assert result.forces.shape == (cantilever_model.num_nodes, 3)


class TestReset:
    def test_reset_restores_initial(self, single_tet_model):
        solver = TissueSolver(single_tet_model)
        solver.step_dummy()
        solver.step_dummy()
        assert solver.time > 0

        solver.reset()
        assert solver.time == 0.0
        state = solver.get_current_state()
        np.testing.assert_array_equal(state.positions, single_tet_model.nodes)


class TestGetCurrentState:
    def test_initial_state(self, single_tet_model):
        solver = TissueSolver(single_tet_model)
        state = solver.get_current_state()
        np.testing.assert_array_equal(state.positions, single_tet_model.nodes)
        np.testing.assert_allclose(state.displacements, 0.0)
        assert state.max_displacement() == 0.0
