"""TissueSolver: wraps Newton's VBD/XPBD solver for soft tissue FEM."""

from __future__ import annotations

import numpy as np
import warp as wp
import newton

from .loading import Gravity, BodyForce, PrescribedDisplacement
from .model import TissueModel
from .results import SimulationResults


def _ensure_positive_winding(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Ensure all tetrahedra have positive volume (right-hand winding).

    Newton requires positive-volume tets. If det([v1-v0, v2-v0, v3-v0]) < 0,
    swap two vertices to fix orientation.

    Args:
        nodes: (N, 3) vertex positions.
        elements: (M, 4) tet connectivity (modified in-place and returned).

    Returns:
        The corrected elements array.
    """
    elements = elements.copy()
    v0 = nodes[elements[:, 0]]
    d1 = nodes[elements[:, 1]] - v0
    d2 = nodes[elements[:, 2]] - v0
    d3 = nodes[elements[:, 3]] - v0
    det = (
        d1[:, 0] * (d2[:, 1] * d3[:, 2] - d2[:, 2] * d3[:, 1])
        - d1[:, 1] * (d2[:, 0] * d3[:, 2] - d2[:, 2] * d3[:, 0])
        + d1[:, 2] * (d2[:, 0] * d3[:, 1] - d2[:, 1] * d3[:, 0])
    )
    inverted = det < 0
    if np.any(inverted):
        elements[inverted, 2], elements[inverted, 3] = (
            elements[inverted, 3].copy(),
            elements[inverted, 2].copy(),
        )
    return elements


class TissueSolver:
    """High-level solver for soft tissue FEM simulation.

    Wraps Newton's VBD or XPBD solver with Neo-Hookean hyperelasticity on
    tetrahedral meshes. Uses the TetMesh API for per-element material support.

    Args:
        model: The tissue model to solve.
        dt: Time step size [s]. Default 1/60 s.
        num_substeps: Number of sub-steps per frame.
        iterations: Solver iterations per sub-step.
        device: Compute device. ``"cpu"`` or ``"cuda:0"``.
        k_damp: Rayleigh damping coefficient (dimensionless). Default 0.01.
        solver_type: ``"vbd"`` or ``"xpbd"``. Default ``"vbd"``.
    """

    def __init__(
        self,
        model: TissueModel,
        dt: float = 1.0 / 60.0,
        num_substeps: int = 10,
        iterations: int = 20,
        device: str = "cpu",
        k_damp: float = 0.01,
        solver_type: str = "vbd",
    ):
        self._model = model
        self._dt = dt
        self._num_substeps = num_substeps
        self._iterations = iterations
        self._device = device
        self._solver_type = solver_type
        self._time = 0.0
        self._initial_positions = model.nodes.copy()

        wp.init()

        # Build the Newton model
        self._newton_model = self._build_newton_model(k_damp)

        # Create solver
        if solver_type == "vbd":
            from newton.solvers import SolverVBD

            self._solver = SolverVBD(
                model=self._newton_model,
                iterations=iterations,
            )
        else:
            from newton.solvers import SolverXPBD

            self._solver = SolverXPBD(
                model=self._newton_model,
                iterations=iterations,
            )

        # Double-buffered state
        self._state_0 = self._newton_model.state()
        self._state_1 = self._newton_model.state()

        # Control and contacts
        self._control = self._newton_model.control()
        self._contacts = self._newton_model.contacts()

    def _build_newton_model(self, k_damp: float) -> newton.Model:
        """Construct the Newton Model from TissueModel specification."""
        mat = self._model.material
        k_mu, k_lambda = mat.to_lame_arrays(
            self._model.num_elements, self._model.elements
        )
        density = mat.get_density()
        if isinstance(density, np.ndarray):
            density = float(density.mean())

        # Ensure positive-volume winding for Newton
        elements = _ensure_positive_winding(
            self._model.nodes.astype(np.float32),
            self._model.elements,
        )

        # Build TetMesh with per-element material arrays
        tet_mesh = newton.TetMesh(
            vertices=self._model.nodes.astype(np.float32),
            tet_indices=elements.flatten(),
            k_mu=k_mu.astype(np.float32),
            k_lambda=k_lambda.astype(np.float32),
            k_damp=np.full(self._model.num_elements, k_damp, dtype=np.float32),
            density=density,
        )

        builder = newton.ModelBuilder()
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            mesh=tet_mesh,
        )

        # Fix boundary nodes by setting mass to zero (kinematic)
        for i in self._model.fixed_node_indices:
            builder.particle_mass[int(i)] = 0.0

        # VBD requires graph coloring
        if self._solver_type == "vbd":
            builder.color()

        newton_model = builder.finalize(device=self._device)

        # Set gravity from loading conditions
        for lc in self._model.loading_conditions:
            if isinstance(lc, (Gravity, BodyForce)):
                newton_model.gravity = lc.acceleration.astype(np.float32)
                break

        return newton_model

    def _apply_non_gravity_forces(self) -> None:
        """Apply external point forces to the current state.

        Gravity/body forces are handled natively by Newton via model.gravity,
        so only non-gravity forces (PointForce, etc.) are applied here.
        """
        forces = np.zeros((self._model.num_nodes, 3), dtype=np.float32)
        masses = self._model.compute_lumped_masses().astype(np.float32)

        has_non_gravity = False
        for lc in self._model.loading_conditions:
            if isinstance(lc, (Gravity, BodyForce, PrescribedDisplacement)):
                continue
            lc.apply(forces, self._model.nodes.astype(np.float32), masses)
            has_non_gravity = True

        if has_non_gravity:
            forces_wp = wp.array(forces, dtype=wp.vec3, device=self._device)
            wp.copy(self._state_0.particle_f, forces_wp)

    def _apply_prescribed_displacements(self) -> None:
        """Apply prescribed displacement boundary conditions."""
        for lc in self._model.loading_conditions:
            if isinstance(lc, PrescribedDisplacement):
                disp = lc.get_displacement(self._time).astype(np.float32)
                positions_np = self._state_0.particle_q.numpy()
                for idx in lc.node_indices:
                    positions_np[idx] = (
                        self._initial_positions[idx].astype(np.float32) + disp
                    )
                self._state_0.particle_q.assign(
                    wp.array(positions_np, dtype=wp.vec3, device=self._device)
                )

    @property
    def model(self) -> TissueModel:
        return self._model

    @property
    def time(self) -> float:
        return self._time

    def reset(self) -> None:
        """Reset the solver to the initial (undeformed) configuration."""
        self._state_0 = self._newton_model.state()
        self._state_1 = self._newton_model.state()
        self._time = 0.0

    def step(self) -> SimulationResults:
        """Advance one frame (num_substeps sub-steps).

        Returns:
            SimulationResults with current positions, displacements,
            velocities, and forces.
        """
        for _ in range(self._num_substeps):
            self._state_0.clear_forces()
            self._apply_non_gravity_forces()
            self._apply_prescribed_displacements()

            self._solver.step(
                self._state_0,
                self._state_1,
                self._control,
                self._contacts,
                self._dt,
            )

            self._state_0, self._state_1 = self._state_1, self._state_0

        self._time += self._dt * self._num_substeps
        return self.get_current_state()

    def step_dummy(self) -> SimulationResults:
        """Advance one frame with dummy (zero-deformation) results.

        Useful for testing the API without GPU dependencies.
        """
        self._time += self._dt * self._num_substeps
        return SimulationResults(
            positions=self._initial_positions.copy(),
            displacements=np.zeros_like(self._initial_positions),
            velocities=np.zeros_like(self._initial_positions),
            forces=None,
            time=self._time,
            converged=True,
            num_iterations=0,
        )

    def solve_static(
        self, max_frames: int = 1000, tol: float = 1e-6
    ) -> SimulationResults:
        """Solve for quasi-static equilibrium.

        Runs dynamic steps with damping until the maximum velocity magnitude
        drops below the tolerance, indicating the system has settled.

        Args:
            max_frames: Maximum number of frames before giving up.
            tol: Convergence tolerance on max velocity magnitude [m/s].

        Returns:
            SimulationResults. ``converged`` is True if tolerance was reached.
        """
        converged = False
        frame = 0
        for frame in range(max_frames):
            self.step()

            velocities_np = self._state_0.particle_qd.numpy()
            max_vel = float(np.max(np.linalg.norm(velocities_np, axis=1)))

            if max_vel < tol:
                converged = True
                break

        result = self.get_current_state()
        result.converged = converged
        result.num_iterations = (frame + 1) * self._num_substeps
        return result

    def solve_static_dummy(
        self, max_frames: int = 1000, tol: float = 1e-6
    ) -> SimulationResults:
        """Return dummy static solution for API testing."""
        self._time = max_frames * self._dt * self._num_substeps
        return SimulationResults(
            positions=self._initial_positions.copy(),
            displacements=np.zeros_like(self._initial_positions),
            velocities=None,
            forces=self._model.assemble_forces(),
            time=self._time,
            converged=True,
            num_iterations=0,
        )

    def get_current_state(self) -> SimulationResults:
        """Extract the current simulation state as a SimulationResults object."""
        positions_np = self._state_0.particle_q.numpy().astype(np.float64)
        velocities_np = self._state_0.particle_qd.numpy().astype(np.float64)

        return SimulationResults(
            positions=positions_np,
            displacements=positions_np - self._initial_positions,
            velocities=velocities_np,
            forces=None,
            time=self._time,
            converged=True,
            num_iterations=0,
        )
