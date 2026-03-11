"""Simulation result data structures."""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class SimulationResults:
    """Container for FEM simulation output.

    All arrays use SI units (meters, Newtons, m/s, Pascals).

    Attributes:
        positions: Current node positions, shape (N, 3) [m].
        displacements: Node displacements from reference, shape (N, 3) [m].
        velocities: Node velocities, shape (N, 3) [m/s]. None for static solve.
        forces: Nodal reaction forces, shape (N, 3) [N]. None if unavailable.
        time: Simulation time [s].
        converged: True if the solver converged (static) or completed (dynamic).
        num_iterations: Number of solver iterations used.
    """

    positions: np.ndarray
    displacements: np.ndarray
    velocities: np.ndarray | None = None
    forces: np.ndarray | None = None
    time: float = 0.0
    converged: bool = True
    num_iterations: int = 0

    @property
    def num_nodes(self) -> int:
        return self.positions.shape[0]

    def max_displacement(self) -> float:
        """Maximum displacement magnitude across all nodes."""
        return float(np.max(np.linalg.norm(self.displacements, axis=1)))

    def displacement_at(self, node_index: int) -> np.ndarray:
        """Displacement vector at a specific node."""
        return self.displacements[node_index].copy()

    def von_mises_stress(self) -> np.ndarray | None:
        """Per-element von Mises stress [Pa].

        TODO: Requires strain tensor computation from the deformation gradient.
        Will be implemented when the Newton solver backend is connected.
        """
        return None
