"""Loading condition definitions for FEM simulation.

Loading conditions define external forces applied to the model:
point forces, body forces (gravity), and prescribed displacements.
"""

from __future__ import annotations

import abc
from typing import Callable

import numpy as np


class LoadingCondition(abc.ABC):
    """Abstract base class for loading conditions."""

    @abc.abstractmethod
    def apply(
        self,
        forces: np.ndarray,
        positions: np.ndarray,
        masses: np.ndarray,
    ) -> np.ndarray:
        """Apply this loading condition to the force array.

        Modifies forces in-place and returns it.

        Args:
            forces: Nodal force array, shape (N, 3) [N]. Modified in-place.
            positions: Current node positions, shape (N, 3) [m].
            masses: Lumped nodal masses, shape (N,) [kg].

        Returns:
            The modified forces array.
        """


class PointForce(LoadingCondition):
    """Concentrated force applied to specific nodes.

    Args:
        node_indices: Indices of nodes to load.
        force_vector: Force vector [N], shape (3,). Applied to each listed node.
    """

    def __init__(self, node_indices: list[int] | np.ndarray, force_vector):
        self.node_indices = np.asarray(node_indices, dtype=np.intp)
        self.force_vector = np.asarray(force_vector, dtype=np.float64)

    def apply(self, forces, positions, masses):
        forces[self.node_indices] += self.force_vector
        return forces


class BodyForce(LoadingCondition):
    """Force per unit mass applied to all nodes (e.g., acceleration field).

    The force on node i is ``masses[i] * acceleration``.

    Args:
        acceleration: Acceleration vector [m/s^2], shape (3,).
    """

    def __init__(self, acceleration):
        self.acceleration = np.asarray(acceleration, dtype=np.float64)

    def apply(self, forces, positions, masses):
        forces += masses[:, np.newaxis] * self.acceleration[np.newaxis, :]
        return forces


class Gravity(BodyForce):
    """Gravitational body force.

    Args:
        g: Gravity vector [m/s^2]. Default: [0, -9.81, 0] (y-down).
    """

    def __init__(self, g=None):
        if g is None:
            g = np.array([0.0, -9.81, 0.0])
        super().__init__(acceleration=g)


class PrescribedDisplacement(LoadingCondition):
    """Kinematic constraint: prescribed displacement for specific nodes.

    Args:
        node_indices: Indices of nodes with prescribed displacement.
        displacement: Either a fixed (3,) displacement vector [m],
            or a callable ``fn(time: float) -> np.ndarray`` returning (3,).
    """

    def __init__(
        self,
        node_indices: list[int] | np.ndarray,
        displacement: np.ndarray | Callable[[float], np.ndarray],
    ):
        self.node_indices = np.asarray(node_indices, dtype=np.intp)
        if callable(displacement):
            self._fn = displacement
        else:
            disp = np.asarray(displacement, dtype=np.float64)
            self._fn = lambda t: disp  # noqa: E731

    def get_displacement(self, time: float = 0.0) -> np.ndarray:
        """Return the prescribed displacement vector at the given time."""
        return self._fn(time)

    def apply(self, forces, positions, masses):
        # Prescribed displacements are handled by the solver, not via forces.
        # This is a no-op for the force assembly; the solver reads these
        # conditions directly when updating node positions.
        return forces
