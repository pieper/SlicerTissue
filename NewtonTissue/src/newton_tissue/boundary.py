"""Boundary condition definitions for FEM simulation.

Boundary conditions identify which nodes are fixed (Dirichlet BCs).
Newton implements this by setting particle mass to zero, making nodes kinematic.
"""

from __future__ import annotations

import abc

import numpy as np


class BoundaryCondition(abc.ABC):
    """Abstract base class for boundary conditions."""

    @abc.abstractmethod
    def get_fixed_node_indices(self, nodes: np.ndarray) -> np.ndarray:
        """Return indices of nodes that should be fixed.

        Args:
            nodes: Node positions array, shape (N, 3).

        Returns:
            Integer array of fixed node indices.
        """


class FixedBC(BoundaryCondition):
    """Fix specific nodes by index.

    Args:
        node_indices: Indices of nodes to fix. List, tuple, or ndarray.
    """

    def __init__(self, node_indices: list[int] | np.ndarray):
        self._indices = np.asarray(node_indices, dtype=np.intp)

    def get_fixed_node_indices(self, nodes: np.ndarray) -> np.ndarray:
        return self._indices


class FixedByPredicate(BoundaryCondition):
    """Fix nodes matching a predicate function.

    Args:
        predicate: Function taking a (3,) position array, returning bool.
            Example: ``lambda p: p[0] < 0.01`` fixes all nodes with x < 0.01m.
    """

    def __init__(self, predicate):
        self._predicate = predicate

    def get_fixed_node_indices(self, nodes: np.ndarray) -> np.ndarray:
        mask = np.array([self._predicate(nodes[i]) for i in range(len(nodes))])
        return np.nonzero(mask)[0]


class FixedByBox(BoundaryCondition):
    """Fix all nodes inside an axis-aligned bounding box.

    Args:
        lower: Lower corner of the box, shape (3,).
        upper: Upper corner of the box, shape (3,).
    """

    def __init__(self, lower, upper):
        self._lower = np.asarray(lower, dtype=np.float64)
        self._upper = np.asarray(upper, dtype=np.float64)

    def get_fixed_node_indices(self, nodes: np.ndarray) -> np.ndarray:
        inside = np.all(
            (nodes >= self._lower) & (nodes <= self._upper), axis=1
        )
        return np.nonzero(inside)[0]
