"""Analytical Euler-Bernoulli cantilever beam solutions.

Provides closed-form deflection and stress for cantilever beams under
standard loading conditions. Useful for FEM verification and visualization
before the GPU solver backend is connected.

Coordinate convention:
    x: along beam axis (0 = fixed wall, L = free tip)
    y: vertical (bending direction, gravity in -y)
    z: lateral

Sign convention:
    Positive deflection = downward (-y direction)
    Positive bending moment = sagging (bottom in tension)
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class BeamProperties:
    """Cross-section and material properties for a rectangular beam.

    Args:
        L: Beam length [m] (x-direction).
        b: Cross-section width [m] (z-direction).
        h: Cross-section height [m] (y-direction).
        E: Young's modulus [Pa].
        nu: Poisson's ratio.
        density: Mass density [kg/m^3].
    """

    L: float
    b: float
    h: float
    E: float
    nu: float
    density: float

    @property
    def A(self) -> float:
        """Cross-section area [m^2]."""
        return self.b * self.h

    @property
    def I(self) -> float:
        """Second moment of area about z-axis (bending about z) [m^4]."""
        return self.b * self.h**3 / 12.0

    @property
    def y_neutral(self) -> float:
        """Neutral axis y-coordinate [m]."""
        return self.h / 2.0

    @property
    def weight_per_length(self) -> float:
        """Self-weight per unit length q = rho * g * A [N/m]."""
        return self.density * 9.81 * self.A


def deflection_tip_load(x: np.ndarray, P: float, beam: BeamProperties) -> np.ndarray:
    """Deflection of cantilever under tip point load P (downward positive).

    delta(x) = P * x^2 * (3L - x) / (6EI)
    """
    EI = beam.E * beam.I
    return P * x**2 * (3.0 * beam.L - x) / (6.0 * EI)


def deflection_uniform_load(
    x: np.ndarray, q: float, beam: BeamProperties
) -> np.ndarray:
    """Deflection of cantilever under uniform distributed load q [N/m].

    delta(x) = q * x^2 * (x^2 + 6L^2 - 4Lx) / (24EI)
    """
    EI = beam.E * beam.I
    L = beam.L
    return q * x**2 * (x**2 + 6.0 * L**2 - 4.0 * L * x) / (24.0 * EI)


def deflection_self_weight(x: np.ndarray, beam: BeamProperties) -> np.ndarray:
    """Deflection under self-weight (gravity in -y)."""
    return deflection_uniform_load(x, beam.weight_per_length, beam)


def moment_tip_load(x: np.ndarray, P: float, beam: BeamProperties) -> np.ndarray:
    """Bending moment M(x) for tip load P. Positive = bottom in tension."""
    return P * (beam.L - x)


def moment_uniform_load(
    x: np.ndarray, q: float, beam: BeamProperties
) -> np.ndarray:
    """Bending moment M(x) for uniform load q."""
    return q * (beam.L - x) ** 2 / 2.0


def moment_self_weight(x: np.ndarray, beam: BeamProperties) -> np.ndarray:
    """Bending moment under self-weight."""
    return moment_uniform_load(x, beam.weight_per_length, beam)


def bending_stress(
    x: np.ndarray, y: np.ndarray, moment_fn, beam: BeamProperties
) -> np.ndarray:
    """Axial bending stress sigma_xx(x, y) = M(x) * (y - y_neutral) / I.

    Positive stress = tension.

    Args:
        x: x-coordinates of points.
        y: y-coordinates of points.
        moment_fn: Callable(x, beam) -> M(x) array.
        beam: Beam properties.

    Returns:
        Stress array, same shape as x.
    """
    M = moment_fn(x, beam)
    return M * (y - beam.y_neutral) / beam.I


def deform_nodes(
    nodes: np.ndarray, deflection_fn, beam: BeamProperties
) -> np.ndarray:
    """Apply analytical deflection to a set of 3D node positions.

    Assumes beam axis is along x, bending in -y.

    Args:
        nodes: (N, 3) node positions [m].
        deflection_fn: Callable(x, beam) -> delta(x) array (downward positive).
        beam: Beam properties.

    Returns:
        Deformed node positions, shape (N, 3).
    """
    deformed = nodes.copy()
    x = nodes[:, 0]
    delta = deflection_fn(x, beam)
    deformed[:, 1] -= delta  # deflect downward
    return deformed
