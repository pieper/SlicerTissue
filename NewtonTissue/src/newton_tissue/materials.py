"""Material property definitions for soft tissue FEM simulation.

Supports specification via Young's modulus + Poisson's ratio or Lamé parameters.
Internally everything is stored as Lamé parameters (mu, lambda) since that is
what Newton's TetMesh expects (k_mu, k_lambda).

Conversion formulas:
    mu  = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2*nu))
"""

from __future__ import annotations

import abc

import numpy as np


class Material(abc.ABC):
    """Abstract base class for material definitions."""

    @abc.abstractmethod
    def get_mu(self) -> float | np.ndarray:
        """First Lamé parameter (shear modulus) [Pa]."""

    @abc.abstractmethod
    def get_lambda(self) -> float | np.ndarray:
        """Second Lamé parameter [Pa]."""

    @abc.abstractmethod
    def get_density(self) -> float | np.ndarray:
        """Mass density [kg/m^3]."""

    @abc.abstractmethod
    def to_lame_arrays(self, num_elements: int) -> tuple[np.ndarray, np.ndarray]:
        """Return per-element (k_mu, k_lambda) arrays of shape (num_elements,)."""


def _young_poisson_to_lame(E: float, nu: float) -> tuple[float, float]:
    """Convert Young's modulus and Poisson's ratio to Lamé parameters."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def _lame_to_young_poisson(mu: float, lam: float) -> tuple[float, float]:
    """Convert Lamé parameters to Young's modulus and Poisson's ratio."""
    E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    nu = lam / (2.0 * (lam + mu))
    return E, nu


class IsotropicMaterial(Material):
    """Homogeneous isotropic linear elastic material.

    Create from either (E, nu) or (mu, lam). Exactly one pair must be provided.

    Args:
        E: Young's modulus [Pa]. Must be > 0.
        nu: Poisson's ratio (dimensionless). Must be in (-1, 0.5).
        mu: First Lamé parameter (shear modulus) [Pa]. Must be > 0.
        lam: Second Lamé parameter [Pa].
        density: Mass density [kg/m^3]. Default 1000.0 (water).
    """

    def __init__(
        self,
        E: float | None = None,
        nu: float | None = None,
        mu: float | None = None,
        lam: float | None = None,
        density: float = 1000.0,
    ):
        have_young = E is not None and nu is not None
        have_lame = mu is not None and lam is not None

        if have_young == have_lame:
            raise ValueError(
                "Specify exactly one of (E, nu) or (mu, lam), not both or neither."
            )

        if have_young:
            if E <= 0:
                raise ValueError(f"Young's modulus must be positive, got {E}")
            if nu <= -1.0 or nu >= 0.5:
                raise ValueError(
                    f"Poisson's ratio must be in (-1, 0.5), got {nu}"
                )
            self._mu, self._lam = _young_poisson_to_lame(E, nu)
        else:
            if mu <= 0:
                raise ValueError(f"Shear modulus (mu) must be positive, got {mu}")
            self._mu = mu
            self._lam = lam

        if density <= 0:
            raise ValueError(f"Density must be positive, got {density}")
        self._density = density

    @classmethod
    def from_young_poisson(
        cls, E: float, nu: float, density: float = 1000.0
    ) -> IsotropicMaterial:
        return cls(E=E, nu=nu, density=density)

    @classmethod
    def from_lame(
        cls, mu: float, lam: float, density: float = 1000.0
    ) -> IsotropicMaterial:
        return cls(mu=mu, lam=lam, density=density)

    @property
    def mu(self) -> float:
        return self._mu

    @property
    def lam(self) -> float:
        return self._lam

    @property
    def E(self) -> float:
        E, _ = _lame_to_young_poisson(self._mu, self._lam)
        return E

    @property
    def nu(self) -> float:
        _, nu = _lame_to_young_poisson(self._mu, self._lam)
        return nu

    @property
    def density(self) -> float:
        return self._density

    def get_mu(self) -> float:
        return self._mu

    def get_lambda(self) -> float:
        return self._lam

    def get_density(self) -> float:
        return self._density

    def to_lame_arrays(self, num_elements: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.full(num_elements, self._mu, dtype=np.float32),
            np.full(num_elements, self._lam, dtype=np.float32),
        )

    def __repr__(self) -> str:
        return (
            f"IsotropicMaterial(E={self.E:.1f}, nu={self.nu:.4f}, "
            f"mu={self._mu:.1f}, lam={self._lam:.1f}, "
            f"density={self._density:.1f})"
        )


class HeterogeneousMaterial(Material):
    """Per-element material properties.

    Args:
        k_mu: Per-element shear modulus array, shape (num_elements,) [Pa].
        k_lambda: Per-element second Lamé parameter array, shape (num_elements,) [Pa].
        density: Scalar or per-element density [kg/m^3].
    """

    def __init__(
        self,
        k_mu: np.ndarray,
        k_lambda: np.ndarray,
        density: float | np.ndarray = 1000.0,
    ):
        k_mu = np.asarray(k_mu, dtype=np.float32)
        k_lambda = np.asarray(k_lambda, dtype=np.float32)
        if k_mu.ndim != 1 or k_lambda.ndim != 1:
            raise ValueError("k_mu and k_lambda must be 1D arrays")
        if k_mu.shape != k_lambda.shape:
            raise ValueError(
                f"k_mu and k_lambda must have same shape, got {k_mu.shape} and {k_lambda.shape}"
            )
        if np.any(k_mu <= 0):
            raise ValueError("All k_mu values must be positive")

        self._k_mu = k_mu
        self._k_lambda = k_lambda
        self._density = (
            np.asarray(density, dtype=np.float32)
            if isinstance(density, np.ndarray)
            else float(density)
        )

    @property
    def num_elements(self) -> int:
        return len(self._k_mu)

    def get_mu(self) -> np.ndarray:
        return self._k_mu

    def get_lambda(self) -> np.ndarray:
        return self._k_lambda

    def get_density(self) -> float | np.ndarray:
        return self._density

    def to_lame_arrays(self, num_elements: int) -> tuple[np.ndarray, np.ndarray]:
        if num_elements != len(self._k_mu):
            raise ValueError(
                f"Expected {len(self._k_mu)} elements, got {num_elements}"
            )
        return self._k_mu.copy(), self._k_lambda.copy()

    def __repr__(self) -> str:
        return (
            f"HeterogeneousMaterial(num_elements={self.num_elements}, "
            f"mu=[{self._k_mu.min():.1f}, {self._k_mu.max():.1f}], "
            f"lam=[{self._k_lambda.min():.1f}, {self._k_lambda.max():.1f}])"
        )


# ── Tissue presets ──────────────────────────────────────────────────────────

PROSTATE_PERIPHERAL = IsotropicMaterial(E=20.0e3, nu=0.48, density=1040.0)
"""Prostate peripheral zone: E ≈ 20 kPa, ν ≈ 0.48, ρ = 1040 kg/m³."""

PROSTATE_TRANSITION = IsotropicMaterial(E=40.0e3, nu=0.45, density=1040.0)
"""Prostate transition zone: E ≈ 40 kPa, ν ≈ 0.45, ρ = 1040 kg/m³."""
